"""
COMPACT-UAV training entry: waypoint regression with TBPTT over episodes.

Adapted from jamel_compact/train.py for the TravelUAV task:
  - memory is carried step-to-step within an episode and detached at chunk
    boundaries (TBPTT, chunk_size steps); reset at episode boundaries;
  - loss = waypoint (cosine direction + L1 distance) + COMPACT aux losses;
  - base VLM frozen by default, optional LoRA on top.

Example:
    python train_compact_uav.py \
        --data_path ../../data/uav_dataset/trainset.json \
        --dataset_path ../../data/raw_dataset/extracted \
        --base_model_name Qwen/Qwen3-VL-2B-Instruct \
        --lora_rank 32 --lora_alpha 64 \
        --output_dir work_dirs/compact-uav-2b-lora32
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compact.config import CompactConfig
from compact_uav_model import CompactUAVModel
from dataset_uav import UAVEpisodeDataset, process_step


def parse_args():
    p = argparse.ArgumentParser(description="Train COMPACT-UAV waypoint policy")
    # data
    p.add_argument('--data_path', type=str, required=True,
                   help='trainset.json (list of {json, frame})')
    p.add_argument('--dataset_path', type=str, required=True,
                   help='raw dataset root with <Map>/<episode>/ folders')
    # model
    p.add_argument('--base_model_name', type=str,
                   default='Qwen/Qwen3-VL-2B-Instruct')
    p.add_argument('--resume_from', type=str, default=None,
                   help='COMPACT-UAV checkpoint dir to resume weights from')
    p.add_argument('--lora_rank', type=int, default=32)
    p.add_argument('--lora_alpha', type=int, default=64)
    p.add_argument('--lora_dropout', type=float, default=0.0)
    p.add_argument('--train_base', action='store_true',
                   help='unfreeze base VLM (dense SFT; no LoRA allowed)')
    p.add_argument('--no_memory', action='store_true',
                   help='ablation: reset memory every step (no carry)')
    # training
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--chunk_size', type=int, default=8)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--memory_learning_rate', type=float, default=2e-5,
                   help='lr for side memories + waypoint head + prev_action_mlp')
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--grad_accum_steps', type=int, default=16,
                   help='optimizer steps per N TBPTT chunks')
    p.add_argument('--max_episode_steps', type=int, default=None)
    p.add_argument('--max_episodes', type=int, default=None,
                   help='debug: cap number of episodes per epoch')
    p.add_argument('--max_length', type=int, default=8192)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log_steps', type=int, default=10)
    p.add_argument('--save_steps', type=int, default=500)
    p.add_argument('--no_gradient_checkpointing', action='store_true')
    p.add_argument('--adam8bit', action='store_true',
                   help='use bitsandbytes 8-bit AdamW (needed on <=8GB GPUs; '
                        'fp32 AdamW states for ~335M trainable params cost '
                        '~2.7GB VRAM)')
    p.add_argument('--dry_run', action='store_true',
                   help='smoke test: 2 episodes, chunk_size=2, no saving')
    return p.parse_args()


def detach_states(states):
    if states is None:
        return None
    return [s.detach() if torch.is_tensor(s) else None for s in states]


def main():
    args = parse_args()
    if args.dry_run:
        args.max_episodes = 2
        args.chunk_size = 2
        args.max_episode_steps = 4
        args.epochs = 1
        args.grad_accum_steps = 1
        print('[train] DRY RUN mode')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = CompactConfig.from_args(
        base_model_name=args.base_model_name,
        freeze_base=not args.train_base,
        lora_rank=0 if args.train_base else args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        chunk_size=args.chunk_size,
        max_length=args.max_length,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        bf16=True,
    )

    if args.resume_from:
        model = CompactUAVModel.from_pretrained(
            args.resume_from, config_override=config)
    else:
        model = CompactUAVModel(config, use_memory=not args.no_memory)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    counts = model.count_parameters()
    print(f"[train] params: base={counts['base']/1e9:.2f}B "
          f"new={counts['new']/1e6:.1f}M "
          f"(memory={counts['memory']/1e6:.1f}M lora={counts['lora']/1e6:.1f}M)")

    # ── Optimizer: LoRA params at lr, new modules at memory_learning_rate ──
    llm_trainable = [p for p in model.llm.parameters() if p.requires_grad]
    new_modules = [model.side_memories, model.action_embed,
                   model.waypoint_emb, model.waypoints_fc,
                   model.waypoints_output, model.prev_action_mlp]
    new_params = [p for m in new_modules for p in m.parameters()]
    new_params.append(model.null_action)
    param_groups = []
    if llm_trainable:
        param_groups.append({'params': llm_trainable, 'lr': args.learning_rate})
    param_groups.append({'params': new_params, 'lr': args.memory_learning_rate})
    if args.adam8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(param_groups,
                                        weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups,
                                      weight_decay=args.weight_decay)

    dataset = UAVEpisodeDataset(
        args.data_path, args.dataset_path,
        max_episode_steps=args.max_episode_steps,
    )
    processor = model.processor
    tokenizer = model.tokenizer
    assert processor is not None and tokenizer is not None

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0  # optimizer steps
    micro_step = 0   # chunks since last optimizer step
    running = {}

    for epoch in range(args.epochs):
        order = list(range(len(dataset)))
        random.shuffle(order)
        if args.max_episodes:
            order = order[:args.max_episodes]

        for ep_idx in order:
            episode = dataset[ep_idx]
            if episode is None:
                continue
            steps = episode['steps']

            memory_states, variance_states, e_prev_list = None, None, None
            for chunk_start in range(0, len(steps), args.chunk_size):
                chunk = steps[chunk_start:chunk_start + args.chunk_size]
                chunk_loss = 0.0
                for step in chunk:
                    inputs = process_step(step, processor, tokenizer,
                                          max_length=args.max_length)
                    model_inputs = {
                        k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in inputs.items()
                        if k in ('input_ids', 'attention_mask', 'sentinel_mask',
                                 'observation_mask', 'pixel_values',
                                 'image_grid_thw', 'mm_token_type_ids')
                        and v is not None
                    }
                    prev_wp = inputs['prev_waypoint']
                    if prev_wp is not None:
                        prev_wp = prev_wp.to(device)
                    out = model(
                        **model_inputs,
                        prev_waypoints=prev_wp,
                        memory_states=memory_states,
                        variance_states=variance_states,
                        e_prev_list=e_prev_list,
                        waypoint_labels=inputs['waypoint_label'].to(device),
                    )
                    chunk_loss = chunk_loss + out['loss'] / len(chunk)
                    memory_states = out['new_memory']
                    variance_states = out['new_variance']
                    e_prev_list = out['e_list']
                    for k, v in out['loss_dict'].items():
                        running[k] = running.get(k, 0.0) + float(v) / len(chunk)

                chunk_loss.backward()
                micro_step += 1

                # TBPTT: detach carried state at chunk boundary
                memory_states = detach_states(memory_states)
                variance_states = detach_states(variance_states)
                e_prev_list = detach_states(e_prev_list)

                if micro_step % args.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in param_groups for p in g['params']],
                        args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % args.log_steps == 0:
                        n = args.log_steps * args.grad_accum_steps
                        msg = ' '.join(
                            f'{k}={v / n:.4f}' for k, v in sorted(running.items()))
                        print(f'[train] epoch {epoch} step {global_step} {msg}',
                              flush=True)
                        running = {}
                    if not args.dry_run and global_step % args.save_steps == 0:
                        ckpt = os.path.join(args.output_dir,
                                            f'checkpoint-{global_step}')
                        model.save_pretrained(ckpt)

            # episode boundary: drop memory (next episode starts fresh)
            memory_states = variance_states = e_prev_list = None

        if not args.dry_run:
            model.save_pretrained(
                os.path.join(args.output_dir, f'epoch-{epoch}'))

    if not args.dry_run:
        model.save_pretrained(os.path.join(args.output_dir, 'final'))
    print('[train] done')


if __name__ == '__main__':
    main()
