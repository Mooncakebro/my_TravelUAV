"""
Configuration for JAMEL-COMPACT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CompactConfig:
    """All hyperparameters for JAMEL-COMPACT model, training, and eval."""

    # ── Model ──
    base_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    mem_dim: int = 512                # reduced memory dimension d_mem
    num_mem_tokens: int = 16           # N_m memory tokens per layer
    num_heads: int = 8                # attention heads in side memory
    freeze_base: bool = True          # freeze pretrained LLM weights by default
    model_parallel: bool = False      # shard frozen base layers across visible GPUs
    num_act_tokens: int = 1           # action tokens in input sequence
    lora_rank: int = 0                # 0 = disabled; >0 enables base-model LoRA
    lora_alpha: int = 0               # must be >0 when lora_rank > 0
    lora_dropout: float = 0.0
    lora_target_modules: str = (
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    lora_bias: str = "none"

    # ── Model version (for checkpoint compatibility) ──
    model_version: int = 2            # v1=1, v2=2 (U1/U2 change side_memories.pt keys)

    # ── Hierarchical hyperparameters ──
    lambda_shallow: float = 0.70
    lambda_mid: float = 0.85
    lambda_deep: float = 0.95
    inject_shallow: float = 0.8
    inject_mid: float = 0.5
    inject_deep: float = 0.3
    alpha_confidence: float = 0.1     # learning rate for confidence update (v1 only)

    # ── v2: observation / Kalman / injection ──
    num_obs_tokens: int = 4           # k learned latent queries for observation pooling (U1)
    gamma_e: float = 1.0              # surprise inflation factor for variance predict (U2)
    surprise_clip: float = 10.0       # max surprise fed into variance inflation (stability)
    r_min: float = 0.01               # floor on learned observation noise R (stability)
    inject_gate_init: float = 0.1      # lets the zero-init output projection receive gradients
    lambda_obs: float = 0.01          # weight for observation-prediction loss L_obs (U3)
    lambda_nll: float = 0.01          # weight for Gaussian NLL loss L_nll (U2; replaces lambda_uncert)
    obs_loss: str = "mse"             # "mse" or "infonce" for L_obs variant (U3)
    memory_conditioned_generate: bool = True  # F5: use KV-cache prefill for generation

    # ── v2: U4 experiment flags (all default off) ──
    shared_memory: bool = False       # U4a: single shared memory across layers
    slow_memory: bool = False         # U4b: two-timescale EMA track
    slow_memory_eta: float = 0.05     # U4b: EMA update rate
    sparse_write_k: int = 0           # U4c: top-k sparse writes (0 = off)

    # ── Training ──
    output_dir: str = "outputs/compact_ckpt"
    learning_rate: float = 2e-5
    memory_learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    bf16: bool = True
    seed: int = 42
    log_steps: int = 10
    save_steps: int = 500
    val_steps: int = 200
    tbptt_detach: bool = True         # F6: detach carried state at chunk boundaries
    coverage_weight_eta: float = 0.0  # F7: 0 = off; >0 weights high-novelty samples

    # ── Loss weights ──
    lambda_mem: float = 0.001        # memory regularization weight
    lambda_uncert: float = 0.1       # uncertainty calibration weight (v1: MSE; v2: NLL)
    beta_entropy: float = 0.01        # entropy regularization in mem loss (v1 only)

    # ── Data ──
    max_length: int = 8192
    train_file: str = ""
    val_file: str = ""
    val_ratio: float = 0.05
    image_resize: tuple = (640, 360)  # WEB_MODEL_IMAGE_SIZE
    chunk_size: int = 8  # F6: session-chunked training default (was 1 in v1)

    # ── Eval ──
    eval_output: str = "outputs/compact_eval"
    max_steps: int = 50
    num_sessions: int = 3
    temperature: float = 0.8
    top_p: float = 0.9
    scalewob_root: str = "env/browser_env/scalewob-env"
    apps_mode: str = "test10"
    freeze_memory_init: bool = False  # F5 ablation: never write new_memory back to session state

    # ── TensorBoard ──
    tb_log_dir: str = "outputs/compact_tb"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_args(cls, **kwargs) -> "CompactConfig":
        cfg = cls()
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
