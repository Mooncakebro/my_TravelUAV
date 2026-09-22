"""Tiny-config smoke test for CompactUAVModel (no pretrained weights)."""
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from compact.config import CompactConfig
from compact_uav_model import CompactUAVModel

TMP = '/tmp/tiny_qwen3vl'

if not os.path.exists(TMP):
    cfg = AutoConfig.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')
    cfg.text_config.hidden_size = 256
    cfg.text_config.intermediate_size = 512
    cfg.text_config.num_hidden_layers = 2
    cfg.text_config.num_attention_heads = 4
    cfg.text_config.num_key_value_heads = 2
    cfg.text_config.num_experts = 0 if hasattr(cfg.text_config, 'num_experts') else None
    if hasattr(cfg.vision_config, 'depth'):
        cfg.vision_config.depth = 2
    if hasattr(cfg.vision_config, 'hidden_size'):
        cfg.vision_config.hidden_size = 64
    if hasattr(cfg.vision_config, 'intermediate_size'):
        cfg.vision_config.intermediate_size = 128
    if hasattr(cfg.vision_config, 'num_heads'):
        cfg.vision_config.num_heads = 4
    cfg.save_pretrained(TMP)
    print('tiny config saved')

from transformers import AutoModelForImageTextToText, AutoTokenizer
if not os.path.exists(os.path.join(TMP, 'model.safetensors')):
    cfg = AutoConfig.from_pretrained(TMP)
    model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.float32)
    model.save_pretrained(TMP)
    print('tiny random model saved')
if not os.path.exists(os.path.join(TMP, 'tokenizer_config.json')):
    AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct').save_pretrained(TMP)
    AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-2B-Instruct').save_pretrained(TMP)
    print('processor/tokenizer saved into tiny dir')

_lora = int(os.environ.get('SMOKE_LORA_RANK', '0'))
config = CompactConfig.from_args(
    base_model_name=TMP, bf16=False, gradient_checkpointing=False,
    lora_rank=_lora, lora_alpha=_lora * 2,
)
m = CompactUAVModel(config)
print('model built:', m.count_parameters())

processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')
tokenizer = m.tokenizer

from dataset_uav import process_step
step = {
    'image_paths': [],
    'prompt_text': 'Stage:cruise\n\nPrevious displacement:0.9,0.1,0.0\n\nCurrent position:1.0,2.0,-3.0\n\nCurrent image (in order: front, left, right, rear, down):\n\nInstruction:There is a target. Please control the drone and find the target.',
    'waypoint_label': np.array([1.0, 0.0, 0.0, 5.0], dtype=np.float32),
    'prev_waypoint': None,
}
# fake 5 view images
tmpdir = Path('/tmp/uav_imgs'); tmpdir.mkdir(exist_ok=True)
for cam in ['frontcamera', 'leftcamera', 'rightcamera', 'rearcamera', 'downcamera']:
    p = tmpdir / f'{cam}.png'
    Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8)).save(p)
    step['image_paths'].append(str(p))

inputs = process_step(step, processor, tokenizer)
print({k: (v.shape if torch.is_tensor(v) else v) for k, v in inputs.items() if k not in ('prev_waypoint',)})

# step 1: episode start (prev=None), with labels -> loss
out1 = m(**{k: v for k, v in inputs.items() if k in
            ('input_ids', 'attention_mask', 'sentinel_mask', 'observation_mask',
             'pixel_values', 'image_grid_thw', 'mm_token_type_ids') and v is not None},
         prev_waypoints=None,
         waypoint_labels=inputs['waypoint_label'])
print('step1 pred shape:', out1['predicted_waypoints'].shape,
      'loss:', float(out1['loss']))
out1['loss'].backward()
grads = {n: p.grad is not None for n, p in m.named_parameters() if p.requires_grad}
no_grad = [n for n, g in grads.items() if not g]
print('trainable params:', len(grads), 'missing grads:', no_grad[:10])
assert not no_grad, f'params without grad: {no_grad}'

# step 2: carry memory, teacher-forced prev action
m.zero_grad()
out2 = m(**{k: v for k, v in inputs.items() if k in
            ('input_ids', 'attention_mask', 'sentinel_mask', 'observation_mask',
             'pixel_values', 'image_grid_thw', 'mm_token_type_ids') and v is not None},
         prev_waypoints=inputs['waypoint_label'],
         memory_states=[s.detach() for s in out1['new_memory']],
         variance_states=[s.detach() for s in out1['new_variance']],
         e_prev_list=out1['e_list'],
         waypoint_labels=inputs['waypoint_label'])
print('step2 pred:', out2['predicted_waypoints'].detach().numpy().round(3),
      'loss:', float(out2['loss']))

# memory actually changes across steps?
diff = (out1['new_memory'][0] - out2['new_memory'][0]).abs().max()
print('memory diff across steps:', float(diff))

# save/load roundtrip
m.save_pretrained('/tmp/compact_uav_ckpt')
m2 = CompactUAVModel.from_pretrained('/tmp/compact_uav_ckpt')
sd1 = m.waypoints_output.state_dict()['weight']
sd2 = m2.waypoints_output.state_dict()['weight']
assert torch.equal(sd1, sd2), 'waypoint head not preserved'
print('save/load roundtrip OK')
print('SMOKE TEST PASSED')
