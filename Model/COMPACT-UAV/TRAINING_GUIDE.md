# COMPACT-UAV training and evaluation

This guide uses the copied COMPACT implementation in this repository. The policy
action is always the model's raw four-vector `[direction_x, direction_y,
direction_z, distance]`. Training regresses that vector; closed-loop inference
converts it directly to one world-frame XYZ target and linearly samples the
endpoint into the simulator's five-point path API. It does not run the
trajectory-refinement network.

## Environment check

The tested environment is `lerobot` (Torch 2.10.0+cu128, Transformers 4.57.6,
PEFT 0.21.0). No packages were installed for this check. `yolov8-py38` has
Torch 2.4.1 and Transformers 4.46.3, which is too old for Qwen3-VL; use the
`lerobot` environment or a server environment with Transformers >= 4.57 and
PEFT installed.

```bash
cd /home/spc/memory_arena/TravelUAV
export LD_LIBRARY_PATH=/home/spc/anaconda3/envs/lerobot/lib:${LD_LIBRARY_PATH:-}
conda run -n lerobot python -c \
  "import torch,transformers,peft; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.device_count())"
```

Run the model-only forward/backward smoke test before submitting a large job:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  conda run -n lerobot python Model/COMPACT-UAV/smoke_test.py
```

## Multi-A800 training

Run from the COMPACT directory. `torchrun` assigns one process per GPU; each
rank uses a deterministic shard of complete episodes, so memory is reset only
at an episode boundary. The default split is deterministic and episode-level:
95% training and 5% validation (`--eval_fraction 0.05`). Validation loss is
all-reduced across ranks after each epoch. The best model is written to
`<output_dir>/best`, with the selection recorded in `best_metrics.json`.

```bash
cd /home/spc/memory_arena/TravelUAV/Model/COMPACT-UAV
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 train_compact_uav.py \
  --base_model_name Qwen/Qwen3-VL-2B-Instruct \
  --data_path ../../data/uav_dataset/trainset.json \
  --dataset_path ../../data/raw_dataset/extracted \
  --output_dir work_dirs/compact-uav-2b-lora32 \
  --lora_rank 32 --lora_alpha 64 \
  --chunk_size 8 --grad_accum_steps 16 \
  --epochs 3 --eval_fraction 0.05 --eval_every 1 \
  --save_steps 500
```

For a quick data/model integration check, add `--dry_run`; it limits training
to two episodes and validation to one episode and does not save checkpoints.
Use `--no_memory` for the reset-memory ablation. `--adam8bit` is available for
smaller GPUs, but the intended A800 run can use the default AdamW.

## Final closed-loop evaluation

Start the AirSim server first, then evaluate the `best` checkpoint. The
trajectory-model and CLIP image-processor arguments are intentionally absent
for COMPACT-UAV. GroundingDINO assets and the simulator environment are still
required by the unchanged stop/metric pipeline.

```bash
# terminal 1
cd /home/spc/memory_arena/TravelUAV/airsim_plugin
python AirVLNSimulatorServerTool.py --port 30000 --root_path /path/to/TravelUAV_env

# terminal 2
cd /home/spc/memory_arena/TravelUAV
CUDA_VISIBLE_DEVICES=0 python -u src/vlnce_src/eval.py \
  --run_type eval --policy compact_uav --name COMPACT-UAV \
  --gpu_id 0 --simulator_tool_port 25000 --batchSize 2 \
  --always_help True --use_gt True --maxWaypoints 200 \
  --dataset_path data/raw_dataset/extracted \
  --eval_json_path data/uav_dataset/seen_valset.json \
  --eval_save_path data/eval_closeloop/compact_uav_best \
  --model_path Model/COMPACT-UAV/work_dirs/compact-uav-2b-lora32/best \
  --map_spawn_area_json_path data/meta/map_spawnarea_info.json \
  --object_name_json_path data/meta/object_description.json \
  --groundingdino_config src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --groundingdino_model_path src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth

bash scripts/metric.sh
```

Use `--use_gt False` for a no-ground-truth-assistance result after the assisted
sanity run. Keep the same image ordering and prompt construction in training
and evaluation: `front, left, right, rear, down`, with the literal dataset
`<image>` marker stripped before Qwen's chat template inserts five image slots.
