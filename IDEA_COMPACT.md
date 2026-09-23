# IDEA: COMPACT-UAV — Porting the COMPACT Memory Architecture to the TRAVEL UAV-VLN Benchmark

Status: planning / pre-implementation
Date: 2026-09-20

## 1. Goal

Replace the TravelLLM policy's MLLM (LLaMA-UAV = LLaMA-VID / Vicuna-7B + EVA-ViT-G, LoRA) with our
**COMPACT model (Qwen3-VL-2B/4B + per-layer recurrent side memory)**, while keeping the rest of the
TRAVEL closed-loop pipeline (trajectory predictor, GroundingDINO stop monitor, assist hints, AirSim
execution, metrics) unchanged.

**Constraint: the original COMPACT repo (`/home/spc/JAMEL-COMPACT`) must NOT be modified.**
Any COMPACT code needed here is **copied** into this repo (see §6).

Scientific motivation: TravelUAV's policy receives **no visual history** — history enters only as two
text lines ("Previous displacement", "Current position"). COMPACT's learned memory is a principled
replacement for this textual hack; the comparison "VLM + text history" vs "VLM + memory" on an
identical action interface is the core experiment.

## 2. TravelUAV repo — who is who

| Piece | Path | Role |
|---|---|---|
| Simulator bridge | `airsim_plugin/AirVLNSimulatorServerTool.py`, `AirVLNSimulatorClientTool.py` | Hosts AirSim/Unreal envs; serves 5 RGB + 5 depth cameras @ 256×256; executes `moveOnPathAsync` (1 m/s, ForwardOnly, lookahead 3) |
| Closed-loop env | `src/vlnce_src/env_uav.py`, `closeloop_util.py` | Episode loop, collision/stuck detection, success = stop within 20 m of target |
| Assist module | `src/vlnce_src/assist.py` | "Stage:" prompt hints from depth rules / GroundingDINO / GT trajectory (`--always_help True --use_gt True` in eval) |
| Policy wrapper | `src/model_wrapper/travel_llm.py` | LLM → 4-dim waypoint → traj model → 7 refined world-frame waypoints; DINO stop |
| MLLM (to be replaced) | `Model/LLaMA-UAV/llamavid/model/language_model/llava_llama_uav.py` | `LlavaLlamaAttForCausalLM`: sentinel slot + waypoint regression head |
| Trajectory predictor (kept) | `Model/LLaMA-UAV/llamavid/model/vis_traj_arch.py` | `VisionTrajectoryGenerator`: front camera + 3-dim waypoint → 7 future waypoints |
| Data prep tools | `Model/LLaMA-UAV/tools/generate_merged_json.py`, `preprocess_image2tensor.py` | Build `merged_data.json` per episode; CLIP-preprocess images (EVA-specific, must be redone for Qwen) |
| Train scripts | `Model/LLaMA-UAV/scripts/llm/train_uav_llm.sh`, `scripts/traj/train_traj_completion.sh` | Stage 1 LLM, stage 2 traj predictor |
| Entry points | `src/vlnce_src/dagger.py`, `eval.py`; `scripts/dagger_NYC.sh`, `eval.sh`, `metric.sh` | Closed-loop DAgger collection / evaluation |

### 2.1 The sentinel-slot mechanism (the action interface we keep)

TravelUAV's LLM **never generates text** (`lm_head` is never called; no CE loss). It works like a VLA:

1. One extra token position is appended to the prompt (just before `</s>`). It is not a vocab token —
   its input embedding is overwritten with a learned vector
   (`self.waypoint_emb = nn.Embedding(1, hidden_size)`, `llava_llama_uav.py:59,147`).
   A dedicated "query slot", BERT-`[CLS]`-style.
2. The decoder runs normally; the slot attends to all image tokens + instruction.
3. The final-layer hidden state at that slot feeds `waypoints_fc` (`4096→2048→64`) +
   `waypoints_output` (`64→4`) → **unit direction xyz + distance**
   (`llava_llama_uav.py:60-65`).
4. Loss: cosine-similarity on direction + L1 on distance (`use_angle_and_norm_loss=True`,
   `llava_llama_uav.py:168-175`).
5. At inference, `forward(..., return_waypoints=True)` returns the head output directly
   (`travel_llm.py:40-51`), renormalized to `dir * distance`.

Note: `orientations` input and the `<his>` history-embedding insertion branches are effectively dead
code — history enters the shipped model only as prompt text.

## 3. Dataset

### 3.1 What we have

- Split JSONs already downloaded to this repo (from HF `wangxiangyu0814/TravelUAV_data_json`):
  - `data/uav_dataset/trainset.json` — **427,933 samples**, each
    `{"json": "<Map>/<episode_id>/merged_data.json", "frame": N}`. Frames within one episode are
    consecutive (`frame: 1, 2, 3, ...`) → can be grouped into sequences for memory/TBPTT training.
  - `data/uav_dataset/seen_valset.json` (9.1 MB), `unseen_valset.json` (10.7 MB)
  - `data/meta/map_spawnarea_info.json`, `data/meta/object_description.json`
    (94 target objects, e.g. `{"object_name": "SM_African_elephant", "object_desc": "Brown elephant"}`)
- Train map distribution (samples): NYCEnvironmentMegapa 111,693; TropicalIsland 59,011;
  NewYorkCity 48,269; Carla_Town01/02/03/04/05/06/07/10HD/15 (15k–39k each); ModernCityMap 13,780;
  Brushify*/Japanese_Street/London_Street/NordicHarbour/WesterTown/BattlefieldKitDesert (150–3k each).
- Raw dataset: HF `wangxiangyu0814/TravelUAV`, **~483 GB total**, per-map split-zip archives
  (`.z01…/.zip` parts, reassemble with `zip -s 0 <Map>.zip --out <Map>_full.zip && unzip` or 7z).
  - Download: **handled by the user** into `/media/spc/新加卷/TravelUAV_dataset`
    (802 GB free; extract map-by-map, delete archives as we go — extracted data roughly doubles usage).
  - Symlink already in place: `data/raw_dataset -> /media/spc/新加卷/TravelUAV_dataset`.

### 3.2 Raw episode structure (per `<dataset_root>/<Map>/<episode_id>/`)

- `log/000000.json …` — per-frame sensor state (`sensors.state.position`, `orientation` quaternion)
- 5 RGB folders: `frontcamera/ leftcamera/ rightcamera/ rearcamera/ downcamera/*.png` (256×256)
- 5 depth folders: `*_depth/*.png`
- `object_description.json` — list of captions for the target; one is sampled into the instruction
- `mark.json` — object name + GT target world position (eval)
- generated: `merged_data.json` — keys: `trajectory` (6-dof states in start frame),
  `trajectory_raw`, `trajectory_raw_detailed`, `index`, `length`, `image_feature_path`,
  `conversations` = `[human: "<image>\n" + instruction, gpt: ""]`.

Instruction template (`generate_merged_json.py:109`):

> There is a target in the %orientation_description% of uav. Using your front as the x-axis and your
> right as the y-axis, The target is at a yaw angle of %orientation_value% degrees from you.
> %object_description% Please control the drone and find the target.

### 3.3 One training sample (reference LLM pipeline)

Current frame's 5 views (224×224, EVA-ViT-G, Q-Former-compressed to ~17 tokens/view) + prompt
(`Stage:<take off/cruise/landing/left/right>` + `Previous displacement:` + `Current position:` +
`Current image: <image>` + `Instruction:`, template `imgsp_uav`) → label = next frame's GT position
relative to current pose, rotated into a target-aligned yaw frame, as (unit direction, norm)
(`train_uav_notice.py:388-399, 750-785, 945-968`).

**The shipped `rgb_imgs.tensor` files are CLIP/EVA-specific — for Qwen3-VL we must re-preprocess
from the raw PNGs with the Qwen processor (new tool, §5 step 2).**

## 4. COMPACT recap (source: `/home/spc/JAMEL-COMPACT`, read-only reference)

- Base: `Qwen/Qwen3-VL-2B-Instruct` (default; 4B/8B supported), base frozen by default, optional LoRA.
- Memory: one `SideMemoryModule` per decoder layer. State `M ∈ R^{16×512}` + variance track `P`.
  Per step: **predict** (action-conditioned FiLM-GRU transition) → frozen decoder layer in place →
  **observe** (hidden states → d_mem, k=4 latent queries attention-pool over prompt positions) →
  **correct** (learned Kalman update) → **inject** (cross-attn write-back via zero-init `delta_up`,
  tanh gate — at init the wrapped model is exactly the base VLM).
- GUI actions: pure text `<action>...</action>` BrowserGym calls — **not used here**; we attach the
  waypoint head instead.
- Training: TBPTT, chunks of 8 consecutive steps, state detached at chunk boundaries; aux losses
  (observation-prediction MSE + Gaussian NLL + memory L2).
- Input: 1 image/step; here extended to 5 images/step (Qwen3-VL natively supports multi-image).

## 5. Implementation plan (COMPACT-UAV)

### Architecture

1. **Wrap Qwen3-VL with COMPACT side memory** (copied code, §6) — unchanged mechanism.
2. **Sentinel slot + waypoint head**, ported from `llava_llama_uav.py`:
   - append one slot before end of prompt; input embedding = learned `waypoint_emb`;
   - final hidden state at slot → `waypoints_fc` (hidden→2048→64, adjust input dim: 2048 for 2B,
     2560 for 4B) → `waypoints_output` (64→4);
   - loss = cosine(direction) + L1(distance), same as reference;
   - the slot's hidden state is automatically memory-conditioned (memory injects at every layer).
     Verify `extract_observation` prompt-position pooling treats the slot sanely.
3. **Five views**: prompt contains 5 `<image>` placeholders in fixed order
   (front/left/right/rear/down) with text labels. ~64 visual tokens/view @ 256×256 → ~320 total,
   well within `max_length=8192`.
4. **Previous-action input**: replace "mean-pool previous action text tokens" with a small MLP
   embedding the **previous 4-dim waypoint** (direction+distance) into the FiLM-GRU control space.
   - Training: teacher forcing with GT previous waypoint.
   - Inference: the predicted 4-dim waypoint itself is the action for now (trajectory refinement
     is downstream and is not fed back). Episode start: learned null-action embedding / zeros.
5. **Trajectory predictor**: keep frozen, LLM-agnostic (input = front camera + 3-dim waypoint).
   Reuse released checkpoint `wangxiangyu0814/traveluav-traj-model` initially; optionally retrain on
   COMPACT-UAV's waypoint distribution later (`scripts/traj/train_traj_completion.sh`).
6. **Stop / assist / execution / metrics**: unchanged (GroundingDINO + assist module + AirSim client).

### Training data pipeline (new)

- Group `trainset.json` entries by episode → sequences of consecutive frames (TBPTT chunks of 8,
  memory detached at boundaries, matching COMPACT's `chunk_size=8`).
- New dataloader collate: 5 PNGs → Qwen processor; prompt template mirrors `imgsp_uav`
  (Stage/Previous displacement/Current position kept verbatim for the first ablation; a
  memory-only variant dropping them is the ablation that isolates COMPACT's contribution).
- Labels: same 4-dim waypoint target convention as reference.

### New files (all inside this repo)

```
Model/COMPACT-UAV/
  compact/                       # copied jamel_compact/{model,config,lora,loss}.py
                                 # + minimal extensions (embed_override hook,
                                 #   hidden_states/loss_obs/loss_nll in result,
                                 #   skip_lm_head, tuple-API get_image_features,
                                 #   batched DeepStack offset fix)
  compact_uav_model.py           # CompactUAVModel: sentinel slot + waypoint head
                                 # + prev_action_mlp (4→hidden) + null_action
  dataset_uav.py                 # episode-grouped TBPTT dataset, Qwen processor
                                 # on the fly (NO EVA rgb_imgs.tensor needed)
  train_compact_uav.py           # TBPTT training entry (--dry_run for smoke tests)
  smoke_test.py                  # tiny-config end-to-end test (passed, see §9)
src/model_wrapper/compact_uav.py # eval/DAgger wrapper (policy selector below)
```

Wiring: `--policy compact_uav` (default `travelllm`) added to
`src/common/param.py`, selected in `src/vlnce_src/eval.py` and `dagger.py`.
`--model_path` points to the COMPACT-UAV checkpoint dir when using the new policy.

## 6. Step-by-step commands

### 6.0 One-time setup

```bash
# env (follow README.md) — llamauav conda env, torch 2.0.1 cu118, pip install -e Model/LLaMA-UAV, -r requirement.txt
# model zoo: vicuna-7b-v1.5, eva_vit_g.pth (only needed for baselines/traj predictor vision tower)
# GroundingDINO ckpt -> src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth

# copy COMPACT code (read-only source, do NOT modify original)
# already done — Model/COMPACT-UAV/compact/ = {__init__,config,lora,loss,model}.py
```

### 6.1 Data preparation (after the user finishes the raw download)

```bash
cd data/raw_dataset   # = /media/spc/新加卷/TravelUAV_dataset

# per map: reassemble split zip, extract, delete archive to stay within disk budget
MAP=NYCEnvironmentMegapa   # repeat per map, big training maps first
zip -s 0 $MAP.zip --out ${MAP}_full.zip && unzip -q ${MAP}_full.zip -d extracted/ && rm ${MAP}_full.zip $MAP.z* $MAP.zip

# merged jsons (reference tool, works as-is)
cd /home/spc/memory_arena/TravelUAV/Model/LLaMA-UAV
python tools/generate_merged_json.py --root_dir /home/spc/memory_arena/TravelUAV/data/raw_dataset/extracted

# NOTE: no image preprocessing step needed for COMPACT-UAV — dataset_uav.py
# loads raw PNGs and runs the Qwen3-VL processor on the fly.
```

### 6.2 Train COMPACT-UAV (stage 1: waypoint policy)

```bash
cd /home/spc/memory_arena/TravelUAV/Model/COMPACT-UAV
torchrun --standalone --nproc_per_node=8 train_compact_uav.py \
    --base_model_name Qwen/Qwen3-VL-2B-Instruct \
    --data_path ../../data/uav_dataset/trainset.json \
    --dataset_path ../../data/raw_dataset/extracted \
    --lora_rank 32 --lora_alpha 64 \
    --chunk_size 8 \
    --output_dir work_dirs/compact-uav-2b-lora32
# For a single GPU, use `python -u` instead of `torchrun`.
# smoke test first: add --dry_run (2 episodes, 4 steps, no saving)
# ablations: --no_memory (reset memory every step), --train_base (dense SFT)
```

Requires an env with torch + transformers>=4.57 + peft (verified working in the
`lerobot` conda env with `LD_LIBRARY_PATH=/home/spc/anaconda3/envs/lerobot/lib`).

### 6.3 Trajectory predictor (stage 2)

```bash
# option A (start here): reuse released ckpt
hf download wangxiangyu0814/traveluav-traj-model --local-dir Model/LLaMA-UAV/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4
# option B (later): retrain on COMPACT-UAV waypoint outputs
bash Model/LLaMA-UAV/scripts/traj/train_traj_completion.sh
```

### 6.4 Closed-loop eval

```bash
# terminal 1: simulator server (needs downloaded envs from TravelUAV_env, see README)
cd airsim_plugin && python AirVLNSimulatorServerTool.py --port 30000 --root_path /path/to/envs

# terminal 2: eval with COMPACT-UAV wrapper (--policy compact_uav)
cd /home/spc/memory_arena/TravelUAV
CUDA_VISIBLE_DEVICES=0 python -u src/vlnce_src/eval.py \
    --run_type eval --policy compact_uav --name COMPACT-UAV --gpu_id 0 \
    --simulator_tool_port 25000 --batchSize 2 \
    --always_help True --use_gt True --maxWaypoints 200 \
    --dataset_path data/raw_dataset/extracted \
    --eval_save_path data/eval_closeloop/compact_uav_test \
    --model_path Model/COMPACT-UAV/work_dirs/compact-uav-2b-lora32/final \
    --traj_model_path Model/LLaMA-UAV/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4 \
    --vision_tower Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor Model/LLaMA-UAV/llamavid/processor/clip-patch14-224 \
    --eval_json_path data/uav_dataset/seen_valset.json \
    --map_spawn_area_json_path data/meta/map_spawnarea_info.json \
    --object_name_json_path data/meta/object_description.json \
    --groundingdino_config src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth

bash scripts/metric.sh   # SR / SPL / NE etc. on eval_save_path outputs
```

## 9. Implementation status (2026-09-21)

- All modules implemented and py_compile-clean; `smoke_test.py` **passed** on a
  tiny random-config Qwen3-VL (2 layers, hidden 256) in the `lerobot` env:
  5×256×256 images → 406 tokens total; waypoint pred [B,4]; finite loss; grads
  flow to all trainable params (side memory + heads + LoRA; base frozen);
  memory carries and changes across steps; save/load roundtrip OK.
- **Real-weights GPU test passed** (`/home/spc/LLMs/Qwen3-VL-2B-Instruct`,
  LoRA-32, base frozen): 4.95 GB loaded, 5.75 GB peak fwd+bwd. Params:
  2.13B base + 335.5M new (300.7M side memory, 34.9M LoRA). Identical
  predictions with/without carried memory at init — confirms the zero-init
  injection design (model == base VLM at step 0).
- **Real-data mini training run passed**: BattlefieldKitDesert extracted
  (455 episodes, 1957 samples → `data/uav_dataset/trainset_battlefield.json`),
  31 optimizer steps over 5 episodes; waypoint loss 2.24 → 0.15–0.6
  (angle 0.066 → 0.017, norm 2.17 → 0.13). Checkpoint save/load verified.
- **8 GB GPU (RTX 4060 Ti) constraints, measured**: fp32 AdamW states for the
  335M trainable params OOM → use `--adam8bit` (bitsandbytes, installed in
  lerobot env); TBPTT `chunk_size >= 2` OOMs (graph spans steps) →
  `chunk_size 1` + `--grad_accum_steps 8` fits. Real training needs a bigger
  GPU for the intended chunk_size=8; this card is fine for debugging/eval.
  Also set `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- NOT yet verified: closed-loop eval (needs AirSim envs + a combined env with
  llamavid deps AND transformers>=4.57 — the traj predictor import via
  travel_util pulls in llamavid; test this pairing early).

## 7. Ablations / experiment matrix

| Variant | History source | Purpose |
|---|---|---|
| LLaMA-UAV (reference) | text only | baseline numbers from paper |
| COMPACT-UAV w/ text history | text + memory | drop-in replacement, same interface |
| COMPACT-UAV memory-only | memory only | isolates memory contribution |
| COMPACT-UAV 2B vs 4B | — | capacity scaling |

## 8. Risks / open questions

1. **Capacity**: 2B/4B vs 7B on waypoint regression — empirical.
2. **Previous-action embedding**: the only genuinely new module (4-dim → FiLM-GRU control MLP).
3. **Eval assistance**: reference eval uses `--always_help True --use_gt True` (GT-derived stage hints).
   Keep identical flags for fair comparison; optionally add a no-GT-hint run (`--use_gt False`,
   rule/DINO-based hints).
4. **Traj predictor distribution shift**: it was trained on LLaMA-UAV waypoint outputs; if
   COMPACT-UAV's waypoint distribution differs a lot, retrain stage 2 on COMPACT-UAV outputs.
5. **Disk budget**: ~483 GB archives + extraction won't fit simultaneously in 802 GB — extract
   map-by-map, delete archives progressively.
6. **TBPTT chunk vs episode alignment**: chunk boundaries must fall on episode steps; never split a
   single step's 5-view input across chunks; reset memory at episode boundaries.
