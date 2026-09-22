"""
COMPACT-UAV training dataset: episode-grouped TravelUAV data.

Replicates the reference pipeline (Model/LLaMA-UAV/llamavid/train/train_uav/
train_uav_notice.py) exactly for prompt text and waypoint-label math:

  - label at step f (1-based) = next frame's GT position relative to the
    current position, rotated into the target-aligned frame
    (rotation_matrix_from_vector on the episode's final xy), stored as
    (unit direction xyz, distance).
  - prompt = "Stage:... / Previous displacement:... / Current position:... /
    Current image: <5 views> / Instruction: ..." — same strings as the
    reference, but rendered through the Qwen3-VL chat template with 5 image
    placeholders (front, left, right, rear, down).

Differences from the reference:
  - Images are loaded from raw PNGs and processed with the Qwen3-VL
    AutoProcessor on the fly (the shipped rgb_imgs.tensor is EVA-specific).
  - Samples are yielded as episode-aligned step sequences for TBPTT, plus a
    teacher-forced previous-action vector (previous step's label).
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

CAMERA_FOLDERS = ['frontcamera', 'leftcamera', 'rightcamera',
                  'rearcamera', 'downcamera']
CAMERA_NAMES = ['front', 'left', 'right', 'rear', 'down']


# ── Geometry helpers (identical to train_uav_notice.py / travel_util.py) ──

def rotation_matrix_from_vector(x, y):
    v_x = np.array([x, y, 0])
    v_x = v_x / np.linalg.norm(v_x)
    v_y = np.array([-v_x[1], v_x[0], 0])
    v_y = v_y / np.linalg.norm(v_y)
    v_z = np.array([0, 0, 1])
    return np.column_stack((v_x, v_y, v_z))


def transform_point(point, rotation_matrix):
    return np.dot(point, rotation_matrix)


def waypoint2angle(waypoints):
    angle_and_norm = []
    for waypoint in waypoints:
        norm = np.linalg.norm(waypoint)
        angle = waypoint / (norm + 1e-6)
        angle_and_norm.append([angle[0], angle[1], angle[2], norm])
    return np.array(angle_and_norm)


def get_stage(trajectory, frame_num):
    """Ported verbatim from train_uav_notice.py: get_stage."""
    def turning_stage(p0, p1, p2):
        prev_vec = p1 - p0
        now_vec = p2 - p1
        delta_angle = np.arccos(
            np.dot(prev_vec, now_vec) / (np.linalg.norm(prev_vec) + 1e-6)
            / (np.linalg.norm(now_vec) + 1e-6)) * 180 / np.pi
        if delta_angle > 25 and delta_angle < 120:
            if int(np.cross(prev_vec, now_vec)) > 0:
                return 'right'
            else:
                return 'left'
        return 'cruise'

    assist = 0
    trajectory = np.asarray(trajectory)
    z_values = trajectory[:, 2]
    now_z = z_values[frame_num - 1]
    future_z = z_values[min(frame_num + 2, len(z_values) - 1)]
    stage = 'cruise'
    if now_z - future_z > 5:
        stage = 'take off'
    elif now_z - future_z < -5:
        stage = 'landing'
    prev_vec = np.array([0, 0, 0])
    if frame_num >= 2 and frame_num < len(trajectory):
        prev_vec = np.array(trajectory[frame_num - 1, :3]
                            - trajectory[frame_num - 2, :3])
        if stage == 'cruise':
            stage = turning_stage(trajectory[frame_num - 2, :2],
                                  trajectory[frame_num - 1, :2],
                                  trajectory[frame_num, :2])
    if frame_num >= 1 and frame_num < len(trajectory) - 1:
        future_p = trajectory[frame_num + 1, :2]
        next_p = trajectory[frame_num, :2]
        next_stage = turning_stage(trajectory[frame_num - 1, :2],
                                   next_p, future_p)
        future_z = z_values[min(frame_num + 3, len(z_values) - 1)]
        if trajectory[frame_num, 2] - future_z < -5:
            next_stage = 'landing'
        if next_stage == 'left' or next_stage == 'right' or next_stage == 'landing':
            assist = 1
    return stage, prev_vec, assist


def build_prompt_text(stage: str, delta_str: str, cur_str: str,
                      instruction: str) -> str:
    """Same fields as the reference imgsp_uav prompt (travel_util.py:213)."""
    return (
        f'Stage:{stage}\n\n'
        f'Previous displacement:{delta_str}\n\n'
        f'Current position:{cur_str}\n\n'
        f'Current image (in order: front, left, right, rear, down):\n\n'
        f'Instruction:{instruction}'
    )


# ── Episode dataset ──

class UAVEpisodeDataset(Dataset):
    """One item = one episode's worth of consecutive steps (for TBPTT).

    Args:
        data_path:    trainset.json — list of {"json": rel/path/merged_data.json,
                      "frame": N} (frames are 1-based into merged trajectory).
        dataset_path: root containing <Map>/<episode>/ folders with
                      merged_data.json + the 5 camera PNG folders.
        max_episode_steps: optional cap on steps per episode (None = all).
    """

    def __init__(self, data_path: str, dataset_path: str,
                 max_episode_steps: Optional[int] = None):
        self.dataset_path = dataset_path
        self.max_episode_steps = max_episode_steps
        with open(data_path, 'r') as f:
            entries = json.load(f)

        grouped: Dict[str, List[int]] = {}
        for e in entries:
            grouped.setdefault(e['json'], []).append(e['frame'])
        self.episodes = sorted(
            (json_rel, sorted(frames)) for json_rel, frames in grouped.items()
        )
        print(f"[dataset] {len(entries)} steps in {len(self.episodes)} episodes "
              f"from {data_path}")

    def __len__(self):
        return len(self.episodes)

    def _image_paths(self, traj_dir: str, real_index: int) -> List[str]:
        name = str(real_index).zfill(6) + '.png'
        return [os.path.join(traj_dir, cam, name) for cam in CAMERA_FOLDERS]

    def __getitem__(self, idx) -> Optional[dict]:
        json_rel, frames = self.episodes[idx]
        traj_dir = os.path.join(self.dataset_path, *json_rel.split('/')[:-1])
        json_path = os.path.join(self.dataset_path, json_rel)
        try:
            with open(json_path, 'r') as f:
                merged = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[dataset] skip episode {json_rel}: {e}")
            return None

        trajectory = np.asarray(merged['trajectory'], dtype=np.float64)
        index_list = merged['index']
        # instruction: strip the "<image>\n" prefix from the human turn
        instruction = merged['conversations'][0]['value']
        if instruction.startswith('<image>'):
            instruction = instruction[len('<image>'):].strip()

        x_t, y_t = trajectory[-1][0], trajectory[-1][1]
        rot_to_target = rotation_matrix_from_vector(x_t, y_t)

        steps = []
        for frame_num in frames:
            if self.max_episode_steps and len(steps) >= self.max_episode_steps:
                break
            if frame_num < 1 or frame_num > len(index_list):
                continue
            real_index = index_list[frame_num - 1]
            img_paths = self._image_paths(traj_dir, real_index)
            if not all(os.path.isfile(p) for p in img_paths):
                # truncate episode at the first missing frame (prev-action
                # chain must stay consecutive)
                print(f"[dataset] {json_rel} frame {frame_num}: missing PNGs, "
                      "truncating episode")
                break

            stage, prev_vec, _assist = get_stage(trajectory, frame_num)

            prev_delta = transform_point(prev_vec, rot_to_target)
            prev_delta = prev_delta / (np.linalg.norm(prev_delta) + 1e-8)
            delta_str = ','.join(str(round(v, 1)) for v in prev_delta)

            cur_pos = transform_point(trajectory[frame_num - 1][:3], rot_to_target)
            cur_str = ','.join(str(round(v, 1)) for v in cur_pos)

            # waypoint label: next <=7 GT positions relative to current pos,
            # rotated into the target-aligned frame, (unit dir, norm)
            cur_xyz = trajectory[frame_num - 1, 0:3]
            future = trajectory[frame_num:min(len(trajectory), frame_num + 7), 0:3]
            if len(future) == 0:
                future = np.array([cur_xyz] * 7)
            elif len(future) < 7:
                future = np.array([future[i] if i < len(future) else future[-1]
                                   for i in range(7)])
            future = future - cur_xyz
            future = transform_point(future, rot_to_target)
            label = waypoint2angle(future)[0]  # [4]

            steps.append({
                'frame': frame_num,
                'image_paths': img_paths,
                'prompt_text': build_prompt_text(stage, delta_str, cur_str,
                                                 instruction),
                'waypoint_label': label.astype(np.float32),
            })

        if len(steps) < 2:
            return None

        # teacher-forced previous action: previous step's label
        for i, step in enumerate(steps):
            step['prev_waypoint'] = (steps[i - 1]['waypoint_label']
                                     if i > 0 else None)
            step['is_episode_start'] = (i == 0)

        return {
            'episode': json_rel,
            'steps': steps,
            'rot_to_target': rot_to_target,
        }


# ── Step processing with the Qwen3-VL processor ──

def process_step(step: dict, processor, tokenizer, max_length: int = 8192,
                 sentinel_token_id: Optional[int] = None) -> dict:
    """Build model inputs for one step.

    Returns dict with input_ids [1, N+1], attention_mask, sentinel_mask,
    observation_mask (attention minus sentinel), pixel_values,
    image_grid_thw, mm_token_type_ids (if the processor emits them).
    """
    from PIL import Image

    images = [Image.open(p).convert('RGB') for p in step['image_paths']]
    content = [{'type': 'text',
                'text': step['prompt_text'].split(
                    'Current image (in order: front, left, right, rear, down):'
                )[0] + 'Current image (in order: front, left, right, rear, down):'}]
    content += [{'type': 'image'} for _ in images]
    content.append({'type': 'text',
                    'text': '\n\nInstruction:'
                            + step['prompt_text'].split('Instruction:')[-1]})
    messages = [{'role': 'user', 'content': content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=images, return_tensors='pt',
                       padding=True)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # left-truncate to max_length, keeping room for the sentinel slot
    if input_ids.shape[1] > max_length - 1:
        excess = input_ids.shape[1] - (max_length - 1)
        input_ids = input_ids[:, excess:]
        attention_mask = attention_mask[:, excess:]
        if inputs.get('mm_token_type_ids') is not None:
            inputs['mm_token_type_ids'] = inputs['mm_token_type_ids'][:, excess:]

    # append the sentinel slot (embedding overridden inside the model)
    if sentinel_token_id is None:
        sentinel_token_id = tokenizer.pad_token_id
        if sentinel_token_id is None:
            sentinel_token_id = tokenizer.eos_token_id
    B, N = input_ids.shape
    sentinel_col = torch.full((B, 1), sentinel_token_id, dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids, sentinel_col], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype)], dim=1)

    sentinel_mask = torch.zeros(B, N + 1, dtype=torch.bool)
    sentinel_mask[:, -1] = True
    observation_mask = attention_mask.clone()
    observation_mask[:, -1] = 0  # sentinel excluded from memory pooling

    out = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'sentinel_mask': sentinel_mask,
        'observation_mask': observation_mask,
        'pixel_values': inputs.get('pixel_values'),
        'image_grid_thw': inputs.get('image_grid_thw'),
        'mm_token_type_ids': inputs.get('mm_token_type_ids'),
        'waypoint_label': torch.as_tensor(
            step['waypoint_label'], dtype=torch.float32).unsqueeze(0),
        'prev_waypoint': (None if step['prev_waypoint'] is None else
                          torch.as_tensor(step['prev_waypoint'],
                                          dtype=torch.float32).unsqueeze(0)),
    }
    return out
