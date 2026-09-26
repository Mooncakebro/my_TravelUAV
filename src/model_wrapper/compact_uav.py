"""
COMPACT-UAV closed-loop wrapper — drop-in replacement for TravelModelWrapper.

Same public interface (prepare_inputs / run / predict_done / eval) so
src/vlnce_src/eval.py and dagger.py can switch policies with --policy.

Differences from TravelModelWrapper:
  - the MLLM is CompactUAVModel (Qwen3-VL + side memory + waypoint head);
  - prompts are built with the Qwen3-VL chat template (5 view images);
  - COMPACT memory is carried across steps within an episode and reset when
    an episode slot is replaced or truncated;
  - the previous predicted 4-dim waypoint (the policy action, before trajectory
    refinement) is fed back as the FiLM-GRU control input.

The GroundingDINO stop monitor is reused unchanged. The trajectory predictor is
intentionally bypassed: the predicted 4D waypoint is the policy action for this
  experiment. AirSim requires a short XYZ path, so that one action is converted
  to world coordinates and linearly sampled for the simulator's five-point API.

NOTE(env): COMPACT-UAV evaluation needs the Qwen3-VL/PEFT stack and the
existing AirSim/GroundingDINO runtime. It does not import the LLaMA-UAV
trajectory predictor because that model is bypassed for this action ablation.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(str(os.getcwd())).resolve()))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'Model' / 'COMPACT-UAV'))

from src.model_wrapper.base_model import BaseModelWrapper
from src.vlnce_src.dino_monitor_online import DinoMonitor


def _rotation_matrix_from_vector(x, y):
    v_x = np.asarray([x, y, 0.0], dtype=np.float64)
    v_x /= np.linalg.norm(v_x) + 1e-12
    v_y = np.asarray([-v_x[1], v_x[0], 0.0])
    v_y /= np.linalg.norm(v_y) + 1e-12
    return np.column_stack((v_x, v_y, [0.0, 0.0, 1.0]))


def _transform_point(point, rotation_matrix):
    return np.dot(point, rotation_matrix)


class CompactUAVModelWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        from compact_uav_model import CompactUAVModel
        self._rotation_matrix_from_vector = _rotation_matrix_from_vector
        self._transform_point = _transform_point

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CompactUAVModel.from_pretrained(model_args.model_path)
        self.model.to(self.device)
        self.tokenizer = self.model.tokenizer
        self.processor = self.model.processor

        self.model_args = model_args
        self.data_args = data_args
        self.dino_moinitor = None

        # per-slot episode state (memory carried across steps)
        self._ep_ids = None        # id() of each slot's episode list
        self._ep_lens = None       # last seen length per slot
        self._memory = None        # (memory_states, variance_states, e_prev)
        self._prev_wp = None       # [B, 4] previous policy action per slot
        self.last_actions_4d = None # CPU [B, 4], useful for logging/evaluation

    # ── prompt construction (mirrors travel_util.prepare_data_to_inputs) ──

    def _build_prompt(self, episode, target_point, assist_notice):
        # Training strips the dataset's literal <image> marker because Qwen's
        # chat template inserts the actual image placeholders. Keep eval exact.
        from dataset_uav import normalize_instruction
        instruction = normalize_instruction(episode[-1]['instruction'])
        if assist_notice is not None:
            stage = assist_notice
        else:
            stage = 'cruise' if len(episode) > 20 else 'take off'

        rot = np.array(episode[0]['sensors']['imu']["rotation"])
        pos = np.array(episode[0]['sensors']['state']['position'])
        deltas = [np.array(src['sensors']['state']['position']) - pos
                  for src in episode if 'rgb' in src]
        history_waypoint = np.array([rot.T @ d for d in deltas])

        target_rel = np.array(rot.T @ (target_point - pos))
        rotation_to_target = self._rotation_matrix_from_vector(
            target_rel[0], target_rel[1])
        history_waypoint = self._transform_point(history_waypoint,
                                                 rotation_to_target)

        if len(history_waypoint) >= 2:
            delta = history_waypoint[-1] - history_waypoint[-2]
        else:
            delta = np.array([0, 0, -4.5])
        delta = delta / (np.linalg.norm(delta) + 1e-8)
        delta_str = ','.join(str(round(v, 1)) for v in delta)
        cur_str = ','.join(str(round(v, 1)) for v in history_waypoint[-1])

        prompt_text = (
            f'Stage:{stage}\n\n'
            f'Previous displacement:{delta_str}\n\n'
            f'Current position:{cur_str}\n\n'
            f'Current image (in order: front, left, right, rear, down):\n\n'
            f'Instruction:{instruction}'
        )
        return prompt_text, rotation_to_target

    def _process_one(self, prompt_text, images):
        """Qwen3-VL processor call for a single step. images: 5 HWC arrays."""
        from PIL import Image
        pil_images = [Image.fromarray(im) for im in images]
        head, instruction = prompt_text.split(
            'Current image (in order: front, left, right, rear, down):')
        content = [{'type': 'text',
                    'text': head + 'Current image (in order: front, left, '
                                   'right, rear, down):'}]
        content += [{'type': 'image'} for _ in pil_images]
        content.append({'type': 'text',
                        'text': '\n\nInstruction:' + instruction.split(
                            'Instruction:')[-1]})
        messages = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=pil_images,
                                return_tensors='pt', padding=True)
        # Keep the exact truncation/sentinel budget used by dataset_uav.py.
        max_length = int(getattr(self.model.config, 'max_length', 8192))
        if inputs['input_ids'].shape[1] > max_length - 1:
            excess = inputs['input_ids'].shape[1] - (max_length - 1)
            inputs['input_ids'] = inputs['input_ids'][:, excess:]
            inputs['attention_mask'] = inputs['attention_mask'][:, excess:]
            if inputs.get('mm_token_type_ids') is not None:
                inputs['mm_token_type_ids'] = \
                    inputs['mm_token_type_ids'][:, excess:]
        return inputs

    # ── memory state management ──

    def _update_episode_tracking(self, episodes):
        """Reset memory rows for slots whose episode was replaced/truncated."""
        B = len(episodes)
        reset_mask = [True] * B
        if self._ep_ids is not None and len(self._ep_ids) == B:
            reset_mask = [
                id(episodes[i]) != self._ep_ids[i]
                or len(episodes[i]) < self._ep_lens[i]
                for i in range(B)
            ]
        self._ep_ids = [id(ep) for ep in episodes]
        self._ep_lens = [len(ep) for ep in episodes]

        if self._memory is None or self._memory[0][0].shape[0] != B:
            self._memory = self.model.init_memory(B, self.device) + (None,)
            self._prev_wp = None
            return

        if any(reset_mask):
            memory_states, variance_states, e_prev = self._memory
            fresh_m, fresh_p = self.model.init_memory(B, self.device)
            mask = torch.tensor(reset_mask, device=self.device)
            for l in range(len(memory_states)):
                memory_states[l] = torch.where(
                    mask.view(B, 1, 1), fresh_m[l], memory_states[l])
                variance_states[l] = torch.where(
                    mask.view(B, 1), fresh_p[l], variance_states[l])
            self._memory = (memory_states, variance_states, None)
            if self._prev_wp is not None:
                for i, r in enumerate(reset_mask):
                    if r:
                        self._prev_wp[i] = 0.0

    # ── BaseModelWrapper interface ──

    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        self._update_episode_tracking(episodes)

        processed, rot_to_targets = [], []
        for i, ep in enumerate(episodes):
            prompt_text, rot = self._build_prompt(
                ep, target_positions[i],
                assist_notices[i] if assist_notices is not None else None)
            images = None
            for src in ep[::-1]:
                if 'rgb' in src:
                    images = src['rgb']
                    break
            processed.append(self._process_one(prompt_text, images))
            rot_to_targets.append(rot)

        # batch: right-pad input_ids to longest, concat pixel_values
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(p['input_ids'].shape[1] for p in processed)
        B = len(processed)

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        mm_ids = None
        if all(p.get('mm_token_type_ids') is not None for p in processed):
            mm_ids = torch.zeros((B, max_len), dtype=torch.long)
        pixel_values = []
        image_grid_thw = []
        for b, p in enumerate(processed):
            n = p['input_ids'].shape[1]
            input_ids[b, :n] = p['input_ids'][0]
            attention_mask[b, :n] = p['attention_mask'][0]
            if mm_ids is not None:
                mm_ids[b, :n] = p['mm_token_type_ids'][0]
            pixel_values.append(p['pixel_values'])
            image_grid_thw.append(p['image_grid_thw'])

        # append the sentinel slot (same position for the whole batch)
        sentinel_col = torch.full((B, 1), pad_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, sentinel_col], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(B, 1, dtype=torch.long)], dim=1)
        if mm_ids is not None:
            mm_ids = torch.cat(
                [mm_ids, torch.zeros(B, 1, dtype=torch.long)], dim=1)
        sentinel_mask = torch.zeros(B, max_len + 1, dtype=torch.bool)
        sentinel_mask[:, -1] = True
        observation_mask = attention_mask.clone()
        observation_mask[:, -1] = 0

        inputs = {
            'input_ids': input_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device),
            'sentinel_mask': sentinel_mask.to(self.device),
            'observation_mask': observation_mask.to(self.device),
            'pixel_values': torch.cat(pixel_values, dim=0).to(self.device),
            'image_grid_thw': torch.cat(image_grid_thw, dim=0).to(self.device),
        }
        if mm_ids is not None:
            inputs['mm_token_type_ids'] = mm_ids.to(self.device)

        return inputs, rot_to_targets

    def run(self, inputs, episodes, rot_to_targets):
        memory_states, variance_states, e_prev = self._memory
        prev_wp = None
        if self._prev_wp is not None:
            prev_wp = self._prev_wp.to(self.device)

        predicted, new_m, new_v, new_e = self.model.predict_waypoint(
            memory_states=memory_states,
            variance_states=variance_states,
            e_prev_list=e_prev,
            prev_waypoints=prev_wp,
            **inputs,
        )
        self._memory = (new_m, new_v, new_e)
        self._prev_wp = predicted.detach().float().cpu()

        # The raw 4D output is the action. Convert its target-frame direction
        # and distance into one world-frame XYZ target; do not invoke the
        # downstream trajectory-refinement model in this policy.
        waypoints = predicted.cpu().to(dtype=torch.float32).numpy()
        self.last_actions_4d = waypoints.copy()
        world_paths = []
        for waypoint, episode, rot_to_target in zip(
                waypoints, episodes, rot_to_targets):
            local_target = (waypoint[:3]
                            / (1e-6 + np.linalg.norm(waypoint[:3]))
                            * waypoint[3])
            rot_0 = np.asarray(episode[0]['sensors']['imu']['rotation'])
            rot = np.asarray(episode[-1]['sensors']['imu']['rotation'])
            pos = np.asarray(episode[-1]['sensors']['state']['position'])
            # target-frame -> initial local frame -> current local frame.
            current_target = rot.T @ rot_0 @ rot_to_target @ local_target
            world_target = rot @ current_target + pos
            # move_path_by_waypoints expects five points and indexes all of
            # them. Linear interpolation is transport-only; the endpoint is
            # still exactly the raw policy action.
            world_paths.append(np.linspace(pos, world_target, 5,
                                           dtype=np.float32))
        return world_paths

    def eval(self):
        self.model.eval()

    def predict_done(self, episodes, object_infos):
        prediction_dones = []
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            prediction_dones.append(
                self.dino_moinitor.get_dino_results(episodes[i],
                                                    object_infos[i]))
        return prediction_dones
