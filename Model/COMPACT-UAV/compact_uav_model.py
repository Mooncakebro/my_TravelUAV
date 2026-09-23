"""
COMPACT-UAV model: JAMELCompactWrapper + waypoint regression head.

Ports the TravelUAV/LLaMA-UAV sentinel-slot action interface onto the
COMPACT architecture (Qwen3-VL + per-layer side memory):

  - One sentinel slot is appended at the end of the prompt; its input
    embedding is replaced by a learned `waypoint_emb` vector (via the
    `embed_override` hook in the copied compact/model.py).
  - The final-layer (post-norm) hidden state at the sentinel position feeds
    `waypoints_fc` (hidden -> hidden/2 -> 64, ReLU) + `waypoints_output`
    (64 -> 4), producing (unit direction xyz, distance) — mirroring
    Model/LLaMA-UAV/llamavid/model/language_model/llava_llama_uav.py.
  - Loss = cosine-direction loss on xyz + L1 on distance
    (+ COMPACT auxiliary losses: lambda_obs * L_obs + lambda_nll * L_nll
    + lambda_mem * L_mem).
  - The previous action fed to the FiLM-GRU predict step is the previous
    4-dim waypoint embedded by `prev_action_mlp` (4 -> hidden), replacing
    COMPACT's text-action token pooling. Episode start uses a learned
    `null_action` vector.

The base VLM is untouched (frozen by default, optional LoRA); only the
side memories, waypoint head, prev_action_mlp and null_action are new
trainable parameters.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compact.config import CompactConfig
from compact.model import JAMELCompactWrapper
from compact.lora import lora_enabled


class CompactUAVModel(JAMELCompactWrapper):
    """COMPACT wrapper with a waypoint regression head for UAV-VLN."""

    def __init__(
        self,
        config: CompactConfig,
        lora_adapter_path: str | Path | None = None,
        lora_is_trainable: bool = True,
        use_memory: bool = True,
        waypoint_loss_scale: float = 1.0,
    ):
        super().__init__(
            config,
            lora_adapter_path=lora_adapter_path,
            lora_is_trainable=lora_is_trainable,
        )
        d = self.hidden_dim
        self.use_memory = use_memory
        self.waypoint_loss_scale = waypoint_loss_scale

        # ── Sentinel-slot waypoint head (mirrors llava_llama_uav.py) ──
        self.waypoint_emb = nn.Embedding(1, d)
        self.waypoints_fc = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 64),
        )
        self.waypoints_output = nn.Linear(64, 4)

        # ── Previous-action embedding for the FiLM-GRU control input ──
        self.prev_action_mlp = nn.Sequential(
            nn.Linear(4, d),
            nn.Tanh(),
            nn.Linear(d, d),
        )
        self.null_action = nn.Parameter(torch.zeros(4))

        llm_dtype = next(self.llm.parameters()).dtype
        for module in (self.waypoint_emb, self.waypoints_fc,
                       self.waypoints_output, self.prev_action_mlp):
            module.to(dtype=llm_dtype)
        self.null_action.data = self.null_action.data.to(llm_dtype)

    # ── Previous action encoding ──

    def encode_prev_action(
        self,
        prev_waypoints: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """prev_waypoints: [B, 4] or None (episode start -> learned null)."""
        if prev_waypoints is None:
            prev_waypoints = self.null_action.unsqueeze(0).expand(batch_size, -1)
        prev_waypoints = prev_waypoints.to(
            device=self._module_device(self.prev_action_mlp),
            dtype=self.null_action.dtype,
        )
        return self.prev_action_mlp(prev_waypoints)  # [B, d]

    # ── Waypoint forward ──

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentinel_mask: torch.Tensor,
        prev_waypoints: Optional[torch.Tensor] = None,
        memory_states: Optional[List[torch.Tensor]] = None,
        variance_states: Optional[List[torch.Tensor]] = None,
        e_prev_list: Optional[List[Optional[torch.Tensor]]] = None,
        waypoint_labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        observation_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Args:
            sentinel_mask:   [B, N] bool — True at the waypoint sentinel slot.
            prev_waypoints:  [B, 4] previous 4-dim waypoint (None = episode start).
            waypoint_labels: [B, 4] GT (unit direction xyz, distance); enables loss.
            observation_mask: [B, N] — sentinel slot must be 0 here (excluded
                             from memory observation pooling).

        Returns dict: predicted_waypoints [B, 4], new_memory, new_variance,
        e_list, and (if labels given) loss + loss_dict.
        """
        B = input_ids.shape[0]
        device = input_ids.device

        if not self.use_memory:
            memory_states, variance_states = None, None
            e_prev_list = None

        action_embed_input = self.encode_prev_action(prev_waypoints, B, device)

        result = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_embed_input=action_embed_input,
            memory_states=memory_states,
            variance_states=variance_states,
            e_prev_list=e_prev_list,
            observation_mask=observation_mask,
            mm_token_type_ids=mm_token_type_ids,
            labels=None,  # no CE loss — waypoint regression only
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            embed_override=(sentinel_mask, self.waypoint_emb.weight[0]),
            skip_lm_head=True,
            **kwargs,
        )

        hidden = result["hidden_states"]  # [B, N, d] post final norm
        wp_h = hidden[sentinel_mask.to(hidden.device)]  # [B, d]
        predicted = self.waypoints_output(self.waypoints_fc(wp_h))  # [B, 4]

        out = {
            "predicted_waypoints": predicted,
            "new_memory": result["new_memory"],
            "new_variance": result["new_variance"],
            "e_list": result["e_list"],
        }

        if waypoint_labels is not None:
            labels = waypoint_labels.to(
                device=predicted.device, dtype=torch.float32,
            )
            pred_f = predicted.float()
            angle_loss = (1 - F.cosine_similarity(
                pred_f[:, :3], labels[:, :3], dim=-1,
            )).mean()
            norm_loss = F.l1_loss(pred_f[:, 3], labels[:, 3])
            loss_wp = self.waypoint_loss_scale * (angle_loss + norm_loss)

            # COMPACT aux losses + memory L2 (mirrors compact/loss.py weights)
            loss_obs = result["loss_obs"].float()
            loss_nll = result["loss_nll"].float()
            loss_mem = torch.stack([
                m.float().pow(2).sum() for m in result["new_memory"]
            ]).mean()

            total = (
                loss_wp
                + self.config.lambda_obs * loss_obs
                + self.config.lambda_nll * loss_nll
                + self.config.lambda_mem * loss_mem
            )
            out["loss"] = total
            out["loss_dict"] = {
                "total": total.detach(),
                "waypoint": loss_wp.detach(),
                "angle": angle_loss.detach(),
                "norm": norm_loss.detach(),
                "obs": loss_obs.detach(),
                "nll": loss_nll.detach(),
                "mem_l2": loss_mem.detach(),
            }

        return out

    # ── Inference helper ──

    @torch.no_grad()
    def predict_waypoint(self, memory_states, variance_states, e_prev_list,
                         **forward_kwargs) -> Tuple[torch.Tensor, list, list, list]:
        """Single eval step. Returns (raw 4-dim waypoint, new_memory,
        new_variance, e_list). Caller renormalizes dir * distance."""
        was_training = self.training
        self.eval()
        out = self.forward(
            memory_states=memory_states,
            variance_states=variance_states,
            e_prev_list=e_prev_list,
            **forward_kwargs,
        )
        if was_training:
            self.train()
        return (
            out["predicted_waypoints"],
            out["new_memory"],
            out["new_variance"],
            out["e_list"],
        )

    # ── Save / Load ──

    def _uav_head_state(self) -> dict:
        return {
            "waypoint_emb": self.waypoint_emb.state_dict(),
            "waypoints_fc": self.waypoints_fc.state_dict(),
            "waypoints_output": self.waypoints_output.state_dict(),
            "prev_action_mlp": self.prev_action_mlp.state_dict(),
            "null_action": self.null_action.detach().cpu(),
            "use_memory": self.use_memory,
            "waypoint_loss_scale": self.waypoint_loss_scale,
        }

    def save_pretrained(self, save_directory, state_dict=None):
        super().save_pretrained(save_directory, state_dict=state_dict)
        save_path = Path(save_directory) / "side_memory"
        save_path.mkdir(exist_ok=True)
        torch.save(self._uav_head_state(), save_path / "uav_heads.pt")
        print(f"[save] COMPACT-UAV waypoint heads saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_directory, config_override=None,
                        model_parallel_override=None) -> "CompactUAVModel":
        model = super().from_pretrained(
            load_directory,
            config_override=config_override,
            model_parallel_override=model_parallel_override,
        )
        head_path = Path(load_directory) / "side_memory" / "uav_heads.pt"
        if head_path.exists():
            state = torch.load(head_path, map_location="cpu", weights_only=False)
            model.waypoint_emb.load_state_dict(state["waypoint_emb"])
            model.waypoints_fc.load_state_dict(state["waypoints_fc"])
            model.waypoints_output.load_state_dict(state["waypoints_output"])
            model.prev_action_mlp.load_state_dict(state["prev_action_mlp"])
            model.null_action.data = state["null_action"].to(
                model.null_action.device, model.null_action.dtype,
            )
            model.use_memory = state.get("use_memory", True)
            model.waypoint_loss_scale = state.get("waypoint_loss_scale", 1.0)
            print(f"[load] COMPACT-UAV waypoint heads loaded from {head_path}")
        else:
            print(f"[load] WARNING: no uav_heads.pt in {head_path.parent}; "
                  "waypoint head is randomly initialized")
        return model
