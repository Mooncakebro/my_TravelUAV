"""
Loss functions for JAMEL-COMPACT v2.

Four-term loss:
  1. L_act  — Coverage-weighted Cross-Entropy (F7: sample_weights from coverage delta)
  2. L_obs  — Observation prediction MSE (U3: obs_model predicts z_target)
  3. L_nll  — Gaussian NLL for variance calibration (U2: calibrates R_psi)
  4. L_mem  — L2 regularization on memory states

Changes from v1:
  - Removed: Bernoulli entropy (was for pinned confidence C, not applicable to variance P)
  - Removed: Uncertainty calibration MSE (replaced by L_nll which calibrates R directly)
  - Added:   Coverage-weighted CE (F7) — per-sample weights from coverage delta
  - Added:   L_obs and L_nll collected per-layer from forward (U2/U3)
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from .config import CompactConfig


def compute_compact_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    memory_states: List[torch.Tensor],
    variance_states: List[torch.Tensor],
    config: Optional[CompactConfig] = None,
    loss_obs: Optional[torch.Tensor] = None,
    loss_nll: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the JAMEL-COMPACT v2 total loss.

    L_total = L_act + lambda_obs * L_obs + lambda_nll * L_nll + lambda_mem * L_mem

    Args:
        logits:            [B, N, vocab_size] — model output logits
        labels:            [B, N] — token labels (-100 for ignore)
        memory_states:     List of [B, N_m, d_mem] — updated memory per layer
        variance_states: List of [B, N_m] — variance P per layer
        config:            CompactConfig with loss weights
        loss_obs:          scalar — per-layer averaged observation MSE (from forward)
        loss_nll:          scalar — per-layer averaged Gaussian NLL (from forward)
        sample_weights:    [B] — per-sample loss weights (F7: 1 + eta * max(cov_delta, 0))

    Returns:
        (total_loss, loss_dict)
    """
    if config is None:
        config = CompactConfig()

    device = logits.device
    dtype = logits.dtype

    # ── 1. L_act: Coverage-weighted Cross-Entropy (F7) ──
    # Shift labels: predict token t+1 from token t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    if sample_weights is not None:
        # F7: Per-sample weighted CE
        # Compute per-sample CE (reduction='none', then mean over valid tokens)
        ce_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none',
        ).view(shift_labels.shape)  # [B, N-1]

        # Mask: valid labels (not -100)
        valid_mask = (shift_labels != -100).float()  # [B, N-1]
        # Per-sample sum / count
        ce_sum = ce_per_token.sum(dim=1)  # [B]
        valid_count = valid_mask.sum(dim=1).clamp(min=1)  # [B]
        ce_per_sample = ce_sum / valid_count  # [B]

        # F7: Unweighted CE (for logging — compare against weighted)
        loss_action_unweighted = ce_per_sample.mean()

        # Apply sample weights
        weights = sample_weights.to(device=device, dtype=dtype)
        loss_action = (ce_per_sample * weights).sum() / weights.sum().clamp(min=1)
    else:
        loss_action = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss_action_unweighted = loss_action.detach()

    # ── 2. L_obs: Observation prediction MSE (U3) ──
    if loss_obs is None:
        loss_obs = torch.tensor(0.0, device=device, dtype=dtype)
    loss_obs = loss_obs.to(device=device, dtype=dtype)

    # ── 3. L_nll: Gaussian NLL for variance calibration (U2) ──
    if loss_nll is None:
        loss_nll = torch.tensor(0.0, device=device, dtype=dtype)
    loss_nll = loss_nll.to(device=device, dtype=dtype)

    # ── 4. L_mem: L2 regularization on memory states ──
    # L_mem = (1/L) Σ_l ||M_l||²_2, matching the method definition.
    loss_mem = torch.tensor(0.0, device=device, dtype=dtype)
    L = len(memory_states)
    for M in memory_states:
        loss_mem = loss_mem + M.float().pow(2).sum().to(
            device=device, dtype=dtype,
        )
    loss_mem = loss_mem / L

    # ── Total ──
    loss_total = (
        loss_action
        + config.lambda_obs * loss_obs
        + config.lambda_nll * loss_nll
        + config.lambda_mem * loss_mem
    )

    loss_dict = {
        "total": loss_total.detach(),
        "action": loss_action.detach(),
        "action_unweighted": loss_action_unweighted.detach(),
        "obs": loss_obs.detach(),
        "nll": loss_nll.detach(),
        "mem_l2": loss_mem.detach(),
    }

    return loss_total, loss_dict
