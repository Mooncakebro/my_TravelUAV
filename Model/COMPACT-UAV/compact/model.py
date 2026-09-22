"""
JAMEL-COMPACT model: wraps a pretrained LLM with per-layer side memory.

This module implements the core architecture:
  - FiLMGRUCell: action-modulated GRU for memory state prediction
  - SideMemoryModule: per-layer memory with Predict→Correct→Inject cycle
  - JAMELCompactWrapper: wraps a HuggingFace LLM (e.g. Qwen3-VL-2B/8B)

The pretrained LLM's self-attention and FFN are NOT replaced — they are
called in-place.  Only the side memory modules are new parameters.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from .config import CompactConfig
from .lora import (
    configure_lora,
    enable_lora_gradient_checkpointing,
    lora_enabled,
    validate_lora_adapter_path,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FiLM-GRU Cell
# ═══════════════════════════════════════════════════════════════════════════════

class FiLMGRUCell(nn.Module):
    """
    FiLM-modulated GRU cell for memory state prediction.

    The action embedding modulates the GRU state transition via
    Feature-wise Linear Modulation (FiLM):

        γ, β = MLP(a_emb)
        h_new = GRU(W_proj(a_emb), γ ⊙ h_old + β)

    This makes the action an explicit *control variable* that steers
    how memory evolves — analogous to u_t in state-space models.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.film_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
        )
        self.action_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_prev: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_prev: [B*N_m, d_mem] — previous memory state (flattened)
            a_emb:  [B*N_m, d_mem] — action embedding (expanded to all tokens)
        Returns:
            h_new:  [B*N_m, d_mem] — predicted memory state
        """
        gamma_beta = self.film_mlp(a_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h_modulated = gamma * h_prev + beta
        gru_input = self.action_proj(a_emb)
        return self.gru(gru_input, h_modulated)


# ═══════════════════════════════════════════════════════════════════════════════
# Side Memory Module (per layer)
# ═══════════════════════════════════════════════════════════════════════════════

class SideMemoryModule(nn.Module):
    """
    v2: Per-layer side memory with learned Kalman filter.

    Changes from v1:
    - F3: Masked observation pooling (uses attention_mask, no padding)
    - F4: Zero-initialized injection (delta_up=0, learnable gate, model=base at init)
    - U1: Multi-token observation (k=4 learned latent queries → non-rank-1 innovation)
    - U2: Learned Kalman track (variance P replaces pinned confidence C;
          learned Q_theta, R_psi, adaptive K = P_hat/(P_hat+R))
    - U3: Observation model + surprise (obs_model predicts observations;
          surprise e feeds next step's variance inflation)

    Dimension flow:
      • Main stream H:  d (e.g. 2048 or 4096)
      • Memory state M: d_mem (e.g. 512)
      • Observation Z:  h is down-projected first, then pooled to [B, k, d_mem]
      • Variance P:     [B, N_m] (per-slot scalar)
    """

    def __init__(self, layer_idx: int, num_layers: int, hidden_dim: int,
                 mem_dim: int = 512, num_mem: int = 16, num_heads: int = 8,
                 num_obs_tokens: int = 4, config: Optional[CompactConfig] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_mem = num_mem
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim
        self.num_obs_tokens = num_obs_tokens  # U1: k

        # ── Down/up projections (d ↔ d_mem) ──
        # obs_down projects h to d_mem BEFORE observation attention, so the
        # attention operates in the compressed d_mem space (not hidden_dim).
        # This keeps parameter count proportional to d_mem, not d.
        self.obs_down = nn.Linear(hidden_dim, mem_dim)
        self.action_down = nn.Linear(hidden_dim, mem_dim)
        self.h_down = nn.Linear(hidden_dim, mem_dim)
        self.delta_up = nn.Linear(mem_dim, hidden_dim)

        # F4: Zero-initialize delta_up so model = base LLM at init
        nn.init.zeros_(self.delta_up.weight)
        nn.init.zeros_(self.delta_up.bias)

        # delta_up is zero-initialized, so the branch is exactly zero at init.
        # The gate must be nonzero: zeroing both factors makes all injection
        # gradients zero and permanently disconnects memory from action loss.
        gate_init = config.inject_gate_init if config is not None else 0.1
        self.inject_gate = nn.Parameter(torch.tensor([float(gate_init)]))

        # ── Memory Predict: FiLM-GRU (in d_mem) ──
        self.gru = FiLMGRUCell(mem_dim)

        # ── U1: Learned observation queries [k, d_mem] ──
        # (in d_mem space — attention operates in compressed space)
        self.obs_queries = nn.Parameter(torch.randn(num_obs_tokens, mem_dim) * 0.02)

        # ── U1: Multi-token observation attention pooling (in d_mem space) ──
        # Use valid number of heads for d_mem (must divide evenly)
        obs_heads = min(num_heads, mem_dim)
        while obs_heads > 0 and mem_dim % obs_heads != 0:
            obs_heads -= 1
        obs_heads = max(obs_heads, 1)
        self.obs_attn = nn.MultiheadAttention(
            mem_dim, obs_heads, batch_first=True,
        )

        # ── Innovation cross-attention (now with k KV tokens, not 1) ──
        self.mem_cross_attn = nn.MultiheadAttention(
            mem_dim, num_heads, batch_first=True,
        )
        self.innovation_proj = nn.Linear(mem_dim, mem_dim)

        # ── U2: Learned process noise Q_theta: Linear(d_mem → N_m) + softplus ──
        self.Q_theta = nn.Linear(mem_dim, num_mem)

        # ── U2: Learned observation noise R_psi: MLP(d_mem → 128 → N_m) + softplus ──
        self.R_psi = nn.Sequential(
            nn.Linear(mem_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_mem),
        )

        # ── U3: Observation model MLP(d_mem → d_mem → d_mem) ──
        self.obs_model = nn.Sequential(
            nn.Linear(mem_dim, mem_dim),
            nn.GELU(),
            nn.Linear(mem_dim, mem_dim),
        )

        # ── Memory Injection cross-attention (in d_mem) ──
        self.inject_cross_attn = nn.MultiheadAttention(
            mem_dim, num_heads, batch_first=True,
        )
        self.inject_norm = nn.Identity()

        # ── Hierarchical hyperparameters ──
        if config is not None:
            lam_s, lam_m, lam_d = (config.lambda_shallow, config.lambda_mid,
                                   config.lambda_deep)
            inj_s, inj_m, inj_d = (config.inject_shallow, config.inject_mid,
                                   config.inject_deep)
            self.gamma_e = config.gamma_e
            self.surprise_clip = config.surprise_clip
            self.r_min = config.r_min
        else:
            lam_s, lam_m, lam_d = 0.70, 0.85, 0.95
            inj_s, inj_m, inj_d = 0.8, 0.5, 0.3
            self.gamma_e = 1.0
            self.surprise_clip = 10.0
            self.r_min = 0.01

        if layer_idx < num_layers // 3:
            lam_init, inj_val = lam_s, inj_s
        elif layer_idx < 2 * num_layers // 3:
            lam_init, inj_val = lam_m, inj_m
        else:
            lam_init, inj_val = lam_d, inj_d

        # U2: Learnable per-slot variance decay λ_l (sigmoid-constrained)
        # sigmoid(λ_l_raw) = lam_init → λ_l_raw = logit(lam_init)
        import math
        lam_clamped = max(min(lam_init, 0.999), 0.001)
        lam_raw_init = math.log(lam_clamped / (1 - lam_clamped))
        self.lambda_l_raw = nn.Parameter(torch.full((num_mem,), lam_raw_init))

        # F4: Keep w_max for hierarchical injection scaling
        self.inject_w_max = inj_val

        # ── Learnable initial memory (in d_mem) ──
        self.init_memory = nn.Parameter(torch.randn(num_mem, mem_dim) * 0.02)
        # U2: Initial variance P_0
        self.init_variance = 0.5

    def predict(self, m_prev: torch.Tensor, p_prev: torch.Tensor,
                action_embed: torch.Tensor,
                e_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """FiLM-GRU predict + variance prediction (U2).

        Args:
            m_prev:  [B, N_m, d_mem] — previous memory state
            p_prev:  [B, N_m] — previous variance
            action_embed: [B, d] — raw action embedding
            e_prev:  [B] — previous step's surprise (detached), or None at chunk start

        Returns:
            m_hat: [B, N_m, d_mem] — predicted memory state
            p_hat: [B, N_m] — predicted variance
            q_noise: [B, N_m] — learned process-noise contribution
            surprise_inflation: [B, N_m] — surprise-driven variance contribution
        """
        B, N_m, d_mem = m_prev.shape
        a_down = self.action_down(action_embed)  # [B, d_mem]

        # FiLM-GRU predict (same as v1)
        m_prev_flat = m_prev.reshape(B * N_m, d_mem)
        a_flat = a_down.unsqueeze(1).expand(-1, N_m, -1).reshape(B * N_m, d_mem)
        m_hat = self.gru(m_prev_flat, a_flat).view(B, N_m, d_mem)

        # U2: Variance predict with learned process noise + adaptive inflation
        lam = torch.sigmoid(self.lambda_l_raw)  # [N_m]
        q_noise = F.softplus(self.Q_theta(a_down))  # [B, N_m]
        p_hat = lam.unsqueeze(0) * p_prev + q_noise  # [B, N_m]
        surprise_inflation = torch.zeros_like(p_hat)
        if e_prev is not None and self.gamma_e > 0:
            # Clamp the surprise before inflating variance: e is an MSE in an
            # unbounded space, and without a cap a single large e drives
            # P_hat -> inf, K -> 1, and the memory into a write-overwrite
            # feedback loop (observed as mem_l2/nll blow-up during training).
            e_inflate = e_prev.detach().clamp(max=self.surprise_clip)
            surprise_inflation = self.gamma_e * e_inflate.unsqueeze(-1)
            p_hat = p_hat + surprise_inflation  # broadcast [B,1]→[B,N_m]

        return m_hat, p_hat, q_noise, surprise_inflation

    def correct(self, m_hat: torch.Tensor, p_hat: torch.Tensor,
                z_down: torch.Tensor, gain_mode: str = "learned",
                fixed_gain: float = 0.5,
                q_noise: Optional[torch.Tensor] = None,
                surprise_inflation: Optional[torch.Tensor] = None,
                p_prev: Optional[torch.Tensor] = None,
                return_diagnostics: bool = False) -> Tuple[torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor,
                                                               torch.Tensor, Optional[dict]]:
        """Learned Kalman filter update (U2) + observation model (U3).

        Args:
            m_hat:  [B, N_m, d_mem] — predicted memory
            p_hat:  [B, N_m] — predicted variance
            z_down: [B, k, d_mem] — multi-token observation (from U1)

        Returns:
            m_new:     [B, N_m, d_mem] — corrected memory
            p_new:     [B, N_m] — corrected variance
            e:         [B] — surprise (detached, for next step's predict)
            loss_obs:  scalar — observation prediction MSE (for L_obs)
            loss_nll:  scalar — Gaussian NLL (for L_nll)
        """
        B, N_m, d_mem = m_hat.shape
        eps = 1e-8

        # ── Innovation: cross-attention with k KV tokens (U1) ──
        delta_raw, _ = self.mem_cross_attn(m_hat, z_down, z_down)  # [B, N_m, d_mem]
        delta_m = self.innovation_proj(delta_raw)

        # ── U2: Learned observation noise R ──
        z_mean = z_down.mean(dim=1)  # [B, d_mem]
        # r_min floor: keeps log R finite and bounds e/R so the NLL cannot
        # explode when the learned noise collapses toward zero.
        R = F.softplus(self.R_psi(z_mean)) + self.r_min  # [B, N_m]

        # ── U2: Kalman gain K = P_hat / (P_hat + R) ──
        learned_K = p_hat / (p_hat + R + eps)  # [B, N_m]
        if gain_mode == "learned":
            K = learned_K
        elif gain_mode == "fixed":
            K = torch.full_like(learned_K, float(fixed_gain)).clamp(0.0, 1.0)
        elif gain_mode == "zero":
            K = torch.zeros_like(learned_K)
        elif gain_mode == "one":
            K = torch.ones_like(learned_K)
        else:
            raise ValueError(
                f"Unknown gain_mode={gain_mode!r}; expected learned, fixed, zero, or one"
            )
        K_exp = K.unsqueeze(-1)  # [B, N_m, 1]

        # ── Kalman update ──
        m_new = m_hat + K_exp * delta_m
        p_new = (1 - K_exp.squeeze(-1)) * p_hat  # [B, N_m]

        # ── U3: Observation model — predict what we'll see ──
        z_pred = self.obs_model(m_hat.mean(dim=1))  # [B, d_mem]
        z_target = z_mean.detach()  # [B, d_mem]
        e_per_sample = F.mse_loss(z_pred, z_target, reduction='none').mean(dim=-1)  # [B]

        # L_obs: trains the observation model
        loss_obs = F.mse_loss(z_pred, z_target)

        # L_nll: Gaussian NLL calibrates R against actual surprise.
        # NOTE: e_per_sample is ALREADY the squared residual (per-sample MSE),
        # so the NLL term is e / R — do NOT square it again (squaring makes
        # the loss quartic in the residual and its gradient ~ -e^2/(2R^2),
        # which explodes and starves the action CE under global grad clipping).
        # Computed in float32 for bf16 stability.
        e_detached = e_per_sample.detach().unsqueeze(-1).float()  # [B, 1]
        R_f = R.float()
        loss_nll = 0.5 * (torch.log(R_f) + e_detached / R_f).mean()
        loss_nll = loss_nll.to(R.dtype)

        # Surprise for next step (detached)
        e = e_per_sample.detach()  # [B]

        diagnostics = None
        if return_diagnostics:
            def _slot_values(value: torch.Tensor) -> list:
                return value.detach().float().cpu().tolist()

            diagnostics = {
                "layer": self.layer_idx,
                "p_prev": _slot_values(p_prev if p_prev is not None else p_hat),
                "q_noise": _slot_values(
                    q_noise if q_noise is not None else torch.zeros_like(p_hat)
                ),
                "surprise_inflation": _slot_values(
                    surprise_inflation
                    if surprise_inflation is not None
                    else torch.zeros_like(p_hat)
                ),
                "p_hat": _slot_values(p_hat),
                "r": _slot_values(R),
                "k_learned": _slot_values(learned_K),
                "k": _slot_values(K),
                "p_new": _slot_values(p_new),
                "innovation_norm": _slot_values(delta_m.norm(dim=-1)),
                "write_norm": _slot_values((K_exp * delta_m).norm(dim=-1)),
                "memory_pred_norm": _slot_values(m_hat.norm(dim=-1)),
                "memory_new_norm": _slot_values(m_new.norm(dim=-1)),
                "surprise": _slot_values(e),
            }

        if return_diagnostics:
            return m_new, p_new, e, loss_obs, loss_nll, diagnostics
        return m_new, p_new, e, loss_obs, loss_nll

    def inject(self, h: torch.Tensor, m_new: torch.Tensor,
               return_diagnostics: bool = False):
        """F4: Zero-output-projection gated injection."""
        h_down = self.h_down(h)
        delta_down, _ = self.inject_cross_attn(h_down, m_new, m_new)
        delta_up = self.delta_up(delta_down)  # zero-init → 0 at start
        w = self.inject_w_max * torch.tanh(self.inject_gate)
        injected_delta = w * delta_up
        injected_h = h + injected_delta
        if return_diagnostics:
            hidden_norm = h.detach().float().norm(dim=-1).mean()
            injection_norm = injected_delta.detach().float().norm(dim=-1).mean()
            return injected_h, {
                "injection_norm": float(injection_norm.cpu().item()),
                "hidden_norm": float(hidden_norm.cpu().item()),
                "injection_ratio": float(
                    (injection_norm / hidden_norm.clamp_min(1e-8)).cpu().item()
                ),
                "gate": float(torch.tanh(self.inject_gate).detach().float().cpu().item()),
                "layer_weight": float(self.inject_w_max),
            }
        return injected_h

    def extract_observation(self, h: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """U1+F3: Masked multi-token observation pooling.

        Projects h down to d_mem FIRST, then k learned latent queries attend
        over non-padding positions in the compressed d_mem space, producing
        [B, k, d_mem]. This keeps the attention O(d_mem²) instead of O(d²).

        Args:
            h:                [B, N, d] — hidden states from the pretrained layer
            attention_mask:   [B, N] — 1 for real tokens, 0 for padding

        Returns:
            z_down: [B, k, d_mem] — multi-token observation
        """
        B = h.shape[0]
        # F3: Build key_padding_mask (True = padding to be masked out)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # [B, N]

        # Project h to d_mem BEFORE attention (compresses the key/value space)
        h_down = self.obs_down(h)  # [B, N, d_mem]

        # U1: k learned queries attend over compressed hidden states
        queries = self.obs_queries.unsqueeze(0).expand(B, -1, -1)  # [B, k, d_mem]
        z_down, _ = self.obs_attn(
            queries, h_down, h_down, key_padding_mask=key_padding_mask,
        )  # [B, k, d_mem]
        return z_down


# ═══════════════════════════════════════════════════════════════════════════════
# JAMEL-COMPACT Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class JAMELCompactWrapper(nn.Module):
    """
    Wraps a pretrained HuggingFace LLM (e.g. Qwen3-VL-2B/8B) with per-layer
    side memory, WITHOUT modifying the pretrained model's internal structure.

    The pretrained layer's self_attn and FFN are called in-place — their
    weights are loaded from the checkpoint.  Only SideMemoryModule parameters
    are new (randomly initialized).

    Supports save/load of the full model (base + side memory) and standalone
    side-memory-only checkpoints.
    """

    def __init__(
        self,
        config: CompactConfig,
        lora_adapter_path: str | Path | None = None,
        lora_is_trainable: bool = True,
    ):
        super().__init__()
        self.config = config

        if lora_enabled(config.lora_rank) and not config.freeze_base:
            raise ValueError(
                "COMPACT LoRA requires freeze_base=True because dense base weights "
                "remain frozen while LoRA adapters are trained"
            )

        # ── Load pretrained LLM ──
        # Qwen3-VL is a multimodal model — AutoModelForCausalLM won't work.
        # We try causal first (for text-only models like Qwen3-8B), then fall
        # back to ImageTextToText (for vision-language models like Qwen3-VL).
        dtype = torch.bfloat16 if config.bf16 else torch.float32
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if config.model_parallel:
            try:
                import accelerate
            except ImportError as exc:
                raise RuntimeError(
                    "Model-parallel loading requires `accelerate`."
                ) from exc
            load_kwargs.update({
                "device_map": "balanced_low_0",
                "low_cpu_mem_usage": True,
            })
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                **load_kwargs,
            )
        except (ValueError, OSError) as e:
            print(f"[model] AutoModelForCausalLM failed ({e}), "
                  f"trying AutoModelForImageTextToText...")
            from transformers import AutoModelForImageTextToText
            self.llm = AutoModelForImageTextToText.from_pretrained(
                config.base_model_name,
                **load_kwargs,
            )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name, trust_remote_code=True,
            )
        except Exception:
            self.tokenizer = None
        try:
            self.processor = AutoProcessor.from_pretrained(
                config.base_model_name, trust_remote_code=True,
            )
        except Exception:
            self.processor = None

        # ── Infer architecture ──
        self.hidden_dim = self._infer_hidden_size(self.llm)
        self.num_layers = self._infer_num_layers(self.llm)
        self.num_mem = config.num_mem_tokens
        self.mem_dim = config.mem_dim
        self.num_act_tokens = config.num_act_tokens

        # ── Create side memory modules (NEW parameters) ──
        self.side_memories = nn.ModuleList([
            SideMemoryModule(
                l, self.num_layers, self.hidden_dim,
                mem_dim=config.mem_dim,
                num_mem=config.num_mem_tokens,
                num_heads=config.num_heads,
                num_obs_tokens=config.num_obs_tokens,
                config=config,
            )
            for l in range(self.num_layers)
        ])

        # ── Action embedding (NEW) ──
        self.action_embed = nn.Linear(self.hidden_dim, self.hidden_dim)

        # ── Cast new modules to the same dtype as the pretrained LLM ──
        # (All new parameters are created in float32 by default)
        llm_dtype = next(self.llm.parameters()).dtype
        self.action_embed = self.action_embed.to(dtype=llm_dtype)
        for sm in self.side_memories:
            sm.to(dtype=llm_dtype)

        self.model_parallel = bool(config.model_parallel)
        self.input_device = next(self.llm.parameters()).device
        if self.model_parallel:
            self._configure_model_parallel(llm_dtype)

        # ── Optionally freeze the base LLM ──
        if config.freeze_base:
            for param in self.llm.parameters():
                param.requires_grad = False

        self.llm = configure_lora(
            self.llm,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias=config.lora_bias,
            adapter_path=lora_adapter_path,
            is_trainable=lora_is_trainable,
        )

        if config.gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
            self.llm.config.use_cache = False
            if lora_enabled(config.lora_rank) and lora_is_trainable:
                enable_lora_gradient_checkpointing(self.llm)

    # ── Architecture helpers ──

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        for candidate in module.modules():
            for attribute_name in ("weight", "bias"):
                tensor = getattr(candidate, attribute_name, None)
                if (
                    isinstance(tensor, torch.Tensor)
                    and tensor.device.type != "meta"
                ):
                    return tensor.device
            for buffer in candidate.buffers(recurse=False):
                if buffer.device.type != "meta":
                    return buffer.device
        for parameter in module.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        return torch.device("cpu")

    @staticmethod
    def _to_device(value, device: torch.device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(JAMELCompactWrapper._to_device(v, device) for v in value)
        if isinstance(value, list):
            return [JAMELCompactWrapper._to_device(v, device) for v in value]
        if isinstance(value, dict):
            return {
                key: JAMELCompactWrapper._to_device(item, device)
                for key, item in value.items()
            }
        return value

    def _configure_model_parallel(self, llm_dtype: torch.dtype) -> None:
        device_map = getattr(self.llm, "hf_device_map", None)
        if not device_map:
            raise RuntimeError("Transformers did not create a model device map.")

        decoder_layers = self._get_decoder_layers()
        layer_devices = []
        for layer, side_memory in zip(decoder_layers, self.side_memories):
            layer_device = self._module_device(layer)
            if layer_device.type != "cuda":
                raise RuntimeError(
                    "The 8B model did not fit entirely on the selected GPUs; "
                    f"a decoder layer was offloaded to {layer_device}. Select "
                    "more GPUs or reduce max_length."
                )
            side_memory.to(device=layer_device, dtype=llm_dtype)
            layer_devices.append(str(layer_device))

        embedding_device = self._module_device(self._get_input_embeddings())
        self.action_embed.to(device=embedding_device, dtype=llm_dtype)
        self.input_device = embedding_device

        distribution = {}
        for layer_device in layer_devices:
            distribution[layer_device] = distribution.get(layer_device, 0) + 1
        print(f"[model] Model-parallel decoder distribution: {distribution}")
        print(f"[model] Input/vision device: {self.input_device}")

    def _run_decoder_layer(self, layer: nn.Module, h: torch.Tensor, **kwargs):
        if self.config.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            def layer_forward(hidden_states):
                return layer(hidden_states, **kwargs)

            return checkpoint(layer_forward, h, use_reentrant=False)
        return layer(h, **kwargs)

    @staticmethod
    def _infer_hidden_size(model) -> int:
        config = model.config
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return int(config.text_config.hidden_size)
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        raise ValueError("Cannot infer hidden_size")

    @staticmethod
    def _infer_num_layers(model) -> int:
        config = model.config
        if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
            return int(config.text_config.num_hidden_layers)
        if hasattr(config, "num_hidden_layers"):
            return int(config.num_hidden_layers)
        raise ValueError("Cannot infer num_layers")

    def _get_decoder_layers(self):
        # Try multiple paths to find decoder layers across model architectures.
        # Cast to list to check if already a ModuleList/Sequential.
        model = self.llm

        candidates = [
            # Qwen3-VL via AutoModelForImageTextToText:
            #   model (Qwen3VLModel) → language_model → layers
            lambda m: getattr(getattr(m, "model", None), "language_model", None),
            # Qwen3 text-only: model → layers
            lambda m: getattr(m, "model", None),
            # Llama/Mistral: transformer → h
            lambda m: getattr(m, "transformer", None),
            # Direct language_model attribute
            lambda m: getattr(m, "language_model", None),
        ]

        for get_wrapper in candidates:
            wrapper = get_wrapper(model)
            if wrapper is None:
                continue
            # wrapper may already be the layer list, or may have .layers / .h
            if hasattr(wrapper, "layers"):
                return wrapper.layers
            if hasattr(wrapper, "h"):
                return wrapper.h
            # wrapper might itself be a layer list
            if isinstance(wrapper, (list, nn.ModuleList)):
                return wrapper

        # Last resort: print model attributes for debugging
        attrs = sorted([a for a in dir(model) if not a.startswith("_") and not callable(getattr(model, a))])
        raise ValueError(
            f"Cannot find decoder layers in {type(model).__name__}. "
            f"Top-level attributes: {attrs[:20]}"
        )

    def _get_lm_head(self):
        """Get the LM head, handling multimodal model wrappers."""
        if hasattr(self.llm, "lm_head"):
            return self.llm.lm_head
        if hasattr(self.llm, "language_model") and hasattr(self.llm.language_model, "lm_head"):
            return self.llm.language_model.lm_head
        raise ValueError("Cannot find lm_head in this model")

    def _get_final_norm(self):
        """Get the pretrained decoder's final normalization layer."""
        model = self.llm
        candidates = [
            lambda m: getattr(
                getattr(getattr(m, "model", None), "language_model", None),
                "norm", None,
            ),
            lambda m: getattr(getattr(m, "language_model", None), "norm", None),
            lambda m: getattr(getattr(m, "model", None), "norm", None),
            lambda m: getattr(getattr(m, "transformer", None), "ln_f", None),
            lambda m: getattr(m, "norm", None),
        ]
        for get_norm in candidates:
            norm = get_norm(model)
            if norm is not None:
                return norm
        return None

    def _apply_final_norm(self, h: torch.Tensor) -> torch.Tensor:
        norm = self._get_final_norm()
        if norm is None:
            return h
        return norm(h.to(self._module_device(norm)))

    def _compute_position_embeddings(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
    ):
        """
        Compute rotary position embeddings for Qwen3-VL layers.

        Qwen3-VL's decoder layer requires `position_embeddings` as a kwarg.
        For image prompts, position IDs come from the base model's 3D M-RoPE
        helper so manual layer execution matches the pretrained forward path.
        """
        # Find the rotary embedding module
        rotary_emb = self._find_rotary_emb()
        if rotary_emb is None:
            return None  # model doesn't use RoPE

        # Qwen3-VL uses 3D M-RoPE for multimodal sequences. Ask the base
        # multimodal backbone for the same position IDs it uses in its own
        # forward path; this is essential for wrapper/base equivalence.
        if (image_grid_thw is not None and input_ids is not None
                and mm_token_type_ids is None):
            # mm_token_type_ids are only required by backbones exposing
            # compute_3d_position_ids; older ones fall back to get_rope_index.
            _has_3d_helper = any(
                hasattr(b, "compute_3d_position_ids")
                for b in (getattr(self.llm, "model", None), self.llm)
                if b is not None
            )
            if _has_3d_helper:
                raise ValueError(
                    "image_grid_thw requires processor-provided "
                    "mm_token_type_ids for correct Qwen3-VL M-RoPE positions"
                )
        position_ids = None
        backbone_candidates = [
            getattr(self.llm, "model", None),
            self.llm,
        ]
        for backbone in backbone_candidates:
            if backbone is None or not hasattr(backbone, "compute_3d_position_ids"):
                continue
            try:
                position_ids = backbone.compute_3d_position_ids(
                    input_ids=input_ids,
                    inputs_embeds=h,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                )
            except TypeError:
                # Older Transformers/Qwen versions do not accept
                # mm_token_type_ids or past_key_values in this helper.
                try:
                    position_ids = backbone.compute_3d_position_ids(
                        input_ids=input_ids,
                        inputs_embeds=h,
                        image_grid_thw=image_grid_thw,
                        attention_mask=attention_mask,
                    )
                except (TypeError, ValueError):
                    position_ids = None
            except ValueError:
                position_ids = None
            if position_ids is not None:
                break

        # Older Qwen2.5-VL-style backbones expose get_rope_index directly.
        if position_ids is None and input_ids is not None:
            for backbone in backbone_candidates:
                if backbone is None or not hasattr(backbone, "get_rope_index"):
                    continue
                try:
                    position_ids, _ = backbone.get_rope_index(
                        input_ids=input_ids,
                        image_grid_thw=image_grid_thw,
                        attention_mask=attention_mask,
                        mm_token_type_ids=mm_token_type_ids,
                    )
                except (TypeError, ValueError):
                    try:
                        position_ids, _ = backbone.get_rope_index(
                            input_ids=input_ids,
                            image_grid_thw=image_grid_thw,
                            attention_mask=attention_mask,
                        )
                    except (TypeError, ValueError):
                        position_ids = None
                if position_ids is not None:
                    break

        # Text-only/fallback path: a 2D position sequence is expanded by the
        # Qwen3-VL rotary embedding implementation when needed.
        if position_ids is None and attention_mask is not None and attention_mask.dim() == 2:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        elif position_ids is None:
            position_ids = torch.arange(
                h.shape[1], device=h.device
            ).unsqueeze(0).expand(h.shape[0], -1)

        # Compute rotary embeddings for all positions
        cos, sin = rotary_emb(h, position_ids)
        return (cos.to(dtype=h.dtype), sin.to(dtype=h.dtype))

    def _find_rotary_emb(self):
        """Find the rotary embedding module in the model.

        We do NOT cache the result because DataParallel creates fresh replicas
        each forward pass, and a cached reference would point to the original
        device's module, causing device mismatches."""
        model = self.llm
        for path in [
            lambda m: getattr(m, "rotary_emb", None),
            lambda m: getattr(getattr(m, "model", None), "rotary_emb", None),
            lambda m: getattr(getattr(getattr(m, "model", None), "language_model", None), "rotary_emb", None),
        ]:
            result = path(model)
            if result is not None:
                return result
        return None

    def _get_input_embeddings(self):
        """Get token embeddings, handling multimodal model wrappers."""
        try:
            return self.llm.get_input_embeddings()
        except (AttributeError, NotImplementedError):
            pass
        if hasattr(self.llm, "language_model"):
            return self.llm.language_model.get_input_embeddings()
        raise ValueError("Cannot find input embeddings in this model")

    def _has_visual_encoder(self) -> bool:
        """Check if the model has a visual encoder (for VLMs)."""
        return hasattr(self.llm, "visual") or (
            hasattr(self.llm, "model") and hasattr(self.llm.model, "visual")
        )

    @staticmethod
    def _build_causal_attention_mask(
        attention_mask: torch.Tensor, dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Convert a 2D padding mask [B, N] into a 4D causal attention mask
        [B, 1, N, N] suitable for SDPA.

        - Positions where attention_mask == 0 (padding) are masked to -inf.
        - Upper triangle is masked (causal).
        """
        B, N = attention_mask.shape
        # Start with causal mask: lower triangle = 0, upper = -inf
        causal = torch.triu(
            torch.full((N, N), float('-inf'), dtype=dtype, device=attention_mask.device),
            diagonal=1,
        )  # [N, N]
        causal = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, N, N)  # [B, 1, N, N]

        # Apply padding mask: set columns for padding positions to -inf
        if (attention_mask == 0).any():
            pad_mask = attention_mask[:, None, None, :] == 0  # [B, 1, 1, N]
            causal = causal.masked_fill(pad_mask, float('-inf'))

        return causal

    def _inject_visual_features(
        self,
        h: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
    ) -> tuple:
        """
        Process image through the visual encoder and inject features into
        the hidden states at image placeholder token positions.

        For Qwen3-VL, the model has a `visual` module that processes pixel
        values into image features. The visual encoder returns:
          - pooler_output: merged/projected features (dim = LLM hidden_dim)
          - deepstack_features: list of per-layer features for DeepStack injection

        Returns:
            (h, deepstack_features, visual_pos_mask) — updated hidden states,
            deepstack features list, and boolean mask of image token positions.
        """
        model = self.llm

        # Flatten pixel_values and image_grid_thw for the visual encoder.
        # The Qwen3-VL visual encoder expects:
        #   pixel_values:  [total_patches, patch_dim]  (all images concatenated)
        #   image_grid_thw: [num_images, 3]
        # But collate_fn stacks per-sample tensors, adding a batch dim.
        if pixel_values.dim() == 3:
            # [B, num_patches, dim] → [B * num_patches, dim]
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        if image_grid_thw is not None and image_grid_thw.dim() == 3:
            # [B, num_images, 3] → [B * num_images, 3]
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        # Use the model's get_image_features if available (handles merger + split)
        if hasattr(model, "get_image_features"):
            try:
                vision_output = model.get_image_features(
                    pixel_values, image_grid_thw,
                )
                if hasattr(vision_output, "pooler_output"):
                    # newer object-style return
                    image_embeds_list = vision_output.pooler_output
                    deepstack_features = vision_output.deepstack_features or []
                else:
                    # transformers>=4.56 tuple API: (per-image embeds, deepstack)
                    image_embeds_list, deepstack_features = vision_output
                    deepstack_features = deepstack_features or []
                image_embeds = torch.cat(image_embeds_list, dim=0).to(
                    h.device, h.dtype,
                )  # [total_image_tokens, d]
            except Exception as e:
                import traceback
                print(f"[model] get_image_features failed: {e}")
                print(f"  pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}, device: {pixel_values.device}")
                if image_grid_thw is not None:
                    print(f"  image_grid_thw shape: {image_grid_thw.shape}, dtype: {image_grid_thw.dtype}, device: {image_grid_thw.device}")
                print(f"  model type: {type(model).__name__}")
                print(f"  visual dtype: {model.visual.dtype if hasattr(model, 'visual') else 'N/A'}")
                traceback.print_exc()
                return h, [], None
        else:
            # Fallback: call visual encoder directly
            visual = getattr(model, "visual", None)
            if visual is None:
                visual = getattr(getattr(model, "model", None), "visual", None)
            if visual is None:
                return h, [], None

            try:
                vision_output = visual(
                    pixel_values.to(h.dtype),
                    grid_thw=image_grid_thw,
                    return_dict=True,
                )
                # Use pooler_output (merged/projected to LLM dim), not last_hidden_state
                if hasattr(vision_output, "pooler_output"):
                    image_embeds = vision_output.pooler_output
                elif hasattr(vision_output, "last_hidden_state"):
                    image_embeds = vision_output.last_hidden_state
                else:
                    return h, [], None
                image_embeds = image_embeds.to(h.device, h.dtype)
                deepstack_features = getattr(vision_output, "deepstack_features", []) or []
            except Exception as e:
                print(f"[model] Visual encoder failed: {e}")
                return h, [], None

        # Find image placeholder token ID
        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.tokenizer.convert_tokens_to_ids("◣")
            if image_token_id is None or image_token_id == self.tokenizer.unk_token_id:
                return h, [], None

        # Build visual position mask [B, N] and scatter features
        visual_pos_mask = (input_ids == image_token_id)  # [B, N]
        total_img_tokens = visual_pos_mask.sum().item()

        if total_img_tokens == 0:
            return h, [], None

        # Use masked_scatter like Qwen3-VL does
        if total_img_tokens == image_embeds.shape[0]:
            mask_expanded = visual_pos_mask.unsqueeze(-1).expand_as(h)
            h = h.masked_scatter(mask_expanded, image_embeds)
        else:
            # Fallback: per-batch scatter with offset
            offset = 0
            for b in range(h.shape[0]):
                positions = visual_pos_mask[b].nonzero(as_tuple=True)[0]
                n = len(positions)
                if n > 0 and offset + n <= image_embeds.shape[0]:
                    h[b, positions] = image_embeds[offset:offset + n]
                    offset += n

        return h, deepstack_features, visual_pos_mask

    # ── Memory initialization ──

    def init_memory(self, batch_size: int, device: torch.device):
        """Initialize memory states and variance for t=0.

        v2: Returns (memory_states, variance_states) — variance P, not confidence C.
        Variance P_0 = 0.5 per slot.
        Also initializes e_prev (surprise) to None for each layer.
        """
        # Cast to model dtype to avoid bfloat16/float32 mismatches
        llm_dtype = next(self.llm.parameters()).dtype
        m_states, p_states = [], []
        for sm in self.side_memories:
            state_device = self._module_device(sm) if self.model_parallel else device
            m = sm.init_memory.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device=state_device, dtype=llm_dtype)
            p = torch.full((batch_size, self.num_mem), sm.init_variance,
                           device=state_device, dtype=llm_dtype)
            m_states.append(m)
            p_states.append(p)
        return m_states, p_states

    # ── Forward ──

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_embed_input: Optional[torch.Tensor] = None,
        memory_states: Optional[List[torch.Tensor]] = None,
        variance_states: Optional[List[torch.Tensor]] = None,
        action_input_ids: Optional[torch.Tensor] = None,
        action_attention_mask: Optional[torch.Tensor] = None,
        observation_mask: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        deepstack_features: Optional[List] = None,
        visual_pos_mask: Optional[torch.Tensor] = None,
        e_prev_list: Optional[List[Optional[torch.Tensor]]] = None,
        sample_weights: Optional[torch.Tensor] = None,
        embed_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        skip_lm_head: bool = False,
        **kwargs,
    ) -> dict:
        """
        One time step through the full memory-augmented LLM.

        [COMPACT-UAV extension] embed_override: optional (mask, value) tuple;
        positions where mask [B, N] is True have their input embedding replaced
        by value [d] (used for the waypoint sentinel slot). The result dict
        always carries "hidden_states" (post-final-norm), "loss_obs" and
        "loss_nll" so non-text heads can consume them.

        v2 changes:
        - attention_mask is threaded into extract_observation (F3: masked pooling)
        - e_prev_list carries surprise from previous step (U2/U3)
        - collect loss_obs and loss_nll per layer (U2/U3)
        - sample_weights for coverage-weighted SFT (F7)

        Args:
            input_ids:          [B, N] — token IDs (includes image placeholder tokens)
            attention_mask:     [B, N]
            action_embed_input: [B, d] — pre-computed action embedding
            action_input_ids:   [B, L_act] — action tokens when the embedding
                                must be computed inside the wrapped forward
            memory_states:      List of [B, N_m, d_mem]
            variance_states:   List of [B, N_m] — variance P per layer
            observation_mask:   [B, N] — prompt/current-observation tokens only
            mm_token_type_ids:  [B, N] — Qwen3-VL modality IDs for M-RoPE
            labels:             [B, N] — token labels for loss (optional)
            pixel_values:       [B, C, H, W] — processed image tensor (optional)
            image_grid_thw:     [B, 3] — image grid dimensions (optional)
            inputs_embeds:      [B, N, d] — pre-computed embeddings (optional,
                                bypasses visual encoder — used for DataParallel)
            deepstack_features: Pre-computed deepstack features (for DataParallel)
            visual_pos_mask:    Pre-computed visual position mask (for DataParallel)
            e_prev_list:        List of [B] or None — previous step surprise per layer
            sample_weights:     [B] — per-sample loss weights (F7)

        Returns:
            dict with: logits, new_memory, new_variance, loss (if labels),
                       loss_dict, e_list (surprise for next step)
        """
        B = input_ids.shape[0]
        device = input_ids.device
        if memory_states is None or variance_states is None:
            memory_states, variance_states = self.init_memory(B, device)
        if observation_mask is None:
            observation_mask = attention_mask
        else:
            # The dedicated mask may only narrow the model's valid-token mask;
            # never allow padding to enter observation pooling.
            observation_mask = (
                observation_mask.to(dtype=torch.bool)
                & attention_mask.to(dtype=torch.bool)
            )

        # ── Embed tokens ──
        if inputs_embeds is not None:
            # Pre-computed embeddings (visual features already injected)
            h = inputs_embeds
            if deepstack_features is None:
                deepstack_features = []
            if visual_pos_mask is None:
                visual_pos_mask = None
        else:
            embed_layer = self._get_input_embeddings()
            h = embed_layer(input_ids)  # [B, N, d]

            # [COMPACT-UAV] sentinel-slot embedding override (e.g. waypoint token)
            if embed_override is not None:
                override_mask, override_value = embed_override
                h = h.clone()
                h[override_mask.to(h.device)] = override_value.to(
                    device=h.device, dtype=h.dtype
                )

            # ── Process image features if provided ──
            if deepstack_features is None:
                deepstack_features = []
            if pixel_values is not None and self._has_visual_encoder():
                h, deepstack_features, visual_pos_mask = self._inject_visual_features(
                    h, input_ids, pixel_values, image_grid_thw,
                )

        # ── Raw action embedding ──
        if action_embed_input is None:
            if action_input_ids is None or action_attention_mask is None:
                raise ValueError(
                    "Provide action_embed_input or action token IDs and mask."
                )
            action_token_embed = self._get_input_embeddings()
            action_token_device = self._module_device(action_token_embed)
            action_token_hidden = action_token_embed(
                action_input_ids.to(action_token_device)
            )
            action_token_mask = action_attention_mask.to(
                action_token_device
            ).unsqueeze(-1).to(action_token_hidden.dtype)
            action_embed_input = (
                (action_token_hidden * action_token_mask).sum(dim=1)
                / action_token_mask.sum(dim=1).clamp(min=1)
            )

        action_device = self._module_device(self.action_embed)
        action_embed = self.action_embed(
            action_embed_input.to(action_device)
        )  # [B, d]

        # ── Get decoder layers ──
        decoder_layers = self._get_decoder_layers()

        # ── Compute position embeddings if the model uses RoPE ──
        position_embeddings = self._compute_position_embeddings(
            h, attention_mask, input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        # ── Convert attention_mask to 4D causal mask ──
        attention_mask_4d = self._build_causal_attention_mask(
            attention_mask, h.dtype,
        )

        # ── Initialize e_prev_list if not provided (chunk start) ──
        if e_prev_list is None:
            e_prev_list = [None] * len(self.side_memories)

        # ── Process through each layer ──
        new_memory, new_variance = [], []
        e_list = []  # surprise per layer for next step
        loss_obs_terms = []
        loss_nll_terms = []
        L = len(decoder_layers)

        for l, (layer, sm) in enumerate(zip(decoder_layers, self.side_memories)):
            layer_device = self._module_device(layer)
            h = h.to(layer_device)
            memory_state = memory_states[l].to(layer_device)
            variance_state = variance_states[l].to(layer_device)
            action_embed_layer = action_embed.to(layer_device)
            e_prev = e_prev_list[l]
            if isinstance(e_prev, torch.Tensor):
                e_prev = e_prev.to(layer_device)

            # 4a. Memory Predict (FiLM-GRU + variance predict)
            m_hat, p_hat, q_noise, surprise_inflation = sm.predict(
                memory_state, variance_state, action_embed_layer,
                e_prev=e_prev,
            )

            # 4b. Run pretrained layer (self-attn + FFN)
            layer_output = self._run_decoder_layer(
                layer,
                h,
                attention_mask=attention_mask_4d.to(layer_device),
                position_embeddings=self._to_device(
                    position_embeddings, layer_device,
                ),
                **self._to_device(kwargs, layer_device),
            )
            if isinstance(layer_output, tuple):
                h_layer = layer_output[0]
            else:
                h_layer = layer_output

            # 4b.5 DeepStack injection (Qwen3-VL adds visual features to
            #      early decoder layers' hidden states at image positions)
            if deepstack_features and l < len(deepstack_features):
                ds_feat = deepstack_features[l].to(h_layer.device, h_layer.dtype)
                if visual_pos_mask is not None:
                    mask_1d = visual_pos_mask.to(h_layer.device)  # [B, N]
                    ds_offset = 0
                    for b in range(h_layer.shape[0]):
                        positions = mask_1d[b].nonzero(as_tuple=True)[0]
                        n = len(positions)
                        if n > 0 and ds_offset + n <= ds_feat.shape[0]:
                            h_layer[b, positions] = (
                                h_layer[b, positions]
                                + ds_feat[ds_offset:ds_offset + n]
                            )
                            ds_offset += n

            # 4c. Extract observation (F3: masked, U1: multi-token)
            z_down = sm.extract_observation(
                h_layer, observation_mask.to(layer_device),
            )

            # 4d. Memory Correct (learned Kalman + obs model)
            m_new, p_new, e, loss_obs_l, loss_nll_l = sm.correct(
                m_hat, p_hat, z_down, q_noise=q_noise,
                surprise_inflation=surprise_inflation, p_prev=variance_state,
            )

            # 4e. Memory Inject (F4: zero-init gated)
            h = sm.inject(h_layer, m_new)

            new_memory.append(m_new)
            new_variance.append(p_new)
            e_list.append(e)
            loss_obs_terms.append(loss_obs_l)
            loss_nll_terms.append(loss_nll_l)

        # ── LM head (pretrained) ──
        # [COMPACT-UAV] skip_lm_head avoids materializing [B, N, vocab] logits
        # when only hidden states are consumed (waypoint regression).
        normalized_h = self._apply_final_norm(h)
        if skip_lm_head and labels is None:
            logits = None
            loss_device = normalized_h.device
        else:
            lm_head = self._get_lm_head()
            logits = lm_head(normalized_h.to(self._module_device(lm_head)))
            loss_device = logits.device
        loss_obs_total = torch.stack([
            term.to(loss_device) for term in loss_obs_terms
        ]).sum() / L
        loss_nll_total = torch.stack([
            term.to(loss_device) for term in loss_nll_terms
        ]).sum() / L

        result = {
            "logits": logits,
            "hidden_states": normalized_h,  # [COMPACT-UAV] post-norm hidden states
            "loss_obs": loss_obs_total,     # [COMPACT-UAV] always exposed
            "loss_nll": loss_nll_total,     # [COMPACT-UAV] always exposed
            "new_memory": new_memory,
            "new_variance": new_variance,
            "e_list": e_list,  # surprise for next step
        }

        # ── Compute loss if labels provided ──
        if labels is not None:
            from .loss import compute_compact_loss
            loss, loss_dict = compute_compact_loss(
                logits=logits,
                labels=labels.to(loss_device),
                memory_states=new_memory,
                variance_states=new_variance,
                config=self.config,
                loss_obs=loss_obs_total,
                loss_nll=loss_nll_total,
                sample_weights=sample_weights,
            )
            result["loss"] = loss
            result["loss_dict"] = loss_dict

        return result

    # ── Generation (for eval) ──

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_embed_input: torch.Tensor,
        memory_states: List[torch.Tensor],
        variance_states: List[torch.Tensor],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        e_prev_list: Optional[List[Optional[torch.Tensor]]] = None,
        freeze_memory: bool = False,
        gain_mode: str = "learned",
        fixed_gain: float = 0.5,
        collect_diagnostics: bool = False,
        **kwargs,
    ) -> dict:
        """F5: Memory-conditioned generation.

        v1 discarded memory: it ran the compact forward to update memory,
        then called base llm.generate() on the RAW prompt (memory-injected
        hidden states were never used for generation).

        v2: After the memory-update prompt pass, the memory-injected final
        hidden state supplies the first-token logits. Subsequent tokens run
        through the pretrained decoder against the populated KV cache, so the
        continuation remains conditioned on memory-influenced prompt states.

        If memory_conditioned_generate is False, generation uses the base
        llm.generate() path on the raw prompt as an explicit ablation/fallback.
        The memory-conditioned path requires a Transformers version that
        provides DynamicCache-compatible decoder calls.
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # ── 1. Memory update: one forward pass on the prompt ──
        # We use a DynamicCache so the prompt's K/V states are captured
        # for incremental generation in step 2.
        from transformers import DynamicCache
        cache = DynamicCache()

        action_device = self._module_device(self.action_embed)
        action_embed = self.action_embed(action_embed_input.to(action_device))
        decoder_layers = self._get_decoder_layers()
        embed_layer = self._get_input_embeddings()

        # ── Process prompt embeddings ──
        embedding_device = self._module_device(embed_layer)
        input_ids = input_ids.to(embedding_device)
        attention_mask = attention_mask.to(embedding_device)
        action_embed_input = action_embed_input.to(embedding_device)
        h = embed_layer(input_ids)  # [B, N, d]

        # Inject visual features if available
        deepstack_features = []
        visual_pos_mask = None
        if pixel_values is not None and self._has_visual_encoder():
            h, deepstack_features, visual_pos_mask = self._inject_visual_features(
                h, input_ids, pixel_values, image_grid_thw,
            )

        # ── Build 4D causal mask for the prompt ──
        attn_mask_4d = self._build_causal_attention_mask(attention_mask, h.dtype)

        # ── Compute position embeddings for prompt ──
        position_embeddings = self._compute_position_embeddings(
            h,
            attention_mask,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            past_key_values=cache,
        )

        # ── Initialize e_prev_list if not provided ──
        if e_prev_list is None:
            e_prev_list = [None] * len(self.side_memories)

        # ── Process prompt through all layers with memory ──
        # Each layer receives the memory-injected h from the previous layer
        # and populates the DynamicCache with its K/V projections.
        new_memory, new_variance = [], []
        e_list = []
        uncertainty_diagnostics = []
        for l, (layer, sm) in enumerate(zip(decoder_layers, self.side_memories)):
            layer_device = self._module_device(layer)
            h = h.to(layer_device)
            memory_state = memory_states[l].to(layer_device)
            variance_state = variance_states[l].to(layer_device)
            action_embed_layer = action_embed.to(layer_device)
            e_prev = e_prev_list[l]
            if isinstance(e_prev, torch.Tensor):
                e_prev = e_prev.to(layer_device)
            m_hat, p_hat, q_noise, surprise_inflation = sm.predict(
                memory_state, variance_state, action_embed_layer,
                e_prev=e_prev,
            )
            layer_output = layer(
                h,
                attention_mask=attn_mask_4d.to(layer_device),
                position_embeddings=self._to_device(
                    position_embeddings, layer_device,
                ),
                past_key_values=cache,
                use_cache=True,
            )
            h_layer = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            # DeepStack injection
            if deepstack_features and l < len(deepstack_features):
                ds_feat = deepstack_features[l].to(h_layer.device, h_layer.dtype)
                if visual_pos_mask is not None:
                    visual_mask = visual_pos_mask.to(h_layer.device)
                    ds_offset = 0
                    for b in range(h_layer.shape[0]):
                        positions = visual_mask[b].nonzero(as_tuple=True)[0]
                        n = len(positions)
                        if n > 0 and ds_offset + n <= ds_feat.shape[0]:
                            h_layer[b, positions] = (
                                h_layer[b, positions]
                                + ds_feat[ds_offset:ds_offset + n]
                            )
                            ds_offset += n

            z_down = sm.extract_observation(
                h_layer, attention_mask.to(layer_device),
            )
            diagnostics = None
            if collect_diagnostics:
                m_new, p_new, e, _, _, diagnostics = sm.correct(
                    m_hat, p_hat, z_down, gain_mode=gain_mode,
                    fixed_gain=fixed_gain, q_noise=q_noise,
                    surprise_inflation=surprise_inflation, p_prev=variance_state,
                    return_diagnostics=True,
                )
                h, injection_diagnostics = sm.inject(
                    h_layer, m_new, return_diagnostics=True,
                )
                diagnostics.update(injection_diagnostics)
            else:
                m_new, p_new, e, _, _ = sm.correct(
                    m_hat, p_hat, z_down, gain_mode=gain_mode,
                    fixed_gain=fixed_gain, q_noise=q_noise,
                    surprise_inflation=surprise_inflation, p_prev=variance_state,
                )
                h = sm.inject(h_layer, m_new)
            new_memory.append(m_new)
            new_variance.append(p_new)
            e_list.append(e)
            if collect_diagnostics:
                uncertainty_diagnostics.append(diagnostics)

        # If freeze_memory (F5 ablation), don't write back new memory
        if freeze_memory:
            new_memory = memory_states
            new_variance = variance_states

        # ── 2. Token generation ──
        # F5: If memory_conditioned_generate is enabled, we use the
        # memory-injected hidden states h from the prompt pass to get the
        # first-token logits, then generate subsequent tokens with a
        # DynamicCache that accumulates KV states across steps.
        #
        # During the prompt pass above, each decoder layer processed the
        # full prompt h. We capture a DynamicCache from that pass so that
        # each subsequently generated token attends to ALL preceding tokens
        # (prompt + previously generated) — not just itself.
        if self.config.memory_conditioned_generate:
            generated_ids = self._generate_from_hidden_states(
                h, attention_mask, max_new_tokens, temperature, top_p,
                position_embeddings, attn_mask_4d, cache,
            )
        else:
            # Fallback: v1 approach — base llm.generate() on raw prompt
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
            )
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            if pixel_values is not None:
                gen_kwargs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                gen_kwargs["image_grid_thw"] = image_grid_thw
            if mm_token_type_ids is not None:
                gen_kwargs["mm_token_type_ids"] = mm_token_type_ids

            full_ids = self.llm.generate(**gen_kwargs)
            prompt_len = input_ids.shape[1]
            generated_ids = full_ids[:, prompt_len:]

        return {
            "generated_ids": generated_ids,
            "new_memory": new_memory,
            "new_variance": new_variance,
            "e_list": e_list,
            "uncertainty_diagnostics": uncertainty_diagnostics if collect_diagnostics else None,
        }

    @torch.inference_mode()
    def _generate_from_hidden_states(
        self,
        h: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        position_embeddings: Optional[Tuple],
        attn_mask_4d: torch.Tensor,
        cache=None,
    ) -> torch.Tensor:
        """F5: Generate tokens using the KV-cache from the prompt pass.

        The prompt pass in generate() already ran h through all decoder
        layers WITH memory injection AND populated a DynamicCache. We use
        the last position's logits from the memory-injected h to sample
        the first token, then generate subsequent tokens one at a time,
        each attending to the full accumulated context via the cache.

        Args:
            h:                  [B, prompt_len, d] — memory-injected final hidden states
            attention_mask:     [B, prompt_len] — prompt padding mask
            max_new_tokens:     max tokens to generate
            temperature:        sampling temperature (0 = greedy)
            top_p:              nucleus sampling threshold
            position_embeddings: (cos, sin) for the prompt positions
            attn_mask_4d:       [B, 1, prompt_len, prompt_len] — prompt causal mask
            cache:              DynamicCache populated during the prompt pass
        """
        B = h.shape[0]
        llm_dtype = h.dtype
        prompt_len = h.shape[1]
        generated = []
        decoder_layers = self._get_decoder_layers()
        embed_layer = self._get_input_embeddings()
        lm_head = self._get_lm_head()

        # ── Step 1: First token from memory-injected h ──
        normalized_h = self._apply_final_norm(h[:, -1:, :])
        logits = lm_head(normalized_h.to(self._module_device(lm_head)))
        next_token = self._sample_token(logits[:, -1, :], temperature, top_p)
        generated.append(next_token)

        # ── Step 2: Generate remaining tokens using the cache ──
        for _ in range(max_new_tokens - 1):
            # Embed the new token
            embedding_device = self._module_device(embed_layer)
            new_embed = embed_layer(
                next_token.to(embedding_device).unsqueeze(-1)
            )  # [B, 1, d]

            # Position embeddings for the new position. The base multimodal
            # backbone reuses its prompt rope deltas for incremental M-RoPE.
            new_pos_emb = self._compute_position_embeddings(
                new_embed,
                attention_mask=None,
                input_ids=None,
                past_key_values=cache,
            )

            # Run new token through all layers with cache (no memory update
            # during generation — memory was already updated in the prompt pass)
            h_new = new_embed
            for layer in decoder_layers:
                layer_device = self._module_device(layer)
                h_new = h_new.to(layer_device)
                layer_output = layer(
                    h_new,
                    attention_mask=None,  # single token, cache handles history
                    position_embeddings=self._to_device(
                        new_pos_emb, layer_device,
                    ),
                    past_key_values=cache,
                    use_cache=True,
                )
                h_new = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            # Get logits and sample
            normalized_h = self._apply_final_norm(h_new[:, -1:, :])
            logits = lm_head(normalized_h.to(self._module_device(lm_head)))
            next_token = self._sample_token(logits[:, -1, :], temperature, top_p)
            generated.append(next_token)

            # Check for EOS
            eos_id = getattr(self.llm.config, "eos_token_id", None)
            if eos_id is not None and (next_token == eos_id).all():
                break

        return torch.stack(generated, dim=1)  # [B, num_generated]

    @staticmethod
    def _sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Sample a token from logits with temperature and top-p filtering."""
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove,
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # ── Save / Load ──

    def save_pretrained(
        self,
        save_directory: str | Path,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
        """Save the full model: base LLM + side memory modules + config."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        llm_state = None
        side_memory_state = None
        action_embed_state = None
        if state_dict is not None:
            normalized_state = {}
            for key, value in state_dict.items():
                normalized_key = key.replace("_fsdp_wrapped_module.", "")
                normalized_state[normalized_key] = value
            llm_state = {
                key[len("llm."):]: value
                for key, value in normalized_state.items()
                if key.startswith("llm.")
            }
            side_memory_state = {
                key[len("side_memories."):]: value
                for key, value in normalized_state.items()
                if key.startswith("side_memories.")
            }
            action_embed_state = {
                key[len("action_embed."):]: value
                for key, value in normalized_state.items()
                if key.startswith("action_embed.")
            }

        # Save side memory modules
        side_mem_dir = save_path / "side_memory"
        side_mem_dir.mkdir(exist_ok=True)
        torch.save(
            side_memory_state or self.side_memories.state_dict(),
            side_mem_dir / "side_memories.pt",
        )
        torch.save(
            action_embed_state or self.action_embed.state_dict(),
            side_mem_dir / "action_embed.pt",
        )

        # Save config
        config_dict = self.config.to_dict()
        config_dict["hidden_dim"] = self.hidden_dim
        config_dict["num_layers"] = self.num_layers
        (save_path / "compact_config.json").write_text(
            json.dumps(config_dict, indent=2, ensure_ascii=False)
        )

        # Save base LLM, a LoRA adapter, or just a frozen-base reference.
        if lora_enabled(self.config.lora_rank):
            (save_path / "base_model_ref.txt").write_text(self.config.base_model_name)
            self.llm.save_pretrained(
                save_path / "base_model_lora",
                state_dict=llm_state,
            )
            validate_lora_adapter_path(save_path / "base_model_lora")
        elif self.config.freeze_base:
            # Only save a reference to the base model
            (save_path / "base_model_ref.txt").write_text(self.config.base_model_name)
        else:
            self.llm.save_pretrained(
                save_path / "base_model",
                state_dict=llm_state,
            )

        # Save tokenizer and processor
        self.tokenizer.save_pretrained(save_path)
        if self.processor is not None:
            try:
                self.processor.save_pretrained(save_path)
            except Exception:
                pass

        print(f"[save] JAMEL-COMPACT model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_directory: str | Path,
                        config_override: Optional[CompactConfig] = None,
                        model_parallel_override: Optional[bool] = None) -> "JAMELCompactWrapper":
        """Load a saved JAMEL-COMPACT model."""
        load_path = Path(load_directory)

        # Load config
        config_path = load_path / "compact_config.json"
        if config_path.exists():
            config_dict = json.loads(config_path.read_text())
            config = CompactConfig.from_args(**config_dict)
        else:
            config = config_override or CompactConfig()

        if config_override is not None:
            # Override specific fields
            for k, v in config_override.to_dict().items():
                if hasattr(config, k):
                    setattr(config, k, v)

        if model_parallel_override is not None:
            config.model_parallel = model_parallel_override

        # Check for base model reference (frozen) or saved base model
        base_ref_path = load_path / "base_model_ref.txt"
        base_model_dir = load_path / "base_model"
        lora_adapter_dir = load_path / "base_model_lora"
        if base_ref_path.exists():
            config.base_model_name = base_ref_path.read_text().strip()
        elif base_model_dir.exists():
            config.base_model_name = str(base_model_dir)

        if lora_enabled(config.lora_rank) and not lora_adapter_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint enables LoRA but adapter directory is missing: {lora_adapter_dir}"
            )

        # Create model
        model = cls(
            config,
            lora_adapter_path=lora_adapter_dir if lora_enabled(config.lora_rank) else None,
            lora_is_trainable=False,
        )

        # ── Reload tokenizer/processor from the checkpoint top-level dir ──
        # save_pretrained() saves tokenizer + processor to the top-level
        # checkpoint dir, but __init__ loads them from config.base_model_name
        # (which is checkpoint/base_model/ when the base LLM is saved there).
        # That subdirectory only has model weights, not tokenizer/processor
        # files, so the loads silently fail.  Retry from the top-level dir.
        if model.tokenizer is None or model.processor is None:
            try:
                if model.tokenizer is None:
                    model.tokenizer = AutoTokenizer.from_pretrained(
                        str(load_path), trust_remote_code=True,
                    )
                    print(f"[load] Tokenizer loaded from {load_path}")
            except Exception:
                pass
            try:
                if model.processor is None:
                    model.processor = AutoProcessor.from_pretrained(
                        str(load_path), trust_remote_code=True,
                    )
                    print(f"[load] Processor loaded from {load_path}")
            except Exception:
                pass

        # Load side memory weights
        side_mem_dir = load_path / "side_memory"
        if side_mem_dir.exists():
            sm_state = torch.load(side_mem_dir / "side_memories.pt", map_location="cpu")
            for key, value in list(sm_state.items()):
                if key.endswith(".inject_gate") and value.ndim == 0:
                    sm_state[key] = value.unsqueeze(0)
            incompatible = model.side_memories.load_state_dict(sm_state, strict=False)
            if incompatible.unexpected_keys:
                print(f"[load] Ignored legacy side-memory keys: {incompatible.unexpected_keys}")
            ae_state = torch.load(side_mem_dir / "action_embed.pt", map_location="cpu")
            model.action_embed.load_state_dict(ae_state)
            print(f"[load] Side memory modules loaded from {side_mem_dir}")

        return model

    # ── Parameter counting ──

    def count_parameters(self) -> dict:
        """Return parameter counts: base, new, total."""
        llm_params = sum(p.numel() for p in self.llm.parameters())
        lora_params = 0
        if lora_enabled(self.config.lora_rank):
            lora_params = sum(
                p.numel() for p in self.llm.parameters() if p.requires_grad
            )
        base_params = llm_params - lora_params
        memory_params = sum(p.numel() for p in self.side_memories.parameters())
        memory_params += sum(p.numel() for p in self.action_embed.parameters())
        new_params = memory_params + lora_params
        return {
            "base": base_params,
            "new": new_params,
            "memory": memory_params,
            "lora": lora_params,
            "total": llm_params + memory_params,
            "overhead_pct": new_params / base_params * 100,
        }
