"""Shared LoRA configuration helpers for COMPACT and baseline training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEFAULT_LORA_TARGET_MODULES_CSV = ",".join(DEFAULT_LORA_TARGET_MODULES)
LORA_ADAPTER_NAME = "default"
LORA_ADAPTER_WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def lora_enabled(rank: int) -> bool:
    return rank > 0


def normalize_lora_target_modules(value: str | Sequence[str]) -> str | list[str]:
    if isinstance(value, str):
        modules = [module.strip() for module in value.split(",") if module.strip()]
    else:
        modules = [str(module).strip() for module in value if str(module).strip()]

    if modules == ["all-linear"]:
        return "all-linear"
    if "all-linear" in modules:
        raise ValueError("'all-linear' cannot be combined with explicit LoRA target modules")
    if not modules:
        raise ValueError("LoRA target modules cannot be empty")
    return modules


def validate_lora_settings(
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: str | Sequence[str],
    bias: str = "none",
) -> str | list[str]:
    if rank < 0:
        raise ValueError("LoRA rank must be >= 0")
    if alpha < 0:
        raise ValueError("LoRA alpha must be >= 0")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("LoRA dropout must satisfy 0 <= dropout < 1")
    if bias not in {"none", "all", "lora_only"}:
        raise ValueError("LoRA bias must be one of: none, all, lora_only")

    normalized_targets = normalize_lora_target_modules(target_modules)
    if rank == 0:
        if alpha != 0 or dropout != 0.0 or bias != "none":
            raise ValueError(
                "LoRA rank is 0, so alpha/dropout/bias must keep their disabled defaults"
            )
        return normalized_targets
    if alpha <= 0:
        raise ValueError("LoRA alpha must be > 0 when LoRA rank is enabled")
    return normalized_targets


def read_lora_adapter_config(adapter_path: str | Path) -> dict:
    adapter_path = Path(adapter_path)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"LoRA adapter_config.json not found in {adapter_path}")
    try:
        adapter_config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LoRA adapter config: {config_path}") from exc
    if str(adapter_config.get("peft_type", "LORA")).upper() != "LORA":
        raise ValueError(f"Checkpoint is not a LoRA adapter: {config_path}")
    return adapter_config


def validate_lora_adapter_path(adapter_path: str | Path) -> Path:
    adapter_path = Path(adapter_path)
    read_lora_adapter_config(adapter_path)
    if not any(
        (adapter_path / filename).is_file()
        for filename in LORA_ADAPTER_WEIGHT_FILENAMES
    ):
        expected = " or ".join(LORA_ADAPTER_WEIGHT_FILENAMES)
        raise FileNotFoundError(
            f"LoRA adapter weights not found in {adapter_path}; expected {expected}"
        )
    return adapter_path


def _validate_saved_adapter_settings(
    adapter_config: dict,
    *,
    rank: int,
    alpha: int,
) -> None:
    saved_rank = adapter_config.get("r")
    saved_alpha = adapter_config.get("lora_alpha")
    if saved_rank is not None and int(saved_rank) != rank:
        raise ValueError(
            f"LoRA rank mismatch: compact config has {rank}, adapter has {saved_rank}"
        )
    if saved_alpha is not None and int(saved_alpha) != alpha:
        raise ValueError(
            f"LoRA alpha mismatch: compact config has {alpha}, adapter has {saved_alpha}"
        )


def load_lora_adapter(
    model,
    adapter_path: str | Path,
    *,
    is_trainable: bool = False,
    expected_rank: int | None = None,
    expected_alpha: int | None = None,
):
    adapter_path = validate_lora_adapter_path(adapter_path)
    adapter_config = read_lora_adapter_config(adapter_path)
    if expected_rank is not None and expected_alpha is not None:
        _validate_saved_adapter_settings(
            adapter_config,
            rank=expected_rank,
            alpha=expected_alpha,
        )

    try:
        import peft  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Loading this LoRA checkpoint requires PEFT>=0.19.1. Install the "
            "training dependencies with `uv sync --extra train`."
        ) from exc

    try:
        model.load_adapter(
            str(adapter_path),
            adapter_name=LORA_ADAPTER_NAME,
            is_trainable=is_trainable,
        )
        model.set_adapter(LORA_ADAPTER_NAME)
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "This LoRA checkpoint requires Transformers PEFT integration and "
            "PEFT>=0.19.1."
        ) from exc

    active_adapters = getattr(model, "active_adapters", None)
    if callable(active_adapters):
        active = active_adapters()
        if isinstance(active, str):
            active = [active]
        if LORA_ADAPTER_NAME not in active:
            raise RuntimeError(
                f"LoRA adapter loaded but is not active: {LORA_ADAPTER_NAME}"
            )
    return model


def configure_lora(
    model,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: str | Sequence[str],
    bias: str = "none",
    adapter_path: str | Path | None = None,
    is_trainable: bool = True,
):
    normalized_targets = validate_lora_settings(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        bias=bias,
    )
    if not lora_enabled(rank):
        return model

    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise RuntimeError(
            "LoRA training requires PEFT>=0.19.1. Install the training "
            "dependencies with `uv sync --extra train`."
        ) from exc

    if adapter_path is None:
        adapter_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=normalized_targets,
            bias=bias,
        )
        model.add_adapter(adapter_config, adapter_name=LORA_ADAPTER_NAME)
        model.set_adapter(LORA_ADAPTER_NAME)
    else:
        model = load_lora_adapter(
            model,
            adapter_path,
            is_trainable=is_trainable,
            expected_rank=rank,
            expected_alpha=alpha,
        )
    return model


def enable_lora_gradient_checkpointing(model) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
