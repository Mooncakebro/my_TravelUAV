"""
JAMEL-COMPACT: Unified LLM with per-layer side memory.

Subpackage containing the model, data, training, and evaluation code for
the JAMEL-COMPACT framework — a single pretrained LLM (e.g. Qwen3-VL-2B/8B)
augmented with per-layer side memory modules that evolve via FiLM-GRU
prediction and Kalman-Filter correction.
"""

# Lazy imports to avoid requiring torch for data-only operations
def __getattr__(name):
    if name == "JAMELCompactWrapper":
        from .model import JAMELCompactWrapper
        return JAMELCompactWrapper
    if name == "SideMemoryModule":
        from .model import SideMemoryModule
        return SideMemoryModule
    if name == "FiLMGRUCell":
        from .model import FiLMGRUCell
        return FiLMGRUCell
    if name == "CompactConfig":
        from .config import CompactConfig
        return CompactConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "JAMELCompactWrapper",
    "SideMemoryModule",
    "FiLMGRUCell",
    "CompactConfig",
]