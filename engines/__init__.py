"""Tiny experimental engine loops."""

from .qwen3_greedy_engine import (
    ABLATION_VARIANTS,
    BEST_PRACTICE_MODES,
    KernelSetQwen3BestPracticePath,
    KernelSetQwen3ConfigurablePath,
    KernelSetQwen3FullPath,
    KernelStats,
    TORCH_MANUAL_MODES,
    kernel_coverage_for_modes,
    make_rope_cache,
    merge_modes,
    rms_eps,
)
from .tiny_engine import DiffusionResult, GreedyResult, TinyEngine

__all__ = [
    "ABLATION_VARIANTS",
    "BEST_PRACTICE_MODES",
    "DiffusionResult",
    "GreedyResult",
    "KernelSetQwen3BestPracticePath",
    "KernelSetQwen3ConfigurablePath",
    "KernelSetQwen3FullPath",
    "KernelStats",
    "TORCH_MANUAL_MODES",
    "TinyEngine",
    "kernel_coverage_for_modes",
    "make_rope_cache",
    "merge_modes",
    "rms_eps",
]
