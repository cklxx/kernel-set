"""Tiny experimental engine loops."""

from .llm_greedy_engine import (
    ABLATION_VARIANTS,
    BEST_PRACTICE_MODES,
    KernelSetCausalLMBestPracticePath,
    KernelSetCausalLMConfigurablePath,
    KernelSetCausalLMFullPath,
    KernelSetLLMBestPracticePath,
    KernelSetLLMConfigurablePath,
    KernelSetLLMFullPath,
    KernelStats,
    TORCH_MANUAL_MODES,
    kernel_coverage_for_modes,
    make_rope_cache,
    merge_modes,
    rms_eps,
)
from .qwen3_greedy_engine import (
    KernelSetQwen3BestPracticePath,
    KernelSetQwen3ConfigurablePath,
    KernelSetQwen3FullPath,
)
from .tiny_engine import DiffusionResult, GreedyResult, TinyEngine

__all__ = [
    "ABLATION_VARIANTS",
    "BEST_PRACTICE_MODES",
    "DiffusionResult",
    "GreedyResult",
    "KernelSetCausalLMBestPracticePath",
    "KernelSetCausalLMConfigurablePath",
    "KernelSetCausalLMFullPath",
    "KernelSetLLMBestPracticePath",
    "KernelSetLLMConfigurablePath",
    "KernelSetLLMFullPath",
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
