#!/usr/bin/env python3
"""Backward-compatible Qwen3 names for the generic causal-LM engine."""

from __future__ import annotations

from .causal_lm_greedy_engine import (
    ABLATION_VARIANTS,
    BEST_PRACTICE_MODES,
    KernelSetCausalLMBestPracticePath,
    KernelSetCausalLMConfigurablePath,
    KernelSetCausalLMFullPath,
    KernelStats,
    TORCH_MANUAL_MODES,
    kernel_coverage_for_modes,
    make_rope_cache,
    merge_modes,
    rms_eps,
)

KernelSetQwen3FullPath = KernelSetCausalLMFullPath
KernelSetQwen3ConfigurablePath = KernelSetCausalLMConfigurablePath
KernelSetQwen3BestPracticePath = KernelSetCausalLMBestPracticePath

__all__ = [
    "ABLATION_VARIANTS",
    "BEST_PRACTICE_MODES",
    "KernelSetQwen3BestPracticePath",
    "KernelSetQwen3ConfigurablePath",
    "KernelSetQwen3FullPath",
    "KernelStats",
    "TORCH_MANUAL_MODES",
    "kernel_coverage_for_modes",
    "make_rope_cache",
    "merge_modes",
    "rms_eps",
]
