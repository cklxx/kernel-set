#!/usr/bin/env python3
"""Backward-compatible Qwen3 names for the generic LLM engine."""

from __future__ import annotations

from .llm_greedy_engine import (
    ABLATION_VARIANTS,
    BEST_PRACTICE_MODES,
    KERNEL_SET_ENGINE_MODES,
    KernelSetLLMBestPracticePath,
    KernelSetLLMConfigurablePath,
    KernelSetLLMEnginePath,
    KernelSetLLMFullPath,
    KernelStats,
    TORCH_MANUAL_MODES,
    kernel_coverage_for_modes,
    make_rope_cache,
    merge_modes,
    rms_eps,
)

KernelSetQwen3FullPath = KernelSetLLMFullPath
KernelSetQwen3ConfigurablePath = KernelSetLLMConfigurablePath
KernelSetQwen3EnginePath = KernelSetLLMEnginePath
KernelSetQwen3BestPracticePath = KernelSetLLMBestPracticePath

__all__ = [
    "ABLATION_VARIANTS",
    "BEST_PRACTICE_MODES",
    "KERNEL_SET_ENGINE_MODES",
    "KernelSetQwen3BestPracticePath",
    "KernelSetQwen3ConfigurablePath",
    "KernelSetQwen3EnginePath",
    "KernelSetQwen3FullPath",
    "KernelStats",
    "TORCH_MANUAL_MODES",
    "kernel_coverage_for_modes",
    "make_rope_cache",
    "merge_modes",
    "rms_eps",
]
