#!/usr/bin/env python3
"""gen_optimal.py — materialize the SM x dtype x op optimal-selection table.

Pure stdlib (argparse + json). Produces ``providers/optimal.json``: the full
**Cartesian product** of ``logical_op x sm x dtype`` with, for every feasible
cell, the optimal provider, its fallback chain (always terminating in
``kernel-set``), and a ``source`` tag of ``"measured"`` (a real benchmark winner)
or ``"heuristic"`` (the curated arch/dtype baseline).

The elegant layering
--------------------
The table is built by overlaying two reproducible inputs, both embedded below:

1. **HEURISTIC baseline** — the curated ``op x sm x dtype -> {provider,
   fallback_chain}`` cells (mirrors the rank-1 of the runtime dispatch registry,
   derived from ``providers/registry.json`` + ``providers/atomic_ops.json``). It
   defines the *feasible universe*: which (op, sm, dtype) cells exist at all
   (arch-/dtype-infeasible cells are simply absent — e.g. fp8 is omitted on
   sm75, fp32 attention has no flash path).

2. **MEASURED winners** — actual benchmark results parsed from
   ``benchmarks/results/*.md``. A measured winner **OVERRIDES** the heuristic
   provider for its exact (op, sm, dtype) cell: the winning provider is hoisted
   to the front of the fallback chain, the cell is tagged ``source=measured``,
   and the measured ``metric`` + ``gpu`` are recorded.

Fill rule (read this once):

    measured-winner OVERRIDES heuristic  ·  heuristic FILLS the rest
    arch/dtype-infeasible cells are OMITTED  ·  kernel-set is the TERMINAL fallback

Every emitted ``fallback_chain`` ends in ``kernel-set`` so runtime dispatch never
dead-ends.

Reproducibility
---------------
The MEASURED and HEURISTIC blocks below are the canonical, version-controlled
inputs (the measured rows are transcribed from the cited ``source_file`` benchmark
markdown). Regeneration is therefore deterministic and offline — no GPU, no torch,
no network. ``providers/registry.json`` / ``atomic_ops.json`` /
``benchmarks/results`` remain the human source-of-record the embedded data is
distilled from.

Usage
-----
::

    python3 scripts/gen_optimal.py            # write providers/optimal.json
    python3 scripts/gen_optimal.py --check     # exit 1 if checked-in file drifts
    python3 scripts/gen_optimal.py --stdout    # print, do not write
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
OUTPUT = os.path.join(REPO_ROOT, "providers", "optimal.json")

KERNEL_SET = "kernel-set"

# --------------------------------------------------------------------------- #
# INPUT 1 — MEASURED winners (override). Transcribed from the cited benchmark
# markdown under benchmarks/results/. ``winner`` is the impl that was fastest &
# correct for that (logical_op, sm, dtype); ``metric`` is its median latency.
# ``logical_op`` uses the dotted benchmark taxonomy (domain.op); it is mapped to
# the canonical dispatch op id below via _OP_ALIASES.
# --------------------------------------------------------------------------- #
MEASURED = {
    "measured": [
        {"logical_op": "attention.prefill", "sm": 89, "dtype": "fp16",
         "winner": "flashinfer", "metric": "599.0 us",
         "source_file": "benchmarks/results/l4_vs_sota_flashinfer.md"},
        {"logical_op": "attention.decode", "sm": 89, "dtype": "fp16",
         "winner": "flashinfer", "metric": "2283.5 us",
         "source_file": "benchmarks/results/l4_vs_sota_flashinfer.md"},
        {"logical_op": "norm.rmsnorm", "sm": 89, "dtype": "fp16",
         "winner": "liger", "metric": "299.0 us",
         "source_file": "benchmarks/results/l4_vs_sota_flashinfer.md"},
        {"logical_op": "norm.fused_add_rmsnorm", "sm": 89, "dtype": "fp16",
         "winner": "kernel-set", "metric": "614.4 us",
         "source_file": "benchmarks/results/l4_vs_sota_flashinfer.md"},
        {"logical_op": "rope.apply", "sm": 89, "dtype": "fp16",
         "winner": "liger", "metric": "376.8 us",
         "source_file": "benchmarks/results/l4_vs_sota_flashinfer.md"},
        {"logical_op": "act.swiglu", "sm": 89, "dtype": "fp16",
         "winner": "flashinfer", "metric": "1532.9 us",
         "source_file": "benchmarks/results/l4_vs_sota_flashinfer.md"},
        {"logical_op": "gemm.dense", "sm": 89, "dtype": "fp16",
         "winner": "cuBLAS(torch)", "metric": "2689.0 us",
         "source_file": "benchmarks/results/l4_vs_sota_flashinfer.md"},
        {"logical_op": "loss.cross_entropy", "sm": 89, "dtype": "fp16",
         "winner": "liger", "metric": "1386.5 us",
         "source_file": "benchmarks/results/l4_vs_sota_flashinfer.md"},
        {"logical_op": "norm.rmsnorm", "sm": 80, "dtype": "bf16",
         "winner": "kernel-set", "metric": "115.7 us",
         "source_file": "benchmarks/results/a100.md"},
        {"logical_op": "rope.apply", "sm": 80, "dtype": "bf16",
         "winner": "kernel-set", "metric": "119.8 us",
         "source_file": "benchmarks/results/a100.md"},
        {"logical_op": "attention.decode", "sm": 80, "dtype": "bf16",
         "winner": "kernel-set", "metric": "4455.4 us",
         "source_file": "benchmarks/results/a100.md"},
        {"logical_op": "act.swiglu", "sm": 80, "dtype": "bf16",
         "winner": "kernel-set", "metric": "269.3 us",
         "source_file": "benchmarks/results/a100.md"},
        {"logical_op": "loss.cross_entropy", "sm": 80, "dtype": "bf16",
         "winner": "kernel-set", "metric": "1271.8 us",
         "source_file": "benchmarks/results/a100.md"},
    ]
}

# Map the dotted benchmark taxonomy -> canonical dispatch logical_op id.
_OP_ALIASES = {
    "attention.prefill": "attention_prefill",
    "attention.decode": "attention_decode",
    "norm.rmsnorm": "rmsnorm",
    "norm.fused_add_rmsnorm": "fused_add_rmsnorm",
    "rope.apply": "rope",
    "act.swiglu": "swiglu",
    "gemm.dense": "gemm",
    "loss.cross_entropy": "cross_entropy",
}

# Map benchmark ``winner`` impl labels -> canonical provider id (matches the
# runtime dispatch registry's Provider.name and the heuristic provider ids).
_WINNER_ALIASES = {
    "cuBLAS(torch)": "torch",
    "torch-cublas": "torch",
    "flashinfer": "flashinfer",
    "flashinfer-norm": "flashinfer",
    "flashinfer-rope": "flashinfer",
    "flashinfer-act": "flashinfer",
    "liger": "liger",
    "liger-ce": "liger",
    "liger-norm": "liger",
    "liger-rope": "liger",
    "kernel-set": "kernel-set",
}


# --------------------------------------------------------------------------- #
# INPUT 2 — HEURISTIC baseline (fill). The curated op x sm x dtype cells; each
# names the rank-1 ``provider`` and the full ``fallback_chain`` (ending in
# kernel-set). This block is the canonical feasible universe of the Cartesian
# table; any (op, sm, dtype) NOT present here is arch/dtype-infeasible and is
# omitted from the output.
# --------------------------------------------------------------------------- #
HEURISTIC = {
    "cells": [
        {"logical_op": "gemm", "sm": 75, "dtype": "fp16", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 75, "dtype": "bf16", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 75, "dtype": "fp32", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 80, "dtype": "fp16", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 80, "dtype": "bf16", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 80, "dtype": "fp32", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 86, "dtype": "fp16", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 86, "dtype": "bf16", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 86, "dtype": "fp32", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 89, "dtype": "fp16", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 89, "dtype": "bf16", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 89, "dtype": "fp32", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 90, "dtype": "fp16", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 90, "dtype": "bf16", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 90, "dtype": "fp32", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 100, "dtype": "fp16", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 100, "dtype": "bf16", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "gemm", "sm": 100, "dtype": "fp32", "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "fp8_gemm", "sm": 89, "dtype": "fp8",
         "provider": "torch-scaled-mm",
         "fallback_chain": ["torch-scaled-mm", "kernel-set"]},
        {"logical_op": "fp8_gemm", "sm": 90, "dtype": "fp8",
         "provider": "deep_gemm",
         "fallback_chain": ["deep_gemm", "torch-scaled-mm", "sgl-kernel",
                            "kernel-set"]},
        {"logical_op": "fp8_gemm", "sm": 100, "dtype": "fp8",
         "provider": "deep_gemm",
         "fallback_chain": ["deep_gemm", "torch-scaled-mm", "sgl-kernel",
                            "kernel-set"]},
        {"logical_op": "int8_gemm", "sm": 75, "dtype": "int8",
         "provider": "kernel-set", "fallback_chain": ["kernel-set"]},
        {"logical_op": "int8_gemm", "sm": 80, "dtype": "int8",
         "provider": "vllm",
         "fallback_chain": ["vllm", "sgl-kernel", "kernel-set"]},
        {"logical_op": "int8_gemm", "sm": 86, "dtype": "int8",
         "provider": "vllm",
         "fallback_chain": ["vllm", "sgl-kernel", "kernel-set"]},
        {"logical_op": "int8_gemm", "sm": 89, "dtype": "int8",
         "provider": "vllm",
         "fallback_chain": ["vllm", "sgl-kernel", "kernel-set"]},
        {"logical_op": "int8_gemm", "sm": 90, "dtype": "int8",
         "provider": "vllm",
         "fallback_chain": ["vllm", "sgl-kernel", "kernel-set"]},
        {"logical_op": "int8_gemm", "sm": 100, "dtype": "int8",
         "provider": "vllm-marlin-int8",
         "fallback_chain": ["vllm-marlin-int8", "sgl-kernel", "kernel-set"]},
        {"logical_op": "w4a16", "sm": 80, "dtype": "int4",
         "provider": "vllm-marlin",
         "fallback_chain": ["vllm-marlin", "kernel-set"]},
        {"logical_op": "w4a16", "sm": 86, "dtype": "int4",
         "provider": "vllm-marlin",
         "fallback_chain": ["vllm-marlin", "kernel-set"]},
        {"logical_op": "w4a16", "sm": 89, "dtype": "int4",
         "provider": "vllm-marlin",
         "fallback_chain": ["vllm-marlin", "kernel-set"]},
        {"logical_op": "w4a16", "sm": 90, "dtype": "int4",
         "provider": "vllm-machete",
         "fallback_chain": ["vllm-machete", "vllm-marlin", "kernel-set"]},
        {"logical_op": "w4a16", "sm": 100, "dtype": "int4",
         "provider": "vllm-machete",
         "fallback_chain": ["vllm-machete", "vllm-marlin", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 75, "dtype": "fp16",
         "provider": "torch-sdpa",
         "fallback_chain": ["torch-sdpa", "flashinfer", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 75, "dtype": "bf16",
         "provider": "torch-sdpa",
         "fallback_chain": ["torch-sdpa", "flashinfer", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 75, "dtype": "fp32",
         "provider": "torch-sdpa",
         "fallback_chain": ["torch-sdpa", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 80, "dtype": "fp16",
         "provider": "flash-attn",
         "fallback_chain": ["flash-attn", "torch-sdpa", "flashinfer",
                            "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 80, "dtype": "bf16",
         "provider": "flash-attn",
         "fallback_chain": ["flash-attn", "torch-sdpa", "flashinfer",
                            "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 80, "dtype": "fp32",
         "provider": "torch-sdpa",
         "fallback_chain": ["torch-sdpa", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 86, "dtype": "fp16",
         "provider": "flash-attn",
         "fallback_chain": ["flash-attn", "torch-sdpa", "flashinfer",
                            "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 86, "dtype": "bf16",
         "provider": "flash-attn",
         "fallback_chain": ["flash-attn", "torch-sdpa", "flashinfer",
                            "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 86, "dtype": "fp32",
         "provider": "torch-sdpa",
         "fallback_chain": ["torch-sdpa", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 89, "dtype": "fp16",
         "provider": "flash-attn",
         "fallback_chain": ["flash-attn", "torch-sdpa", "flashinfer",
                            "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 89, "dtype": "bf16",
         "provider": "flash-attn",
         "fallback_chain": ["flash-attn", "torch-sdpa", "flashinfer",
                            "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 89, "dtype": "fp8",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 89, "dtype": "fp32",
         "provider": "torch-sdpa",
         "fallback_chain": ["torch-sdpa", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 90, "dtype": "fp16",
         "provider": "flash-attn",
         "fallback_chain": ["flash-attn", "sgl-kernel", "torch-sdpa",
                            "flashinfer", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 90, "dtype": "bf16",
         "provider": "flash-attn",
         "fallback_chain": ["flash-attn", "sgl-kernel", "torch-sdpa",
                            "flashinfer", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 90, "dtype": "fp8",
         "provider": "flash-attn",
         "fallback_chain": ["flash-attn", "flashinfer", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 90, "dtype": "fp32",
         "provider": "torch-sdpa",
         "fallback_chain": ["torch-sdpa", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 100, "dtype": "fp16",
         "provider": "flash-attn-cute",
         "fallback_chain": ["flash-attn-cute", "flash-attn", "sgl-kernel",
                            "torch-sdpa", "flashinfer", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 100, "dtype": "bf16",
         "provider": "flash-attn-cute",
         "fallback_chain": ["flash-attn-cute", "flash-attn", "sgl-kernel",
                            "torch-sdpa", "flashinfer", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 100, "dtype": "fp8",
         "provider": "flash-attn",
         "fallback_chain": ["flash-attn", "flashinfer", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 100, "dtype": "fp32",
         "provider": "torch-sdpa",
         "fallback_chain": ["torch-sdpa", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 75, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 75, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 80, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 80, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 86, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 86, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 89, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 89, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 89, "dtype": "fp8",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 90, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 90, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 90, "dtype": "fp8",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 100, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 100, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "attention_decode", "sm": 100, "dtype": "fp8",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 75, "dtype": "bf16",
         "provider": "kernel-set", "fallback_chain": ["kernel-set"]},
        {"logical_op": "mla_decode", "sm": 75, "dtype": "fp16",
         "provider": "kernel-set", "fallback_chain": ["kernel-set"]},
        {"logical_op": "mla_decode", "sm": 80, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 80, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 86, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 86, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 89, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 89, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 89, "dtype": "fp8",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 90, "dtype": "bf16",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 90, "dtype": "fp16",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 90, "dtype": "fp8",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 100, "dtype": "bf16",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 100, "dtype": "fp16",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 100, "dtype": "fp8",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "flashinfer", "kernel-set"]},
        {"logical_op": "moe", "sm": 75, "dtype": "fp16",
         "provider": "kernel-set", "fallback_chain": ["kernel-set"]},
        {"logical_op": "moe", "sm": 75, "dtype": "bf16",
         "provider": "kernel-set", "fallback_chain": ["kernel-set"]},
        {"logical_op": "moe", "sm": 80, "dtype": "fp16",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 80, "dtype": "bf16",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 80, "dtype": "int4",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 86, "dtype": "fp16",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 86, "dtype": "bf16",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 86, "dtype": "int4",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 89, "dtype": "fp16",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 89, "dtype": "bf16",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 89, "dtype": "fp8",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 89, "dtype": "int4",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 90, "dtype": "fp16",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 90, "dtype": "bf16",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 90, "dtype": "fp8",
         "provider": "deep_gemm",
         "fallback_chain": ["deep_gemm", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 90, "dtype": "int4",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 100, "dtype": "fp16",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 100, "dtype": "bf16",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 100, "dtype": "fp8",
         "provider": "deep_gemm",
         "fallback_chain": ["deep_gemm", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "moe", "sm": 100, "dtype": "int4",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "moe_gate", "sm": 75, "dtype": "fp32",
         "provider": "kernel-set", "fallback_chain": ["kernel-set"]},
        {"logical_op": "moe_gate", "sm": 80, "dtype": "fp32",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "moe_gate", "sm": 86, "dtype": "fp32",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "moe_gate", "sm": 89, "dtype": "fp32",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "moe_gate", "sm": 90, "dtype": "fp32",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "moe_gate", "sm": 100, "dtype": "fp32",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "moe_group_gate", "sm": 75, "dtype": "fp32",
         "provider": "kernel-set", "fallback_chain": ["kernel-set"]},
        {"logical_op": "moe_group_gate", "sm": 80, "dtype": "fp32",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "moe_group_gate", "sm": 86, "dtype": "fp32",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "moe_group_gate", "sm": 89, "dtype": "fp32",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "moe_group_gate", "sm": 90, "dtype": "fp32",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "moe_group_gate", "sm": 100, "dtype": "fp32",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 75, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "vllm", "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 75, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "vllm", "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 75, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 80, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 80, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 80, "dtype": "fp32",
         "provider": "vllm",
         "fallback_chain": ["vllm", "liger", "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 86, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 86, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 86, "dtype": "fp32",
         "provider": "vllm",
         "fallback_chain": ["vllm", "liger", "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 89, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 89, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 89, "dtype": "fp32",
         "provider": "vllm",
         "fallback_chain": ["vllm", "liger", "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 90, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 90, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 90, "dtype": "fp32",
         "provider": "vllm",
         "fallback_chain": ["vllm", "liger", "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 100, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 100, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rmsnorm", "sm": 100, "dtype": "fp32",
         "provider": "vllm",
         "fallback_chain": ["vllm", "liger", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 75, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 75, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 75, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 80, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 80, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 80, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 86, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 86, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 86, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 89, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 89, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 89, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 90, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 90, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 90, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 100, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 100, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "fused_add_rmsnorm", "sm": 100, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 75, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 75, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 75, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 80, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 80, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 80, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 86, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 86, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 86, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 89, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 89, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 89, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 90, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 90, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 90, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 100, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 100, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 100, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "swiglu", "sm": 75, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "vllm", "kernel-set"]},
        {"logical_op": "swiglu", "sm": 75, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "vllm", "kernel-set"]},
        {"logical_op": "swiglu", "sm": 75, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "swiglu", "sm": 80, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "swiglu", "sm": 80, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "swiglu", "sm": 80, "dtype": "fp32",
         "provider": "vllm",
         "fallback_chain": ["vllm", "liger", "kernel-set"]},
        {"logical_op": "swiglu", "sm": 86, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "swiglu", "sm": 86, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "swiglu", "sm": 86, "dtype": "fp32",
         "provider": "vllm",
         "fallback_chain": ["vllm", "liger", "kernel-set"]},
        {"logical_op": "swiglu", "sm": 89, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "swiglu", "sm": 89, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "swiglu", "sm": 89, "dtype": "fp32",
         "provider": "vllm",
         "fallback_chain": ["vllm", "liger", "kernel-set"]},
        {"logical_op": "swiglu", "sm": 90, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "swiglu", "sm": 90, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "swiglu", "sm": 90, "dtype": "fp32",
         "provider": "vllm",
         "fallback_chain": ["vllm", "liger", "kernel-set"]},
        {"logical_op": "swiglu", "sm": 100, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "swiglu", "sm": 100, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "swiglu", "sm": 100, "dtype": "fp32",
         "provider": "vllm",
         "fallback_chain": ["vllm", "liger", "kernel-set"]},
    ]
}

# Ops that have a native kernel-set C-ABI fallback (so kernel-set may terminate
# the chain). All HEURISTIC ops above are ks-backed; cross_entropy is added by a
# measured cell (it has the ks_cross_entropy ABI). This set is used only as a
# sanity assertion when normalizing chains.
_KS_TERMINAL = KERNEL_SET


# --------------------------------------------------------------------------- #
# Build the Cartesian table by overlaying measured winners on the heuristic
# baseline.
# --------------------------------------------------------------------------- #
def _normalize_chain(chain):
    """Ensure a fallback chain is non-empty and terminates in kernel-set,
    with no duplicate kernel-set entries before the terminal one."""
    out = [p for p in chain if p != KERNEL_SET]
    out.append(KERNEL_SET)
    return out


def _hoist(provider, chain):
    """Return ``chain`` with ``provider`` moved to the front (added if absent),
    still terminating in kernel-set."""
    rest = [p for p in chain if p != provider and p != KERNEL_SET]
    return _normalize_chain([provider] + rest)


def build():
    """Materialize the op x sm x dtype Cartesian table. Returns (cells, stats)
    where cells is a list of dicts sorted by (logical_op, sm, dtype)."""
    # 1) Heuristic baseline keyed by (op, sm, dtype).
    table = {}
    for c in HEURISTIC["cells"]:
        key = (c["logical_op"], int(c["sm"]), c["dtype"])
        table[key] = {
            "logical_op": c["logical_op"],
            "sm": int(c["sm"]),
            "dtype": c["dtype"],
            "provider": c["provider"],
            "source": "heuristic",
            "fallback_chain": _normalize_chain(c["fallback_chain"]),
        }

    # 2) Overlay measured winners (override). A measured winner hoists its
    #    provider to the front of the (existing or newly-created) chain.
    measured_keys = set()
    for m in MEASURED["measured"]:
        op = _OP_ALIASES.get(m["logical_op"], m["logical_op"])
        sm = int(m["sm"])
        dtype = m["dtype"]
        winner = _WINNER_ALIASES.get(m["winner"], m["winner"])
        key = (op, sm, dtype)
        base = table.get(key)
        if base is not None:
            chain = _hoist(winner, base["fallback_chain"])
        else:
            # A measured cell with no heuristic baseline (e.g. cross_entropy,
            # which has no heuristic fill row): seed a chain winner -> ks.
            chain = _normalize_chain([winner])
        table[key] = {
            "logical_op": op,
            "sm": sm,
            "dtype": dtype,
            "provider": winner,
            "source": "measured",
            "fallback_chain": chain,
            "metric": m["metric"],
            "gpu": _sm_to_gpu(sm),
            "source_file": m["source_file"],
        }
        measured_keys.add(key)

    cells = [table[k] for k in sorted(table.keys())]
    stats = {
        "total": len(cells),
        "measured": sum(1 for c in cells if c["source"] == "measured"),
        "heuristic": sum(1 for c in cells if c["source"] == "heuristic"),
    }
    return cells, stats


# Representative GPU SKU per SM (for the measured cells' provenance label).
_SM_TO_GPU = {
    75: "t4", 80: "a100", 86: "a10", 89: "l4", 90: "h100", 100: "b200",
}


def _sm_to_gpu(sm):
    return _SM_TO_GPU.get(int(sm), f"sm{sm}")


def render(cells, stats):
    """Serialize the table to the canonical optimal.json text (stable order,
    2-space indent, trailing newline) so ``--check`` is a byte-diff."""
    doc = {
        "schema_version": 1,
        "description": (
            "SM x dtype x logical_op optimal kernel-selection table. The "
            "Cartesian product of (logical_op, sm, dtype); each feasible cell "
            "names the optimal provider, its source (measured benchmark winner "
            "or heuristic baseline), and a fallback_chain terminating in "
            "kernel-set. Regenerate with: python3 scripts/gen_optimal.py."),
        "fill_rule": (
            "measured-winner OVERRIDES heuristic; heuristic fills the rest; "
            "arch/dtype-infeasible cells omitted; kernel-set is the terminal "
            "fallback."),
        "terminal_fallback": KERNEL_SET,
        "stats": stats,
        "cells": cells,
    }
    return json.dumps(doc, indent=2, sort_keys=False) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=OUTPUT,
                    help="optimal.json to write (default: providers/optimal.json)")
    ap.add_argument("--stdout", action="store_true",
                    help="print to stdout, do not write")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the checked-in file differs from a regen")
    args = ap.parse_args(argv)

    cells, stats = build()
    text = render(cells, stats)

    if args.stdout:
        sys.stdout.write(text)
        return 0

    if args.check:
        try:
            with open(args.output) as f:
                current = f.read()
        except FileNotFoundError:
            current = ""
        if current == text:
            print(f"optimal.json is up to date ({stats['total']} cells: "
                  f"{stats['measured']} measured, {stats['heuristic']} "
                  f"heuristic).")
            return 0
        sys.stderr.write(
            "optimal.json is OUT OF DATE — run: python3 scripts/gen_optimal.py\n")
        return 1

    with open(args.output, "w") as f:
        f.write(text)
    print(f"wrote {args.output} ({stats['total']} cells: {stats['measured']} "
          f"measured, {stats['heuristic']} heuristic).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
