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
   defines the *feasible universe*: which (op, sm, dtype) cells exist at all.
   Arch-/dtype-infeasible cells are OMITTED by the capability gate derived from
   ``models/gpu_caps.json`` (the single source of truth): bf16/tf32 < sm80, fp8
   < sm89, fp4 < sm100 are dropped (so e.g. no sm75/bf16 rows exist).

2. **MEASURED winners** — actual benchmark results parsed from
   ``benchmarks/results/*.md``. A measured winner **OVERRIDES** the heuristic
   provider for its exact (op, sm, dtype) cell: the winning provider is hoisted
   to the front of the fallback chain, the cell is tagged ``source=measured``,
   and the measured ``metric`` + ``gpu`` are recorded.

Fill rule (read this once):

    measured-winner OVERRIDES heuristic  ·  heuristic FILLS the rest
    arch/dtype-infeasible cells are OMITTED  ·  kernel-set is the TERMINAL fallback

Every emitted ``fallback_chain`` ends in ``kernel-set`` so runtime dispatch never
dead-ends, and every emitted cell satisfies ``provider == fallback_chain[0]``
(a measured ``kernel-set`` winner emits chain ``["kernel-set"]``).

Design invariants (enforced by ``validate()`` on every build, so
``--check`` fails CI on any violation):

1. ``cell.provider == cell.fallback_chain[0]`` for every cell.
2. every ``provider`` + ``fallback_chain`` entry EXISTS as a runtime provider
   for that ``logical_op`` in ``backends/_registry.py``.
3. no arch/dtype-infeasible cell (bf16/tf32 < sm80, fp8 < sm89, fp4 < sm100),
   per ``models/gpu_caps.json``.
5. every dtype token ``optimal.py`` accepts appears in >=1 feasible cell OR is a
   documented degradation (``_DOCUMENTED_DEGRADATIONS``; fp4 -> kernel-set).
6. the table's ``logical_op`` set matches the dispatch ``OP_ORDER``.

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
        # FP8 BLOCKWISE GEMM (DeepSeek-V3 recipe). sm89 has no fp8 HW provider, so
        # kernel-set's portable blockwise kernel IS the winner there (chain
        # collapses to ["kernel-set"]); DeepGEMM wins on Hopper/Blackwell.
        {"logical_op": "fp8_gemm_blockwise", "sm": 89, "dtype": "fp8",
         "provider": "kernel-set",
         "fallback_chain": ["kernel-set"]},
        {"logical_op": "fp8_gemm_blockwise", "sm": 90, "dtype": "fp8",
         "provider": "deep_gemm",
         "fallback_chain": ["deep_gemm", "sgl-kernel", "kernel-set"]},
        {"logical_op": "fp8_gemm_blockwise", "sm": 100, "dtype": "fp8",
         "provider": "deep_gemm",
         "fallback_chain": ["deep_gemm", "sgl-kernel", "kernel-set"]},
        # Per-token-group (1x128) dynamic fp8 quant. deep_gemm (sm90) only joins
        # the chain on Hopper+; sm89 chain omits it (min_sm gate).
        {"logical_op": "per_token_group_quant", "sm": 89, "dtype": "fp8",
         "provider": "vllm",
         "fallback_chain": ["vllm", "sgl-kernel", "kernel-set"]},
        {"logical_op": "per_token_group_quant", "sm": 90, "dtype": "fp8",
         "provider": "vllm",
         "fallback_chain": ["vllm", "sgl-kernel", "deep_gemm", "kernel-set"]},
        {"logical_op": "per_token_group_quant", "sm": 100, "dtype": "fp8",
         "provider": "vllm",
         "fallback_chain": ["vllm", "sgl-kernel", "deep_gemm", "kernel-set"]},
        # NVFP4 GEMM (Blackwell sm100 native fp4). fp4 is feasible only on sm100+
        # (gpu_caps threshold), so only the Blackwell cell exists; sm120 folds in.
        {"logical_op": "nvfp4_gemm", "sm": 100, "dtype": "fp4",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "vllm", "kernel-set"]},
        # MXFP4 GEMM (OCP microscaling; gpt-oss). Blackwell native.
        {"logical_op": "mxfp4_gemm", "sm": 100, "dtype": "fp4",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "vllm", "torchao", "kernel-set"]},
        # FP8 attention compute. SageAttention (sm80+) wins on Ada; FA3 fp8 on
        # Hopper/Blackwell (sm90+), where it joins the chain.
        {"logical_op": "fp8_attention", "sm": 89, "dtype": "fp8",
         "provider": "sage-attn",
         "fallback_chain": ["sage-attn", "kernel-set"]},
        {"logical_op": "fp8_attention", "sm": 90, "dtype": "fp8",
         "provider": "flash-attn-3",
         "fallback_chain": ["flash-attn-3", "sage-attn", "kernel-set"]},
        {"logical_op": "fp8_attention", "sm": 100, "dtype": "fp8",
         "provider": "flash-attn-3",
         "fallback_chain": ["flash-attn-3", "sage-attn", "kernel-set"]},
        # FP8 KV-cache quantize-on-write. vLLM reshape_and_cache_flash fp8.
        {"logical_op": "fp8_kv_cache", "sm": 89, "dtype": "fp8",
         "provider": "vllm",
         "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "fp8_kv_cache", "sm": 90, "dtype": "fp8",
         "provider": "vllm",
         "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "fp8_kv_cache", "sm": 100, "dtype": "fp8",
         "provider": "vllm",
         "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 75, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 75, "dtype": "bf16",
         "provider": "torch-sdpa",
         "fallback_chain": ["torch-sdpa", "flashinfer", "kernel-set"]},
        {"logical_op": "attention_prefill", "sm": 75, "dtype": "fp32",
         "provider": "kernel-set",
         "fallback_chain": ["kernel-set"]},
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
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
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
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
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
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 100, "dtype": "bf16",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 100, "dtype": "fp16",
         "provider": "sgl-kernel",
         "fallback_chain": ["sgl-kernel", "flashinfer", "kernel-set"]},
        {"logical_op": "mla_decode", "sm": 100, "dtype": "fp8",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
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
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rope", "sm": 80, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rope", "sm": 80, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 86, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rope", "sm": 86, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rope", "sm": 86, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 89, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rope", "sm": 89, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rope", "sm": 89, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 90, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rope", "sm": 90, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rope", "sm": 90, "dtype": "fp32",
         "provider": "vllm", "fallback_chain": ["vllm", "kernel-set"]},
        {"logical_op": "rope", "sm": 100, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
        {"logical_op": "rope", "sm": 100, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "vllm", "liger",
                            "kernel-set"]},
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
        # --- gemma_rmsnorm: Gemma (weight+1) RMSNorm. flashinfer leads on
        #     fp16/bf16 (sm75+), sgl woven in (sm80+); fp32 has no external
        #     provider -> kernel-set. -------------------------------------------
        {"logical_op": "gemma_rmsnorm", "sm": 75, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 75, "dtype": "fp32",
         "provider": "kernel-set",
         "fallback_chain": ["kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 80, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 80, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 80, "dtype": "fp32",
         "provider": "kernel-set",
         "fallback_chain": ["kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 86, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 86, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 86, "dtype": "fp32",
         "provider": "kernel-set",
         "fallback_chain": ["kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 89, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 89, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 89, "dtype": "fp32",
         "provider": "kernel-set",
         "fallback_chain": ["kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 90, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 90, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 90, "dtype": "fp32",
         "provider": "kernel-set",
         "fallback_chain": ["kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 100, "dtype": "fp16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 100, "dtype": "bf16",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "gemma_rmsnorm", "sm": 100, "dtype": "fp32",
         "provider": "kernel-set",
         "fallback_chain": ["kernel-set"]},
        # --- cross_entropy: liger fused CE leads on sm80+ (fp16/bf16/fp32);
        #     torch is the sm75 leader (liger min_sm 80) and the universal
        #     rank-2. (sm89 fp16 + sm80 bf16 are MEASURED overrides below.) ----
        {"logical_op": "cross_entropy", "sm": 75, "dtype": "fp16",
         "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 75, "dtype": "fp32",
         "provider": "torch",
         "fallback_chain": ["torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 80, "dtype": "fp16",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 80, "dtype": "fp32",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 80, "dtype": "bf16",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 86, "dtype": "fp16",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 86, "dtype": "fp32",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 86, "dtype": "bf16",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 89, "dtype": "fp32",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 89, "dtype": "bf16",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 90, "dtype": "fp16",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 90, "dtype": "fp32",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 90, "dtype": "bf16",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 100, "dtype": "fp16",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 100, "dtype": "fp32",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        {"logical_op": "cross_entropy", "sm": 100, "dtype": "bf16",
         "provider": "liger",
         "fallback_chain": ["liger", "torch", "kernel-set"]},
        # --- sampling: fused temp/top-k/top-p over fp32 probs (dtype-agnostic
        #     downstream of the logits). flashinfer leads (sm75+); sgl woven in
        #     (sm80+). ---------------------------------------------------------
        {"logical_op": "sampling", "sm": 75, "dtype": "fp32",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "kernel-set"]},
        {"logical_op": "sampling", "sm": 80, "dtype": "fp32",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "sampling", "sm": 86, "dtype": "fp32",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "sampling", "sm": 89, "dtype": "fp32",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "sampling", "sm": 90, "dtype": "fp32",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
        {"logical_op": "sampling", "sm": 100, "dtype": "fp32",
         "provider": "flashinfer",
         "fallback_chain": ["flashinfer", "sgl-kernel", "kernel-set"]},
    ]
}

# --------------------------------------------------------------------------- #
# CAPABILITY GATE — derived from the single source of truth, models/gpu_caps.json.
# A (sm, dtype) cell is FEASIBLE only if the dtype is supported by that SM. We
# load the named ``sm_thresholds`` ({fp8: 89, bf16: 80, tf32: 80, fp4: 100, ...})
# from the canonical JSON; an absent/malformed file falls back to the inlined
# defaults (kept identical to the JSON). dtypes with no min-SM threshold
# (fp16/fp32/int8/int4) are feasible on every SM in the table.
# --------------------------------------------------------------------------- #
_GPU_CAPS_PATH = os.path.join(REPO_ROOT, "models", "gpu_caps.json")

# Map an optimal-table dtype token -> the gpu_caps capability key it requires.
# fp16/fp32/int4/int8 have no arch threshold (feasible everywhere we model).
_DTYPE_CAP_KEY = {
    "bf16": "bf16",
    "tf32": "tf32",
    "fp8": "fp8",
    "fp4": "fp4",
}

_SM_THRESHOLDS_FALLBACK = {
    "fp8": 89, "bf16": 80, "tf32": 80, "int8": 75, "int4": 75,
    "fp4": 100, "wgmma": 90,
}


def _load_sm_thresholds():
    """{capability: min_sm} from models/gpu_caps.json (the single source of
    truth), else the inlined fallback. Pure stdlib; never raises."""
    try:
        with open(_GPU_CAPS_PATH) as f:
            caps = json.load(f)
        return {k: int(v) for k, v in caps["sm_thresholds"].items()
                if not k.startswith("_")}
    except Exception:
        return dict(_SM_THRESHOLDS_FALLBACK)


_SM_THRESHOLDS = _load_sm_thresholds()


def _dtype_feasible(sm, dtype):
    """Is ``dtype`` supported by ``sm`` per the gpu_caps thresholds? fp16/fp32/
    int4/int8 are feasible on every SM; bf16<80, tf32<80, fp8<89, fp4<100 are
    infeasible and such cells are OMITTED from the table."""
    cap = _DTYPE_CAP_KEY.get(dtype)
    if cap is None:
        return True
    return int(sm) >= int(_SM_THRESHOLDS.get(cap, 0))


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
    """Return ``chain`` with ``provider`` as the winner at index 0, still
    terminating in kernel-set. The INVARIANT is ``provider == chain[0]``.

    A measured ``kernel-set`` winner is special-cased: kernel-set is the
    terminal fallback, so it can never sit non-terminally in the chain. When the
    measured winner IS kernel-set we collapse the chain to ``["kernel-set"]`` so
    the invariant ``provider == chain[0]`` holds (instead of stripping ks from
    the middle and re-appending it at the end, which left a different provider at
    index 0 — the HIGH bug from the review)."""
    if provider == KERNEL_SET:
        return [KERNEL_SET]
    rest = [p for p in chain if p != provider and p != KERNEL_SET]
    return _normalize_chain([provider] + rest)


def build():
    """Materialize the op x sm x dtype Cartesian table. Returns (cells, stats)
    where cells is a list of dicts sorted by (logical_op, sm, dtype).

    Infeasible (sm, dtype) cells (e.g. bf16 on sm75, fp8 on sm80) are OMITTED
    via the gpu_caps capability gate. Measured winners override the heuristic
    baseline; every emitted cell satisfies ``provider == fallback_chain[0]`` and
    terminates in kernel-set."""
    # 1) Heuristic baseline keyed by (op, sm, dtype). Omit infeasible dtypes.
    table = {}
    for c in HEURISTIC["cells"]:
        sm = int(c["sm"])
        if not _dtype_feasible(sm, c["dtype"]):
            continue
        key = (c["logical_op"], sm, c["dtype"])
        table[key] = {
            "logical_op": c["logical_op"],
            "sm": sm,
            "dtype": c["dtype"],
            "provider": c["provider"],
            "source": "heuristic",
            "fallback_chain": _hoist(c["provider"],
                                     _normalize_chain(c["fallback_chain"])),
        }

    # 2) Overlay measured winners (override). A measured winner hoists its
    #    provider to index 0 of the (existing or newly-created) chain.
    measured_keys = set()
    for m in MEASURED["measured"]:
        op = _OP_ALIASES.get(m["logical_op"], m["logical_op"])
        sm = int(m["sm"])
        dtype = m["dtype"]
        winner = _WINNER_ALIASES.get(m["winner"], m["winner"])
        # A measured row must itself be arch/dtype-feasible.
        if not _dtype_feasible(sm, dtype):
            raise AssertionError(
                f"measured row {op}/sm{sm}/{dtype} is dtype-infeasible "
                f"(min_sm for {dtype} is {_SM_THRESHOLDS.get(_DTYPE_CAP_KEY.get(dtype), 0)})")
        key = (op, sm, dtype)
        base = table.get(key)
        if base is not None:
            chain = _hoist(winner, base["fallback_chain"])
        else:
            # A measured cell with no heuristic baseline: seed a chain winner ->
            # ks (or just ["kernel-set"] if the winner IS kernel-set).
            chain = _hoist(winner, _normalize_chain([winner]))
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
    validate(cells)
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


# --------------------------------------------------------------------------- #
# VALIDATION — the design invariants, enforced at generate AND --check time so
# CI's existing `gen_optimal.py --check` gate guards them.
# --------------------------------------------------------------------------- #
# dtype tokens that optimal.py accepts but that resolve to the kernel-set
# fallback by design rather than appearing in a feasible cell. Each must be
# documented here (invariant 5: every accepted dtype token appears in >=1
# feasible cell OR is a documented degradation path).
_DOCUMENTED_DEGRADATIONS = {
    # NVFP4/MXFP4 are now WIRED for the dedicated fp4 ops (nvfp4_gemm/mxfp4_gemm)
    # on Blackwell (sm100+), so fp4 DOES appear in feasible cells. It stays listed
    # here because for every OTHER op (and on sm<100) a fp4 request still degrades
    # to the kernel-set fallback rather than borrowing the int4 cell — invariant 5
    # is satisfied either way (fp4 is present AND documented as a partial path).
    "fp4",
}


def _load_registry_contract():
    """Runtime registry + probe objects for strong source-tree validation.

    Importing the registry is import-safe: adapters are lazy and no provider
    libraries are imported. In the source tree the binding package is expected to
    exist; import failure is fatal so ``--check`` cannot silently skip the
    strongest invariants. Only a stripped installed-wheel context with no binding
    tree gets the narrow fallback (None, None, None).
    """
    import importlib

    bindings = os.path.join(REPO_ROOT, "bindings", "python")
    pkg_dir = os.path.join(bindings, "kernel_set")
    if not os.path.isdir(pkg_dir):
        return None, None, None
    if bindings not in sys.path:
        sys.path.insert(0, bindings)
    try:
        reg = importlib.import_module("kernel_set.backends._registry")
        probe = importlib.import_module("kernel_set.backends._probe")
    except Exception as e:
        raise AssertionError(
            "source-tree registry/probe import failed; cannot validate "
            f"optimal.json against runtime dispatch contract: {e}") from e
    providers = {name: {p.name: p for p in op.providers}
                 for name, op in reg.OPS.items()}
    return providers, list(reg.OP_ORDER), probe


def _accepted_dtype_tokens():
    """The set of canonical dtype tokens optimal.py accepts (its alias map's
    values). Import-safe; returns a conservative default if unavailable."""
    import importlib

    bindings = os.path.join(REPO_ROOT, "bindings", "python")
    if bindings not in sys.path:
        sys.path.insert(0, bindings)
    try:
        opt = importlib.import_module("kernel_set.backends.optimal")
        return set(opt._DTYPE_ALIASES.values())
    except Exception:
        return {"fp16", "bf16", "fp32", "fp8", "int8", "int4", "fp4"}


def validate(cells):
    """Enforce the table's design invariants. Raises AssertionError on any
    violation. Run by build() (and therefore by --check)."""
    # Invariant 1: provider == fallback_chain[0] for EVERY cell.
    for c in cells:
        assert c["fallback_chain"], c
        assert c["fallback_chain"][-1] == KERNEL_SET, \
            f"chain must terminate in kernel-set: {c}"
        assert c["fallback_chain"].count(KERNEL_SET) == 1, \
            f"kernel-set must appear exactly once (terminal): {c}"
        assert c["provider"] == c["fallback_chain"][0], \
            f"provider must lead its chain (provider != chain[0]): {c}"

    # Invariant 3: no infeasible (sm, dtype) cell (bf16<80, tf32<80, fp8<89,
    # fp4<100), per the gpu_caps thresholds.
    for c in cells:
        assert _dtype_feasible(c["sm"], c["dtype"]), \
            f"infeasible dtype cell (dtype unsupported by sm): {c}"

    # Invariant 2: every provider + every fallback_chain entry exists as a
    # runtime provider for that logical_op in the registry, and every
    # non-kernel-set entry is actually selectable under the registry's own
    # callability / arch / dtype gates.
    op_providers, op_order, probe = _load_registry_contract()
    if op_providers is not None:
        for c in cells:
            op = c["logical_op"]
            known = op_providers.get(op)
            assert known is not None, \
                f"logical_op {op!r} has no runtime registry Op: {c}"
            for p in [c["provider"]] + c["fallback_chain"]:
                assert p in known, (
                    f"provider {p!r} not registered for op {op!r} "
                    f"(known: {sorted(known)}): {c}")
                provider = known[p]
                if p == KERNEL_SET:
                    continue
                assert provider.call is not None, (
                    f"provider {p!r} for op {op!r} is not callable: {c}")
                assert int(c["sm"]) >= int(provider.min_sm), (
                    f"provider {p!r} for op {op!r} is gated sm"
                    f"{provider.min_sm}+ but cell is sm{c['sm']}: {c}")
                assert probe.dtype_arch_ok(c["dtype"], int(c["sm"])), (
                    f"dtype {c['dtype']!r} is not hardware-feasible on "
                    f"sm{c['sm']}: {c}")
                assert probe.dtype_ok(c["dtype"], provider.dtypes), (
                    f"provider {p!r} for op {op!r} does not support dtype "
                    f"{c['dtype']!r} (provider dtypes={provider.dtypes!r}): {c}")

        # Invariant 6: the table's logical_ops match the dispatch OP_ORDER (it
        # may be a documented superset — measured-only ops with no Op are not
        # allowed, but extra dispatch ops must each have >=1 cell).
        table_ops = {c["logical_op"] for c in cells}
        missing = [o for o in op_order if o not in table_ops]
        assert not missing, (
            f"dispatch OP_ORDER ops missing from optimal.json (every dispatch "
            f"op must have >=1 feasible cell): {missing}")
        extra = [o for o in table_ops if o not in set(op_order)]
        assert not extra, (
            f"optimal.json names ops not in dispatch OP_ORDER: {extra}")

    # Invariant 5: every dtype token optimal.py accepts appears in >=1 feasible
    # cell OR is a documented degradation path.
    present = {c["dtype"] for c in cells}
    for tok in _accepted_dtype_tokens():
        assert tok in present or tok in _DOCUMENTED_DEGRADATIONS, (
            f"accepted dtype token {tok!r} has no feasible cell and is not a "
            f"documented degradation (add a cell or list it in "
            f"_DOCUMENTED_DEGRADATIONS)")


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
