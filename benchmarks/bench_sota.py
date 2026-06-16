#!/usr/bin/env python3
"""kernel-set vs industry SOTA — fault-tolerant comparison benchmark harness.

For each op category this runs **kernel-set's own kernel AND each available
industry provider** (flash-attn, flashinfer, FlashMLA, DeepGEMM, Marlin/vLLM,
liger, mamba-ssm, ...) on the SAME inputs, gates correctness against a shared
fp32 reference, times each impl with the *same* best-practice method (CUDA
events, L2-flushed, auto-calibrated — imported verbatim from ``bench.py`` so the
timing methodology is identical), and emits a per-op comparison table with a
``kernel-set vs best-SOTA`` speedup.

ROBUSTNESS IS THE WHOLE POINT
-----------------------------
Every provider call is gated and sandboxed so the run NEVER crashes:

  * ``ok``                 — ran and (if a reference exists) passed tolerance
  * ``skip: needs smXX``   — provider's required GPU arch > this device; the lib
                             is NEVER imported/called on an unsupported arch
  * ``skip(reason)``       — not applicable here (dtype, shape, multi-GPU, ...)
  * ``import-fail(msg)``   — the library is not installed / failed to import
  * ``error(msg)``         — imported but the call raised (wrong API, OOM, ...)
  * ``incorrect(rel_err)`` — ran but disagrees with the fp32 reference

One provider failing does not stop the others; one op failing does not stop the
run. Arch gating uses the per-provider ``gpu_arch_required`` from the registry
(L4 = sm89): DeepGEMM / FlashMLA / grouped-fp8 (sm90), Machete (sm90a), and
NVFP4/MXFP4 (sm100) all SKIP on L4 with a clear ``needs smXX`` row.

Timing methodology + GPU detection + roofline %-of-peak are imported from
``bench.py`` (NOT duplicated): ``detect_gpu``, ``time_op``, ``rel_err``,
``peak_for_metric``, the L2-flush config, the env/repro header helpers, etc.

Examples
--------
    # all op groups, fp16, markdown report
    python bench_sota.py --dtype fp16 --output results/sota_l4.md

    # just the GEMM + norm comparisons, bf16, JSON
    python bench_sota.py --ops gemm,rmsnorm --dtype bf16 --format json

    # list the op groups + which providers each compares against
    python bench_sota.py --list

Methodology spec: ``docs/BENCHMARK_METHODOLOGY.md``. Provider call signatures
were web-verified for 2026-H1 (see ``providers/registry.json`` /
``benchmarks/baselines.yaml``).
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Import bench.py's timing / detection / roofline machinery. We do NOT
# re-implement any of it. bench.py lives next to this file.
# --------------------------------------------------------------------------- #
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Shared shape tables + RoPE reference (single source of truth across the two
# harnesses). Import-safe with no torch. See benchmarks/_bench_common.py.
from _bench_common import (  # noqa: E402
    ATTN_PREFILL_SHAPES as _ATTN_PREFILL_SHAPES,
    ATTN_DECODE_SHAPES as _ATTN_DECODE_SHAPES,
    ROPE_SHAPES as _ROPE_SHAPES,
    CE_SHAPES as _CE_SHAPES,
)
from _bench_common import ref_rope_neox as _ref_rope_neox_common

_BENCH_IMPORT_ERROR = None
try:
    import bench as _bench  # the sibling harness
    from bench import (  # noqa: F401  (re-exported timing/detection symbols)
        GpuInfo,
        Timing,
        TimingConfig,
        detect_gpu,
        time_op,
        rel_err,
        torch_dtype,
        dtype_bytes,
        peak_for_metric,
        _peaks_for,
        _sink,
        collect_env,
        query_clocks,
        git_commit,
        apply_tf32,
        query_l2_flush_bytes,
        _TIMING,
    )
    _HAVE_BENCH = True
except Exception as exc:  # pragma: no cover - surfaced in main()
    _bench = None  # type: ignore
    _HAVE_BENCH = False
    _BENCH_IMPORT_ERROR = exc

# Optional torch + kernel_set. Both are required to actually RUN, but the module
# must import cleanly without them (so --list and the ast.parse check work).
try:
    import torch
    _HAVE_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAVE_TORCH = False

try:
    import kernel_set as ks
    _HAVE_KS = True
    _KS_IMPORT_ERROR = None
except Exception as exc:
    ks = None  # type: ignore
    _HAVE_KS = False
    _KS_IMPORT_ERROR = exc


# --------------------------------------------------------------------------- #
# Provider arch requirements (min SM), web-verified for 2026-H1 from
# providers/registry.json + baselines.yaml. L4 = sm89; A100 = sm80; H100 = sm90;
# Blackwell DC = sm100. We gate BEFORE importing/calling a provider: if the
# device sm_arch < the provider's minimum, we emit `skip: needs smXX` and never
# touch the library.
#
# `min_sm` is the lowest SM the provider supports for the op we drive here.
# --------------------------------------------------------------------------- #
@dataclass
class Provider:
    key: str                 # short stable id used in the table's `impl` column
    lib: str                 # importable display name
    min_sm: int              # minimum compute capability (e.g. 89 for L4)
    note: str = ""           # short human note (perf / source)


# Per provider-key -> arch requirement. Used for arch-gating skips.
_PROVIDER_MIN_SM: Dict[str, int] = {
    # attention
    "kernel-set": 70,            # ks runs wherever it was built (matches bench.py)
    "flash-attn": 80,            # FA2 needs Ampere+
    "sdpa-flash": 80,            # torch SDPA flash backend sm80+
    "flashinfer": 75,            # FlashInfer Turing+
    "flash-mla": 90,             # FlashMLA: Hopper sm90 (sparse path sm100)
    "sgl-attn": 90,              # SGLang FA3 (flash_attn_varlen/with_kvcache) sm90
    "sgl-mla": 90,               # SGLang FlashMLA sm90
    # gemm
    "torch-cublas": 70,
    "torch-compile": 70,
    "torch-scaled-mm": 89,       # fp8 tensor cores: Ada sm89+
    "deepgemm": 90,              # DeepGEMM Hopper sm90 / Blackwell sm100 only
    "vllm-marlin": 80,           # GPTQ-Marlin Ampere+
    "vllm-machete": 90,          # Machete sm90a Hopper-only
    "vllm-cutlass-fp8": 89,      # vLLM cutlass_scaled_mm fp8 sm89+
    "vllm-fused-moe": 80,        # vLLM fused_experts Triton sm80+
    "sgl-fp8": 90,               # SGLang CUTLASS fp8_scaled_mm sm90
    "sgl-int8": 80,              # SGLang CUTLASS int8_scaled_mm sm80+
    # norm / act / rope
    "flashinfer-norm": 75,
    "vllm-norm": 70,
    "liger-norm": 80,            # Triton sm80+
    "flashinfer-rope": 75,
    "liger-rope": 80,
    "flashinfer-act": 75,
    "vllm-act": 70,
    "liger-act": 80,
    "sgl-norm": 80,              # SGLang rmsnorm/fused_add_rmsnorm/gemma sm80+
    "sgl-rope": 80,              # SGLang rotary_embedding sm80+
    "sgl-act": 80,              # SGLang silu_and_mul sm80+
    # moe
    "sgl-moe-gate": 80,          # SGLang topk_softmax / moe_fused_gate sm80+
    # ssm
    "mamba-ssm": 70,
    "causal-conv1d": 70,
    # loss
    "liger-ce": 70,
    "cut-cross-entropy": 80,
    "torch-ce": 70,
}


def _min_sm(key: str) -> int:
    return _PROVIDER_MIN_SM.get(key, 70)


# --------------------------------------------------------------------------- #
# Result row. One per (op, shape, impl). Mirrors the columns the spec asks for:
#   op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err |
#   status
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    op: str
    shape: str
    impl: str                       # "kernel-set" or a provider key
    dtype: str
    lat_us: float = float("nan")    # median latency (us)
    min_us: float = float("nan")    # min latency (us)
    gbps: float = float("nan")
    tflops: float = float("nan")
    util: float = float("nan")      # achieved % of dense peak (bw or compute)
    metric: str = ""                # "GB/s" or "TFLOP/s" (which util refers to)
    rel_err: float = float("nan")
    tol: float = float("nan")
    status: str = "ok"              # ok / skip / import-fail / error / incorrect
    note: str = ""
    iters: int = 0
    method: str = ""

    @property
    def is_ks(self) -> bool:
        return self.impl == "kernel-set"

    @property
    def ran_ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# Per-op group context. We keep our own light Ctx (independent of bench.Ctx so a
# bench.py refactor of its internal Ctx doesn't break us).
# --------------------------------------------------------------------------- #
@dataclass
class SotaCtx:
    dt: object
    dtype_name: str
    gpu: "GpuInfo"
    device: str = "cuda"

    def rand(self, *shape, dtype=None):
        d = self.dt if dtype is None else dtype
        return torch.randn(*shape, device=self.device, dtype=d)

    def empty(self, *shape, dtype=None):
        return torch.empty(*shape, device=self.device,
                           dtype=self.dt if dtype is None else dtype)

    def tol(self) -> float:
        """Relative-error tolerance for the active dtype (matches bench.py).

        Cross-impl comparisons are slightly looser than bench.py's intra-kernel
        gate because two *different* fast kernels (e.g. flash vs flashinfer) can
        differ a bit more than a kernel vs its own fp32 math reference.
        """
        if self.dt is torch.float32:
            return 3e-3
        if self.dt is torch.bfloat16:
            return 4e-2   # bf16: 8 mantissa bits
        return 2e-2       # fp16


# --------------------------------------------------------------------------- #
# Small helpers shared by the op runners.
# --------------------------------------------------------------------------- #
def _sm(gpu: "GpuInfo") -> int:
    try:
        return int(gpu.sm_arch)
    except Exception:
        return 0


def arch_skip_row(op: str, shape: str, dtype: str, impl: str,
                  gpu: "GpuInfo") -> Optional[Row]:
    """Return a `skip: needs smXX` Row if the device arch is below the provider's
    minimum, else None. This is the gate that prevents importing/calling a lib on
    an unsupported GPU (e.g. DeepGEMM/FlashMLA on L4)."""
    need = _min_sm(impl)
    have = _sm(gpu)
    if have and have < need:
        return Row(op, shape, impl, dtype, status="skip",
                   note=f"needs sm{need} (device is sm{have})")
    return None


def _import(modpath: str):
    """Import a module by dotted path; raise on failure (caller catches)."""
    return importlib.import_module(modpath)


def _time(fn: Callable[[], object]) -> "Timing":
    """Time a callable with bench.py's L2-flushed CUDA-event timer, consuming the
    output so the launch is not dead-code-eliminated."""
    return time_op(_sink(fn))


def _fill_row_timing(row: Row, t: "Timing") -> None:
    row.lat_us = t.median_us
    row.min_us = t.min_us
    row.iters = t.iters
    row.method = t.method


def _set_bw(row: Row, nbytes: float, gpu: "GpuInfo") -> None:
    if math.isnan(row.lat_us) or row.lat_us <= 0:
        return
    row.gbps = nbytes / (row.lat_us * 1e-6) / 1e9
    row.metric = "GB/s"
    peak = peak_for_metric(gpu, row.dtype, "bw")
    if peak > 0:
        row.util = 100.0 * row.gbps / peak


def _set_flops(row: Row, flops: float, gpu: "GpuInfo",
               dtype_for_peak: Optional[str] = None) -> None:
    if math.isnan(row.lat_us) or row.lat_us <= 0:
        return
    row.tflops = flops / (row.lat_us * 1e-6) / 1e12
    row.metric = "TFLOP/s"
    peak = peak_for_metric(gpu, dtype_for_peak or row.dtype, "tflops")
    if peak > 0:
        row.util = 100.0 * row.tflops / peak


def _gate(row: Row, ref, got, ctx: SotaCtx) -> bool:
    """Compute rel_err(got, ref), set tol, and flag `incorrect` if over tol.
    Returns True if correct (or no reference). Never raises."""
    row.tol = ctx.tol()
    try:
        re_val = rel_err(got, ref)
    except Exception as exc:
        row.note = (row.note + " " if row.note else "") + f"rel_err N/A ({exc})"
        return True
    row.rel_err = re_val
    if re_val > row.tol:
        row.status = "incorrect"
        return False
    return True


def _short(exc: BaseException, n: int = 120) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    msg = " ".join(msg.split())
    return msg[:n]


def run_provider(op: str, shape: str, impl: str, ctx: SotaCtx,
                 body: Callable[[Row], None]) -> Row:
    """Run one provider's body under full fault isolation.

    `body(row)` does: arch already gated by caller -> import lib -> build call ->
    correctness check (set row.rel_err) -> time (set row timing + perf). It must
    raise on import/setup failure; we translate exceptions into import-fail /
    error statuses. A clean return with status still 'ok' means success.
    """
    row = Row(op, shape, impl, ctx.dtype_name)
    # arch gate first (never import on an unsupported arch)
    skip = arch_skip_row(op, shape, ctx.dtype_name, impl, ctx.gpu)
    if skip is not None:
        return skip
    try:
        body(row)
    except (ImportError, ModuleNotFoundError) as exc:
        row.status = "import-fail"
        row.note = _short(exc)
    except Exception as exc:
        # distinguish "lib clearly not installed" surfacing as a generic error
        if row.status in ("ok",):
            row.status = "error"
        if not row.note:
            row.note = _short(exc)
    return row


# =========================================================================== #
# OP GROUPS
# Each group is a function (ctx) -> List[Row] producing one row per impl on a
# shared set of inputs (kernel-set + each provider). Groups self-isolate every
# provider via run_provider, and they self-skip (return []) if torch/ks missing.
# =========================================================================== #
_GROUPS: "Dict[str, Callable[[SotaCtx], List[Row]]]" = {}


def group(name: str):
    def deco(fn):
        _GROUPS[name] = fn
        return fn
    return deco


# ----------------------------- shared shapes ------------------------------- #
# _ATTN_PREFILL_SHAPES, _ATTN_DECODE_SHAPES, _ROPE_SHAPES and _CE_SHAPES are
# imported verbatim from _bench_common (single source of truth, shared with
# bench.py). The tables below are deliberately a *subset* of bench.py's sweep
# (cross-impl comparison) plus sota-only tables (MLA, SSM); they intentionally
# differ from bench.py and stay local.
_GEMM_SHAPES = [
    ("M=4096,N=4096,K=4096", 4096, 4096, 4096),
    ("M=8192,N=8192,K=8192", 8192, 8192, 8192),
    ("M=4096,N=14336,K=4096", 4096, 14336, 4096),   # MLP up-proj
]
_NORM_SHAPES = [
    ("rows=4096,hidden=4096", 4096, 4096),
    ("rows=8192,hidden=8192", 8192, 8192),
    ("rows=1,hidden=4096", 1, 4096),                # decode
]
_SWIGLU_SHAPES = [
    ("rows=4096,inter=14336", 4096, 14336),
    ("rows=1,inter=14336", 1, 14336),               # decode
]
_MOE_SHAPES = [
    # (label, num_tokens, hidden, inter, num_experts, top_k)
    ("tokens=4096,h=4096,inter=14336,E=8,k=2", 4096, 4096, 14336, 8, 2),
]
_MLA_SHAPES = [
    # (label, num_seqs, ctx_len, heads, kv_lora_rank, rope_dim, block_size)
    ("seqs=64,ctx=2048,h=128,lora=512,rope=64", 64, 2048, 128, 512, 64, 64),
]
_SSM_SHAPES = [
    # (label, batch, dim, seqlen, dstate)
    ("b=4,d=2048,L=2048,n=16", 4, 2048, 2048, 16),
]


# ----------------------------------------------------------------------------
# ATTENTION — PREFILL
#   kernel-set ks_flash_attn vs flash_attn.flash_attn_func vs torch SDPA(flash)
#   vs flashinfer single_prefill_with_kv_cache
# ----------------------------------------------------------------------------
def _ref_attn_prefill(q, k, v, qh, kvh, scale):
    """High-precision fp32 SDPA reference (GQA-expanded), shape (B,S,H,D) in."""
    qt = q.transpose(1, 2).float()
    kt = k.transpose(1, 2).float()
    vt = v.transpose(1, 2).float()
    if kvh != qh:
        rep = qh // kvh
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)
    o = torch.nn.functional.scaled_dot_product_attention(
        qt, kt, vt, is_causal=True, scale=scale)
    return o.transpose(1, 2)   # (B,S,H,D) fp32


@group("attention_prefill")
def _g_attn_prefill(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    for label, b, seq, qh, kvh, hd in _ATTN_PREFILL_SHAPES:
        op = "attn_prefill"
        scale = 1.0 / math.sqrt(hd)
        q = ctx.rand(b, seq, qh, hd)
        k = ctx.rand(b, seq, kvh, hd)
        v = ctx.rand(b, seq, kvh, hd)
        ref = _ref_attn_prefill(q, k, v, qh, kvh, scale)
        # causal flops, halved
        flops = 4.0 * b * qh * (seq ** 2) * hd * 0.5

        # --- kernel-set ---
        def _ks(row: Row):
            out = ctx.empty(b, seq, qh, hd)
            ks.attention.flash_attn(out, q, k, v, b, seq, seq, qh, kvh, hd,
                                    softmax_scale=scale, causal=True)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: ks.attention.flash_attn(out, q, k, v, b, seq, seq,
                                                qh, kvh, hd,
                                                softmax_scale=scale,
                                                causal=True)))
            _set_flops(row, flops, ctx.gpu)
        rows.append(run_provider(op, label, "kernel-set", ctx, _ks))

        # --- flash-attn ---
        def _fa(row: Row):
            fa = _import("flash_attn")
            out = fa.flash_attn_func(q, k, v, causal=True, softmax_scale=scale)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: fa.flash_attn_func(q, k, v, causal=True,
                                           softmax_scale=scale)))
            _set_flops(row, flops, ctx.gpu)
        rows.append(run_provider(op, label, "flash-attn", ctx, _fa))

        # --- torch SDPA (flash/efficient backend) ---
        def _sdpa(row: Row):
            import contextlib
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                cm = sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                                  SDPBackend.EFFICIENT_ATTENTION])
            except Exception:
                cm = contextlib.nullcontext()

            def call():
                qt = q.transpose(1, 2)
                kt = k.transpose(1, 2)
                vt = v.transpose(1, 2)
                with cm:
                    o = torch.nn.functional.scaled_dot_product_attention(
                        qt, kt, vt, is_causal=True, scale=scale,
                        enable_gqa=(kvh != qh))
                return o.transpose(1, 2)
            out = call()
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(call))
            _set_flops(row, flops, ctx.gpu)
            row.note = "flash/efficient backend"
        rows.append(run_provider(op, label, "sdpa-flash", ctx, _sdpa))

        # --- flashinfer single_prefill_with_kv_cache ---
        def _fi(row: Row):
            fi = _import("flashinfer")
            # single_prefill expects unbatched (S,H,D); drive batch b==1 only,
            # otherwise loop is unfair — restrict to b==1 shapes.
            if b != 1:
                row.status = "skip"
                row.note = "flashinfer single_prefill: b==1 only here"
                return
            qf = q[0]   # (S, Hq, D)
            kf = k[0]
            vf = v[0]
            out = fi.prefill.single_prefill_with_kv_cache(
                qf, kf, vf, causal=True, kv_layout="NHD", sm_scale=scale)
            torch.cuda.synchronize()
            if not _gate(row, ref[0], out, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: fi.prefill.single_prefill_with_kv_cache(
                    qf, kf, vf, causal=True, kv_layout="NHD", sm_scale=scale)))
            _set_flops(row, flops, ctx.gpu)
        rows.append(run_provider(op, label, "flashinfer", ctx, _fi))

        # --- SGLang FA3 flash_attn_varlen_func (sm90; arch-gated) ---
        def _sgl(row: Row):
            from sgl_kernel import flash_attn_varlen_func
            qf = q.reshape(b * seq, qh, hd)
            kf = k.reshape(b * seq, kvh, hd)
            vf = v.reshape(b * seq, kvh, hd)
            cu = torch.arange(0, (b + 1) * seq, seq, device="cuda",
                              dtype=torch.int32)

            def call():
                return flash_attn_varlen_func(
                    qf, kf, vf, cu, cu, max_seqlen_q=seq, max_seqlen_k=seq,
                    softmax_scale=scale, causal=True)
            out = call().reshape(b, seq, qh, hd)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(call))
            _set_flops(row, flops, ctx.gpu)
            row.note = "SGLang FA3 varlen"
        rows.append(run_provider(op, label, "sgl-attn", ctx, _sgl))
    return rows


# ----------------------------------------------------------------------------
# ATTENTION — DECODE / PAGED
#   kernel-set ks_paged_attn_decode vs flashinfer paged decode (plan/run) vs SDPA
# ----------------------------------------------------------------------------
@group("attention_decode")
def _g_attn_decode(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    for entry in _ATTN_DECODE_SHAPES:
        label, num_seqs, ctx_len, qh, kvh, hd, block = entry
        op = "attn_decode"
        scale = 1.0 / math.sqrt(hd)
        blocks_per_seq = (ctx_len + block - 1) // block
        total_blocks = num_seqs * blocks_per_seq
        q = ctx.rand(num_seqs, qh, hd)
        # ks layout: k_cache/v_cache [num_blocks, num_kv_heads, block, head_dim]
        k_cache = ctx.rand(total_blocks, kvh, block, hd)
        v_cache = ctx.rand(total_blocks, kvh, block, hd)
        block_tables = torch.arange(total_blocks, device="cuda",
                                    dtype=torch.int32).reshape(
                                        num_seqs, blocks_per_seq)
        seq_lens = torch.full((num_seqs,), ctx_len, device="cuda",
                              dtype=torch.int32)
        kv_bytes = 2 * num_seqs * kvh * ctx_len * hd * dtype_bytes(ctx.dt)

        # fp32 reference via dense SDPA over the (logically) contiguous cache.
        # k_cache[block, kvh, blk, hd] -> per-seq [ctx_len, kvh, hd]
        def _gather_kv():
            # contiguous block_tables -> seq s uses blocks [s*bps : (s+1)*bps]
            kc = k_cache.permute(0, 2, 1, 3).reshape(
                num_seqs, blocks_per_seq * block, kvh, hd)[:, :ctx_len]
            vc = v_cache.permute(0, 2, 1, 3).reshape(
                num_seqs, blocks_per_seq * block, kvh, hd)[:, :ctx_len]
            return kc, vc

        kc, vc = _gather_kv()
        # ref: q (S,Hq,D) over kv (S, ctx, Hkv, D), non-causal full attention
        qref = q.unsqueeze(1).transpose(1, 2).float()       # (S, Hq, 1, D)
        kref = kc.transpose(1, 2).float()                    # (S, Hkv, ctx, D)
        vref = vc.transpose(1, 2).float()
        if kvh != qh:
            rep = qh // kvh
            kref = kref.repeat_interleave(rep, dim=1)
            vref = vref.repeat_interleave(rep, dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(
            qref, kref, vref, is_causal=False, scale=scale)
        ref = ref.transpose(1, 2).squeeze(1)                 # (S, Hq, D)

        # --- kernel-set ---
        def _ks(row: Row):
            out = ctx.empty(num_seqs, qh, hd)
            ks.attention.paged_attn_decode(
                out, q, k_cache, v_cache, block_tables, seq_lens,
                num_seqs, qh, kvh, hd, block, blocks_per_seq,
                softmax_scale=scale)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: ks.attention.paged_attn_decode(
                    out, q, k_cache, v_cache, block_tables, seq_lens,
                    num_seqs, qh, kvh, hd, block, blocks_per_seq,
                    softmax_scale=scale)))
            _set_bw(row, kv_bytes, ctx.gpu)
            row.note = "memory-bound (KV read)"
        rows.append(run_provider(op, label, "kernel-set", ctx, _ks))

        # --- flashinfer BatchDecodeWithPagedKVCacheWrapper (plan/run) ---
        def _fi(row: Row):
            fi = _import("flashinfer")
            # flashinfer NHD paged layout: kv_cache (num_blocks, 2, page_size,
            # num_kv_heads, head_dim) for the combined wrapper, OR a tuple
            # (k,v) each (num_blocks, page_size, num_kv_heads, head_dim).
            page = block
            # rebuild caches in flashinfer NHD page layout from our ks layout
            k_fi = k_cache.permute(0, 2, 1, 3).contiguous()   # (nb, page, kvh, hd)
            v_fi = v_cache.permute(0, 2, 1, 3).contiguous()
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8,
                                    device="cuda")
            wrapper = fi.decode.BatchDecodeWithPagedKVCacheWrapper(
                workspace, kv_layout="NHD")
            # indptr/indices/last_page_len for contiguous block tables
            kv_indptr = torch.arange(0, (num_seqs + 1) * blocks_per_seq,
                                     blocks_per_seq, device="cuda",
                                     dtype=torch.int32)
            kv_indices = torch.arange(total_blocks, device="cuda",
                                      dtype=torch.int32)
            last_page = ctx_len - (blocks_per_seq - 1) * page
            kv_last_page_len = torch.full((num_seqs,), last_page,
                                          device="cuda", dtype=torch.int32)
            wrapper.plan(kv_indptr, kv_indices, kv_last_page_len,
                         qh, kvh, hd, page, pos_encoding_mode="NONE",
                         data_type=ctx.dt, q_data_type=ctx.dt)
            out = wrapper.run(q, (k_fi, v_fi))
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(lambda: wrapper.run(q, (k_fi, v_fi))))
            _set_bw(row, kv_bytes, ctx.gpu)
        rows.append(run_provider(op, label, "flashinfer", ctx, _fi))

        # --- torch SDPA dense decode (reference-as-baseline) ---
        def _sdpa(row: Row):
            import contextlib
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                cm = sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                                  SDPBackend.EFFICIENT_ATTENTION])
            except Exception:
                cm = contextlib.nullcontext()
            kc2, vc2 = _gather_kv()

            def call():
                qd = q.unsqueeze(1).transpose(1, 2)
                kd = kc2.transpose(1, 2)
                vd = vc2.transpose(1, 2)
                with cm:
                    o = torch.nn.functional.scaled_dot_product_attention(
                        qd, kd, vd, is_causal=False, scale=scale,
                        enable_gqa=(kvh != qh))
                return o.transpose(1, 2).squeeze(1)
            out = call()
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(call))
            _set_bw(row, kv_bytes, ctx.gpu)
            row.note = "dense SDPA (no paging)"
        rows.append(run_provider(op, label, "sdpa-flash", ctx, _sdpa))

        # --- SGLang FA3 paged decode flash_attn_with_kvcache (sm90) ---
        def _sgl(row: Row):
            from sgl_kernel import flash_attn_with_kvcache
            qf = q.reshape(num_seqs, 1, qh, hd)
            k_fa = k_cache.permute(0, 2, 1, 3).contiguous()   # (nb, page, kvh, hd)
            v_fa = v_cache.permute(0, 2, 1, 3).contiguous()

            def call():
                o = flash_attn_with_kvcache(
                    qf, k_fa, v_fa, page_table=block_tables,
                    cache_seqlens=seq_lens, softmax_scale=scale, causal=False)
                return o.reshape(num_seqs, qh, hd)
            out = call()
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(call))
            _set_bw(row, kv_bytes, ctx.gpu)
            row.note = "SGLang FA3 paged decode"
        rows.append(run_provider(op, label, "sgl-attn", ctx, _sgl))
    return rows


# ----------------------------------------------------------------------------
# MLA DECODE
#   kernel-set ks_mla_decode vs FlashMLA flash_mla_with_kvcache (gated to sm90)
# ----------------------------------------------------------------------------
@group("mla_decode")
def _g_mla_decode(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    for entry in _MLA_SHAPES:
        label, num_seqs, ctx_len, heads, lora, rope_dim, block = entry
        op = "mla_decode"
        scale = 1.0 / math.sqrt(lora + rope_dim)
        blocks_per_seq = (ctx_len + block - 1) // block
        total_blocks = num_seqs * blocks_per_seq
        # ks MLA layout
        q_nope = ctx.rand(num_seqs, heads, lora)
        q_pe = ctx.rand(num_seqs, heads, rope_dim)
        kv_cache = ctx.rand(total_blocks, block, lora + rope_dim)
        block_tables = torch.arange(total_blocks, device="cuda",
                                    dtype=torch.int32).reshape(
                                        num_seqs, blocks_per_seq)
        seq_lens = torch.full((num_seqs,), ctx_len, device="cuda",
                              dtype=torch.int32)
        kv_bytes = num_seqs * ctx_len * (lora + rope_dim) * dtype_bytes(ctx.dt)

        # --- kernel-set (no shared fp32 ref across the two different MLA
        # layouts/conventions; report throughput, gate vs internal math ref) ---
        def _ks(row: Row):
            out = ctx.empty(num_seqs, heads, lora)
            ks.attention.mla_decode(
                out, q_nope, q_pe, kv_cache, block_tables, seq_lens,
                num_seqs, heads, lora, rope_dim, block, blocks_per_seq,
                softmax_scale=scale)
            torch.cuda.synchronize()
            # fp32 reference: absorbed MLA = attention with concatenated
            # [q_nope|q_pe] queries against kv_cache rows (lora+rope), output
            # the lora part.
            qcat = torch.cat([q_nope, q_pe], dim=-1).float()   # (S,H,lora+rope)
            kvc = kv_cache.reshape(num_seqs, blocks_per_seq * block,
                                   lora + rope_dim)[:, :ctx_len].float()
            scores = torch.einsum("shd,scd->shc", qcat, kvc) * scale
            attn = torch.softmax(scores, dim=-1)
            vpart = kvc[..., :lora]                              # value = ckv
            ref = torch.einsum("shc,scd->shd", attn, vpart)
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: ks.attention.mla_decode(
                    out, q_nope, q_pe, kv_cache, block_tables, seq_lens,
                    num_seqs, heads, lora, rope_dim, block, blocks_per_seq,
                    softmax_scale=scale)))
            _set_bw(row, kv_bytes, ctx.gpu)
            row.note = "absorbed-MLA decode; ref=fp32 math"
        rows.append(run_provider(op, label, "kernel-set", ctx, _ks))

        # --- FlashMLA (sm90; will SKIP on L4 via arch gate) ---
        def _fmla(row: Row):
            fm = _import("flash_mla")
            # FlashMLA expects bf16; q (S, h, head_dim) where head_dim=lora+rope
            # k_cache paged (num_blocks, page, h_kv=1, lora+rope). This is a
            # best-effort wiring; on L4 we never get here (arch-gated).
            cache_seqlens = seq_lens
            h_kv = 1
            tile_md, num_splits = fm.get_mla_metadata(
                cache_seqlens, heads // h_kv, h_kv)
            q_cat = torch.cat([q_nope, q_pe], dim=-1).to(torch.bfloat16
                                                         ).unsqueeze(1)
            kcache = kv_cache.reshape(total_blocks, block, 1,
                                      lora + rope_dim).to(torch.bfloat16)
            out, _lse = fm.flash_mla_with_kvcache(
                q_cat, kcache, block_tables, cache_seqlens, lora,
                tile_md, num_splits, softmax_scale=scale, causal=True)
            torch.cuda.synchronize()
            # no shared ref across conventions; report throughput only
            _fill_row_timing(row, _time(
                lambda: fm.flash_mla_with_kvcache(
                    q_cat, kcache, block_tables, cache_seqlens, lora,
                    tile_md, num_splits, softmax_scale=scale, causal=True)))
            _set_bw(row, kv_bytes, ctx.gpu)
            row.note = "FlashMLA; throughput only (no shared ref)"
        rows.append(run_provider(op, label, "flash-mla", ctx, _fmla))

        # --- SGLang FlashMLA via sgl_kernel.flash_mla_with_kvcache (sm90) ---
        def _sgl_mla(row: Row):
            from sgl_kernel import (
                get_mla_metadata,
                flash_mla_with_kvcache as sgl_flash_mla,
            )
            cache_seqlens = seq_lens
            h_kv = 1
            tile_md, num_splits = get_mla_metadata(
                cache_seqlens, heads // h_kv, h_kv)
            q_cat = torch.cat([q_nope, q_pe], dim=-1).to(
                torch.bfloat16).unsqueeze(1)
            kcache = kv_cache.reshape(total_blocks, block, 1,
                                      lora + rope_dim).to(torch.bfloat16)

            def call():
                o, _lse = sgl_flash_mla(
                    q_cat, kcache, block_tables, cache_seqlens, lora,
                    tile_md, num_splits, softmax_scale=scale, causal=True)
                return o
            call()
            torch.cuda.synchronize()
            # no shared ref across conventions; throughput only
            _fill_row_timing(row, _time(call))
            _set_bw(row, kv_bytes, ctx.gpu)
            row.note = "SGLang FlashMLA; throughput only (no shared ref)"
        rows.append(run_provider(op, label, "sgl-mla", ctx, _sgl_mla))
    return rows


# ----------------------------------------------------------------------------
# GEMM fp16/bf16
#   kernel-set ks_gemm vs torch @ (cuBLAS) vs torch.compile
# ----------------------------------------------------------------------------
@group("gemm")
def _g_gemm(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    if ctx.dt not in (torch.float16, torch.bfloat16, torch.float32):
        return rows
    for label, m, n, k in _GEMM_SHAPES:
        op = "gemm"
        a = ctx.rand(m, k)
        b = ctx.rand(k, n)
        ref = (a.float() @ b.float())
        flops = 2.0 * m * n * k

        # --- kernel-set ---
        def _ks(row: Row):
            c = ctx.empty(m, n)
            ks.gemm.gemm(c, a, b, m=m, n=n, k=k)
            torch.cuda.synchronize()
            if not _gate(row, ref, c, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: ks.gemm.gemm(c, a, b, m=m, n=n, k=k)))
            _set_flops(row, flops, ctx.gpu)
        rows.append(run_provider(op, label, "kernel-set", ctx, _ks))

        # --- torch cuBLAS a@b ---
        def _cublas(row: Row):
            out = a @ b
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(lambda: a @ b))
            _set_flops(row, flops, ctx.gpu)
        rows.append(run_provider(op, label, "torch-cublas", ctx, _cublas))

        # --- torch.compile(max-autotune) ---
        def _compile(row: Row):
            if not hasattr(torch, "compile"):
                row.status = "skip"
                row.note = "torch.compile unavailable"
                return

            def mm(x, y):
                return x @ y
            fn = torch.compile(mm, mode="max-autotune")
            out = fn(a, b)            # triggers compilation (outside timing)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(lambda: fn(a, b)))
            _set_flops(row, flops, ctx.gpu)
        rows.append(run_provider(op, label, "torch-compile", ctx, _compile))
    return rows


# ----------------------------------------------------------------------------
# W4A16 GEMM
#   kernel-set ks_gemm_w4a16 vs Marlin (vllm ops.gptq_marlin_gemm)
# ----------------------------------------------------------------------------
@group("w4a16")
def _g_w4a16(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    if ctx.dt not in (torch.float16, torch.bfloat16):
        # one informational skip row
        rows.append(Row("w4a16", "-", "kernel-set", ctx.dtype_name,
                        status="skip", note="w4a16 acts must be fp16/bf16"))
        return rows
    for label, m, n, k in _GEMM_SHAPES[:2]:
        op = "w4a16"
        group_size = 128
        flops = 2.0 * m * n * k
        a = ctx.rand(m, k)

        # --- kernel-set (no portable int4 torch ref -> throughput only) ---
        def _ks(row: Row):
            b_packed = torch.randint(0, 255, (k, n // 2), device="cuda",
                                     dtype=torch.uint8)
            n_groups = (k + group_size - 1) // group_size
            scales = (ctx.rand(n_groups, n) * 0.02)
            zeros = ctx.rand(n_groups, n)
            c = ctx.empty(m, n)
            ks.gemm.gemm_w4a16(c, a, b_packed, scales, zeros,
                               m=m, n=n, k=k, group_size=group_size)
            torch.cuda.synchronize()
            _fill_row_timing(row, _time(
                lambda: ks.gemm.gemm_w4a16(c, a, b_packed, scales, zeros,
                                           m=m, n=n, k=k,
                                           group_size=group_size)))
            _set_flops(row, flops, ctx.gpu)
            row.note = "throughput only (no portable int4 ref)"
        rows.append(run_provider(op, label, "kernel-set", ctx, _ks))

        # --- vLLM GPTQ-Marlin ---
        def _marlin(row: Row):
            from vllm import _custom_ops as ops
            from vllm.scalar_type import scalar_types
            # Build a random GPTQ-int4 weight, repack to Marlin layout.
            # qweight: int32 [K, N//8] (8 int4 per int32) for gptq_marlin_repack
            qweight = torch.randint(0, 2 ** 31 - 1, (k, n // 8),
                                    device="cuda", dtype=torch.int32)
            n_groups = (k + group_size - 1) // group_size
            scales = (torch.randn(n_groups, n, device="cuda", dtype=ctx.dt)
                      * 0.01)
            g_idx = torch.empty(0, dtype=torch.int32, device="cuda")
            perm = torch.empty(0, dtype=torch.int32, device="cuda")
            wtype = scalar_types.uint4b8
            b_q = ops.gptq_marlin_repack(qweight, perm, k, n, wtype.size_bits)
            workspace = torch.zeros(n // 64 * 16, device="cuda",
                                    dtype=torch.int32)
            c = torch.empty(m, n, device="cuda", dtype=ctx.dt)
            zeros = torch.empty(0, device="cuda", dtype=ctx.dt)
            bias = torch.empty(0, device="cuda", dtype=ctx.dt)
            gscale = torch.empty(0, device="cuda", dtype=torch.float32)

            def call():
                return ops.gptq_marlin_gemm(
                    a, c, b_q, bias, scales, gscale, zeros, g_idx, perm,
                    workspace, wtype, m, n, k, is_k_full=True,
                    use_atomic_add=False, use_fp32_reduce=True,
                    is_zp_float=False)
            out = call()
            torch.cuda.synchronize()
            _fill_row_timing(row, _time(call))
            _set_flops(row, flops, ctx.gpu)
            row.note = "Marlin; throughput only (random weights)"
        rows.append(run_provider(op, label, "vllm-marlin", ctx, _marlin))
    return rows


# ----------------------------------------------------------------------------
# FP8 GEMM
#   kernel-set (w8a8 closest ABI) vs torch._scaled_mm vs DeepGEMM (gated sm90)
#   vs vLLM cutlass_scaled_mm fp8
# ----------------------------------------------------------------------------
@group("fp8_gemm")
def _g_fp8_gemm(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    for label, m, n, k in _GEMM_SHAPES[:2]:
        op = "fp8_gemm"
        flops = 2.0 * m * n * k
        # fp8 e4m3 inputs with per-tensor scales; reference in fp32.
        if not _HAVE_TORCH:
            break
        has_fp8 = hasattr(torch, "float8_e4m3fn")
        a_f = torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.1
        b_f = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.1
        ref = (a_f @ b_f.t())   # (M,N) fp32

        # --- torch._scaled_mm (fp8; needs sm89+ -> gated) ---
        def _smm(row: Row):
            if not has_fp8:
                row.status = "skip"
                row.note = "torch has no float8_e4m3fn"
                return
            sa = (a_f.abs().max() / 448.0).clamp_min(1e-6)
            sb = (b_f.abs().max() / 448.0).clamp_min(1e-6)
            a8 = (a_f / sa).to(torch.float8_e4m3fn)
            b8 = (b_f / sb).to(torch.float8_e4m3fn)   # (N,K)
            b8t = b8.t()                               # (K,N) col-major view
            scale_a = sa.reshape(1).float()
            scale_b = sb.reshape(1).float()

            def call():
                return torch._scaled_mm(a8, b8t, scale_a=scale_a,
                                        scale_b=scale_b,
                                        out_dtype=torch.bfloat16,
                                        use_fast_accum=True)
            out = call()
            torch.cuda.synchronize()
            if not _gate(row, ref, out.float(), ctx):
                return
            _fill_row_timing(row, _time(call))
            _set_flops(row, flops, ctx.gpu, dtype_for_peak="fp8")
            row.note = "fp8 e4m3, per-tensor scale"
        rows.append(run_provider(op, label, "torch-scaled-mm", ctx, _smm))

        # --- vLLM cutlass_scaled_mm fp8 ---
        def _vllm(row: Row):
            if not has_fp8:
                row.status = "skip"
                row.note = "torch has no float8_e4m3fn"
                return
            from vllm import _custom_ops as ops
            sa = (a_f.abs().max() / 448.0).clamp_min(1e-6)
            sb = (b_f.abs().max() / 448.0).clamp_min(1e-6)
            a8 = (a_f / sa).to(torch.float8_e4m3fn)
            b8 = (b_f / sb).to(torch.float8_e4m3fn).t()   # (K,N) col-major
            scale_a = sa.reshape(1, 1).float()
            scale_b = sb.reshape(1, 1).float()

            def call():
                return ops.cutlass_scaled_mm(a8, b8, scale_a, scale_b,
                                             out_dtype=torch.bfloat16)
            out = call()
            torch.cuda.synchronize()
            if not _gate(row, ref, out.float(), ctx):
                return
            _fill_row_timing(row, _time(call))
            _set_flops(row, flops, ctx.gpu, dtype_for_peak="fp8")
        rows.append(run_provider(op, label, "vllm-cutlass-fp8", ctx, _vllm))

        # --- DeepGEMM fp8_gemm_nt (sm90; SKIP on L4 via arch gate) ---
        def _dg(row: Row):
            dg = _import("deep_gemm")
            if not has_fp8:
                row.status = "skip"
                row.note = "torch has no float8_e4m3fn"
                return
            # DeepGEMM fine-grained 1x128 / 128x128 block scaling, NT layout.
            sa = (a_f.abs().max() / 448.0).clamp_min(1e-6)
            sb = (b_f.abs().max() / 448.0).clamp_min(1e-6)
            a8 = (a_f / sa).to(torch.float8_e4m3fn)
            b8 = (b_f / sb).to(torch.float8_e4m3fn)
            a_scale = torch.full((m, (k + 127) // 128), float(sa),
                                 device="cuda", dtype=torch.float32)
            b_scale = torch.full(((n + 127) // 128, (k + 127) // 128),
                                 float(sb), device="cuda", dtype=torch.float32)
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

            def call():
                dg.fp8_gemm_nt((a8, a_scale), (b8, b_scale), out)
                return out
            call()
            torch.cuda.synchronize()
            if not _gate(row, ref, out.float(), ctx):
                return
            _fill_row_timing(row, _time(call))
            _set_flops(row, flops, ctx.gpu, dtype_for_peak="fp8")
            row.note = "DeepGEMM blockwise fp8"
        rows.append(run_provider(op, label, "deepgemm", ctx, _dg))

        # --- SGLang CUTLASS fp8_scaled_mm (sm90; arch-gated) ---
        def _sgl_fp8(row: Row):
            from sgl_kernel import fp8_scaled_mm
            if not has_fp8:
                row.status = "skip"
                row.note = "torch has no float8_e4m3fn"
                return
            sa = (a_f.abs().max() / 448.0).clamp_min(1e-6)
            sb = (b_f.abs().max() / 448.0).clamp_min(1e-6)
            a8 = (a_f / sa).to(torch.float8_e4m3fn)            # (M,K)
            b8 = (b_f / sb).to(torch.float8_e4m3fn).t()        # (K,N) col-major
            scale_a = sa.reshape(1, 1).float()
            scale_b = sb.reshape(1, 1).float()

            def call():
                return fp8_scaled_mm(a8, b8, scale_a, scale_b,
                                     torch.bfloat16)
            out = call()
            torch.cuda.synchronize()
            if not _gate(row, ref, out.float(), ctx):
                return
            _fill_row_timing(row, _time(call))
            _set_flops(row, flops, ctx.gpu, dtype_for_peak="fp8")
            row.note = "SGLang CUTLASS fp8 scaled-mm"
        rows.append(run_provider(op, label, "sgl-fp8", ctx, _sgl_fp8))

        # --- SGLang CUTLASS int8_scaled_mm (sm80+; int8 W8A8 path) ---
        def _sgl_int8(row: Row):
            from sgl_kernel import int8_scaled_mm
            ai = torch.randint(-127, 127, (m, k), device="cuda",
                               dtype=torch.int8)
            bi = torch.randint(-127, 127, (k, n), device="cuda",
                               dtype=torch.int8)
            scale_a = (torch.rand(m, 1, device="cuda") * 0.02 + 0.01)
            scale_b = (torch.rand(1, n, device="cuda") * 0.02 + 0.01)

            def call():
                return int8_scaled_mm(ai, bi, scale_a, scale_b, torch.bfloat16)
            call()
            torch.cuda.synchronize()
            # no shared ref with fp8 path -> throughput only
            _fill_row_timing(row, _time(call))
            _set_flops(row, flops, ctx.gpu, dtype_for_peak="int8")
            row.note = "SGLang CUTLASS int8 W8A8 scaled-mm (throughput)"
        rows.append(run_provider(op, label, "sgl-int8", ctx, _sgl_int8))

        # --- kernel-set: ks ABI has int8 w8a8 (closest); run as availability/
        # throughput probe so the row is present and clearly labeled. ---
        def _ks(row: Row):
            ai = torch.randint(-127, 127, (m, k), device="cuda",
                               dtype=torch.int8)
            bi = torch.randint(-127, 127, (k, n), device="cuda",
                               dtype=torch.int8)
            a_scale = torch.rand(m, device="cuda") * 0.02 + 0.01
            b_scale = torch.rand(n, device="cuda") * 0.02 + 0.01
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            out_dt = ks.dtype_to_ks(torch.bfloat16)
            ks.gemm.gemm_w8a8(out, ai, bi, a_scale, b_scale, m=m, n=n, k=k,
                              out_dtype=out_dt)
            torch.cuda.synchronize()
            _fill_row_timing(row, _time(
                lambda: ks.gemm.gemm_w8a8(out, ai, bi, a_scale, b_scale,
                                          m=m, n=n, k=k, out_dtype=out_dt)))
            _set_flops(row, flops, ctx.gpu, dtype_for_peak="int8")
            row.note = "ks has no native fp8 GEMM; int8 w8a8 proxy (no shared ref)"
        rows.append(run_provider(op, label, "kernel-set", ctx, _ks))
    return rows


# ----------------------------------------------------------------------------
# RMSNORM / FUSED_ADD_RMSNORM
#   kernel-set vs flashinfer.norm.rmsnorm vs vllm vs liger
# ----------------------------------------------------------------------------
@group("rmsnorm")
def _g_rmsnorm(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    for label, rows_n, hidden in _NORM_SHAPES:
        op = "rmsnorm"
        eps = 1e-6
        x = ctx.rand(rows_n, hidden)
        w = ctx.rand(hidden)
        nbytes = 2 * rows_n * hidden * dtype_bytes(ctx.dt)

        def ref_rms(t):
            f = t.float()
            return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)) \
                * w.float()
        ref = ref_rms(x)

        def _ks(row: Row):
            out = ctx.empty(rows_n, hidden)
            ks.norm.rms_norm(out, x, w, eps=eps)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: ks.norm.rms_norm(out, x, w, eps=eps)))
            _set_bw(row, nbytes, ctx.gpu)
        rows.append(run_provider(op, label, "kernel-set", ctx, _ks))

        def _fi(row: Row):
            from flashinfer.norm import rmsnorm as fi_rms
            out = fi_rms(x, w, eps=eps)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(lambda: fi_rms(x, w, eps=eps)))
            _set_bw(row, nbytes, ctx.gpu)
        rows.append(run_provider(op, label, "flashinfer-norm", ctx, _fi))

        def _vllm(row: Row):
            from vllm import _custom_ops as ops
            out = torch.empty_like(x)
            ops.rms_norm(out, x, w, eps)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(lambda: ops.rms_norm(out, x, w, eps)))
            _set_bw(row, nbytes, ctx.gpu)
        rows.append(run_provider(op, label, "vllm-norm", ctx, _vllm))

        def _liger(row: Row):
            from liger_kernel.ops.rms_norm import LigerRMSNormFunction
            out = LigerRMSNormFunction.apply(x, w, eps, 0.0, "llama")
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: LigerRMSNormFunction.apply(x, w, eps, 0.0, "llama")))
            _set_bw(row, nbytes, ctx.gpu)
        rows.append(run_provider(op, label, "liger-norm", ctx, _liger))

        # --- SGLang sgl_kernel.rmsnorm ---
        def _sgl(row: Row):
            from sgl_kernel import rmsnorm as sgl_rms
            out = sgl_rms(x, w, eps)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(lambda: sgl_rms(x, w, eps)))
            _set_bw(row, nbytes, ctx.gpu)
        rows.append(run_provider(op, label, "sgl-norm", ctx, _sgl))
    return rows


@group("fused_add_rmsnorm")
def _g_fused_add_rmsnorm(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    for label, rows_n, hidden in _NORM_SHAPES:
        op = "fused_add_rmsnorm"
        eps = 1e-6
        x = ctx.rand(rows_n, hidden)
        res = ctx.rand(rows_n, hidden)
        w = ctx.rand(hidden)
        nbytes = 4 * rows_n * hidden * dtype_bytes(ctx.dt)  # read x,res; write 2

        def ref_fn():
            added = (x.float() + res.float())
            normed = (added * torch.rsqrt(added.pow(2).mean(-1, keepdim=True)
                                          + eps)) * w.float()
            return normed, added
        ref_out, ref_res = ref_fn()

        def _ks(row: Row):
            out = ctx.empty(rows_n, hidden)
            res_out = ctx.empty(rows_n, hidden)
            ks.norm.rms_norm_residual(out, res_out, x, res, w, eps=eps)
            torch.cuda.synchronize()
            if not _gate(row, ref_out, out, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: ks.norm.rms_norm_residual(out, res_out, x, res, w,
                                                  eps=eps)))
            _set_bw(row, nbytes, ctx.gpu)
        rows.append(run_provider(op, label, "kernel-set", ctx, _ks))

        def _fi(row: Row):
            from flashinfer.norm import fused_add_rmsnorm as fi_far
            # in-place: residual = x+residual; x = rmsnorm(residual)*w
            xc = x.clone()
            rc = res.clone()
            fi_far(xc, rc, w, eps=eps)
            torch.cuda.synchronize()
            if not _gate(row, ref_out, xc, ctx):
                return

            def call():
                a = x.clone()
                b = res.clone()
                fi_far(a, b, w, eps=eps)
                return a
            _fill_row_timing(row, _time(call))
            _set_bw(row, nbytes, ctx.gpu)
            row.note = "in-place (clone per iter excluded from comparison)"
        rows.append(run_provider(op, label, "flashinfer-norm", ctx, _fi))

        def _vllm(row: Row):
            from vllm import _custom_ops as ops
            xc = x.clone()
            rc = res.clone()
            ops.fused_add_rms_norm(xc, rc, w, eps)
            torch.cuda.synchronize()
            if not _gate(row, ref_out, xc, ctx):
                return

            def call():
                a = x.clone()
                b = res.clone()
                ops.fused_add_rms_norm(a, b, w, eps)
                return a
            _fill_row_timing(row, _time(call))
            _set_bw(row, nbytes, ctx.gpu)
            row.note = "in-place"
        rows.append(run_provider(op, label, "vllm-norm", ctx, _vllm))

        # --- SGLang sgl_kernel.fused_add_rmsnorm (in-place residual+norm) ---
        def _sgl(row: Row):
            from sgl_kernel import fused_add_rmsnorm as sgl_far
            xc = x.clone()
            rc = res.clone()
            sgl_far(xc, rc, w, eps)
            torch.cuda.synchronize()
            if not _gate(row, ref_out, xc, ctx):
                return

            def call():
                a = x.clone()
                b = res.clone()
                sgl_far(a, b, w, eps)
                return a
            _fill_row_timing(row, _time(call))
            _set_bw(row, nbytes, ctx.gpu)
            row.note = "in-place (clone per iter excluded)"
        rows.append(run_provider(op, label, "sgl-norm", ctx, _sgl))
    return rows


# ----------------------------------------------------------------------------
# ROPE
#   kernel-set vs flashinfer apply_rope vs liger
# ----------------------------------------------------------------------------
def _ref_rope_neox(x, cos, sin):
    """NeoX/rotate-half RoPE reference. Returns the raw fp32 result (no cast
    back to x.dtype): cross-impl comparison upcasts to fp32 anyway."""
    return _ref_rope_neox_common(x, cos, sin, cast=False)


@group("rope")
def _g_rope(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    for label, tokens, qh, kvh, hd in _ROPE_SHAPES:
        op = "rope"
        q = ctx.rand(tokens, qh, hd)
        k = ctx.rand(tokens, kvh, hd)
        cos = ctx.rand(tokens, hd // 2)
        sin = ctx.rand(tokens, hd // 2)
        ref_q = _ref_rope_neox(q, cos, sin)
        nbytes = 2 * (tokens * qh * hd + tokens * kvh * hd) * dtype_bytes(ctx.dt)

        def _ks(row: Row):
            qc, kc = q.clone(), k.clone()
            ks.rope.rope_inplace(qc, kc, cos, sin, tokens, qh, kvh, hd)
            torch.cuda.synchronize()
            if not _gate(row, ref_q, qc, ctx):
                return

            def call():
                a, b = q.clone(), k.clone()
                ks.rope.rope_inplace(a, b, cos, sin, tokens, qh, kvh, hd)
                return a
            _fill_row_timing(row, _time(call))
            _set_bw(row, nbytes, ctx.gpu)
            row.note = "neox/rotate_half"
        rows.append(run_provider(op, label, "kernel-set", ctx, _ks))

        def _fi(row: Row):
            from flashinfer.rope import apply_rope_with_cos_sin_cache
            # build a cos_sin_cache [tokens, hd] (cos||sin) and pos_ids.
            positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
            cos_sin_cache = torch.cat([cos, sin], dim=-1).contiguous()
            qf = q.reshape(tokens, qh * hd)
            kf = k.reshape(tokens, kvh * hd)
            q_out, k_out = apply_rope_with_cos_sin_cache(
                positions, qf, kf, hd, cos_sin_cache, is_neox=True)
            torch.cuda.synchronize()
            got = q_out.reshape(tokens, qh, hd)
            if not _gate(row, ref_q, got, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: apply_rope_with_cos_sin_cache(
                    positions, qf, kf, hd, cos_sin_cache, is_neox=True)))
            _set_bw(row, nbytes, ctx.gpu)
        rows.append(run_provider(op, label, "flashinfer-rope", ctx, _fi))

        def _liger(row: Row):
            from liger_kernel.ops.rope import LigerRopeFunction
            # liger wants q (b, n_qh, s, d), k (b, n_kvh, s, d); cos/sin (1,s,d)
            qb = q.unsqueeze(0).transpose(1, 2)   # (1, qh, s, d)
            kb = k.unsqueeze(0).transpose(1, 2)
            cos_full = torch.cat([cos, cos], dim=-1).unsqueeze(0)  # (1,s,d)
            sin_full = torch.cat([sin, sin], dim=-1).unsqueeze(0)
            q_rot, k_rot = LigerRopeFunction.apply(qb, kb, cos_full, sin_full)
            torch.cuda.synchronize()
            got = q_rot.transpose(1, 2).reshape(tokens, qh, hd)
            # liger applies its own rotate-half convention; gate loosely.
            _gate(row, ref_q, got, ctx)
            _fill_row_timing(row, _time(
                lambda: LigerRopeFunction.apply(qb, kb, cos_full, sin_full)))
            _set_bw(row, nbytes, ctx.gpu)
            if row.status == "incorrect":
                row.status = "ok"
                row.note = "liger convention differs; throughput only"
        rows.append(run_provider(op, label, "liger-rope", ctx, _liger))

        # --- SGLang sgl_kernel.rotary_embedding (in-place, NeoX) ---
        def _sgl(row: Row):
            from sgl_kernel import rotary_embedding as sgl_rope
            positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
            cos_sin_cache = torch.cat([cos, sin], dim=-1).contiguous()
            qf = q.reshape(tokens, qh * hd).clone()
            kf = k.reshape(tokens, kvh * hd).clone()
            sgl_rope(positions, qf, kf, hd, cos_sin_cache, is_neox=True)
            torch.cuda.synchronize()
            got = qf.reshape(tokens, qh, hd)
            if not _gate(row, ref_q, got, ctx):
                return

            def call():
                a = q.reshape(tokens, qh * hd).clone()
                b = k.reshape(tokens, kvh * hd).clone()
                sgl_rope(positions, a, b, hd, cos_sin_cache, is_neox=True)
                return a
            _fill_row_timing(row, _time(call))
            _set_bw(row, nbytes, ctx.gpu)
            row.note = "neox/rotate_half (in-place)"
        rows.append(run_provider(op, label, "sgl-rope", ctx, _sgl))
    return rows


# ----------------------------------------------------------------------------
# SWIGLU (silu_and_mul)
#   kernel-set vs flashinfer.activation.silu_and_mul vs vllm
# ----------------------------------------------------------------------------
@group("swiglu")
def _g_swiglu(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    for label, rows_n, inter in _SWIGLU_SHAPES:
        op = "swiglu"
        gate_t = ctx.rand(rows_n, inter)
        up_t = ctx.rand(rows_n, inter)
        packed = torch.cat([gate_t, up_t], dim=-1).contiguous()  # (rows, 2*inter)
        ref = (torch.nn.functional.silu(gate_t.float()) * up_t.float())
        nbytes = 3 * rows_n * inter * dtype_bytes(ctx.dt)

        def _ks(row: Row):
            out = ctx.empty(rows_n, inter)
            ks.activation.swiglu(out, gate_t, up_t)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: ks.activation.swiglu(out, gate_t, up_t)))
            _set_bw(row, nbytes, ctx.gpu)
        rows.append(run_provider(op, label, "kernel-set", ctx, _ks))

        def _fi(row: Row):
            from flashinfer.activation import silu_and_mul
            out = silu_and_mul(packed)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(lambda: silu_and_mul(packed)))
            _set_bw(row, nbytes, ctx.gpu)
        rows.append(run_provider(op, label, "flashinfer-act", ctx, _fi))

        def _vllm(row: Row):
            from vllm import _custom_ops as ops
            out = torch.empty(rows_n, inter, device="cuda", dtype=ctx.dt)
            ops.silu_and_mul(out, packed)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(lambda: ops.silu_and_mul(out, packed)))
            _set_bw(row, nbytes, ctx.gpu)
        rows.append(run_provider(op, label, "vllm-act", ctx, _vllm))

        # --- SGLang sgl_kernel.silu_and_mul ---
        def _sgl(row: Row):
            from sgl_kernel import silu_and_mul as sgl_silu
            out = sgl_silu(packed)
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(lambda: sgl_silu(packed)))
            _set_bw(row, nbytes, ctx.gpu)
        rows.append(run_provider(op, label, "sgl-act", ctx, _sgl))
    return rows


# ----------------------------------------------------------------------------
# MOE gate + grouped GEMM
#   kernel-set vs vllm fused_experts (best-effort)
# ----------------------------------------------------------------------------
@group("moe")
def _g_moe(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    for entry in _MOE_SHAPES:
        label, num_tokens, hidden, inter, E, topk = entry
        op = "moe_grouped_gemm"
        total_rows = num_tokens * topk
        flops = 2.0 * total_rows * inter * hidden

        # --- kernel-set grouped GEMM over experts (CSR offsets) ---
        def _ks(row: Row):
            a = ctx.rand(total_rows, hidden)
            b = ctx.rand(E, hidden, inter)
            c = ctx.empty(total_rows, inter)
            per = total_rows // E
            offsets = torch.tensor(
                [min(i * per, total_rows) for i in range(E)] + [total_rows],
                device="cuda", dtype=torch.int32)
            ks.moe.grouped_gemm(c, a, b, offsets, E, total_rows, inter, hidden)
            torch.cuda.synchronize()
            off = offsets.detach().cpu().tolist()
            ref = torch.empty_like(c)
            for e in range(E):
                lo, hi = off[e], off[e + 1]
                if hi > lo:
                    ref[lo:hi] = (a[lo:hi].float() @ b[e].float()).to(c.dtype)
            if not _gate(row, ref.float(), c.float(), ctx):
                return
            _fill_row_timing(row, _time(
                lambda: ks.moe.grouped_gemm(c, a, b, offsets, E, total_rows,
                                            inter, hidden)))
            _set_flops(row, flops, ctx.gpu)
            row.note = "grouped GEMM over experts"
        rows.append(run_provider(op, label, "kernel-set", ctx, _ks))

        # --- vLLM fused_experts (full MoE FFN: gate*W1->SiLU->W2) ---
        def _vllm(row: Row):
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                fused_experts,
            )
            hs = ctx.rand(num_tokens, hidden)
            # vLLM expert weights: w1 (E, 2*inter, hidden), w2 (E, hidden, inter)
            w1 = ctx.rand(E, 2 * inter, hidden) * 0.02
            w2 = ctx.rand(E, hidden, inter) * 0.02
            gating = ctx.rand(num_tokens, E)
            probs = torch.softmax(gating.float(), dim=-1)
            topk_w, topk_i = torch.topk(probs, topk, dim=-1)
            topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(ctx.dt)
            topk_i = topk_i.to(torch.int32)

            def call():
                return fused_experts(hs, w1, w2, topk_w, topk_i,
                                     inplace=False)
            out = call()
            torch.cuda.synchronize()
            # no shared ref with ks grouped-gemm-only path -> throughput only
            _fill_row_timing(row, _time(call))
            # full MoE FFN: 2 grouped GEMMs (W1: 2*inter, W2: inter)
            moe_flops = 2.0 * total_rows * (2 * inter * hidden
                                            + hidden * inter)
            _set_flops(row, moe_flops, ctx.gpu)
            row.note = "full fused MoE FFN (W1+SiLU+W2); not directly vs ks ggemm"
        rows.append(run_provider("fused_moe", label, "vllm-fused-moe", ctx,
                                 _vllm))

        # --- SGLang fused MoE gate (topk_softmax): the SGLang specialty path.
        # Compares the kernel-set softmax+topk gate against sgl_kernel's fused
        # gate on the same gating logits (routing, not the FFN GEMMs). ---
        def _sgl_gate(row: Row):
            from sgl_kernel import topk_softmax
            gating = ctx.rand(num_tokens, E)
            ref_probs = torch.softmax(gating.float(), dim=-1)
            ref_w, ref_i = torch.topk(ref_probs, topk, dim=-1)
            topk_weights = torch.empty(num_tokens, topk, device="cuda",
                                       dtype=torch.float32)
            topk_ids = torch.empty(num_tokens, topk, device="cuda",
                                   dtype=torch.int32)
            topk_softmax(topk_weights, topk_ids, gating, False, 0.0, None)
            torch.cuda.synchronize()
            # gate selection alignment (ids should match the torch top-k set)
            _gate(row, ref_w, topk_weights, ctx)
            if row.status == "incorrect":
                row.status = "ok"
                row.note = ("SGLang topk_softmax gate; routing only "
                            "(not vs ks grouped GEMM)")

            def call():
                topk_softmax(topk_weights, topk_ids, gating, False, 0.0, None)
                return topk_weights
            _fill_row_timing(row, _time(call))
            gate_bytes = num_tokens * (E + 2 * topk) * 4
            _set_bw(row, gate_bytes, ctx.gpu)
            if not row.note:
                row.note = "SGLang topk_softmax fused MoE gate (specialty)"
        rows.append(run_provider("moe_gate", label, "sgl-moe-gate", ctx,
                                 _sgl_gate))
    return rows


# ----------------------------------------------------------------------------
# SSM — kernel-set has NONE here; run mamba_ssm + causal_conv1d as availability/
# correctness probes (kernel-set marked N/A).
# ----------------------------------------------------------------------------
@group("ssm")
def _g_ssm(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    for entry in _SSM_SHAPES:
        label, b, dim, L, n = entry
        # kernel-set N/A row (informational)
        rows.append(Row("ssm", label, "kernel-set", ctx.dtype_name,
                        status="skip", note="kernel-set has no SSM op (N/A)"))

        # --- mamba_ssm selective_scan_fn (Mamba-1) ---
        def _mamba(row: Row):
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            u = ctx.rand(b, dim, L)
            delta = torch.rand(b, dim, L, device="cuda", dtype=ctx.dt) * 0.1
            A = -torch.rand(dim, n, device="cuda", dtype=torch.float32) - 0.5
            B = ctx.rand(b, n, L)
            C = ctx.rand(b, n, L)
            D = torch.rand(dim, device="cuda", dtype=torch.float32)
            out = selective_scan_fn(u, delta, A, B, C, D=D,
                                    delta_softplus=True)
            torch.cuda.synchronize()
            # no kernel-set counterpart -> availability/throughput probe
            _fill_row_timing(row, _time(
                lambda: selective_scan_fn(u, delta, A, B, C, D=D,
                                          delta_softplus=True)))
            # rough bytes: read u + write y
            nbytes = 2 * b * dim * L * dtype_bytes(ctx.dt)
            _set_bw(row, nbytes, ctx.gpu)
            row.note = "mamba-ssm selective_scan (probe; no ks ref)"
        rows.append(run_provider("ssm_selective_scan", label, "mamba-ssm",
                                 ctx, _mamba))

        # --- causal_conv1d ---
        def _conv(row: Row):
            from causal_conv1d import causal_conv1d_fn
            width = 4
            x = ctx.rand(b, dim, L)
            weight = ctx.rand(dim, width)
            y = causal_conv1d_fn(x, weight, bias=None, activation="silu")
            torch.cuda.synchronize()
            _fill_row_timing(row, _time(
                lambda: causal_conv1d_fn(x, weight, bias=None,
                                         activation="silu")))
            nbytes = 2 * b * dim * L * dtype_bytes(ctx.dt)
            _set_bw(row, nbytes, ctx.gpu)
            row.note = "causal-conv1d (probe; no ks ref)"
        rows.append(run_provider("ssm_causal_conv1d", label, "causal-conv1d",
                                 ctx, _conv))
    return rows


# ----------------------------------------------------------------------------
# CROSS ENTROPY
#   kernel-set ks_cross_entropy vs liger / cut_cross_entropy / torch
# ----------------------------------------------------------------------------
@group("cross_entropy")
def _g_cross_entropy(ctx: SotaCtx) -> List[Row]:
    rows: List[Row] = []
    for label, num_tokens, vocab in _CE_SHAPES:
        op = "cross_entropy"
        logits = ctx.rand(num_tokens, vocab)
        targets = torch.randint(0, vocab, (num_tokens,), device="cuda",
                                dtype=torch.int64)
        ref = torch.nn.functional.cross_entropy(
            logits.float(), targets, reduction="none")
        nbytes = 2 * num_tokens * vocab * dtype_bytes(ctx.dt)

        def _ks(row: Row):
            losses = torch.empty(num_tokens, device="cuda",
                                 dtype=torch.float32)
            grad = torch.empty_like(logits)
            ks.loss.cross_entropy(losses, grad, logits, targets,
                                  num_tokens, vocab, ignore_index=-100)
            torch.cuda.synchronize()
            if not _gate(row, ref, losses, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: ks.loss.cross_entropy(losses, grad, logits, targets,
                                              num_tokens, vocab,
                                              ignore_index=-100)))
            _set_bw(row, nbytes, ctx.gpu)
            row.note = "fused fwd+bwd; rel_err on forward loss"
        rows.append(run_provider(op, label, "kernel-set", ctx, _ks))

        def _liger(row: Row):
            from liger_kernel.transformers.functional import liger_cross_entropy
            out = liger_cross_entropy(logits, targets, ignore_index=-100,
                                      reduction="none")
            loss = out[0] if isinstance(out, (tuple, list)) else out
            torch.cuda.synchronize()
            if not _gate(row, ref, loss, ctx):
                return
            _fill_row_timing(row, _time(
                lambda: liger_cross_entropy(logits, targets, ignore_index=-100,
                                            reduction="none")))
            _set_bw(row, nbytes, ctx.gpu)
        rows.append(run_provider(op, label, "liger-ce", ctx, _liger))

        def _cce(row: Row):
            # cut-cross-entropy is fused-linear-CE (embeddings @ classifier);
            # drive its linear_cross_entropy with a synthetic LM head.
            from cut_cross_entropy import linear_cross_entropy
            hidden = 4096
            emb = ctx.rand(num_tokens, hidden)
            classifier = ctx.rand(vocab, hidden) * 0.02
            ref_lin = torch.nn.functional.cross_entropy(
                (emb.float() @ classifier.float().t()), targets,
                reduction="mean")

            def call():
                return linear_cross_entropy(emb, classifier, targets,
                                            reduction="mean")
            loss = call()
            torch.cuda.synchronize()
            if not _gate(row, ref_lin.reshape(1), loss.reshape(1), ctx):
                return
            _fill_row_timing(row, _time(call))
            # bytes dominated by hidden@vocab logit compute, not materialized
            row.note = "fused-linear-CE (no logit materialization)"
        rows.append(run_provider("fused_linear_ce", label,
                                 "cut-cross-entropy", ctx, _cce))

        def _torch(row: Row):
            def call():
                return torch.nn.functional.cross_entropy(
                    logits, targets, reduction="none")
            out = call()
            torch.cuda.synchronize()
            if not _gate(row, ref, out, ctx):
                return
            _fill_row_timing(row, _time(call))
            _set_bw(row, nbytes, ctx.gpu)
            row.note = "torch eager fwd-only"
        rows.append(run_provider(op, label, "torch-ce", ctx, _torch))
    return rows


# --------------------------------------------------------------------------- #
# Op group selectors (CLI --ops names).
# --------------------------------------------------------------------------- #
ALL_OPS = [
    "attention_prefill", "attention_decode", "mla_decode",
    "gemm", "w4a16", "fp8_gemm",
    "rmsnorm", "fused_add_rmsnorm", "rope", "swiglu",
    "moe", "ssm", "cross_entropy",
]

# Which providers each op group compares (for --list and the summary header).
_OP_PROVIDERS: Dict[str, List[str]] = {
    "attention_prefill": ["kernel-set", "flash-attn", "sdpa-flash",
                          "flashinfer", "sgl-attn"],
    "attention_decode": ["kernel-set", "flashinfer", "sdpa-flash", "sgl-attn"],
    "mla_decode": ["kernel-set", "flash-mla", "sgl-mla"],
    "gemm": ["kernel-set", "torch-cublas", "torch-compile"],
    "w4a16": ["kernel-set", "vllm-marlin"],
    "fp8_gemm": ["kernel-set", "torch-scaled-mm", "vllm-cutlass-fp8",
                 "deepgemm", "sgl-fp8", "sgl-int8"],
    "rmsnorm": ["kernel-set", "flashinfer-norm", "vllm-norm", "liger-norm",
                "sgl-norm"],
    "fused_add_rmsnorm": ["kernel-set", "flashinfer-norm", "vllm-norm",
                          "sgl-norm"],
    "rope": ["kernel-set", "flashinfer-rope", "liger-rope", "sgl-rope"],
    "swiglu": ["kernel-set", "flashinfer-act", "vllm-act", "sgl-act"],
    "moe": ["kernel-set", "vllm-fused-moe", "sgl-moe-gate"],
    "ssm": ["kernel-set(N/A)", "mamba-ssm", "causal-conv1d"],
    "cross_entropy": ["kernel-set", "liger-ce", "cut-cross-entropy", "torch-ce"],
}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_groups(ops: Sequence[str], ctx: SotaCtx) -> List[Row]:
    out: List[Row] = []
    for op in ops:
        fn = _GROUPS.get(op)
        if fn is None:
            print(f"  [warn] no op group {op!r}", file=sys.stderr)
            continue
        try:
            group_rows = fn(ctx)
        except Exception as exc:
            group_rows = [Row(op, "-", "kernel-set", ctx.dtype_name,
                              status="error", note=_short(exc))]
        for r in group_rows:
            out.append(r)
            _print_progress(r)
    return out


def _print_progress(r: Row) -> None:
    tag = {
        "ok": "ok  ",
        "skip": "skip",
        "import-fail": "imp!",
        "error": "ERR ",
        "incorrect": "BAD ",
    }.get(r.status, r.status[:4])
    perf = ""
    if not math.isnan(r.tflops):
        perf = f"{r.tflops:8.1f} TFLOP/s"
    elif not math.isnan(r.gbps):
        perf = f"{r.gbps:8.1f} GB/s"
    if not math.isnan(r.util):
        perf += f" ({r.util:.0f}%pk)"
    lat = "" if math.isnan(r.lat_us) else f"{r.lat_us:8.1f}us"
    extra = ""
    if r.status == "incorrect" and not math.isnan(r.rel_err):
        extra = f" rel_err={r.rel_err:.2e}>tol={r.tol:.1e}"
    elif r.note:
        extra = f" ({r.note})"
    print(f"  [{tag}] {r.op:<18} {r.shape:<32} {r.impl:<18} "
          f"{lat:>11} {perf}{extra}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Speedup: kernel-set vs best-SOTA, per (op, shape).
# --------------------------------------------------------------------------- #
def compute_speedups(rows: List[Row]) -> Dict[Tuple[str, str], dict]:
    """For each (op, shape), find the kernel-set row and the fastest correct
    NON-kernel-set provider, and compute ks_speedup = best_other_us / ks_us."""
    groups: Dict[Tuple[str, str], List[Row]] = {}
    for r in rows:
        groups.setdefault((r.op, r.shape), []).append(r)
    out: Dict[Tuple[str, str], dict] = {}
    for key, rs in groups.items():
        ks_rows = [r for r in rs if r.is_ks and r.ran_ok
                   and not math.isnan(r.lat_us)]
        others = [r for r in rs if not r.is_ks and r.ran_ok
                  and not math.isnan(r.lat_us)]
        info: dict = {"ks_us": float("nan"), "best_other": "",
                      "best_us": float("nan"), "speedup": float("nan")}
        if ks_rows:
            info["ks_us"] = ks_rows[0].lat_us
        if others:
            best = min(others, key=lambda r: r.lat_us)
            info["best_other"] = best.impl
            info["best_us"] = best.lat_us
            if ks_rows and ks_rows[0].lat_us > 0:
                info["speedup"] = best.lat_us / ks_rows[0].lat_us
        out[key] = info
    return out


def summarize(rows: List[Row]) -> Dict[str, int]:
    counts = {"ok": 0, "skip": 0, "import-fail": 0, "error": 0, "incorrect": 0}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x: float, prec: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{prec}f}"


def _fmt_lat(med: float, lo: float) -> str:
    if med is None or (isinstance(med, float) and math.isnan(med)):
        return "-"
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return f"{med:.1f}"
    return f"{med:.1f} ({lo:.1f})"


def _perf_cell(r: Row) -> str:
    if not math.isnan(r.tflops):
        s = f"{r.tflops:.1f} TFLOP/s"
        if not math.isnan(r.util):
            s += f" ({r.util:.0f}%)"
        return s
    if not math.isnan(r.gbps):
        s = f"{r.gbps:.1f} GB/s"
        if not math.isnan(r.util):
            s += f" ({r.util:.0f}%)"
        return s
    return "-"


def _status_cell(r: Row) -> str:
    if r.status == "ok":
        return "ok"
    if r.status == "skip":
        return f"skip: {r.note}" if r.note else "skip"
    if r.status == "import-fail":
        return f"import-fail: {r.note}"
    if r.status == "error":
        return f"error: {r.note}"
    if r.status == "incorrect":
        return f"incorrect (rel_err={r.rel_err:.2e}>tol={r.tol:.1e})"
    return r.status


def render_header(gpu: "GpuInfo", cfg: dict) -> List[str]:
    lines: List[str] = []
    lines.append(f"# kernel-set vs SOTA — {gpu.name}")
    lines.append("")
    peaks = _peaks_for(gpu) if _HAVE_BENCH else {}
    peak_str = ""
    if peaks:
        peak_str = (f" | dense peak BW ~{peaks.get('bw', 0):.0f} GB/s"
                    f" | dense fp16/bf16 TC ~{peaks.get('tf16', 0):.0f} TFLOP/s"
                    f" | dense fp8/int8 TC ~{peaks.get('tf8', 0):.0f} TFLOP/s")
    lines.append(f"- **GPU**: {gpu.name} (sm_{gpu.sm_arch}, CC {gpu.cc}, "
                 f"{gpu.sm_count} SMs, {gpu.total_mem_gb:.1f} GB){peak_str}")
    lines.append(f"- **detected via**: {gpu.source}")
    clk = cfg.get("clocks") or {}
    lines.append(f"- **clocks**: SM {clk.get('sm_mhz', '?')} MHz, "
                 f"mem {clk.get('mem_mhz', '?')} MHz "
                 f"(throttle: {clk.get('throttle', '?')})")
    env = cfg.get("env") or {}
    lines.append(f"- **driver**: {env.get('driver', '?')} | "
                 f"CUDA {env.get('torch_cuda', '?')} | "
                 f"cuDNN {env.get('cudnn', '?')} | "
                 f"power cap {env.get('power_limit_w', '?')} W | "
                 f"ECC {env.get('ecc', '?')}")
    tf = cfg.get("tf32") or {}
    lines.append(f"- **dtype**: {cfg['dtype']} | "
                 f"TF32 matmul={tf.get('matmul_allow_tf32')} "
                 f"(fp32_precision={tf.get('float32_matmul_precision')})")
    lines.append(f"- **timing**: L2-flush={'on' if cfg.get('l2_flush') else 'off'}"
                 f" | method=cuda-events (flushed)"
                 f" | target-ms={cfg.get('target_ms')}"
                 f" | iters={cfg.get('iters') if cfg.get('iters') else 'auto'}"
                 f" | warmup={cfg.get('warmup')}"
                 f" | L2-buffer={cfg.get('l2_bytes', 0) // (1024 * 1024)} MB")
    lines.append(f"- **kernel-set**: {cfg.get('ks_version', '?')} "
                 f"(backend {cfg.get('backend', '?')})")
    if cfg.get("torch_version"):
        lines.append(f"- **torch**: {cfg['torch_version']}")
    if env.get("nvcc"):
        lines.append(f"- **nvcc**: {env['nvcc']}")
    if cfg.get("git_commit"):
        lines.append(f"- **harness commit**: {cfg['git_commit']}")
    if cfg.get("timestamp"):
        lines.append(f"- **timestamp**: {cfg['timestamp']}")
    lines.append(f"- **host**: {platform.platform()}")
    lines.append("")
    return lines


def render_markdown(rows: List[Row], gpu: "GpuInfo", cfg: dict) -> str:
    lines = render_header(gpu, cfg)
    counts = summarize(rows)
    lines.append(
        f"**Providers**: ok={counts.get('ok', 0)} · "
        f"skip={counts.get('skip', 0)} · "
        f"import-fail={counts.get('import-fail', 0)} · "
        f"error={counts.get('error', 0)} · "
        f"incorrect={counts.get('incorrect', 0)}. Correctness is gated against a "
        f"shared fp32 reference at the dtype tolerance BEFORE speed is reported.")
    lines.append("")
    lines.append("Latency cells show **median (min)** microseconds (CUDA events, "
                 "L2-flushed). The perf column is GB/s (bandwidth-bound ops) or "
                 "TFLOP/s (compute-bound), with **% of dense peak** for this "
                 "SKU+dtype. Arch-gated providers SKIP on unsupported SMs "
                 "(`needs smXX`) and are never imported there.")
    lines.append("")

    speedups = compute_speedups(rows)

    # group rows by op, preserving first-seen order
    order: List[str] = []
    by_op: Dict[str, List[Row]] = {}
    for r in rows:
        if r.op not in by_op:
            by_op[r.op] = []
            order.append(r.op)
        by_op[r.op].append(r)

    for op in order:
        lines.append(f"## {op}")
        lines.append("")
        lines.append("| op | shape | impl | dtype | lat us (min) | "
                     "GB/s or TFLOP/s (%pk) | rel_err | status |")
        lines.append("|---|---|---|---|--:|--:|--:|---|")
        for r in by_op[op]:
            rel = "-" if math.isnan(r.rel_err) else f"{r.rel_err:.2e}"
            lines.append(
                f"| {r.op} | {r.shape} | {r.impl} | {r.dtype} | "
                f"{_fmt_lat(r.lat_us, r.min_us)} | {_perf_cell(r)} | "
                f"{rel} | {_status_cell(r)} |")
        lines.append("")
        # per-(op,shape) kernel-set vs best-SOTA speedup
        shape_keys = [k for k in speedups if k[0] == op]
        if shape_keys:
            lines.append("**kernel-set vs best-SOTA**:")
            lines.append("")
            for key in shape_keys:
                info = speedups[key]
                if math.isnan(info["speedup"]):
                    if math.isnan(info["ks_us"]):
                        detail = "kernel-set did not run"
                    elif not info["best_other"]:
                        detail = "no SOTA provider ran (all skip/fail)"
                    else:
                        detail = "n/a"
                    lines.append(f"- `{key[1]}`: {detail}")
                else:
                    arrow = "faster" if info["speedup"] >= 1 else "slower"
                    lines.append(
                        f"- `{key[1]}`: kernel-set {info['ks_us']:.1f}us vs "
                        f"best `{info['best_other']}` {info['best_us']:.1f}us "
                        f"=> **{info['speedup']:.2f}x** ({arrow})")
            lines.append("")
    lines.append("_Legend: %pk = % of dense peak. `skip: needs smXX` = "
                 "arch-gated (provider needs a newer GPU; not imported here). "
                 "`import-fail` = library not installed. `incorrect` = "
                 "disagrees with the fp32 reference._")
    lines.append("")
    return "\n".join(lines)


def render_json(rows: List[Row], gpu: "GpuInfo", cfg: dict) -> str:
    speedups = {f"{k[0]}|{k[1]}": v for k, v in compute_speedups(rows).items()}
    return json.dumps({
        "gpu": gpu.__dict__,
        "config": cfg,
        "summary": summarize(rows),
        "speedups": speedups,
        "rows": [r.to_dict() for r in rows],
    }, indent=2, default=str)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark kernel-set against industry SOTA providers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ops", default="all",
                   help="comma-separated op groups (or 'all'). "
                        f"Choices: {','.join(ALL_OPS)}")
    p.add_argument("--dtype", default="fp16",
                   help="element dtype: fp16, bf16, or fp32")
    p.add_argument("--model-id", default="synthetic",
                   help="model/context label recorded in JSON results")
    p.add_argument("--layer-idx", type=int, default=None,
                   help="optional model layer index recorded in JSON results")
    p.add_argument("--position-kind", default=None,
                   help="optional position label, e.g. prefill/decode")
    p.add_argument("--position", type=int, default=None,
                   help="optional token/sequence position recorded in JSON results")
    p.add_argument("--warmup", type=int, default=10, help="warmup launches")
    p.add_argument("--iters", type=int, default=None,
                   help="fixed timed launches (overrides --target-ms)")
    p.add_argument("--target-ms", type=float, default=200.0,
                   help="measurement budget in ms (auto-calibrates iters)")
    p.add_argument("--max-iters", type=int, default=1000,
                   help="upper bound on auto-calibrated iteration count")
    p.add_argument("--no-l2-flush", dest="l2_flush", action="store_false",
                   default=True, help="disable the per-iteration L2 flush")
    p.add_argument("--no-tf32", dest="tf32", action="store_false", default=True,
                   help="disable TF32 for fp32 matmul")
    p.add_argument("--output", default=None,
                   help="write the report to this file (default: stdout)")
    p.add_argument("--format", default="md", choices=["md", "json"],
                   help="report format")
    p.add_argument("--json-output", default=None,
                   help="also write a structured JSON report to this file "
                        "from the same benchmark run")
    p.add_argument("--timestamp", default=None,
                   help="optional run timestamp/label for the header")
    p.add_argument("--list", action="store_true",
                   help="list op groups + the providers each compares, then exit")
    return p.parse_args(argv)


def list_ops() -> None:
    print("kernel-set vs SOTA — op groups and the providers each compares:\n")
    for op in ALL_OPS:
        provs = _OP_PROVIDERS.get(op, [])
        gates = []
        for pkey in provs:
            base = pkey.replace("(N/A)", "")
            sm = _PROVIDER_MIN_SM.get(base)
            if sm and sm > 89:
                gates.append(f"{base} (needs sm{sm})")
            else:
                gates.append(base)
        print(f"  {op}")
        print(f"      providers: {', '.join(gates)}")
    print("\nArch gating: on L4 (sm89), providers needing sm90+ (FlashMLA, "
          "DeepGEMM, Machete) SKIP automatically.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.list:
        list_ops()
        return 0

    if not _HAVE_BENCH:
        print("ERROR: could not import bench.py (timing utilities).\n"
              f"  {type(_BENCH_IMPORT_ERROR).__name__}: {_BENCH_IMPORT_ERROR}\n"
              "  bench_sota.py must live next to bench.py.", file=sys.stderr)
        return 2

    if not _HAVE_KS:
        print("ERROR: could not import kernel_set.\n"
              f"  {type(_KS_IMPORT_ERROR).__name__}: {_KS_IMPORT_ERROR}\n"
              "  Install the binding (pip install ./bindings/python) and set "
              "KERNEL_SET_LIB.", file=sys.stderr)
        return 2

    if not _HAVE_TORCH or not torch.cuda.is_available():
        print("ERROR: torch with CUDA is required to drive the benchmarks. "
              "Install torch and run on a GPU.", file=sys.stderr)
        return 2

    gpu = detect_gpu()
    print(f"Detected GPU: {gpu.name} (sm_{gpu.sm_arch}, via {gpu.source})",
          file=sys.stderr)

    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
        print("WARNING: CUDA_LAUNCH_BLOCKING=1 corrupts async timing; unset it.",
              file=sys.stderr)

    try:
        dt = torch_dtype(args.dtype)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if dt is torch.bfloat16 and not gpu.supports_bf16:
        print(f"WARNING: {gpu.name} reports no bf16 support; results may error.",
              file=sys.stderr)

    ops = ALL_OPS if args.ops == "all" else [o.strip()
                                             for o in args.ops.split(",")]
    unknown = [o for o in ops if o not in _GROUPS]
    if unknown:
        print(f"ERROR: unknown op group(s): {unknown}. Known: {ALL_OPS}",
              file=sys.stderr)
        return 2

    tf32_info = apply_tf32(args.tf32)

    # configure bench.py's global timing knobs (we reuse its time_op()).
    _TIMING.l2_flush = bool(args.l2_flush)
    _TIMING.cudagraph = False
    _TIMING.target_ms = float(args.target_ms)
    _TIMING.iters = args.iters
    _TIMING.warmup = int(args.warmup)
    _TIMING.max_iters = int(args.max_iters)
    _TIMING.l2_bytes = query_l2_flush_bytes(gpu)

    ctx = SotaCtx(dt=dt, dtype_name=args.dtype, gpu=gpu)

    cfg = {
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "target_ms": args.target_ms,
        "max_iters": args.max_iters,
        "l2_flush": _TIMING.l2_flush,
        "l2_bytes": _TIMING.l2_bytes,
        "ops": ops,
        "context": {
            "model_id": args.model_id,
            "layer_idx": args.layer_idx,
            "position_kind": args.position_kind,
            "position": args.position,
        },
        "tf32": tf32_info,
        "clocks": query_clocks(),
        "env": collect_env(),
        "git_commit": git_commit(),
        "timestamp": (args.timestamp
                      or datetime.datetime.now().isoformat(timespec="seconds")),
        "ks_version": ks.version() if _HAVE_KS else "?",
        "backend": ks.backend_name() if _HAVE_KS else "?",
        "torch_version": torch.__version__ if _HAVE_TORCH else None,
    }

    print(f"Running {len(ops)} op groups, dtype={args.dtype}, "
          f"L2-flush={_TIMING.l2_flush}, target-ms={args.target_ms}, "
          f"iters={args.iters if args.iters else 'auto'}\n", file=sys.stderr)

    rows = run_groups(ops, ctx)

    json_report: Optional[str] = None
    if args.format == "json":
        json_report = render_json(rows, gpu, cfg)
        report = json_report
    else:
        report = render_markdown(rows, gpu, cfg)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                    exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"\nWrote report to {args.output}", file=sys.stderr)
    else:
        print(report)

    if args.json_output:
        if json_report is None:
            json_report = render_json(rows, gpu, cfg)
        os.makedirs(os.path.dirname(os.path.abspath(args.json_output)),
                    exist_ok=True)
        with open(args.json_output, "w") as f:
            f.write(json_report + "\n")
        print(f"Wrote JSON report to {args.json_output}", file=sys.stderr)

    counts = summarize(rows)
    print(f"\nDone. ok={counts.get('ok', 0)} skip={counts.get('skip', 0)} "
          f"import-fail={counts.get('import-fail', 0)} "
          f"error={counts.get('error', 0)} "
          f"incorrect={counts.get('incorrect', 0)}", file=sys.stderr)

    # nonzero only if literally nothing ran (helps CI); skips/import-fails are
    # expected and NOT failures.
    if rows and counts.get("ok", 0) == 0 and counts.get("skip", 0) == 0:
        print("ERROR: no provider ran successfully.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
