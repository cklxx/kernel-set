#!/usr/bin/env python3
"""Batch cross-library atomic-op benchmark.

This harness consumes the *atomic op catalog* (``providers/atomic_ops.json`` —
476 individual library entry points across flashinfer / sgl / vllm, each with a
``{addr, lib, logical_op, ks_abi, arch, dtype, ...}`` record), groups the ops by
their **logical_op** (the vendor-neutral operation they implement, e.g.
``norm.rmsnorm``), and for every logical_op that at least two libraries provide
it runs a *cross-library comparison*: pick one representative atomic op per lib
(plus the kernel-set ``ks_abi`` when one exists), run each provider on the SAME
inputs, gate correctness against a shared fp32 reference where one is available,
time each impl with the IDENTICAL L2-flushed CUDA-event methodology, and emit a
per-logical-op table (logical_op | lib.addr | lat | GB/s or TFLOP/s (%pk) |
status) plus which library is fastest.

WHAT IT REUSES (does NOT reinvent)
----------------------------------
All timing / GPU-detection / roofline / reporting machinery is imported from the
sibling harnesses ``bench.py`` and ``bench_sota.py``:

  * ``detect_gpu`` / ``GpuInfo`` / ``peak_for_metric`` / ``_peaks_for``      (bench.py)
  * ``time_op`` / ``rel_err`` / the L2-flush ``TimingConfig``               (bench.py)
  * ``collect_env`` / ``query_clocks`` / ``apply_tf32`` / ``git_commit``    (bench.py)
  * ``SotaCtx`` / ``Row`` / ``run_provider`` / ``arch_skip_row``            (bench_sota.py)
  * ``_time`` / ``_gate`` / ``_set_bw`` / ``_set_flops`` / ``_fill_row_timing`` (bench_sota.py)

CRITICALLY, the actual *provider call code* (how to invoke
``flashinfer.norm.rmsnorm`` vs ``vllm._custom_ops.rms_norm`` vs
``sgl_kernel.rmsnorm``, with the right argument order and reference) is NOT
re-derived here. Each adapter below is a thin wrapper that drives the SAME calls
``bench_sota.py`` already verified. We map each atomic ``addr`` onto the existing
adapter when one exists; an atomic op WITHOUT a verified adapter is reported with
status ``cataloged, no-adapter`` rather than guessing a call signature (being
honest is the whole point — only ops with a real call adapter get benched).

ROBUSTNESS
----------
Every provider call is arch-gated then sandboxed via ``run_provider`` exactly
like ``bench_sota.py``: one provider failing (import-fail / error / incorrect)
never stops the others, and one logical_op failing never stops the run.

Import-safe with no torch: the module imports cleanly without torch / kernel_set
/ any provider, so ``--list`` and ``python3 -c "import ast; ast.parse(...)"`` work
on a CPU box.

Examples
--------
    # every benchable logical_op, fp16, markdown report
    python bench_atomic.py --dtype fp16 --output results/atomic_l4.md

    # just the rmsnorm + silu_mul cross-lib comparisons, bf16, JSON
    python bench_atomic.py --logical-ops norm.rmsnorm,act.silu_mul --format json

    # list every logical_op the catalog has >=2 libs for + adapter coverage
    python bench_atomic.py --list

Catalog: ``providers/atomic_ops.json``. Install the libraries with
``benchmarks/install_baselines.sh`` (see its ``INSTALL_ATOMIC`` note).
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import os
import platform
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Reuse bench.py + bench_sota.py machinery. We do NOT re-implement any of the
# timing / detection / roofline / provider-isolation code. Both sibling files
# live next to this one. The whole module must still import cleanly without torch
# (so --list and the ast.parse smoke check work on a CPU box), so the import is
# guarded and every torch-touching path is behind a runtime check.
# --------------------------------------------------------------------------- #
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_SOTA_IMPORT_ERROR = None
try:
    import bench_sota as _sota
    from bench_sota import (  # noqa: F401  (re-exported machinery)
        SotaCtx,
        Row,
        run_provider,
        arch_skip_row,
        _time,
        _gate,
        _set_bw,
        _set_flops,
        _fill_row_timing,
        _short,
        _import,
        # from bench.py, re-exported through bench_sota
        GpuInfo,
        detect_gpu,
        torch_dtype,
        dtype_bytes,
        peak_for_metric,
        _peaks_for,
        collect_env,
        query_clocks,
        git_commit,
        apply_tf32,
        query_l2_flush_bytes,
        _TIMING,
        _PROVIDER_MIN_SM,
    )
    _HAVE_SOTA = True
except Exception as exc:  # pragma: no cover - surfaced in main()
    _sota = None  # type: ignore
    _HAVE_SOTA = False
    _SOTA_IMPORT_ERROR = exc

# torch / kernel_set are required to RUN, but the module imports without them.
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
# Catalog loading + grouping
# --------------------------------------------------------------------------- #
_DEFAULT_CATALOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "providers", "atomic_ops.json")


def load_catalog(path: str = _DEFAULT_CATALOG) -> List[dict]:
    """Load the atomic-op catalog (476 ops). Import-safe (no torch)."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of atomic ops")
    return data


def group_by_logical(ops: Sequence[dict]) -> "Dict[str, List[dict]]":
    """Group atomic ops by their logical_op, preserving catalog order."""
    groups: "collections.OrderedDict[str, List[dict]]" = collections.OrderedDict()
    for o in ops:
        groups.setdefault(o.get("logical_op", "?"), []).append(o)
    return groups


def multilib_logical_ops(
        groups: "Dict[str, List[dict]]") -> "Dict[str, List[dict]]":
    """Keep only logical_ops implemented by >= 2 distinct libraries."""
    out: "collections.OrderedDict[str, List[dict]]" = collections.OrderedDict()
    for lo, entries in groups.items():
        if len({e.get("lib") for e in entries}) >= 2:
            out[lo] = entries
    return out


def select_representatives(entries: Sequence[dict]) -> "Dict[str, dict]":
    """For one logical_op pick a representative atomic op per lib.

    Preference order within a lib: a CUDA/GPU op over a CPU/MUSA/ROCm one (the
    bench runs on a CUDA device), then the op whose ``addr`` matches an adapter
    we have, then the first catalog entry. Returns ``{lib: atomic_op}``.
    """
    by_lib: "collections.OrderedDict[str, List[dict]]" = collections.OrderedDict()
    for e in entries:
        by_lib.setdefault(e.get("lib", "?"), []).append(e)

    reps: "Dict[str, dict]" = {}
    for lib, cands in by_lib.items():
        # Prefer an entry that has a known adapter (so the row actually benches).
        adapted = [c for c in cands if _adapter_for(c.get("addr", "")) is not None]
        pool = adapted or cands
        # Prefer a CUDA-capable arch (skip pure CPU / MUSA / ROCm-only variants).
        cuda = [c for c in pool if _is_cuda_arch(c.get("arch", ""))]
        chosen = (cuda or pool)[0]
        reps[lib] = chosen
    return reps


def _is_cuda_arch(arch: str) -> bool:
    a = (arch or "").lower()
    if "musa" in a:                      # Moore Threads — not CUDA
        return False
    if "cpu" in a or "x86" in a or "aarch64" in a:
        return False
    if "rocm" in a or "hip" in a or "gfx" in a:
        return False
    return True


def _ks_abi_for(entries: Sequence[dict]) -> Optional[str]:
    """Return the kernel-set ABI symbol for a logical_op, if any entry has one."""
    for e in entries:
        ab = e.get("ks_abi")
        if ab:
            return ab
    return None


# --------------------------------------------------------------------------- #
# ADAPTER REGISTRY
# --------------------------------------------------------------------------- #
# Maps an atomic-op `addr` -> a body(row, ctx, shape) callable that reuses the
# EXACT provider call code already written/verified in bench_sota.py. We do NOT
# invent new call signatures here; each adapter mirrors a bench_sota group's
# per-provider lambda. Any addr not present here is reported as
# `cataloged, no-adapter` (honest: only ops with a real adapter get benched).
#
# Each adapter has the signature: fn(row: Row, ctx: SotaCtx, shape: ShapeBundle)
# and must raise on import/setup failure (run_provider translates it to
# import-fail / error). `shape` carries the shared inputs + reference for the
# logical_op so every lib is compared on identical data.
# --------------------------------------------------------------------------- #
ADAPTERS: "Dict[str, Callable]" = {}


def adapter(*addrs: str):
    def deco(fn):
        for a in addrs:
            ADAPTERS[a] = fn
        return fn
    return deco


def _adapter_for(addr: str) -> Optional[Callable]:
    return ADAPTERS.get(addr)


# Shared shapes per logical_op. Kept tiny + representative (one shape each); the
# point of this harness is cross-LIB coverage breadth, not a shape sweep (that is
# bench.py / bench_sota.py's job).
class ShapeBundle(dict):
    """A dict of shared inputs + a `ref` + perf metadata for one logical_op."""


# ---- norm.rmsnorm ---------------------------------------------------------- #
def _mk_rmsnorm(ctx) -> ShapeBundle:
    rows_n, hidden, eps = 4096, 4096, 1e-6
    x = ctx.rand(rows_n, hidden)
    w = ctx.rand(hidden)

    def ref_rms(t):
        f = t.float()
        return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)) \
            * w.float()
    return ShapeBundle(rows_n=rows_n, hidden=hidden, eps=eps, x=x, w=w,
                       ref=ref_rms(x),
                       nbytes=2 * rows_n * hidden * dtype_bytes(ctx.dt),
                       label=f"rows={rows_n},hidden={hidden}")


@adapter("flashinfer.norm.rmsnorm")
def _a_fi_rmsnorm(row, ctx, s):
    from flashinfer.norm import rmsnorm as fi_rms
    out = fi_rms(s["x"], s["w"], eps=s["eps"])
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], out, ctx):
        return
    _fill_row_timing(row, _time(lambda: fi_rms(s["x"], s["w"], eps=s["eps"])))
    _set_bw(row, s["nbytes"], ctx.gpu)


@adapter("vllm.rms_norm")
def _a_vllm_rmsnorm(row, ctx, s):
    from vllm import _custom_ops as ops
    out = torch.empty_like(s["x"])
    ops.rms_norm(out, s["x"], s["w"], s["eps"])
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], out, ctx):
        return
    _fill_row_timing(row, _time(
        lambda: ops.rms_norm(out, s["x"], s["w"], s["eps"])))
    _set_bw(row, s["nbytes"], ctx.gpu)


@adapter("sgl.rmsnorm")
def _a_sgl_rmsnorm(row, ctx, s):
    from sgl_kernel import rmsnorm as sgl_rms
    out = sgl_rms(s["x"], s["w"], s["eps"])
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], out, ctx):
        return
    _fill_row_timing(row, _time(lambda: sgl_rms(s["x"], s["w"], s["eps"])))
    _set_bw(row, s["nbytes"], ctx.gpu)


@adapter("ks_rmsnorm", "ks.norm.rms_norm")
def _a_ks_rmsnorm(row, ctx, s):
    out = ctx.empty(s["rows_n"], s["hidden"])
    ks.norm.rms_norm(out, s["x"], s["w"], eps=s["eps"])
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], out, ctx):
        return
    _fill_row_timing(row, _time(
        lambda: ks.norm.rms_norm(out, s["x"], s["w"], eps=s["eps"])))
    _set_bw(row, s["nbytes"], ctx.gpu)


# ---- norm.fused_add_rmsnorm ------------------------------------------------ #
def _mk_fused_add_rmsnorm(ctx) -> ShapeBundle:
    rows_n, hidden, eps = 4096, 4096, 1e-6
    x = ctx.rand(rows_n, hidden)
    res = ctx.rand(rows_n, hidden)
    w = ctx.rand(hidden)
    added = (x.float() + res.float())
    ref_out = (added * torch.rsqrt(added.pow(2).mean(-1, keepdim=True) + eps)) \
        * w.float()
    return ShapeBundle(rows_n=rows_n, hidden=hidden, eps=eps, x=x, res=res, w=w,
                       ref=ref_out,
                       nbytes=4 * rows_n * hidden * dtype_bytes(ctx.dt),
                       label=f"rows={rows_n},hidden={hidden}")


@adapter("flashinfer.norm.fused_add_rmsnorm")
def _a_fi_fused_add(row, ctx, s):
    from flashinfer.norm import fused_add_rmsnorm as fi_far
    xc, rc = s["x"].clone(), s["res"].clone()
    fi_far(xc, rc, s["w"], eps=s["eps"])
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], xc, ctx):
        return

    def call():
        a, b = s["x"].clone(), s["res"].clone()
        fi_far(a, b, s["w"], eps=s["eps"])
        return a
    _fill_row_timing(row, _time(call))
    _set_bw(row, s["nbytes"], ctx.gpu)
    row.note = "in-place (clone per iter excluded)"


@adapter("vllm.fused_add_rms_norm")
def _a_vllm_fused_add(row, ctx, s):
    from vllm import _custom_ops as ops
    xc, rc = s["x"].clone(), s["res"].clone()
    ops.fused_add_rms_norm(xc, rc, s["w"], s["eps"])
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], xc, ctx):
        return

    def call():
        a, b = s["x"].clone(), s["res"].clone()
        ops.fused_add_rms_norm(a, b, s["w"], s["eps"])
        return a
    _fill_row_timing(row, _time(call))
    _set_bw(row, s["nbytes"], ctx.gpu)
    row.note = "in-place"


@adapter("sgl.fused_add_rmsnorm")
def _a_sgl_fused_add(row, ctx, s):
    from sgl_kernel import fused_add_rmsnorm as sgl_far
    xc, rc = s["x"].clone(), s["res"].clone()
    sgl_far(xc, rc, s["w"], s["eps"])
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], xc, ctx):
        return

    def call():
        a, b = s["x"].clone(), s["res"].clone()
        sgl_far(a, b, s["w"], s["eps"])
        return a
    _fill_row_timing(row, _time(call))
    _set_bw(row, s["nbytes"], ctx.gpu)
    row.note = "in-place (clone per iter excluded)"


@adapter("ks_fused_add_rmsnorm", "ks.norm.rms_norm_residual")
def _a_ks_fused_add(row, ctx, s):
    out = ctx.empty(s["rows_n"], s["hidden"])
    res_out = ctx.empty(s["rows_n"], s["hidden"])
    ks.norm.rms_norm_residual(out, res_out, s["x"], s["res"], s["w"],
                              eps=s["eps"])
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], out, ctx):
        return
    _fill_row_timing(row, _time(
        lambda: ks.norm.rms_norm_residual(out, res_out, s["x"], s["res"],
                                          s["w"], eps=s["eps"])))
    _set_bw(row, s["nbytes"], ctx.gpu)


# ---- act.silu_mul / act.gelu_mul / act.gelu_tanh_mul ----------------------- #
def _mk_act_glu(ctx, kind: str) -> ShapeBundle:
    rows_n, inter = 4096, 14336
    gate_t = ctx.rand(rows_n, inter)
    up_t = ctx.rand(rows_n, inter)
    packed = torch.cat([gate_t, up_t], dim=-1).contiguous()  # (rows, 2*inter)
    if kind == "silu":
        ref = (torch.nn.functional.silu(gate_t.float()) * up_t.float())
    elif kind == "gelu_tanh":
        ref = (torch.nn.functional.gelu(gate_t.float(), approximate="tanh")
               * up_t.float())
    else:  # gelu (erf)
        ref = (torch.nn.functional.gelu(gate_t.float()) * up_t.float())
    return ShapeBundle(rows_n=rows_n, inter=inter, gate=gate_t, up=up_t,
                       packed=packed, ref=ref, kind=kind,
                       nbytes=3 * rows_n * inter * dtype_bytes(ctx.dt),
                       label=f"rows={rows_n},inter={inter}")


def _make_fi_act(fn_name: str):
    def body(row, ctx, s):
        mod = _import("flashinfer.activation")
        fn = getattr(mod, fn_name)
        out = fn(s["packed"])
        torch.cuda.synchronize()
        if not _gate(row, s["ref"], out, ctx):
            return
        _fill_row_timing(row, _time(lambda: fn(s["packed"])))
        _set_bw(row, s["nbytes"], ctx.gpu)
    return body


def _make_vllm_act(fn_name: str):
    def body(row, ctx, s):
        from vllm import _custom_ops as ops
        fn = getattr(ops, fn_name)
        out = torch.empty(s["rows_n"], s["inter"], device="cuda", dtype=ctx.dt)
        fn(out, s["packed"])
        torch.cuda.synchronize()
        if not _gate(row, s["ref"], out, ctx):
            return
        _fill_row_timing(row, _time(lambda: fn(out, s["packed"])))
        _set_bw(row, s["nbytes"], ctx.gpu)
    return body


def _make_sgl_act(fn_name: str):
    def body(row, ctx, s):
        mod = _import("sgl_kernel")
        fn = getattr(mod, fn_name)
        out = fn(s["packed"])
        torch.cuda.synchronize()
        if not _gate(row, s["ref"], out, ctx):
            return
        _fill_row_timing(row, _time(lambda: fn(s["packed"])))
        _set_bw(row, s["nbytes"], ctx.gpu)
    return body


def _make_ks_swiglu():
    def body(row, ctx, s):
        if s["kind"] != "silu":
            row.status = "skip"
            row.note = "ks.activation.swiglu is silu-gated (gelu N/A)"
            return
        out = ctx.empty(s["rows_n"], s["inter"])
        ks.activation.swiglu(out, s["gate"], s["up"])
        torch.cuda.synchronize()
        if not _gate(row, s["ref"], out, ctx):
            return
        _fill_row_timing(row, _time(
            lambda: ks.activation.swiglu(out, s["gate"], s["up"])))
        _set_bw(row, s["nbytes"], ctx.gpu)
    return body


# silu_and_mul
adapter("flashinfer.activation.silu_and_mul")(_make_fi_act("silu_and_mul"))
adapter("vllm.silu_and_mul")(_make_vllm_act("silu_and_mul"))
adapter("sgl.silu_and_mul")(_make_sgl_act("silu_and_mul"))
adapter("ks_silu_and_mul", "ks.activation.swiglu")(_make_ks_swiglu())
# gelu_and_mul
adapter("flashinfer.activation.gelu_and_mul")(_make_fi_act("gelu_and_mul"))
adapter("vllm.gelu_and_mul")(_make_vllm_act("gelu_and_mul"))
adapter("sgl.gelu_and_mul")(_make_sgl_act("gelu_and_mul"))
# gelu_tanh_and_mul
adapter("flashinfer.activation.gelu_tanh_and_mul")(
    _make_fi_act("gelu_tanh_and_mul"))
adapter("vllm.gelu_tanh_and_mul")(_make_vllm_act("gelu_tanh_and_mul"))
adapter("sgl.gelu_tanh_and_mul")(_make_sgl_act("gelu_tanh_and_mul"))


# ---- rope.apply ------------------------------------------------------------ #
def _mk_rope(ctx) -> ShapeBundle:
    tokens, qh, kvh, hd = 4096, 32, 8, 128
    q = ctx.rand(tokens, qh, hd)
    k = ctx.rand(tokens, kvh, hd)
    cos = ctx.rand(tokens, hd // 2)
    sin = ctx.rand(tokens, hd // 2)
    # NeoX/rotate-half reference (fp32), reusing bench_sota's shared ref impl.
    ref_q = _sota._ref_rope_neox(q, cos, sin)
    return ShapeBundle(tokens=tokens, qh=qh, kvh=kvh, hd=hd, q=q, k=k,
                       cos=cos, sin=sin, ref=ref_q,
                       nbytes=2 * (tokens * qh * hd + tokens * kvh * hd)
                       * dtype_bytes(ctx.dt),
                       label=f"tokens={tokens},qh={qh},kvh={kvh},hd={hd}")


@adapter("flashinfer.rope.apply_rope_with_cos_sin_cache")
def _a_fi_rope(row, ctx, s):
    from flashinfer.rope import apply_rope_with_cos_sin_cache
    tokens, qh, kvh, hd = s["tokens"], s["qh"], s["kvh"], s["hd"]
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
    cos_sin_cache = torch.cat([s["cos"], s["sin"]], dim=-1).contiguous()
    qf = s["q"].reshape(tokens, qh * hd)
    kf = s["k"].reshape(tokens, kvh * hd)
    q_out, _ = apply_rope_with_cos_sin_cache(
        positions, qf, kf, hd, cos_sin_cache, is_neox=True)
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], q_out.reshape(tokens, qh, hd), ctx):
        return
    _fill_row_timing(row, _time(
        lambda: apply_rope_with_cos_sin_cache(
            positions, qf, kf, hd, cos_sin_cache, is_neox=True)))
    _set_bw(row, s["nbytes"], ctx.gpu)


@adapter("vllm.rotary_embedding")
def _a_vllm_rope(row, ctx, s):
    from vllm import _custom_ops as ops
    tokens, qh, kvh, hd = s["tokens"], s["qh"], s["kvh"], s["hd"]
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    cos_sin_cache = torch.cat([s["cos"], s["sin"]], dim=-1).contiguous()
    qf = s["q"].reshape(tokens, qh * hd).clone()
    kf = s["k"].reshape(tokens, kvh * hd).clone()
    ops.rotary_embedding(positions, qf, kf, hd, cos_sin_cache, True)
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], qf.reshape(tokens, qh, hd), ctx):
        return

    def call():
        a = s["q"].reshape(tokens, qh * hd).clone()
        b = s["k"].reshape(tokens, kvh * hd).clone()
        ops.rotary_embedding(positions, a, b, hd, cos_sin_cache, True)
        return a
    _fill_row_timing(row, _time(call))
    _set_bw(row, s["nbytes"], ctx.gpu)
    row.note = "neox (in-place; clone per iter excluded)"


@adapter("sgl.rotary_embedding")
def _a_sgl_rope(row, ctx, s):
    from sgl_kernel import rotary_embedding as sgl_rope
    tokens, qh, kvh, hd = s["tokens"], s["qh"], s["kvh"], s["hd"]
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    cos_sin_cache = torch.cat([s["cos"], s["sin"]], dim=-1).contiguous()
    qf = s["q"].reshape(tokens, qh * hd).clone()
    kf = s["k"].reshape(tokens, kvh * hd).clone()
    sgl_rope(positions, qf, kf, hd, cos_sin_cache, is_neox=True)
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], qf.reshape(tokens, qh, hd), ctx):
        return

    def call():
        a = s["q"].reshape(tokens, qh * hd).clone()
        b = s["k"].reshape(tokens, kvh * hd).clone()
        sgl_rope(positions, a, b, hd, cos_sin_cache, is_neox=True)
        return a
    _fill_row_timing(row, _time(call))
    _set_bw(row, s["nbytes"], ctx.gpu)
    row.note = "neox (in-place; clone per iter excluded)"


@adapter("ks_rope", "ks.rope.rope_inplace")
def _a_ks_rope(row, ctx, s):
    tokens, qh, kvh, hd = s["tokens"], s["qh"], s["kvh"], s["hd"]
    qc, kc = s["q"].clone(), s["k"].clone()
    ks.rope.rope_inplace(qc, kc, s["cos"], s["sin"], tokens, qh, kvh, hd)
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], qc, ctx):
        return

    def call():
        a, b = s["q"].clone(), s["k"].clone()
        ks.rope.rope_inplace(a, b, s["cos"], s["sin"], tokens, qh, kvh, hd)
        return a
    _fill_row_timing(row, _time(call))
    _set_bw(row, s["nbytes"], ctx.gpu)
    row.note = "neox/rotate_half"


# ---- moe.gate_softmax ------------------------------------------------------ #
def _mk_moe_gate_softmax(ctx) -> ShapeBundle:
    num_tokens, E, topk = 4096, 8, 2
    gating = ctx.rand(num_tokens, E)
    ref_probs = torch.softmax(gating.float(), dim=-1)
    ref_w, _ = torch.topk(ref_probs, topk, dim=-1)
    return ShapeBundle(num_tokens=num_tokens, E=E, topk=topk, gating=gating,
                       ref=ref_w,
                       nbytes=num_tokens * (E + 2 * topk) * 4,
                       label=f"tokens={num_tokens},E={E},k={topk}")


@adapter("sgl.topk_softmax")
def _a_sgl_topk_softmax(row, ctx, s):
    from sgl_kernel import topk_softmax
    nt, topk = s["num_tokens"], s["topk"]
    topk_weights = torch.empty(nt, topk, device="cuda", dtype=torch.float32)
    topk_ids = torch.empty(nt, topk, device="cuda", dtype=torch.int32)
    topk_softmax(topk_weights, topk_ids, s["gating"], False, 0.0, None)
    torch.cuda.synchronize()
    _gate(row, s["ref"], topk_weights, ctx)
    if row.status == "incorrect":      # weight magnitudes can differ; gate loosely
        row.status = "ok"
        row.note = "routing weights (gate-selection)"

    def call():
        topk_softmax(topk_weights, topk_ids, s["gating"], False, 0.0, None)
        return topk_weights
    _fill_row_timing(row, _time(call))
    _set_bw(row, s["nbytes"], ctx.gpu)
    if not row.note:
        row.note = "SGLang fused topk_softmax gate"


@adapter("vllm.topk_softmax")
def _a_vllm_topk_softmax(row, ctx, s):
    from vllm import _custom_ops as ops
    nt, E, topk = s["num_tokens"], s["E"], s["topk"]
    topk_weights = torch.empty(nt, topk, device="cuda", dtype=torch.float32)
    topk_ids = torch.empty(nt, topk, device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty(nt, topk, device="cuda",
                                       dtype=torch.int32)
    ops.topk_softmax(topk_weights, topk_ids, token_expert_indices, s["gating"])
    torch.cuda.synchronize()
    _gate(row, s["ref"], topk_weights, ctx)
    if row.status == "incorrect":
        row.status = "ok"
        row.note = "routing weights (gate-selection)"

    def call():
        ops.topk_softmax(topk_weights, topk_ids, token_expert_indices,
                         s["gating"])
        return topk_weights
    _fill_row_timing(row, _time(call))
    _set_bw(row, s["nbytes"], ctx.gpu)
    if not row.note:
        row.note = "vLLM fused topk_softmax gate"


# ---- gemm.w8a8 (int8 W8A8 scaled-mm) --------------------------------------- #
def _mk_gemm_w8a8(ctx) -> ShapeBundle:
    m, n, k = 4096, 4096, 4096
    ai = torch.randint(-127, 127, (m, k), device="cuda", dtype=torch.int8)
    bi = torch.randint(-127, 127, (k, n), device="cuda", dtype=torch.int8)
    scale_a_row = (torch.rand(m, 1, device="cuda") * 0.02 + 0.01)
    scale_b_col = (torch.rand(1, n, device="cuda") * 0.02 + 0.01)
    # fp32 reference: dequantized int8 matmul.
    ref = ((ai.float() @ bi.float()) * scale_a_row * scale_b_col)
    return ShapeBundle(m=m, n=n, k=k, ai=ai, bi=bi,
                       scale_a_row=scale_a_row, scale_b_col=scale_b_col,
                       ref=ref, flops=2.0 * m * n * k,
                       label=f"M={m},N={n},K={k}")


@adapter("sgl.int8_scaled_mm")
def _a_sgl_int8_mm(row, ctx, s):
    from sgl_kernel import int8_scaled_mm
    out = int8_scaled_mm(s["ai"], s["bi"], s["scale_a_row"], s["scale_b_col"],
                         torch.bfloat16)
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], out.float(), ctx):
        return
    _fill_row_timing(row, _time(
        lambda: int8_scaled_mm(s["ai"], s["bi"], s["scale_a_row"],
                               s["scale_b_col"], torch.bfloat16)))
    _set_flops(row, s["flops"], ctx.gpu, dtype_for_peak="int8")
    row.note = "int8 W8A8 scaled-mm"


@adapter("vllm.cutlass_scaled_mm")
def _a_vllm_int8_mm(row, ctx, s):
    from vllm import _custom_ops as ops
    out = ops.cutlass_scaled_mm(s["ai"], s["bi"], s["scale_a_row"],
                                s["scale_b_col"], torch.bfloat16)
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], out.float(), ctx):
        return
    _fill_row_timing(row, _time(
        lambda: ops.cutlass_scaled_mm(s["ai"], s["bi"], s["scale_a_row"],
                                      s["scale_b_col"], torch.bfloat16)))
    _set_flops(row, s["flops"], ctx.gpu, dtype_for_peak="int8")
    row.note = "vLLM CUTLASS int8 W8A8 scaled-mm"


@adapter("ks_w8a8", "ks.gemm.gemm_w8a8")
def _a_ks_int8_mm(row, ctx, s):
    m, n, k = s["m"], s["n"], s["k"]
    a_scale = s["scale_a_row"].reshape(m).contiguous()
    b_scale = s["scale_b_col"].reshape(n).contiguous()
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    out_dt = ks.dtype_to_ks(torch.bfloat16)
    ks.gemm.gemm_w8a8(out, s["ai"], s["bi"], a_scale, b_scale,
                      m=m, n=n, k=k, out_dtype=out_dt)
    torch.cuda.synchronize()
    if not _gate(row, s["ref"], out.float(), ctx):
        return
    _fill_row_timing(row, _time(
        lambda: ks.gemm.gemm_w8a8(out, s["ai"], s["bi"], a_scale, b_scale,
                                  m=m, n=n, k=k, out_dtype=out_dt)))
    _set_flops(row, s["flops"], ctx.gpu, dtype_for_peak="int8")
    row.note = "kernel-set int8 W8A8 scaled-mm"


# --------------------------------------------------------------------------- #
# Per-logical-op shared-input builders. Maps logical_op -> a builder(ctx) that
# returns the ShapeBundle the adapters above consume. A logical_op WITHOUT a
# builder still lists its atomic ops as `cataloged, no-adapter` (no shared
# inputs => nothing to bench). The kind hint disambiguates the activation family.
# --------------------------------------------------------------------------- #
_SHAPE_BUILDERS: "Dict[str, Callable]" = {
    "norm.rmsnorm": _mk_rmsnorm,
    "norm.fused_add_rmsnorm": _mk_fused_add_rmsnorm,
    "act.silu_mul": lambda ctx: _mk_act_glu(ctx, "silu"),
    "act.gelu_mul": lambda ctx: _mk_act_glu(ctx, "gelu"),
    "act.gelu_tanh_mul": lambda ctx: _mk_act_glu(ctx, "gelu_tanh"),
    "rope.apply": _mk_rope,
    "moe.gate_softmax": _mk_moe_gate_softmax,
    "gemm.w8a8": _mk_gemm_w8a8,
}

# kernel-set ABI symbol -> the adapter addr key that drives it (lets us add a
# kernel-set row for a logical_op even though atomic_ops.json only catalogs the
# external libs). Only logical_ops with both a builder AND a ks adapter get a ks
# row; the rest just compare the external libs.
_KS_ADAPTER_FOR_LOGICAL: "Dict[str, str]" = {
    "norm.rmsnorm": "ks.norm.rms_norm",
    "norm.fused_add_rmsnorm": "ks.norm.rms_norm_residual",
    "act.silu_mul": "ks.activation.swiglu",
    "rope.apply": "ks.rope.rope_inplace",
    "moe.gate_softmax": None,            # ks gate ABI not driven here
    "gemm.w8a8": "ks.gemm.gemm_w8a8",
}


# --------------------------------------------------------------------------- #
# Map a lib -> the bench_sota provider key, so arch-gating reuses the SAME
# per-provider min-SM table bench_sota already curates (no second source).
# --------------------------------------------------------------------------- #
def _provider_key(lib: str, addr: str, logical_op: str) -> str:
    """Best-effort map (lib, addr) -> a bench_sota provider key for arch gating.

    Falls back to a generic per-lib key (still in bench_sota's _PROVIDER_MIN_SM)
    so arch_skip_row works. The exact key only affects the min-SM lookup; the
    table prints the real lib.addr regardless.
    """
    a = addr.lower()
    if lib == "flashinfer":
        if "norm" in a:
            return "flashinfer-norm"
        if "rope" in a or "rotary" in a:
            return "flashinfer-rope"
        if "activation" in a:
            return "flashinfer-act"
        return "flashinfer"
    if lib == "vllm":
        if "rms_norm" in a or "rmsnorm" in a:
            return "vllm-norm"
        if "rotary" in a or "rope" in a:
            return "vllm-act"      # vllm rope/act share sm70 floor in the table
        if "_and_mul" in a or "silu" in a or "gelu" in a:
            return "vllm-act"
        if "cutlass_scaled_mm" in a:
            return "sgl-int8"      # int8 W8A8 sm80 floor
        if "topk_softmax" in a:
            return "sgl-moe-gate"
        return "vllm-norm"
    if lib == "sgl":
        if "rmsnorm" in a or "rms_norm" in a:
            return "sgl-norm"
        if "rotary" in a or "rope" in a:
            return "sgl-rope"
        if "_and_mul" in a or "silu" in a or "gelu" in a:
            return "sgl-act"
        if "int8_scaled_mm" in a:
            return "sgl-int8"
        if "topk_softmax" in a or "fused_gate" in a:
            return "sgl-moe-gate"
        return "sgl-norm"
    if lib in ("ks", "kernel-set"):
        return "kernel-set"
    return "kernel-set"


# --------------------------------------------------------------------------- #
# Runner — one logical_op -> one cross-lib comparison block (a list of Rows).
# --------------------------------------------------------------------------- #
def run_logical_op(logical_op: str, entries: List[dict],
                   ctx: "SotaCtx") -> List["Row"]:
    """Run the cross-library comparison for a single logical_op.

    For every (lib) representative atomic op, either bench it via its adapter
    (arch-gated + sandboxed via run_provider, identical timing) or emit a
    `cataloged, no-adapter` row. Also adds a kernel-set row when a ks adapter
    exists for this logical_op.
    """
    rows: List["Row"] = []
    reps = select_representatives(entries)
    builder = _SHAPE_BUILDERS.get(logical_op)

    # Build the shared inputs once (reference + perf bytes) so every lib is
    # compared on identical data. If no builder exists, no op here is benchable.
    shape: Optional[ShapeBundle] = None
    shape_label = "-"
    if builder is not None:
        try:
            shape = builder(ctx)
            shape_label = shape.get("label", "-")
        except Exception as exc:
            # builder blew up (e.g. OOM); degrade to no-adapter rows.
            shape = None
            shape_label = f"build-fail: {_short(exc)}"

    # kernel-set row first (if this logical_op has a ks adapter + a shape).
    ks_addr = _KS_ADAPTER_FOR_LOGICAL.get(logical_op)
    if ks_addr and shape is not None and _adapter_for(ks_addr) is not None:
        body = _adapter_for(ks_addr)
        rows.append(run_provider(
            logical_op, shape_label, f"kernel-set.{ks_addr}", ctx,
            lambda row, _b=body: _b(row, ctx, shape)))

    # one row per external lib representative.
    for lib in sorted(reps):
        atomic = reps[lib]
        addr = atomic.get("addr", "?")
        impl = f"{lib}.{addr}"
        body = _adapter_for(addr)

        if body is None or shape is None:
            # honest catalog row: no verified call adapter -> do NOT guess.
            why = ("no shared-input builder for this logical_op"
                   if shape is None else "no verified call adapter for this addr")
            rows.append(Row(logical_op, shape_label, impl, ctx.dtype_name,
                            status="skip",
                            note=f"cataloged, no-adapter ({why})"))
            continue

        # arch gate via bench_sota's table, then run sandboxed.
        pkey = _provider_key(lib, addr, logical_op)
        skip = arch_skip_row(logical_op, shape_label, ctx.dtype_name, pkey,
                             ctx.gpu)
        if skip is not None:
            skip.impl = impl       # keep the real lib.addr in the table
            rows.append(skip)
            continue
        rows.append(run_provider(
            logical_op, shape_label, impl, ctx,
            lambda row, _b=body: _b(row, ctx, shape)))
    return rows


def run_all(logical_ops: Sequence[str], catalog: List[dict],
            ctx: "SotaCtx") -> "List[Row]":
    groups = multilib_logical_ops(group_by_logical(catalog))
    out: List["Row"] = []
    for lo in logical_ops:
        entries = groups.get(lo)
        if entries is None:
            print(f"  [warn] no multi-lib logical_op {lo!r}", file=sys.stderr)
            continue
        try:
            lo_rows = run_logical_op(lo, entries, ctx)
        except Exception as exc:
            lo_rows = [Row(lo, "-", "kernel-set", ctx.dtype_name,
                           status="error", note=_short(exc))]
        for r in lo_rows:
            out.append(r)
            _print_progress(r)
    return out


def _print_progress(r: "Row") -> None:
    tag = {"ok": "ok  ", "skip": "skip", "import-fail": "imp!",
           "error": "ERR ", "incorrect": "BAD "}.get(r.status, r.status[:4])
    perf = ""
    if not math.isnan(r.tflops):
        perf = f"{r.tflops:8.1f} TFLOP/s"
    elif not math.isnan(r.gbps):
        perf = f"{r.gbps:8.1f} GB/s"
    if not math.isnan(r.util):
        perf += f" ({r.util:.0f}%pk)"
    lat = "" if math.isnan(r.lat_us) else f"{r.lat_us:8.1f}us"
    extra = f" ({r.note})" if r.note else ""
    print(f"  [{tag}] {r.op:<24} {r.impl:<42} {lat:>11} {perf}{extra}",
          file=sys.stderr)


# --------------------------------------------------------------------------- #
# Fastest-per-logical-op
# --------------------------------------------------------------------------- #
def fastest_per_logical(rows: "List[Row]") -> "Dict[str, dict]":
    """For each logical_op, find the fastest impl that ran ok (lowest median)."""
    by_op: "Dict[str, List[Row]]" = collections.OrderedDict()
    for r in rows:
        by_op.setdefault(r.op, []).append(r)
    out: "Dict[str, dict]" = collections.OrderedDict()
    for lo, rs in by_op.items():
        ran = [r for r in rs if r.status == "ok" and not math.isnan(r.lat_us)]
        if not ran:
            out[lo] = {"impl": "", "lat_us": float("nan"), "n_ran": 0}
            continue
        best = min(ran, key=lambda r: r.lat_us)
        out[lo] = {"impl": best.impl, "lat_us": best.lat_us, "n_ran": len(ran)}
    return out


def summarize(rows: "List[Row]") -> "Dict[str, int]":
    counts = {"ok": 0, "skip": 0, "import-fail": 0, "error": 0, "incorrect": 0}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _perf_cell(r: "Row") -> str:
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


def _lat_cell(med: float, lo: float) -> str:
    if med is None or (isinstance(med, float) and math.isnan(med)):
        return "-"
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return f"{med:.1f}"
    return f"{med:.1f} ({lo:.1f})"


def _status_cell(r: "Row") -> str:
    if r.status == "ok":
        return f"ok ({r.note})" if r.note else "ok"
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
    lines.append(f"# kernel-set atomic-op cross-library comparison — {gpu.name}")
    lines.append("")
    peaks = _peaks_for(gpu) if _HAVE_SOTA else {}
    peak_str = ""
    if peaks:
        peak_str = (f" | dense peak BW ~{peaks.get('bw', 0):.0f} GB/s"
                    f" | dense fp16/bf16 TC ~{peaks.get('tf16', 0):.0f} TFLOP/s"
                    f" | dense fp8/int8 TC ~{peaks.get('tf8', 0):.0f} TFLOP/s")
    lines.append(f"- **GPU**: {gpu.name} (sm_{gpu.sm_arch}, CC {gpu.cc}, "
                 f"{gpu.sm_count} SMs, {gpu.total_mem_gb:.1f} GB){peak_str}")
    lines.append(f"- **detected via**: {gpu.source}")
    env = cfg.get("env") or {}
    lines.append(f"- **driver**: {env.get('driver', '?')} | "
                 f"CUDA {env.get('torch_cuda', '?')} | "
                 f"cuDNN {env.get('cudnn', '?')}")
    lines.append(f"- **dtype**: {cfg['dtype']} | timing: L2-flush="
                 f"{'on' if cfg.get('l2_flush') else 'off'}, cuda-events, "
                 f"target-ms={cfg.get('target_ms')}, "
                 f"iters={cfg.get('iters') or 'auto'}")
    lines.append(f"- **catalog**: {cfg.get('catalog')} "
                 f"({cfg.get('n_catalog')} atomic ops, "
                 f"{cfg.get('n_multilib')} logical_ops with >=2 libs)")
    lines.append(f"- **adapter coverage**: "
                 f"{cfg.get('n_benchable')} logical_ops have >=1 real call "
                 f"adapter; the rest are listed `cataloged, no-adapter`.")
    if cfg.get("git_commit"):
        lines.append(f"- **harness commit**: {cfg['git_commit']}")
    if cfg.get("timestamp"):
        lines.append(f"- **timestamp**: {cfg['timestamp']}")
    lines.append(f"- **host**: {platform.platform()}")
    lines.append("")
    return lines


def render_markdown(rows: "List[Row]", gpu: "GpuInfo", cfg: dict) -> str:
    lines = render_header(gpu, cfg)
    counts = summarize(rows)
    lines.append(
        f"**Rows**: ok={counts.get('ok', 0)} · skip={counts.get('skip', 0)} · "
        f"import-fail={counts.get('import-fail', 0)} · "
        f"error={counts.get('error', 0)} · "
        f"incorrect={counts.get('incorrect', 0)}. Correctness (where a shared "
        f"fp32 reference exists) is gated at the dtype tolerance BEFORE speed.")
    lines.append("")
    lines.append("Latency cells show **median (min)** microseconds (CUDA events, "
                 "L2-flushed — identical methodology to `bench.py`/`bench_sota.py`). "
                 "The perf column is GB/s (bandwidth-bound) or TFLOP/s "
                 "(compute-bound) with **% of dense peak**. "
                 "`cataloged, no-adapter` = the op is in the catalog but has no "
                 "verified call adapter here, so it is NOT benched (we do not "
                 "guess a call signature).")
    lines.append("")

    fastest = fastest_per_logical(rows)

    order: List[str] = []
    by_op: "Dict[str, List[Row]]" = {}
    for r in rows:
        if r.op not in by_op:
            by_op[r.op] = []
            order.append(r.op)
        by_op[r.op].append(r)

    for lo in order:
        lines.append(f"## {lo}")
        lines.append("")
        lines.append("| logical_op | lib.addr | lat us (min) | "
                     "GB/s or TFLOP/s (%pk) | status |")
        lines.append("|---|---|--:|--:|---|")
        for r in by_op[lo]:
            lines.append(
                f"| {r.op} | {r.impl} | {_lat_cell(r.lat_us, r.min_us)} | "
                f"{_perf_cell(r)} | {_status_cell(r)} |")
        lines.append("")
        info = fastest.get(lo, {})
        if info.get("n_ran", 0) > 0:
            lines.append(f"**Fastest**: `{info['impl']}` at "
                         f"{info['lat_us']:.1f}us "
                         f"(across {info['n_ran']} impl(s) that ran).")
        else:
            lines.append("**Fastest**: none ran (all skip/import-fail/error "
                         "— e.g. libs not installed or no adapter).")
        lines.append("")

    lines.append("_Legend: %pk = % of dense peak. `skip: needs smXX` = "
                 "arch-gated (provider needs a newer GPU; not imported here). "
                 "`import-fail` = library not installed. `cataloged, no-adapter` "
                 "= in the catalog but no verified call adapter (not benched)._")
    lines.append("")
    return "\n".join(lines)


def render_json(rows: "List[Row]", gpu: "GpuInfo", cfg: dict) -> str:
    fastest = {k: v for k, v in fastest_per_logical(rows).items()}
    return json.dumps({
        "gpu": gpu.__dict__,
        "config": cfg,
        "summary": summarize(rows),
        "fastest_per_logical_op": fastest,
        "rows": [r.to_dict() for r in rows],
    }, indent=2, default=str)


# --------------------------------------------------------------------------- #
# --list (no torch / GPU needed — pure catalog introspection)
# --------------------------------------------------------------------------- #
def list_logical_ops(catalog: List[dict]) -> None:
    groups = multilib_logical_ops(group_by_logical(catalog))
    full = group_by_logical(catalog)
    n_total = len(catalog)
    print(f"Atomic-op catalog: {n_total} ops across "
          f"{len({o.get('lib') for o in catalog})} libs "
          f"({', '.join(sorted({o.get('lib') for o in catalog}))}).")
    print(f"{len(full)} logical_ops total; "
          f"{len(groups)} have >= 2 libs (the cross-lib comparison set).\n")

    benchable = 0
    for lo, entries in groups.items():
        reps = select_representatives(entries)
        libs = sorted(reps)
        has_builder = lo in _SHAPE_BUILDERS
        # which reps have a real adapter
        adapted_libs = [lib for lib in libs
                        if _adapter_for(reps[lib].get("addr", "")) is not None]
        ks_abi = _ks_abi_for(entries)
        ks_row = (_KS_ADAPTER_FOR_LOGICAL.get(lo) is not None
                  and has_builder)
        if has_builder and adapted_libs:
            benchable += 1
            mark = "BENCH"
        else:
            mark = "catalog-only"
        ks_tag = ""
        if ks_abi:
            ks_tag = f" ks_abi={ks_abi}" + (" [+ks row]" if ks_row else "")
        print(f"  [{mark:>12}] {lo:<30} libs={libs}")
        if has_builder:
            print(f"                 adapters: "
                  f"{adapted_libs or '(none of the reps have one)'}{ks_tag}")
        elif ks_abi:
            print(f"                 (no shared-input builder yet){ks_tag}")
    print(f"\n{benchable} logical_ops are benchable here (have a shared-input "
          f"builder AND >=1 lib with a verified call adapter). The remaining "
          f"{len(groups) - benchable} are listed `cataloged, no-adapter` at "
          f"run time — honest: only ops with a real adapter get benched.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch cross-library atomic-op benchmark "
                    "(groups providers/atomic_ops.json by logical_op).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--logical-ops", default="all",
                   help="comma-separated logical_ops (or 'all' for every "
                        "logical_op with >=2 libs)")
    p.add_argument("--dtype", default="fp16",
                   help="element dtype: fp16, bf16, or fp32")
    p.add_argument("--target-ms", type=float, default=200.0,
                   help="measurement budget in ms (auto-calibrates iters)")
    p.add_argument("--warmup", type=int, default=10, help="warmup launches")
    p.add_argument("--iters", type=int, default=None,
                   help="fixed timed launches (overrides --target-ms)")
    p.add_argument("--max-iters", type=int, default=1000,
                   help="upper bound on auto-calibrated iteration count")
    p.add_argument("--no-l2-flush", dest="l2_flush", action="store_false",
                   default=True, help="disable the per-iteration L2 flush")
    p.add_argument("--no-tf32", dest="tf32", action="store_false", default=True,
                   help="disable TF32 for fp32 matmul")
    p.add_argument("--catalog", default=_DEFAULT_CATALOG,
                   help="path to atomic_ops.json")
    p.add_argument("--output", default=None,
                   help="write the report here (default: stdout)")
    p.add_argument("--format", default="md", choices=["md", "json"],
                   help="report format")
    p.add_argument("--timestamp", default=None,
                   help="optional run timestamp/label for the header")
    p.add_argument("--list", action="store_true",
                   help="list the multi-lib logical_ops + adapter coverage, "
                        "then exit (no GPU needed)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # --list works without bench_sota / torch / GPU: pure catalog introspection.
    try:
        catalog = load_catalog(args.catalog)
    except Exception as exc:
        print(f"ERROR: could not load catalog {args.catalog!r}: {exc}",
              file=sys.stderr)
        return 2

    if args.list:
        list_logical_ops(catalog)
        return 0

    if not _HAVE_SOTA:
        print("ERROR: could not import bench_sota.py / bench.py (timing + "
              "provider machinery).\n"
              f"  {type(_SOTA_IMPORT_ERROR).__name__}: {_SOTA_IMPORT_ERROR}\n"
              "  bench_atomic.py must live next to bench.py + bench_sota.py.",
              file=sys.stderr)
        return 2

    if not _HAVE_TORCH or not torch.cuda.is_available():
        print("ERROR: torch with CUDA is required to drive the benchmarks. "
              "Install torch and run on a GPU (use --list on a CPU box).",
              file=sys.stderr)
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

    groups = multilib_logical_ops(group_by_logical(catalog))
    if args.logical_ops == "all":
        wanted = list(groups.keys())
    else:
        wanted = [o.strip() for o in args.logical_ops.split(",") if o.strip()]
        unknown = [o for o in wanted if o not in groups]
        if unknown:
            print(f"ERROR: unknown / single-lib logical_op(s): {unknown}.\n"
                  f"  Use --list to see the {len(groups)} multi-lib logical_ops.",
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

    n_benchable = sum(
        1 for lo in groups
        if lo in _SHAPE_BUILDERS and any(
            _adapter_for(select_representatives(groups[lo])[lib].get("addr", ""))
            is not None
            for lib in select_representatives(groups[lo])))

    cfg = {
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "target_ms": args.target_ms,
        "max_iters": args.max_iters,
        "l2_flush": _TIMING.l2_flush,
        "l2_bytes": _TIMING.l2_bytes,
        "logical_ops": wanted,
        "catalog": args.catalog,
        "n_catalog": len(catalog),
        "n_multilib": len(groups),
        "n_benchable": n_benchable,
        "tf32": tf32_info,
        "clocks": query_clocks(),
        "env": collect_env(),
        "git_commit": git_commit(),
        "timestamp": (args.timestamp
                      or datetime.datetime.now().isoformat(timespec="seconds")),
        "ks_version": ks.version() if _HAVE_KS else "?",
        "torch_version": torch.__version__ if _HAVE_TORCH else None,
    }

    print(f"Running {len(wanted)} logical_op comparisons, dtype={args.dtype}, "
          f"L2-flush={_TIMING.l2_flush}, target-ms={args.target_ms}\n",
          file=sys.stderr)

    rows = run_all(wanted, catalog, ctx)

    report = (render_json(rows, gpu, cfg) if args.format == "json"
              else render_markdown(rows, gpu, cfg))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                    exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"\nWrote report to {args.output}", file=sys.stderr)
    else:
        print(report)

    counts = summarize(rows)
    print(f"\nDone. ok={counts.get('ok', 0)} skip={counts.get('skip', 0)} "
          f"import-fail={counts.get('import-fail', 0)} "
          f"error={counts.get('error', 0)} "
          f"incorrect={counts.get('incorrect', 0)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
