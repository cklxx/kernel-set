#!/usr/bin/env python3
"""kernel-set benchmark harness.

Detects the GPU, then benchmarks each kernel category over representative LLM
shapes and reports:

  * latency (microseconds, median over `--iters` timed launches),
  * achieved memory bandwidth (GB/s) for bandwidth-bound ops, or compute
    throughput (TFLOP/s) for compute-bound ops,
  * correctness (relative error) vs a PyTorch reference where one exists, and
  * the kernel-set / PyTorch speedup.

Timing uses CUDA events (``torch.cuda.Event``) with a warmup phase, and falls
back to a CPU wall-clock timer (with ``torch.cuda.synchronize``) only when torch
is unavailable.

Examples
--------
    # everything, fp16
    python bench.py --dtype fp16

    # just norm + activation, bf16, more iters, write a markdown report
    python bench.py --ops rmsnorm,layernorm,swiglu --dtype bf16 \
        --iters 100 --output results/l4.md --format md

    # list the op categories this harness knows about
    python bench.py --list-ops

Library discovery: the kernel-set shared library is located by the
``kernel_set`` binding via ``KERNEL_SET_LIB`` / ``KERNEL_SET_LIB_DIR`` (see
``bindings/python/README.md``). ``build_and_bench.sh`` sets these for you.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Optional imports. The harness needs kernel_set; torch is needed for the
# tensor-convenience paths (which is how every benchmark below drives kernels)
# and for the reference comparisons. We degrade gracefully with clear errors.
# --------------------------------------------------------------------------- #
try:
    import torch
    _HAVE_TORCH = True
except Exception:  # pragma: no cover - torch is effectively required on a GPU box
    torch = None  # type: ignore
    _HAVE_TORCH = False

try:
    import kernel_set as ks
    _HAVE_KS = True
    _KS_IMPORT_ERROR = None
except Exception as exc:  # surfaced in main() with a helpful hint
    ks = None  # type: ignore
    _HAVE_KS = False
    _KS_IMPORT_ERROR = exc


# --------------------------------------------------------------------------- #
# GPU detection
# --------------------------------------------------------------------------- #
@dataclass
class GpuInfo:
    name: str = "unknown"
    compute_major: int = 0
    compute_minor: int = 0
    sm_count: int = 0
    total_mem_gb: float = 0.0
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_tf32: bool = False
    source: str = "unknown"   # "kernel_set", "torch", "nvidia-smi" or "none"

    @property
    def sm_arch(self) -> int:
        """SM architecture as a 2-digit int, e.g. 89 for L4, 80 for A100."""
        return self.compute_major * 10 + self.compute_minor

    @property
    def cc(self) -> str:
        return f"{self.compute_major}.{self.compute_minor}"

    @property
    def slug(self) -> str:
        """Filesystem-friendly short name, e.g. 'l4', 'a100', 'rtx4090'."""
        n = self.name.lower()
        for key in ("a100", "h100", "h200", "l4", "l40s", "l40", "a10g", "a10",
                    "t4", "v100", "4090", "3090", "a6000"):
            if key in n.replace(" ", ""):
                return key
        # fall back to a sanitized device name
        safe = "".join(c if c.isalnum() else "-" for c in n).strip("-")
        return safe or f"sm{self.sm_arch}"


# Known peak specs (memory bandwidth GB/s, dense tensor-core TFLOP/s for the
# relevant dtype) for roofline context. These are *marketing* peaks used only to
# print a "% of peak" hint; absence of an entry is fine.
_GPU_PEAKS: Dict[str, Dict[str, float]] = {
    # name-substring : {bw_gbps, tf16 (fp16/bf16 tensor-core dense, no sparsity)}
    "a100": {"bw": 1935.0, "tf16": 312.0, "tf8": 624.0},
    "h100": {"bw": 3350.0, "tf16": 989.0, "tf8": 1979.0},
    "l4":   {"bw": 300.0,  "tf16": 121.0, "tf8": 242.0},
    "l40":  {"bw": 864.0,  "tf16": 181.0, "tf8": 362.0},
    "t4":   {"bw": 320.0,  "tf16": 65.0,  "tf8": 0.0},
    "v100": {"bw": 900.0,  "tf16": 125.0, "tf8": 0.0},
    "4090": {"bw": 1008.0, "tf16": 165.0, "tf8": 330.0},
    "a10":  {"bw": 600.0,  "tf16": 125.0, "tf8": 250.0},
}


def _peaks_for(gpu: GpuInfo) -> Dict[str, float]:
    n = gpu.name.lower().replace(" ", "")
    for key, peaks in _GPU_PEAKS.items():
        if key in n:
            return peaks
    return {}


def detect_gpu() -> GpuInfo:
    """Detect the GPU via ks_get_device_properties, then torch, then nvidia-smi."""
    # 1) kernel-set's own device query (vendor-neutral, matches what kernels see).
    if _HAVE_KS:
        try:
            if ks.device_count() > 0:
                p = ks.get_device_properties(0)
                return GpuInfo(
                    name=p.name,
                    compute_major=p.compute_major,
                    compute_minor=p.compute_minor,
                    sm_count=p.multiprocessor_count,
                    total_mem_gb=p.total_global_memory / (1024 ** 3),
                    supports_bf16=p.supports_bf16,
                    supports_fp8=p.supports_fp8,
                    supports_tf32=p.supports_tf32,
                    source="kernel_set",
                )
        except Exception:
            pass

    # 2) torch.
    if _HAVE_TORCH and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            major, minor = props.major, props.minor
            return GpuInfo(
                name=props.name,
                compute_major=major,
                compute_minor=minor,
                sm_count=getattr(props, "multi_processor_count", 0),
                total_mem_gb=props.total_memory / (1024 ** 3),
                supports_bf16=(major >= 8),
                supports_fp8=(major >= 9 or (major == 8 and minor == 9)),
                supports_tf32=(major >= 8),
                source="torch",
            )
        except Exception:
            pass

    # 3) nvidia-smi (no GPU libraries needed).
    info = _detect_via_nvidia_smi()
    if info is not None:
        return info

    return GpuInfo(source="none")


def _detect_via_nvidia_smi() -> Optional[GpuInfo]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None
    if not out:
        return None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    name = parts[0] if parts else "unknown"
    major = minor = 0
    if len(parts) >= 2 and "." in parts[1]:
        try:
            major, minor = (int(x) for x in parts[1].split("."))
        except Exception:
            pass
    mem_gb = 0.0
    if len(parts) >= 3:
        try:
            mem_gb = float(parts[2]) / 1024.0   # MiB -> GiB
        except Exception:
            pass
    return GpuInfo(
        name=name,
        compute_major=major,
        compute_minor=minor,
        total_mem_gb=mem_gb,
        supports_bf16=(major >= 8),
        supports_fp8=(major >= 9 or (major == 8 and minor == 9)),
        supports_tf32=(major >= 8),
        source="nvidia-smi",
    )


# --------------------------------------------------------------------------- #
# dtype helpers
# --------------------------------------------------------------------------- #
_DTYPE_ALIASES = {
    "fp16": "float16", "f16": "float16", "half": "float16", "float16": "float16",
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "fp32": "float32", "f32": "float32", "float32": "float32",
}


def torch_dtype(name: str):
    canon = _DTYPE_ALIASES.get(name.lower())
    if canon is None:
        raise ValueError(f"unknown dtype {name!r}; choose from fp16/bf16/fp32")
    return getattr(torch, canon)


def dtype_bytes(dt) -> int:
    return torch.finfo(dt).bits // 8


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def time_cuda(fn: Callable[[], None], warmup: int, iters: int) -> float:
    """Median per-call latency in microseconds, timed with CUDA events.

    `fn` must enqueue exactly one logical op (no host sync inside). We time each
    iteration individually so we can take a median (robust to clock/boost jitter
    and the occasional context switch).
    """
    if _HAVE_TORCH and torch.cuda.is_available():
        for _ in range(max(1, warmup)):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        return statistics.median(times_ms) * 1e3   # ms -> us

    # CPU wall-clock fallback (no torch CUDA): coarse but functional.
    for _ in range(max(1, warmup)):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    op: str
    shape: str
    dtype: str
    ks_us: float = float("nan")
    ref_us: float = float("nan")
    gbps: float = float("nan")
    tflops: float = float("nan")
    rel_err: float = float("nan")
    speedup: float = float("nan")
    status: str = "ok"            # "ok", "skip", or "error: ..."
    note: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# Benchmark registry
# --------------------------------------------------------------------------- #
# Each benchmark is a function (ctx) -> Result. `ctx` bundles the run config
# (dtype, gpu, iters, warmup) and a couple of helpers. Benchmarks are grouped
# by op-category name (the argparse --ops selector). A category may register
# several shapes.
@dataclass
class Ctx:
    dt: object                       # torch dtype
    dtype_name: str
    gpu: GpuInfo
    warmup: int
    iters: int
    device: str = "cuda"

    def rand(self, *shape, dtype=None):
        d = self.dt if dtype is None else dtype
        if d in (torch.int8, torch.int32, torch.int64):
            return torch.randint(-8, 8, shape, device=self.device, dtype=d)
        return torch.randn(*shape, device=self.device, dtype=d)

    def empty(self, *shape, dtype=None):
        return torch.empty(*shape, device=self.device,
                           dtype=self.dt if dtype is None else dtype)


# op-category -> list of (shape_label, builder). builder(ctx) -> Result
_REGISTRY: "Dict[str, List[Tuple[str, Callable[[Ctx], Result]]]]" = {}


def register(op: str, label: str):
    def deco(fn):
        _REGISTRY.setdefault(op, []).append((label, fn))
        return fn
    return deco


def rel_err(a, b) -> float:
    """Max relative error between two tensors (computed in fp32)."""
    a = a.detach().float()
    b = b.detach().float()
    denom = b.abs().max().clamp_min(1e-6)
    return (a - b).abs().max().item() / denom.item()


# ------------------------------- NORM -------------------------------------- #
# Representative hidden sizes across Llama-3-8B (4096), 70B (8192),
# Mistral-7B (4096), Qwen2-7B (3584), and a decode-vs-prefill row count.
_NORM_SHAPES = [
    ("rows=4096,hidden=4096", 4096, 4096),
    ("rows=8192,hidden=8192", 8192, 8192),
    ("rows=2048,hidden=3584", 2048, 3584),
    ("rows=1,hidden=4096",    1,    4096),   # decode (single token)
]


def _bench_rmsnorm(ctx: Ctx, rows: int, hidden: int) -> Result:
    r = Result("rmsnorm", f"rows={rows},hidden={hidden}", ctx.dtype_name)
    x = ctx.rand(rows, hidden)
    w = ctx.rand(hidden)
    out = ctx.empty(rows, hidden)
    eps = 1e-6

    def run():
        ks.norm.rms_norm(out, x, w, eps=eps)

    r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    # bytes: read x + write out (+ read w once, negligible) ~ 2 * rows*hidden
    nbytes = 2 * rows * hidden * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9

    # reference
    def ref_rms(t):
        f = t.float()
        return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(t.dtype) * w

    ref = ref_rms(x)
    ks.norm.rms_norm(out, x, w, eps=eps)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    r.ref_us = time_cuda(lambda: ref_rms(x), ctx.warmup, ctx.iters)
    r.speedup = r.ref_us / r.ks_us
    return r


def _bench_layernorm(ctx: Ctx, rows: int, hidden: int) -> Result:
    r = Result("layernorm", f"rows={rows},hidden={hidden}", ctx.dtype_name)
    x = ctx.rand(rows, hidden)
    w = ctx.rand(hidden)
    b = ctx.rand(hidden)
    out = ctx.empty(rows, hidden)
    eps = 1e-5

    def run():
        ks.norm.layer_norm(out, x, w, b, eps=eps)

    r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    nbytes = 2 * rows * hidden * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9

    ref = torch.nn.functional.layer_norm(
        x.float(), (hidden,), w.float(), b.float(), eps).to(ctx.dt)
    ks.norm.layer_norm(out, x, w, b, eps=eps)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    r.ref_us = time_cuda(
        lambda: torch.nn.functional.layer_norm(x, (hidden,), w, b, eps),
        ctx.warmup, ctx.iters)
    r.speedup = r.ref_us / r.ks_us
    return r


for _lbl, _rows, _hid in _NORM_SHAPES:
    register("rmsnorm", _lbl)(
        (lambda rows, hid: lambda c: _bench_rmsnorm(c, rows, hid))(_rows, _hid))
    register("layernorm", _lbl)(
        (lambda rows, hid: lambda c: _bench_layernorm(c, rows, hid))(_rows, _hid))


# ----------------------------- ACTIVATION ---------------------------------- #
# SwiGLU intermediate sizes: Llama-3-8B (14336), Mistral (14336), 70B (28672).
_SWIGLU_SHAPES = [
    ("rows=4096,inter=14336", 4096, 14336),
    ("rows=2048,inter=28672", 2048, 28672),
    ("rows=1,inter=14336",    1,    14336),   # decode
]


def _bench_swiglu(ctx: Ctx, rows: int, inter: int) -> Result:
    r = Result("swiglu", f"rows={rows},inter={inter}", ctx.dtype_name)
    gate = ctx.rand(rows, inter)
    up = ctx.rand(rows, inter)
    out = ctx.empty(rows, inter)

    def run():
        ks.activation.swiglu(out, gate, up)

    r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    # read gate + read up + write out = 3 * rows*inter
    nbytes = 3 * rows * inter * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9

    ref = (torch.nn.functional.silu(gate.float()) * up.float()).to(ctx.dt)
    ks.activation.swiglu(out, gate, up)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    r.ref_us = time_cuda(
        lambda: torch.nn.functional.silu(gate) * up, ctx.warmup, ctx.iters)
    r.speedup = r.ref_us / r.ks_us
    return r


for _lbl, _rows, _inter in _SWIGLU_SHAPES:
    register("swiglu", _lbl)(
        (lambda rows, inter: lambda c: _bench_swiglu(c, rows, inter))(_rows, _inter))


# -------------------------------- ROPE ------------------------------------- #
# [num_tokens, heads, head_dim]; Llama-3-8B: 32 q heads, 8 kv heads, head_dim 128.
_ROPE_SHAPES = [
    ("tokens=4096,qh=32,kvh=8,hd=128", 4096, 32, 8, 128),
    ("tokens=1,qh=32,kvh=8,hd=128",    1,    32, 8, 128),   # decode
]


def _ref_rope_neox(x, cos, sin):
    # x: [tokens, heads, hd]; cos/sin: [tokens, hd/2]
    hd = x.shape[-1]
    half = hd // 2
    x1 = x[..., :half].float()
    x2 = x[..., half:].float()
    c = cos.float().unsqueeze(1)   # [tokens,1,hd/2]
    s = sin.float().unsqueeze(1)
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.cat([o1, o2], dim=-1).to(x.dtype)


def _bench_rope(ctx: Ctx, tokens: int, qh: int, kvh: int, hd: int) -> Result:
    r = Result("rope", f"tokens={tokens},qh={qh},kvh={kvh},hd={hd}", ctx.dtype_name)
    q = ctx.rand(tokens, qh, hd)
    k = ctx.rand(tokens, kvh, hd)
    cos = ctx.rand(tokens, hd // 2)
    sin = ctx.rand(tokens, hd // 2)
    q0 = q.clone()
    k0 = k.clone()

    def run():
        # in-place; restore inputs is not needed for timing (idempotent layout)
        ks.rope.rope_inplace(q, k, cos, sin, tokens, qh, kvh, hd)

    r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    nbytes = 2 * (tokens * qh * hd + tokens * kvh * hd) * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9

    # correctness on a fresh copy
    qc, kc = q0.clone(), k0.clone()
    ref_q = _ref_rope_neox(qc, cos, sin)
    ks.rope.rope_inplace(qc, kc, cos, sin, tokens, qh, kvh, hd)
    torch.cuda.synchronize()
    r.rel_err = rel_err(qc, ref_q)
    r.note = "neox/rotate_half"
    return r


for _lbl, _tk, _qh, _kvh, _hd in _ROPE_SHAPES:
    register("rope", _lbl)(
        (lambda tk, qh, kvh, hd: lambda c: _bench_rope(c, tk, qh, kvh, hd))(
            _tk, _qh, _kvh, _hd))


# ------------------------------ ATTENTION ---------------------------------- #
# Prefill: dense flash attention. Decode: paged attention.
_ATTN_PREFILL_SHAPES = [
    # (batch, seqlen, qheads, kvheads, head_dim)
    ("b=1,seq=2048,qh=32,kvh=8,hd=128", 1, 2048, 32, 8, 128),
    ("b=4,seq=1024,qh=32,kvh=8,hd=128", 4, 1024, 32, 8, 128),
]
_ATTN_DECODE_SHAPES = [
    # (num_seqs, ctx_len, qheads, kvheads, head_dim, block_size)
    ("seqs=64,ctx=2048,qh=32,kvh=8,hd=128", 64, 2048, 32, 8, 128, 16),
    ("seqs=256,ctx=1024,qh=32,kvh=8,hd=128", 256, 1024, 32, 8, 128, 16),
]


def _bench_attn_prefill(ctx: Ctx, b, seq, qh, kvh, hd) -> Result:
    r = Result("attention_prefill",
               f"b={b},seq={seq},qh={qh},kvh={kvh},hd={hd}", ctx.dtype_name)
    q = ctx.rand(b, seq, qh, hd)
    k = ctx.rand(b, seq, kvh, hd)
    v = ctx.rand(b, seq, kvh, hd)
    out = ctx.empty(b, seq, qh, hd)
    scale = 1.0 / math.sqrt(hd)

    def run():
        ks.attention.flash_attn(out, q, k, v, b, seq, seq, qh, kvh, hd,
                                softmax_scale=scale, causal=True)

    r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    # causal attention FLOPs ~ 2 * (QK^T + softmax*V) * 0.5 (causal)
    # ~ 4 * b * qh * seq^2 * hd  (2 matmuls, factor 2 for MAC), halved for causal
    flops = 4.0 * b * qh * (seq ** 2) * hd * 0.5
    r.tflops = flops / (r.ks_us * 1e-6) / 1e12

    # reference: torch SDPA with GQA expansion
    ref = _ref_sdpa(q, k, v, qh, kvh, causal=True, scale=scale)
    ks.attention.flash_attn(out, q, k, v, b, seq, seq, qh, kvh, hd,
                            softmax_scale=scale, causal=True)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    r.ref_us = time_cuda(
        lambda: _ref_sdpa(q, k, v, qh, kvh, causal=True, scale=scale),
        ctx.warmup, ctx.iters)
    r.speedup = r.ref_us / r.ks_us
    return r


def _ref_sdpa(q, k, v, qh, kvh, causal, scale):
    # q: [b, s, qh, hd]; k/v: [b, s, kvh, hd] -> [b, h, s, hd]
    qt = q.transpose(1, 2).float()
    kt = k.transpose(1, 2).float()
    vt = v.transpose(1, 2).float()
    if kvh != qh:
        rep = qh // kvh
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)
    o = torch.nn.functional.scaled_dot_product_attention(
        qt, kt, vt, is_causal=causal, scale=scale)
    return o.transpose(1, 2).to(q.dtype)


def _bench_attn_decode(ctx: Ctx, num_seqs, ctx_len, qh, kvh, hd, block) -> Result:
    r = Result("attention_decode",
               f"seqs={num_seqs},ctx={ctx_len},qh={qh},kvh={kvh},hd={hd}",
               ctx.dtype_name)
    blocks_per_seq = (ctx_len + block - 1) // block
    total_blocks = num_seqs * blocks_per_seq
    q = ctx.rand(num_seqs, qh, hd)
    k_cache = ctx.rand(total_blocks, kvh, block, hd)
    v_cache = ctx.rand(total_blocks, kvh, block, hd)
    block_tables = torch.arange(total_blocks, device="cuda", dtype=torch.int32)\
        .reshape(num_seqs, blocks_per_seq)
    seq_lens = torch.full((num_seqs,), ctx_len, device="cuda", dtype=torch.int32)
    out = ctx.empty(num_seqs, qh, hd)
    scale = 1.0 / math.sqrt(hd)

    def run():
        ks.attention.paged_attn_decode(
            out, q, k_cache, v_cache, block_tables, seq_lens,
            num_seqs, qh, kvh, hd, block, blocks_per_seq, softmax_scale=scale)

    r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    # decode is memory-bound: read whole KV cache each step.
    kv_bytes = 2 * num_seqs * kvh * ctx_len * hd * dtype_bytes(ctx.dt)
    r.gbps = kv_bytes / (r.ks_us * 1e-6) / 1e9
    r.note = "memory-bound (KV read)"
    return r


for _e in _ATTN_PREFILL_SHAPES:
    register("attention", _e[0])(
        (lambda a: lambda c: _bench_attn_prefill(c, *a[1:]))(_e))
for _e in _ATTN_DECODE_SHAPES:
    register("attention", _e[0])(
        (lambda a: lambda c: _bench_attn_decode(c, *a[1:]))(_e))


# -------------------------------- GEMM ------------------------------------- #
# (M, N, K) for QKV/MLP-ish projections at typical batch*seq token counts.
_GEMM_SHAPES = [
    ("M=4096,N=4096,K=4096", 4096, 4096, 4096),
    ("M=8192,N=8192,K=8192", 8192, 8192, 8192),
    ("M=4096,N=14336,K=4096", 4096, 14336, 4096),   # MLP up-proj
    ("M=2048,N=4096,K=14336", 2048, 4096, 14336),   # MLP down-proj
]


def _bench_gemm(ctx: Ctx, m, n, k) -> Result:
    r = Result(f"gemm_{ctx.dtype_name}", f"M={m},N={n},K={k}", ctx.dtype_name)
    a = ctx.rand(m, k)
    b = ctx.rand(k, n)
    c = ctx.empty(m, n)

    def run():
        ks.gemm.gemm(c, a, b, m=m, n=n, k=k)

    r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    flops = 2.0 * m * n * k
    r.tflops = flops / (r.ks_us * 1e-6) / 1e12

    ref = (a.float() @ b.float()).to(ctx.dt)
    ks.gemm.gemm(c, a, b, m=m, n=n, k=k)
    torch.cuda.synchronize()
    r.rel_err = rel_err(c, ref)
    r.ref_us = time_cuda(lambda: a @ b, ctx.warmup, ctx.iters)
    r.speedup = r.ref_us / r.ks_us
    return r


for _lbl, _m, _n, _k in _GEMM_SHAPES:
    # registered under both "gemm" and the dtype-specific selectors below
    register("gemm", _lbl)(
        (lambda m, n, k: lambda c: _bench_gemm(c, m, n, k))(_m, _n, _k))


# ------------------------------- W8A8 GEMM --------------------------------- #
def _bench_w8a8(ctx: Ctx, m, n, k) -> Result:
    r = Result("w8a8", f"M={m},N={n},K={k}", ctx.dtype_name)
    a = torch.randint(-127, 127, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 127, (k, n), device="cuda", dtype=torch.int8)
    a_scale = torch.rand(m, device="cuda", dtype=torch.float32) * 0.02 + 0.01
    b_scale = torch.rand(n, device="cuda", dtype=torch.float32) * 0.02 + 0.01
    out = ctx.empty(m, n)
    out_dt = ks.dtype_to_ks(ctx.dt)

    def run():
        ks.gemm.gemm_w8a8(out, a, b, a_scale, b_scale, m=m, n=n, k=k,
                          out_dtype=out_dt)

    r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    flops = 2.0 * m * n * k
    r.tflops = flops / (r.ks_us * 1e-6) / 1e12

    # reference: int32 matmul then per-token x per-channel dequant
    acc = (a.int() @ b.int()).float()
    ref = (acc * a_scale.unsqueeze(1) * b_scale.unsqueeze(0)).to(ctx.dt)
    ks.gemm.gemm_w8a8(out, a, b, a_scale, b_scale, m=m, n=n, k=k, out_dtype=out_dt)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    r.note = "int8xint8->acc int32, per-token/per-channel dequant"
    return r


for _lbl, _m, _n, _k in _GEMM_SHAPES[:3]:
    register("w8a8", _lbl)(
        (lambda m, n, k: lambda c: _bench_w8a8(c, m, n, k))(_m, _n, _k))


# ------------------------------- W4A16 GEMM -------------------------------- #
def _bench_w4a16(ctx: Ctx, m, n, k, group_size=128) -> Result:
    r = Result("w4a16", f"M={m},N={n},K={k},g={group_size}", ctx.dtype_name)
    if ctx.dt not in (torch.float16, torch.bfloat16):
        r.status = "skip"
        r.note = "w4a16 activations must be fp16/bf16"
        return r
    a = ctx.rand(m, k)
    # b_packed: two int4 per byte -> [K, N/2] uint8 (N must be even)
    b_packed = torch.randint(0, 255, (k, n // 2), device="cuda", dtype=torch.uint8)
    n_groups = (k + group_size - 1) // group_size
    scales = ctx.rand(n_groups, n) * 0.02
    zeros = ctx.rand(n_groups, n)
    out = ctx.empty(m, n)

    def run():
        ks.gemm.gemm_w4a16(out, a, b_packed, scales, zeros,
                           m=m, n=n, k=k, group_size=group_size)

    try:
        r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    except Exception as exc:
        r.status = f"error: {type(exc).__name__}: {exc}"
        return r
    flops = 2.0 * m * n * k
    r.tflops = flops / (r.ks_us * 1e-6) / 1e12
    r.note = "no portable torch int4 ref; throughput only"
    return r


for _lbl, _m, _n, _k in _GEMM_SHAPES[:3]:
    register("w4a16", _lbl)(
        (lambda m, n, k: lambda c: _bench_w4a16(c, m, n, k))(_m, _n, _k))


# -------------------------------- MoE -------------------------------------- #
# DeepSeek/Mixtral-ish: tokens routed to top-k of E experts, grouped GEMM.
_MOE_SHAPES = [
    # (num_tokens, hidden, inter, num_experts, top_k)
    ("tokens=4096,h=4096,inter=14336,E=8,k=2", 4096, 4096, 14336, 8, 2),
    ("tokens=2048,h=2048,inter=1408,E=64,k=6", 2048, 2048, 1408, 64, 6),
]


def _bench_moe_gate(ctx: Ctx, num_tokens, hidden, inter, E, k) -> Result:
    r = Result("moe_gate", f"tokens={num_tokens},E={E},k={k}", ctx.dtype_name)
    logits = ctx.rand(num_tokens, E)
    weights = torch.empty(num_tokens, k, device="cuda", dtype=torch.float32)
    indices = torch.empty(num_tokens, k, device="cuda", dtype=torch.int32)

    def run():
        ks.moe.gate_softmax_topk(weights, indices, logits,
                                 num_tokens, E, k, renormalize=True)

    r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)

    # reference: softmax + topk + renormalize
    probs = torch.softmax(logits.float(), dim=-1)
    ref_w, ref_i = torch.topk(probs, k, dim=-1)
    ref_w = ref_w / ref_w.sum(-1, keepdim=True)
    ks.moe.gate_softmax_topk(weights, indices, logits, num_tokens, E, k,
                             renormalize=True)
    torch.cuda.synchronize()
    # match by sorting weights (index order may differ on ties)
    w_sorted = torch.sort(weights.float(), dim=-1, descending=True).values
    rw_sorted = torch.sort(ref_w, dim=-1, descending=True).values
    r.rel_err = rel_err(w_sorted, rw_sorted)
    r.note = "softmax top-k gating"
    return r


def _bench_moe_grouped_gemm(ctx: Ctx, num_tokens, hidden, inter, E, k) -> Result:
    r = Result("moe_grouped_gemm",
               f"tokens={num_tokens},h={hidden},E={E},k={k}", ctx.dtype_name)
    total_rows = num_tokens * k
    a = ctx.rand(total_rows, hidden)
    b = ctx.rand(E, hidden, inter)
    c = ctx.empty(total_rows, inter)
    # even split across experts (CSR offsets)
    per = total_rows // E
    offsets = torch.tensor(
        [min(i * per, total_rows) for i in range(E)] + [total_rows],
        device="cuda", dtype=torch.int32)

    def run():
        ks.moe.grouped_gemm(c, a, b, offsets, E, total_rows, inter, hidden)

    try:
        r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    except Exception as exc:
        r.status = f"error: {type(exc).__name__}: {exc}"
        return r
    flops = 2.0 * total_rows * inter * hidden
    r.tflops = flops / (r.ks_us * 1e-6) / 1e12
    r.note = "grouped GEMM over experts"
    return r


for _e in _MOE_SHAPES:
    register("moe", _e[0] + ",gate")(
        (lambda a: lambda c: _bench_moe_gate(c, *a[1:]))(_e))
    register("moe", _e[0] + ",ggemm")(
        (lambda a: lambda c: _bench_moe_grouped_gemm(c, *a[1:]))(_e))


# ------------------------------ SAMPLING ----------------------------------- #
_SAMPLING_SHAPES = [
    ("seqs=256,vocab=32000", 256, 32000),     # Llama vocab
    ("seqs=64,vocab=128256", 64, 128256),     # Llama-3 vocab
]


def _bench_sampling(ctx: Ctx, num_seqs, vocab) -> Result:
    r = Result("sampling", f"seqs={num_seqs},vocab={vocab}", ctx.dtype_name)
    logits = ctx.rand(num_seqs, vocab)
    tokens = torch.empty(num_seqs, device="cuda", dtype=torch.int32)

    def run():
        ks.sampling.argmax(tokens, logits, num_seqs, vocab)

    r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    nbytes = num_seqs * vocab * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9

    ref = logits.float().argmax(-1).to(torch.int32)
    ks.sampling.argmax(tokens, logits, num_seqs, vocab)
    torch.cuda.synchronize()
    match = (tokens == ref).float().mean().item()
    r.rel_err = 1.0 - match   # fraction mismatched (0 == perfect)
    r.ref_us = time_cuda(lambda: logits.argmax(-1), ctx.warmup, ctx.iters)
    r.speedup = r.ref_us / r.ks_us
    r.note = "argmax (greedy); rel_err = mismatch fraction"
    return r


for _lbl, _s, _v in _SAMPLING_SHAPES:
    register("sampling", _lbl)(
        (lambda s, v: lambda c: _bench_sampling(c, s, v))(_s, _v))


# ---------------------------- CROSS ENTROPY -------------------------------- #
_CE_SHAPES = [
    ("tokens=4096,vocab=32000", 4096, 32000),
    ("tokens=8192,vocab=128256", 8192, 128256),
]


def _bench_cross_entropy(ctx: Ctx, num_tokens, vocab) -> Result:
    r = Result("cross_entropy", f"tokens={num_tokens},vocab={vocab}", ctx.dtype_name)
    logits = ctx.rand(num_tokens, vocab)
    targets = torch.randint(0, vocab, (num_tokens,), device="cuda", dtype=torch.int64)
    losses = torch.empty(num_tokens, device="cuda", dtype=torch.float32)
    grad = torch.empty_like(logits)

    def run():
        ks.loss.cross_entropy(losses, grad, logits, targets,
                              num_tokens, vocab, ignore_index=-100)

    r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    # read logits + write grad ~ 2 * tokens*vocab
    nbytes = 2 * num_tokens * vocab * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9

    ref = torch.nn.functional.cross_entropy(
        logits.float(), targets, reduction="none")
    ks.loss.cross_entropy(losses, grad, logits, targets, num_tokens, vocab,
                          ignore_index=-100)
    torch.cuda.synchronize()
    r.rel_err = rel_err(losses, ref)
    r.ref_us = time_cuda(
        lambda: torch.nn.functional.cross_entropy(
            logits, targets, reduction="none"),
        ctx.warmup, ctx.iters)
    r.speedup = r.ref_us / r.ks_us
    r.note = "fused fwd+bwd; rel_err on forward loss"
    return r


for _lbl, _t, _v in _CE_SHAPES:
    register("cross_entropy", _lbl)(
        (lambda t, v: lambda c: _bench_cross_entropy(c, t, v))(_t, _v))


# -------------------------------- ADAMW ------------------------------------ #
_ADAMW_SHAPES = [
    ("n=4096*4096", 4096 * 4096),
    ("n=8192*8192", 8192 * 8192),
]


def _bench_adamw(ctx: Ctx, n) -> Result:
    r = Result("adamw", f"n={n}", ctx.dtype_name)
    param = ctx.rand(n)
    grad = ctx.rand(n)
    exp_avg = torch.zeros(n, device="cuda", dtype=torch.float32)
    exp_avg_sq = torch.zeros(n, device="cuda", dtype=torch.float32)

    def run():
        ks.optimizer.adamw(param, grad, exp_avg, exp_avg_sq,
                           lr=1e-3, step=1)

    r.ks_us = time_cuda(run, ctx.warmup, ctx.iters)
    # read+write param (dt) + grad (dt) + 2 fp32 states r/w
    nbytes = (3 * n * dtype_bytes(ctx.dt)) + (4 * n * 4)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9

    # reference: a single fused AdamW step on fresh copies.
    p2 = param.detach().clone().float()
    g2 = grad.detach().clone().float()
    m2 = torch.zeros_like(p2)
    v2 = torch.zeros_like(p2)
    lr, b1, b2, eps, wd, step = 1e-3, 0.9, 0.999, 1e-8, 0.0, 1
    m2 = b1 * m2 + (1 - b1) * g2
    v2 = b2 * v2 + (1 - b2) * g2 * g2
    mhat = m2 / (1 - b1 ** step)
    vhat = v2 / (1 - b2 ** step)
    p2 = p2 - lr * (mhat / (vhat.sqrt() + eps) + wd * p2)
    pk = param.detach().clone()
    ek = torch.zeros(n, device="cuda", dtype=torch.float32)
    sk = torch.zeros(n, device="cuda", dtype=torch.float32)
    ks.optimizer.adamw(pk, grad, ek, sk, lr=1e-3, step=1)
    torch.cuda.synchronize()
    r.rel_err = rel_err(pk, p2.to(ctx.dt))
    r.note = "fused AdamW step; memory-bound"
    return r


for _lbl, _n in _ADAMW_SHAPES:
    register("adamw", _lbl)(
        (lambda n: lambda c: _bench_adamw(c, n))(_n))


# --------------------------------------------------------------------------- #
# Op aliases: the user-facing op names map to one or more registry categories.
# This lets `--ops gemm` cover the fp16/bf16 selector and keeps the help tidy.
# --------------------------------------------------------------------------- #
ALL_OPS = [
    "rmsnorm", "layernorm", "swiglu", "rope", "attention",
    "gemm", "w8a8", "w4a16", "moe", "sampling", "cross_entropy", "adamw",
]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_benchmarks(ops: Sequence[str], ctx: Ctx,
                   shape_filter: Optional[str]) -> List[Result]:
    results: List[Result] = []
    for op in ops:
        cases = _REGISTRY.get(op, [])
        if not cases:
            print(f"  [warn] no benchmark registered for op {op!r}", file=sys.stderr)
            continue
        for label, fn in cases:
            if shape_filter and shape_filter not in label:
                continue
            try:
                res = fn(ctx)
            except Exception as exc:
                res = Result(op, label, ctx.dtype_name,
                             status=f"error: {type(exc).__name__}: {exc}")
            results.append(res)
            _print_progress(res)
    return results


def _print_progress(r: Result) -> None:
    if r.status == "skip":
        print(f"  [skip] {r.op:<18} {r.shape:<30} ({r.note})", file=sys.stderr)
        return
    if r.status.startswith("error"):
        print(f"  [ERR ] {r.op:<18} {r.shape:<30} {r.status}", file=sys.stderr)
        return
    perf = ""
    if not math.isnan(r.tflops):
        perf = f"{r.tflops:8.1f} TFLOP/s"
    elif not math.isnan(r.gbps):
        perf = f"{r.gbps:8.1f} GB/s"
    extra = ""
    if not math.isnan(r.speedup):
        extra += f"  speedup={r.speedup:5.2f}x"
    if not math.isnan(r.rel_err):
        extra += f"  rel_err={r.rel_err:.2e}"
    print(f"  [ok  ] {r.op:<18} {r.shape:<30} {r.ks_us:9.2f} us  {perf}{extra}",
          file=sys.stderr)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x: float, prec: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{prec}f}"


def render_markdown(results: List[Result], gpu: GpuInfo, cfg: dict) -> str:
    lines: List[str] = []
    lines.append(f"# kernel-set benchmark — {gpu.name}")
    lines.append("")
    peaks = _peaks_for(gpu)
    peak_str = ""
    if peaks:
        peak_str = (f" | peak BW ~{peaks.get('bw', 0):.0f} GB/s"
                    f" | peak fp16 TC ~{peaks.get('tf16', 0):.0f} TFLOP/s")
    lines.append(f"- **GPU**: {gpu.name} (sm_{gpu.sm_arch}, CC {gpu.cc}, "
                 f"{gpu.sm_count} SMs, {gpu.total_mem_gb:.1f} GB){peak_str}")
    lines.append(f"- **detected via**: {gpu.source}")
    lines.append(f"- **dtype**: {cfg['dtype']} | warmup={cfg['warmup']} "
                 f"iters={cfg['iters']}")
    lines.append(f"- **kernel-set**: {cfg.get('ks_version', '?')} "
                 f"(backend {cfg.get('backend', '?')})")
    if cfg.get("torch_version"):
        lines.append(f"- **torch**: {cfg['torch_version']}")
    lines.append(f"- **host**: {platform.platform()}")
    lines.append("")
    lines.append("Latency is the median over the timed iterations (CUDA events). "
                 "`rel_err` is max relative error vs the PyTorch reference; "
                 "`speedup` is ref_us / ks_us.")
    lines.append("")
    lines.append("| op | shape | dtype | ks (us) | ref (us) | GB/s | TFLOP/s | "
                 "rel_err | speedup | notes |")
    lines.append("|---|---|---|--:|--:|--:|--:|--:|--:|---|")
    for r in results:
        if r.status == "skip":
            lines.append(f"| {r.op} | {r.shape} | {r.dtype} | skip | - | - | - "
                         f"| - | - | {r.note or 'skipped'} |")
            continue
        if r.status.startswith("error"):
            lines.append(f"| {r.op} | {r.shape} | {r.dtype} | err | - | - | - "
                         f"| - | - | {r.status} |")
            continue
        lines.append(
            f"| {r.op} | {r.shape} | {r.dtype} | {_fmt(r.ks_us)} | "
            f"{_fmt(r.ref_us)} | {_fmt(r.gbps, 1)} | {_fmt(r.tflops, 1)} | "
            f"{_fmt(r.rel_err, 2) if math.isnan(r.rel_err) else f'{r.rel_err:.2e}'} | "
            f"{_fmt(r.speedup)}{'x' if not math.isnan(r.speedup) else ''} | "
            f"{r.note} |")
    lines.append("")
    return "\n".join(lines)


def render_json(results: List[Result], gpu: GpuInfo, cfg: dict) -> str:
    return json.dumps({
        "gpu": gpu.__dict__,
        "config": cfg,
        "results": [r.to_dict() for r in results],
    }, indent=2)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark kernel-set kernels against PyTorch on the local GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ops", default="all",
                   help="comma-separated op categories (or 'all'). "
                        f"Choices: {','.join(ALL_OPS)}")
    p.add_argument("--dtype", default="fp16",
                   help="element dtype: fp16, bf16, or fp32")
    p.add_argument("--shape", default=None,
                   help="only run shapes whose label contains this substring")
    p.add_argument("--warmup", type=int, default=10, help="warmup launches")
    p.add_argument("--iters", type=int, default=50, help="timed launches")
    p.add_argument("--output", default=None,
                   help="write the report to this file (default: stdout)")
    p.add_argument("--format", default="md", choices=["md", "json"],
                   help="report format")
    p.add_argument("--list-ops", action="store_true",
                   help="list op categories + shapes and exit")
    p.add_argument("--gpu-only", action="store_true",
                   help="print detected GPU info as JSON and exit "
                        "(used by build_and_bench.sh to name the results file)")
    return p.parse_args(argv)


def list_ops() -> None:
    print("Benchmark catalog (op category -> shapes):")
    for op in ALL_OPS:
        cases = _REGISTRY.get(op, [])
        print(f"\n  {op}")
        for label, _ in cases:
            print(f"      - {label}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_ops:
        list_ops()
        return 0

    if args.gpu_only:
        gpu = detect_gpu()
        print(json.dumps(gpu.__dict__))
        return 0

    if not _HAVE_KS:
        print("ERROR: could not import kernel_set.\n"
              f"  {type(_KS_IMPORT_ERROR).__name__}: {_KS_IMPORT_ERROR}\n"
              "  Install the binding (pip install ./bindings/python) and point "
              "KERNEL_SET_LIB at the built libkernel_set.so.", file=sys.stderr)
        return 2

    gpu = detect_gpu()
    print(f"Detected GPU: {gpu.name} (sm_{gpu.sm_arch}, via {gpu.source})",
          file=sys.stderr)

    if not _HAVE_TORCH or not torch.cuda.is_available():
        print("ERROR: torch with CUDA is required to drive the benchmarks "
              "(tensors + CUDA event timing). Install torch and run on a GPU.",
              file=sys.stderr)
        return 2

    # dtype selection + capability guards
    dt = torch_dtype(args.dtype)
    if dt is torch.bfloat16 and not gpu.supports_bf16:
        print(f"WARNING: {gpu.name} reports no bf16 support; results may be "
              "emulated or error.", file=sys.stderr)

    ops = ALL_OPS if args.ops == "all" else [o.strip() for o in args.ops.split(",")]
    unknown = [o for o in ops if o not in _REGISTRY and o not in ALL_OPS]
    if unknown:
        print(f"ERROR: unknown op(s): {unknown}. Known: {ALL_OPS}", file=sys.stderr)
        return 2

    ctx = Ctx(dt=dt, dtype_name=args.dtype, gpu=gpu,
              warmup=args.warmup, iters=args.iters)

    cfg = {
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "ops": ops,
        "ks_version": ks.version() if _HAVE_KS else "?",
        "backend": ks.backend_name() if _HAVE_KS else "?",
        "torch_version": torch.__version__ if _HAVE_TORCH else None,
    }

    print(f"Running {len(ops)} op categories, dtype={args.dtype}, "
          f"warmup={args.warmup}, iters={args.iters}\n", file=sys.stderr)
    results = run_benchmarks(ops, ctx, args.shape)

    if args.format == "json":
        report = render_json(results, gpu, cfg)
    else:
        report = render_markdown(results, gpu, cfg)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"\nWrote report to {args.output}", file=sys.stderr)
    else:
        print(report)

    # surface a nonzero exit if every benchmark errored (helps CI)
    ran = [r for r in results if r.status == "ok"]
    if results and not ran:
        print("ERROR: all benchmarks failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
