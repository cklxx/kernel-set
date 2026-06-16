#!/usr/bin/env python3
"""kernel-set benchmark harness.

Detects the GPU, then benchmarks each kernel category over representative LLM
shapes and reports:

  * latency (microseconds, robust stats over an auto-calibrated number of timed
    launches): median with the min in parentheses, plus p20/p80 spread,
  * achieved memory bandwidth (GB/s) for bandwidth-bound ops, or compute
    throughput (TFLOP/s) for compute-bound ops, each as a % of the SKU's dense
    peak for the active dtype,
  * correctness (relative error) vs the strongest available PyTorch reference,
    *gated before* reporting speed (a kernel that fails tolerance is flagged
    INCORRECT rather than presented as a clean speedup), and
  * the kernel-set / PyTorch speedup vs the fastest baseline.

Timing uses CUDA events (``torch.cuda.Event``) with a budget-derived warmup,
an L2-cache flush between every measured iteration (Triton ``do_bench`` style)
so each launch reads cold from HBM, and an optional CUDA-graph replay path that
amortizes launch overhead for tiny / launch-bound ops.

Methodology spec: ``docs/BENCHMARK_METHODOLOGY.md``.

Examples
--------
    # everything, fp16
    python bench.py --dtype fp16

    # just norm + activation, bf16, fill a 300ms budget, write a markdown report
    python bench.py --ops rmsnorm,layernorm,swiglu --dtype bf16 \
        --target-ms 300 --output results/l4.md --format md

    # launch-bound decode ops via cuda graphs (warm-L2, launch overhead removed)
    python bench.py --ops attention --shape decode --cudagraph --no-l2-flush

    # lock clocks (if permitted) for low-variance numbers
    python bench.py --lock-clocks --dtype bf16

    # list the op categories this harness knows about
    python bench.py --list-ops

Library discovery: the kernel-set shared library is located by the
``kernel_set`` binding via ``KERNEL_SET_LIB`` / ``KERNEL_SET_LIB_DIR`` (see
``bindings/python/README.md``). ``build_and_bench.sh`` sets these for you.
"""

from __future__ import annotations

import argparse
import datetime
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

# Shared shape tables + RoPE reference (single source of truth across the two
# harnesses). Import-safe with no torch. See benchmarks/_bench_common.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bench_common import (  # noqa: E402
    ATTN_PREFILL_SHAPES as _ATTN_PREFILL_SHAPES,
    ATTN_DECODE_SHAPES as _ATTN_DECODE_SHAPES,
    ROPE_SHAPES as _ROPE_SHAPES,
    CE_SHAPES as _CE_SHAPES,
    ELEMENTWISE_SHAPES as _ELEMENTWISE_SHAPES,
    EMBEDDING_SHAPES as _EMBEDDING_SHAPES,
    QUANT_SHAPES as _QUANT_SHAPES,
    DEQUANT_INT4_SHAPES as _DEQUANT_INT4_SHAPES,
    MOE_PERMUTE_SHAPES as _MOE_PERMUTE_SHAPES,
    RESHAPE_CACHE_SHAPES as _RESHAPE_CACHE_SHAPES,
    ATTN_BWD_SHAPES as _ATTN_BWD_SHAPES,
    FLCE_SHAPES as _FLCE_SHAPES,
    OPTIM_SHAPES as _OPTIM_SHAPES,
    LOG_SOFTMAX_SHAPES as _LOG_SOFTMAX_SHAPES,
    ref_rope_neox as _ref_rope_neox,
)

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


# Per-SKU DENSE peak specs for the roofline / %-of-peak columns:
#   bw   = HBM/GDDR bandwidth in GB/s
#   tf16 = dense bf16/fp16 tensor-core TFLOP/s (NO 2x sparsity)
#   tf8  = dense fp8/int8 tensor-core TFLOP/s (NO sparsity); 0 if unsupported
# Numbers are vendor dense peaks (sparsity stripped). SXM/PCIe/NVL and 40/80GB
# variants differ; we disambiguate a few by detected memory size below.
# An achieved value over ~105% of these almost always means warm cache (no L2
# flush) or dead-code elimination — that is the smell the audit calls out.
_GPU_PEAKS: Dict[str, Dict[str, float]] = {
    # name-substring : {bw, tf16, tf8}
    "a100": {"bw": 2039.0, "tf16": 312.0, "tf8": 624.0},   # SXM 80GB; PCIe/40GB adjusted below
    "h200": {"bw": 4800.0, "tf16": 989.0, "tf8": 1979.0},  # SXM 141GB
    "h100": {"bw": 3350.0, "tf16": 989.0, "tf8": 1979.0},  # SXM 80GB; PCIe/NVL adjusted below
    "l40s": {"bw": 864.0,  "tf16": 362.0, "tf8": 733.0},
    "l40":  {"bw": 864.0,  "tf16": 181.0, "tf8": 362.0},
    "l4":   {"bw": 300.0,  "tf16": 121.0, "tf8": 242.0},
    "4090": {"bw": 1008.0, "tf16": 165.0, "tf8": 330.0},   # RTX 4090 (Ada)
    "3090": {"bw": 936.0,  "tf16": 71.0,  "tf8": 0.0},
    "a6000": {"bw": 768.0, "tf16": 155.0, "tf8": 310.0},
    "a10g": {"bw": 600.0,  "tf16": 70.0,  "tf8": 140.0},
    "a10":  {"bw": 600.0,  "tf16": 125.0, "tf8": 250.0},
    "t4":   {"bw": 320.0,  "tf16": 65.0,  "tf8": 130.0},
    "v100": {"bw": 900.0,  "tf16": 125.0, "tf8": 0.0},
}


def _peaks_for(gpu: GpuInfo) -> Dict[str, float]:
    """Dense peaks for the detected SKU, disambiguated by memory size where it
    matters (A100 40 vs 80GB, H100 PCIe vs SXM). Returns {} if unknown."""
    n = gpu.name.lower().replace(" ", "")
    peaks: Dict[str, float] = {}
    for key, p in _GPU_PEAKS.items():
        if key in n:
            peaks = dict(p)
            break
    if not peaks:
        return {}
    mem = gpu.total_mem_gb
    # A100 40GB (HBM2) tops out ~1555 GB/s vs 80GB (HBM2e) ~2039 GB/s.
    if "a100" in n and 0 < mem < 60:
        peaks["bw"] = 1555.0
    # H100 PCIe has lower bandwidth (~2039) than SXM (~3350); NVL ~3900.
    if "h100" in n:
        if "pcie" in n or (0 < mem < 90):
            peaks["bw"] = 2039.0
        elif "nvl" in n:
            peaks["bw"] = 3900.0
    return peaks


def peak_for_metric(gpu: GpuInfo, dtype_name: str, metric: str) -> float:
    """Return the dense peak for the given metric ('bw' GB/s, or 'tflops' for the
    active dtype). 0.0 when unknown so callers can skip the %-of-peak column."""
    peaks = _peaks_for(gpu)
    if not peaks:
        return 0.0
    if metric == "bw":
        return peaks.get("bw", 0.0)
    # compute peak depends on dtype class
    name = (dtype_name or "").lower()
    if name in ("int8", "fp8", "w8a8", "e4m3", "e5m2"):
        return peaks.get("tf8", 0.0)
    # fp16/bf16 (and fp32-via-TF32 -> use the same TC dense peak as a ceiling)
    return peaks.get("tf16", 0.0)


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
# Clock control + environment / repro metadata
# --------------------------------------------------------------------------- #
def _nvsmi_query(field: str) -> Optional[str]:
    """Query a single nvidia-smi --query-gpu field for device 0. None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out.splitlines()[0].strip() if out else None
    except Exception:
        return None


def query_clocks() -> Dict[str, object]:
    """Current SM + memory clocks (MHz) and active throttle reasons via
    nvidia-smi. Always best-effort; missing values are None."""
    info: Dict[str, object] = {"sm_mhz": None, "mem_mhz": None,
                               "throttle": None}
    sm = _nvsmi_query("clocks.sm")
    mem = _nvsmi_query("clocks.mem")
    try:
        info["sm_mhz"] = int(float(sm)) if sm not in (None, "", "[N/A]") else None
    except Exception:
        info["sm_mhz"] = None
    try:
        info["mem_mhz"] = int(float(mem)) if mem not in (None, "", "[N/A]") else None
    except Exception:
        info["mem_mhz"] = None
    thr = _nvsmi_query("clocks_throttle_reasons.active")
    info["throttle"] = thr
    return info


def lock_clocks() -> Dict[str, object]:
    """Attempt to enable persistence mode and lock GPU clocks to a mid value via
    nvidia-smi. Returns a status dict; NEVER raises and NEVER fails the run if
    locking is not permitted (e.g. Colab) — it just reports that it could not."""
    status: Dict[str, object] = {"requested": True, "locked": False,
                                 "lock_mhz": None, "detail": ""}

    def _run(cmd: List[str]) -> Tuple[bool, str]:
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=20)
            return p.returncode == 0, p.stdout.decode(errors="ignore").strip()
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    # persistence mode (often needs root)
    _run(["nvidia-smi", "-pm", "1"])

    # pick a sustainable graphics clock: the median of supported SM clocks.
    lock_mhz = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
            stderr=subprocess.DEVNULL).decode(errors="ignore")
        gr = []
        for line in out.splitlines():
            ls = line.strip()
            if ls.startswith("Graphics") and "MHz" in ls:
                try:
                    gr.append(int(ls.split(":")[1].strip().split()[0]))
                except Exception:
                    pass
        if gr:
            gr = sorted(set(gr))
            lock_mhz = gr[len(gr) // 2]   # mid value (sustainable, not max-boost)
    except Exception:
        lock_mhz = None

    if lock_mhz is not None:
        ok, detail = _run(["nvidia-smi", f"--lock-gpu-clocks={lock_mhz}"])
        status["lock_mhz"] = lock_mhz
        status["locked"] = ok
        status["detail"] = detail or ("locked" if ok else "lock not permitted")
    else:
        status["detail"] = "could not read SUPPORTED_CLOCKS; not locked"
    return status


def reset_clocks() -> None:
    """Best-effort reset of any locked clocks. Never raises."""
    for cmd in (["nvidia-smi", "--reset-gpu-clocks"],
                ["nvidia-smi", "--reset-memory-clocks"]):
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20)
        except Exception:
            pass


def collect_env() -> Dict[str, object]:
    """Driver / CUDA / cuDNN / TF32 / ECC metadata for the repro header.
    All fields best-effort; missing ones are None."""
    env: Dict[str, object] = {}
    env["driver"] = _nvsmi_query("driver_version")
    env["power_limit_w"] = _nvsmi_query("power.limit")
    env["ecc"] = _nvsmi_query("ecc.mode.current")
    if _HAVE_TORCH:
        env["torch_cuda"] = getattr(torch.version, "cuda", None)
        try:
            env["cudnn"] = torch.backends.cudnn.version()
        except Exception:
            env["cudnn"] = None
    else:
        env["torch_cuda"] = None
        env["cudnn"] = None
    # toolkit nvcc (if present on PATH)
    try:
        out = subprocess.check_output(["nvcc", "--version"],
                                      stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if "release" in line:
                env["nvcc"] = line.strip()
                break
    except Exception:
        env["nvcc"] = None
    return env


def git_commit() -> Optional[str]:
    """Short git commit hash of the harness, or None."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.check_output(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
        return out or None
    except Exception:
        return None


def apply_tf32(allow: bool) -> Dict[str, object]:
    """Pin TF32 settings explicitly (BP-08) and return what was set so it can be
    reported. allow=True enables TF32 for matmul+cudnn (fp16/bf16 paths are
    unaffected; this governs fp32 GEMM internal precision)."""
    info: Dict[str, object] = {"matmul_allow_tf32": None,
                               "cudnn_allow_tf32": None,
                               "float32_matmul_precision": None}
    if not _HAVE_TORCH:
        return info
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow)
        torch.backends.cudnn.allow_tf32 = bool(allow)
        torch.set_float32_matmul_precision("high" if allow else "highest")
        info["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        info["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
        info["float32_matmul_precision"] = "high" if allow else "highest"
    except Exception:
        pass
    return info


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
@dataclass
class Timing:
    """Per-call latency distribution in microseconds, plus how it was measured."""
    median_us: float = float("nan")
    min_us: float = float("nan")
    p20_us: float = float("nan")
    p80_us: float = float("nan")
    iters: int = 0
    method: str = "event"   # "event" (flushed or warm eager) or "cudagraph"

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float], method: str) -> "Timing":
        us = sorted(s * 1e3 for s in samples_ms)   # ms -> us, sorted
        if not us:
            return cls(method=method)

        def _pct(p: float) -> float:
            if len(us) == 1:
                return us[0]
            idx = p * (len(us) - 1)
            lo = int(math.floor(idx))
            hi = int(math.ceil(idx))
            if lo == hi:
                return us[lo]
            return us[lo] + (us[hi] - us[lo]) * (idx - lo)

        return cls(
            median_us=statistics.median(us),
            min_us=us[0],
            p20_us=_pct(0.20),
            p80_us=_pct(0.80),
            iters=len(us),
            method=method,
        )


# Global timing knobs, set once in main() from the CLI. Kept as module state so
# the per-op bench functions keep their existing (ctx) signature.
@dataclass
class TimingConfig:
    l2_flush: bool = True
    cudagraph: bool = False
    target_ms: float = 200.0
    iters: Optional[int] = None       # explicit override; None => auto-calibrate
    warmup: int = 10                  # floor; budget may raise it
    max_iters: int = 1000
    l2_bytes: int = 256 * 1024 * 1024  # fallback; replaced by ~2x L2 query


_TIMING = TimingConfig()
_L2_BUFFER = None   # lazily-allocated device scratch for the L2 flush


def query_l2_flush_bytes(gpu: "GpuInfo") -> int:
    """~2x the GPU L2 cache size in bytes (Triton do_bench uses a buffer larger
    than L2 so the prior iteration's footprint is evicted). Falls back to 256MB
    when the L2 size cannot be queried."""
    l2 = 0
    if _HAVE_TORCH and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            l2 = int(getattr(props, "L2_cache_size", 0) or 0)
        except Exception:
            l2 = 0
    if l2 <= 0:
        return 256 * 1024 * 1024
    # 2x L2, but never smaller than 64MB and cap at 512MB to bound the per-iter cost.
    want = 2 * l2
    return max(64 * 1024 * 1024, min(want, 512 * 1024 * 1024))


def _get_l2_buffer():
    global _L2_BUFFER
    if not (_HAVE_TORCH and torch.cuda.is_available()):
        return None
    n = max(1, _TIMING.l2_bytes // 4)
    if _L2_BUFFER is None or _L2_BUFFER.numel() != n:
        _L2_BUFFER = torch.empty(n, dtype=torch.int32, device="cuda")
    return _L2_BUFFER


def _flush_l2() -> None:
    buf = _get_l2_buffer()
    if buf is not None:
        buf.zero_()


def _estimate_iters(fn: Callable[[], None]) -> Tuple[int, float]:
    """Estimate one-iter cost (ms) from a few flushed launches, then derive the
    iteration count to fill the target-ms budget. Returns (n_iters, est_ms)."""
    if _TIMING.iters is not None:
        return max(1, _TIMING.iters), float("nan")
    # a few quick samples to estimate cost (with L2 flush so the estimate matches
    # the real measurement regime)
    n_probe = 5
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if _TIMING.l2_flush:
        _flush_l2()
    start.record()
    for _ in range(n_probe):
        fn()
    end.record()
    torch.cuda.synchronize()
    est_ms = max(start.elapsed_time(end) / n_probe, 1e-4)
    n = int(_TIMING.target_ms / est_ms)
    n = max(_TIMING.warmup, n)             # always at least a warmup's worth
    n = max(20, min(n, _TIMING.max_iters))  # floor 20 for a stable median
    return n, est_ms


def _time_cuda_events(fn: Callable[[], None], warmup: int, iters: int) -> Timing:
    """Flushed (or warm) per-iteration CUDA-event timing.

    `fn` must enqueue exactly one logical op (no host sync inside). Each iteration
    is timed individually; when L2 flush is enabled the scratch buffer is zeroed
    immediately before each timed launch so the op reads cold from HBM. We enqueue
    all iterations then synchronize once (no inner sync), then read elapsed_time.
    """
    for _ in range(max(1, warmup)):
        if _TIMING.l2_flush:
            _flush_l2()
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if _TIMING.l2_flush:
            _flush_l2()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return Timing.from_samples(times_ms, method="event")


def _time_cudagraph(fn: Callable[[], None], warmup: int, iters: int) -> Timing:
    """Capture `fn` into a CUDA graph and replay it to amortize launch overhead.

    NOTE (BP-22): L2 is NOT flushed between graph replays, so these are WARM-L2,
    launch-overhead-removed numbers — not directly comparable to flushed-eager.
    Falls back to event timing if capture is unavailable or fails.
    """
    if not hasattr(torch.cuda, "CUDAGraph"):
        return _time_cuda_events(fn, warmup, iters)

    # warm up + let any lazy allocation/autotune settle before capture
    for _ in range(max(3, warmup)):
        fn()
    torch.cuda.synchronize()

    # number of fn() invocations baked into one replay window
    replays = max(10, min(iters, 50))
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(replays):
                fn()
    except Exception:
        # capture not supported for this op (e.g. dynamic shapes / sync inside)
        return _time_cuda_events(fn, warmup, iters)

    # warm the graph itself
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    # each measured window = one replay (replays invocations); divide by replays
    windows = max(20, min(iters, _TIMING.max_iters))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(windows)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(windows)]
    for i in range(windows):
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) / replays for s, e in zip(starts, ends)]
    return Timing.from_samples(times_ms, method="cudagraph")


def time_op(fn: Callable[[], None]) -> Timing:
    """Time one logical op and return a full latency distribution.

    Honors the global TimingConfig: budget-derived iteration count (or --iters
    override), optional L2 flush per iteration, and an optional CUDA-graph replay
    path for launch-bound ops. Requires torch+CUDA (main() guarantees this).
    """
    n_iters, _ = _estimate_iters(fn)
    warmup = _TIMING.warmup
    if _TIMING.cudagraph:
        return _time_cudagraph(fn, warmup, n_iters)
    return _time_cuda_events(fn, warmup, n_iters)


def time_cuda(fn: Callable[[], None], warmup: int = 0, iters: int = 0) -> float:
    """Backward-compatible shim: returns the median latency in microseconds.

    Retained so the per-op bench functions can call ``time_cuda(run, ...)`` as
    before; the ``warmup``/``iters`` args are ignored in favor of the global
    TimingConfig (auto-calibration + L2 flush). Use :func:`time_op` for the full
    distribution.
    """
    return time_op(fn).median_us


# Consume timed-ref outputs so the launch is not dead-code-eliminated (BP-10).
# We accumulate one element of the output into a persistent DEVICE scalar — this
# observes the result and keeps it live WITHOUT a host sync (a host sync would
# serialize and corrupt the async event timing).
_DCE_SINK = None


def _sink(fn: Callable[[], object]) -> Callable[[], None]:
    """Wrap a function that returns a tensor so its output is consumed each call,
    preventing the timed launch from being optimized away. The consume is a
    device-side accumulate (no host sync)."""
    global _DCE_SINK
    if _HAVE_TORCH and torch.cuda.is_available():
        _DCE_SINK = torch.zeros((), device="cuda", dtype=torch.float32)

    def wrapped() -> None:
        out = fn()
        if (_DCE_SINK is not None and out is not None
                and hasattr(out, "numel") and out.numel() > 0):
            try:
                _DCE_SINK.add_(out.reshape(-1)[0].float())
            except Exception:
                pass
    return wrapped


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    op: str
    shape: str
    dtype: str
    ks_us: float = float("nan")        # representative (median) kernel-set latency
    ref_us: float = float("nan")       # representative (median) baseline latency
    # full latency distribution (us) for the kernel-set op
    ks_min_us: float = float("nan")
    ks_p20_us: float = float("nan")
    ks_p80_us: float = float("nan")
    ref_min_us: float = float("nan")
    gbps: float = float("nan")
    tflops: float = float("nan")
    bw_util: float = float("nan")      # achieved GB/s as % of dense peak
    compute_util: float = float("nan")  # achieved TFLOP/s as % of dense peak
    rel_err: float = float("nan")
    tol: float = float("nan")          # tolerance the correctness was gated at
    is_correct: Optional[bool] = None  # None = no reference available
    speedup: float = float("nan")
    baseline: str = ""                 # which baseline the speedup is vs
    method: str = ""                   # "event" / "cudagraph"
    n_iters: int = 0
    status: str = "ok"            # "ok", "skip", "INCORRECT", or "error: ..."
    note: str = ""

    def set_ks_timing(self, t: "Timing") -> None:
        self.ks_us = t.median_us
        self.ks_min_us = t.min_us
        self.ks_p20_us = t.p20_us
        self.ks_p80_us = t.p80_us
        self.method = t.method
        self.n_iters = t.iters

    def set_ref_timing(self, t: "Timing") -> None:
        self.ref_us = t.median_us
        self.ref_min_us = t.min_us

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

    def tol(self) -> float:
        """Op-appropriate relative-error tolerance for the active dtype."""
        if self.dt is torch.float32:
            return 2e-3
        if self.dt is torch.bfloat16:
            return 3e-2   # bf16 has only 8 mantissa bits
        return 1e-2       # fp16


def _fill_util(r: Result, gpu: GpuInfo) -> None:
    """Populate bw_util / compute_util (% of dense peak) from gbps / tflops."""
    if not math.isnan(r.gbps):
        peak = peak_for_metric(gpu, r.dtype, "bw")
        if peak > 0:
            r.bw_util = 100.0 * r.gbps / peak
    if not math.isnan(r.tflops):
        # w8a8 is int8 math; tag it so the peak lookup picks tf8
        dn = "int8" if r.op == "w8a8" else r.dtype
        peak = peak_for_metric(gpu, dn, "tflops")
        if peak > 0:
            r.compute_util = 100.0 * r.tflops / peak


def gate_correctness(r: Result, ctx: Ctx) -> bool:
    """Set is_correct/tol/status from r.rel_err BEFORE trusting the speed.

    Returns True if the op passed (or has no reference). A failing op is marked
    status='INCORRECT' so the report does not present it as a clean speedup.
    """
    r.tol = ctx.tol()
    if math.isnan(r.rel_err):
        r.is_correct = None     # no reference => cannot gate
        return True
    r.is_correct = r.rel_err <= r.tol
    if not r.is_correct and r.status == "ok":
        r.status = "INCORRECT"
    return bool(r.is_correct)


def _is_arch_unsupported(exc: Exception) -> bool:
    """True if a kernel_set error is ARCH_UNSUPPORTED (e.g. fp8 quant on sm80),
    so the caller can mark skip(arch) rather than failing the whole run.

    KernelSetError carries the numeric ks_status_t as ``.status`` (7 ==
    KS_ERROR_ARCH_UNSUPPORTED) and a symbolic ``.status_name``; we also fall back
    to a substring match so the guard is robust if the error type changes."""
    if getattr(exc, "status", None) == 7:
        return True
    name = (getattr(exc, "status_name", "") or "").upper()
    if "ARCH_UNSUPPORTED" in name:
        return True
    return "ARCH_UNSUPPORTED" in str(exc).upper()


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

    # reference
    def ref_rms(t):
        f = t.float()
        return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(t.dtype) * w

    # correctness FIRST (gate timing): materialize output for the rel_err read.
    ref = ref_rms(x)
    ks.norm.rms_norm(out, x, w, eps=eps)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # bytes: read x + write out (+ read w once, negligible) ~ 2 * rows*hidden
    nbytes = 2 * rows * hidden * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)

    r.set_ref_timing(time_op(_sink(lambda: ref_rms(x))))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager"
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

    ref = torch.nn.functional.layer_norm(
        x.float(), (hidden,), w.float(), b.float(), eps).to(ctx.dt)
    ks.norm.layer_norm(out, x, w, b, eps=eps)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    nbytes = 2 * rows * hidden * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)

    r.set_ref_timing(time_op(_sink(
        lambda: torch.nn.functional.layer_norm(x, (hidden,), w, b, eps))))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager"
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

    ref = (torch.nn.functional.silu(gate.float()) * up.float()).to(ctx.dt)
    ks.activation.swiglu(out, gate, up)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # read gate + read up + write out = 3 * rows*inter
    nbytes = 3 * rows * inter * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)

    r.set_ref_timing(time_op(_sink(
        lambda: torch.nn.functional.silu(gate) * up)))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager"
    return r


for _lbl, _rows, _inter in _SWIGLU_SHAPES:
    register("swiglu", _lbl)(
        (lambda rows, inter: lambda c: _bench_swiglu(c, rows, inter))(_rows, _inter))


# -------------------------------- ROPE ------------------------------------- #
# [num_tokens, heads, head_dim]; Llama-3-8B: 32 q heads, 8 kv heads, head_dim 128.
# _ROPE_SHAPES + _ref_rope_neox imported from _bench_common (shared with bench_sota).


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

    # correctness FIRST on a fresh copy
    qc, kc = q0.clone(), k0.clone()
    ref_q = _ref_rope_neox(qc, cos, sin)
    ks.rope.rope_inplace(qc, kc, cos, sin, tokens, qh, kvh, hd)
    torch.cuda.synchronize()
    r.rel_err = rel_err(qc, ref_q)
    r.note = "neox/rotate_half"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    nbytes = 2 * (tokens * qh * hd + tokens * kvh * hd) * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


for _lbl, _tk, _qh, _kvh, _hd in _ROPE_SHAPES:
    register("rope", _lbl)(
        (lambda tk, qh, kvh, hd: lambda c: _bench_rope(c, tk, qh, kvh, hd))(
            _tk, _qh, _kvh, _hd))


# ------------------------------ ATTENTION ---------------------------------- #
# Prefill: dense flash attention. Decode: paged attention.
# _ATTN_PREFILL_SHAPES + _ATTN_DECODE_SHAPES imported from _bench_common.


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

    # correctness FIRST: high-precision fp32 SDPA reference (math backend).
    ref = _ref_sdpa(q, k, v, qh, kvh, causal=True, scale=scale, fp32=True)
    ks.attention.flash_attn(out, q, k, v, b, seq, seq, qh, kvh, hd,
                            softmax_scale=scale, causal=True)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # causal attention FLOPs ~ 2 * (QK^T + softmax*V) * 0.5 (causal)
    # ~ 4 * b * qh * seq^2 * hd  (2 matmuls, factor 2 for MAC), halved for causal
    flops = 4.0 * b * qh * (seq ** 2) * hd * 0.5
    r.tflops = flops / (r.ks_us * 1e-6) / 1e12
    _fill_util(r, ctx.gpu)

    # baseline timed in the KERNEL's dtype with the flash/efficient SDPA backend
    # (fair precision + strongest torch path), not fp32-math.
    r.set_ref_timing(time_op(_sink(
        lambda: _ref_sdpa(q, k, v, qh, kvh, causal=True, scale=scale,
                          fp32=False))))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "sdpa(flash/efficient)"
    return r


def _sdpa_backends():
    """Return a context manager pinning SDPA to the flash/efficient backends
    (excluding the slow math fallback), or a no-op CM if unavailable."""
    import contextlib
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION])
    except Exception:
        return contextlib.nullcontext()


def _ref_sdpa(q, k, v, qh, kvh, causal, scale, fp32: bool):
    """torch SDPA reference with GQA expansion.

    fp32=True  -> high-precision math backend, for the correctness baseline.
    fp32=False -> kernel dtype + flash/efficient backend, for a FAIR timed
                  baseline (matches the kernel's precision, BP-08).
    """
    cast = (lambda t: t.float()) if fp32 else (lambda t: t)
    qt = cast(q.transpose(1, 2))
    kt = cast(k.transpose(1, 2))
    vt = cast(v.transpose(1, 2))
    if kvh != qh:
        rep = qh // kvh
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)
    if fp32:
        o = torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt, is_causal=causal, scale=scale)
    else:
        with _sdpa_backends():
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

    r.set_ks_timing(time_op(run))
    # decode is memory-bound: read whole KV cache each step.
    kv_bytes = 2 * num_seqs * kvh * ctx_len * hd * dtype_bytes(ctx.dt)
    r.gbps = kv_bytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
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

    # correctness FIRST vs an fp32-accumulated reference.
    ref = (a.float() @ b.float()).to(ctx.dt)
    ks.gemm.gemm(c, a, b, m=m, n=n, k=k)
    torch.cuda.synchronize()
    r.rel_err = rel_err(c, ref)
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    flops = 2.0 * m * n * k
    r.tflops = flops / (r.ks_us * 1e-6) / 1e12
    _fill_util(r, ctx.gpu)

    # Baselines at the kernel dtype (TF32 pinned globally): cuBLAS eager `a@b`
    # and, when available, torch.compile(max-autotune). Headline vs the FASTEST.
    cublas = time_op(_sink(lambda: a @ b))
    best, best_name = cublas, "cublas(a@b)"
    compiled = _maybe_compile_matmul(a, b)
    if compiled is not None:
        comp = time_op(_sink(compiled))
        if comp.median_us < best.median_us:
            best, best_name = comp, "torch.compile"
        r.note = (f"cublas={cublas.median_us:.1f}us "
                  f"compile={comp.median_us:.1f}us")
    r.set_ref_timing(best)
    r.speedup = r.ref_us / r.ks_us
    r.baseline = best_name
    return r


def _maybe_compile_matmul(a, b):
    """Return a callable computing a@b via torch.compile(max-autotune), or None
    if torch.compile is unavailable / fails. Cached across shapes is unsafe
    (shapes differ), so we compile per call-site lazily and tolerate failure."""
    if not hasattr(torch, "compile"):
        return None
    try:
        def mm(x, y):
            return x @ y
        fn = torch.compile(mm, mode="max-autotune")
        # trigger compilation once so it is not in the timed window
        _ = fn(a, b)
        torch.cuda.synchronize()
        return lambda: fn(a, b)
    except Exception:
        return None


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

    # correctness FIRST. reference: int matmul (done in fp64 — CUDA has no
    # integer matmul, and fp64 represents these |sum| < 2^53 products exactly)
    # then dequant.
    acc = (a.double() @ b.double()).float()
    ref = (acc * a_scale.unsqueeze(1) * b_scale.unsqueeze(0)).to(ctx.dt)
    ks.gemm.gemm_w8a8(out, a, b, a_scale, b_scale, m=m, n=n, k=k, out_dtype=out_dt)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    flops = 2.0 * m * n * k
    r.tflops = flops / (r.ks_us * 1e-6) / 1e12
    _fill_util(r, ctx.gpu)   # int8 -> tf8 peak
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
        r.set_ks_timing(time_op(run))
    except Exception as exc:
        r.status = f"error: {type(exc).__name__}: {exc}"
        return r
    flops = 2.0 * m * n * k
    r.tflops = flops / (r.ks_us * 1e-6) / 1e12
    _fill_util(r, ctx.gpu)
    r.note = "no portable torch int4 ref; throughput only (uncorrectness-gated)"
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

    # correctness FIRST. reference: softmax + topk + renormalize
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
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
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

    # correctness FIRST: per-expert torch GEMM over the CSR offsets.
    try:
        ks.moe.grouped_gemm(c, a, b, offsets, E, total_rows, inter, hidden)
        torch.cuda.synchronize()
        off = offsets.detach().cpu().tolist()
        ref = torch.empty_like(c)
        for e in range(E):
            lo, hi = off[e], off[e + 1]
            if hi > lo:
                ref[lo:hi] = (a[lo:hi].float() @ b[e].float()).to(c.dtype)
        r.rel_err = rel_err(c, ref)
    except Exception as exc:
        r.status = f"error: {type(exc).__name__}: {exc}"
        return r
    if not gate_correctness(r, ctx):
        return r

    try:
        r.set_ks_timing(time_op(run))
    except Exception as exc:
        r.status = f"error: {type(exc).__name__}: {exc}"
        return r
    flops = 2.0 * total_rows * inter * hidden
    r.tflops = flops / (r.ks_us * 1e-6) / 1e12
    _fill_util(r, ctx.gpu)
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

    # correctness FIRST
    ref = logits.float().argmax(-1).to(torch.int32)
    ks.sampling.argmax(tokens, logits, num_seqs, vocab)
    torch.cuda.synchronize()
    match = (tokens == ref).float().mean().item()
    r.rel_err = 1.0 - match   # fraction mismatched (0 == perfect)
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    nbytes = num_seqs * vocab * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)

    r.set_ref_timing(time_op(_sink(lambda: logits.argmax(-1))))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager"
    r.note = "argmax (greedy); rel_err = mismatch fraction"
    return r


for _lbl, _s, _v in _SAMPLING_SHAPES:
    register("sampling", _lbl)(
        (lambda s, v: lambda c: _bench_sampling(c, s, v))(_s, _v))


# ---------------------------- CROSS ENTROPY -------------------------------- #
# _CE_SHAPES imported from _bench_common (shared with bench_sota).


def _bench_cross_entropy(ctx: Ctx, num_tokens, vocab) -> Result:
    r = Result("cross_entropy", f"tokens={num_tokens},vocab={vocab}", ctx.dtype_name)
    logits = ctx.rand(num_tokens, vocab)
    targets = torch.randint(0, vocab, (num_tokens,), device="cuda", dtype=torch.int64)
    losses = torch.empty(num_tokens, device="cuda", dtype=torch.float32)
    grad = torch.empty_like(logits)

    def run():
        ks.loss.cross_entropy(losses, grad, logits, targets,
                              num_tokens, vocab, ignore_index=-100)

    # correctness FIRST (on the forward loss)
    ref = torch.nn.functional.cross_entropy(
        logits.float(), targets, reduction="none")
    ks.loss.cross_entropy(losses, grad, logits, targets, num_tokens, vocab,
                          ignore_index=-100)
    torch.cuda.synchronize()
    r.rel_err = rel_err(losses, ref)
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # read logits + write grad ~ 2 * tokens*vocab
    nbytes = 2 * num_tokens * vocab * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)

    r.set_ref_timing(time_op(_sink(
        lambda: torch.nn.functional.cross_entropy(
            logits, targets, reduction="none"))))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager(fwd-only)"
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
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # read+write param (dt) + grad (dt) + 2 fp32 states r/w
    nbytes = (3 * n * dtype_bytes(ctx.dt)) + (4 * n * 4)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    r.note = "fused AdamW step; memory-bound"
    return r


for _lbl, _n in _ADAMW_SHAPES:
    register("adamw", _lbl)(
        (lambda n: lambda c: _bench_adamw(c, n))(_n))


# =========================================================================== #
# Full-op-coverage benchmarks: every remaining kernel-set compute op, each with
# a correctness gate (vs the strongest available torch reference — autograd for
# the backward kernels) + perf (GB/s for bandwidth-bound, TFLOP/s for FLCE).
# =========================================================================== #

# --------------------------- NORM BACKWARD --------------------------------- #
def _bench_rmsnorm_bwd(ctx: Ctx, rows: int, hidden: int) -> Result:
    r = Result("rmsnorm_bwd", f"rows={rows},hidden={hidden}", ctx.dtype_name)
    x = ctx.rand(rows, hidden)
    w = ctx.rand(hidden)
    g = ctx.rand(rows, hidden)                       # grad_out
    eps = 1e-6
    grad_input = ctx.empty(rows, hidden)
    grad_weight_fp32 = torch.zeros(hidden, device="cuda", dtype=torch.float32)

    def run():
        grad_weight_fp32.zero_()
        ks.norm.rms_norm_backward(grad_input, grad_weight_fp32, g, x, w,
                                  rows=rows, cols=hidden, eps=eps)

    # reference via torch autograd through the RMSNorm forward (fp32 math).
    xv = x.float().detach().requires_grad_(True)
    wv = w.float().detach().requires_grad_(True)
    rms = torch.rsqrt(xv.pow(2).mean(-1, keepdim=True) + eps)
    y = (xv * rms) * wv
    y.backward(g.float())
    ref_gi, ref_gw = xv.grad, wv.grad

    grad_weight_fp32.zero_()
    ks.norm.rms_norm_backward(grad_input, grad_weight_fp32, g, x, w,
                              rows=rows, cols=hidden, eps=eps)
    torch.cuda.synchronize()
    r.rel_err = max(rel_err(grad_input, ref_gi),
                    rel_err(grad_weight_fp32, ref_gw))
    r.note = "autograd ref (grad_input + grad_weight_fp32)"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # read grad_out + input, write grad_input (+ atomic grad_weight, negligible)
    nbytes = 3 * rows * hidden * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


def _bench_layernorm_bwd(ctx: Ctx, rows: int, hidden: int) -> Result:
    r = Result("layernorm_bwd", f"rows={rows},hidden={hidden}", ctx.dtype_name)
    x = ctx.rand(rows, hidden)
    w = ctx.rand(hidden)
    b = ctx.rand(hidden)
    g = ctx.rand(rows, hidden)
    eps = 1e-5
    grad_input = ctx.empty(rows, hidden)
    grad_weight_fp32 = torch.zeros(hidden, device="cuda", dtype=torch.float32)
    grad_bias_fp32 = torch.zeros(hidden, device="cuda", dtype=torch.float32)

    def run():
        grad_weight_fp32.zero_()
        grad_bias_fp32.zero_()
        ks.norm.layer_norm_backward(grad_input, grad_weight_fp32, grad_bias_fp32,
                                    g, x, w, rows=rows, cols=hidden, eps=eps)

    # reference via torch autograd through F.layer_norm (fp32 math).
    xv = x.float().detach().requires_grad_(True)
    wv = w.float().detach().requires_grad_(True)
    bv = b.float().detach().requires_grad_(True)
    y = torch.nn.functional.layer_norm(xv, (hidden,), wv, bv, eps)
    y.backward(g.float())
    ref_gi, ref_gw, ref_gb = xv.grad, wv.grad, bv.grad

    grad_weight_fp32.zero_()
    grad_bias_fp32.zero_()
    ks.norm.layer_norm_backward(grad_input, grad_weight_fp32, grad_bias_fp32,
                                g, x, w, rows=rows, cols=hidden, eps=eps)
    torch.cuda.synchronize()
    r.rel_err = max(rel_err(grad_input, ref_gi),
                    rel_err(grad_weight_fp32, ref_gw),
                    rel_err(grad_bias_fp32, ref_gb))
    r.note = "autograd ref (grad_input + grad_weight/bias_fp32)"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    nbytes = 3 * rows * hidden * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


# layer_norm forward is already registered under "layernorm" (_bench_layernorm).
for _lbl, _rows, _hid in _NORM_SHAPES:
    register("rmsnorm_bwd", _lbl)(
        (lambda rows, hid: lambda c: _bench_rmsnorm_bwd(c, rows, hid))(_rows, _hid))
    register("layernorm_bwd", _lbl)(
        (lambda rows, hid: lambda c: _bench_layernorm_bwd(c, rows, hid))(_rows, _hid))


# ------------------------- ACTIVATION: GeGLU + bwd ------------------------- #
def _bench_geglu(ctx: Ctx, rows: int, inter: int) -> Result:
    r = Result("geglu", f"rows={rows},inter={inter}", ctx.dtype_name)
    gate = ctx.rand(rows, inter)
    up = ctx.rand(rows, inter)
    out = ctx.empty(rows, inter)

    def run():
        ks.activation.geglu(out, gate, up)

    # exact (erf) GELU * up, fp32 reference.
    ref = (torch.nn.functional.gelu(gate.float()) * up.float()).to(ctx.dt)
    ks.activation.geglu(out, gate, up)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    nbytes = 3 * rows * inter * dtype_bytes(ctx.dt)   # read gate+up, write out
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)

    r.set_ref_timing(time_op(_sink(
        lambda: torch.nn.functional.gelu(gate) * up)))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager"
    return r


def _bench_swiglu_bwd(ctx: Ctx, rows: int, inter: int) -> Result:
    r = Result("swiglu_bwd", f"rows={rows},inter={inter}", ctx.dtype_name)
    gate = ctx.rand(rows, inter)
    up = ctx.rand(rows, inter)
    g = ctx.rand(rows, inter)                          # grad_out
    grad_gate = ctx.empty(rows, inter)
    grad_up = ctx.empty(rows, inter)

    def run():
        ks.activation.swiglu_backward(grad_gate, grad_up, g, gate, up)

    # autograd through silu(gate)*up.
    gv = gate.float().detach().requires_grad_(True)
    uv = up.float().detach().requires_grad_(True)
    y = torch.nn.functional.silu(gv) * uv
    y.backward(g.float())
    ref_gg, ref_gu = gv.grad, uv.grad

    ks.activation.swiglu_backward(grad_gate, grad_up, g, gate, up)
    torch.cuda.synchronize()
    r.rel_err = max(rel_err(grad_gate, ref_gg), rel_err(grad_up, ref_gu))
    r.note = "autograd ref (grad_gate + grad_up)"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # read grad_out + gate + up, write grad_gate + grad_up = 5 * rows*inter
    nbytes = 5 * rows * inter * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


for _lbl, _rows, _inter in _SWIGLU_SHAPES:
    register("geglu", _lbl)(
        (lambda rows, inter: lambda c: _bench_geglu(c, rows, inter))(_rows, _inter))
    register("swiglu_bwd", _lbl)(
        (lambda rows, inter: lambda c: _bench_swiglu_bwd(c, rows, inter))(_rows, _inter))


# ----------------------------- ROPE BACKWARD ------------------------------- #
def _bench_rope_bwd(ctx: Ctx, tokens: int, qh: int, kvh: int, hd: int) -> Result:
    r = Result("rope_bwd", f"tokens={tokens},qh={qh},kvh={kvh},hd={hd}",
               ctx.dtype_name)
    grad_q = ctx.rand(tokens, qh, hd)
    grad_k = ctx.rand(tokens, kvh, hd)
    cos = ctx.rand(tokens, hd // 2)
    sin = ctx.rand(tokens, hd // 2)
    gq0, gk0 = grad_q.clone(), grad_k.clone()

    def run():
        # in-place; rotates the grads by the conjugate rotation each call.
        grad_q.copy_(gq0)
        grad_k.copy_(gk0)
        ks.rope.rope_backward(grad_q, grad_k, cos, sin,
                              tokens, qh, kvh, hd)

    # RoPE backward is the conjugate (transpose) rotation, i.e. RoPE with -sin.
    # That is exactly autograd through the forward NeoX RoPE:
    #   forward o = R(theta) x  =>  grad_x = R(theta)^T grad_o = R(-theta) grad_o.
    gqc, gkc = gq0.clone(), gk0.clone()
    ref_gq = _ref_rope_neox(gqc, cos, -sin)   # conjugate rotation
    ks.rope.rope_backward(gqc, gkc, cos, sin, tokens, qh, kvh, hd)
    torch.cuda.synchronize()
    r.rel_err = rel_err(gqc, ref_gq)
    r.note = "conjugate-rotation ref (neox)"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    nbytes = 2 * (tokens * qh * hd + tokens * kvh * hd) * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


for _lbl, _tk, _qh, _kvh, _hd in _ROPE_SHAPES:
    register("rope_bwd", _lbl)(
        (lambda tk, qh, kvh, hd: lambda c: _bench_rope_bwd(c, tk, qh, kvh, hd))(
            _tk, _qh, _kvh, _hd))


# ------------------------------- EMBEDDING --------------------------------- #
def _bench_embedding_lookup(ctx: Ctx, tokens, vocab, embed_dim) -> Result:
    r = Result("embedding", f"tokens={tokens},vocab={vocab},d={embed_dim}",
               ctx.dtype_name)
    table = ctx.rand(vocab, embed_dim)
    idx = torch.randint(0, vocab, (tokens,), device="cuda", dtype=torch.int32)
    out = ctx.empty(tokens, embed_dim)

    def run():
        ks.embedding.embedding_lookup(out, table, idx, tokens, embed_dim)

    # reference: torch index_select.
    ref = table.index_select(0, idx.to(torch.int64))
    ks.embedding.embedding_lookup(out, table, idx, tokens, embed_dim)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    r.note = "index_select ref"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # read+write the gathered rows = 2 * tokens*embed_dim (table read is sparse).
    nbytes = 2 * tokens * embed_dim * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)

    r.set_ref_timing(time_op(_sink(
        lambda: table.index_select(0, idx.to(torch.int64)))))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager(index_select)"
    return r


def _bench_embedding_bwd(ctx: Ctx, tokens, vocab, embed_dim) -> Result:
    r = Result("embedding_bwd", f"tokens={tokens},vocab={vocab},d={embed_dim}",
               ctx.dtype_name)
    idx = torch.randint(0, vocab, (tokens,), device="cuda", dtype=torch.int32)
    g = ctx.rand(tokens, embed_dim)                    # grad_out
    grad_table_fp32 = torch.zeros(vocab, embed_dim, device="cuda",
                                  dtype=torch.float32)

    def run():
        grad_table_fp32.zero_()
        ks.embedding.embedding_backward(grad_table_fp32, g, idx,
                                        tokens, embed_dim)

    # reference: scatter-add into a zero table (autograd of index_select).
    ref = torch.zeros(vocab, embed_dim, device="cuda", dtype=torch.float32)
    ref.index_add_(0, idx.to(torch.int64), g.float())

    grad_table_fp32.zero_()
    ks.embedding.embedding_backward(grad_table_fp32, g, idx, tokens, embed_dim)
    torch.cuda.synchronize()
    r.rel_err = rel_err(grad_table_fp32, ref)
    r.note = "scatter-add (index_add) ref; grad_table fp32"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # read grad_out (dt) + scatter-add into fp32 table (read+write touched rows).
    nbytes = tokens * embed_dim * dtype_bytes(ctx.dt) + 2 * tokens * embed_dim * 4
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


for _lbl, _tk, _v, _d in _EMBEDDING_SHAPES:
    register("embedding", _lbl)(
        (lambda tk, v, d: lambda c: _bench_embedding_lookup(c, tk, v, d))(
            _tk, _v, _d))
    register("embedding_bwd", _lbl)(
        (lambda tk, v, d: lambda c: _bench_embedding_bwd(c, tk, v, d))(
            _tk, _v, _d))


# ------------------------------ ELEMENTWISE -------------------------------- #
def _bench_ew_add(ctx: Ctx, n) -> Result:
    r = Result("ew_add", f"n={n}", ctx.dtype_name)
    a = ctx.rand(n)
    b = ctx.rand(n)
    out = ctx.empty(n)

    def run():
        ks.elementwise.add(out, a, b, n)

    ref = (a + b)
    ks.elementwise.add(out, a, b, n)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    if not gate_correctness(r, ctx):
        return r
    r.set_ks_timing(time_op(run))
    r.gbps = (3 * n * dtype_bytes(ctx.dt)) / (r.ks_us * 1e-6) / 1e9   # 2 read+1 write
    _fill_util(r, ctx.gpu)
    r.set_ref_timing(time_op(_sink(lambda: a + b)))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager"
    return r


def _bench_ew_mul(ctx: Ctx, n) -> Result:
    r = Result("ew_mul", f"n={n}", ctx.dtype_name)
    a = ctx.rand(n)
    b = ctx.rand(n)
    out = ctx.empty(n)

    def run():
        ks.elementwise.mul(out, a, b, n)

    ref = (a * b)
    ks.elementwise.mul(out, a, b, n)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    if not gate_correctness(r, ctx):
        return r
    r.set_ks_timing(time_op(run))
    r.gbps = (3 * n * dtype_bytes(ctx.dt)) / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    r.set_ref_timing(time_op(_sink(lambda: a * b)))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager"
    return r


def _bench_ew_add_residual(ctx: Ctx, n) -> Result:
    r = Result("ew_add_residual", f"n={n}", ctx.dtype_name)
    x = ctx.rand(n)
    res0 = ctx.rand(n)
    residual = res0.clone()

    def run():
        residual.copy_(res0)
        ks.elementwise.add_residual(residual, x, n)

    # in-place: residual <- residual + x
    rc = res0.clone()
    ref = res0 + x
    ks.elementwise.add_residual(rc, x, n)
    torch.cuda.synchronize()
    r.rel_err = rel_err(rc, ref)
    r.note = "in-place residual += x"
    if not gate_correctness(r, ctx):
        return r
    r.set_ks_timing(time_op(run))
    # read residual + x, write residual = 3 * n
    r.gbps = (3 * n * dtype_bytes(ctx.dt)) / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


def _bench_ew_scale(ctx: Ctx, n) -> Result:
    r = Result("ew_scale", f"n={n}", ctx.dtype_name)
    x = ctx.rand(n)
    out = ctx.empty(n)
    s = 1.2345

    def run():
        ks.elementwise.scale(out, x, s, n)

    ref = (x * s)
    ks.elementwise.scale(out, x, s, n)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    if not gate_correctness(r, ctx):
        return r
    r.set_ks_timing(time_op(run))
    r.gbps = (2 * n * dtype_bytes(ctx.dt)) / (r.ks_us * 1e-6) / 1e9   # read+write
    _fill_util(r, ctx.gpu)
    r.set_ref_timing(time_op(_sink(lambda: x * s)))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager"
    return r


def _bench_ew_cast(ctx: Ctx, n) -> Result:
    # cast the active dtype -> fp32 (correctness-checkable, dtype-stable).
    r = Result("ew_cast", f"n={n}", ctx.dtype_name)
    x = ctx.rand(n)
    out = torch.empty(n, device="cuda", dtype=torch.float32)
    dst = ks.dtype_to_ks(torch.float32)

    def run():
        ks.elementwise.cast(out, dst, x, n=n)

    ref = x.to(torch.float32)
    ks.elementwise.cast(out, dst, x, n=n)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    r.note = f"{ctx.dtype_name}->fp32 cast"
    if not gate_correctness(r, ctx):
        return r
    r.set_ks_timing(time_op(run))
    # read src (dt) + write dst (fp32)
    r.gbps = (n * dtype_bytes(ctx.dt) + n * 4) / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    r.set_ref_timing(time_op(_sink(lambda: x.to(torch.float32))))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager(.to)"
    return r


def _bench_ew_axpby(ctx: Ctx, n) -> Result:
    r = Result("ew_axpby", f"n={n}", ctx.dtype_name)
    a = ctx.rand(n)
    b = ctx.rand(n)
    out = ctx.empty(n)
    alpha, beta = 0.75, -0.5

    def run():
        ks.elementwise.axpby(out, a, alpha, b, beta, n)

    ref = (a * alpha + b * beta)
    ks.elementwise.axpby(out, a, alpha, b, beta, n)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    r.note = "out = a*alpha + b*beta"
    if not gate_correctness(r, ctx):
        return r
    r.set_ks_timing(time_op(run))
    r.gbps = (3 * n * dtype_bytes(ctx.dt)) / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    r.set_ref_timing(time_op(_sink(lambda: a * alpha + b * beta)))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager"
    return r


_EW_BUILDERS = [
    ("add", _bench_ew_add),
    ("mul", _bench_ew_mul),
    ("add_residual", _bench_ew_add_residual),
    ("scale", _bench_ew_scale),
    ("cast", _bench_ew_cast),
    ("axpby", _bench_ew_axpby),
]
for _op_name, _fn in _EW_BUILDERS:
    for _lbl, _n in _ELEMENTWISE_SHAPES:
        register("elementwise", f"{_op_name},{_lbl}")(
            (lambda fn, n: lambda c: fn(c, n))(_fn, _n))


# ------------------------------ QUANTIZATION ------------------------------- #
def _bench_quantize_fp8(ctx: Ctx, rows, cols) -> Result:
    r = Result("quantize_fp8", f"rows={rows},cols={cols}", ctx.dtype_name)
    x = ctx.rand(rows, cols)
    out = torch.empty(rows, cols, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.zeros(1, device="cuda", dtype=torch.float32)   # per-tensor
    deq = torch.empty(rows, cols, device="cuda", dtype=torch.float32)

    def run():
        ks.quant.quantize_fp8(out, scale, x, rows, cols,
                              fp8_dtype=ks.DType.F8E4M3,
                              mode=ks.QuantMode.PER_TENSOR)

    # correctness: round-trip dequant(quant(x)) ~ x within fp8 quant tolerance.
    # (FP8 e4m3 has ~2-3 mantissa bits -> ~6-12% per-element error; gate on the
    # relative L2 of the round-trip, which the round-trip preserves well.)
    try:
        ks.quant.quantize_fp8(out, scale, x, rows, cols,
                              fp8_dtype=ks.DType.F8E4M3,
                              mode=ks.QuantMode.PER_TENSOR)
        ks.quant.dequantize_fp8(deq, out, scale, rows, cols,
                                fp8_dtype=ks.DType.F8E4M3,
                                mode=ks.QuantMode.PER_TENSOR)
        torch.cuda.synchronize()
    except Exception as exc:
        if _is_arch_unsupported(exc):
            r.status = "skip"
            r.note = "fp8 not supported on this arch (ARCH_UNSUPPORTED)"
            return r
        r.status = f"error: {type(exc).__name__}: {exc}"
        return r
    # relative L2 of the round-trip (robust to per-element fp8 rounding noise).
    xf = x.float()
    r.rel_err = ((deq - xf).norm() / xf.norm().clamp_min(1e-6)).item()
    r.tol = 0.10   # fp8 e4m3 round-trip tolerance (overrides dtype tol)
    r.is_correct = r.rel_err <= r.tol
    if not r.is_correct and r.status == "ok":
        r.status = "INCORRECT"
    r.note = "round-trip dequant(quant(x))~x; rel-L2 gate @0.10"
    if not r.is_correct:
        return r

    r.set_ks_timing(time_op(run))
    # read x (dt), write fp8 out (1 byte) + scale (negligible)
    nbytes = rows * cols * dtype_bytes(ctx.dt) + rows * cols * 1
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


def _bench_dequantize_fp8(ctx: Ctx, rows, cols) -> Result:
    r = Result("dequantize_fp8", f"rows={rows},cols={cols}", ctx.dtype_name)
    x = ctx.rand(rows, cols)
    q = torch.empty(rows, cols, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.zeros(1, device="cuda", dtype=torch.float32)
    out = ctx.empty(rows, cols)

    try:
        ks.quant.quantize_fp8(q, scale, x, rows, cols,
                              fp8_dtype=ks.DType.F8E4M3,
                              mode=ks.QuantMode.PER_TENSOR)
        torch.cuda.synchronize()
    except Exception as exc:
        if _is_arch_unsupported(exc):
            r.status = "skip"
            r.note = "fp8 not supported on this arch (ARCH_UNSUPPORTED)"
            return r
        r.status = f"error: {type(exc).__name__}: {exc}"
        return r

    def run():
        ks.quant.dequantize_fp8(out, q, scale, rows, cols,
                                fp8_dtype=ks.DType.F8E4M3,
                                mode=ks.QuantMode.PER_TENSOR)

    # reference: float(fp8) * scale, computed by upcasting the fp8 codes via torch.
    ref = (q.float() * scale).to(ctx.dt)
    ks.quant.dequantize_fp8(out, q, scale, rows, cols,
                            fp8_dtype=ks.DType.F8E4M3,
                            mode=ks.QuantMode.PER_TENSOR)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    r.note = "ref = float(fp8)*scale"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # read fp8 (1 byte), write out (dt)
    nbytes = rows * cols * 1 + rows * cols * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


def _bench_quantize_int8(ctx: Ctx, rows, cols) -> Result:
    r = Result("quantize_int8", f"rows={rows},cols={cols}", ctx.dtype_name)
    x = ctx.rand(rows, cols)
    out = torch.empty(rows, cols, device="cuda", dtype=torch.int8)
    scale = torch.zeros(rows, device="cuda", dtype=torch.float32)   # per-token
    deq = torch.empty(rows, cols, device="cuda", dtype=torch.float32)

    def run():
        ks.quant.quantize_int8(out, scale, x, rows, cols,
                               mode=ks.QuantMode.PER_TOKEN)

    # correctness: reference symmetric per-token quant scale = amax/127, then
    # compare both the codes and the round-trip.
    try:
        ks.quant.quantize_int8(out, scale, x, rows, cols,
                               mode=ks.QuantMode.PER_TOKEN)
        ks.quant.dequantize_int8(deq, out, scale, rows, cols,
                                 mode=ks.QuantMode.PER_TOKEN)
        torch.cuda.synchronize()
    except Exception as exc:
        r.status = f"error: {type(exc).__name__}: {exc}"
        return r
    # gate on (a) the per-token scale matching the symmetric amax/127 reference
    # and (b) the round-trip dequant(quant(x)) ~ x within one quant step.
    xf = x.float()
    ref_scale = (xf.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    r.rel_err = max(
        rel_err(scale, ref_scale),
        ((deq - xf).norm() / xf.norm().clamp_min(1e-6)).item(),
    )
    r.tol = 0.03   # int8 per-token round-trip + scale tolerance
    r.is_correct = r.rel_err <= r.tol
    if not r.is_correct and r.status == "ok":
        r.status = "INCORRECT"
    r.note = "ref scale=amax/127; round-trip rel-L2 gate @0.03"
    if not r.is_correct:
        return r

    r.set_ks_timing(time_op(run))
    nbytes = rows * cols * dtype_bytes(ctx.dt) + rows * cols * 1
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


def _bench_dequantize_int8(ctx: Ctx, rows, cols) -> Result:
    r = Result("dequantize_int8", f"rows={rows},cols={cols}", ctx.dtype_name)
    x = ctx.rand(rows, cols)
    q = torch.empty(rows, cols, device="cuda", dtype=torch.int8)
    scale = torch.zeros(rows, device="cuda", dtype=torch.float32)
    out = ctx.empty(rows, cols)
    try:
        ks.quant.quantize_int8(q, scale, x, rows, cols,
                               mode=ks.QuantMode.PER_TOKEN)
        torch.cuda.synchronize()
    except Exception as exc:
        r.status = f"error: {type(exc).__name__}: {exc}"
        return r

    def run():
        ks.quant.dequantize_int8(out, q, scale, rows, cols,
                                 mode=ks.QuantMode.PER_TOKEN)

    # reference: int8 codes * per-token scale.
    ref = (q.float() * scale.unsqueeze(1)).to(ctx.dt)
    ks.quant.dequantize_int8(out, q, scale, rows, cols,
                             mode=ks.QuantMode.PER_TOKEN)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    r.note = "ref = int8*scale (per-token)"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    nbytes = rows * cols * 1 + rows * cols * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


def _bench_dequantize_int4(ctx: Ctx, k, n, group_size) -> Result:
    r = Result("dequantize_int4", f"K={k},N={n},g={group_size}", ctx.dtype_name)
    if ctx.dt not in (torch.float16, torch.bfloat16, torch.float32):
        r.status = "skip"
        r.note = "int4 dequant out must be float"
        return r
    # packed int32 [K/8, N], 8 nibbles (codes 0..15) per word along K.
    qweight = torch.randint(0, 2 ** 31 - 1, (k // 8, n), device="cuda",
                            dtype=torch.int32)
    n_groups = k // group_size
    scales = (ctx.rand(n_groups, n).abs() * 0.02 + 1e-3).to(ctx.dt)
    zeros = torch.full((n_groups, n), 8.0, device="cuda").to(ctx.dt)  # symmetric
    out = ctx.empty(k, n)

    def run():
        ks.quant.dequantize_int4(out, qweight, scales, zeros, k, n, group_size)

    # exact AWQ/GPTQ unpack reference (matches dequant_int4.cu):
    #   q = (word >> (4*(kk%8))) & 0xF;  g = kk/group_size
    #   out[kk,nn] = (q - zero[g,nn]) * scale[g,nn]
    try:
        ks.quant.dequantize_int4(out, qweight, scales, zeros, k, n, group_size)
        torch.cuda.synchronize()
        kk = torch.arange(k, device="cuda")
        word = qweight[(kk >> 3)]                       # [K, N] gathered words
        shift = ((kk & 7) * 4).view(k, 1)
        codes = ((word >> shift) & 0xF).float()         # [K, N] in [0,15]
        g = (kk // group_size)
        zexp = zeros.float()[g]                         # [K, N]
        sexp = scales.float()[g]                        # [K, N]
        ref = ((codes - zexp) * sexp).to(ctx.dt)
        r.rel_err = rel_err(out, ref)
    except Exception as exc:
        r.status = f"error: {type(exc).__name__}: {exc}"
        return r
    r.note = "exact AWQ/GPTQ unpack ref"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # read packed (4 bytes / 8 codes) + scales/zeros, write [K,N] out (dt).
    nbytes = (k // 8) * n * 4 + k * n * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


for _lbl, _rows, _cols in _QUANT_SHAPES:
    register("quant", f"quantize_fp8,{_lbl}")(
        (lambda rw, cl: lambda c: _bench_quantize_fp8(c, rw, cl))(_rows, _cols))
    register("quant", f"dequantize_fp8,{_lbl}")(
        (lambda rw, cl: lambda c: _bench_dequantize_fp8(c, rw, cl))(_rows, _cols))
    register("quant", f"quantize_int8,{_lbl}")(
        (lambda rw, cl: lambda c: _bench_quantize_int8(c, rw, cl))(_rows, _cols))
    register("quant", f"dequantize_int8,{_lbl}")(
        (lambda rw, cl: lambda c: _bench_dequantize_int8(c, rw, cl))(_rows, _cols))
for _lbl, _k, _n, _g in _DEQUANT_INT4_SHAPES:
    register("quant", f"dequantize_int4,{_lbl}")(
        (lambda kk, nn, gg: lambda c: _bench_dequantize_int4(c, kk, nn, gg))(
            _k, _n, _g))


# --------------------------- MoE PERMUTE/UNPERMUTE ------------------------- #
def _moe_permutation_setup(ctx: Ctx, tokens, E, k):
    """Build the routing indices + permutation (compute_permutation is the
    setup helper) shared by the permute / unpermute benchmarks."""
    logits = ctx.rand(tokens, E)
    weights = torch.empty(tokens, k, device="cuda", dtype=torch.float32)
    indices = torch.empty(tokens, k, device="cuda", dtype=torch.int32)
    ks.moe.gate_softmax_topk(weights, indices, logits, tokens, E, k,
                             renormalize=True)
    sorted_token_ids = torch.empty(tokens * k, device="cuda", dtype=torch.int32)
    expert_offsets = torch.empty(E + 1, device="cuda", dtype=torch.int32)
    ks.moe.compute_permutation(sorted_token_ids, expert_offsets, indices,
                               tokens, E, k)
    torch.cuda.synchronize()
    return weights, indices, sorted_token_ids, expert_offsets


def _bench_moe_permute(ctx: Ctx, tokens, hidden, E, k) -> Result:
    r = Result("moe_permute", f"tokens={tokens},h={hidden},E={E},k={k}",
               ctx.dtype_name)
    _, _, sorted_ids, _ = _moe_permutation_setup(ctx, tokens, E, k)
    x = ctx.rand(tokens, hidden)
    permuted = ctx.empty(tokens * k, hidden)

    def run():
        ks.moe.permute(permuted, x, sorted_ids, tokens, k, hidden)

    # reference: gather rows of x by sorted_token_ids (// k maps to source token).
    ks.moe.permute(permuted, x, sorted_ids, tokens, k, hidden)
    torch.cuda.synchronize()
    src = (sorted_ids.to(torch.int64) // k)            # source-token per dest row
    ref = x.index_select(0, src)
    r.rel_err = rel_err(permuted, ref)
    r.note = "gather-by-sorted_token_ids ref"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    nbytes = 2 * tokens * k * hidden * dtype_bytes(ctx.dt)   # read+write rows
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


def _bench_moe_unpermute(ctx: Ctx, tokens, hidden, E, k) -> Result:
    r = Result("moe_unpermute", f"tokens={tokens},h={hidden},E={E},k={k}",
               ctx.dtype_name)
    weights, _, sorted_ids, _ = _moe_permutation_setup(ctx, tokens, E, k)
    permuted = ctx.rand(tokens * k, hidden)            # expert outputs
    out = ctx.empty(tokens, hidden)

    def run():
        ks.moe.unpermute(out, permuted, sorted_ids, weights,
                         tokens, k, hidden)

    # reference: out[token] = sum_k weight[token,k] * permuted[pos(token,k)].
    # sorted_ids[pos] = token*k + slot, so dest token = id // k, weight slot id%k.
    ks.moe.unpermute(out, permuted, sorted_ids, weights, tokens, k, hidden)
    torch.cuda.synchronize()
    ids = sorted_ids.to(torch.int64)
    dst_tok = ids // k
    slot = ids % k
    w_per_row = weights[dst_tok, slot].unsqueeze(1).float()   # [tokens*k, 1]
    contrib = permuted.float() * w_per_row
    ref = torch.zeros(tokens, hidden, device="cuda", dtype=torch.float32)
    ref.index_add_(0, dst_tok, contrib)
    r.rel_err = rel_err(out, ref.to(ctx.dt))
    r.note = "weighted scatter-add ref"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    nbytes = (tokens * k * hidden + tokens * hidden) * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


for _lbl, _tk, _h, _E, _k in _MOE_PERMUTE_SHAPES:
    register("moe_permute", _lbl)(
        (lambda tk, h, E, k: lambda c: _bench_moe_permute(c, tk, h, E, k))(
            _tk, _h, _E, _k))
    register("moe_unpermute", _lbl)(
        (lambda tk, h, E, k: lambda c: _bench_moe_unpermute(c, tk, h, E, k))(
            _tk, _h, _E, _k))


# --------------------------- ATTENTION: cache + bwd ----------------------- #
def _bench_reshape_and_cache(ctx: Ctx, tokens, kvh, hd, block) -> Result:
    r = Result("reshape_and_cache",
               f"tokens={tokens},kvh={kvh},hd={hd},blk={block}", ctx.dtype_name)
    key = ctx.rand(tokens, kvh, hd)
    value = ctx.rand(tokens, kvh, hd)
    # enough blocks to hold all tokens, one token per distinct slot.
    num_blocks = (tokens + block - 1) // block + 1
    k_cache = ctx.rand(num_blocks, kvh, block, hd)
    v_cache = ctx.rand(num_blocks, kvh, block, hd)
    # distinct flat slots = block_id*block + offset; pick the first `tokens`.
    slot_mapping = torch.arange(tokens, device="cuda", dtype=torch.int32)

    def run():
        ks.attention.reshape_and_cache(k_cache, v_cache, key, value,
                                       slot_mapping, tokens, kvh, hd, block)

    # correctness: scatter K/V into the cache, then gather back by slot and
    # compare to the original key/value.
    ks.attention.reshape_and_cache(k_cache, v_cache, key, value,
                                   slot_mapping, tokens, kvh, hd, block)
    torch.cuda.synchronize()
    slots = slot_mapping.to(torch.int64)
    blk_id = slots // block
    off = slots % block
    # cache layout [num_blocks, kvh, block, hd] -> gather [tokens, kvh, hd]
    got_k = k_cache[blk_id, :, off, :]                 # [tokens, kvh, hd]
    got_v = v_cache[blk_id, :, off, :]
    r.rel_err = max(rel_err(got_k, key), rel_err(got_v, value))
    r.note = "scatter-then-gather-by-slot ref"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # read K+V, write K+V into cache = 4 * tokens*kvh*hd
    nbytes = 4 * tokens * kvh * hd * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


def _bench_flash_attn_bwd(ctx: Ctx, b, seq, qh, kvh, hd) -> Result:
    r = Result("flash_attn_bwd",
               f"b={b},seq={seq},qh={qh},kvh={kvh},hd={hd}", ctx.dtype_name)
    q = ctx.rand(b, seq, qh, hd)
    k = ctx.rand(b, seq, kvh, hd)
    v = ctx.rand(b, seq, kvh, hd)
    out = ctx.empty(b, seq, qh, hd)
    g = ctx.rand(b, seq, qh, hd)                       # grad_out
    lse = torch.empty(qh, b * seq, device="cuda", dtype=torch.float32)
    scale = 1.0 / math.sqrt(hd)
    grad_q = ctx.empty(b, seq, qh, hd)
    grad_k = ctx.empty(b, seq, kvh, hd)
    grad_v = ctx.empty(b, seq, kvh, hd)

    # forward (produces out + softmax_lse needed by the backward).
    try:
        ks.attention.flash_attn(out, q, k, v, b, seq, seq, qh, kvh, hd,
                                softmax_lse=lse, softmax_scale=scale, causal=True)
        torch.cuda.synchronize()
    except Exception as exc:
        if _is_arch_unsupported(exc):
            r.status = "skip"
            r.note = "flash_attn fwd ARCH_UNSUPPORTED"
            return r
        r.status = f"error: {type(exc).__name__}: {exc}"
        return r

    def run():
        ks.attention.flash_attn_backward(
            grad_q, grad_k, grad_v, g, q, k, v, out, lse,
            b, seq, seq, qh, kvh, hd, softmax_scale=scale, causal=True)

    # reference: autograd through the fp32 SDPA math reference (GQA-expanded).
    qv = q.float().detach().transpose(1, 2).requires_grad_(True)
    kv = k.float().detach().transpose(1, 2).requires_grad_(True)
    vv = v.float().detach().transpose(1, 2).requires_grad_(True)
    rep = qh // kvh
    ke = kv.repeat_interleave(rep, dim=1) if kvh != qh else kv
    ve = vv.repeat_interleave(rep, dim=1) if kvh != qh else vv
    o = torch.nn.functional.scaled_dot_product_attention(
        qv, ke, ve, is_causal=True, scale=scale)
    o.transpose(1, 2).backward(g.float())
    ref_gq = qv.grad.transpose(1, 2)
    ref_gk = kv.grad.transpose(1, 2)
    ref_gv = vv.grad.transpose(1, 2)

    try:
        ks.attention.flash_attn_backward(
            grad_q, grad_k, grad_v, g, q, k, v, out, lse,
            b, seq, seq, qh, kvh, hd, softmax_scale=scale, causal=True)
        torch.cuda.synchronize()
    except Exception as exc:
        if _is_arch_unsupported(exc):
            r.status = "skip"
            r.note = "flash_attn_backward ARCH_UNSUPPORTED"
            return r
        r.status = f"error: {type(exc).__name__}: {exc}"
        return r
    r.rel_err = max(rel_err(grad_q, ref_gq), rel_err(grad_k, ref_gk),
                    rel_err(grad_v, ref_gv))
    r.note = "autograd-through-SDPA ref (grad_q/k/v)"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # backward attention ~ 2.5x the forward causal FLOPs (BWD ~ 5 matmuls).
    flops = 2.5 * (4.0 * b * qh * (seq ** 2) * hd * 0.5)
    r.tflops = flops / (r.ks_us * 1e-6) / 1e12
    _fill_util(r, ctx.gpu)
    return r


for _lbl, _tk, _kvh, _hd, _blk in _RESHAPE_CACHE_SHAPES:
    register("reshape_and_cache", _lbl)(
        (lambda tk, kvh, hd, blk:
         lambda c: _bench_reshape_and_cache(c, tk, kvh, hd, blk))(
            _tk, _kvh, _hd, _blk))
for _lbl, _b, _seq, _qh, _kvh, _hd in _ATTN_BWD_SHAPES:
    register("flash_attn_bwd", _lbl)(
        (lambda b, seq, qh, kvh, hd:
         lambda c: _bench_flash_attn_bwd(c, b, seq, qh, kvh, hd))(
            _b, _seq, _qh, _kvh, _hd))


# ------------------- FUSED LINEAR CROSS ENTROPY (FLCE) -------------------- #
def _bench_flce(ctx: Ctx, tokens, hidden_dim, vocab) -> Result:
    r = Result("fused_linear_ce",
               f"tokens={tokens},d={hidden_dim},vocab={vocab}", ctx.dtype_name)
    hidden = ctx.rand(tokens, hidden_dim)
    weight = ctx.rand(vocab, hidden_dim)               # LM head [vocab, d]
    targets = torch.randint(0, vocab, (tokens,), device="cuda", dtype=torch.int64)
    losses = torch.empty(tokens, device="cuda", dtype=torch.float32)
    grad_hidden = ctx.empty(tokens, hidden_dim)
    grad_weight_fp32 = torch.zeros(vocab, hidden_dim, device="cuda",
                                   dtype=torch.float32)

    def run():
        grad_weight_fp32.zero_()
        ks.loss.fused_linear_cross_entropy(
            losses, grad_hidden, grad_weight_fp32, hidden, weight, targets,
            tokens, hidden_dim, vocab, ignore_index=-100)

    # reference: torch F.cross_entropy(hidden @ W^T, targets) + autograd grads.
    # The kernel emits per-token losses and SUM-reduction grads (grad of logits
    # = softmax - onehot, no 1/n scaling — reduction is left to the caller, see
    # fused_linear_cross_entropy.cu). So back-prop the SUMMED loss for the grads
    # and compare the per-token loss vector separately.
    hv = hidden.float().detach().requires_grad_(True)
    wv = weight.float().detach().requires_grad_(True)
    logits = hv @ wv.t()                               # [tokens, vocab]
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
    loss.backward()
    ref_losses = torch.nn.functional.cross_entropy(
        logits.detach(), targets, reduction="none")
    ref_gh, ref_gw = hv.grad, wv.grad

    grad_weight_fp32.zero_()
    ks.loss.fused_linear_cross_entropy(
        losses, grad_hidden, grad_weight_fp32, hidden, weight, targets,
        tokens, hidden_dim, vocab, ignore_index=-100)
    torch.cuda.synchronize()
    r.rel_err = max(rel_err(losses, ref_losses),
                    rel_err(grad_hidden, ref_gh),
                    rel_err(grad_weight_fp32, ref_gw))
    r.note = "F.cross_entropy(h@W^T)+autograd; per-token loss, sum-reduction grads"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # FLCE is compute-bound (two GEMMs: logits = h@W^T fwd, grads bwd).
    # FLOPs ~ 2*tokens*vocab*hidden (fwd) + 2*tokens*vocab*hidden (grad_hidden)
    #        + 2*tokens*vocab*hidden (grad_weight) = 6*tokens*vocab*hidden.
    flops = 6.0 * tokens * vocab * hidden_dim
    r.tflops = flops / (r.ks_us * 1e-6) / 1e12
    _fill_util(r, ctx.gpu)
    return r


for _lbl, _tk, _d, _v in _FLCE_SHAPES:
    register("fused_linear_ce", _lbl)(
        (lambda tk, d, v: lambda c: _bench_flce(c, tk, d, v))(_tk, _d, _v))


# ----------------------------- OPTIMIZER ---------------------------------- #
def _bench_sgd_momentum(ctx: Ctx, n) -> Result:
    r = Result("sgd_momentum", f"n={n}", ctx.dtype_name)
    param0 = ctx.rand(n)
    grad = ctx.rand(n)
    lr, mf, wd = 1e-2, 0.9, 0.01

    def run():
        pk = param0.clone()
        mk = torch.zeros(n, device="cuda", dtype=torch.float32)
        ks.optimizer.sgd_momentum(pk, grad, mk, lr,
                                  momentum_factor=mf, weight_decay=wd)

    # reference: one SGD-with-momentum step (decoupled? -> standard coupled wd:
    # d = grad + wd*param; buf = mf*buf + d (buf starts 0); param -= lr*buf).
    p_ref = param0.float().clone()
    g_ref = grad.float()
    d = g_ref + wd * p_ref
    buf = d                                  # mf*0 + d
    p_ref = p_ref - lr * buf

    pk = param0.clone()
    mk = torch.zeros(n, device="cuda", dtype=torch.float32)
    ks.optimizer.sgd_momentum(pk, grad, mk, lr,
                              momentum_factor=mf, weight_decay=wd)
    torch.cuda.synchronize()
    r.rel_err = rel_err(pk, p_ref.to(ctx.dt))
    r.note = "manual SGD+momentum step ref"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    # read+write param (dt) + read grad (dt) + read+write momentum (fp32)
    nbytes = 3 * n * dtype_bytes(ctx.dt) + 2 * n * 4
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


def _bench_global_grad_norm(ctx: Ctx, n) -> Result:
    r = Result("global_grad_norm", f"n={n}", ctx.dtype_name)
    # split into a few tensors (typical: per-parameter grads).
    n1 = n // 2
    n2 = n - n1
    g1 = ctx.rand(n1)
    g2 = ctx.rand(n2)
    out_norm = torch.zeros(1, device="cuda", dtype=torch.float32)

    def run():
        ks.optimizer.global_grad_norm(out_norm, [g1, g2])

    # reference: torch.linalg.vector_norm over the concatenated grads (fp32).
    ref = torch.linalg.vector_norm(
        torch.cat([g1.float().reshape(-1), g2.float().reshape(-1)]))
    ks.optimizer.global_grad_norm(out_norm, [g1, g2])
    torch.cuda.synchronize()
    r.rel_err = rel_err(out_norm, ref.reshape(1))
    r.note = "sqrt(sum||g||^2) vs torch vector_norm"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    nbytes = n * dtype_bytes(ctx.dt)                   # read all grads once
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    return r


for _lbl, _n in _OPTIM_SHAPES:
    register("sgd_momentum", _lbl)(
        (lambda n: lambda c: _bench_sgd_momentum(c, n))(_n))
    register("global_grad_norm", _lbl)(
        (lambda n: lambda c: _bench_global_grad_norm(c, n))(_n))


# ------------------------------- SAMPLING --------------------------------- #
# argmax is already covered by the "sampling" category (_bench_sampling). Add a
# stand-alone argmax registration + log_softmax to close coverage explicitly.
def _bench_argmax(ctx: Ctx, num_seqs, vocab) -> Result:
    r = Result("argmax", f"seqs={num_seqs},vocab={vocab}", ctx.dtype_name)
    logits = ctx.rand(num_seqs, vocab)
    tokens = torch.empty(num_seqs, device="cuda", dtype=torch.int32)

    def run():
        ks.sampling.argmax(tokens, logits, num_seqs, vocab)

    ref = logits.float().argmax(-1).to(torch.int32)
    ks.sampling.argmax(tokens, logits, num_seqs, vocab)
    torch.cuda.synchronize()
    r.rel_err = 1.0 - (tokens == ref).float().mean().item()   # mismatch fraction
    r.note = "torch.argmax ref; rel_err = mismatch fraction"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    nbytes = num_seqs * vocab * dtype_bytes(ctx.dt)
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    r.set_ref_timing(time_op(_sink(lambda: logits.argmax(-1))))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager"
    return r


def _bench_log_softmax(ctx: Ctx, rows, cols) -> Result:
    r = Result("log_softmax", f"rows={rows},vocab={cols}", ctx.dtype_name)
    x = ctx.rand(rows, cols)
    out = ctx.empty(rows, cols)

    def run():
        ks.sampling.log_softmax(out, x, rows, cols)

    ref = torch.log_softmax(x.float(), dim=-1).to(ctx.dt)
    ks.sampling.log_softmax(out, x, rows, cols)
    torch.cuda.synchronize()
    r.rel_err = rel_err(out, ref)
    r.note = "torch.log_softmax(fp32) ref"
    if not gate_correctness(r, ctx):
        return r

    r.set_ks_timing(time_op(run))
    nbytes = 2 * rows * cols * dtype_bytes(ctx.dt)     # read + write
    r.gbps = nbytes / (r.ks_us * 1e-6) / 1e9
    _fill_util(r, ctx.gpu)
    r.set_ref_timing(time_op(_sink(lambda: torch.log_softmax(x, dim=-1))))
    r.speedup = r.ref_us / r.ks_us
    r.baseline = "eager"
    return r


for _lbl, _s, _v in _SAMPLING_SHAPES:
    register("argmax", _lbl)(
        (lambda s, v: lambda c: _bench_argmax(c, s, v))(_s, _v))
for _lbl, _rows, _cols in _LOG_SOFTMAX_SHAPES:
    register("log_softmax", _lbl)(
        (lambda rows, cols: lambda c: _bench_log_softmax(c, rows, cols))(
            _rows, _cols))


# --------------------------------------------------------------------------- #
# Op aliases: the user-facing op names map to one or more registry categories.
# This lets `--ops gemm` cover the fp16/bf16 selector and keeps the help tidy.
# --------------------------------------------------------------------------- #
ALL_OPS = [
    # forward / GEMM / attention / MoE / sampling / loss / optimizer (existing)
    "rmsnorm", "layernorm", "swiglu", "rope", "attention",
    "gemm", "w8a8", "w4a16", "moe", "sampling", "cross_entropy", "adamw",
    # full-op-coverage additions (every remaining kernel-set compute op):
    "rmsnorm_bwd", "layernorm_bwd",                     # norm backward
    "geglu", "swiglu_bwd",                              # activation
    "rope_bwd",                                         # rope backward
    "embedding", "embedding_bwd",                       # embedding fwd/bwd
    "elementwise",                                      # add/mul/add_residual/scale/cast/axpby
    "quant",                                            # fp8/int8 quant+dequant, int4 dequant
    "moe_permute", "moe_unpermute",                     # MoE permute/unpermute
    "reshape_and_cache", "flash_attn_bwd",             # attention cache + backward
    "fused_linear_ce",                                  # fused-linear-cross-entropy
    "sgd_momentum", "global_grad_norm",                # optimizer
    "argmax", "log_softmax",                            # sampling
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
    if r.status == "INCORRECT":
        print(f"  [BAD ] {r.op:<18} {r.shape:<30} INCORRECT "
              f"rel_err={r.rel_err:.2e} > tol={r.tol:.1e} (speed not reported)",
              file=sys.stderr)
        return
    perf = ""
    if not math.isnan(r.tflops):
        perf = f"{r.tflops:8.1f} TFLOP/s"
        if not math.isnan(r.compute_util):
            perf += f" ({r.compute_util:.0f}% pk)"
    elif not math.isnan(r.gbps):
        perf = f"{r.gbps:8.1f} GB/s"
        if not math.isnan(r.bw_util):
            perf += f" ({r.bw_util:.0f}% pk)"
    extra = ""
    if not math.isnan(r.speedup):
        extra += f"  speedup={r.speedup:5.2f}x"
    if not math.isnan(r.rel_err):
        extra += f"  rel_err={r.rel_err:.2e}"
    tag = r.method[:1].upper() if r.method else " "
    print(f"  [ok {tag}] {r.op:<18} {r.shape:<30} "
          f"{_fmt_lat(r.ks_us, r.ks_min_us):>16} us  {perf}{extra}",
          file=sys.stderr)


# --------------------------------------------------------------------------- #
# Aggregate score (fast_p): joint correctness + speed
# --------------------------------------------------------------------------- #
def compute_fast_p(results: List[Result]) -> Dict[str, object]:
    """fast_p = fraction of ops that are BOTH correct (within tolerance) AND at
    least p-times the baseline speed. We report fast_1 as the headline (correct
    AND >= baseline). Also returns correct/incorrect/error counts and the mean
    speedup over correct-only ops.

    Ops without a reference (is_correct is None) are excluded from the
    correctness-gated population but counted separately.
    """
    # population = ops that actually ran a comparison (have a baseline speedup)
    comparable = [r for r in results
                  if r.status not in ("skip",) and not r.status.startswith("error")
                  and r.is_correct is not None and not math.isnan(r.speedup)]
    n_error = sum(1 for r in results if r.status.startswith("error"))
    n_skip = sum(1 for r in results if r.status == "skip")
    n_correct = sum(1 for r in results if r.is_correct is True)
    n_incorrect = sum(1 for r in results if r.is_correct is False)
    n_noref = sum(1 for r in results
                  if r.is_correct is None and r.status == "ok")

    def fast_p(p: float) -> float:
        if not comparable:
            return float("nan")
        hits = sum(1 for r in comparable
                   if r.is_correct and r.speedup >= p)
        return hits / len(comparable)

    correct_speedups = [r.speedup for r in comparable if r.is_correct]
    mean_correct_speedup = (statistics.mean(correct_speedups)
                            if correct_speedups else float("nan"))

    return {
        "fast_0": fast_p(0.0),   # correct at any speed
        "fast_1": fast_p(1.0),   # correct AND >= baseline (headline)
        "fast_2": fast_p(2.0),   # correct AND >= 2x baseline
        "n_comparable": len(comparable),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_error": n_error,
        "n_skip": n_skip,
        "n_noref": n_noref,
        "mean_speedup_correct": mean_correct_speedup,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x: float, prec: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{prec}f}"


def _fmt_lat(med: float, lo: float) -> str:
    """'median (min)' latency cell in microseconds."""
    if med is None or (isinstance(med, float) and math.isnan(med)):
        return "-"
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return f"{med:.1f}"
    return f"{med:.1f} ({lo:.1f})"


def _fmt_util(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.0f}%"


def render_markdown(results: List[Result], gpu: GpuInfo, cfg: dict) -> str:
    lines: List[str] = []
    lines.append(f"# kernel-set benchmark — {gpu.name}")
    lines.append("")
    peaks = _peaks_for(gpu)
    peak_str = ""
    if peaks:
        peak_str = (f" | dense peak BW ~{peaks.get('bw', 0):.0f} GB/s"
                    f" | dense fp16/bf16 TC ~{peaks.get('tf16', 0):.0f} TFLOP/s")
    lines.append(f"- **GPU**: {gpu.name} (sm_{gpu.sm_arch}, CC {gpu.cc}, "
                 f"{gpu.sm_count} SMs, {gpu.total_mem_gb:.1f} GB){peak_str}")
    lines.append(f"- **detected via**: {gpu.source}")

    # clocks (+ locked?)
    clk = cfg.get("clocks") or {}
    lk = cfg.get("clock_lock") or {}
    clk_str = (f"SM {clk.get('sm_mhz', '?')} MHz, mem {clk.get('mem_mhz', '?')} MHz")
    if lk.get("requested"):
        clk_str += (" | locked" if lk.get("locked")
                    else f" | lock requested but NOT applied ({lk.get('detail', '')})")
    else:
        clk_str += " | clocks UNLOCKED (boost/throttle not controlled)"
    if clk.get("throttle") and clk.get("throttle") not in ("Not Active", "[N/A]"):
        clk_str += f" | throttle: {clk.get('throttle')}"
    lines.append(f"- **clocks**: {clk_str}")

    env = cfg.get("env") or {}
    lines.append(f"- **driver**: {env.get('driver', '?')} | "
                 f"CUDA {env.get('torch_cuda', '?')} | "
                 f"cuDNN {env.get('cudnn', '?')} | "
                 f"power cap {env.get('power_limit_w', '?')} W | "
                 f"ECC {env.get('ecc', '?')}")
    tf = cfg.get("tf32") or {}
    lines.append(f"- **dtype**: {cfg['dtype']} | "
                 f"TF32 matmul={tf.get('matmul_allow_tf32')} "
                 f"cudnn={tf.get('cudnn_allow_tf32')} "
                 f"(fp32_precision={tf.get('float32_matmul_precision')})")
    lines.append(f"- **timing**: L2-flush={'on' if cfg.get('l2_flush') else 'off'}"
                 f" | method={'cudagraph (warm-L2, launch-overhead removed)' if cfg.get('cudagraph') else 'cuda-events (flushed)'}"
                 f" | target-ms={cfg.get('target_ms')}"
                 f" | iters={cfg.get('iters') if cfg.get('iters') else 'auto'}"
                 f" | warmup={cfg.get('warmup')}"
                 f" | L2-flush-buffer={cfg.get('l2_bytes', 0) // (1024 * 1024)} MB")
    lines.append(f"- **launch overhead included**: "
                 f"{'no (graph replay)' if cfg.get('cudagraph') else 'yes (single launch / event pair)'}")
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

    # fast_p aggregate (joint correctness + speed)
    fp = compute_fast_p(results)
    lines.append(
        f"**fast_1 = {_fmt(fp['fast_1'] * 100, 0) if not math.isnan(fp['fast_1']) else '-'}%** "
        f"of comparable ops are BOTH correct AND >= baseline speed "
        f"(fast_0={_fmt(fp['fast_0'] * 100, 0) if not math.isnan(fp['fast_0']) else '-'}%, "
        f"fast_2={_fmt(fp['fast_2'] * 100, 0) if not math.isnan(fp['fast_2']) else '-'}%). "
        f"correct={fp['n_correct']} · incorrect={fp['n_incorrect']} · "
        f"error={fp['n_error']} · skip={fp['n_skip']} · no-ref={fp['n_noref']}. "
        f"mean speedup over correct-only = "
        f"{_fmt(fp['mean_speedup_correct'])}x.")
    lines.append("")
    lines.append("Latency cells show **median (min)** microseconds over the "
                 "auto-calibrated timed iterations (CUDA events, L2-flushed unless "
                 "noted). `GB/s`/`TFLOP/s` cells append the achieved value's "
                 "**% of dense peak** for this SKU+dtype. `rel_err` is gated at the "
                 "dtype tolerance BEFORE speed is reported; a kernel that fails is "
                 "marked **INCORRECT**. `speedup` is best_baseline_us / ks_us.")
    lines.append("")
    lines.append("| op | shape | dtype | ks us (min) | ref us (min) | "
                 "GB/s (%pk) | TFLOP/s (%pk) | rel_err | spd | base | iters | m | notes |")
    lines.append("|---|---|---|--:|--:|--:|--:|--:|--:|---|--:|:-:|---|")
    for r in results:
        if r.status == "skip":
            lines.append(f"| {r.op} | {r.shape} | {r.dtype} | skip | - | - | - "
                         f"| - | - | - | - | - | {r.note or 'skipped'} |")
            continue
        if r.status.startswith("error"):
            lines.append(f"| {r.op} | {r.shape} | {r.dtype} | err | - | - | - "
                         f"| - | - | - | - | - | {r.status} |")
            continue
        if r.status == "INCORRECT":
            lines.append(
                f"| {r.op} | {r.shape} | {r.dtype} | INCORRECT | - | - | - | "
                f"{r.rel_err:.2e} | - | - | - | - | "
                f"rel_err > tol={r.tol:.1e}; speed not reported |")
            continue
        rel = ("-" if math.isnan(r.rel_err) else f"{r.rel_err:.2e}")
        gb = _fmt(r.gbps, 1)
        if not math.isnan(r.gbps) and not math.isnan(r.bw_util):
            gb = f"{r.gbps:.1f} ({_fmt_util(r.bw_util)})"
        tf_c = _fmt(r.tflops, 1)
        if not math.isnan(r.tflops) and not math.isnan(r.compute_util):
            tf_c = f"{r.tflops:.1f} ({_fmt_util(r.compute_util)})"
        lines.append(
            f"| {r.op} | {r.shape} | {r.dtype} | "
            f"{_fmt_lat(r.ks_us, r.ks_min_us)} | "
            f"{_fmt_lat(r.ref_us, r.ref_min_us)} | {gb} | {tf_c} | "
            f"{rel} | "
            f"{_fmt(r.speedup)}{'x' if not math.isnan(r.speedup) else ''} | "
            f"{r.baseline or '-'} | {r.n_iters or '-'} | "
            f"{(r.method[:1].upper() if r.method else '-')} | "
            f"{r.note} |")
    lines.append("")
    lines.append("_Legend: m = timing method (E=cuda-events flushed, "
                 "C=cudagraph replay). %pk = % of dense peak. spd = speedup vs "
                 "the fastest baseline named in `base`._")
    lines.append("")
    return "\n".join(lines)


def render_json(results: List[Result], gpu: GpuInfo, cfg: dict) -> str:
    return json.dumps({
        "gpu": gpu.__dict__,
        "config": cfg,
        "aggregate": compute_fast_p(results),
        "results": [r.to_dict() for r in results],
    }, indent=2, default=str)


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
    p.add_argument("--model-id", default="synthetic",
                   help="model/context label recorded in JSON results")
    p.add_argument("--layer-idx", type=int, default=None,
                   help="optional model layer index recorded in JSON results")
    p.add_argument("--position-kind", default=None,
                   help="optional position label, e.g. prefill/decode")
    p.add_argument("--position", type=int, default=None,
                   help="optional token/sequence position recorded in JSON results")
    p.add_argument("--warmup", type=int, default=10, help="warmup launches (floor)")
    p.add_argument("--iters", type=int, default=None,
                   help="fixed timed launches (overrides --target-ms "
                        "auto-calibration)")
    p.add_argument("--target-ms", type=float, default=200.0,
                   help="target measurement budget in ms; iteration count is "
                        "auto-calibrated to fill it (capped by --max-iters)")
    p.add_argument("--max-iters", type=int, default=1000,
                   help="upper bound on auto-calibrated iteration count")
    # L2 flush (Triton do_bench style); ON by default. --no-l2-flush disables.
    p.add_argument("--l2-flush", dest="l2_flush", action="store_true",
                   default=True,
                   help="zero a >L2 scratch buffer before each timed launch so "
                        "each iteration reads cold from HBM (default: ON)")
    p.add_argument("--no-l2-flush", dest="l2_flush", action="store_false",
                   help="disable the per-iteration L2 flush (warm-cache timing)")
    p.add_argument("--cudagraph", action="store_true",
                   help="time via CUDA-graph replay to amortize launch overhead "
                        "(tiny/launch-bound ops). NOTE: graph replay does NOT "
                        "flush L2 (warm-cache) — mutually exclusive with the "
                        "per-iter L2 flush; falls back to events if capture fails")
    p.add_argument("--lock-clocks", action="store_true",
                   help="attempt nvidia-smi persistence + lock-gpu-clocks to a "
                        "sustainable value (no-op if not permitted, e.g. Colab); "
                        "current clocks are always queried and reported")
    p.add_argument("--tf32", dest="tf32", action="store_true", default=True,
                   help="enable TF32 for fp32 matmul/cudnn (default: ON; "
                        "recorded in the header)")
    p.add_argument("--no-tf32", dest="tf32", action="store_false",
                   help="disable TF32 (fp32 GEMMs run at full fp32 precision)")
    p.add_argument("--output", default=None,
                   help="write the report to this file (default: stdout)")
    p.add_argument("--format", default="md", choices=["md", "json"],
                   help="report format")
    p.add_argument("--json-output", default=None,
                   help="also write a structured JSON report to this file "
                        "from the same benchmark run")
    p.add_argument("--timestamp", default=None,
                   help="optional run timestamp/label to record in the header")
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

    # diagnostic flags that corrupt timing should not be on while benchmarking
    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
        print("WARNING: CUDA_LAUNCH_BLOCKING=1 destroys async overlap and "
              "corrupts timing; unset it for performance runs.", file=sys.stderr)

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

    # ---- fairness: pin TF32 explicitly and record it (BP-08) ----
    tf32_info = apply_tf32(args.tf32)

    # ---- clock control + reporting (BP-05) ----
    clock_lock = {"requested": False, "locked": False, "detail": ""}
    if args.lock_clocks:
        clock_lock = lock_clocks()
        if clock_lock.get("locked"):
            print(f"Locked GPU clocks to ~{clock_lock.get('lock_mhz')} MHz.",
                  file=sys.stderr)
        else:
            print(f"NOTE: clock lock not applied ({clock_lock.get('detail')}); "
                  "running with whatever clock the GPU chooses.", file=sys.stderr)
    clocks = query_clocks()
    env = collect_env()

    # ---- timing config (L2 flush, cudagraph, budget) ----
    if args.cudagraph and args.l2_flush:
        # graph replay does not flush L2 (BP-22); they are mutually exclusive
        # per-iter. cudagraph wins; warn that the numbers are warm-cache.
        print("NOTE: --cudagraph replays do NOT flush L2; disabling per-iter "
              "L2 flush for graph timing (numbers are warm-cache, launch-"
              "overhead-removed).", file=sys.stderr)
    _TIMING.l2_flush = bool(args.l2_flush and not args.cudagraph)
    _TIMING.cudagraph = bool(args.cudagraph)
    _TIMING.target_ms = float(args.target_ms)
    _TIMING.iters = args.iters
    _TIMING.warmup = int(args.warmup)
    _TIMING.max_iters = int(args.max_iters)
    _TIMING.l2_bytes = query_l2_flush_bytes(gpu)

    ctx = Ctx(dt=dt, dtype_name=args.dtype, gpu=gpu,
              warmup=args.warmup, iters=args.iters or 0)

    cfg = {
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,            # None => auto
        "target_ms": args.target_ms,
        "max_iters": args.max_iters,
        "l2_flush": _TIMING.l2_flush,
        "l2_bytes": _TIMING.l2_bytes,
        "cudagraph": _TIMING.cudagraph,
        "ops": ops,
        "context": {
            "model_id": args.model_id,
            "layer_idx": args.layer_idx,
            "position_kind": args.position_kind,
            "position": args.position,
        },
        "tf32": tf32_info,
        "clocks": clocks,
        "clock_lock": clock_lock,
        "env": env,
        "git_commit": git_commit(),
        "timestamp": args.timestamp or datetime.datetime.now().isoformat(timespec="seconds"),
        "ks_version": ks.version() if _HAVE_KS else "?",
        "backend": ks.backend_name() if _HAVE_KS else "?",
        "torch_version": torch.__version__ if _HAVE_TORCH else None,
    }

    print(f"Running {len(ops)} op categories, dtype={args.dtype}, "
          f"L2-flush={_TIMING.l2_flush}, cudagraph={_TIMING.cudagraph}, "
          f"target-ms={args.target_ms}, "
          f"iters={args.iters if args.iters else 'auto'}\n", file=sys.stderr)
    try:
        results = run_benchmarks(ops, ctx, args.shape)
    finally:
        if clock_lock.get("locked"):
            reset_clocks()
            print("Reset GPU clocks.", file=sys.stderr)

    json_report: Optional[str] = None
    if args.format == "json":
        json_report = render_json(results, gpu, cfg)
        report = json_report
    else:
        report = render_markdown(results, gpu, cfg)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"\nWrote report to {args.output}", file=sys.stderr)
    else:
        print(report)

    if args.json_output:
        if json_report is None:
            json_report = render_json(results, gpu, cfg)
        os.makedirs(os.path.dirname(os.path.abspath(args.json_output)),
                    exist_ok=True)
        with open(args.json_output, "w") as f:
            f.write(json_report + "\n")
        print(f"Wrote JSON report to {args.json_output}", file=sys.stderr)

    # surface a nonzero exit if every benchmark errored (helps CI)
    ran = [r for r in results if r.status == "ok"]
    if results and not ran:
        print("ERROR: all benchmarks failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
