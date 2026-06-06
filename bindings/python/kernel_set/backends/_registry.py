"""Curated provider table for the best-backend dispatcher.

Derived from ``providers/registry.json`` (127 ops, ranked providers each with an
exact ``python_call`` / ``install`` / ``gpu_arch`` / ``dtypes``) but pared down
to the logical ops that have *real, callable* providers we can route to. For
each provider we ship:

* ``rank``        — registry rank (1 = industry-best); lower wins.
* ``min_sm``      — minimum compute capability gate (e.g. DeepGEMM/FlashMLA 90,
                    NVFP4 100). Below this the provider is skipped silently.
* ``dtypes``      — free-form dtype support string (from the registry).
* ``import_check``— registry import snippet used by the availability probe.
* ``call``        — a *lazy* thin adapter ``(args...) -> out``. It imports the
                    library on first use and mirrors the call signatures already
                    proven in ``benchmarks/bench_sota.py``. The kernel-set
                    provider's adapter calls into the C-ABI binding op modules.

The kernel-set C ABI provider is appended last to *every* op as the portable
fallback (``min_sm`` low, always selectable) so dispatch never dead-ends.

Adapters take and return torch tensors with the same ergonomics as the existing
``kernel_set.<module>`` wrappers; they raise only when actually invoked on an
unsupported host (the dispatcher never calls an arch-/import-gated adapter).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

KERNEL_SET = "kernel-set"
SGL_KERNEL = "sgl-kernel"


@dataclass
class Provider:
    """One ranked backend for one logical op."""

    name: str                       # short, stable id (matches registry `lib`)
    rank: int                       # 1 = best; ascending preference
    min_sm: int                     # arch gate (compute capability)
    dtypes: str                     # free-form supported dtype string
    import_check: str               # probe snippet (top module(s) extracted)
    call: Callable                  # lazy adapter; imports lib on first use
    note: str = ""                  # short human note (perf / source)


@dataclass
class Op:
    """A logical op with its ranked provider chain (ks fallback appended)."""

    name: str
    domain: str
    abi: Optional[str]              # kernel-set C ABI symbol (None if no ks op)
    providers: List[Provider] = field(default_factory=list)


# =========================================================================== #
# Lazy import helper — keep heavy libs out of import time.
# =========================================================================== #
def _imp(modpath: str):
    import importlib

    return importlib.import_module(modpath)


# External-provider adapters for the newly-wired quant ops (NVFP4/MXFP4 GEMM,
# per-token-group fp8 quant, fp8 KV-cache, fp8 attention). Import-safe: every
# adapter does its heavy imports lazily inside the function body.
from ._quant_ext import (  # noqa: E402
    _nvfp4_gemm_flashinfer, _nvfp4_gemm_vllm,
    _mxfp4_gemm_flashinfer, _mxfp4_gemm_vllm, _mxfp4_gemm_torchao,
    _nvfp4_quantize_vllm, _nvfp4_quantize_flashinfer,
    _reshape_and_cache_fp8_vllm,
    _per_token_group_quant_fp8_vllm, _per_token_group_quant_fp8_sgl,
    _per_token_group_quant_fp8_deepgemm,
    _fp8_attention_sage, _fp8_attention_flashinfer,
)


def _scale(head_dim: int, scale: Optional[float]) -> float:
    return float(scale) if scale else 1.0 / math.sqrt(head_dim)


class ProviderCallUnsupported(NotImplementedError):
    """Raised by an adapter when this provider cannot serve the requested
    optional call features, allowing dispatch to try the next provider."""


def _kw_meaningful(name, value) -> bool:
    if name == "softcap" or name == "logits_soft_cap":
        return value is not None and float(value) != 0.0
    if name == "window_size":
        return value not in (None, (-1, -1), [-1, -1])
    if name == "window_left":
        return value is not None and int(value) >= 0
    return value is not None


def _attention_extras_set(window_size=None, softcap=0.0, sinks=None,
                          custom_mask=None, packed_custom_mask=None,
                          alibi_slopes=None) -> bool:
    return (_kw_meaningful("window_size", window_size) or
            _kw_meaningful("softcap", softcap) or sinks is not None or
            custom_mask is not None or packed_custom_mask is not None or
            alibi_slopes is not None)


def _flashinfer_window_left(window_size, label: str):
    if not _kw_meaningful("window_size", window_size):
        return None
    if isinstance(window_size, (tuple, list)):
        if len(window_size) != 2:
            raise ProviderCallUnsupported(
                f"{label}: window_size must be (left, right)")
        left, right = window_size
        if right not in (None, -1, 0):
            raise ProviderCallUnsupported(
                f"{label}: FlashInfer supports left-window attention only")
        return int(left)
    return int(window_size)


def _call_with_optional_kwargs(fn, args, base_kwargs, optional_kwargs,
                               label: str):
    """Call ``fn`` with only requested optional kwargs it actually supports.

    Plain-path defaults are not passed, so older provider versions keep working.
    If a caller requested a real feature and this installed function has no such
    kwarg, raise ProviderCallUnsupported so dispatch can try the next backend.
    """
    requested = {k: v for k, v in optional_kwargs.items()
                 if _kw_meaningful(k, v)}
    call_kwargs = dict(base_kwargs)
    if requested:
        try:
            sig = inspect.signature(fn)
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in sig.parameters.values())
            missing = [k for k in requested
                       if not has_varkw and k not in sig.parameters]
        except (TypeError, ValueError):
            missing = []
        if missing:
            raise ProviderCallUnsupported(
                f"{label}: installed provider does not support "
                f"{', '.join(missing)}")
        call_kwargs.update(requested)
    try:
        return fn(*args, **call_kwargs)
    except TypeError as exc:
        msg = str(exc)
        if requested and ("unexpected keyword" in msg or
                          "got an unexpected" in msg):
            raise ProviderCallUnsupported(
                f"{label}: installed provider rejected optional attention "
                f"kwargs {sorted(requested)}") from exc
        raise


def _attention_ks_unsupported():
    raise NotImplementedError(
        "kernel-set attention fallback does not yet support "
        "window_size/softcap/sinks/custom_mask/packed_custom_mask/alibi_slopes; install "
        "flash-attn or flashinfer for these attention features.")


# =========================================================================== #
# ATTENTION — prefill (dense). q/k/v: (batch, seqlen, heads, head_dim).
# =========================================================================== #
def _attn_prefill_flash_attn(q, k, v, *, causal=True, softmax_scale=None,
                             window_size=None, softcap=0.0, sinks=None,
                             custom_mask=None, packed_custom_mask=None,
                             alibi_slopes=None, **_):
    if custom_mask is not None or packed_custom_mask is not None:
        raise ProviderCallUnsupported(
            "flash-attn dense prefill does not support custom/tree masks")
    fa = _imp("flash_attn")
    return _call_with_optional_kwargs(
        fa.flash_attn_func, (q, k, v),
        {"causal": causal, "softmax_scale": softmax_scale},
        {"window_size": window_size, "softcap": softcap, "sinks": sinks,
         "alibi_slopes": alibi_slopes},
        "flash-attn prefill")


def _attn_prefill_sdpa(q, k, v, *, causal=True, softmax_scale=None,
                       window_size=None, softcap=0.0, sinks=None,
                       custom_mask=None, packed_custom_mask=None,
                       alibi_slopes=None, **_):
    if _attention_extras_set(window_size, softcap, sinks, custom_mask,
                             packed_custom_mask, alibi_slopes):
        raise ProviderCallUnsupported(
            "torch SDPA adapter cannot serve window/softcap/sinks/custom_mask/alibi_slopes")
    import torch
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    enable_gqa = q.shape[-2] != k.shape[-2]
    o = torch.nn.functional.scaled_dot_product_attention(
        qt, kt, vt, is_causal=causal, scale=softmax_scale,
        enable_gqa=enable_gqa)
    return o.transpose(1, 2)


def _attn_prefill_flashinfer(q, k, v, *, causal=True, softmax_scale=None,
                             window_size=None, softcap=0.0, sinks=None,
                             custom_mask=None, packed_custom_mask=None,
                             alibi_slopes=None, **_):
    fi = _imp("flashinfer")
    if q.shape[0] != 1:
        raise NotImplementedError("flashinfer single_prefill adapter is b==1")
    window_left = _flashinfer_window_left(window_size, "flashinfer prefill")
    pos_mode = "ALIBI" if alibi_slopes is not None else "NONE"
    out = _call_with_optional_kwargs(
        fi.prefill.single_prefill_with_kv_cache, (q[0], k[0], v[0]),
        {"causal": causal, "kv_layout": "NHD", "sm_scale": softmax_scale,
         "pos_encoding_mode": pos_mode},
        {"window_left": window_left, "logits_soft_cap": softcap,
         "sinks": sinks, "custom_mask": custom_mask,
         "packed_custom_mask": packed_custom_mask},
        "flashinfer prefill")
    return out.unsqueeze(0)


def _attn_prefill_fa4(q, k, v, *, causal=True, softmax_scale=None,
                      window_size=None, softcap=0.0, sinks=None,
                      custom_mask=None, packed_custom_mask=None,
                      alibi_slopes=None, **_):
    # FlashAttention-4 (Blackwell sm100): the CuTe-DSL kernel exported as
    # flash_attn.cute.flash_attn_func. Same dense (b, s, h, d) ergonomics as FA2.
    if custom_mask is not None or packed_custom_mask is not None:
        raise ProviderCallUnsupported(
            "FlashAttention-4 dense prefill does not support custom/tree masks")
    fac = _imp("flash_attn.cute")
    return _call_with_optional_kwargs(
        fac.flash_attn_func, (q, k, v),
        {"causal": causal, "softmax_scale": softmax_scale},
        {"window_size": window_size, "softcap": softcap, "sinks": sinks,
         "alibi_slopes": alibi_slopes},
        "flash-attn-cute prefill")


def _attn_prefill_ks(q, k, v, *, causal=True, softmax_scale=None,
                     window_size=None, softcap=0.0, sinks=None,
                     custom_mask=None, packed_custom_mask=None,
                     alibi_slopes=None, **_):
    if _attention_extras_set(window_size, softcap, sinks, custom_mask,
                             packed_custom_mask, alibi_slopes):
        _attention_ks_unsupported()
    from .. import attention
    import torch
    b, s, qh, hd = q.shape
    kvh = k.shape[-2]
    out = torch.empty_like(q)
    attention.flash_attn(out, q, k, v, b, s, k.shape[1], qh, kvh, hd,
                         softmax_scale=_scale(hd, softmax_scale), causal=causal)
    return out


# =========================================================================== #
# ATTENTION — paged decode. q: (num_seqs, heads, head_dim); paged caches.
# =========================================================================== #
def _attn_decode_flashinfer(q, k_cache, v_cache, block_tables, seq_lens, *,
                            block_size, max_blocks_per_seq, softmax_scale=None,
                            window_size=None, softcap=0.0, sinks=None,
                            custom_mask=None, packed_custom_mask=None,
                            alibi_slopes=None, **_):
    import torch
    fi = _imp("flashinfer")
    num_seqs, qh, hd = q.shape
    kvh = k_cache.shape[1]
    device = q.device
    # ks layout (nb, kvh, page, hd) -> flashinfer NHD (nb, page, kvh, hd)
    k_fi = k_cache.permute(0, 2, 1, 3).contiguous()
    v_fi = v_cache.permute(0, 2, 1, 3).contiguous()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = fi.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD")
    bps = max_blocks_per_seq
    kv_indptr = torch.arange(0, (num_seqs + 1) * bps, bps,
                             device=device, dtype=torch.int32)
    kv_indices = torch.arange(num_seqs * bps, device=device, dtype=torch.int32)
    last = (seq_lens - (bps - 1) * block_size).clamp(min=1).to(torch.int32)
    window_left = _flashinfer_window_left(window_size, "flashinfer decode")
    pos_mode = "ALIBI" if alibi_slopes is not None else "NONE"
    _call_with_optional_kwargs(
        wrapper.plan,
        (kv_indptr, kv_indices, last, qh, kvh, hd, block_size),
        {"pos_encoding_mode": pos_mode, "data_type": q.dtype,
         "q_data_type": q.dtype},
        {"window_left": window_left, "logits_soft_cap": softcap,
         "sinks": sinks, "custom_mask": custom_mask,
         "packed_custom_mask": packed_custom_mask},
        "flashinfer decode plan")
    return wrapper.run(q, (k_fi, v_fi))


def _attn_decode_fa3(q, k_cache, v_cache, block_tables, seq_lens, *,
                     block_size, max_blocks_per_seq, softmax_scale=None,
                     window_size=None, softcap=0.0, sinks=None,
                     custom_mask=None, packed_custom_mask=None,
                     alibi_slopes=None, **_):
    if custom_mask is not None or packed_custom_mask is not None:
        raise ProviderCallUnsupported(
            "flash-attn-3 paged decode does not support custom/tree masks")
    from flash_attn_interface import flash_attn_with_kvcache
    qf = q.unsqueeze(1)
    k_fa = k_cache.permute(0, 2, 1, 3).contiguous()
    v_fa = v_cache.permute(0, 2, 1, 3).contiguous()
    out = _call_with_optional_kwargs(
        flash_attn_with_kvcache, (qf, k_fa, v_fa),
        {"page_table": block_tables, "cache_seqlens": seq_lens,
         "softmax_scale": softmax_scale, "causal": False},
        {"window_size": window_size, "softcap": softcap, "sinks": sinks,
         "alibi_slopes": alibi_slopes},
        "flash-attn-3 decode")
    return out.reshape_as(q)


def _attn_decode_ks(q, k_cache, v_cache, block_tables, seq_lens, *,
                    block_size, max_blocks_per_seq, softmax_scale=None,
                    window_size=None, softcap=0.0, sinks=None,
                    custom_mask=None, packed_custom_mask=None,
                    alibi_slopes=None, **_):
    if _attention_extras_set(window_size, softcap, sinks, custom_mask,
                             packed_custom_mask, alibi_slopes):
        _attention_ks_unsupported()
    from .. import attention
    import torch
    num_seqs, qh, hd = q.shape
    kvh = k_cache.shape[1]
    out = torch.empty_like(q)
    attention.paged_attn_decode(
        out, q, k_cache, v_cache, block_tables, seq_lens, num_seqs, qh, kvh, hd,
        block_size, max_blocks_per_seq, softmax_scale=_scale(hd, softmax_scale))
    return out


def _patch_embed_torch(x, weight, bias=None, *, stride=1, padding=0,
                       dilation=1, groups=1, **_):
    import torch
    if x.ndim == 4:
        return torch.nn.functional.conv2d(
            x, weight, bias, stride=stride, padding=padding,
            dilation=dilation, groups=groups)
    if x.ndim == 5:
        return torch.nn.functional.conv3d(
            x, weight, bias, stride=stride, padding=padding,
            dilation=dilation, groups=groups)
    raise ValueError("patch_embed expects a 4D image or 5D video tensor")


def _flex_attention_torch(q, k, v, *, score_mod=None, block_mask=None,
                          mask_mod=None, create_mask_kwargs=None, **kw):
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )
    if block_mask is None and mask_mod is not None:
        create_mask_kwargs = dict(create_mask_kwargs or {})
        block_mask = create_block_mask(mask_mod, **create_mask_kwargs)
    call_kw = dict(kw)
    if score_mod is not None:
        call_kw["score_mod"] = score_mod
    if block_mask is not None:
        call_kw["block_mask"] = block_mask
    return flex_attention(q, k, v, **call_kw)


def _varlen_pad_flash_attn(x, indices=None, *, mode="unpad",
                           attention_mask=None, batch=None, seqlen=None,
                           cu_seqlens=None, max_seqlen=None, **kw):
    from flash_attn.bert_padding import (
        index_first_axis,
        pad_input,
        unpad_input,
    )
    if mode in ("unpad", "unpack"):
        if attention_mask is not None:
            return unpad_input(x, attention_mask, **kw)
        if indices is None:
            raise ValueError(
                "varlen_pad(mode='unpad') requires attention_mask or indices")
        flat = x.reshape(-1, *x.shape[2:]) if getattr(x, "ndim", 0) > 2 else x
        return index_first_axis(flat, indices), indices, cu_seqlens, max_seqlen
    if mode in ("pad", "pack"):
        if indices is None or batch is None or seqlen is None:
            raise ValueError(
                "varlen_pad(mode='pad') requires indices, batch, and seqlen")
        return pad_input(x, indices, batch, seqlen)
    raise ValueError("varlen_pad mode must be 'unpad'/'unpack' or 'pad'/'pack'")


# =========================================================================== #
# GEMM (dense fp16/bf16). a: (M,K), b: (K,N) -> (M,N).
# =========================================================================== #
def _gemm_torch(a, b, **_):
    return a @ b


def _gemm_ks(a, b, **_):
    from .. import gemm
    import torch
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    gemm.gemm(c, a, b, m=m, n=n, k=k)
    return c


# =========================================================================== #
# FP8 GEMM (blockwise / scaled). a,b fp8; per-tensor or block scales.
# =========================================================================== #
def _fp8_gemm_deepgemm(a8, b8, a_scale, b_scale, *, out_dtype=None, **_):
    import torch
    dg = _imp("deep_gemm")
    m = a8.shape[0]
    n = b8.shape[0]
    out = torch.empty(m, n, device=a8.device,
                      dtype=out_dtype or torch.bfloat16)
    dg.fp8_gemm_nt((a8, a_scale), (b8, b_scale), out)
    return out


def _fp8_gemm_torch(a8, b8, a_scale, b_scale, *, out_dtype=None, **_):
    import torch
    return torch._scaled_mm(a8, b8.t(), scale_a=a_scale, scale_b=b_scale,
                            out_dtype=out_dtype or torch.bfloat16,
                            use_fast_accum=True)


def _fp8_gemm_vllm_cutlass(a8, b8, a_scale, b_scale, *, out_dtype=None,
                           bias=None, **_):
    # vLLM CUTLASS scaled-mm. Public fp8_gemm uses the DeepGEMM NT weight layout
    # (B as [N,K]); vLLM expects the column-major [K,N] operand.
    import torch
    from vllm import _custom_ops as ops
    b_arg = b8.t().contiguous() if getattr(b8, "ndim", 0) == 2 else b8
    return ops.cutlass_scaled_mm(a8, b_arg, a_scale, b_scale,
                                 out_dtype or torch.bfloat16, bias)


def _fp8_gemm_fbgemm(a8, b8, a_scale, b_scale, *, out_dtype=None, **_):
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401 - registers torch.ops
    import torch
    return torch.ops.fbgemm.f8f8bf16_rowwise(a8, b8, a_scale, b_scale)


def _fp8_gemm_ks(a8, b8, a_scale, b_scale, *, out_dtype=None, **_):
    # kernel-set has no native fp8 GEMM; the closest ABI is int8 w8a8. We expose
    # the ks dense GEMM symbol as the portable fallback path here.
    from .. import gemm
    import torch
    m, k = a8.shape
    n = b8.shape[0]
    c = torch.empty(m, n, device=a8.device, dtype=out_dtype or torch.bfloat16)
    gemm.gemm(c, a8.to(c.dtype), b8.to(c.dtype).t().contiguous(),
              m=m, n=n, k=k)
    return c


def _fp8_gemm_blockwise_ks(a8, b8, a_scale, b_scale, *, block_n=128,
                           block_k=128, out_dtype=None, **_):
    # Native blockwise fp8 GEMM (ks_gemm_fp8_blockwise): the portable sm80+
    # terminal for the DeepSeek-V3 recipe. A is [M,K], B is [K,N] (row-major),
    # a_scale [M, ceil(K/block_k)], b_scale [ceil(K/block_k), ceil(N/block_n)].
    from .. import gemm
    import torch
    m, k = a8.shape
    n = b8.shape[1]
    out = torch.empty(m, n, device=a8.device, dtype=out_dtype or torch.bfloat16)
    gemm.gemm_fp8_blockwise(out, a8, b8, a_scale, b_scale, m=m, n=n, k=k,
                            block_n=block_n, block_k=block_k)
    return out


def _per_token_group_quant_ks(x, *, group_size=128, fp8_dtype=None, **_):
    # Native per-token-group fp8 quant (ks_quantize_fp8_group): the activation
    # format the blockwise fp8 GEMM consumes. Returns (fp8 out, fp32 scale).
    from .. import quant
    from ..enums import DType
    import torch
    rows, cols = x.shape
    ngroups = (cols + group_size - 1) // group_size
    out = torch.empty(rows, cols, device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(rows, ngroups, device=x.device, dtype=torch.float32)
    quant.quantize_fp8_group(out, scale, x, rows=rows, cols=cols,
                             group_size=group_size,
                             fp8_dtype=fp8_dtype or DType.F8E4M3)
    return out, scale


def _ks_unsupported(label):
    """A kernel-set terminal adapter for ops with NO portable ks kernel (fp4,
    fp8 attention, fp8 KV-cache). It keeps the chain validly terminated at
    kernel-set for the optimal table, but raises a clear, actionable error if it
    is ever actually selected (i.e. no external provider is installed). These ops
    require a Blackwell/Hopper GPU + the relevant library by design."""
    def _raise(*_a, **_k):
        raise NotImplementedError(
            f"{label}: no portable kernel-set implementation. Install a provider "
            "(FlashInfer / vLLM / DeepGEMM / SGLang / SageAttention) — this op "
            "dispatches to the external backend on supported GPUs.")
    return _raise


def _vllm_scalar_type(name: str):
    from vllm.scalar_type import scalar_types
    return getattr(scalar_types, name)


def _marlin_workspace(n: int, device):
    import torch
    return torch.zeros(max(1, n // 64 * 16), device=device, dtype=torch.int32)


# =========================================================================== #
# W4A16 GEMM. a: fp16/bf16 (M,K); packed int4 weights + group scales/zeros.
# vLLM owns the SOTA mixed-input INT4xFP16 kernels: GPTQ/AWQ-Marlin (sm80+) and
# Machete (sm90a, CUTLASS TMA+WGMMA weight-prepack — beats Marlin on Hopper).
# =========================================================================== #
def _w4a16_marlin(a, b_packed, scales, zeros, *, group_size=128,
                  workspace=None, g_idx=None, perm=None, is_zp_float=False, **_):
    # Unified vLLM Marlin. GPTQ symmetric weights use uint4b8; AWQ/zero-point
    # weights use uint4 and pass b_zeros. Weights are already Marlin-prepacked.
    from vllm import _custom_ops as ops
    m, k = a.shape
    n = scales.shape[1]
    if workspace is None:
        workspace = _marlin_workspace(n, a.device)
    b_q_type = _vllm_scalar_type("uint4" if zeros is not None else "uint4b8")
    return ops.marlin_gemm(
        a, None, b_packed, None, scales, None, None, zeros, g_idx, perm,
        workspace, b_q_type, m, n, k, True, False, False, is_zp_float)


def _w4a16_machete(a, b_packed, scales, zeros, *, group_size=128, **_):
    # vLLM Machete (sm90a): CUTLASS TMA+WGMMA mixed-input GEMM with weight
    # pre-packing — the Hopper-optimal W4A16/W4A8 path (beats Marlin on sm90).
    # ops.machete_mm(A, B, b_type, out_type, group_scales, group_zeros,
    #   group_size, channel_scales, token_scales, schedule). Weights are
    # machete_prepack_B-packed by the caller.
    from vllm import _custom_ops as ops
    return ops.machete_mm(
        a, b_packed, _vllm_scalar_type("uint4b8"), a.dtype,
        scales, zeros, group_size, None, None, None)


def _w4a16_gemlite(a, b_packed, scales, zeros, *, group_size=128, bias=None,
                   out_dtype=None, **_):
    import torch
    try:
        from gemlite import DType
    except Exception:
        DType = None
    from gemlite.core import GemLiteLinear
    m, k = a.shape
    n = scales.shape[1] if getattr(scales, "ndim", 0) >= 2 else b_packed.shape[0]
    if DType is not None:
        dt = DType.BF16 if a.dtype is torch.bfloat16 else DType.FP16
        kwargs = {"input_dtype": dt, "output_dtype": dt}
    else:
        kwargs = {"input_dtype": a.dtype, "output_dtype": out_dtype or a.dtype}
    layer = GemLiteLinear(W_nbits=4, group_size=group_size, in_features=k,
                          out_features=n, **kwargs)
    layer.pack(b_packed, scales, zeros, bias)
    return layer(a.reshape(-1, k)).reshape(*a.shape[:-1], n)


def _w4a16_torchao_int4(a, b_packed, scales, zeros=None, *, group_size=128,
                        **_):
    import torch
    from torchao.quantization import Int4WeightOnlyConfig, quantize_  # noqa: F401
    if zeros is not None:
        raise NotImplementedError(
            "torchao-int4 expects prepacked scales_and_zeros in `scales`")
    return torch.ops.aten._weight_int4pack_mm(a, b_packed, group_size, scales)


def _w4a8_machete(a8, b_packed, b_scales=None, a_scales=None, *,
                  b_zeros=None, group_size=None, out_dtype=None,
                  b_channel_scales=None, a_token_scales=None, schedule=None,
                  **_):
    # Machete W4A8: int4 weight with fp8/int8 activation and token/channel scales.
    import torch
    from vllm import _custom_ops as ops
    channel_scales = b_channel_scales if b_channel_scales is not None else b_scales
    token_scales = a_token_scales if a_token_scales is not None else a_scales
    return ops.machete_mm(
        a8, b_packed, _vllm_scalar_type("int4"),
        out_dtype or torch.bfloat16, None, b_zeros, group_size,
        channel_scales, token_scales, schedule)


def _w4a8_marlin(a8, b_packed, b_scales, a_scales=None, *, global_scale=None,
                 workspace=None, size_m=None, size_n=None, size_k=None,
                 **_):
    # Unified Marlin QQQ/W4A8: signed int4 weights + int8/fp8 activations.
    from vllm import _custom_ops as ops
    m, k = a8.shape
    size_m = size_m or m
    size_k = size_k or k
    size_n = size_n or (b_scales.shape[-1] if b_scales is not None
                        else b_packed.shape[1])
    if workspace is None:
        workspace = _marlin_workspace(size_n, a8.device)
    return ops.marlin_gemm(
        a8, None, b_packed, None, b_scales, a_scales, global_scale,
        None, None, None, workspace, _vllm_scalar_type("int4"),
        size_m, size_n, size_k, True, False, False, False)


def _w8a16_fp8_marlin(a, b_packed, b_scales, *, global_scale=None,
                      workspace=None, size_m=None, size_n=None, size_k=None,
                      g_idx=None, perm=None, bias=None, use_fp32_reduce=False,
                      **_):
    # Unified Marlin FP8 weight-only: fp16/bf16 activations with fp8-e4m3
    # weights dequantized in software. On sm89+ the native fp8_gemm op is
    # preferred for true FP8 tensor-core GEMM; this op covers the sm80/86/89
    # no-fp8-TC / FP8-checkpoint niche without overloading int8_gemm.
    from vllm import _custom_ops as ops
    m, k = a.shape
    size_m = size_m or m
    size_k = size_k or k
    size_n = size_n or (b_scales.shape[-1] if b_scales is not None
                        else b_packed.shape[1])
    if workspace is None:
        workspace = _marlin_workspace(size_n, a.device)
    return ops.marlin_gemm(
        a, None, b_packed, bias, b_scales, None, global_scale,
        None, g_idx, perm, workspace, _vllm_scalar_type("float8_e4m3fn"),
        size_m, size_n, size_k, True, False, use_fp32_reduce, False)


def _sparse_2_4_gemm_vllm(a, bt_meta, bt_q, scale_a, scale_b, *,
                          out_dtype=None, bias=None, **_):
    # CUTLASS 2:4 sparse scaled GEMM over weights compressed offline via
    # ops.cutlass_sparse_compress. The int4+2:4 Sparse-Marlin sub-path is
    # deferred because vLLM removed gptq_marlin_24_gemm; it needs vendoring
    # IST-DASLab/Sparse-Marlin before we can expose a callable provider.
    from vllm import _custom_ops as ops
    return ops.cutlass_scaled_sparse_mm(
        a, bt_meta, bt_q, scale_a, scale_b, out_dtype or a.dtype, bias)


def _bitnet_gemm_bitblas(a, b_ternary, scale=None, *, out_dtype=None,
                         matmul=None, config=None, **kw):
    # BitBLAS is the tractable Python-installable W1.58/A8 provider. The native
    # microsoft/BitNet CUDA kernels use source-build-only 2-bit packed dp4a and
    # are deferred until a vendored/native target lands.
    if matmul is None:
        from bitblas import Matmul
        if config is None:
            m, k = a.shape
            n = b_ternary.shape[0] if getattr(b_ternary, "ndim", 0) == 2 \
                else kw.pop("size_n", None)
            out_name = str(out_dtype or getattr(a, "dtype", "float16"))
            if out_name.startswith("torch."):
                out_name = out_name[len("torch."):]
            config = {
                "M": kw.pop("size_m", m),
                "N": kw.pop("size_n", n),
                "K": kw.pop("size_k", k),
                "A_dtype": "int8" if "int8" in str(getattr(a, "dtype", ""))
                else "float16",
                "W_dtype": "int2",
                "out_dtype": out_name,
                "accum_dtype": "int32",
                "layout": "nt",
                "bitnet": True,
            }
        matmul = Matmul(config)
    if scale is not None:
        try:
            return matmul(a, b_ternary, scale=scale, out_dtype=out_dtype, **kw)
        except TypeError:
            return matmul(a, b_ternary, scale, **kw)
    return matmul(a, b_ternary, **kw)


def _w4a16_ks(a, b_packed, scales, zeros, *, group_size=128, **_):
    from .. import gemm
    import torch
    m, k = a.shape
    n = scales.shape[1]
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    gemm.gemm_w4a16(c, a, b_packed, scales, zeros, m=m, n=n, k=k,
                    group_size=group_size)
    return c


# =========================================================================== #
# RMSNORM. x: (rows, hidden); w: (hidden,).
# =========================================================================== #
def _rmsnorm_flashinfer(x, w, *, eps=1e-6, **_):
    from flashinfer.norm import rmsnorm as fi_rms
    return fi_rms(x, w, eps=eps)


def _rmsnorm_quack(x, w, *, eps=1e-6, **_):
    from quack import rmsnorm
    return rmsnorm(x, w, eps)


def _rmsnorm_vllm(x, w, *, eps=1e-6, **_):
    import torch
    from vllm import _custom_ops as ops
    out = torch.empty_like(x)
    ops.rms_norm(out, x, w, eps)
    return out


def _rmsnorm_liger(x, w, *, eps=1e-6, **_):
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    return LigerRMSNormFunction.apply(x, w, eps, 0.0, "llama")


def _rmsnorm_ks(x, w, *, eps=1e-6, **_):
    from .. import norm
    import torch
    out = torch.empty_like(x)
    norm.rms_norm(out, x, w, eps=eps)
    return out


# =========================================================================== #
# FUSED_ADD_RMSNORM. in-place: residual += x; x = rmsnorm(residual)*w.
# Adapter returns (normed, new_residual) out-of-place for ergonomic parity.
# =========================================================================== #
def _far_flashinfer(x, residual, w, *, eps=1e-6, **_):
    from flashinfer.norm import fused_add_rmsnorm as fi_far
    xc, rc = x.clone(), residual.clone()
    fi_far(xc, rc, w, eps=eps)
    return xc, rc


def _far_vllm(x, residual, w, *, eps=1e-6, **_):
    from vllm import _custom_ops as ops
    xc, rc = x.clone(), residual.clone()
    ops.fused_add_rms_norm(xc, rc, w, eps)
    return xc, rc


def _far_ks(x, residual, w, *, eps=1e-6, **_):
    from .. import norm
    import torch
    out = torch.empty_like(x)
    res_out = torch.empty_like(residual)
    norm.rms_norm_residual(out, res_out, x, residual, w, eps=eps)
    return out, res_out


# =========================================================================== #
# ROPE. q: (tokens, qheads, hd); k: (tokens, kvheads, hd); cos/sin (tokens,hd/2)
# Returns (q_rot, k_rot). NeoX / rotate_half convention.
# =========================================================================== #
def _rope_flashinfer(q, k, cos, sin, *, interleaved=False, **_):
    import torch
    from flashinfer.rope import apply_rope_with_cos_sin_cache
    tokens, qh, hd = q.shape
    kvh = k.shape[1]
    positions = torch.arange(tokens, device=q.device, dtype=torch.int32)
    cos_sin_cache = torch.cat([cos, sin], dim=-1).contiguous()
    qf = q.reshape(tokens, qh * hd)
    kf = k.reshape(tokens, kvh * hd)
    q_out, k_out = apply_rope_with_cos_sin_cache(
        positions, qf, kf, hd, cos_sin_cache, is_neox=not interleaved)
    return q_out.reshape(tokens, qh, hd), k_out.reshape(tokens, kvh, hd)


def _rope_vllm(q, k, cos, sin, *, interleaved=False, **_):
    import torch
    from vllm import _custom_ops as ops
    tokens, qh, hd = q.shape
    kvh = k.shape[1]
    positions = torch.arange(tokens, device=q.device, dtype=torch.int64)
    cos_sin_cache = torch.cat([cos, sin], dim=-1).contiguous()
    qf = q.reshape(tokens, qh * hd).clone()
    kf = k.reshape(tokens, kvh * hd).clone()
    ops.rotary_embedding(positions, qf, kf, hd, cos_sin_cache,
                         not interleaved)
    return qf.reshape(tokens, qh, hd), kf.reshape(tokens, kvh, hd)


def _rope_liger(q, k, cos, sin, *, interleaved=False, **_):
    # Liger Triton RoPE (LigerRopeFunction). It wants q (b, n_qh, s, d),
    # k (b, n_kvh, s, d) and full-width cos/sin (1, s, d); our ergonomic layout
    # is (tokens, heads, head_dim) with half-width cos/sin (tokens, head_dim/2).
    # Mirror the bench_sota.py adapter: add a batch dim, transpose to heads-major,
    # double the cos/sin to full width, then map the result back.
    import torch
    from liger_kernel.ops.rope import LigerRopeFunction
    tokens, qh, hd = q.shape
    kvh = k.shape[1]
    qb = q.unsqueeze(0).transpose(1, 2)   # (1, qh, s, d)
    kb = k.unsqueeze(0).transpose(1, 2)   # (1, kvh, s, d)
    cos_full = torch.cat([cos, cos], dim=-1).unsqueeze(0)  # (1, s, d)
    sin_full = torch.cat([sin, sin], dim=-1).unsqueeze(0)
    q_rot, k_rot = LigerRopeFunction.apply(qb, kb, cos_full, sin_full)
    return (q_rot.transpose(1, 2).reshape(tokens, qh, hd),
            k_rot.transpose(1, 2).reshape(tokens, kvh, hd))


def _mrope_vllm(q, k, cos, sin, mrope_section, *, positions=None,
                mrope_interleaved=False, rotary_dim=None, **kw):
    # vLLM Triton mRoPE for Qwen2.5/3-VL: 3-axis sections and optional partial
    # rotary_dim. The installed vLLM function owns layout details.
    from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope
    return triton_mrope(
        q, k, cos, sin, mrope_section, positions=positions,
        mrope_interleaved=mrope_interleaved, rotary_dim=rotary_dim, **kw)


def _rope_ks(q, k, cos, sin, *, interleaved=False, **_):
    from .. import rope
    tokens, qh, hd = q.shape
    kvh = k.shape[1]
    qc, kc = q.clone(), k.clone()
    rope.rope_inplace(qc, kc, cos, sin, tokens, qh, kvh, hd,
                      interleaved=interleaved)
    return qc, kc


# =========================================================================== #
# SWIGLU (silu_and_mul). gate, up: (rows, inter) -> (rows, inter).
# =========================================================================== #
def _swiglu_flashinfer(gate, up, **_):
    import torch
    from flashinfer.activation import silu_and_mul
    packed = torch.cat([gate, up], dim=-1).contiguous()
    return silu_and_mul(packed)


def _swiglu_vllm(gate, up, **_):
    import torch
    from vllm import _custom_ops as ops
    packed = torch.cat([gate, up], dim=-1).contiguous()
    out = torch.empty(gate.shape[0], gate.shape[1], device=gate.device,
                      dtype=gate.dtype)
    ops.silu_and_mul(out, packed)
    return out


def _swiglu_liger(gate, up, **_):
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    return LigerSiLUMulFunction.apply(gate, up)


def _fused_rmsnorm_gated_fla(x, weight, gate, *, eps=1e-6,
                             activation="silu", **kw):
    import torch
    from fla.modules import FusedRMSNormGated
    mod = FusedRMSNormGated(weight.shape[-1], eps=eps, activation=activation,
                            **kw)
    if hasattr(mod, "to"):
        mod = mod.to(device=x.device, dtype=x.dtype)
    if hasattr(mod, "weight"):
        with torch.no_grad():
            mod.weight.copy_(weight)
    return mod(x, gate)


def _fused_rmsnorm_gated_ks(x, weight, gate, *, eps=1e-6, activation="silu", **_):
    # Portable kernel-set fallback (matches FLA convention: norm then gate;
    # GPU-verified vs FLA FusedRMSNormGated, rel ~2.5e-3).
    import torch
    from .. import norm as _norm
    out = torch.empty_like(x)
    _norm.fused_rmsnorm_gated(out, x, weight, gate, activation=activation, eps=eps)
    return out


def _swiglu_ks(gate, up, **_):
    from .. import activation
    import torch
    out = torch.empty_like(gate)
    activation.swiglu(out, gate, up)
    return out


# =========================================================================== #
# CROSS_ENTROPY. logits: (tokens, vocab); targets: (tokens,) int64.
# Returns per-token loss (reduction="none").
# =========================================================================== #
def _ce_liger(logits, targets, *, ignore_index=-100, **_):
    from liger_kernel.transformers.functional import liger_cross_entropy
    out = liger_cross_entropy(logits, targets, ignore_index=ignore_index,
                              reduction="none")
    return out[0] if isinstance(out, (tuple, list)) else out


def _ce_quack(logits, targets, *, ignore_index=-100, **_):
    from quack import cross_entropy
    out = cross_entropy(logits, targets, ignore_index=ignore_index)
    return out[0] if isinstance(out, (tuple, list)) else out


def _ce_torch(logits, targets, *, ignore_index=-100, **_):
    import torch
    return torch.nn.functional.cross_entropy(
        logits, targets, ignore_index=ignore_index, reduction="none")


def _ce_ks(logits, targets, *, ignore_index=-100, **_):
    from .. import loss
    import torch
    num_tokens, vocab = logits.shape
    losses = torch.empty(num_tokens, device=logits.device, dtype=torch.float32)
    grad = torch.empty_like(logits)
    loss.cross_entropy(losses, grad, logits, targets, num_tokens, vocab,
                       ignore_index=ignore_index)
    return losses


def _fused_linear_ce_liger(hidden, lm_head_weight, targets, *, bias=None,
                           ce_weight=None, ignore_index=-100,
                           lse_square_scale=0.0, label_smoothing=0.0,
                           reduction="mean", softcap=None,
                           return_z_loss=False, accum_dtype=None,
                           use_token_scaling=False,
                           return_token_accuracy=False,
                           return_predicted_tokens=False, **_):
    from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction,
    )
    out = LigerFusedLinearCrossEntropyFunction.apply(
        hidden, lm_head_weight, targets, bias, ce_weight, ignore_index,
        lse_square_scale, label_smoothing, reduction, softcap, return_z_loss,
        accum_dtype, use_token_scaling, return_token_accuracy,
        return_predicted_tokens)
    if return_z_loss or return_token_accuracy or return_predicted_tokens:
        return out
    return out[0] if isinstance(out, (tuple, list)) else out


def _fused_linear_ce_ks(hidden, lm_head_weight, targets, *, ignore_index=-100,
                        label_smoothing=0.0, chunk_size=0,
                        reduction="mean", **_):
    from .. import loss
    import torch
    num_tokens, hidden_dim = hidden.shape
    vocab = lm_head_weight.shape[0]
    losses = torch.empty(num_tokens, device=hidden.device, dtype=torch.float32)
    grad_hidden = torch.empty_like(hidden)
    grad_weight_fp32 = torch.empty(vocab, hidden_dim,
                                   device=lm_head_weight.device,
                                   dtype=torch.float32)
    loss.fused_linear_cross_entropy(
        losses, grad_hidden, grad_weight_fp32, hidden, lm_head_weight, targets,
        num_tokens, hidden_dim, vocab, ignore_index=ignore_index,
        label_smoothing=label_smoothing, chunk_size=chunk_size)
    if reduction == "none":
        return losses
    valid = targets != ignore_index
    selected = losses[valid] if getattr(valid, "ndim", 0) else losses
    if reduction == "sum":
        return selected.sum()
    if reduction == "mean":
        return selected.mean()
    return losses


def _muon_torch(grad, *, steps=5, eps=1e-7, **_):
    import torch  # noqa: F401
    # Muon is matmul-bound on the tensor backend; torch/cuBLAS is the provider
    # path until a dedicated optimizer kernel is justified.
    a, b, c = 3.4445, -4.7750, 2.0315
    X = grad.bfloat16()
    tr = X.shape[-2] > X.shape[-1]
    if tr:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if tr:
        X = X.mT
    return X.to(grad.dtype)


# =========================================================================== #
# MOE (fused experts). hidden, w1, w2, topk_weights, topk_ids.
# =========================================================================== #
def _moe_vllm(hidden, w1, w2, topk_weights, topk_ids, **_):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    return fused_experts(hidden, w1, w2, topk_weights, topk_ids, inplace=False)


def _moe_flashinfer_cutlass(hidden, w1, w2, topk_weights, topk_ids, *,
                            out_dtype=None, quant_scales=None, **kw):
    from flashinfer.fused_moe import cutlass_fused_moe
    return cutlass_fused_moe(
        hidden, topk_ids, topk_weights, w1, w2, out_dtype,
        quant_scales=quant_scales, **kw)


def _moe_ks(a, b, expert_offsets, *, num_experts, n, k, **_):
    from .. import moe
    import torch
    total_rows = a.shape[0]
    c = torch.empty(total_rows, n, device=a.device, dtype=a.dtype)
    moe.grouped_gemm(c, a, b, expert_offsets, num_experts, total_rows, n, k)
    return c


# =========================================================================== #
# SGL-KERNEL adapters. SGLang's `sgl_kernel` is the alignment target for the
# hard ops. Each adapter imports `sgl_kernel` lazily and calls the EXACT public
# API exported from its vendored python/ (third_party/sglang/sgl-kernel/). It is
# ranked #1 for the MoE gate ops (its specialty) and competitively elsewhere.
# =========================================================================== #
def _rmsnorm_sgl(x, w, *, eps=1e-6, **_):
    # sgl_kernel.rmsnorm(input, weight, eps, out=None, enable_pdl=None) -> out
    sk = _imp("sgl_kernel")
    return sk.rmsnorm(x, w, eps)


def _far_sgl(x, residual, w, *, eps=1e-6, **_):
    # sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps) is in-place:
    # residual += input; input = rmsnorm(residual)*weight. Clone for ergonomic
    # out-of-place parity with the other fused_add_rmsnorm adapters.
    sk = _imp("sgl_kernel")
    xc, rc = x.clone(), residual.clone()
    sk.fused_add_rmsnorm(xc, rc, w, eps)
    return xc, rc


def _gemma_rmsnorm_sgl(x, w, *, eps=1e-6, **_):
    # sgl_kernel.gemma_rmsnorm(input, weight, eps, out=None) -> out
    # out = (x / RMS(x)) * (weight + 1)
    sk = _imp("sgl_kernel")
    return sk.gemma_rmsnorm(x, w, eps)


def _gemma_rmsnorm_flashinfer(x, w, *, eps=1e-6, **_):
    # flashinfer.norm.gemma_rmsnorm(input, weight, eps=..., out=None) -> out.
    from flashinfer.norm import gemma_rmsnorm
    return gemma_rmsnorm(x, w, eps=eps)


def _gemma_rmsnorm_ks(x, w, *, eps=1e-6, **_):
    # kernel-set has no gemma-specific wrapper; realize the (weight+1) scale via
    # the portable rms_norm path so the fallback never imports sgl_kernel.
    from .. import norm
    import torch
    out = torch.empty_like(x)
    norm.rms_norm(out, x, w + 1, eps=eps)
    return out


def _rope_sgl(q, k, cos, sin, *, interleaved=False, **_):
    # sgl_kernel.rotary_embedding(positions, query, key, head_size,
    #     cos_sin_cache, is_neox=True) updates query/key in place. It expects
    # flattened (tokens, heads*head_dim) q/k and a (max_pos, head_dim) cache.
    import torch
    sk = _imp("sgl_kernel")
    tokens, qh, hd = q.shape
    kvh = k.shape[1]
    positions = torch.arange(tokens, device=q.device, dtype=torch.int64)
    cos_sin_cache = torch.cat([cos, sin], dim=-1).contiguous()
    qf = q.reshape(tokens, qh * hd).clone()
    kf = k.reshape(tokens, kvh * hd).clone()
    sk.rotary_embedding(positions, qf, kf, hd, cos_sin_cache,
                        is_neox=not interleaved)
    return qf.reshape(tokens, qh, hd), kf.reshape(tokens, kvh, hd)


def _swiglu_sgl(gate, up, **_):
    # sgl_kernel.silu_and_mul(input, out=None) -> out; input is the packed
    # (rows, 2*inter) gate||up tensor.
    import torch
    sk = _imp("sgl_kernel")
    packed = torch.cat([gate, up], dim=-1).contiguous()
    return sk.silu_and_mul(packed)


def _fp8_gemm_sgl(a8, b8, a_scale, b_scale, *, out_dtype=None, **_):
    # sgl_kernel.fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype,
    #     bias=None) -> Tensor. mat_b is the (K,N) column-major fp8 operand.
    import torch
    sk = _imp("sgl_kernel")
    return sk.fp8_scaled_mm(a8, b8.t().contiguous().t(), a_scale, b_scale,
                            out_dtype or torch.bfloat16)


def _int8_gemm_sgl(a8, b8, a_scale, b_scale, *, out_dtype=None, **_):
    # sgl_kernel.int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype,
    #     bias=None) -> Tensor. a8 int8 (M,K); b8 int8 (K,N).
    import torch
    sk = _imp("sgl_kernel")
    return sk.int8_scaled_mm(a8, b8, a_scale, b_scale,
                             out_dtype or torch.bfloat16)


def _int8_gemm_vllm(a8, b8, a_scale, b_scale, *, out_dtype=None, **_):
    # vLLM CUTLASS INT8 W8A8 (SmoothQuant / compressed-tensors, symmetric +
    # azp): ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None).
    # The registry's true rank-1 INT8 kernel across sm75-sm90. a8 int8 (M,K);
    # b8 int8 (K,N) column-major.
    import torch
    from vllm import _custom_ops as ops
    return ops.cutlass_scaled_mm(a8, b8, scale_a=a_scale, scale_b=b_scale,
                                 out_dtype=out_dtype or torch.bfloat16)


def _int8_gemm_marlin(a8, b8, a_scale, b_scale, *, out_dtype=None, **_):
    # Unified vLLM Marlin INT8. a8 is int8/Char with per-token a_scale; b8 is the
    # Marlin-prepacked signed-int8 weight with per-channel/group b_scale.
    from vllm import _custom_ops as ops
    m, k = a8.shape
    n = b8.shape[1]
    workspace = _marlin_workspace(n, a8.device)
    return ops.marlin_gemm(
        a8, None, b8, None, b_scale, a_scale, None, None, None, None,
        workspace, _vllm_scalar_type("int8"), m, n, k, True, False, False,
        False)


def _int8_gemm_gemlite(a8, b8, a_scale=None, b_scale=None, *, out_dtype=None,
                       **kw):
    helper = _imp("gemlite.helper")
    if hasattr(helper, "A8W8_INT8_dynamic"):
        return helper.A8W8_INT8_dynamic(
            a8, b8, a_scale=a_scale, b_scale=b_scale, out_dtype=out_dtype, **kw)
    raise NotImplementedError("gemlite A8W8_INT8_dynamic helper is unavailable")


def _int8_gemm_ks(a8, b8, a_scale, b_scale, *, out_dtype=None, **_):
    from .. import gemm
    import torch
    m, k = a8.shape
    n = b8.shape[1]
    out = torch.empty(m, n, device=a8.device, dtype=out_dtype or torch.bfloat16)
    out_dt = None
    try:
        from .. import dtype_to_ks
        out_dt = dtype_to_ks(out.dtype)
    except Exception:
        pass
    gemm.gemm_w8a8(out, a8, b8, a_scale, b_scale, m=m, n=n, k=k,
                   out_dtype=out_dt)
    return out


def _sampling_sgl(probs, *, top_k=None, top_p=None, **_):
    # sgl_kernel exports the fused renorm-by-threshold sampling primitives on the
    # CUDA path: top_k_renorm_prob(probs, top_k) and top_p_renorm_prob(probs,
    # top_p). Together with categorical sampling these realize top-k/top-p
    # filtering (top_k_first order). Public dispatch returns sampled token ids.
    import torch
    sk = _imp("sgl_kernel")
    out = probs
    if top_k is not None:
        out = sk.top_k_renorm_prob(out, top_k)
    if top_p is not None:
        out = sk.top_p_renorm_prob(out, top_p)
    return torch.multinomial(out, num_samples=1).squeeze(-1).to(torch.int32)


def _sampling_flashinfer(probs, *, top_k=None, top_p=None, **_):
    fs = _imp("flashinfer.sampling")
    if top_k is not None and top_p is not None:
        return fs.top_k_top_p_sampling_from_probs(probs, top_k, top_p)
    if top_k is not None:
        return fs.top_k_sampling_from_probs(probs, top_k)
    if top_p is not None:
        return fs.top_p_sampling_from_probs(probs, top_p)
    return fs.sampling_from_probs(probs)


def _min_p_sampling_flashinfer(probs, min_p, *, indices=None,
                               deterministic=True, generator=None, seed=None,
                               offset=None, **_):
    fs = _imp("flashinfer.sampling")
    return fs.min_p_sampling_from_probs(
        probs, min_p, indices=indices, deterministic=deterministic,
        generator=generator, seed=seed, offset=offset)


def _chain_speculative_sampling_flashinfer(
        draft_probs, draft_token_ids, target_probs, *,
        maybe_output_accepted_token_num=None,
        maybe_output_emitted_draft_token_num=None,
        deterministic=True, generator=None, seed=None, offset=None, **_):
    fs = _imp("flashinfer.sampling")
    return fs.chain_speculative_sampling(
        draft_probs, draft_token_ids, target_probs,
        maybe_output_accepted_token_num=maybe_output_accepted_token_num,
        maybe_output_emitted_draft_token_num=(
            maybe_output_emitted_draft_token_num),
        deterministic=deterministic, generator=generator, seed=seed,
        offset=offset)


def _moe_gate_sgl(gating_output, *, top_k, renormalize=False,
                  moe_softcapping=0.0, correction_bias=None, **_):
    # sgl_kernel.topk_softmax(topk_weights, topk_ids, gating_output,
    #     renormalize, moe_softcapping, correction_bias) writes the output
    # buffers in place. gating_output: (num_tokens, num_experts).
    import torch
    sk = _imp("sgl_kernel")
    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty(num_tokens, top_k, device=gating_output.device,
                               dtype=torch.float32)
    topk_ids = torch.empty(num_tokens, top_k, device=gating_output.device,
                           dtype=torch.int32)
    sk.topk_softmax(topk_weights, topk_ids, gating_output, renormalize,
                    moe_softcapping, correction_bias)
    return topk_weights, topk_ids


def _moe_group_gate_sgl(gating_output, bias, *, num_expert_group, topk_group,
                        top_k, num_fused_shared_experts=0,
                        routed_scaling_factor=0.0,
                        apply_routed_scaling_factor_on_output=False, **_):
    # sgl_kernel.moe_fused_gate(input_tensor, bias, num_expert_group,
    #     topk_group, topk, num_fused_shared_experts, routed_scaling_factor,
    #     apply_routed_scaling_factor_on_output) -> (topk_weights, topk_ids).
    # SGLang's hierarchical sigmoid grouped-topk gate (DeepSeek-V3 style).
    sk = _imp("sgl_kernel")
    return sk.moe_fused_gate(gating_output, bias, num_expert_group, topk_group,
                             top_k, num_fused_shared_experts,
                             routed_scaling_factor,
                             apply_routed_scaling_factor_on_output)


def _moe_gate_vllm(gating_output, *, top_k, renormalize=False, **_):
    # vLLM fused softmax + top-k routing gate (rank-2 alignment target):
    # ops.topk_softmax(topk_weights, topk_ids, token_expert_indices,
    #   gating_output) writes the buffers in place.
    import torch
    from vllm import _custom_ops as ops
    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty(num_tokens, top_k, device=gating_output.device,
                               dtype=torch.float32)
    topk_ids = torch.empty(num_tokens, top_k, device=gating_output.device,
                           dtype=torch.int32)
    token_expert_indices = torch.empty_like(topk_ids)
    ops.topk_softmax(topk_weights, topk_ids, token_expert_indices,
                     gating_output.float())
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def _moe_group_gate_vllm(gating_output, bias, *, num_expert_group, topk_group,
                         top_k, renormalize=True, routed_scaling_factor=1.0,
                         **_):
    # vLLM grouped (DeepSeek-V3 biased) top-k gate (rank-2):
    # grouped_topk(hidden_states, gating_output, topk, renormalize,
    #   num_expert_group, topk_group, scoring_func, e_score_correction_bias).
    from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
    topk_weights, topk_ids = grouped_topk(
        gating_output, gating_output, top_k, renormalize,
        num_expert_group, topk_group, scoring_func="sigmoid",
        e_score_correction_bias=bias)
    return topk_weights, topk_ids


def _moe_gate_ks(gating_output, *, top_k, renormalize=False, **_):
    from .. import moe
    import torch
    num_tokens, num_experts = gating_output.shape
    out_w = torch.empty(num_tokens, top_k, device=gating_output.device,
                        dtype=torch.float32)
    out_i = torch.empty(num_tokens, top_k, device=gating_output.device,
                        dtype=torch.int32)
    moe.gate_softmax_topk(out_w, out_i, gating_output, num_tokens, num_experts,
                          top_k, renormalize=renormalize)
    return out_w, out_i


def _moe_group_gate_ks(gating_output, bias, *, num_expert_group, topk_group,
                       top_k, routed_scaling_factor=1.0, renormalize=True,
                       **_):
    from .. import moe
    import torch
    num_tokens, num_experts = gating_output.shape
    out_w = torch.empty(num_tokens, top_k, device=gating_output.device,
                        dtype=torch.float32)
    out_i = torch.empty(num_tokens, top_k, device=gating_output.device,
                        dtype=torch.int32)
    moe.gate_sigmoid_group_topk(out_w, out_i, gating_output, num_tokens,
                                num_experts, num_expert_group, topk_group,
                                top_k, correction_bias=bias,
                                renormalize=renormalize,
                                routed_scaling_factor=routed_scaling_factor)
    return out_w, out_i


def _sampling_ks(probs, *, top_k=None, top_p=None, **_):
    # kernel-set fused temp/top-k/top-p sampler returns sampled token ids; here
    # we expose it as the portable fallback for the renorm/sample op group.
    from .. import sampling
    import torch
    num_seqs, vocab = probs.shape
    out_tokens = torch.empty(num_seqs, device=probs.device, dtype=torch.int32)
    top_ks = (torch.full((num_seqs,), int(top_k), device=probs.device,
                         dtype=torch.int32) if top_k is not None else None)
    top_ps = (torch.full((num_seqs,), float(top_p), device=probs.device,
                         dtype=torch.float32) if top_p is not None else None)
    sampling.sample(out_tokens, probs, num_seqs, vocab,
                    top_ks=top_ks, top_ps=top_ps)
    return out_tokens


def _attention_state_merge_flashinfer(v, s, v_other=None, s_other=None, **_):
    cascade = _imp("flashinfer.cascade")
    if v_other is None:
        return cascade.merge_states(v, s)
    return cascade.merge_state(v, s, v_other, s_other)


def _attention_state_merge_ks(v, s, v_other=None, s_other=None, **_):
    # Portable kernel-set 2-way log-sum-exp merge (GPU-verified rel ~1.7e-3).
    # N-way merge_states (v_other is None) needs FlashInfer.
    import torch
    from .. import attention as _attn
    if v_other is None:
        raise NotImplementedError(
            "ks attention_state_merge fallback supports 2-way merge only; "
            "N-way merge_states needs FlashInfer")
    rows = 1
    for d in v.shape[:-1]:
        rows *= int(d)
    v_dim = int(v.shape[-1])
    out = torch.empty_like(v)
    lse = torch.empty_like(s, dtype=torch.float32)
    _attn.attention_state_merge(out, lse, v, s.float(), v_other, s_other.float(),
                                n_rows=rows, v_dim=v_dim)
    return out, lse


def _mxfp8_quantize_vllm(x, problem_sizes, expert_offsets,
                         blockscale_offsets, *, quant_output=None,
                         scale_factor=None, **_):
    import torch
    from vllm import _custom_ops as ops
    if quant_output is None:
        dtype = getattr(torch, "float8_e4m3fn", x.dtype)
        quant_output = torch.empty_like(x, dtype=dtype)
    if scale_factor is None:
        nblocks = max(1, (x.numel() + 31) // 32)
        scale_factor = torch.empty(nblocks, device=x.device, dtype=torch.uint8)
    ops.mxfp8_experts_quant(
        x, problem_sizes, expert_offsets, blockscale_offsets,
        quant_output, scale_factor)
    return quant_output, scale_factor


def _apply_token_bitmask_xgrammar(logits, bitmask, *, indices=None,
                                  vocab_size=None, backend="cuda", **_):
    import xgrammar as xgr
    if vocab_size is None:
        vocab_size = logits.shape[-1]
    xgr.apply_token_bitmask_inplace(
        logits, bitmask, indices=indices, vocab_size=vocab_size,
        backend=backend)
    return logits


def _copy_result_to_out(out, result):
    if isinstance(result, (tuple, list)):
        result = result[0]
    if result is out:
        return out
    if hasattr(out, "copy_"):
        out.copy_(result)
        return out
    raise TypeError("external provider returned a tensor but out is not copyable")


def _first_result(result):
    return result[0] if isinstance(result, (tuple, list)) else result


def _fla_scale(scale):
    if scale is None:
        return None
    try:
        return None if float(scale) == 0.0 else scale
    except (TypeError, ValueError):
        return scale


def _selective_scan_mamba(out, x, dt, A, B, C, D=None, z=None, dt_bias=None,
                          *, delta_softplus=False, state=None, chunk_size=None,
                          decode=False, use_ssd=False, **kw):
    if decode or state is not None:
        from mamba_ssm.ops.triton.selective_state_update import (
            selective_state_update)
        result = selective_state_update(
            state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias,
            dt_softplus=delta_softplus, **kw)
        return _copy_result_to_out(out, result)
    if use_ssd or chunk_size is not None:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        result = mamba_chunk_scan_combined(
            x, dt, A, B, C, chunk_size=chunk_size, D=D, z=z,
            dt_bias=dt_bias, dt_softplus=delta_softplus, **kw)
        return _copy_result_to_out(out, result)
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    result = selective_scan_fn(
        x, dt, A, B, C, D=D, z=z, delta_bias=dt_bias,
        delta_softplus=delta_softplus)
    return _copy_result_to_out(out, result)


def _selective_scan_ks(out, x, dt, A, B, C, D=None, z=None, dt_bias=None,
                       *, delta_softplus=False, batch=None, dim=None,
                       seqlen=None, dstate=None, dtype=None, stream=None, **_):
    from .. import ssm
    return ssm.selective_scan(
        out, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias,
        delta_softplus=delta_softplus, batch=batch, dim=dim, seqlen=seqlen,
        dstate=dstate, dtype=dtype, stream=stream)


def _causal_conv1d_external(out, x, weight, bias=None, *, silu=False,
                            conv_state=None, decode=False, **_):
    if decode or conv_state is not None:
        from causal_conv1d import causal_conv1d_update
        result = causal_conv1d_update(
            x, conv_state, weight, bias, activation="silu" if silu else None)
        return _copy_result_to_out(out, result)
    from causal_conv1d import causal_conv1d_fn
    result = causal_conv1d_fn(
        x, weight, bias, activation="silu" if silu else None)
    return _copy_result_to_out(out, result)


def _causal_conv1d_ks(out, x, weight, bias=None, *, batch=None, dim=None,
                      seqlen=None, width=None, silu=False, dtype=None,
                      stream=None, **_):
    from .. import ssm
    return ssm.causal_conv1d(
        out, x, weight, bias=bias, batch=batch, dim=dim, seqlen=seqlen,
        width=width, silu=silu, dtype=dtype, stream=stream)


def _mamba2_ssd_chunk_scan_mamba(*args, **kw):
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
    )
    return _first_result(mamba_chunk_scan_combined_varlen(*args, **kw))


def _gated_delta_rule_fla(q, k, v, g, beta, *, g_is_vector=0,
                          use_qk_l2norm=0, scale=0.0, initial_state=None,
                          output_final_state=False, state_v_first=False,
                          cu_seqlens=None, cu_seqlens_cpu=None,
                          use_beta_sigmoid_in_kernel=False,
                          allow_neg_eigval=False, recurrent=False, **kw):
    if recurrent:
        from fla.ops import fused_recurrent_gated_delta_rule
        result = fused_recurrent_gated_delta_rule(
            q, k, v, g, beta, scale=_fla_scale(scale),
            initial_state=initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens, **kw)
        return _first_result(result)
    from fla.ops import chunk_gated_delta_rule, chunk_kda
    is_vector_gate = bool(g_is_vector) or getattr(g, "ndim", 0) == 4
    fn = chunk_kda if is_vector_gate else chunk_gated_delta_rule
    result = fn(
        q, k, v, g, beta, scale=_fla_scale(scale),
        initial_state=initial_state, output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm),
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval, state_v_first=state_v_first,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, **kw)
    return _first_result(result)


def _gated_delta_rule_ks(q, k, v, g, beta, *, batch=None, seqlen=None,
                         heads=None, k_dim=None, v_dim=None, g_is_vector=0,
                         use_qk_l2norm=0, scale=0.0, dtype=None, stream=None,
                         **_):
    from .. import linear_attn
    import torch
    b, t, h, kd = q.shape
    vd = v.shape[-1]
    out = torch.empty(b, t, h, vd, device=q.device, dtype=v.dtype)
    linear_attn.gated_delta_rule(
        out, q, k, v, g, beta,
        batch=b if batch is None else batch,
        seqlen=t if seqlen is None else seqlen,
        heads=h if heads is None else heads,
        k_dim=kd if k_dim is None else k_dim,
        v_dim=vd if v_dim is None else v_dim,
        g_is_vector=g_is_vector,
        use_qk_l2norm=use_qk_l2norm,
        scale=scale,
        dtype=dtype,
        stream=stream)
    return out


def _gated_linear_attn_fla(q, k, v, g=None, head_decay=None, *, gate_mode=0,
                           scale=0.0, initial_state=None,
                           output_final_state=False, state_v_first=False,
                           cu_seqlens=None, cu_seqlens_cpu=None,
                           layer_idx=None, num_layers=None, recurrent=False,
                           **_):
    if recurrent:
        from fla.ops import fused_recurrent_gla, fused_recurrent_simple_gla
        fn = (fused_recurrent_simple_gla if gate_mode in (1, "simple") or
              g is None or head_decay is not None else fused_recurrent_gla)
        result = fn(
            q, k, v, g if g is not None else head_decay, scale=_fla_scale(scale),
            initial_state=initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens)
        return _first_result(result)
    from fla.ops import chunk_gla, chunk_lightning_attn, chunk_simple_gla
    if gate_mode in (2, "lightning") or layer_idx is not None or num_layers is not None:
        if layer_idx is None or num_layers is None:
            raise ValueError("lightning attention requires layer_idx and num_layers")
        result = chunk_lightning_attn(
            q, k, v, layer_idx=layer_idx, num_layers=num_layers,
            scale=_fla_scale(scale), initial_state=initial_state,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens)
    elif (gate_mode in (1, "simple") or g is None or head_decay is not None
          or getattr(g, "ndim", 0) == 3):
        result = chunk_simple_gla(
            q, k, v, g=g, g_gamma=None if g is not None else head_decay,
            scale=_fla_scale(scale), initial_state=initial_state,
            output_final_state=output_final_state, state_v_first=state_v_first,
            cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu)
    else:
        result = chunk_gla(
            q, k, v, g, scale=_fla_scale(scale),
            initial_state=initial_state, output_final_state=output_final_state,
            state_v_first=state_v_first, cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu)
    return _first_result(result)


def _gated_linear_attn_ks(q, k, v, g=None, head_decay=None, *, batch=None,
                          seqlen=None, heads=None, k_dim=None, v_dim=None,
                          gate_mode=0, scale=0.0, dtype=None, stream=None,
                          **_):
    from .. import linear_attn
    import torch
    b, t, h, kd = q.shape
    vd = v.shape[-1]
    out = torch.empty(b, t, h, vd, device=q.device, dtype=v.dtype)
    linear_attn.gated_linear_attn(
        out, q, k, v, g, head_decay,
        batch=b if batch is None else batch,
        seqlen=t if seqlen is None else seqlen,
        heads=h if heads is None else heads,
        k_dim=kd if k_dim is None else k_dim,
        v_dim=vd if v_dim is None else v_dim,
        gate_mode=gate_mode,
        scale=scale,
        dtype=dtype,
        stream=stream)
    return out


def _rwkv_wkv7_fla(r, w, k, v, a, b, *, scale=0.0, initial_state=None,
                   output_final_state=False, cu_seqlens=None,
                   cu_seqlens_cpu=None, safe_gate=False, chunk_size=None,
                   recurrent=False, **_):
    if recurrent:
        from fla.ops import fused_recurrent_rwkv7
        result = fused_recurrent_rwkv7(
            r, w, k, v, a, b, scale=1.0 if _fla_scale(scale) is None else scale,
            initial_state=initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens)
        return _first_result(result)
    from fla.ops import chunk_rwkv7
    result = chunk_rwkv7(
        r, w, k, v, a, b, scale=1.0 if _fla_scale(scale) is None else scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
        safe_gate=safe_gate, chunk_size=chunk_size)
    return _first_result(result)


def _rwkv_wkv7_ks(r, w, k, v, a, b, *, batch=None, seqlen=None, heads=None,
                  k_dim=None, v_dim=None, scale=0.0, dtype=None, stream=None,
                  **_):
    from .. import linear_attn
    import torch
    bt, t, h, kd = r.shape
    vd = v.shape[-1]
    out = torch.empty(bt, t, h, vd, device=r.device, dtype=v.dtype)
    linear_attn.rwkv_wkv7(
        out, r, w, k, v, a, b,
        batch=bt if batch is None else batch,
        seqlen=t if seqlen is None else seqlen,
        heads=h if heads is None else heads,
        k_dim=kd if k_dim is None else k_dim,
        v_dim=vd if v_dim is None else v_dim,
        scale=scale,
        dtype=dtype,
        stream=stream)
    return out


def _moe_deepgemm(a, b, expert_offsets=None, *, num_experts=None, n, k,
                  a_scale=None, b_scale=None, masked_m=None, expected_m=None,
                  **_):
    # DeepGEMM grouped FP8 GEMM. If masked_m is supplied, use the CUDA-graph
    # decode path; otherwise use the contiguous prefill path.
    import torch
    dg = _imp("deep_gemm")
    total_rows = a.shape[0]
    out = torch.empty(total_rows, n, device=a.device, dtype=torch.bfloat16)
    if masked_m is not None:
        dg.m_grouped_fp8_gemm_nt_masked(
            (a, a_scale), (b, b_scale), out, masked_m, expected_m)
        return out
    if expert_offsets is None or num_experts is None:
        raise ValueError("DeepGEMM contiguous MoE needs expert_offsets/num_experts")
    # Build m_indices: expert id per row from the contiguous expert offsets.
    m_indices = torch.empty(total_rows, device=a.device, dtype=torch.int32)
    for e in range(num_experts):
        lo = int(expert_offsets[e])
        hi = int(expert_offsets[e + 1])
        m_indices[lo:hi] = e
    dg.m_grouped_fp8_gemm_nt_contiguous((a, a_scale), (b, b_scale), out,
                                        m_indices)
    return out


def _moe_sgl(a, b, expert_offsets, *, num_experts, n, k,
             a_scale=None, b_scale=None, **_):
    # SGLang CUTLASS blockwise-fp8 grouped GEMM (the stated alignment target):
    # sgl_kernel.fp8_blockwise_scaled_grouped_mm(out, a, b, a_scale, b_scale,
    #   stride/offset tensors, expert_offsets, ...). This is the *real* sgl
    # grouped GEMM, not a delegation to the ks kernel.
    import torch
    sk = _imp("sgl_kernel")
    total_rows = a.shape[0]
    out = torch.empty(total_rows, n, device=a.device, dtype=torch.bfloat16)
    sk.fp8_blockwise_scaled_grouped_mm(
        out, a, b, a_scale, b_scale, expert_offsets, num_experts)
    return out


def _reshape_and_cache_fp8_flashinfer(key, value, key_cache, value_cache,
                                      slot_mapping, k_scale=None, v_scale=None,
                                      *, kv_layout="NHD", **kw):
    from flashinfer.page import append_paged_kv_cache
    append_paged_kv_cache(
        key, value, key_cache, value_cache, slot_mapping,
        kv_layout=kv_layout, k_scale=k_scale, v_scale=v_scale, **kw)
    return key_cache, value_cache


def _reshape_and_cache_fp8_sgl(key, value, key_cache, value_cache, slot_mapping,
                               k_scale=None, v_scale=None, *,
                               kv_cache_dtype="fp8", **_):
    sk = _imp("sgl_kernel")
    sk.reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping,
                               kv_cache_dtype, k_scale, v_scale)
    return key_cache, value_cache


def _attn_prefill_sgl(q, k, v, *, causal=True, softmax_scale=None,
                      window_size=None, softcap=0.0, sinks=None,
                      custom_mask=None, packed_custom_mask=None,
                      alibi_slopes=None, **_):
    # sgl_kernel.flash_attn_varlen_func (FA3) needs cu_seqlens + max_seqlens.
    # Build them for the dense (batch, seqlen, heads, head_dim) layout.
    if custom_mask is not None or packed_custom_mask is not None:
        raise ProviderCallUnsupported(
            "sgl-kernel dense prefill adapter does not support custom/tree masks")
    import torch
    sk = _imp("sgl_kernel")
    b, s, qh, hd = q.shape
    kvh = k.shape[-2]
    qf = q.reshape(b * s, qh, hd)
    kf = k.reshape(b * s, kvh, hd)
    vf = v.reshape(b * s, kvh, hd)
    cu = torch.arange(0, (b + 1) * s, s, device=q.device, dtype=torch.int32)
    out = _call_with_optional_kwargs(
        sk.flash_attn_varlen_func, (qf, kf, vf, cu, cu),
        {"max_seqlen_q": s, "max_seqlen_k": s,
         "softmax_scale": _scale(hd, softmax_scale), "causal": causal},
        {"window_size": window_size, "softcap": softcap, "sinks": sinks,
         "alibi_slopes": alibi_slopes},
        "sgl-kernel prefill")
    return out.reshape(b, s, qh, hd)


def _attn_decode_sgl(q, k_cache, v_cache, block_tables, seq_lens, *,
                     block_size, max_blocks_per_seq, softmax_scale=None,
                     window_size=None, softcap=0.0, sinks=None,
                     custom_mask=None, packed_custom_mask=None,
                     alibi_slopes=None, **_):
    # sgl_kernel.flash_attn_with_kvcache: paged decode with FA3. ks caches are
    # (num_blocks, kvh, page, hd); FA3 wants (num_blocks, page, kvh, hd).
    if custom_mask is not None or packed_custom_mask is not None:
        raise ProviderCallUnsupported(
            "sgl-kernel decode adapter does not support custom/tree masks")
    import torch
    sk = _imp("sgl_kernel")
    num_seqs, qh, hd = q.shape
    kvh = k_cache.shape[1]
    qf = q.reshape(num_seqs, 1, qh, hd)
    k_fa = k_cache.permute(0, 2, 1, 3).contiguous()
    v_fa = v_cache.permute(0, 2, 1, 3).contiguous()
    out = _call_with_optional_kwargs(
        sk.flash_attn_with_kvcache, (qf, k_fa, v_fa),
        {"page_table": block_tables, "cache_seqlens": seq_lens,
         "softmax_scale": _scale(hd, softmax_scale), "causal": False},
        {"window_size": window_size, "softcap": softcap, "sinks": sinks,
         "alibi_slopes": alibi_slopes},
        "sgl-kernel decode")
    return out.reshape(num_seqs, qh, hd)


def _mla_decode_ks(q_nope, q_pe, kv_cache, block_tables, seq_lens, *,
                   heads, lora, rope_dim, block_size, max_blocks_per_seq,
                   softmax_scale=None, **_):
    # kernel-set absorbed-MLA decode (ks.attention.mla_decode).
    from .. import attention
    import torch
    num_seqs = q_nope.shape[0]
    out = torch.empty(num_seqs, heads, lora, device=q_nope.device,
                      dtype=q_nope.dtype)
    attention.mla_decode(
        out, q_nope, q_pe, kv_cache, block_tables, seq_lens,
        num_seqs, heads, lora, rope_dim, block_size, max_blocks_per_seq,
        softmax_scale=_scale(lora + rope_dim, softmax_scale))
    return out


def _mla_decode_flashinfer(q_nope, q_pe, kv_cache, block_tables, seq_lens, *,
                           heads, lora, rope_dim, block_size,
                           max_blocks_per_seq, softmax_scale=None, **_):
    # FlashInfer MLA paged decode (BatchMLAPagedAttentionWrapper). Portable
    # across sm80+ (A100/L4/H100/B200) — the only SOTA MLA path pre-Hopper, where
    # FlashMLA (sm90) is gated out. Absorbed-MLA: ckv (lora) + kpe (rope_dim).
    import torch
    fi = _imp("flashinfer")
    num_seqs = q_nope.shape[0]
    device = q_nope.device
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = fi.mla.BatchMLAPagedAttentionWrapper(workspace, backend="auto")
    bps = max_blocks_per_seq
    q_indptr = torch.arange(num_seqs + 1, device=device, dtype=torch.int32)
    kv_indptr = torch.arange(0, (num_seqs + 1) * bps, bps,
                             device=device, dtype=torch.int32)
    kv_indices = torch.arange(num_seqs * bps, device=device, dtype=torch.int32)
    wrapper.plan(q_indptr, kv_indptr, kv_indices, seq_lens.to(torch.int32),
                 heads, lora, rope_dim, block_size,
                 False, _scale(lora + rope_dim, softmax_scale),
                 q_nope.dtype, kv_cache.dtype)
    ckv = kv_cache[..., :lora].contiguous()
    kpe = kv_cache[..., lora:lora + rope_dim].contiguous()
    return wrapper.run(q_nope, q_pe, ckv, kpe)


def _mla_decode_flash_mla(q_nope, q_pe, kv_cache, block_tables, seq_lens, *,
                          heads, lora, rope_dim, block_size,
                          max_blocks_per_seq, softmax_scale=None, **_):
    import torch
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
    h_kv = 1
    tile_md, num_splits = get_mla_metadata(seq_lens, heads // h_kv, h_kv)
    q_cat = torch.cat([q_nope, q_pe], dim=-1).to(torch.bfloat16).unsqueeze(1)
    kcache = kv_cache.reshape(
        kv_cache.shape[0], block_size, h_kv, lora + rope_dim).to(torch.bfloat16)
    out, _lse = flash_mla_with_kvcache(
        q_cat, kcache, block_tables, seq_lens, lora, tile_md, num_splits,
        softmax_scale=_scale(lora + rope_dim, softmax_scale), causal=True)
    return out


def _mla_decode_sgl(q_nope, q_pe, kv_cache, block_tables, seq_lens, *,
                    heads, lora, rope_dim, block_size, max_blocks_per_seq,
                    softmax_scale=None, **_):
    # sgl_kernel.flash_mla_with_kvcache (FlashMLA, Hopper sm90). Build the tile
    # scheduler metadata via get_mla_metadata, then run the absorbed-MLA decode.
    import torch
    sk = _imp("sgl_kernel")
    num_seqs = q_nope.shape[0]
    h_kv = 1
    tile_md, num_splits = sk.get_mla_metadata(seq_lens, heads // h_kv, h_kv)
    q_cat = torch.cat([q_nope, q_pe], dim=-1).to(torch.bfloat16).unsqueeze(1)
    total_blocks = kv_cache.shape[0]
    kcache = kv_cache.reshape(total_blocks, block_size, 1,
                              lora + rope_dim).to(torch.bfloat16)
    out, _lse = sk.flash_mla_with_kvcache(
        q_cat, kcache, block_tables, seq_lens, lora, tile_md, num_splits,
        softmax_scale=_scale(lora + rope_dim, softmax_scale), causal=True)
    return out


def _sparse_mla_attention_flash_mla(
        q_nope, q_pe, kv_cache, block_tables=None, seq_lens=None, indices=None,
        *, heads=None, lora=None, rope_dim=None, block_size=None,
        max_blocks_per_seq=None, topk=None, softmax_scale=None, is_fp8=False,
        prefill=False, causal=True, h_kv=1, **kw):
    # Sparse FlashMLA consumes a per-query top-k index tensor and only gathers
    # those KV positions. This is not a flag on dense MLA decode.
    import torch
    from flash_mla import (
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
    if indices is None:
        raise ValueError("sparse_mla_attention requires an indices tensor")
    if topk is None:
        topk = indices.shape[-1]
    if lora is None:
        lora = q_nope.shape[-1]
    if rope_dim is None:
        rope_dim = q_pe.shape[-1]
    if heads is None:
        heads = q_nope.shape[-2]
    q_cat = torch.cat([q_nope, q_pe], dim=-1).to(torch.bfloat16)
    if not prefill:
        q_cat = q_cat.unsqueeze(1) if q_cat.ndim == 3 else q_cat
    if block_size is not None and kv_cache.ndim == 3:
        kcache = kv_cache.reshape(
            kv_cache.shape[0], block_size, h_kv, lora + rope_dim)
    else:
        kcache = kv_cache
    kcache = kcache.to(torch.bfloat16)
    sm_scale = _scale(lora + rope_dim, softmax_scale)
    if prefill or block_tables is None or seq_lens is None:
        out = flash_mla_sparse_fwd(
            q_cat, kcache, indices, topk=topk, softmax_scale=sm_scale,
            causal=causal, is_fp8=is_fp8, **kw)
        return _first_result(out)
    tile_md, num_splits = get_mla_metadata(
        seq_lens, heads // h_kv, h_kv, is_fp8=is_fp8, topk=topk)
    out, _lse = flash_mla_with_kvcache(
        q_cat, kcache, block_tables, seq_lens, lora, tile_md, num_splits,
        softmax_scale=sm_scale, causal=causal, is_fp8=is_fp8,
        indices=indices, topk=topk, **kw)
    return out


def _dsa_indexer_logits_deepgemm(q, kv, *args, paged=False,
                                 block_tables=None, seq_lens=None, **kw):
    dg = _imp("deep_gemm")
    if paged or block_tables is not None or seq_lens is not None:
        return dg.fp8_paged_mqa_logits(
            q, kv, block_tables, seq_lens, *args, **kw)
    return dg.fp8_mqa_logits(q, kv, *args, **kw)


def _dsa_topk_select_flashinfer(scores, topk, *, indices_out=None,
                                largest=True, sorted=False, **kw):
    fi = _imp("flashinfer")
    fn = getattr(fi, "top_k", None)
    if fn is None:
        try:
            topk_mod = _imp("flashinfer.top_k")
            fn = getattr(topk_mod, "top_k", None) or getattr(topk_mod, "topk", None)
        except Exception as exc:
            raise ProviderCallUnsupported(
                "flashinfer top_k selector is not exposed by this install") from exc
    if fn is None:
        raise ProviderCallUnsupported(
            "flashinfer top_k selector is not exposed by this install")
    result = fn(scores, topk, largest=largest, sorted=sorted, **kw)
    indices = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 \
        else _first_result(result)
    if indices_out is not None:
        indices_out.copy_(indices)
        return indices_out
    return indices


def _dsa_topk_select_ks(scores, topk, *, indices_out=None, **_):
    # Portable kernel-set fallback (k passes of arg-max; GPU-verified).
    import torch
    from .. import attention as _attn
    rows = 1
    for d in scores.shape[:-1]:
        rows *= int(d)
    n_cols = int(scores.shape[-1])
    if indices_out is None:
        indices_out = torch.empty((rows, int(topk)), device=scores.device,
                                  dtype=torch.int32)
    _attn.dsa_topk_select(indices_out, scores, n_rows=rows, n_cols=n_cols,
                          topk=int(topk))
    return indices_out


def _nsa_selection_attention_fla(q, k, v, *args, **kw):
    ops = _imp("fla.ops")
    fn = (getattr(ops, "parallel_nsa", None) or
          getattr(ops, "native_sparse_attention", None))
    if fn is None:
        raise ProviderCallUnsupported(
            "FLA install does not expose parallel_nsa/native_sparse_attention")
    return _first_result(fn(q, k, v, *args, **kw))


# =========================================================================== #
# THE CURATED TABLE. Provider lists are in rank order (1 = best).
# A kernel-set fallback is appended to every op via _ks(...) below.
# Provider metadata (min_sm / dtypes / import_check) is lifted from
# providers/registry.json.
# =========================================================================== #
def _ks_provider(call, abi, note="portable C-ABI fallback") -> Provider:
    # min_sm=0: ks is the always-selectable portable fallback; its own runtime
    # decides device/dtype support. dtypes="" => no dtype gate here.
    return Provider(KERNEL_SET, rank=99, min_sm=0, dtypes="",
                    import_check="", call=call, note=note)


def _sgl_provider(rank, call, *, min_sm=80, dtypes="fp16, bf16",
                  note="SGLang sgl-kernel") -> Provider:
    """A sgl-kernel provider entry. import_check is `import sgl_kernel`.
    Default arch gate sm80 (Ampere+); fp8/FlashMLA paths pass min_sm=90."""
    return Provider(SGL_KERNEL, rank=rank, min_sm=min_sm, dtypes=dtypes,
                    import_check="import sgl_kernel", call=call, note=note)


# TODO(P2): defer int4+2:4 Sparse-Marlin and native microsoft/BitNet CUDA
# wiring until vendoring/model targets land.


_OPS_RAW: List[Op] = [
    Op("attention_prefill", "attention", None, [
        # FA4 (CuTe-DSL) is the Blackwell-optimal path; gated sm100 so it only
        # outranks FA2/FA3 on B200. flash-attn (FA2 sm80/89, FA3 auto on sm90)
        # remains rank-1 for Ampere/Ada/Hopper.
        Provider("flash-attn-cute", 0, 100, "fp16, bf16",
                 "import flash_attn.cute",
                 _attn_prefill_fa4,
                 "FlashAttention-4 CuTe-DSL (Blackwell sm100; SWA/softcap "
                 "when exposed by installed build)"),
        Provider("flash-attn", 1, 80, "fp16, bf16",
                 "from flash_attn import flash_attn_func",
                 _attn_prefill_flash_attn,
                 "FA2/FA3 dense attention; forwards window_size, softcap, "
                 "and sinks when the installed API supports them"),
        _sgl_provider(2, _attn_prefill_sgl, min_sm=90,
                      note="SGLang FA3 varlen prefill; forwards SWA/softcap/sinks "
                      "when present"),
        Provider("torch-sdpa", 3, 80, "fp16, bf16, fp32",
                 "import torch",
                 _attn_prefill_sdpa,
                 "PyTorch SDPA plain attention only; skipped for "
                 "window/softcap/sinks/custom masks"),
        Provider("flashinfer", 4, 75, "fp16, bf16, fp8",
                 "from flashinfer.prefill import single_prefill_with_kv_cache",
                 _attn_prefill_flashinfer,
                 "FlashInfer prefill (b==1 adapter); forwards window_left, "
                 "logits_soft_cap, and custom/packed masks when supported"),
        _ks_provider(_attn_prefill_ks, "ks_flash_attn"),
    ]),
    Op("attention_decode", "attention", None, [
        Provider("flashinfer", 1, 75, "fp16, bf16, fp8 KV",
                 "from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper",
                 _attn_decode_flashinfer,
                 "FlashInfer paged decode; forwards window_left/logits_soft_cap "
                 "and supported mask/sink kwargs"),
        _sgl_provider(2, _attn_decode_sgl, min_sm=90,
                      note="SGLang FA3 paged decode; forwards SWA/softcap/sinks "
                      "when present"),
        Provider("flash-attn-3", 3, 90, "fp16, bf16, fp8 e4m3",
                 "from flash_attn_interface import flash_attn_with_kvcache",
                 _attn_decode_fa3,
                 "native FlashAttention-3 paged decode; forwards window_size, "
                 "softcap, and sinks when supported"),
        _ks_provider(_attn_decode_ks, "ks_paged_attn_decode"),
    ]),
    Op("mla_decode", "attention", None, [
        # Hopper/Blackwell: FlashMLA (DeepSeek official, sm90) is unbeatable.
        _sgl_provider(1, _mla_decode_sgl, min_sm=90,
                      note="SGLang FlashMLA (flash_mla_with_kvcache, sm90)"),
        # Pre-Hopper (A100 sm80 / L4 sm89): FlashInfer MLA is the only SOTA path
        # — wired at rank 2 (sm80+) so Ampere/Ada no longer drop to the ~1%-BW
        # ks fallback. On sm90 it sits behind FlashMLA; on sm80/89 it wins.
        Provider("flashinfer", 2, 80, "fp16, bf16, fp8 KV",
                 "from flashinfer.mla import BatchMLAPagedAttentionWrapper",
                 _mla_decode_flashinfer,
                 "FlashInfer MLA paged decode (portable sm80+)"),
        Provider("flash-mla", 3, 90, "bf16, fp8 e4m3 KV",
                 "from flash_mla import flash_mla_with_kvcache, get_mla_metadata",
                 _mla_decode_flash_mla,
                 "native DeepSeek FlashMLA package (Hopper+)"),
        _ks_provider(_mla_decode_ks, "ks_mla_decode",
                     "kernel-set absorbed-MLA decode"),
    ]),
    Op("sparse_mla_attention", "attention", None, [
        Provider("flash-mla", 1, 90, "bf16, fp8 e4m3 KV",
                 "from flash_mla import flash_mla_sparse_fwd, "
                 "flash_mla_with_kvcache, get_mla_metadata",
                 _sparse_mla_attention_flash_mla,
                 "FlashMLA sparse/top-k indexed MLA prefill+decode (Hopper+)"),
        _ks_provider(_ks_unsupported("sparse_mla_attention"), "",
                     "no portable sparse MLA; needs FlashMLA"),
    ]),
    Op("dsa_indexer_logits", "attention", None, [
        Provider("deep_gemm", 1, 90, "fp8 e4m3, fp32 logits",
                 "import deep_gemm; deep_gemm.fp8_mqa_logits; "
                 "deep_gemm.fp8_paged_mqa_logits",
                 _dsa_indexer_logits_deepgemm,
                 "DeepGEMM FP8 lightning-indexer MQA logits (Hopper+)"),
        _ks_provider(_ks_unsupported("dsa_indexer_logits"), "",
                     "no portable DSA indexer; needs DeepGEMM"),
    ]),
    Op("dsa_topk_select", "attention", "ks_dsa_topk_select", [
        Provider("flashinfer", 1, 80,
                 "fp32, fp16, bf16 scores; int32 indices",
                 "import flashinfer; flashinfer.top_k",
                 _dsa_topk_select_flashinfer,
                 "FlashInfer radix/top-k selector for sparse-attention indices"),
        _ks_provider(_dsa_topk_select_ks, "ks_dsa_topk_select",
                     "portable k-pass arg-max top-k (correctness-first)"),
    ]),
    Op("nsa_selection_attention", "attention", None, [
        Provider("flash-linear-attention", 1, 80, "fp16, bf16",
                 "import fla.ops",
                 _nsa_selection_attention_fla,
                 "FLA Native Sparse Attention selection branch"),
        _ks_provider(_ks_unsupported("nsa_selection_attention"), "",
                     "no portable NSA selection attention; needs FLA"),
    ]),
    Op("gemm", "gemm-dense", "ks_gemm", [
        # Dense fp16/bf16/tf32 GEMM is the most NVIDIA-tuned primitive there is;
        # cuBLAS/cuBLASLt (via torch @) is the optimal provider on EVERY arch
        # (A100/L4/H100/B200 — it auto-selects the WGMMA/tcgen05 path per arch).
        # CUTLASS only wins when you need a fused epilogue (not exposed here).
        # ks is the rank-99 portable fallback ONLY (measured 0.03-0.10x cuBLAS).
        Provider("torch", 1, 70, "fp16, bf16, fp32, tf32",
                 "import torch",
                 _gemm_torch, "cuBLAS/cuBLASLt tensor-core GEMM (arch-optimal)"),
        _ks_provider(_gemm_ks, "ks_gemm",
                     "portable C-ABI fallback (slow; correctness only)"),
    ]),
    Op("fp8_gemm", "gemm-quant", "ks_gemm_w8a8", [
        Provider("deep_gemm", 1, 90, "fp8 e4m3, fp32 block scales",
                 "import deep_gemm",
                 _fp8_gemm_deepgemm, "DeepGEMM blockwise fp8 (Hopper/Blackwell)"),
        Provider("vllm-cutlass", 2, 89, "fp8 e4m3/e5m2",
                 "from vllm import _custom_ops as ops; ops.cutlass_scaled_mm; "
                 "ops.scaled_fp8_quant",
                 _fp8_gemm_vllm_cutlass,
                 "vLLM CUTLASS scaled fp8 mm (dynamic/per-token scales)"),
        Provider("fbgemm", 2, 90, "fp8 e4m3, bf16 out",
                 "import torch; import fbgemm_gpu.experimental.gen_ai; "
                 "torch.ops.fbgemm.f8f8bf16_rowwise",
                 _fp8_gemm_fbgemm,
                 "FBGEMM GenAI rowwise fp8 GEMM"),
        Provider("torch-scaled-mm", 2, 89, "fp8 e4m3/e5m2",
                 "import torch",
                 _fp8_gemm_torch, "torch._scaled_mm fp8 tensor cores (Ada+)"),
        _sgl_provider(3, _fp8_gemm_sgl, min_sm=90, dtypes="fp8 e4m3",
                      note="SGLang CUTLASS fp8_scaled_mm (sm90)"),
        _ks_provider(_fp8_gemm_ks, "ks_gemm_w8a8",
                     "no native fp8; dense-cast fallback"),
    ]),
    Op("int8_gemm", "gemm-quant", "ks_gemm_w8a8", [
        # Blackwell (sm100): CUTLASS INT8 is unsupported, so Marlin-int8 (vLLM)
        # is the rank-0 sm100 path; gated sm100 so it only wins on B200.
        Provider("vllm-marlin-int8", 0, 100, "int8 w8a8",
                 "from vllm import _custom_ops as ops; ops.marlin_gemm; "
                 "ops.gptq_marlin_repack",
                 _int8_gemm_marlin, "Marlin INT8 (vLLM, Blackwell sm100)"),
        # Registry true rank-1 across sm75-sm90: vLLM CUTLASS INT8 W8A8
        # (SmoothQuant / compressed-tensors, symmetric + azp).
        Provider("vllm", 1, 80, "int8 w8a8",
                 "from vllm import _custom_ops",
                 _int8_gemm_vllm, "vLLM CUTLASS int8 W8A8 (SmoothQuant)"),
        # sgl-kernel int8_scaled_mm is the rank-2 alignment target.
        _sgl_provider(2, _int8_gemm_sgl, min_sm=80, dtypes="int8 w8a8",
                      note="SGLang CUTLASS int8_scaled_mm"),
        Provider("gemlite", 3, 80, "int8 w8a8",
                 "from gemlite.core import GemLiteLinear",
                 _int8_gemm_gemlite,
                 "GemLite Triton INT8 split-K fallback"),
        _ks_provider(_int8_gemm_ks, "ks_gemm_w8a8",
                     "int8 W8A8 scaled-mm (native ABI; portable fallback)"),
    ]),
    Op("w4a16", "gemm-quant", "ks_gemm_w4a16", [
        # Hopper (sm90a): Machete (CUTLASS TMA+WGMMA weight-prepack) beats
        # Marlin — rank-0, gated sm90 so it only outranks Marlin on Hopper+.
        Provider("vllm-machete", 0, 90, "int4 weights, fp16/bf16 acts",
                 "from vllm import _custom_ops as ops; ops.machete_mm; "
                 "ops.machete_prepack_B",
                 _w4a16_machete, "Machete W4A16/W4A8 (vLLM, Hopper sm90a)"),
        # Ampere/Ada/Blackwell: GPTQ/AWQ-Marlin is the de-facto W4A16 kernel
        # (~4x over fp16). Now WIRED (was call=None) so it actually dispatches.
        Provider("vllm-marlin", 1, 80, "int4 weights, fp16/bf16 acts",
                 "from vllm import _custom_ops as ops; ops.marlin_gemm; "
                 "ops.gptq_marlin_repack; ops.awq_marlin_repack",
                 _w4a16_marlin, "GPTQ/AWQ-Marlin (Ampere+; NVFP4-Marlin sm100)"),
        Provider("gemlite", 2, 80, "int4 weights, fp16/bf16 acts",
                 "from gemlite.core import GemLiteLinear",
                 _w4a16_gemlite,
                 "GemLite Triton low-bit fallback (portable, batch-friendly)"),
        Provider("torchao-int4", 3, 80, "int4 weight, bf16 act",
                 "from torchao.quantization import quantize_, Int4WeightOnlyConfig",
                 _w4a16_torchao_int4,
                 "torchao tinygemm int4 weight-only path"),
        _ks_provider(_w4a16_ks, "ks_gemm_w4a16",
                     "portable INT4 fallback (correctness only)"),
    ]),
    Op("w4a8", "gemm-quant", None, [
        Provider("vllm-machete", 0, 90, "int4 weights, int8/fp8 acts",
                 "from vllm import _custom_ops as ops; ops.machete_mm; "
                 "ops.machete_prepack_B",
                 _w4a8_machete, "Machete W4A8 (Hopper+)"),
        Provider("vllm-marlin", 1, 80, "int4 weights, int8/fp8 acts",
                 "from vllm import _custom_ops as ops; ops.marlin_gemm; "
                 "ops.gptq_marlin_repack",
                 _w4a8_marlin, "Marlin QQQ/W4A8 (Ampere/Ada)"),
        _ks_provider(_ks_unsupported("w4a8 GEMM"), "",
                     "no portable W4A8 kernel; needs vLLM Machete/Marlin"),
    ]),
    Op("w8a16_fp8", "gemm-quant", None, [
        Provider("vllm-fp8-marlin", 1, 80,
                 "fp8 weights, fp16, bf16 acts",
                 "from vllm import _custom_ops as ops; ops.marlin_gemm; "
                 "ops.gptq_marlin_repack; ops.awq_marlin_repack",
                 _w8a16_fp8_marlin,
                 "FP8 weight-only Marlin for Ampere/Ada no-fp8-TC serving"),
        _ks_provider(_ks_unsupported("w8a16_fp8 GEMM"), "",
                     "no portable FP8 weight-only GEMM; needs vLLM Marlin"),
    ]),
    Op("sparse_2_4_gemm", "gemm-quant", None, [
        Provider("vllm-cutlass-sparse", 0, 90,
                 "fp8 e4m3/int8 weights, fp16/bf16 out",
                 "from vllm import _custom_ops as ops; "
                 "ops.cutlass_scaled_sparse_mm; "
                 "ops.cutlass_sparse_compress",
                 _sparse_2_4_gemm_vllm,
                 "vLLM CUTLASS 2:4 sparse scaled GEMM (Hopper+)"),
        _ks_provider(_ks_unsupported("sparse_2_4_gemm"), "",
                     "no portable 2:4 sparse GEMM; needs vLLM CUTLASS"),
    ]),
    Op("bitnet_gemm", "gemm-quant", None, [
        Provider("bitblas", 0, 80,
                 "ternary/int2 weights, int8 or fp16 activations",
                 "from bitblas import Matmul",
                 _bitnet_gemm_bitblas,
                 "BitBLAS A8W1.58 ternary BitLinear GEMM (Ampere+)"),
        _ks_provider(_ks_unsupported("bitnet_gemm"), "",
                     "no portable ternary BitLinear GEMM; needs BitBLAS"),
    ]),
    # FP8 BLOCKWISE GEMM (DeepSeek-V3 128x128 weight / 1x128 act recipe). DeepGEMM
    # is the Hopper/Blackwell reference; kernel-set's ks_gemm_fp8_blockwise is the
    # portable sm80+ terminal (real blockwise kernel, not a dense cast) so the
    # recipe is also covered on A100/A800 where there is no fp8 hardware provider.
    Op("fp8_gemm_blockwise", "gemm-quant", "ks_gemm_fp8_blockwise", [
        Provider("deep_gemm", 1, 90, "fp8 e4m3, fp32 block scales",
                 "import deep_gemm",
                 _fp8_gemm_deepgemm,
                 "DeepGEMM blockwise fp8 NT (Hopper/Blackwell)"),
        _sgl_provider(2, _fp8_gemm_sgl, min_sm=90, dtypes="fp8 e4m3",
                      note="SGLang CUTLASS blockwise fp8 grouped/scaled mm"),
        _ks_provider(_fp8_gemm_blockwise_ks, "ks_gemm_fp8_blockwise",
                     "portable blockwise fp8 (sm80+, software dequant)"),
    ]),
    # Per-token-GROUP (1x128) dynamic fp8 activation quant — the format the
    # blockwise fp8 GEMM consumes. ks_quantize_fp8_group is the native terminal.
    Op("per_token_group_quant", "gemm-quant", "ks_quantize_fp8_group", [
        Provider("vllm", 1, 89, "fp8 e4m3",
                 "from vllm import _custom_ops",
                 _per_token_group_quant_fp8_vllm,
                 "vLLM per-token-group fp8 quant (scaled_fp8_quant group_shape)"),
        _sgl_provider(2, _per_token_group_quant_fp8_sgl, min_sm=89,
                      dtypes="fp8 e4m3",
                      note="SGLang sgl_per_token_group_quant_8bit"),
        Provider("deep_gemm", 3, 90, "fp8 e4m3",
                 "import deep_gemm",
                 _per_token_group_quant_fp8_deepgemm,
                 "DeepGEMM TMA-aligned per-token cast"),
        _ks_provider(_per_token_group_quant_ks, "ks_quantize_fp8_group",
                     "native per-token-group fp8 quant (sm89+)"),
    ]),
    # NVFP4 GEMM (Blackwell native 4-bit float: e2m1 + e4m3 1x16 block scale +
    # fp32 global). No portable ks kernel — Blackwell + FlashInfer/vLLM only.
    Op("nvfp4_gemm", "gemm-quant", None, [
        Provider("flashinfer", 1, 100, "fp4 nvfp4 e2m1",
                 "from flashinfer.gemm import mm_fp4",
                 _nvfp4_gemm_flashinfer, "FlashInfer NVFP4 mm_fp4 (Blackwell)"),
        Provider("vllm", 2, 100, "fp4 nvfp4",
                 "from vllm import _custom_ops",
                 _nvfp4_gemm_vllm, "vLLM cutlass_scaled_fp4_mm (Blackwell)"),
        _ks_provider(_ks_unsupported("nvfp4 GEMM"), "",
                     "no portable fp4 kernel; needs Blackwell + FlashInfer/vLLM"),
    ]),
    # MXFP4 GEMM (OCP microscaling 4-bit: e2m1 + E8M0 block-32 scale). gpt-oss.
    Op("mxfp4_gemm", "gemm-quant", None, [
        Provider("flashinfer", 1, 100, "fp4 mxfp4 e2m1",
                 "from flashinfer.gemm import mm_fp4",
                 _mxfp4_gemm_flashinfer, "FlashInfer MXFP4 mm_fp4 (Blackwell)"),
        Provider("vllm", 2, 100, "fp4 mxfp4",
                 "from vllm import _custom_ops",
                 _mxfp4_gemm_vllm, "vLLM Marlin-MXFP4 (Blackwell)"),
        Provider("torchao", 3, 100, "fp4 mxfp4",
                 "import torchao",
                 _mxfp4_gemm_torchao, "torchao MXFP4 inference"),
        _ks_provider(_ks_unsupported("mxfp4 GEMM"), "",
                     "no portable fp4 kernel; needs FlashInfer/vLLM/torchao"),
    ]),
    Op("fp4_quantize", "gemm-quant", None, [
        Provider("vllm", 1, 100,
                 "fp16, bf16 -> fp4 e2m1 + fp8-e4m3 block scale",
                 "from vllm import _custom_ops as ops; ops.scaled_fp4_quant",
                 _nvfp4_quantize_vllm,
                 "vLLM scaled_fp4_quant activation quant (Blackwell)"),
        Provider("flashinfer", 2, 100,
                 "fp16, bf16 -> fp4 e2m1 + fp8/ue8m0 block scale",
                 "from flashinfer import fp4_quantization",
                 _nvfp4_quantize_flashinfer,
                 "FlashInfer fp4_quantization NVFP4/MXFP4 activation quant"),
        _ks_provider(_ks_unsupported("fp4_quantize"), "",
                     "no portable FP4 quantizer; needs vLLM/FlashInfer"),
    ]),
    Op("mxfp8_quantize", "gemm-quant", None, [
        Provider("vllm", 1, 100,
                 "fp16, bf16 -> mxfp8/fp8 e4m3 with E8M0 block scale",
                 "from vllm import _custom_ops as ops; ops.mxfp8_experts_quant",
                 _mxfp8_quantize_vllm,
                 "vLLM MXFP8 expert quantizer (Blackwell microscaling)"),
        _ks_provider(_ks_unsupported("mxfp8_quantize"), "",
                     "no portable MXFP8 quantizer; needs vLLM/torchao"),
    ]),
    # FP8 attention compute (fp8 QK^T/PV, fp32 softmax) — distinct from fp8 KV
    # store. SageAttention / FlashAttention-3 fp8. ks attention is fp16/bf16 only.
    Op("fp8_attention", "attention", None, [
        Provider("sage-attn", 1, 80, "fp8 e4m3, int8 QK",
                 "import sageattention",
                 _fp8_attention_sage,
                 "SageAttention (INT8 QK + FP8 PV, sm80+)"),
        Provider("flash-attn-3", 2, 90, "fp8 e4m3",
                 "import flash_attn_interface",
                 _fp8_attention_flashinfer,
                 "FlashAttention-3 fp8 (Hopper sm90)"),
        _ks_provider(_ks_unsupported("fp8 attention"), "",
                     "ks attention is fp16/bf16; needs FA3 / SageAttention"),
    ]),
    # FP8 KV-cache quantize-on-write (reshape_and_cache_flash fp8). 2x KV memory.
    # ks_reshape_and_cache is dtype-preserving (no quant), so vLLM is the path.
    Op("fp8_kv_cache", "attention", None, [
        Provider("vllm", 1, 89, "fp8 e4m3 KV",
                 "from vllm import _custom_ops",
                 _reshape_and_cache_fp8_vllm,
                 "vLLM reshape_and_cache_flash (fp8 KV, quantize-on-write)"),
        Provider("flashinfer", 2, 89, "fp8 e4m3 KV, nvfp4 KV",
                 "from flashinfer.page import append_paged_kv_cache",
                 _reshape_and_cache_fp8_flashinfer,
                 "FlashInfer paged KV append / quant-on-write"),
        _sgl_provider(3, _reshape_and_cache_fp8_sgl, min_sm=89,
                      dtypes="fp8 e4m3 KV",
                      note="SGLang reshape_and_cache_flash fp8 KV write"),
        _ks_provider(_ks_unsupported("fp8 KV-cache quant"), "",
                     "ks_reshape_and_cache is dtype-preserving; needs vLLM"),
    ]),
    Op("patch_embed", "gemm-dense", None, [
        Provider("torch", 1, 70, "fp16, bf16, fp32",
                 "import torch",
                 _patch_embed_torch,
                 "torch/cuDNN conv2d/conv3d patch projection"),
        _ks_provider(_ks_unsupported("patch_embed"), "",
                     "no portable patch embedding kernel; needs torch/cuDNN"),
    ]),
    Op("flex_attention", "attention", None, [
        Provider("torch", 1, 80, "fp16, bf16, fp32",
                 "from torch.nn.attention.flex_attention import "
                 "flex_attention, create_block_mask",
                 _flex_attention_torch,
                 "PyTorch FlexAttention score_mod/block_mask path"),
        _ks_provider(_ks_unsupported("flex_attention"), "",
                     "no portable FlexAttention kernel; needs PyTorch"),
    ]),
    Op("varlen_pad", "attention", None, [
        Provider("flash-attn", 1, 80, "fp16, bf16, fp32",
                 "from flash_attn.bert_padding import unpad_input, pad_input, "
                 "index_first_axis",
                 _varlen_pad_flash_attn,
                 "flash-attn bert_padding unpad/pad/index helpers"),
        _ks_provider(_ks_unsupported("varlen_pad"), "",
                     "no portable varlen pack/unpack kernel; needs flash-attn"),
    ]),
    Op("attention_state_merge", "attention", "ks_attention_state_merge", [
        Provider("flashinfer", 1, 80, "fp16, bf16, fp8",
                 "import flashinfer.cascade; flashinfer.cascade.merge_state; "
                 "flashinfer.cascade.merge_states",
                 _attention_state_merge_flashinfer,
                 "FlashInfer cascade merge_state/merge_states (online softmax)"),
        _ks_provider(_attention_state_merge_ks, "ks_attention_state_merge",
                     "portable 2-way log-sum-exp state merge (fp32)"),
    ]),
    Op("rmsnorm", "norm-act-rope", "ks_rmsnorm", [
        Provider("quack", 0, 90, "fp16, bf16, fp32",
                 "from quack import rmsnorm",
                 _rmsnorm_quack, "Quack CuTe-DSL speed-of-light RMSNorm"),
        Provider("flashinfer", 1, 75, "fp16, bf16",
                 "from flashinfer.norm import rmsnorm",
                 _rmsnorm_flashinfer, "FlashInfer fused RMSNorm"),
        _sgl_provider(2, _rmsnorm_sgl,
                      note="SGLang rmsnorm (FlashInfer-derived, PDL sm90)"),
        Provider("vllm", 3, 70, "fp16, bf16, fp32",
                 "from vllm import _custom_ops",
                 _rmsnorm_vllm, "vLLM custom RMSNorm"),
        Provider("liger", 4, 80, "fp16, bf16, fp32",
                 "from liger_kernel.ops.rms_norm import LigerRMSNormFunction",
                 _rmsnorm_liger, "Liger Triton RMSNorm"),
        _ks_provider(_rmsnorm_ks, "ks_rmsnorm"),
    ]),
    Op("fused_add_rmsnorm", "norm-act-rope", "ks_fused_add_rmsnorm", [
        Provider("flashinfer", 1, 75, "fp16, bf16",
                 "from flashinfer.norm import fused_add_rmsnorm",
                 _far_flashinfer, "FlashInfer fused add-RMSNorm"),
        _sgl_provider(2, _far_sgl,
                      note="SGLang fused_add_rmsnorm (in-place residual+norm)"),
        Provider("vllm", 3, 70, "fp16, bf16, fp32",
                 "from vllm import _custom_ops",
                 _far_vllm, "vLLM fused add-RMSNorm"),
        _ks_provider(_far_ks, "ks_fused_add_rmsnorm"),
    ]),
    Op("fused_rmsnorm_gated", "norm-act-rope", "ks_fused_rmsnorm_gated", [
        Provider("flash-linear-attention", 1, 80, "fp16, bf16",
                 "from fla.modules import FusedRMSNormGated",
                 _fused_rmsnorm_gated_fla,
                 "FLA fused RMSNorm plus SiLU/sigmoid gate"),
        _ks_provider(_fused_rmsnorm_gated_ks, "ks_fused_rmsnorm_gated",
                     "portable gated RMSNorm (norm-then-gate, matches FLA)"),
    ]),
    Op("gemma_rmsnorm", "norm-act-rope", "ks_gemma_rmsnorm", [
        Provider("flashinfer", 1, 75, "fp16, bf16",
                 "from flashinfer.norm import gemma_rmsnorm",
                 _gemma_rmsnorm_flashinfer, "FlashInfer gemma_rmsnorm"),
        _sgl_provider(2, _gemma_rmsnorm_sgl,
                      note="SGLang gemma_rmsnorm ((weight+1) scale)"),
        _ks_provider(_gemma_rmsnorm_ks, "ks_gemma_rmsnorm",
                     "Gemma-style (weight+1) RMSNorm"),
    ]),
    Op("rope", "norm-act-rope", "ks_rope", [
        Provider("flashinfer", 1, 75, "fp16, bf16",
                 "from flashinfer.rope import apply_rope_with_cos_sin_cache",
                 _rope_flashinfer, "FlashInfer RoPE"),
        _sgl_provider(2, _rope_sgl,
                      note="SGLang rotary_embedding (NeoX/interleaved)"),
        Provider("vllm", 3, 70, "fp16, bf16, fp32",
                 "from vllm import _custom_ops",
                 _rope_vllm, "vLLM rotary_embedding"),
        Provider("liger", 4, 80, "fp16, bf16, fp32",
                 "from liger_kernel.ops.rope import LigerRopeFunction",
                 _rope_liger, "Liger Triton RoPE"),
        _ks_provider(_rope_ks, "ks_rope"),
    ]),
    Op("mrope", "norm-act-rope", None, [
        Provider("vllm", 1, 70, "fp16, bf16",
                 "from vllm.model_executor.layers.rotary_embedding.mrope "
                 "import triton_mrope",
                 _mrope_vllm,
                 "vLLM Triton multimodal/3D RoPE with mrope_section"),
        _ks_provider(_ks_unsupported("mrope"), "",
                     "no portable multimodal RoPE; needs vLLM"),
    ]),
    Op("swiglu", "norm-act-rope", "ks_silu_and_mul", [
        Provider("flashinfer", 1, 75, "fp16, bf16",
                 "from flashinfer.activation import silu_and_mul",
                 _swiglu_flashinfer, "FlashInfer silu_and_mul"),
        _sgl_provider(2, _swiglu_sgl, note="SGLang silu_and_mul"),
        Provider("vllm", 3, 70, "fp16, bf16, fp32",
                 "from vllm import _custom_ops",
                 _swiglu_vllm, "vLLM silu_and_mul"),
        Provider("liger", 4, 80, "fp16, bf16, fp32",
                 "from liger_kernel.ops.swiglu import LigerSiLUMulFunction",
                 _swiglu_liger, "Liger SwiGLU"),
        _ks_provider(_swiglu_ks, "ks_silu_and_mul"),
    ]),
    Op("cross_entropy", "loss-optim-misc", "ks_cross_entropy", [
        Provider("quack", 0, 90, "fp32, bf16, fp16",
                 "from quack import cross_entropy",
                 _ce_quack, "Quack CuTe-DSL cross-entropy"),
        Provider("liger", 1, 70, "fp32, bf16, fp16",
                 "from liger_kernel.transformers.functional import "
                 "liger_cross_entropy",
                 _ce_liger, "Liger fused cross-entropy"),
        Provider("torch", 2, 70, "fp32, bf16, fp16",
                 "import torch",
                 _ce_torch, "torch.nn.functional.cross_entropy"),
        _ks_provider(_ce_ks, "ks_cross_entropy"),
    ]),
    Op("fused_linear_ce", "loss-optim-misc", "ks_fused_linear_cross_entropy", [
        Provider("liger", 1, 80, "fp16, bf16, fp32",
                 "from liger_kernel.ops.fused_linear_cross_entropy import "
                 "LigerFusedLinearCrossEntropyFunction",
                 _fused_linear_ce_liger,
                 "Liger fused LM-head matmul + CE (no logits materialized)"),
        _ks_provider(_fused_linear_ce_ks, "ks_fused_linear_cross_entropy",
                     "chunked fused-linear CE fallback"),
    ]),
    Op("muon", "loss-optim-misc", None, [
        Provider("torch", 1, 80, "fp16, bf16, fp32",
                 "import torch",
                 _muon_torch,
                 "Muon Newton-Schulz orthogonalized update (matmul/cuBLAS-bound)"),
        _ks_provider(_ks_unsupported("muon"), "",
                     "no portable Muon optimizer kernel; torch matmul is the path"),
    ]),
    Op("moe", "moe-comm", "ks_moe_grouped_gemm", [
        # Hopper/Blackwell: DeepGEMM grouped FP8 (DeepSeek-V3 production,
        # Mega-MoE) is the reference MoE GEMM — rank-1, gated sm90.
        Provider("deep_gemm", 1, 90, "fp8 e4m3, fp32 block scales",
                 "import deep_gemm; from deep_gemm import "
                 "m_grouped_fp8_gemm_nt_masked",
                 _moe_deepgemm,
                 "DeepGEMM grouped FP8 (contiguous/masked, Hopper/Blackwell)"),
        # sgl-kernel CUTLASS blockwise-fp8 grouped GEMM — the stated alignment
        # target (rank-2, sm90). Now calls the REAL sgl grouped mm (was a stub
        # that delegated to the ks kernel).
        _sgl_provider(2, _moe_sgl, min_sm=90, dtypes="fp8 e4m3",
                      note="SGLang CUTLASS grouped FP8 (fp8_blockwise_..._mm)"),
        Provider("flashinfer-cutlass-moe", 2, 89, "fp16, bf16, fp8, nvfp4",
                 "from flashinfer.fused_moe import cutlass_fused_moe",
                 _moe_flashinfer_cutlass,
                 "FlashInfer CUTLASS fused MoE (gather+GEMM+scatter)"),
        # Ampere/Ada (no FP8 hw): vLLM Triton fused_experts (bf16/INT4 grouped).
        Provider("vllm", 3, 80, "bf16, fp16, fp8, int4",
                 "from vllm.model_executor.layers.fused_moe.fused_moe import "
                 "fused_experts",
                 _moe_vllm, "vLLM fused_experts / fused_moe (Triton, sm80+)"),
        _ks_provider(_moe_ks, "ks_moe_grouped_gemm",
                     "grouped GEMM over experts (portable fallback)"),
    ]),
    Op("moe_gate", "moe-comm", "ks_moe_gate_softmax_topk", [
        _sgl_provider(1, _moe_gate_sgl, dtypes="fp16, bf16, fp32",
                      note="SGLang topk_softmax fused gate (specialty)"),
        Provider("vllm", 2, 80, "fp16, bf16, fp32",
                 "from vllm import _custom_ops",
                 _moe_gate_vllm, "vLLM topk_softmax gate"),
        _ks_provider(_moe_gate_ks, "ks_moe_gate_softmax_topk",
                     "softmax + top-k gate (near-parity fallback)"),
    ]),
    Op("moe_group_gate", "moe-comm", "ks_moe_gate_sigmoid_group_topk", [
        _sgl_provider(1, _moe_group_gate_sgl, dtypes="fp16, bf16, fp32",
                      note="SGLang moe_fused_gate grouped-topk (specialty)"),
        Provider("vllm", 2, 80, "fp16, bf16, fp32",
                 "from vllm import _custom_ops",
                 _moe_group_gate_vllm, "vLLM grouped_topk gate"),
        _ks_provider(_moe_group_gate_ks, "ks_moe_gate_sigmoid_group_topk",
                     "sigmoid + group-limited top-k gate (near-parity)"),
    ]),
    Op("sampling", "sampling-logitproc", "ks_sample", [
        Provider("flashinfer", 1, 75, "fp32 probs",
                 "import flashinfer.sampling",
                 _sampling_flashinfer, "FlashInfer fused top-k/top-p sampling"),
        _sgl_provider(2, _sampling_sgl, dtypes="fp32 probs",
                      note="SGLang renorm + categorical sampling"),
        _ks_provider(_sampling_ks, "ks_sample",
                     "fused temp/top-k/top-p sampler"),
    ]),
    Op("min_p_sampling", "sampling-logitproc", None, [
        Provider("flashinfer", 1, 75, "fp32, fp16, bf16 probs",
                 "import flashinfer.sampling; "
                 "flashinfer.sampling.min_p_sampling_from_probs",
                 _min_p_sampling_flashinfer,
                 "FlashInfer sorting-free min-p filter + sample"),
        _ks_provider(_ks_unsupported("min_p_sampling"), "",
                     "ks_sample has no min-p path; needs FlashInfer"),
    ]),
    Op("chain_speculative_sampling", "sampling-logitproc", None, [
        Provider("flashinfer", 1, 75, "fp32, fp16, bf16 probs; int32 token ids",
                 "import flashinfer.sampling; "
                 "flashinfer.sampling.chain_speculative_sampling",
                 _chain_speculative_sampling_flashinfer,
                 "FlashInfer draft/target chain speculative accept-reject"),
        _ks_provider(_ks_unsupported("chain_speculative_sampling"), "",
                     "no portable speculative verifier; needs FlashInfer"),
    ]),
    Op("apply_token_bitmask", "sampling-logitproc", None, [
        Provider("xgrammar", 1, 70,
                 "fp32, fp16, bf16 logits; int32 packed bitmask",
                 "import xgrammar; xgrammar.apply_token_bitmask_inplace; "
                 "xgrammar.allocate_token_bitmask",
                 _apply_token_bitmask_xgrammar,
                 "xgrammar CUDA/Triton vocab bitmask apply for guided decode"),
        _ks_provider(_ks_unsupported("apply_token_bitmask"), "",
                     "no portable token-bitmask kernel; needs xgrammar"),
    ]),
    Op("selective_scan", "ssm", "ks_selective_scan", [
        Provider("mamba-ssm", 1, 80, "fp16, bf16, fp32",
                 "from mamba_ssm.ops.selective_scan_interface import "
                 "selective_scan_fn; "
                 "from mamba_ssm.ops.triton.ssd_combined import "
                 "mamba_chunk_scan_combined; "
                 "from mamba_ssm.ops.triton.selective_state_update import "
                 "selective_state_update",
                 _selective_scan_mamba,
                 "Mamba selective_scan_fn + SSD chunk/update"),
        _ks_provider(_selective_scan_ks, "ks_selective_scan",
                     "portable selective scan"),
    ]),
    Op("mamba2_ssd_chunk_scan", "ssm", None, [
        Provider("mamba-ssm", 1, 80, "fp16, bf16, fp32",
                 "from mamba_ssm.ops.triton.ssd_combined import "
                 "mamba_chunk_scan_combined_varlen",
                 _mamba2_ssd_chunk_scan_mamba,
                 "Mamba-2 SSD varlen chunk scan"),
        _ks_provider(_ks_unsupported("mamba2_ssd_chunk_scan"), "",
                     "Mamba-1 selective_scan is the dense recurrent fallback; "
                     "SSD chunk scan needs mamba-ssm"),
    ]),
    Op("causal_conv1d", "ssm", "ks_causal_conv1d", [
        Provider("causal-conv1d", 1, 80, "fp16, bf16, fp32",
                 "from causal_conv1d import causal_conv1d_fn, "
                 "causal_conv1d_update",
                 _causal_conv1d_external,
                 "causal-conv1d fused prefill/update kernels"),
        _ks_provider(_causal_conv1d_ks, "ks_causal_conv1d",
                     "portable causal depthwise conv1d"),
    ]),
    Op("gated_delta_rule", "linear-attn", "ks_gated_delta_rule", [
        Provider("flash-linear-attention", 1, 80, "fp16, bf16",
                 "from fla.ops import chunk_gated_delta_rule, chunk_kda, "
                 "fused_recurrent_gated_delta_rule",
                 _gated_delta_rule_fla,
                 "FLA chunk/recurrent gated delta rule / KDA"),
        _ks_provider(_gated_delta_rule_ks, "ks_gated_delta_rule",
                     "portable gated delta rule"),
    ]),
    Op("gated_linear_attn", "linear-attn", "ks_gated_linear_attn", [
        Provider("flash-linear-attention", 1, 80, "fp16, bf16",
                 "from fla.ops import chunk_gla, chunk_simple_gla, "
                 "chunk_lightning_attn, fused_recurrent_gla, "
                 "fused_recurrent_simple_gla",
                 _gated_linear_attn_fla,
                 "FLA chunk/recurrent GLA / simple GLA / lightning"),
        _ks_provider(_gated_linear_attn_ks, "ks_gated_linear_attn",
                     "portable gated linear attention"),
    ]),
    Op("rwkv_wkv7", "linear-attn", "ks_rwkv_wkv7", [
        Provider("flash-linear-attention", 1, 80, "fp16, bf16",
                 "from fla.ops import chunk_rwkv7, fused_recurrent_rwkv7",
                 _rwkv_wkv7_fla,
                 "FLA chunk/recurrent RWKV-7 WKV"),
        _ks_provider(_rwkv_wkv7_ks, "ks_rwkv_wkv7",
                     "portable RWKV-7 WKV"),
    ]),
]


# Build {op_name: Op} preserving order.
OPS: "Dict[str, Op]" = {op.name: op for op in _OPS_RAW}
OP_ORDER: List[str] = [op.name for op in _OPS_RAW]


# =========================================================================== #
# SINGLE SOURCE OF TRUTH for ranking: the Cartesian optimal-selection table
# (providers/optimal.json, generated by scripts/gen_optimal.py and loaded by
# backends.optimal). The dispatcher orders each op's providers by that table's
# per-(op, sm, dtype) fallback_chain instead of the hard-coded Provider.rank,
# so measured benchmark winners (and the curated heuristic baseline) drive
# selection from one place. The static `rank` survives only as the tie-break /
# fallback for providers the table doesn't mention and for queries the table has
# no cell for (e.g. sm unknown). This keeps every arch gate + availability probe
# intact: the table decides the *order*, the probes decide what is *runnable*.
# =========================================================================== #
def optimal_order(op: str, sm: Optional[int], dtype) -> "List[Provider]":
    """Providers for ``op`` ordered by the optimal table for ``(op, sm, dtype)``.

    The table's ``fallback_chain`` is a (usually) subsequence of the static
    provider order; we weave any provider it doesn't mention back in at its
    static-rank position (just after the last chain member that precedes it in
    the static order), so the static order is preserved wherever the table has
    no opinion, and a measured winner is promoted exactly where the table says.
    The kernel-set fallback (static rank 99) always sorts last."""
    providers = OPS[op].providers
    try:
        from .optimal import optimal_chain
        chain = optimal_chain(op, sm, dtype)
    except Exception:
        chain = []
    if chain == [KERNEL_SET]:
        return [p for p in providers if p.name == KERNEL_SET]
    if chain:
        by_name = {p.name: p for p in providers}
        ordered = [by_name[name] for name in chain if name in by_name]
        named = {p.name for p in ordered}
        extras = [p for p in providers if p.name not in named]
        extras = sorted(extras, key=lambda p: (p.rank, p.name))
        if ordered and ordered[-1].name == KERNEL_SET:
            return ordered[:-1] + extras + ordered[-1:]
        return ordered + extras
    chain_idx = {name: i for i, name in enumerate(chain)}
    keys: Dict[str, tuple] = {}
    last_seen = -1  # chain index of the last chain member seen in static order
    for si, p in enumerate(providers):
        if p.name in chain_idx:
            ci = chain_idx[p.name]
            last_seen = ci
            keys[p.name] = (ci, 0, p.rank, si)
        else:
            # Not named by this chain: keep it adjacent to the last preceding
            # chain member, ordered by its static rank then static index.
            keys[p.name] = (last_seen, 1, p.rank, si)
    return sorted(providers, key=lambda p: keys[p.name])
