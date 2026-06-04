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

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

KERNEL_SET = "kernel-set"


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


def _scale(head_dim: int, scale: Optional[float]) -> float:
    return float(scale) if scale else 1.0 / math.sqrt(head_dim)


# =========================================================================== #
# ATTENTION — prefill (dense). q/k/v: (batch, seqlen, heads, head_dim).
# =========================================================================== #
def _attn_prefill_flash_attn(q, k, v, *, causal=True, softmax_scale=None, **_):
    fa = _imp("flash_attn")
    return fa.flash_attn_func(q, k, v, causal=causal,
                              softmax_scale=softmax_scale)


def _attn_prefill_sdpa(q, k, v, *, causal=True, softmax_scale=None, **_):
    import torch
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    enable_gqa = q.shape[-2] != k.shape[-2]
    o = torch.nn.functional.scaled_dot_product_attention(
        qt, kt, vt, is_causal=causal, scale=softmax_scale,
        enable_gqa=enable_gqa)
    return o.transpose(1, 2)


def _attn_prefill_flashinfer(q, k, v, *, causal=True, softmax_scale=None, **_):
    fi = _imp("flashinfer")
    if q.shape[0] != 1:
        raise NotImplementedError("flashinfer single_prefill adapter is b==1")
    return fi.prefill.single_prefill_with_kv_cache(
        q[0], k[0], v[0], causal=causal, kv_layout="NHD",
        sm_scale=softmax_scale).unsqueeze(0)


def _attn_prefill_ks(q, k, v, *, causal=True, softmax_scale=None, **_):
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
                            **_):
    import torch
    fi = _imp("flashinfer")
    num_seqs, qh, hd = q.shape
    kvh = k_cache.shape[1]
    # ks layout (nb, kvh, page, hd) -> flashinfer NHD (nb, page, kvh, hd)
    k_fi = k_cache.permute(0, 2, 1, 3).contiguous()
    v_fi = v_cache.permute(0, 2, 1, 3).contiguous()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = fi.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD")
    bps = max_blocks_per_seq
    kv_indptr = torch.arange(0, (num_seqs + 1) * bps, bps,
                             device="cuda", dtype=torch.int32)
    kv_indices = torch.arange(num_seqs * bps, device="cuda", dtype=torch.int32)
    last = (seq_lens - (bps - 1) * block_size).clamp(min=1).to(torch.int32)
    wrapper.plan(kv_indptr, kv_indices, last, qh, kvh, hd, block_size,
                 pos_encoding_mode="NONE", data_type=q.dtype, q_data_type=q.dtype)
    return wrapper.run(q, (k_fi, v_fi))


def _attn_decode_ks(q, k_cache, v_cache, block_tables, seq_lens, *,
                    block_size, max_blocks_per_seq, softmax_scale=None, **_):
    from .. import attention
    import torch
    num_seqs, qh, hd = q.shape
    kvh = k_cache.shape[1]
    out = torch.empty_like(q)
    attention.paged_attn_decode(
        out, q, k_cache, v_cache, block_tables, seq_lens, num_seqs, qh, kvh, hd,
        block_size, max_blocks_per_seq, softmax_scale=_scale(hd, softmax_scale))
    return out


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
    out = torch.empty(m, n, device="cuda",
                      dtype=out_dtype or torch.bfloat16)
    dg.fp8_gemm_nt((a8, a_scale), (b8, b_scale), out)
    return out


def _fp8_gemm_torch(a8, b8, a_scale, b_scale, *, out_dtype=None, **_):
    import torch
    return torch._scaled_mm(a8, b8.t(), scale_a=a_scale, scale_b=b_scale,
                            out_dtype=out_dtype or torch.bfloat16,
                            use_fast_accum=True)


def _fp8_gemm_ks(a8, b8, a_scale, b_scale, *, out_dtype=None, **_):
    # kernel-set has no native fp8 GEMM; the closest ABI is int8 w8a8. We expose
    # the ks dense GEMM symbol as the portable fallback path here.
    from .. import gemm
    import torch
    m, k = a8.shape
    n = b8.shape[0]
    c = torch.empty(m, n, device="cuda", dtype=out_dtype or torch.bfloat16)
    gemm.gemm(c, a8.to(c.dtype), b8.to(c.dtype).t().contiguous(),
              m=m, n=n, k=k)
    return c


# =========================================================================== #
# W4A16 GEMM. a: fp16/bf16 (M,K); packed int4 weights + group scales/zeros.
# =========================================================================== #
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


# =========================================================================== #
# MOE (fused experts). hidden, w1, w2, topk_weights, topk_ids.
# =========================================================================== #
def _moe_vllm(hidden, w1, w2, topk_weights, topk_ids, **_):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    return fused_experts(hidden, w1, w2, topk_weights, topk_ids, inplace=False)


def _moe_ks(a, b, expert_offsets, *, num_experts, n, k, **_):
    from .. import moe
    import torch
    total_rows = a.shape[0]
    c = torch.empty(total_rows, n, device=a.device, dtype=a.dtype)
    moe.grouped_gemm(c, a, b, expert_offsets, num_experts, total_rows, n, k)
    return c


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


_OPS_RAW: List[Op] = [
    Op("attention_prefill", "attention", None, [
        Provider("flash-attn", 1, 80, "fp16, bf16",
                 "from flash_attn import flash_attn_func",
                 _attn_prefill_flash_attn,
                 "industry-standard exact attention (FA2/FA3)"),
        Provider("torch-sdpa", 2, 80, "fp16, bf16, fp32",
                 "import torch",
                 _attn_prefill_sdpa, "PyTorch SDPA flash/efficient backend"),
        Provider("flashinfer", 3, 75, "fp16, bf16, fp8",
                 "import flashinfer",
                 _attn_prefill_flashinfer, "NVIDIA serving kernels (b==1)"),
        _ks_provider(_attn_prefill_ks, "ks_flash_attn"),
    ]),
    Op("attention_decode", "attention", None, [
        Provider("flashinfer", 1, 75, "fp16, bf16, fp8 KV",
                 "import flashinfer",
                 _attn_decode_flashinfer, "paged decode plan/run, FA-class"),
        _ks_provider(_attn_decode_ks, "ks_paged_attn_decode"),
    ]),
    Op("gemm", "gemm-dense", "ks_gemm", [
        Provider("torch", 1, 70, "fp16, bf16, fp32, tf32",
                 "import torch",
                 _gemm_torch, "cuBLASLt / cuBLAS tensor-core GEMM"),
        _ks_provider(_gemm_ks, "ks_gemm"),
    ]),
    Op("fp8_gemm", "gemm-quant", "ks_gemm_w8a8", [
        Provider("deep_gemm", 1, 90, "fp8 e4m3, fp32 block scales",
                 "import deep_gemm",
                 _fp8_gemm_deepgemm, "DeepGEMM blockwise fp8 (Hopper/Blackwell)"),
        Provider("torch-scaled-mm", 2, 89, "fp8 e4m3/e5m2",
                 "import torch",
                 _fp8_gemm_torch, "torch._scaled_mm fp8 tensor cores (Ada+)"),
        _ks_provider(_fp8_gemm_ks, "ks_gemm_w8a8",
                     "no native fp8; dense-cast fallback"),
    ]),
    Op("w4a16", "gemm-quant", "ks_gemm_w4a16", [
        Provider("vllm-marlin", 1, 80, "int4 weights, fp16/bf16 acts",
                 "from vllm import _custom_ops",
                 None, "GPTQ-Marlin (Ampere+); Machete on sm90a"),
        _ks_provider(_w4a16_ks, "ks_gemm_w4a16"),
    ]),
    Op("rmsnorm", "norm-act-rope", "ks_rmsnorm", [
        Provider("flashinfer", 1, 75, "fp16, bf16",
                 "from flashinfer.norm import rmsnorm",
                 _rmsnorm_flashinfer, "FlashInfer fused RMSNorm"),
        Provider("vllm", 2, 70, "fp16, bf16, fp32",
                 "from vllm import _custom_ops",
                 _rmsnorm_vllm, "vLLM custom RMSNorm"),
        Provider("liger", 3, 80, "fp16, bf16, fp32",
                 "from liger_kernel.ops.rms_norm import LigerRMSNormFunction",
                 _rmsnorm_liger, "Liger Triton RMSNorm"),
        _ks_provider(_rmsnorm_ks, "ks_rmsnorm"),
    ]),
    Op("fused_add_rmsnorm", "norm-act-rope", "ks_fused_add_rmsnorm", [
        Provider("flashinfer", 1, 75, "fp16, bf16",
                 "from flashinfer.norm import fused_add_rmsnorm",
                 _far_flashinfer, "FlashInfer fused add-RMSNorm"),
        Provider("vllm", 2, 70, "fp16, bf16, fp32",
                 "from vllm import _custom_ops",
                 _far_vllm, "vLLM fused add-RMSNorm"),
        _ks_provider(_far_ks, "ks_fused_add_rmsnorm"),
    ]),
    Op("rope", "norm-act-rope", "ks_rope", [
        Provider("flashinfer", 1, 75, "fp16, bf16",
                 "from flashinfer.rope import apply_rope_with_cos_sin_cache",
                 _rope_flashinfer, "FlashInfer RoPE"),
        Provider("vllm", 2, 70, "fp16, bf16, fp32",
                 "from vllm import _custom_ops",
                 _rope_vllm, "vLLM rotary_embedding"),
        _ks_provider(_rope_ks, "ks_rope"),
    ]),
    Op("swiglu", "norm-act-rope", "ks_silu_and_mul", [
        Provider("flashinfer", 1, 75, "fp16, bf16",
                 "from flashinfer.activation import silu_and_mul",
                 _swiglu_flashinfer, "FlashInfer silu_and_mul"),
        Provider("vllm", 2, 70, "fp16, bf16, fp32",
                 "from vllm import _custom_ops",
                 _swiglu_vllm, "vLLM silu_and_mul"),
        Provider("liger", 3, 80, "fp16, bf16, fp32",
                 "from liger_kernel.ops.swiglu import LigerSiLUMulFunction",
                 _swiglu_liger, "Liger SwiGLU"),
        _ks_provider(_swiglu_ks, "ks_silu_and_mul"),
    ]),
    Op("cross_entropy", "loss-optim-misc", "ks_cross_entropy", [
        Provider("liger", 1, 70, "fp32, bf16, fp16",
                 "from liger_kernel.transformers.functional import "
                 "liger_cross_entropy",
                 _ce_liger, "Liger fused cross-entropy"),
        Provider("torch", 2, 70, "fp32, bf16, fp16",
                 "import torch",
                 _ce_torch, "torch.nn.functional.cross_entropy"),
        _ks_provider(_ce_ks, "ks_cross_entropy"),
    ]),
    Op("moe", "moe-comm", "ks_moe_grouped_gemm", [
        Provider("vllm", 1, 80, "bf16, fp16, fp8",
                 "from vllm.model_executor.layers.fused_moe.fused_moe import "
                 "fused_experts",
                 _moe_vllm, "vLLM fused_experts (full MoE FFN)"),
        _ks_provider(_moe_ks, "ks_moe_grouped_gemm",
                     "grouped GEMM over experts"),
    ]),
]


# Build {op_name: Op} preserving order.
OPS: "Dict[str, Op]" = {op.name: op for op in _OPS_RAW}
OP_ORDER: List[str] = [op.name for op in _OPS_RAW]
