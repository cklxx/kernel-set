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


def _attn_prefill_fa4(q, k, v, *, causal=True, softmax_scale=None, **_):
    # FlashAttention-4 (Blackwell sm100): the CuTe-DSL kernel exported as
    # flash_attn.cute.flash_attn_func. Same dense (b, s, h, d) ergonomics as FA2.
    fac = _imp("flash_attn.cute")
    return fac.flash_attn_func(q, k, v, causal=causal,
                               softmax_scale=softmax_scale)


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
# vLLM owns the SOTA mixed-input INT4xFP16 kernels: GPTQ/AWQ-Marlin (sm80+) and
# Machete (sm90a, CUTLASS TMA+WGMMA weight-prepack — beats Marlin on Hopper).
# =========================================================================== #
def _w4a16_marlin(a, b_packed, scales, zeros, *, group_size=128,
                  workspace=None, **_):
    # vLLM GPTQ-Marlin: ops.gptq_marlin_gemm(a, b_q_weight, b_scales, b_zeros,
    #   g_idx, perm, workspace, num_bits=4, size_m, size_n, size_k, is_k_full=...)
    # The weights must already be Marlin-prepacked (gptq_marlin_repack); callers
    # supply the prepacked `b_packed` + `scales` (+ optional `zeros`).
    import torch
    from vllm import _custom_ops as ops
    m, k = a.shape
    n = scales.shape[1]
    if workspace is None:
        workspace = torch.zeros(n // 64 * 16, device=a.device, dtype=torch.int32)
    return ops.gptq_marlin_gemm(
        a, b_packed, scales, zeros, workspace, num_bits=4,
        size_m=m, size_n=n, size_k=k, is_k_full=True)


def _w4a16_machete(a, b_packed, scales, zeros, *, group_size=128, **_):
    # vLLM Machete (sm90a): CUTLASS TMA+WGMMA mixed-input GEMM with weight
    # pre-packing — the Hopper-optimal W4A16/W4A8 path (beats Marlin on sm90).
    # ops.machete_mm(a, b_q, b_type, b_group_scales, b_group_zeros,
    #   b_group_size, ...). Weights are machete_prepack_B-packed by the caller.
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    return ops.machete_mm(
        a=a, b_q=b_packed, b_type=scalar_types.uint4b8,
        b_group_scales=scales, b_group_zeros=zeros,
        b_group_size=group_size, out_type=a.dtype)


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
    # Marlin INT8 (vLLM) — the Blackwell (sm100) fallback where CUTLASS INT8 is
    # unsupported. ops.gptq_marlin_24_gemm / marlin int8 path; weights are
    # Marlin-prepacked. Exposed as the sm100 rank-1 int8 provider.
    import torch
    from vllm import _custom_ops as ops
    m, k = a8.shape
    n = b8.shape[1]
    workspace = torch.zeros(n // 64 * 16, device=a8.device, dtype=torch.int32)
    return ops.gptq_marlin_gemm(
        a8, b8, b_scale, a_scale, workspace, num_bits=8,
        size_m=m, size_n=n, size_k=k, is_k_full=True)


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
    # filtering (top_k_first order). We return the renormalized probs.
    sk = _imp("sgl_kernel")
    out = probs
    if top_k is not None:
        out = sk.top_k_renorm_prob(out, top_k)
    if top_p is not None:
        out = sk.top_p_renorm_prob(out, top_p)
    return out


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


def _moe_deepgemm(a, b, expert_offsets, *, num_experts, n, k,
                  a_scale=None, b_scale=None, **_):
    # DeepGEMM grouped (contiguous) FP8 GEMM — the reference MoE GEMM on
    # Hopper/Blackwell (DeepSeek-V3 production, Mega-MoE). Contiguous-layout
    # grouped fp8: m_grouped_fp8_gemm_nt_contiguous((a, a_s), (b, b_s), out,
    # m_indices). `expert_offsets` is converted to the per-row m_indices map.
    import torch
    dg = _imp("deep_gemm")
    total_rows = a.shape[0]
    out = torch.empty(total_rows, n, device=a.device, dtype=torch.bfloat16)
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


def _attn_prefill_sgl(q, k, v, *, causal=True, softmax_scale=None, **_):
    # sgl_kernel.flash_attn_varlen_func (FA3) needs cu_seqlens + max_seqlens.
    # Build them for the dense (batch, seqlen, heads, head_dim) layout.
    import torch
    sk = _imp("sgl_kernel")
    b, s, qh, hd = q.shape
    kvh = k.shape[-2]
    qf = q.reshape(b * s, qh, hd)
    kf = k.reshape(b * s, kvh, hd)
    vf = v.reshape(b * s, kvh, hd)
    cu = torch.arange(0, (b + 1) * s, s, device=q.device, dtype=torch.int32)
    out = sk.flash_attn_varlen_func(
        qf, kf, vf, cu, cu, max_seqlen_q=s, max_seqlen_k=s,
        softmax_scale=_scale(hd, softmax_scale), causal=causal)
    return out.reshape(b, s, qh, hd)


def _attn_decode_sgl(q, k_cache, v_cache, block_tables, seq_lens, *,
                     block_size, max_blocks_per_seq, softmax_scale=None, **_):
    # sgl_kernel.flash_attn_with_kvcache: paged decode with FA3. ks caches are
    # (num_blocks, kvh, page, hd); FA3 wants (num_blocks, page, kvh, hd).
    import torch
    sk = _imp("sgl_kernel")
    num_seqs, qh, hd = q.shape
    kvh = k_cache.shape[1]
    qf = q.reshape(num_seqs, 1, qh, hd)
    k_fa = k_cache.permute(0, 2, 1, 3).contiguous()
    v_fa = v_cache.permute(0, 2, 1, 3).contiguous()
    out = sk.flash_attn_with_kvcache(
        qf, k_fa, v_fa, page_table=block_tables, cache_seqlens=seq_lens,
        softmax_scale=_scale(hd, softmax_scale), causal=False)
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
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = fi.mla.BatchMLAPagedAttentionWrapper(workspace, backend="auto")
    bps = max_blocks_per_seq
    q_indptr = torch.arange(num_seqs + 1, device="cuda", dtype=torch.int32)
    kv_indptr = torch.arange(0, (num_seqs + 1) * bps, bps,
                             device="cuda", dtype=torch.int32)
    kv_indices = torch.arange(num_seqs * bps, device="cuda", dtype=torch.int32)
    wrapper.plan(q_indptr, kv_indptr, kv_indices, seq_lens.to(torch.int32),
                 heads, lora, rope_dim, block_size,
                 False, _scale(lora + rope_dim, softmax_scale),
                 q_nope.dtype, kv_cache.dtype)
    ckv = kv_cache[..., :lora].contiguous()
    kpe = kv_cache[..., lora:lora + rope_dim].contiguous()
    return wrapper.run(q_nope, q_pe, ckv, kpe)


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


_OPS_RAW: List[Op] = [
    Op("attention_prefill", "attention", None, [
        # FA4 (CuTe-DSL) is the Blackwell-optimal path; gated sm100 so it only
        # outranks FA2/FA3 on B200. flash-attn (FA2 sm80/89, FA3 auto on sm90)
        # remains rank-1 for Ampere/Ada/Hopper.
        Provider("flash-attn-cute", 0, 100, "fp16, bf16",
                 "import flash_attn.cute",
                 _attn_prefill_fa4,
                 "FlashAttention-4 CuTe-DSL (Blackwell sm100)"),
        Provider("flash-attn", 1, 80, "fp16, bf16",
                 "from flash_attn import flash_attn_func",
                 _attn_prefill_flash_attn,
                 "industry-standard exact attention (FA2 sm80/89, FA3 sm90)"),
        _sgl_provider(2, _attn_prefill_sgl, min_sm=90,
                      note="SGLang FA3 (flash_attn_varlen_func, sm90)"),
        Provider("torch-sdpa", 3, 80, "fp16, bf16, fp32",
                 "import torch",
                 _attn_prefill_sdpa, "PyTorch SDPA flash/efficient backend"),
        Provider("flashinfer", 4, 75, "fp16, bf16, fp8",
                 "import flashinfer",
                 _attn_prefill_flashinfer, "NVIDIA serving kernels (b==1)"),
        _ks_provider(_attn_prefill_ks, "ks_flash_attn"),
    ]),
    Op("attention_decode", "attention", None, [
        Provider("flashinfer", 1, 75, "fp16, bf16, fp8 KV",
                 "import flashinfer",
                 _attn_decode_flashinfer, "paged decode plan/run, FA-class"),
        _sgl_provider(2, _attn_decode_sgl, min_sm=90,
                      note="SGLang FA3 paged decode (flash_attn_with_kvcache)"),
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
                 "import flashinfer",
                 _mla_decode_flashinfer,
                 "FlashInfer MLA paged decode (portable sm80+)"),
        _ks_provider(_mla_decode_ks, "ks_mla_decode",
                     "kernel-set absorbed-MLA decode"),
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
                 "from vllm import _custom_ops",
                 _int8_gemm_marlin, "Marlin INT8 (vLLM, Blackwell sm100)"),
        # Registry true rank-1 across sm75-sm90: vLLM CUTLASS INT8 W8A8
        # (SmoothQuant / compressed-tensors, symmetric + azp).
        Provider("vllm", 1, 80, "int8 w8a8",
                 "from vllm import _custom_ops",
                 _int8_gemm_vllm, "vLLM CUTLASS int8 W8A8 (SmoothQuant)"),
        # sgl-kernel int8_scaled_mm is the rank-2 alignment target.
        _sgl_provider(2, _int8_gemm_sgl, min_sm=80, dtypes="int8 w8a8",
                      note="SGLang CUTLASS int8_scaled_mm"),
        _ks_provider(_int8_gemm_ks, "ks_gemm_w8a8",
                     "int8 W8A8 scaled-mm (native ABI; portable fallback)"),
    ]),
    Op("w4a16", "gemm-quant", "ks_gemm_w4a16", [
        # Hopper (sm90a): Machete (CUTLASS TMA+WGMMA weight-prepack) beats
        # Marlin — rank-0, gated sm90 so it only outranks Marlin on Hopper+.
        Provider("vllm-machete", 0, 90, "int4 weights, fp16/bf16 acts",
                 "from vllm import _custom_ops",
                 _w4a16_machete, "Machete W4A16/W4A8 (vLLM, Hopper sm90a)"),
        # Ampere/Ada/Blackwell: GPTQ/AWQ-Marlin is the de-facto W4A16 kernel
        # (~4x over fp16). Now WIRED (was call=None) so it actually dispatches.
        Provider("vllm-marlin", 1, 80, "int4 weights, fp16/bf16 acts",
                 "from vllm import _custom_ops",
                 _w4a16_marlin, "GPTQ/AWQ-Marlin (Ampere+; NVFP4-Marlin sm100)"),
        _ks_provider(_w4a16_ks, "ks_gemm_w4a16",
                     "portable INT4 fallback (correctness only)"),
    ]),
    Op("rmsnorm", "norm-act-rope", "ks_rmsnorm", [
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
    Op("gemma_rmsnorm", "norm-act-rope", "ks_gemma_rmsnorm", [
        Provider("flashinfer", 1, 75, "fp16, bf16",
                 "from flashinfer.norm import gemma_rmsnorm",
                 None, "FlashInfer gemma_rmsnorm"),
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
        _ks_provider(_rope_ks, "ks_rope"),
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
        # Hopper/Blackwell: DeepGEMM grouped FP8 (DeepSeek-V3 production,
        # Mega-MoE) is the reference MoE GEMM — rank-1, gated sm90.
        Provider("deep_gemm", 1, 90, "fp8 e4m3, fp32 block scales",
                 "import deep_gemm",
                 _moe_deepgemm,
                 "DeepGEMM grouped FP8 (contiguous, Hopper/Blackwell)"),
        # sgl-kernel CUTLASS blockwise-fp8 grouped GEMM — the stated alignment
        # target (rank-2, sm90). Now calls the REAL sgl grouped mm (was a stub
        # that delegated to the ks kernel).
        _sgl_provider(2, _moe_sgl, min_sm=90, dtypes="fp8 e4m3",
                      note="SGLang CUTLASS grouped FP8 (fp8_blockwise_..._mm)"),
        # Ampere/Ada (no FP8 hw): vLLM Triton fused_experts (bf16/INT4 grouped).
        Provider("vllm", 3, 80, "bf16, fp16, fp8",
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
                 "import flashinfer",
                 None, "FlashInfer fused top-k/top-p sampling"),
        _sgl_provider(2, _sampling_sgl, dtypes="fp32 probs",
                      note="SGLang top_k_renorm_prob / top_p_renorm_prob"),
        _ks_provider(_sampling_ks, "ks_sample",
                     "fused temp/top-k/top-p sampler"),
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
