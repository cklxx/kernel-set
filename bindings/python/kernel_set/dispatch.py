"""Best-available-backend dispatch layer for kernel-set.

This realizes kernel-set's strategy — *prefer industry-best, don't reinvent*.
For each logical op, callers automatically get the strongest **installed**
industry kernel (flash-attn, FlashInfer, DeepGEMM, Marlin/vLLM, Liger, ...),
with kernel-set's own portable C-ABI kernel as the always-available fallback.

Quick start
-----------
::

    import kernel_set as ks

    # auto-routes to liger / flashinfer / vllm / ... or ks fallback
    out = ks.dispatch.rms_norm(x, w, eps=1e-6)
    o   = ks.dispatch.attention_prefill(q, k, v, causal=True)

    ks.dispatch.which("rmsnorm")          # -> 'flashinfer' (or 'kernel-set')
    ks.dispatch.which("fp8_gemm", gpu="h100", dtype="fp8")   # -> 'deep_gemm'
    ks.dispatch.available()               # -> {op: [selectable providers...]}

Selection
---------
For an op + (gpu arch, dtype) we walk the rank-ordered provider chain and pick
the first provider that is **selectable**:

* its library imports (cached probe), AND
* the device compute-capability meets the provider's ``min_sm`` gate
  (e.g. DeepGEMM / FlashMLA need sm90, NVFP4 sm100), AND
* its dtype-support string covers the requested dtype (when one is given).

Arch-/import-gated providers are skipped *silently*. The chain always ends at
the kernel-set C-ABI provider, which is treated as selectable everywhere, so
dispatch never dead-ends.

Import-safety & laziness
------------------------
This module imports cleanly with no torch, no CUDA, and even without the
kernel-set shared library built — the probes just report ``unavailable`` and
introspection (``which`` / ``available``) still works, reporting the kernel-set
fallback. Heavy provider libraries are imported only when an op is actually
dispatched (or its availability probed).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .backends import (
    KERNEL_SET,
    OP_ORDER,
    OPS,
    Provider,
    arch_ok,
    can_import,
    dtype_arch_ok,
    dtype_ok,
    optimal_order,
    resolve_sm,
)

__all__ = [
    # public op API (auto-routing)
    "attention_prefill",
    "attention_decode",
    "mla_decode",
    "gemm",
    "fp8_gemm",
    "fp8_gemm_blockwise",
    "int8_gemm",
    "w4a16",
    "w4a8",
    "w8a16_fp8",
    "sparse_2_4_gemm",
    "bitnet_gemm",
    "per_token_group_quant",
    "nvfp4_gemm",
    "mxfp4_gemm",
    "fp8_attention",
    "fp8_kv_cache",
    "rms_norm",
    "fused_add_rmsnorm",
    "gemma_rmsnorm",
    "rope",
    "swiglu",
    "cross_entropy",
    "fused_linear_ce",
    "moe",
    "moe_gate",
    "moe_group_gate",
    "sampling",
    "selective_scan",
    "causal_conv1d",
    "gated_delta_rule",
    "gated_linear_attn",
    "rwkv_wkv7",
    # introspection
    "which",
    "which_provider",
    "available",
    "providers",
    "ops",
    "chain",
    "Backend",
    "resolve_sm",
    "reset_cache",
]


# --------------------------------------------------------------------------- #
# Selection cache: keyed by (op, sm, dtype-token). Cleared by reset_cache().
# --------------------------------------------------------------------------- #
_SELECT_CACHE: Dict[tuple, Provider] = {}


def reset_cache() -> None:
    """Drop the cached selections (e.g. after installing a new provider)."""
    _SELECT_CACHE.clear()


def _is_selectable(p: Provider, sm: Optional[int], dtype) -> bool:
    """A provider is selectable if it imports, meets the arch gate, supports the
    dtype, AND the dtype is feasible on this SM at all. The kernel-set fallback
    is always selectable."""
    if p.name == KERNEL_SET:
        return True
    # Capability-aware dtype gate FIRST: even if the provider's min_sm is low and
    # its dtype string lists fp8/bf16/fp4, the hardware must actually support
    # that dtype (fp8>=sm89, bf16>=sm80, fp4>=sm100). This stops e.g. FlashInfer
    # (min_sm=75) being selected for fp8 on sm75.
    if not dtype_arch_ok(dtype, sm):
        return False
    if not arch_ok(p.min_sm, sm):
        return False
    if not dtype_ok(dtype, p.dtypes):
        return False
    if p.call is None:
        # Providers without a wired call adapter can be *named* by `which`
        # (chain introspection) but are not runtime-dispatchable.
        return False
    return can_import(p.import_check)


def select(op: str, gpu=None, dtype=None) -> Provider:
    """Return the highest-rank selectable :class:`Provider` for ``op`` given the
    GPU arch / dtype. Always returns at least the kernel-set fallback."""
    if op not in OPS:
        raise KeyError(
            f"unknown dispatch op {op!r}; known: {', '.join(OP_ORDER)}")
    sm = resolve_sm(gpu)
    from .backends import normalize_dtype
    key = (op, sm, normalize_dtype(dtype))
    cached = _SELECT_CACHE.get(key)
    if cached is not None:
        return cached
    chosen = None
    for p in optimal_order(op, sm, dtype):  # optimal-table order (ks last)
        if _is_selectable(p, sm, dtype):
            chosen = p
            break
    if chosen is None:  # pragma: no cover - ks fallback is always selectable
        chosen = next(p for p in OPS[op].providers if p.name == KERNEL_SET)
    _SELECT_CACHE[key] = chosen
    return chosen


# --------------------------------------------------------------------------- #
# Backend handle — a bound (op, gpu, dtype) selection you can call repeatedly.
# --------------------------------------------------------------------------- #
class Backend:
    """A resolved best-backend for one op. ``Backend("rmsnorm")(x, w)`` runs the
    strongest installed provider; ``.name`` reports which one was chosen."""

    __slots__ = ("op", "gpu", "dtype", "_provider")

    def __init__(self, op: str, gpu=None, dtype=None):
        self.op = op
        self.gpu = gpu
        self.dtype = dtype
        self._provider = select(op, gpu, dtype)

    @property
    def name(self) -> str:
        return self._provider.name

    @property
    def provider(self) -> Provider:
        return self._provider

    def __call__(self, *args, **kwargs):
        return self._provider.call(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Backend op={self.op!r} -> {self.name!r}>"


def _dispatch(op: str, args, kwargs):
    """Resolve and invoke the best provider for ``op``. ``gpu``/``dtype`` may be
    passed as kwargs to influence selection (and are consumed, not forwarded)."""
    gpu = kwargs.pop("_gpu", None)
    dtype = kwargs.pop("_dtype", None)
    return select(op, gpu, dtype).call(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Public auto-routing op API. Signatures mirror the kernel-set wrappers but
# return the output tensor (and auto-pick the backend).
# --------------------------------------------------------------------------- #
def attention_prefill(q, k, v, *, causal=True, softmax_scale=None, **kw):
    """Dense prefill attention. q/k/v: ``(batch, seqlen, heads, head_dim)``."""
    return _dispatch("attention_prefill", (q, k, v),
                     dict(causal=causal, softmax_scale=softmax_scale, **kw))


def attention_decode(q, k_cache, v_cache, block_tables, seq_lens, *,
                     block_size, max_blocks_per_seq, softmax_scale=None, **kw):
    """Paged KV-cache decode. q: ``(num_seqs, heads, head_dim)``."""
    return _dispatch(
        "attention_decode", (q, k_cache, v_cache, block_tables, seq_lens),
        dict(block_size=block_size, max_blocks_per_seq=max_blocks_per_seq,
             softmax_scale=softmax_scale, **kw))


def mla_decode(q_nope, q_pe, kv_cache, block_tables, seq_lens, *, heads, lora,
               rope_dim, block_size, max_blocks_per_seq, softmax_scale=None,
               **kw):
    """Absorbed-MLA paged decode (DeepSeek-style). Routes to SGLang FlashMLA on
    sm90, else the kernel-set MLA decode."""
    return _dispatch(
        "mla_decode", (q_nope, q_pe, kv_cache, block_tables, seq_lens),
        dict(heads=heads, lora=lora, rope_dim=rope_dim, block_size=block_size,
             max_blocks_per_seq=max_blocks_per_seq,
             softmax_scale=softmax_scale, **kw))


def gemm(a, b, **kw):
    """Dense GEMM ``a @ b``. a: ``(M,K)``, b: ``(K,N)``."""
    return _dispatch("gemm", (a, b), kw)


def fp8_gemm(a8, b8, a_scale, b_scale, *, out_dtype=None, **kw):
    """FP8 (blockwise / scaled) GEMM. a8/b8 fp8; per-tensor or block scales."""
    return _dispatch("fp8_gemm", (a8, b8, a_scale, b_scale),
                     dict(out_dtype=out_dtype, **kw))


def int8_gemm(a8, b8, a_scale, b_scale, *, out_dtype=None, **kw):
    """INT8 W8A8 scaled GEMM. a8/b8 int8; per-row/col scales. Routes to SGLang
    int8_scaled_mm, else the kernel-set w8a8 path."""
    return _dispatch("int8_gemm", (a8, b8, a_scale, b_scale),
                     dict(out_dtype=out_dtype, **kw))


def w4a16(a, b_packed, scales, zeros, *, group_size=128, **kw):
    """W4A16 GEMM: fp16/bf16 acts x packed int4 weights with group scales."""
    return _dispatch("w4a16", (a, b_packed, scales, zeros),
                     dict(group_size=group_size, **kw))


def w4a8(a8, b_packed, b_scales, a_scales=None, *, global_scale=None,
         group_size=None, out_dtype=None, **kw):
    """W4A8 GEMM: int8/fp8 acts x packed int4 weights with token/channel scales."""
    return _dispatch(
        "w4a8", (a8, b_packed, b_scales, a_scales),
        dict(global_scale=global_scale, group_size=group_size,
             out_dtype=out_dtype, **kw))


def w8a16_fp8(a, b_packed, b_scales, *, global_scale=None, out_dtype=None,
              **kw):
    """FP8 weight-only Marlin GEMM: fp16/bf16 acts x fp8-e4m3 weights.

    This is distinct from ``fp8_gemm`` (native FP8 tensor-core W8A8) and from
    ``int8_gemm``. It targets FP8 checkpoints on sm80/86/89 paths where the
    activation compute dtype remains fp16/bf16.
    """
    return _dispatch(
        "w8a16_fp8", (a, b_packed, b_scales),
        dict(global_scale=global_scale, out_dtype=out_dtype, **kw))


def sparse_2_4_gemm(a, bt_meta, bt_q, scale_a, scale_b, *, out_dtype=None,
                    bias=None, **kw):
    """2:4 structured sparse scaled GEMM over pre-compressed fp8/int8 weights."""
    return _dispatch(
        "sparse_2_4_gemm", (a, bt_meta, bt_q, scale_a, scale_b),
        dict(out_dtype=out_dtype, bias=bias, **kw))


def bitnet_gemm(a, b_ternary, scale=None, *, out_dtype=None, **kw):
    """BitNet W1.58A8 ternary BitLinear GEMM via BitBLAS when available."""
    return _dispatch(
        "bitnet_gemm", (a, b_ternary, scale),
        dict(out_dtype=out_dtype, **kw))


def fp8_gemm_blockwise(a8, b8, a_scale, b_scale, *, block_n=128, block_k=128,
                       out_dtype=None, **kw):
    """FP8 BLOCKWISE GEMM (DeepSeek-V3 recipe): 128x128 weight block / 1x128 act
    tile, two-level fp32 accumulation. Routes to DeepGEMM (Hopper/Blackwell);
    kernel-set's portable blockwise kernel is the sm80+ fallback."""
    return _dispatch("fp8_gemm_blockwise", (a8, b8, a_scale, b_scale),
                     dict(block_n=block_n, block_k=block_k,
                          out_dtype=out_dtype, **kw))


def per_token_group_quant(x, *, group_size=128, **kw):
    """Per-token-group (1x128) dynamic fp8 activation quant — the format the
    blockwise fp8 GEMM consumes. Returns ``(fp8_out, fp32_scale)``."""
    return _dispatch("per_token_group_quant", (x,),
                     dict(group_size=group_size, **kw))


def nvfp4_gemm(a4, b4, a_scale, b_scale, *, alpha=None, out_dtype=None, **kw):
    """NVFP4 GEMM (Blackwell e2m1 + e4m3 1x16 block scale + fp32 global). Routes
    to FlashInfer / vLLM cutlass_scaled_fp4_mm (sm100+ only)."""
    return _dispatch("nvfp4_gemm", (a4, b4, a_scale, b_scale),
                     dict(alpha=alpha, out_dtype=out_dtype, **kw))


def mxfp4_gemm(a4, b4, a_scale, b_scale, *, alpha=None, out_dtype=None, **kw):
    """MXFP4 GEMM (OCP microscaling e2m1 + E8M0 block-32 scale; gpt-oss). Routes
    to FlashInfer / vLLM Marlin-MXFP4 / torchao."""
    return _dispatch("mxfp4_gemm", (a4, b4, a_scale, b_scale),
                     dict(alpha=alpha, out_dtype=out_dtype, **kw))


def fp8_attention(q, k, v, *, causal=True, softmax_scale=None, **kw):
    """FP8 attention compute (fp8 QK^T/PV, fp32 softmax). Routes to
    SageAttention / FlashAttention-3 fp8 (no portable ks fp8 attention)."""
    return _dispatch("fp8_attention", (q, k, v),
                     dict(causal=causal, softmax_scale=softmax_scale, **kw))


def fp8_kv_cache(key, value, key_cache, value_cache, slot_mapping, *,
                 k_scale=None, v_scale=None, **kw):
    """FP8 KV-cache quantize-on-write (reshape_and_cache_flash fp8). Routes to
    vLLM (kernel-set's reshape_and_cache is dtype-preserving, no quant)."""
    return _dispatch("fp8_kv_cache",
                     (key, value, key_cache, value_cache, slot_mapping),
                     dict(k_scale=k_scale, v_scale=v_scale, **kw))


def rms_norm(x, w, *, eps=1e-6, **kw):
    """RMSNorm. x: ``(rows, hidden)``; w: ``(hidden,)``."""
    return _dispatch("rmsnorm", (x, w), dict(eps=eps, **kw))


def fused_add_rmsnorm(x, residual, w, *, eps=1e-6, **kw):
    """Fused add-RMSNorm. Returns ``(normed, new_residual)``."""
    return _dispatch("fused_add_rmsnorm", (x, residual, w), dict(eps=eps, **kw))


def gemma_rmsnorm(x, w, *, eps=1e-6, **kw):
    """Gemma-style RMSNorm: ``(x / RMS(x)) * (w + 1)``."""
    return _dispatch("gemma_rmsnorm", (x, w), dict(eps=eps, **kw))


def rope(q, k, cos, sin, *, interleaved=False, **kw):
    """Rotary embedding. Returns ``(q_rot, k_rot)``. NeoX / rotate_half."""
    return _dispatch("rope", (q, k, cos, sin),
                     dict(interleaved=interleaved, **kw))


def swiglu(gate, up, **kw):
    """SwiGLU (silu_and_mul): ``silu(gate) * up``."""
    return _dispatch("swiglu", (gate, up), kw)


def cross_entropy(logits, targets, *, ignore_index=-100, **kw):
    """Cross-entropy. Returns per-token loss (reduction='none')."""
    return _dispatch("cross_entropy", (logits, targets),
                     dict(ignore_index=ignore_index, **kw))


def fused_linear_ce(hidden, lm_head_weight, targets, *, bias=None,
                    ce_weight=None, ignore_index=-100,
                    label_smoothing=0.0, reduction="mean", **kw):
    """Fused LM-head linear + cross-entropy without materializing logits."""
    return _dispatch(
        "fused_linear_ce", (hidden, lm_head_weight, targets),
        dict(bias=bias, ce_weight=ce_weight, ignore_index=ignore_index,
             label_smoothing=label_smoothing, reduction=reduction, **kw))


def moe(*args, **kw):
    """Mixture-of-Experts. The strongest provider (vLLM fused_experts) takes
    ``(hidden, w1, w2, topk_weights, topk_ids)``; the kernel-set fallback takes
    the grouped-GEMM form ``(a, b, expert_offsets, num_experts=, n=, k=)``."""
    return _dispatch("moe", args, kw)


def moe_gate(gating_output, *, top_k, renormalize=False, **kw):
    """MoE softmax + top-k routing gate. SGLang ``topk_softmax`` is rank-1.
    Returns ``(topk_weights, topk_ids)``."""
    return _dispatch("moe_gate", (gating_output,),
                     dict(top_k=top_k, renormalize=renormalize, **kw))


def moe_group_gate(gating_output, bias, *, num_expert_group, topk_group, top_k,
                   **kw):
    """MoE sigmoid + group-limited top-k gate (DeepSeek-V3 style). SGLang
    ``moe_fused_gate`` is rank-1. Returns ``(topk_weights, topk_ids)``."""
    return _dispatch("moe_group_gate", (gating_output, bias),
                     dict(num_expert_group=num_expert_group,
                          topk_group=topk_group, top_k=top_k, **kw))


def sampling(probs, *, top_k=None, top_p=None, **kw):
    """Top-k / top-p sampling (logit processing). Routes to FlashInfer / SGLang
    renorm-by-threshold sampling or the kernel-set fused sampler."""
    return _dispatch("sampling", (probs,),
                     dict(top_k=top_k, top_p=top_p, **kw))


def selective_scan(out, x, dt, A, B, C, D=None, z=None, dt_bias=None, *,
                   delta_softplus=False, batch=None, dim=None, seqlen=None,
                   dstate=None, dtype=None, stream=None, **kw):
    """Mamba selective scan. Signature mirrors ``kernel_set.ssm.selective_scan``."""
    return _dispatch(
        "selective_scan", (out, x, dt, A, B, C),
        dict(D=D, z=z, dt_bias=dt_bias, delta_softplus=delta_softplus,
             batch=batch, dim=dim, seqlen=seqlen, dstate=dstate, dtype=dtype,
             stream=stream, **kw))


def causal_conv1d(out, x, weight, bias=None, *, batch=None, dim=None,
                  seqlen=None, width=None, silu=False, dtype=None, stream=None,
                  **kw):
    """Depthwise causal conv1d. Signature mirrors ``kernel_set.ssm.causal_conv1d``."""
    return _dispatch(
        "causal_conv1d", (out, x, weight),
        dict(bias=bias, batch=batch, dim=dim, seqlen=seqlen, width=width,
             silu=silu, dtype=dtype, stream=stream, **kw))


def gated_delta_rule(q, k, v, g, beta, *, batch=None, seqlen=None, heads=None,
                     k_dim=None, v_dim=None, g_is_vector=0,
                     use_qk_l2norm=0, scale=0.0, dtype=None, stream=None, **kw):
    """Gated delta rule / KDA linear attention. Inputs are ``[B, T, H, D]``."""
    return _dispatch(
        "gated_delta_rule", (q, k, v, g, beta),
        dict(batch=batch, seqlen=seqlen, heads=heads, k_dim=k_dim, v_dim=v_dim,
             g_is_vector=g_is_vector, use_qk_l2norm=use_qk_l2norm, scale=scale,
             dtype=dtype, stream=stream, **kw))


def gated_linear_attn(q, k, v, g=None, head_decay=None, *, batch=None,
                      seqlen=None, heads=None, k_dim=None, v_dim=None,
                      gate_mode=0, scale=0.0, dtype=None, stream=None, **kw):
    """GLA / simple GLA / lightning attention. Inputs are ``[B, T, H, D]``."""
    return _dispatch(
        "gated_linear_attn", (q, k, v, g, head_decay),
        dict(batch=batch, seqlen=seqlen, heads=heads, k_dim=k_dim, v_dim=v_dim,
             gate_mode=gate_mode, scale=scale, dtype=dtype, stream=stream, **kw))


def rwkv_wkv7(r, w, k, v, a, b, *, batch=None, seqlen=None, heads=None,
              k_dim=None, v_dim=None, scale=0.0, dtype=None, stream=None, **kw):
    """RWKV-7 WKV linear attention. Inputs are ``[B, T, H, D]``."""
    return _dispatch(
        "rwkv_wkv7", (r, w, k, v, a, b),
        dict(batch=batch, seqlen=seqlen, heads=heads, k_dim=k_dim, v_dim=v_dim,
             scale=scale, dtype=dtype, stream=stream, **kw))


# --------------------------------------------------------------------------- #
# Introspection.
# --------------------------------------------------------------------------- #
def which(op: str, gpu=None, dtype=None) -> str:
    """Provider name that WOULD be selected for ``op`` on this host (or for the
    named ``gpu`` / ``dtype``). E.g. ``'flashinfer'`` or ``'kernel-set'``."""
    return select(op, gpu, dtype).name


# Alias matching the spec's `which(op, gpu, dtype) -> provider name`.
which_provider = which


def available(gpu=None, dtype=None) -> "Dict[str, List[str]]":
    """``{op: [selectable provider names in rank order]}`` for this host (or the
    named ``gpu``/``dtype``). The kernel-set fallback always appears last."""
    sm = resolve_sm(gpu)
    out: Dict[str, List[str]] = {}
    for op in OP_ORDER:
        sel = [p.name for p in optimal_order(op, sm, dtype)
               if _is_selectable(p, sm, dtype)]
        out[op] = sel
    return out


def providers(op: str) -> "List[str]":
    """Full rank-ordered provider chain for ``op`` (regardless of availability),
    ending with ``'kernel-set'``."""
    if op not in OPS:
        raise KeyError(
            f"unknown dispatch op {op!r}; known: {', '.join(OP_ORDER)}")
    return [p.name for p in OPS[op].providers]


def ops() -> "List[str]":
    """All logical ops the dispatcher knows about, in display order."""
    return list(OP_ORDER)


def chain(op: str, gpu=None, dtype=None) -> "List[dict]":
    """Detailed per-provider view for ``op``: name, rank, min_sm, dtypes, and
    whether each provider is selectable here (used by ``ksctl backends``)."""
    if op not in OPS:
        raise KeyError(f"unknown dispatch op {op!r}")
    sm = resolve_sm(gpu)
    rows = []
    for p in optimal_order(op, sm, dtype):
        rows.append({
            "name": p.name,
            "rank": p.rank,
            "min_sm": p.min_sm,
            "dtypes": p.dtypes,
            "note": p.note,
            "selectable": _is_selectable(p, sm, dtype),
            "is_kernel_set": p.name == KERNEL_SET,
        })
    return rows
