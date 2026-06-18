"""Fused attention kernels (``attention.h``).

FlashAttention-style prefill (dense + varlen), paged KV-cache decode,
reshape-and-cache, DeepSeek MLA decode, and FlashAttention backward.

Conventions (see attention.h):
  * ``head_dim`` is contiguous (innermost).
  * GQA/MQA: ``num_kv_heads`` may be < ``num_heads``.
  * ``softmax_scale <= 0`` => ``1/sqrt(head_dim)``.
  * int32 index tensors (cu_seqlens, block_tables, seq_lens, slot_mapping) are
    passed as ``POINTER(c_int32)`` — torch tensors must be ``int32`` & CUDA.
"""

from __future__ import annotations

from ctypes import POINTER, c_float, c_int32, cast
from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = [
    "flash_attn",
    "flash_attn_varlen",
    "paged_attn_decode",
    "paged_attn_decode_split_k",
    "reshape_and_cache",
    "mla_decode",
    "mla_decode_split_k",
    "attention_state_merge",
    "dsa_topk_select",
    "flash_attn_backward",
]

_I32P = POINTER(c_int32)
_F32P = POINTER(c_float)


def _f32(obj: TensorLike, *, name: str):
    """Resolve an fp32 tensor/pointer to a ``POINTER(c_float)`` (required)."""
    if obj is None:
        raise ValueError(f"{name} must not be None")
    return cast(ptr(obj, name=name, allow_none=False), _F32P)


def _i32(obj: TensorLike, *, name: str):
    """Resolve an int32 index tensor/pointer to a ``POINTER(c_int32)``.

    ``None`` is forbidden (these indices are required by the kernels that take
    them). torch tensors must already be int32.
    """
    if obj is None:
        raise ValueError(f"{name} must not be None")
    return cast(ptr(obj, name=name, allow_none=False), _I32P)


def flash_attn(
    out: TensorLike,
    q: TensorLike,
    k: TensorLike,
    v: TensorLike,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_lse: TensorLike = None,
    softmax_scale: float = 0.0,
    causal: bool = False,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Dense prefill (FlashAttention-2 forward) for a uniform ``[batch, seqlen]``.

    q: ``[batch, seqlen_q, num_heads, head_dim]``;
    k,v: ``[batch, seqlen_k, num_kv_heads, head_dim]``;
    ``softmax_lse``: ``[num_heads, batch*seqlen_q]`` fp32 (NULL if not training).
    Returns ``out``.
    """
    dt = infer_dtype(q, dtype)
    check(
        lib.ks_flash_attn(
            ptr(out, name="out"),
            ptr(softmax_lse, name="softmax_lse"),
            ptr(q, name="q"),
            ptr(k, name="k"),
            ptr(v, name="v"),
            int(batch), int(seqlen_q), int(seqlen_k),
            int(num_heads), int(num_kv_heads), int(head_dim),
            float(softmax_scale), 1 if causal else 0, dt,
            default_stream(stream, q),
        ),
        "ks_flash_attn",
    )
    return out


def flash_attn_varlen(
    out: TensorLike,
    q: TensorLike,
    k: TensorLike,
    v: TensorLike,
    cu_seqlens_q: TensorLike,
    cu_seqlens_k: TensorLike,
    batch: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_lse: TensorLike = None,
    softmax_scale: float = 0.0,
    causal: bool = False,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Variable-length packed prefill.

    q: ``[total_q, num_heads, head_dim]``; k,v: ``[total_kv, num_kv_heads,
    head_dim]``; ``cu_seqlens_*``: ``[batch+1]`` int32 prefix sums. Returns
    ``out``.
    """
    dt = infer_dtype(q, dtype)
    check(
        lib.ks_flash_attn_varlen(
            ptr(out, name="out"),
            ptr(softmax_lse, name="softmax_lse"),
            ptr(q, name="q"),
            ptr(k, name="k"),
            ptr(v, name="v"),
            _i32(cu_seqlens_q, name="cu_seqlens_q"),
            _i32(cu_seqlens_k, name="cu_seqlens_k"),
            int(batch), int(max_seqlen_q), int(max_seqlen_k),
            int(num_heads), int(num_kv_heads), int(head_dim),
            float(softmax_scale), 1 if causal else 0, dt,
            default_stream(stream, q),
        ),
        "ks_flash_attn_varlen",
    )
    return out


def paged_attn_decode(
    out: TensorLike,
    q: TensorLike,
    k_cache: TensorLike,
    v_cache: TensorLike,
    block_tables: TensorLike,
    seq_lens: TensorLike,
    num_seqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
    softmax_scale: float = 0.0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Paged KV-cache decode (FlashDecoding), one query position per sequence.

    q: ``[num_seqs, num_heads, head_dim]``; k_cache/v_cache: ``[num_blocks,
    num_kv_heads, block_size, head_dim]``; block_tables: ``[num_seqs,
    max_blocks_per_seq]`` int32; seq_lens: ``[num_seqs]`` int32. Returns ``out``.
    """
    dt = infer_dtype(q, dtype)
    check(
        lib.ks_paged_attn_decode(
            ptr(out, name="out"),
            ptr(q, name="q"),
            ptr(k_cache, name="k_cache"),
            ptr(v_cache, name="v_cache"),
            _i32(block_tables, name="block_tables"),
            _i32(seq_lens, name="seq_lens"),
            int(num_seqs), int(num_heads), int(num_kv_heads), int(head_dim),
            int(block_size), int(max_blocks_per_seq),
            float(softmax_scale), dt, default_stream(stream, q),
        ),
        "ks_paged_attn_decode",
    )
    return out


def _torch_empty_like_shape(like: TensorLike, shape, *, dtype=None, name: str):
    if not hasattr(like, "device") or not hasattr(like, "dtype"):
        raise ValueError(f"{name} is required for raw-pointer inputs")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional torch
        raise ValueError(f"{name} is required when torch is unavailable") from exc
    return torch.empty(tuple(int(s) for s in shape), device=like.device,
                       dtype=dtype or like.dtype)


def paged_attn_decode_split_k(
    out: TensorLike,
    q: TensorLike,
    k_cache: TensorLike,
    v_cache: TensorLike,
    block_tables: TensorLike,
    seq_lens: TensorLike,
    num_seqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
    num_splits: int,
    partial_out: TensorLike = None,
    partial_lse: TensorLike = None,
    softmax_scale: float = 0.0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Split-K paged KV-cache decode for long-context decode.

    ``partial_out`` is ``[num_splits, num_seqs, num_heads, head_dim]`` in model
    dtype and ``partial_lse`` is ``[num_splits, num_seqs, num_heads]`` fp32. For
    torch tensors these workspaces are allocated automatically when omitted.
    """
    if int(num_splits) <= 1:
        return paged_attn_decode(
            out, q, k_cache, v_cache, block_tables, seq_lens, num_seqs,
            num_heads, num_kv_heads, head_dim, block_size, max_blocks_per_seq,
            softmax_scale=softmax_scale, dtype=dtype, stream=stream)
    if partial_out is None:
        partial_out = _torch_empty_like_shape(
            q, (num_splits, num_seqs, num_heads, head_dim),
            name="partial_out")
    if partial_lse is None:
        try:
            import torch
            partial_lse = _torch_empty_like_shape(
                q, (num_splits, num_seqs, num_heads), dtype=torch.float32,
                name="partial_lse")
        except ValueError:
            raise
    dt = infer_dtype(q, dtype)
    check(
        lib.ks_paged_attn_decode_split_k(
            ptr(out, name="out"),
            ptr(partial_out, name="partial_out"),
            _f32(partial_lse, name="partial_lse"),
            ptr(q, name="q"),
            ptr(k_cache, name="k_cache"),
            ptr(v_cache, name="v_cache"),
            _i32(block_tables, name="block_tables"),
            _i32(seq_lens, name="seq_lens"),
            int(num_seqs), int(num_heads), int(num_kv_heads), int(head_dim),
            int(block_size), int(max_blocks_per_seq), int(num_splits),
            float(softmax_scale), dt, default_stream(stream, q),
        ),
        "ks_paged_attn_decode_split_k",
    )
    return out


def reshape_and_cache(
    k_cache: TensorLike,
    v_cache: TensorLike,
    key: TensorLike,
    value: TensorLike,
    slot_mapping: TensorLike,
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """Write new K/V into a paged cache at the given slots.

    key/value: ``[num_tokens, num_kv_heads, head_dim]``; slot_mapping:
    ``[num_tokens]`` int32 (flat slot = block_id*block_size + offset). Returns
    ``(k_cache, v_cache)``.
    """
    dt = infer_dtype(key, dtype)
    check(
        lib.ks_reshape_and_cache(
            ptr(k_cache, name="k_cache"),
            ptr(v_cache, name="v_cache"),
            ptr(key, name="key"),
            ptr(value, name="value"),
            _i32(slot_mapping, name="slot_mapping"),
            int(num_tokens), int(num_kv_heads), int(head_dim), int(block_size),
            dt, default_stream(stream, key),
        ),
        "ks_reshape_and_cache",
    )
    return k_cache, v_cache


def mla_decode(
    out: TensorLike,
    q_nope: TensorLike,
    q_pe: TensorLike,
    kv_cache: TensorLike,
    block_tables: TensorLike,
    seq_lens: TensorLike,
    num_seqs: int,
    num_heads: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
    softmax_scale: float = 0.0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """DeepSeek Multi-head Latent Attention decode over a compressed KV cache.

    q_nope: ``[num_seqs, num_heads, kv_lora_rank]``; q_pe: ``[num_seqs,
    num_heads, rope_dim]``; kv_cache: ``[num_blocks, block_size, kv_lora_rank +
    rope_dim]``. Returns ``out``.
    """
    dt = infer_dtype(q_nope, dtype)
    check(
        lib.ks_mla_decode(
            ptr(out, name="out"),
            ptr(q_nope, name="q_nope"),
            ptr(q_pe, name="q_pe"),
            ptr(kv_cache, name="kv_cache"),
            _i32(block_tables, name="block_tables"),
            _i32(seq_lens, name="seq_lens"),
            int(num_seqs), int(num_heads), int(kv_lora_rank), int(rope_dim),
            int(block_size), int(max_blocks_per_seq),
            float(softmax_scale), dt, default_stream(stream, q_nope),
        ),
        "ks_mla_decode",
    )
    return out


def mla_decode_split_k(
    out: TensorLike,
    q_nope: TensorLike,
    q_pe: TensorLike,
    kv_cache: TensorLike,
    block_tables: TensorLike,
    seq_lens: TensorLike,
    num_seqs: int,
    num_heads: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
    num_splits: int,
    partial_out: TensorLike = None,
    partial_lse: TensorLike = None,
    softmax_scale: float = 0.0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Split-K DeepSeek MLA decode.

    ``partial_out`` is ``[num_splits, num_seqs, num_heads, kv_lora_rank]`` in
    model dtype and ``partial_lse`` is ``[num_splits, num_seqs, num_heads]`` fp32.
    For torch tensors these workspaces are allocated automatically when omitted.
    """
    if int(num_splits) <= 1:
        return mla_decode(
            out, q_nope, q_pe, kv_cache, block_tables, seq_lens, num_seqs,
            num_heads, kv_lora_rank, rope_dim, block_size, max_blocks_per_seq,
            softmax_scale=softmax_scale, dtype=dtype, stream=stream)
    if partial_out is None:
        partial_out = _torch_empty_like_shape(
            q_nope, (num_splits, num_seqs, num_heads, kv_lora_rank),
            name="partial_out")
    if partial_lse is None:
        try:
            import torch
            partial_lse = _torch_empty_like_shape(
                q_nope, (num_splits, num_seqs, num_heads), dtype=torch.float32,
                name="partial_lse")
        except ValueError:
            raise
    dt = infer_dtype(q_nope, dtype)
    check(
        lib.ks_mla_decode_split_k(
            ptr(out, name="out"),
            ptr(partial_out, name="partial_out"),
            _f32(partial_lse, name="partial_lse"),
            ptr(q_nope, name="q_nope"),
            ptr(q_pe, name="q_pe"),
            ptr(kv_cache, name="kv_cache"),
            _i32(block_tables, name="block_tables"),
            _i32(seq_lens, name="seq_lens"),
            int(num_seqs), int(num_heads), int(kv_lora_rank), int(rope_dim),
            int(block_size), int(max_blocks_per_seq), int(num_splits),
            float(softmax_scale), dt, default_stream(stream, q_nope),
        ),
        "ks_mla_decode_split_k",
    )
    return out


def attention_state_merge(
    out: TensorLike,
    lse: TensorLike,
    out_a: TensorLike,
    lse_a: TensorLike,
    out_b: TensorLike,
    lse_b: TensorLike,
    n_rows: Optional[int] = None,
    v_dim: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Merge two partial attention states by log-sum-exp (cascade / chunked /
    ring attention): ``out = softmax-weighted combine of (out_a,lse_a),(out_b,
    lse_b)``; ``lse = logaddexp(lse_a, lse_b)``. ``out*`` are ``[n_rows, v_dim]``
    (model dtype); ``lse*`` are ``[n_rows]`` fp32. ``out`` may alias ``out_a``.
    Returns ``out``.
    """
    if n_rows is None or v_dim is None:
        try:
            if hasattr(out_a, "shape") and len(out_a.shape) >= 2:
                r = 1
                for s in out_a.shape[:-1]:
                    r *= s
                n_rows, v_dim = int(r), int(out_a.shape[-1])
        except Exception:
            pass
    if n_rows is None or v_dim is None:
        raise ValueError("n_rows and v_dim are required for raw-pointer inputs")
    dt = infer_dtype(out_a, dtype)
    check(
        lib.ks_attention_state_merge(
            ptr(out, name="out"),
            _f32(lse, name="lse"),
            ptr(out_a, name="out_a"),
            _f32(lse_a, name="lse_a"),
            ptr(out_b, name="out_b"),
            _f32(lse_b, name="lse_b"),
            int(n_rows), int(v_dim), dt, default_stream(stream, out_a),
        ),
        "ks_attention_state_merge",
    )
    return out


def dsa_topk_select(
    indices: TensorLike,
    scores: TensorLike,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    topk: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Row-wise top-k KV selection (DeepSeek sparse attention). ``scores`` is
    ``[n_rows, n_cols]`` (model dtype); ``indices`` is ``[n_rows, topk]`` int32,
    largest-score first, ``-1`` padded. Returns ``indices``.
    """
    if n_rows is None or n_cols is None:
        try:
            if hasattr(scores, "shape") and len(scores.shape) >= 2:
                r = 1
                for s in scores.shape[:-1]:
                    r *= s
                n_rows, n_cols = int(r), int(scores.shape[-1])
        except Exception:
            pass
    if topk is None and hasattr(indices, "shape"):
        topk = int(indices.shape[-1])
    if n_rows is None or n_cols is None or topk is None:
        raise ValueError("n_rows, n_cols and topk are required for raw-pointer inputs")
    dt = infer_dtype(scores, dtype)
    check(
        lib.ks_dsa_topk_select(
            _i32(indices, name="indices"),
            ptr(scores, name="scores"),
            int(n_rows), int(n_cols), int(topk), dt,
            default_stream(stream, scores),
        ),
        "ks_dsa_topk_select",
    )
    return indices


def flash_attn_backward(
    grad_q: TensorLike,
    grad_k: TensorLike,
    grad_v: TensorLike,
    grad_out: TensorLike,
    q: TensorLike,
    k: TensorLike,
    v: TensorLike,
    out: TensorLike,
    softmax_lse: TensorLike,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_scale: float = 0.0,
    causal: bool = False,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """FlashAttention backward (training). Requires the forward ``out`` and
    ``softmax_lse``. Returns ``(grad_q, grad_k, grad_v)``."""
    dt = infer_dtype(q, dtype)
    check(
        lib.ks_flash_attn_backward(
            ptr(grad_q, name="grad_q"),
            ptr(grad_k, name="grad_k"),
            ptr(grad_v, name="grad_v"),
            ptr(grad_out, name="grad_out"),
            ptr(q, name="q"),
            ptr(k, name="k"),
            ptr(v, name="v"),
            ptr(out, name="out"),
            ptr(softmax_lse, name="softmax_lse"),
            int(batch), int(seqlen_q), int(seqlen_k),
            int(num_heads), int(num_kv_heads), int(head_dim),
            float(softmax_scale), 1 if causal else 0, dt,
            default_stream(stream, q),
        ),
        "ks_flash_attn_backward",
    )
    return grad_q, grad_k, grad_v
