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

from ctypes import POINTER, c_int32, cast
from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = [
    "flash_attn",
    "flash_attn_varlen",
    "paged_attn_decode",
    "reshape_and_cache",
    "mla_decode",
    "flash_attn_backward",
]

_I32P = POINTER(c_int32)


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
