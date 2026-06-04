"""Rotary position embeddings (``rope.h``).

Tensors are ``[num_tokens, num_heads, head_dim]`` (contiguous over head_dim).
``cos``/``sin`` are ``[num_tokens, head_dim/2]`` (or caches ``[max_pos,
head_dim/2]`` gathered by ``positions``). ``interleaved=True`` selects the GPT-J
convention; ``False`` is NeoX / rotate_half.
"""

from __future__ import annotations

from ctypes import POINTER, c_int32, cast
from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = ["rope_inplace", "rope", "rope_gather", "rope_backward"]

_I32P = POINTER(c_int32)


def rope_inplace(
    q: TensorLike,
    k: TensorLike,
    cos: TensorLike,
    sin: TensorLike,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    interleaved: bool = False,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """In-place RoPE applied to ``q`` and ``k`` (``k`` may be ``None`` to skip).
    Returns ``(q, k)``."""
    dt = infer_dtype(q, dtype)
    check(
        lib.ks_rope_inplace(
            ptr(q, name="q"), ptr(k, name="k"),
            ptr(cos, name="cos"), ptr(sin, name="sin"),
            int(num_tokens), int(num_q_heads), int(num_kv_heads), int(head_dim),
            1 if interleaved else 0, dt, default_stream(stream, q),
        ),
        "ks_rope_inplace",
    )
    return q, k


def rope(
    q_out: TensorLike,
    k_out: TensorLike,
    q: TensorLike,
    k: TensorLike,
    cos: TensorLike,
    sin: TensorLike,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    interleaved: bool = False,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """Out-of-place RoPE writing to ``q_out``/``k_out``. Returns ``(q_out,
    k_out)``."""
    dt = infer_dtype(q, dtype)
    check(
        lib.ks_rope(
            ptr(q_out, name="q_out"), ptr(k_out, name="k_out"),
            ptr(q, name="q"), ptr(k, name="k"),
            ptr(cos, name="cos"), ptr(sin, name="sin"),
            int(num_tokens), int(num_q_heads), int(num_kv_heads), int(head_dim),
            1 if interleaved else 0, dt, default_stream(stream, q),
        ),
        "ks_rope",
    )
    return q_out, k_out


def rope_gather(
    q: TensorLike,
    k: TensorLike,
    cos_cache: TensorLike,
    sin_cache: TensorLike,
    positions: TensorLike,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    interleaved: bool = False,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """In-place RoPE with gathered cos/sin caches ``[max_pos, head_dim/2]``
    indexed by ``positions`` (int32 ``[num_tokens]``). Returns ``(q, k)``."""
    dt = infer_dtype(q, dtype)
    pos = cast(ptr(positions, name="positions", allow_none=False), _I32P)
    check(
        lib.ks_rope_gather(
            ptr(q, name="q"), ptr(k, name="k"),
            ptr(cos_cache, name="cos_cache"), ptr(sin_cache, name="sin_cache"),
            pos,
            int(num_tokens), int(num_q_heads), int(num_kv_heads), int(head_dim),
            1 if interleaved else 0, dt, default_stream(stream, q),
        ),
        "ks_rope_gather",
    )
    return q, k


def rope_backward(
    grad_q: TensorLike,
    grad_k: TensorLike,
    cos: TensorLike,
    sin: TensorLike,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    interleaved: bool = False,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """RoPE backward (rotate gradients by the conjugate rotation). Returns
    ``(grad_q, grad_k)``."""
    dt = infer_dtype(grad_q, dtype)
    check(
        lib.ks_rope_backward(
            ptr(grad_q, name="grad_q"), ptr(grad_k, name="grad_k"),
            ptr(cos, name="cos"), ptr(sin, name="sin"),
            int(num_tokens), int(num_q_heads), int(num_kv_heads), int(head_dim),
            1 if interleaved else 0, dt, default_stream(stream, grad_q),
        ),
        "ks_rope_backward",
    )
    return grad_q, grad_k
