"""Token embedding lookup / scatter (``embedding.h``)."""

from __future__ import annotations

from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = ["embedding_lookup", "embedding_backward"]


def _infer_indices_i64(indices, indices_i64):
    """Infer the indices_i64 flag from a torch tensor when not given."""
    if indices_i64 is not None:
        return 1 if indices_i64 else 0
    try:
        import torch

        if isinstance(indices, torch.Tensor):
            return 1 if indices.dtype == torch.int64 else 0
    except Exception:
        pass
    return 0  # default to int32 indices


def embedding_lookup(
    out: TensorLike,
    table: TensorLike,
    indices: TensorLike,
    num_tokens: int,
    embed_dim: int,
    indices_i64: Optional[bool] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out[i, :] = table[indices[i], :]``.

    table: ``[vocab_size, embed_dim]``; indices: ``[num_tokens]`` int32 (or int64
    if ``indices_i64``); out: ``[num_tokens, embed_dim]``. The index dtype is
    inferred from a torch ``indices`` tensor when ``indices_i64`` is omitted.
    """
    dt = infer_dtype(table, dtype)
    check(
        lib.ks_embedding_lookup(
            ptr(out, name="out"), ptr(table, name="table"),
            ptr(indices, name="indices"),
            _infer_indices_i64(indices, indices_i64),
            int(num_tokens), int(embed_dim), dt, default_stream(stream, table),
        ),
        "ks_embedding_lookup",
    )
    return out


def embedding_backward(
    grad_table_fp32: TensorLike,
    grad_out: TensorLike,
    indices: TensorLike,
    num_tokens: int,
    embed_dim: int,
    indices_i64: Optional[bool] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``grad_table[indices[i], :] += grad_out[i, :]`` (atomic scatter-add).
    ``grad_table_fp32`` is fp32 ``[vocab_size, embed_dim]``. ``dtype`` is
    ``grad_out``'s dtype. Returns ``grad_table_fp32``."""
    dt = infer_dtype(grad_out, dtype)
    check(
        lib.ks_embedding_backward(
            ptr(grad_table_fp32, name="grad_table_fp32"),
            ptr(grad_out, name="grad_out"),
            ptr(indices, name="indices"),
            _infer_indices_i64(indices, indices_i64),
            int(num_tokens), int(embed_dim), dt,
            default_stream(stream, grad_out),
        ),
        "ks_embedding_backward",
    )
    return grad_table_fp32
