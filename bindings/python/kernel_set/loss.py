"""Training loss kernels (``loss.h``): memory-efficient cross-entropy."""

from __future__ import annotations

from ctypes import POINTER, c_float, cast
from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = ["cross_entropy", "fused_linear_cross_entropy"]

_F32P = POINTER(c_float)


def _f32(obj, *, name, allow_none=False):
    raw = ptr(obj, name=name, allow_none=allow_none)
    return cast(raw, _F32P) if raw else _F32P()


def _infer_targets_i64(targets, targets_i64):
    if targets_i64 is not None:
        return 1 if targets_i64 else 0
    try:
        import torch

        if isinstance(targets, torch.Tensor):
            return 1 if targets.dtype == torch.int64 else 0
    except Exception:
        pass
    return 0


def cross_entropy(
    losses: TensorLike,
    grad_logits: TensorLike,
    logits: TensorLike,
    targets: TensorLike,
    num_tokens: int,
    vocab: int,
    targets_i64: Optional[bool] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """Fused cross-entropy forward+backward in one pass.

    logits: ``[num_tokens, vocab]`` (may alias ``grad_logits`` for in-place
    grad); targets: ``[num_tokens]`` class ids; losses: fp32 ``[num_tokens]``.
    Returns ``(losses, grad_logits)``.
    """
    dt = infer_dtype(logits, dtype)
    check(
        lib.ks_cross_entropy(
            _f32(losses, name="losses"),
            ptr(grad_logits, name="grad_logits"),
            ptr(logits, name="logits"),
            ptr(targets, name="targets"),
            _infer_targets_i64(targets, targets_i64),
            int(num_tokens), int(vocab), int(ignore_index),
            float(label_smoothing), dt, default_stream(stream, logits),
        ),
        "ks_cross_entropy",
    )
    return losses, grad_logits


def fused_linear_cross_entropy(
    losses: TensorLike,
    grad_hidden: TensorLike,
    grad_weight_fp32: TensorLike,
    hidden: TensorLike,
    weight: TensorLike,
    targets: TensorLike,
    num_tokens: int,
    hidden_dim: int,
    vocab: int,
    targets_i64: Optional[bool] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    chunk_size: int = 0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """Fused-linear-cross-entropy: computes CE from hidden states and the LM head
    weight in chunks WITHOUT materializing ``[num_tokens, vocab]`` logits, and
    writes ``grad_hidden`` / ``grad_weight_fp32``.

    hidden: ``[num_tokens, hidden_dim]``; weight: ``[vocab, hidden_dim]``;
    grad_hidden: ``[num_tokens, hidden_dim]``; grad_weight_fp32: ``[vocab,
    hidden_dim]``. Returns ``(losses, grad_hidden, grad_weight_fp32)``.
    """
    dt = infer_dtype(hidden, dtype)
    check(
        lib.ks_fused_linear_cross_entropy(
            _f32(losses, name="losses"),
            ptr(grad_hidden, name="grad_hidden"),
            ptr(grad_weight_fp32, name="grad_weight_fp32"),
            ptr(hidden, name="hidden"),
            ptr(weight, name="weight"),
            ptr(targets, name="targets"),
            _infer_targets_i64(targets, targets_i64),
            int(num_tokens), int(hidden_dim), int(vocab), int(ignore_index),
            float(label_smoothing), int(chunk_size), dt,
            default_stream(stream, hidden),
        ),
        "ks_fused_linear_cross_entropy",
    )
    return losses, grad_hidden, grad_weight_fp32
