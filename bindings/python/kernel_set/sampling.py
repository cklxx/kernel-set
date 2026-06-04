"""Softmax and token sampling for autoregressive decoding (``sampling.h``).

logits are ``[num_seqs, vocab_size]``. Sampling kernels select one token per
sequence into ``out_tokens`` ``[num_seqs]`` int32. Randomness uses a
counter-based RNG seeded by ``(seed, philox_offset)`` for reproducibility.
"""

from __future__ import annotations

from ctypes import POINTER, c_float, c_int32, cast
from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = ["softmax", "log_softmax", "argmax", "sample"]

_I32P = POINTER(c_int32)
_F32P = POINTER(c_float)


def _i32(obj, *, name, allow_none=False):
    raw = ptr(obj, name=name, allow_none=allow_none)
    return cast(raw, _I32P) if raw else _I32P()


def _f32(obj, *, name, allow_none=False):
    raw = ptr(obj, name=name, allow_none=allow_none)
    return cast(raw, _F32P) if raw else _F32P()


def softmax(
    out: TensorLike,
    input: TensorLike,
    rows: int,
    cols: int,
    temperature: float = 1.0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = softmax(input / temperature)`` along the last dim
    (``temperature <= 0`` => 1)."""
    dt = infer_dtype(input, dtype)
    check(
        lib.ks_softmax(
            ptr(out, name="out"), ptr(input, name="input"),
            int(rows), int(cols), float(temperature), dt,
            default_stream(stream, input),
        ),
        "ks_softmax",
    )
    return out


def log_softmax(
    out: TensorLike,
    input: TensorLike,
    rows: int,
    cols: int,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = log_softmax(input)`` along the last dim."""
    dt = infer_dtype(input, dtype)
    check(
        lib.ks_log_softmax(
            ptr(out, name="out"), ptr(input, name="input"),
            int(rows), int(cols), dt, default_stream(stream, input),
        ),
        "ks_log_softmax",
    )
    return out


def argmax(
    out_tokens: TensorLike,
    logits: TensorLike,
    num_seqs: int,
    vocab_size: int,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Greedy: ``out_tokens[s] = argmax_v logits[s, v]`` (int32)."""
    dt = infer_dtype(logits, dtype)
    check(
        lib.ks_argmax(
            _i32(out_tokens, name="out_tokens"), ptr(logits, name="logits"),
            int(num_seqs), int(vocab_size), dt, default_stream(stream, logits),
        ),
        "ks_argmax",
    )
    return out_tokens


def sample(
    out_tokens: TensorLike,
    logits: TensorLike,
    num_seqs: int,
    vocab_size: int,
    out_probs: TensorLike = None,
    temperatures: TensorLike = None,
    top_ks: TensorLike = None,
    top_ps: TensorLike = None,
    seed: int = 0,
    philox_offset: int = 0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """Combined temperature + top-k + top-p sampling.

    ``temperatures``/``top_ps`` are fp32 ``[num_seqs]`` (or ``None``); ``top_ks``
    is int32 ``[num_seqs]`` (or ``None``); ``out_probs`` is fp32 ``[num_seqs]``
    (or ``None``). Returns ``(out_tokens, out_probs)``.
    """
    dt = infer_dtype(logits, dtype)
    check(
        lib.ks_sample(
            _i32(out_tokens, name="out_tokens"),
            _f32(out_probs, name="out_probs", allow_none=True),
            ptr(logits, name="logits"),
            _f32(temperatures, name="temperatures", allow_none=True),
            _i32(top_ks, name="top_ks", allow_none=True),
            _f32(top_ps, name="top_ps", allow_none=True),
            int(num_seqs), int(vocab_size),
            int(seed), int(philox_offset), dt, default_stream(stream, logits),
        ),
        "ks_sample",
    )
    return out_tokens, out_probs
