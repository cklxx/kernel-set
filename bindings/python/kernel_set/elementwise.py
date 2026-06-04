"""Elementwise & casting kernels (``elementwise.h``).

All ops run over ``n`` contiguous elements. ``n`` is inferred from a torch
tensor's ``numel()`` when omitted.
"""

from __future__ import annotations

from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = ["add", "mul", "add_residual", "scale", "cast", "axpby"]


def _numel(reference, n):
    if n is not None:
        return int(n)
    if hasattr(reference, "numel"):
        return int(reference.numel())
    raise ValueError("n (element count) is required for raw-pointer inputs")


def add(
    out: TensorLike,
    a: TensorLike,
    b: TensorLike,
    n: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = a + b`` over ``n`` elements."""
    n = _numel(a, n)
    dt = infer_dtype(a, dtype)
    check(
        lib.ks_add(ptr(out, name="out"), ptr(a, name="a"), ptr(b, name="b"), n,
                   dt, default_stream(stream, a)),
        "ks_add",
    )
    return out


def mul(
    out: TensorLike,
    a: TensorLike,
    b: TensorLike,
    n: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = a * b`` (Hadamard) over ``n`` elements."""
    n = _numel(a, n)
    dt = infer_dtype(a, dtype)
    check(
        lib.ks_mul(ptr(out, name="out"), ptr(a, name="a"), ptr(b, name="b"), n,
                   dt, default_stream(stream, a)),
        "ks_mul",
    )
    return out


def add_residual(
    residual: TensorLike,
    x: TensorLike,
    n: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """In-place residual accumulation: ``residual <- residual + x``."""
    n = _numel(residual, n)
    dt = infer_dtype(residual, dtype)
    check(
        lib.ks_add_residual(ptr(residual, name="residual"), ptr(x, name="x"), n,
                            dt, default_stream(stream, residual)),
        "ks_add_residual",
    )
    return residual


def scale(
    out: TensorLike,
    x: TensorLike,
    scale: float,
    n: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = x * scale`` (scalar) over ``n`` elements."""
    n = _numel(x, n)
    dt = infer_dtype(x, dtype)
    check(
        lib.ks_scale(ptr(out, name="out"), ptr(x, name="x"), float(scale), n,
                     dt, default_stream(stream, x)),
        "ks_scale",
    )
    return out


def cast(
    out: TensorLike,
    dst_dtype: int,
    inp: TensorLike,
    src_dtype: Optional[int] = None,
    n: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out[i] = cast<dst>(in[i])``. ``dst_dtype`` is required; ``src_dtype`` is
    inferred from a torch ``inp`` tensor when omitted."""
    n = _numel(inp, n)
    src = infer_dtype(inp, src_dtype)
    check(
        lib.ks_cast(ptr(out, name="out"), int(dst_dtype), ptr(inp, name="inp"),
                    src, n, default_stream(stream, inp)),
        "ks_cast",
    )
    return out


def axpby(
    out: TensorLike,
    a: TensorLike,
    alpha: float,
    b: TensorLike,
    beta: float,
    n: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = a*alpha + b*beta`` (fused AXPBY) over ``n`` elements."""
    n = _numel(a, n)
    dt = infer_dtype(a, dtype)
    check(
        lib.ks_axpby(ptr(out, name="out"), ptr(a, name="a"), float(alpha),
                     ptr(b, name="b"), float(beta), n, dt,
                     default_stream(stream, a)),
        "ks_axpby",
    )
    return out
