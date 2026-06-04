"""Activation kernels and gated-MLP fusions (``activation.h``).

Elementwise ops run over ``n`` contiguous elements. Gated variants implement
the SwiGLU / GeGLU pattern used by Llama-family MLPs.
"""

from __future__ import annotations

from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = [
    "silu",
    "gelu",
    "relu",
    "swiglu",
    "swiglu_packed",
    "geglu",
    "swiglu_backward",
]


def _numel(reference, n):
    if n is not None:
        return int(n)
    if hasattr(reference, "numel"):
        return int(reference.numel())
    raise ValueError("n (element count) is required for raw-pointer inputs")


def _rows_inter(reference, rows, inter):
    """Infer (rows, inter) from a 2-D torch tensor when not given."""
    if rows is not None and inter is not None:
        return int(rows), int(inter)
    if hasattr(reference, "shape") and len(reference.shape) >= 2:
        r = 1
        for s in reference.shape[:-1]:
            r *= s
        return int(r), int(reference.shape[-1])
    raise ValueError("rows and inter are required for raw-pointer inputs")


def silu(
    out: TensorLike,
    input: TensorLike,
    n: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = silu(x) = x * sigmoid(x)`` over ``n`` elements."""
    n = _numel(input, n)
    dt = infer_dtype(input, dtype)
    check(
        lib.ks_silu(ptr(out, name="out"), ptr(input, name="input"), n, dt,
                    default_stream(stream, input)),
        "ks_silu",
    )
    return out


def gelu(
    out: TensorLike,
    input: TensorLike,
    n: Optional[int] = None,
    tanh_approx: bool = False,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = gelu(x)``. ``tanh_approx=True`` selects the tanh approximation."""
    n = _numel(input, n)
    dt = infer_dtype(input, dtype)
    check(
        lib.ks_gelu(ptr(out, name="out"), ptr(input, name="input"), n,
                    1 if tanh_approx else 0, dt, default_stream(stream, input)),
        "ks_gelu",
    )
    return out


def relu(
    out: TensorLike,
    input: TensorLike,
    n: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = relu(x)`` over ``n`` elements."""
    n = _numel(input, n)
    dt = infer_dtype(input, dtype)
    check(
        lib.ks_relu(ptr(out, name="out"), ptr(input, name="input"), n, dt,
                    default_stream(stream, input)),
        "ks_relu",
    )
    return out


def swiglu(
    out: TensorLike,
    gate: TensorLike,
    up: TensorLike,
    rows: Optional[int] = None,
    inter: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = silu(gate) * up``. ``gate``/``up`` are each ``[rows, inter]``."""
    rows, inter = _rows_inter(gate, rows, inter)
    dt = infer_dtype(gate, dtype)
    check(
        lib.ks_swiglu(ptr(out, name="out"), ptr(gate, name="gate"),
                      ptr(up, name="up"), rows, inter, dt,
                      default_stream(stream, gate)),
        "ks_swiglu",
    )
    return out


def swiglu_packed(
    out: TensorLike,
    input: TensorLike,
    rows: Optional[int] = None,
    inter: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Packed SwiGLU: ``input`` is ``[rows, 2*inter]`` (gate = first half, up =
    second half); ``out`` is ``[rows, inter]``."""
    if rows is None or inter is None:
        # input's last dim is 2*inter.
        r, two_inter = _rows_inter(input, rows, None)
        rows = r if rows is None else int(rows)
        inter = two_inter // 2 if inter is None else int(inter)
    dt = infer_dtype(input, dtype)
    check(
        lib.ks_swiglu_packed(ptr(out, name="out"), ptr(input, name="input"),
                             int(rows), int(inter), dt,
                             default_stream(stream, input)),
        "ks_swiglu_packed",
    )
    return out


def geglu(
    out: TensorLike,
    gate: TensorLike,
    up: TensorLike,
    rows: Optional[int] = None,
    inter: Optional[int] = None,
    tanh_approx: bool = False,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = gelu(gate) * up`` (GeGLU). ``tanh_approx`` selects the
    approximation."""
    rows, inter = _rows_inter(gate, rows, inter)
    dt = infer_dtype(gate, dtype)
    check(
        lib.ks_geglu(ptr(out, name="out"), ptr(gate, name="gate"),
                     ptr(up, name="up"), rows, inter,
                     1 if tanh_approx else 0, dt, default_stream(stream, gate)),
        "ks_geglu",
    )
    return out


def swiglu_backward(
    grad_gate: TensorLike,
    grad_up: TensorLike,
    grad_out: TensorLike,
    gate: TensorLike,
    up: TensorLike,
    rows: Optional[int] = None,
    inter: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """SwiGLU backward; produces ``grad_gate``, ``grad_up`` from ``grad_out`` and
    the forward ``gate``/``up``. Returns ``(grad_gate, grad_up)``."""
    rows, inter = _rows_inter(gate, rows, inter)
    dt = infer_dtype(gate, dtype)
    check(
        lib.ks_swiglu_backward(
            ptr(grad_gate, name="grad_gate"),
            ptr(grad_up, name="grad_up"),
            ptr(grad_out, name="grad_out"),
            ptr(gate, name="gate"),
            ptr(up, name="up"),
            rows,
            inter,
            dt,
            default_stream(stream, gate),
        ),
        "ks_swiglu_backward",
    )
    return grad_gate, grad_up
