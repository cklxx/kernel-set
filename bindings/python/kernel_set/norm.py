"""Normalization kernels (``norm.h``): RMSNorm, LayerNorm + fused residual.

Layout: 2-D row-major ``[rows, cols]``; each row of ``cols`` is normalized
independently. ``weight``/``bias`` have ``cols`` elements. Reductions accumulate
in fp32 internally.

Every wrapper accepts torch CUDA tensors *or* raw int device pointers. When
torch tensors are passed, ``rows``/``cols``/``dtype``/``stream`` are inferred;
raw-pointer callers must pass them explicitly.
"""

from __future__ import annotations

from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = [
    "rms_norm",
    "rms_norm_residual",
    "layer_norm",
    "rms_norm_backward",
    "layer_norm_backward",
]


def _rows_cols(reference, rows, cols):
    """Infer (rows, cols) from a 2-D torch tensor when not given."""
    if rows is not None and cols is not None:
        return int(rows), int(cols)
    try:
        import torch  # noqa: F401

        if hasattr(reference, "shape") and len(reference.shape) >= 2:
            r = 1
            for s in reference.shape[:-1]:
                r *= s
            return int(r), int(reference.shape[-1])
    except Exception:
        pass
    raise ValueError("rows and cols are required for raw-pointer inputs")


def rms_norm(
    out: TensorLike,
    input: TensorLike,
    weight: TensorLike,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    eps: float = 1e-6,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = (x / rms(x)) * weight``, ``rms(x) = sqrt(mean(x^2) + eps)``.

    Returns ``out`` for chaining.
    """
    rows, cols = _rows_cols(input, rows, cols)
    dt = infer_dtype(input, dtype)
    check(
        lib.ks_rms_norm(
            ptr(out, name="out"),
            ptr(input, name="input"),
            ptr(weight, name="weight"),
            rows,
            cols,
            float(eps),
            dt,
            default_stream(stream, input),
        ),
        "ks_rms_norm",
    )
    return out


def rms_norm_residual(
    out: TensorLike,
    residual_out: TensorLike,
    input: TensorLike,
    residual: TensorLike,
    weight: TensorLike,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    eps: float = 1e-6,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """Fused add-RMSNorm: ``x = input + residual`` (-> ``residual_out``),
    ``out = rms_norm(x) * weight``. ``residual_out`` may alias ``residual``.

    Returns ``(out, residual_out)``.
    """
    rows, cols = _rows_cols(input, rows, cols)
    dt = infer_dtype(input, dtype)
    check(
        lib.ks_rms_norm_residual(
            ptr(out, name="out"),
            ptr(residual_out, name="residual_out"),
            ptr(input, name="input"),
            ptr(residual, name="residual"),
            ptr(weight, name="weight"),
            rows,
            cols,
            float(eps),
            dt,
            default_stream(stream, input),
        ),
        "ks_rms_norm_residual",
    )
    return out, residual_out


def layer_norm(
    out: TensorLike,
    input: TensorLike,
    weight: TensorLike,
    bias: TensorLike = None,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    eps: float = 1e-5,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``out = (x - mean) / sqrt(var + eps) * weight + bias``. ``bias`` may be
    ``None``. Returns ``out``."""
    rows, cols = _rows_cols(input, rows, cols)
    dt = infer_dtype(input, dtype)
    check(
        lib.ks_layer_norm(
            ptr(out, name="out"),
            ptr(input, name="input"),
            ptr(weight, name="weight"),
            ptr(bias, name="bias"),
            rows,
            cols,
            float(eps),
            dt,
            default_stream(stream, input),
        ),
        "ks_layer_norm",
    )
    return out


def rms_norm_backward(
    grad_input: TensorLike,
    grad_weight_fp32: TensorLike,
    grad_out: TensorLike,
    input: TensorLike,
    weight: TensorLike,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    eps: float = 1e-6,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """RMSNorm backward. ``grad_weight_fp32`` is an fp32 ``[cols]`` accumulator.
    Returns ``(grad_input, grad_weight_fp32)``."""
    rows, cols = _rows_cols(input, rows, cols)
    dt = infer_dtype(input, dtype)
    check(
        lib.ks_rms_norm_backward(
            ptr(grad_input, name="grad_input"),
            ptr(grad_weight_fp32, name="grad_weight_fp32"),
            ptr(grad_out, name="grad_out"),
            ptr(input, name="input"),
            ptr(weight, name="weight"),
            rows,
            cols,
            float(eps),
            dt,
            default_stream(stream, input),
        ),
        "ks_rms_norm_backward",
    )
    return grad_input, grad_weight_fp32


def layer_norm_backward(
    grad_input: TensorLike,
    grad_weight_fp32: TensorLike,
    grad_bias_fp32: TensorLike,
    grad_out: TensorLike,
    input: TensorLike,
    weight: TensorLike,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    eps: float = 1e-5,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """LayerNorm backward. ``grad_weight_fp32``/``grad_bias_fp32`` are fp32
    ``[cols]`` accumulators. Returns ``(grad_input, grad_weight_fp32,
    grad_bias_fp32)``."""
    rows, cols = _rows_cols(input, rows, cols)
    dt = infer_dtype(input, dtype)
    check(
        lib.ks_layer_norm_backward(
            ptr(grad_input, name="grad_input"),
            ptr(grad_weight_fp32, name="grad_weight_fp32"),
            ptr(grad_bias_fp32, name="grad_bias_fp32"),
            ptr(grad_out, name="grad_out"),
            ptr(input, name="input"),
            ptr(weight, name="weight"),
            rows,
            cols,
            float(eps),
            dt,
            default_stream(stream, input),
        ),
        "ks_layer_norm_backward",
    )
    return grad_input, grad_weight_fp32, grad_bias_fp32
