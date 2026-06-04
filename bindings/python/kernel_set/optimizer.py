"""Fused optimizer step kernels (``optimizer.h``).

Each kernel updates a single parameter tensor of ``n`` elements in place. State
tensors (``exp_avg``, ``exp_avg_sq``, ``momentum``) are fp32; ``param``/``grad``
may be lower precision. An optional ``master_param`` holds the fp32 master copy.
"""

from __future__ import annotations

from ctypes import POINTER, c_float, c_int64, c_void_p, cast
from typing import Optional, Sequence

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = ["adamw", "sgd_momentum", "global_grad_norm"]

_F32P = POINTER(c_float)


def _f32(obj, *, name, allow_none=False):
    raw = ptr(obj, name=name, allow_none=allow_none)
    return cast(raw, _F32P) if raw else _F32P()


def _numel(reference, n):
    if n is not None:
        return int(n)
    if hasattr(reference, "numel"):
        return int(reference.numel())
    raise ValueError("n (element count) is required for raw-pointer inputs")


def adamw(
    param: TensorLike,
    grad: TensorLike,
    exp_avg: TensorLike,
    exp_avg_sq: TensorLike,
    lr: float,
    step: int,
    master_param: TensorLike = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    grad_scale: float = 1.0,
    n: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """AdamW (decoupled weight decay) in-place update. ``step`` is the 1-based
    iteration for bias correction. ``exp_avg``/``exp_avg_sq`` are fp32 state.
    Returns ``param``."""
    n = _numel(param, n)
    dt = infer_dtype(param, dtype)
    check(
        lib.ks_adamw(
            ptr(param, name="param"), ptr(master_param, name="master_param"),
            ptr(grad, name="grad"),
            _f32(exp_avg, name="exp_avg"), _f32(exp_avg_sq, name="exp_avg_sq"),
            float(lr), float(beta1), float(beta2), float(eps),
            float(weight_decay), int(step), float(grad_scale),
            n, dt, default_stream(stream, param),
        ),
        "ks_adamw",
    )
    return param


def sgd_momentum(
    param: TensorLike,
    grad: TensorLike,
    momentum: TensorLike,
    lr: float,
    master_param: TensorLike = None,
    momentum_factor: float = 0.9,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    grad_scale: float = 1.0,
    n: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """SGD with (optional Nesterov) momentum and weight decay, in place.
    ``momentum`` is fp32 state. Returns ``param``."""
    n = _numel(param, n)
    dt = infer_dtype(param, dtype)
    check(
        lib.ks_sgd_momentum(
            ptr(param, name="param"), ptr(master_param, name="master_param"),
            ptr(grad, name="grad"), _f32(momentum, name="momentum"),
            float(lr), float(momentum_factor), float(weight_decay),
            1 if nesterov else 0, float(grad_scale), n, dt,
            default_stream(stream, param),
        ),
        "ks_sgd_momentum",
    )
    return param


def global_grad_norm(
    out_norm: TensorLike,
    grads: Sequence[TensorLike],
    sizes: Optional[Sequence[int]] = None,
    dtype: int = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Compute the global L2 norm of a list of grad tensors (for clipping).

    ``grads`` is a sequence of torch tensors or raw int pointers; ``sizes`` is
    their element counts (inferred from torch tensors when omitted). ``out_norm``
    is an fp32 ``[1]`` device scalar = ``sqrt(sum ||g||^2)``. ``dtype`` is the
    grads' element dtype (inferred from the first torch tensor if omitted).
    Returns ``out_norm``.
    """
    num = len(grads)
    if sizes is None:
        sizes = []
        for g in grads:
            if hasattr(g, "numel"):
                sizes.append(int(g.numel()))
            else:
                raise ValueError(
                    "sizes is required when grads contains raw pointers"
                )
    if len(sizes) != num:
        raise ValueError("grads and sizes must have the same length")

    if dtype is None:
        dtype = infer_dtype(grads[0] if num else None, None)

    grad_arr = (c_void_p * num)(*(ptr(g, name=f"grads[{i}]") for i, g in enumerate(grads)))
    size_arr = (c_int64 * num)(*[int(s) for s in sizes])

    check(
        lib.ks_global_grad_norm(
            _f32(out_norm, name="out_norm"),
            cast(grad_arr, POINTER(c_void_p)),
            cast(size_arr, POINTER(c_int64)),
            int(num), int(dtype),
            default_stream(stream, grads[0] if num else None),
        ),
        "ks_global_grad_norm",
    )
    return out_norm
