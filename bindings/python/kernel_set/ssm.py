"""State-space model (Mamba / SSM) kernels (``ssm.h``).

Three primitives back the Mamba block:
  * :func:`causal_conv1d`        — depthwise causal 1-D convolution (+ optional SiLU)
  * :func:`selective_scan`       — the data-dependent (selective) SSM scan
  * :func:`selective_scan_update` — single-step decode update of the SSM state

Layout (row-major, channel-major like the reference Mamba CUDA):
  * Sequence tensors are ``[batch, dim, seqlen]`` — the sequence axis is innermost
    (contiguous), ``dim`` is the middle stride, ``batch`` is outermost.
  * ``B``/``C`` carry the selective projections shared across all ``dim`` channels:
    ``[batch, dstate, seqlen]`` (n_groups == 1).

Dtype policy. The bulk activation tensors (``x``, ``dt``, ``B``, ``C``, ``z``,
``out``, conv ``weight``/``bias``) use the dispatched ``dtype`` (f32 / f16 /
bf16). The recurrence parameters ``A``, ``D`` and ``dt_bias`` are ALWAYS fp32
device buffers and ``selective_scan_update`` keeps ``state`` in fp32 across decode
steps. Pass ``None`` for the optional ``bias`` / ``D`` / ``z`` / ``dt_bias``
arguments to skip them.

Every wrapper accepts torch CUDA tensors *or* raw int device pointers. When torch
tensors are passed, the shape ints / ``dtype`` / ``stream`` are inferred from the
named tensor; raw-pointer callers must pass them explicitly.
"""

from __future__ import annotations

from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = [
    "causal_conv1d",
    "selective_scan",
    "selective_scan_update",
]


def _conv_shapes(x, batch, dim, seqlen):
    """Infer (batch, dim, seqlen) from a 3-D ``[batch, dim, seqlen]`` tensor."""
    if batch is not None and dim is not None and seqlen is not None:
        return int(batch), int(dim), int(seqlen)
    if hasattr(x, "shape") and len(getattr(x, "shape")) == 3:
        s = x.shape
        return int(s[0]), int(s[1]), int(s[2])
    raise ValueError(
        "batch, dim and seqlen are required for raw-pointer inputs"
    )


def _conv_width(weight, width):
    """Infer the conv ``width`` from a 2-D ``[dim, width]`` weight tensor."""
    if width is not None:
        return int(width)
    if hasattr(weight, "shape") and len(getattr(weight, "shape")) == 2:
        return int(weight.shape[-1])
    raise ValueError("width is required for raw-pointer weight inputs")


def causal_conv1d(
    out: TensorLike,
    x: TensorLike,
    weight: TensorLike,
    bias: TensorLike = None,
    batch: Optional[int] = None,
    dim: Optional[int] = None,
    seqlen: Optional[int] = None,
    width: Optional[int] = None,
    silu: bool = False,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Depthwise causal 1-D convolution.

    ``x``/``out`` are ``[batch, dim, seqlen]`` (``dtype``); ``weight`` is the
    per-channel kernel ``[dim, width]`` (``dtype``); ``bias`` is an optional fp32
    ``[dim]`` buffer (or ``None``). Causal with implicit zero left-padding::

        out[b,d,t] = bias[d] + sum_k weight[d,k] * x[b,d, t-(width-1)+k]

    If ``silu`` is true a SiLU activation is applied to the result. Returns
    ``out``.
    """
    batch, dim, seqlen = _conv_shapes(x, batch, dim, seqlen)
    width = _conv_width(weight, width)
    dt = infer_dtype(x, dtype)
    check(
        lib.ks_causal_conv1d(
            ptr(out, name="out"),
            ptr(x, name="x"),
            ptr(weight, name="weight"),
            ptr(bias, name="bias"),
            int(batch),
            int(dim),
            int(seqlen),
            int(width),
            1 if silu else 0,
            dt,
            default_stream(stream, x),
        ),
        "ks_causal_conv1d",
    )
    return out


def selective_scan(
    out: TensorLike,
    x: TensorLike,
    dt: TensorLike,
    A: TensorLike,
    B: TensorLike,
    C: TensorLike,
    D: TensorLike = None,
    z: TensorLike = None,
    dt_bias: TensorLike = None,
    delta_softplus: bool = False,
    batch: Optional[int] = None,
    dim: Optional[int] = None,
    seqlen: Optional[int] = None,
    dstate: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Mamba selective scan (forward).

    ``x``/``dt``/``out`` are ``[batch, dim, seqlen]``; ``B``/``C`` are
    ``[batch, dstate, seqlen]`` (shared across ``dim``). ``A`` is fp32
    ``[dim, dstate]``; ``D``/``dt_bias`` are fp32 ``[dim]`` or ``None``; ``z`` is
    a ``[batch, dim, seqlen]`` SiLU gate or ``None``. ``dtype`` covers the
    activation tensors; ``batch``/``dim``/``seqlen`` are inferred from ``x`` and
    ``dstate`` from ``A``/``B`` when omitted. Returns ``out``.
    """
    batch, dim, seqlen = _conv_shapes(x, batch, dim, seqlen)
    if dstate is None:
        if hasattr(A, "shape") and len(getattr(A, "shape")) == 2:
            dstate = int(A.shape[-1])
        elif hasattr(B, "shape") and len(getattr(B, "shape")) == 3:
            dstate = int(B.shape[1])
        else:
            raise ValueError("dstate is required for raw-pointer inputs")
    dty = infer_dtype(x, dtype)
    check(
        lib.ks_selective_scan(
            ptr(out, name="out"),
            ptr(x, name="x"),
            ptr(dt, name="dt"),
            ptr(A, name="A"),
            ptr(B, name="B"),
            ptr(C, name="C"),
            ptr(D, name="D"),
            ptr(z, name="z"),
            ptr(dt_bias, name="dt_bias"),
            1 if delta_softplus else 0,
            int(batch),
            int(dim),
            int(seqlen),
            int(dstate),
            dty,
            default_stream(stream, x),
        ),
        "ks_selective_scan",
    )
    return out


def selective_scan_update(
    state: TensorLike,
    out: TensorLike,
    x: TensorLike,
    dt: TensorLike,
    A: TensorLike,
    B: TensorLike,
    C: TensorLike,
    D: TensorLike = None,
    z: TensorLike = None,
    dt_bias: TensorLike = None,
    delta_softplus: bool = False,
    batch: Optional[int] = None,
    dim: Optional[int] = None,
    dstate: Optional[int] = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Single-step selective-scan decode update (``seqlen == 1``).

    Advances the SSM state by one position and updates ``state`` in place.
    ``state`` is the fp32 recurrent state ``[batch, dim, dstate]`` (read and
    written). ``x``/``dt``/``out`` are ``[batch, dim]``; ``B``/``C`` are
    ``[batch, dstate]``; ``A`` is fp32 ``[dim, dstate]``; ``D``/``dt_bias`` are
    fp32 ``[dim]`` or ``None``; ``z`` is a ``[batch, dim]`` gate or ``None``.
    ``batch``/``dim`` are inferred from ``x`` and ``dstate`` from ``A``/``B``
    /``state`` when omitted. Returns ``out``.
    """
    if batch is None or dim is None:
        if hasattr(x, "shape") and len(getattr(x, "shape")) == 2:
            batch = int(x.shape[0]) if batch is None else int(batch)
            dim = int(x.shape[1]) if dim is None else int(dim)
        else:
            raise ValueError("batch and dim are required for raw-pointer inputs")
    if dstate is None:
        if hasattr(A, "shape") and len(getattr(A, "shape")) == 2:
            dstate = int(A.shape[-1])
        elif hasattr(B, "shape") and len(getattr(B, "shape")) == 2:
            dstate = int(B.shape[-1])
        elif hasattr(state, "shape") and len(getattr(state, "shape")) == 3:
            dstate = int(state.shape[-1])
        else:
            raise ValueError("dstate is required for raw-pointer inputs")
    dty = infer_dtype(x, dtype)
    check(
        lib.ks_selective_scan_update(
            ptr(state, name="state"),
            ptr(out, name="out"),
            ptr(x, name="x"),
            ptr(dt, name="dt"),
            ptr(A, name="A"),
            ptr(B, name="B"),
            ptr(C, name="C"),
            ptr(D, name="D"),
            ptr(z, name="z"),
            ptr(dt_bias, name="dt_bias"),
            1 if delta_softplus else 0,
            int(batch),
            int(dim),
            int(dstate),
            dty,
            default_stream(stream, x),
        ),
        "ks_selective_scan_update",
    )
    return out
