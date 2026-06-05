"""General matrix multiply and fused/quantized variants (``gemm.h``).

Row-major convention. For ``op(A) [M,K] x op(B) [K,N] = C [M,N]``:
  * ``trans_a``/``trans_b`` transpose the respective operand.
  * leading dimensions are the row stride (in elements) of the stored matrix.

FP16/BF16 paths target tensor cores with fp32 accumulation.
"""

from __future__ import annotations

from ctypes import POINTER, c_float, cast
from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr
from .enums import Activation, QuantMode

__all__ = [
    "gemm",
    "gemm_bias_act",
    "gemm_batched",
    "gemm_w8a8",
    "gemm_w4a16",
    "gemm_fp8",
    "gemm_fp8_blockwise",
]

_F32P = POINTER(c_float)


def _f32(obj: TensorLike, *, name: str):
    """Resolve an fp32 scale tensor/pointer to a ``POINTER(c_float)`` (or NULL)."""
    raw = ptr(obj, name=name, allow_none=True)
    return cast(raw, _F32P) if raw else _F32P()


def gemm(
    c: TensorLike,
    a: TensorLike,
    b: TensorLike,
    m: int,
    n: int,
    k: int,
    trans_a: bool = False,
    trans_b: bool = False,
    lda: int = 0,
    ldb: int = 0,
    ldc: int = 0,
    alpha: float = 1.0,
    beta: float = 0.0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``C = alpha * op(A) @ op(B) + beta * C``.

    Leading dims default to the natural row stride of the *stored* operand when
    passed as 0: ``lda = K if !trans_a else M``, ``ldb = N if !trans_b else K``,
    ``ldc = N``.
    """
    if lda == 0:
        lda = m if trans_a else k
    if ldb == 0:
        ldb = k if trans_b else n
    if ldc == 0:
        ldc = n
    dt = infer_dtype(a, dtype)
    check(
        lib.ks_gemm(
            ptr(c, name="c"), ptr(a, name="a"), ptr(b, name="b"),
            int(m), int(n), int(k),
            1 if trans_a else 0, 1 if trans_b else 0,
            int(lda), int(ldb), int(ldc),
            float(alpha), float(beta), dt, default_stream(stream, a),
        ),
        "ks_gemm",
    )
    return c


def gemm_bias_act(
    d: TensorLike,
    a: TensorLike,
    b: TensorLike,
    m: int,
    n: int,
    k: int,
    bias: TensorLike = None,
    alpha: float = 1.0,
    act: int = Activation.NONE,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """``D = act(alpha * A @ B + bias)``. ``bias`` is ``[N]`` or ``None``."""
    dt = infer_dtype(a, dtype)
    check(
        lib.ks_gemm_bias_act(
            ptr(d, name="d"), ptr(a, name="a"), ptr(b, name="b"),
            ptr(bias, name="bias"),
            int(m), int(n), int(k),
            float(alpha), int(act), dt, default_stream(stream, a),
        ),
        "ks_gemm_bias_act",
    )
    return d


def gemm_batched(
    c: TensorLike,
    a: TensorLike,
    b: TensorLike,
    batch: int,
    m: int,
    n: int,
    k: int,
    trans_a: bool = False,
    trans_b: bool = False,
    stride_a: Optional[int] = None,
    stride_b: Optional[int] = None,
    stride_c: Optional[int] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Strided-batched GEMM. Each batch ``i`` uses ``base + i*stride_{a,b,c}``.

    Strides default to the contiguous per-matrix element counts when omitted.
    """
    if stride_a is None:
        stride_a = m * k
    if stride_b is None:
        stride_b = k * n
    if stride_c is None:
        stride_c = m * n
    dt = infer_dtype(a, dtype)
    check(
        lib.ks_gemm_batched(
            ptr(c, name="c"), ptr(a, name="a"), ptr(b, name="b"),
            int(batch), int(m), int(n), int(k),
            1 if trans_a else 0, 1 if trans_b else 0,
            int(stride_a), int(stride_b), int(stride_c),
            float(alpha), float(beta), dt, default_stream(stream, a),
        ),
        "ks_gemm_batched",
    )
    return c


def gemm_w8a8(
    c: TensorLike,
    a_i8: TensorLike,
    b_i8: TensorLike,
    a_scale: TensorLike,
    b_scale: TensorLike,
    m: int,
    n: int,
    k: int,
    bias: TensorLike = None,
    a_mode: int = QuantMode.PER_TOKEN,
    b_mode: int = QuantMode.PER_CHANNEL,
    out_dtype: int = None,
    stream: TensorLike = None,
) -> TensorLike:
    """W8A8 GEMM: int8 ``A [M,K]`` x int8 ``B [K,N]`` -> ``out_dtype`` ``C [M,N]``.

    ``a_scale``/``b_scale`` are fp32 device buffers ([M]/[N] or [1]); ``bias`` is
    ``[N]`` in ``out_dtype`` or ``None``. ``out_dtype`` (a ``DType``) is
    required.
    """
    if out_dtype is None:
        # try to infer from the output tensor if it is a torch tensor
        from ._tensor import _is_torch_tensor  # local import to avoid cycles

        if _is_torch_tensor(c):
            from ._tensor import dtype_to_ks

            out_dtype = dtype_to_ks(c.dtype)
        else:
            raise ValueError("out_dtype is required for ks_gemm_w8a8")
    check(
        lib.ks_gemm_w8a8(
            ptr(c, name="c"), ptr(a_i8, name="a_i8"), ptr(b_i8, name="b_i8"),
            _f32(a_scale, name="a_scale"), _f32(b_scale, name="b_scale"),
            ptr(bias, name="bias"),
            int(m), int(n), int(k),
            int(a_mode), int(b_mode), int(out_dtype),
            default_stream(stream, c if not isinstance(c, int) else a_i8),
        ),
        "ks_gemm_w8a8",
    )
    return c


def gemm_w4a16(
    c: TensorLike,
    a: TensorLike,
    b_packed: TensorLike,
    scales: TensorLike,
    zeros: TensorLike,
    m: int,
    n: int,
    k: int,
    group_size: int,
    bias: TensorLike = None,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """W4A16 GEMM: fp16/bf16 ``A [M,K]`` x int4 weights ``B [K,N]`` with
    group-wise scales/zeros (AWQ/GPTQ). ``b_packed`` holds two int4 per byte;
    ``scales``/``zeros`` are ``[K/group_size, N]``. ``dtype`` is the activation
    /output dtype."""
    dt = infer_dtype(a, dtype)
    check(
        lib.ks_gemm_w4a16(
            ptr(c, name="c"), ptr(a, name="a"), ptr(b_packed, name="b_packed"),
            ptr(scales, name="scales"), ptr(zeros, name="zeros"),
            ptr(bias, name="bias"),
            int(m), int(n), int(k), int(group_size), dt,
            default_stream(stream, a),
        ),
        "ks_gemm_w4a16",
    )
    return c


def gemm_fp8(
    out: TensorLike,
    a_fp8: TensorLike,
    b_fp8: TensorLike,
    a_scale: TensorLike,
    b_scale: TensorLike,
    m: int,
    n: int,
    k: int,
    a_mode: int = QuantMode.PER_TOKEN,
    b_mode: int = QuantMode.PER_CHANNEL,
    fp8_dtype: int = None,
    out_dtype: int = None,
    stream: TensorLike = None,
) -> TensorLike:
    """FP8 GEMM: fp8 ``A [M,K]`` x fp8 ``B [K,N]`` -> ``out_dtype`` ``C [M,N]``.

    Both operands are FP8 (e4m3 or e5m2, selected by ``fp8_dtype``), dequantized
    to fp32 by the supplied scales before the matmul::

        C = (a_scale (x) b_scale) * (A_fp8 @ B_fp8)

    ``a_scale``/``b_scale`` are fp32 device buffers ([M]/[N] or [1] for
    per-tensor). Row-major, non-transposed (A row stride K, B row stride N).
    ``fp8_dtype`` (a ``DType``: ``F8E4M3`` or ``F8E5M2``) is the operand dtype,
    inferred from ``a_fp8`` when omitted; ``out_dtype`` (a ``DType``) is required
    and is inferred from a torch ``out`` tensor when omitted.
    """
    if fp8_dtype is None:
        fp8_dtype = infer_dtype(a_fp8, None)
    if out_dtype is None:
        # try to infer from the output tensor if it is a torch tensor
        from ._tensor import _is_torch_tensor  # local import to avoid cycles

        if _is_torch_tensor(out):
            from ._tensor import dtype_to_ks

            out_dtype = dtype_to_ks(out.dtype)
        else:
            raise ValueError("out_dtype is required for ks_gemm_fp8")
    check(
        lib.ks_gemm_fp8(
            ptr(out, name="out"), ptr(a_fp8, name="a_fp8"),
            ptr(b_fp8, name="b_fp8"),
            _f32(a_scale, name="a_scale"), _f32(b_scale, name="b_scale"),
            int(m), int(n), int(k),
            int(a_mode), int(b_mode), int(fp8_dtype), int(out_dtype),
            default_stream(stream, out if not isinstance(out, int) else a_fp8),
        ),
        "ks_gemm_fp8",
    )
    return out


def gemm_fp8_blockwise(
    out: TensorLike,
    a_fp8: TensorLike,
    b_fp8: TensorLike,
    a_scale: TensorLike,
    b_scale: TensorLike,
    m: int,
    n: int,
    k: int,
    block_n: int = 128,
    block_k: int = 128,
    fp8_dtype: int = None,
    out_dtype: int = None,
    stream: TensorLike = None,
) -> TensorLike:
    """FP8 BLOCKWISE GEMM (DeepSeek-V3 recipe): fp8 ``A [M,K]`` x fp8 ``B [K,N]``
    -> ``out_dtype`` ``C [M,N]`` with fine-grained fp32 scales and two-level
    accumulation::

        a_scale: [M, ceil(K/block_k)]               (1 x block_k act tiles)
        b_scale: [ceil(K/block_k), ceil(N/block_n)] (block_k x block_n blocks)

    Portable SIMT software-dequant path (runs sm_80+); the hardware fp8-MMA
    upgrade is DeepGEMM/CUTLASS via ``kernel_set.dispatch.fp8_gemm_blockwise``.
    ``fp8_dtype`` (``F8E4M3``/``F8E5M2``) is inferred from ``a_fp8`` when omitted;
    ``out_dtype`` is inferred from a torch ``out`` tensor when omitted.
    """
    if fp8_dtype is None:
        fp8_dtype = infer_dtype(a_fp8, None)
    if out_dtype is None:
        from ._tensor import _is_torch_tensor

        if _is_torch_tensor(out):
            from ._tensor import dtype_to_ks

            out_dtype = dtype_to_ks(out.dtype)
        else:
            raise ValueError("out_dtype is required for ks_gemm_fp8_blockwise")
    check(
        lib.ks_gemm_fp8_blockwise(
            ptr(out, name="out"), ptr(a_fp8, name="a_fp8"),
            ptr(b_fp8, name="b_fp8"),
            _f32(a_scale, name="a_scale"), _f32(b_scale, name="b_scale"),
            int(m), int(n), int(k), int(block_n), int(block_k),
            int(fp8_dtype), int(out_dtype),
            default_stream(stream, out if not isinstance(out, int) else a_fp8),
        ),
        "ks_gemm_fp8_blockwise",
    )
    return out
