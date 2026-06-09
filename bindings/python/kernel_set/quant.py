"""Quantization / dequantization kernels (``quant.h``).

Scales are fp32 device buffers. "Dynamic" quantizers compute the scale from the
data and write it out; "static"/dequant take a precomputed scale.
"""

from __future__ import annotations

from ctypes import POINTER, c_float, cast
from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr
from .enums import DType, QuantMode

__all__ = [
    "quantize_fp8",
    "quantize_fp8_group",
    "dequantize_fp8",
    "quantize_int8",
    "dequantize_int8",
    "dequantize_int4",
    "repack_int4",
    "quantize_nvfp4",
]

_F32P = POINTER(c_float)


def _f32(obj, *, name, allow_none=False):
    raw = ptr(obj, name=name, allow_none=allow_none)
    return cast(raw, _F32P) if raw else _F32P()


def quantize_fp8(
    out: TensorLike,
    scale: TensorLike,
    input: TensorLike,
    rows: int,
    cols: int,
    in_dtype: Optional[int] = None,
    fp8_dtype: int = DType.F8E4M3,
    mode: int = QuantMode.PER_TENSOR,
    stream: TensorLike = None,
):
    """Dynamic FP8 quantization. ``out`` is FP8 ``[rows, cols]``; ``scale`` is
    fp32 ``[1]`` (per-tensor) or ``[rows]`` (per-token). Returns ``(out, scale)``.
    """
    dt = infer_dtype(input, in_dtype)
    check(
        lib.ks_quantize_fp8(
            ptr(out, name="out"), _f32(scale, name="scale"),
            ptr(input, name="input"),
            int(rows), int(cols), dt, int(fp8_dtype), int(mode),
            default_stream(stream, input),
        ),
        "ks_quantize_fp8",
    )
    return out, scale


def quantize_fp8_group(
    out: TensorLike,
    scale: TensorLike,
    input: TensorLike,
    rows: int,
    cols: int,
    group_size: int = 128,
    in_dtype: Optional[int] = None,
    fp8_dtype: int = DType.F8E4M3,
    stream: TensorLike = None,
):
    """Per-token-GROUP dynamic FP8 quantization (1 x ``group_size`` tiles): the
    DeepGEMM blockwise activation format. ``out`` is FP8 ``[rows, cols]``;
    ``scale`` is fp32 ``[rows, ceil(cols/group_size)]`` (one scale per
    (row, col-group)). Returns ``(out, scale)``."""
    dt = infer_dtype(input, in_dtype)
    check(
        lib.ks_quantize_fp8_group(
            ptr(out, name="out"), _f32(scale, name="scale"),
            ptr(input, name="input"),
            int(rows), int(cols), int(group_size), dt, int(fp8_dtype),
            default_stream(stream, input),
        ),
        "ks_quantize_fp8_group",
    )
    return out, scale


def dequantize_fp8(
    out: TensorLike,
    input: TensorLike,
    scale: TensorLike,
    rows: int,
    cols: int,
    out_dtype: Optional[int] = None,
    fp8_dtype: int = DType.F8E4M3,
    mode: int = QuantMode.PER_TENSOR,
    stream: TensorLike = None,
) -> TensorLike:
    """Dequantize FP8 ``input`` -> ``out_dtype`` ``out``. ``scale`` is fp32."""
    dt = infer_dtype(out, out_dtype)
    check(
        lib.ks_dequantize_fp8(
            ptr(out, name="out"), ptr(input, name="input"),
            _f32(scale, name="scale"),
            int(rows), int(cols), dt, int(fp8_dtype), int(mode),
            default_stream(stream, out),
        ),
        "ks_dequantize_fp8",
    )
    return out


def quantize_int8(
    out: TensorLike,
    scale: TensorLike,
    input: TensorLike,
    rows: int,
    cols: int,
    in_dtype: Optional[int] = None,
    mode: int = QuantMode.PER_TOKEN,
    stream: TensorLike = None,
):
    """Dynamic symmetric INT8 quantization. ``out`` int8 + fp32 ``scale``.
    Returns ``(out, scale)``."""
    dt = infer_dtype(input, in_dtype)
    check(
        lib.ks_quantize_int8(
            ptr(out, name="out"), _f32(scale, name="scale"),
            ptr(input, name="input"),
            int(rows), int(cols), dt, int(mode),
            default_stream(stream, input),
        ),
        "ks_quantize_int8",
    )
    return out, scale


def dequantize_int8(
    out: TensorLike,
    input: TensorLike,
    scale: TensorLike,
    rows: int,
    cols: int,
    out_dtype: Optional[int] = None,
    mode: int = QuantMode.PER_TOKEN,
    stream: TensorLike = None,
) -> TensorLike:
    """Dequantize INT8 ``input`` -> ``out_dtype`` ``out`` using fp32 ``scale``."""
    dt = infer_dtype(out, out_dtype)
    check(
        lib.ks_dequantize_int8(
            ptr(out, name="out"), ptr(input, name="input"),
            _f32(scale, name="scale"),
            int(rows), int(cols), dt, int(mode),
            default_stream(stream, out),
        ),
        "ks_dequantize_int8",
    )
    return out


def dequantize_int4(
    out: TensorLike,
    qweight_packed: TensorLike,
    scales: TensorLike,
    zeros: TensorLike,
    k: int,
    n: int,
    group_size: int,
    out_dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Dequantize AWQ/GPTQ group-wise INT4 weights.

    ``qweight_packed``: int32 ``[k/8, n]`` (8 signed-less nibbles per word, packed
    along K. Nibble j (j=0..7) of word is ``(word >> (4*j)) & 0xF``).
    ``scales``/``zeros``: ``out_dtype`` ``[k/group_size, n]``.
    """
    dt = infer_dtype(out, out_dtype)
    check(
        lib.ks_dequantize_int4(
            ptr(out, name="out"), ptr(qweight_packed, name="qweight_packed"),
            ptr(scales, name="scales"), ptr(zeros, name="zeros"),
            int(k), int(n), int(group_size), dt,
            default_stream(stream, out),
        ),
        "ks_dequantize_int4",
    )
    return out


def repack_int4(
    out_packed: TensorLike,
    qweight: TensorLike,
    perm: TensorLike,
    size_k: int,
    size_n: int,
    num_bits: int = 4,
    stream: TensorLike = None,
) -> TensorLike:
    """Repack GPTQ/AWQ weights to Marlin/Machete format."""
    check(
        lib.ks_repack_int4(
            ptr(out_packed, name="out_packed"), ptr(qweight, name="qweight"),
            ptr(perm, name="perm", allow_none=True),
            int(size_k), int(size_n), int(num_bits),
            default_stream(stream, out_packed),
        ),
        "ks_repack_int4",
    )
    return out_packed


def quantize_nvfp4(
    out_fp4: TensorLike,
    out_scales: TensorLike,
    input: TensorLike,
    global_scale: float,
    rows: int,
    cols: int,
    in_dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """NVFP4 quantization correctness fallback."""
    dt = infer_dtype(input, in_dtype)
    check(
        lib.ks_quantize_nvfp4(
            ptr(out_fp4, name="out_fp4"), ptr(out_scales, name="out_scales"),
            ptr(input, name="input"), float(global_scale),
            int(rows), int(cols), dt,
            default_stream(stream, input),
        ),
        "ks_quantize_nvfp4",
    )
    return out_fp4, out_scales
