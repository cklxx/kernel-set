"""Pythonic ``IntEnum`` wrappers around the C ABI enums.

These mirror ``include/kernel_set/types.h`` and ``runtime.h``. They are plain
``IntEnum`` subclasses, so members are interchangeable with the raw integer
constants in :mod:`kernel_set._lib` and can be passed straight to ctypes.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["Status", "DType", "Activation", "QuantMode", "MemcpyKind"]


class Status(IntEnum):
    """``ks_status_t`` — return code of every C ABI entry point."""

    SUCCESS = 0
    ERROR_INVALID_ARGUMENT = 1
    ERROR_UNSUPPORTED_DTYPE = 2
    ERROR_UNSUPPORTED_SHAPE = 3
    ERROR_CUDA = 4
    ERROR_NOT_IMPLEMENTED = 5
    ERROR_OUT_OF_MEMORY = 6
    ERROR_ARCH_UNSUPPORTED = 7
    ERROR_INTERNAL = 8


class DType(IntEnum):
    """``ks_dtype_t`` — numeric element types understood across the ABI."""

    F32 = 0
    F16 = 1
    BF16 = 2
    F8E4M3 = 3
    F8E5M2 = 4
    F64 = 5
    I64 = 6
    I32 = 7
    I8 = 8
    U8 = 9
    I4 = 10


class Activation(IntEnum):
    """``ks_activation_t`` — fused-epilogue activation selector."""

    NONE = 0
    RELU = 1
    GELU = 2       # exact (erf) GELU
    GELU_TANH = 3  # tanh approximation
    SILU = 4       # x * sigmoid(x)


class QuantMode(IntEnum):
    """``ks_quant_mode_t`` — quantization granularity."""

    PER_TENSOR = 0
    PER_TOKEN = 1
    PER_CHANNEL = 2
    GROUPWISE = 3


class MemcpyKind(IntEnum):
    """``ks_memcpy_kind_t`` — direction for :func:`kernel_set.runtime.memcpy`."""

    HOST_TO_DEVICE = 0
    DEVICE_TO_HOST = 1
    DEVICE_TO_DEVICE = 2
