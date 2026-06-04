"""Runtime / device-management wrappers (``runtime.h``).

Library introspection, device queries, stream lifecycle and a small device
memory allocator. Tensor-aware callers (PyTorch) usually skip the memory and
stream helpers and pass their own pointers/streams into the op modules, but
these are handy for bindings used without a tensor library and for diagnostics.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_int, c_size_t, c_void_p
from dataclasses import dataclass
from typing import List, Optional

from . import _lib
from ._lib import KsDeviceProperties, check, lib
from ._tensor import ptr, stream_ptr
from .enums import MemcpyKind

__all__ = [
    "version",
    "backend_name",
    "status_string",
    "last_error_string",
    "dtype_size_bits",
    "dtype_name",
    "device_count",
    "set_device",
    "get_device",
    "DeviceProperties",
    "get_device_properties",
    "Stream",
    "stream_create",
    "stream_destroy",
    "stream_synchronize",
    "malloc_device",
    "free_device",
    "memcpy",
    "memset_device",
]


# ---- introspection --------------------------------------------------------


def version() -> str:
    """Semantic version of the loaded library, e.g. ``"0.1.0"``."""
    return lib.ks_version().decode("utf-8")


def backend_name() -> str:
    """Backend this build targets: ``"cuda"``, ``"hip"`` or ``"none"``."""
    return lib.ks_backend_name().decode("utf-8")


def status_string(status: int) -> str:
    """Human-readable name for a ``ks_status_t`` value."""
    return _lib.status_string(int(status))


def last_error_string() -> str:
    """Thread-local description of the last backend runtime error."""
    return _lib.last_error_string()


def dtype_size_bits(dtype: int) -> int:
    """Element size in bits (e.g. ``DType.I4`` -> 4, ``DType.F32`` -> 32)."""
    return int(lib.ks_dtype_size_bits(int(dtype)))


def dtype_name(dtype: int) -> str:
    """Short token for a dtype, e.g. ``"bf16"``."""
    return lib.ks_dtype_name(int(dtype)).decode("utf-8")


# ---- device queries -------------------------------------------------------


def device_count() -> int:
    """Number of visible GPU devices."""
    out = c_int(0)
    check(lib.ks_device_count(byref(out)), "ks_device_count")
    return out.value


def set_device(device: int) -> None:
    """Make ``device`` the current device for this thread."""
    check(lib.ks_set_device(int(device)), "ks_set_device")


def get_device() -> int:
    """Return the current device ordinal."""
    out = c_int(0)
    check(lib.ks_get_device(byref(out)), "ks_get_device")
    return out.value


@dataclass
class DeviceProperties:
    """Pythonic view of ``ks_device_properties_t``."""

    name: str
    compute_major: int
    compute_minor: int
    multiprocessor_count: int
    max_threads_per_block: int
    max_shared_memory_per_block: int
    warp_size: int
    total_global_memory: int
    supports_bf16: bool
    supports_fp8: bool
    supports_tf32: bool

    @property
    def compute_capability(self) -> str:
        return f"{self.compute_major}.{self.compute_minor}"


def get_device_properties(device: int) -> DeviceProperties:
    """Query :class:`DeviceProperties` for ``device``."""
    props = KsDeviceProperties()
    check(
        lib.ks_get_device_properties(int(device), byref(props)),
        "ks_get_device_properties",
    )
    return DeviceProperties(
        name=props.name.decode("utf-8", "replace"),
        compute_major=props.compute_major,
        compute_minor=props.compute_minor,
        multiprocessor_count=props.multiprocessor_count,
        max_threads_per_block=props.max_threads_per_block,
        max_shared_memory_per_block=props.max_shared_memory_per_block,
        warp_size=props.warp_size,
        total_global_memory=props.total_global_memory,
        supports_bf16=bool(props.supports_bf16),
        supports_fp8=bool(props.supports_fp8),
        supports_tf32=bool(props.supports_tf32),
    )


# ---- streams --------------------------------------------------------------


def stream_create() -> int:
    """Create a new GPU stream and return it as a raw integer handle."""
    out = c_void_p()
    check(lib.ks_stream_create(byref(out)), "ks_stream_create")
    return out.value or 0


def stream_destroy(stream: int) -> None:
    """Destroy a stream created with :func:`stream_create`."""
    check(lib.ks_stream_destroy(stream_ptr(stream)), "ks_stream_destroy")


def stream_synchronize(stream: int = 0) -> None:
    """Block until all work on ``stream`` completes (0 = default stream)."""
    check(lib.ks_stream_synchronize(stream_ptr(stream)), "ks_stream_synchronize")


class Stream:
    """Context-manager wrapper around a ``kernel_set`` stream.

    Usage::

        with ks.runtime.Stream() as s:
            ks.norm.rms_norm(out, x, w, stream=s.handle)
            s.synchronize()
    """

    def __init__(self) -> None:
        self.handle: int = stream_create()
        self._owned = True

    def synchronize(self) -> None:
        stream_synchronize(self.handle)

    def destroy(self) -> None:
        if self._owned and self.handle:
            stream_destroy(self.handle)
            self.handle = 0
            self._owned = False

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *exc) -> None:
        self.destroy()

    def __int__(self) -> int:
        return self.handle


# ---- device memory --------------------------------------------------------


def malloc_device(nbytes: int) -> int:
    """Allocate ``nbytes`` of device memory; returns a raw int pointer."""
    out = c_void_p()
    check(lib.ks_malloc_device(byref(out), c_size_t(int(nbytes))), "ks_malloc_device")
    return out.value or 0


def free_device(pointer: int) -> None:
    """Free device memory allocated by :func:`malloc_device`."""
    check(lib.ks_free_device(ptr(pointer)), "ks_free_device")


def memcpy(dst, src, nbytes: int, kind: int, stream=None) -> None:
    """Copy ``nbytes`` between host/device buffers.

    ``dst``/``src`` may be raw int pointers or torch tensors; ``kind`` is a
    :class:`~kernel_set.enums.MemcpyKind`.
    """
    check(
        lib.ks_memcpy(
            ptr(dst, name="dst"),
            ptr(src, name="src"),
            c_size_t(int(nbytes)),
            int(kind),
            stream_ptr(stream),
        ),
        "ks_memcpy",
    )


def memset_device(dst, value: int, nbytes: int, stream=None) -> None:
    """Set ``nbytes`` of device memory at ``dst`` to byte ``value``."""
    check(
        lib.ks_memset_device(
            ptr(dst, name="dst"),
            int(value),
            c_size_t(int(nbytes)),
            stream_ptr(stream),
        ),
        "ks_memset_device",
    )
