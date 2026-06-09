"""kernel_set — Python bindings for the kernel-set LLM kernel library.

A thin, dependency-free ctypes layer over the frozen C ABI declared in
``include/kernel_set/*.h``. The prebuilt shared library
(``libkernel_set.so`` / ``.dylib`` / ``kernel_set.dll``) is located and
``dlopen``-ed at import time — no compilation happens on install.

Quick start
-----------
::

    import torch
    import kernel_set as ks

    x = torch.randn(4, 4096, device="cuda", dtype=torch.float16)
    w = torch.ones(4096, device="cuda", dtype=torch.float16)
    out = torch.empty_like(x)

    ks.norm.rms_norm(out, x, w, eps=1e-6)   # dtype/shape/stream inferred
    torch.cuda.synchronize()

Every wrapper accepts torch CUDA tensors *or* raw integer device pointers; with
raw pointers you must pass shapes/dtype explicitly.

Layout
------
* :mod:`kernel_set._lib`     — raw ctypes FFI (every function, argtypes/restype).
* :mod:`kernel_set.runtime`  — device/stream/memory + introspection.
* op modules: :mod:`norm`, :mod:`activation`, :mod:`attention`, :mod:`gemm`,
  :mod:`moe`, :mod:`rope`, :mod:`quant`, :mod:`sampling`, :mod:`embedding`,
  :mod:`elementwise`, :mod:`loss`, :mod:`optimizer`, :mod:`ssm`.
"""

from __future__ import annotations

# Pythonic enums are pure-Python and always import (no shared library needed).
from .enums import (  # noqa: F401
    Activation,
    DType,
    MemcpyKind,
    QuantMode,
    Status,
)

# Best-available-backend dispatch. This is import-safe with no torch / CUDA /
# shared library — it only routes (lazily) to whatever is installed, falling
# back to the kernel-set C ABI. Imported up front so ``kernel_set.dispatch`` is
# available even on hosts where the prebuilt shared library is absent.
from . import dispatch  # noqa: F401

# Everything below this point requires the prebuilt shared library (loaded at
# import time by ``_lib``). On a host without the ``.so``/``.dylib`` (e.g. CPU-
# only CI), that import raises ``OSError``; we degrade gracefully so the package
# (and ``kernel_set.dispatch`` introspection) still imports. The op modules and
# FFI symbols then resolve only once the library is present.
_LIB_AVAILABLE = False
_LIB_IMPORT_ERROR = None
try:
    # Core error / FFI handle.
    from ._lib import KernelSetError, KsDeviceProperties, check, lib  # noqa: F401

    # dtype interop helpers.
    from ._tensor import (  # noqa: F401
        KS_TO_TORCH,
        TORCH_TO_KS,
        dtype_to_ks,
        infer_dtype,
        ks_to_torch_dtype,
        ptr,
        stream_ptr,
    )

    # Op category modules (ks.norm.rms_norm(...), etc.).
    from . import (  # noqa: F401
        activation,
        attention,
        elementwise,
        embedding,
        gemm,
        linear_attn,
        loss,
        moe,
        norm,
        optimizer,
        quant,
        rope,
        runtime,
        sampling,
        ssm,
    )

    from .gemm import gemm_nvfp4  # noqa: F401
    from .quant import quantize_nvfp4, repack_int4  # noqa: F401

    # Frequently-used runtime functions promoted to the top level.
    from .runtime import (  # noqa: F401
        Stream,
        backend_name,
        device_count,
        dtype_name,
        dtype_size_bits,
        get_device,
        get_device_properties,
        set_device,
        stream_synchronize,
        version,
    )

    _LIB_AVAILABLE = True
except OSError as exc:  # shared library not located/loadable on this host
    _LIB_IMPORT_ERROR = exc


def lib_available() -> bool:
    """True if the prebuilt kernel-set shared library loaded at import time.

    When False, :mod:`kernel_set.dispatch` still works (routing to installed
    industry providers), but the kernel-set C-ABI fallback path is unavailable
    and the per-op FFI modules (``norm``, ``gemm``, ...) are not imported.
    """
    return _LIB_AVAILABLE


def lib_version() -> str:
    """Version string reported by the loaded shared library."""
    if not _LIB_AVAILABLE:
        raise RuntimeError(
            "kernel-set shared library not loaded: " + str(_LIB_IMPORT_ERROR))
    return version()


__version__ = "0.1.0"

__all__ = [
    "__version__",
    "lib",
    "lib_version",
    "lib_available",
    # best-available-backend dispatch
    "dispatch",
    # errors / interop
    "KernelSetError",
    "KsDeviceProperties",
    "check",
    "dtype_to_ks",
    "ks_to_torch_dtype",
    "infer_dtype",
    "ptr",
    "stream_ptr",
    "TORCH_TO_KS",
    "KS_TO_TORCH",
    # enums
    "Status",
    "DType",
    "Activation",
    "QuantMode",
    "MemcpyKind",
    # runtime convenience
    "Stream",
    "version",
    "backend_name",
    "dtype_name",
    "dtype_size_bits",
    "device_count",
    "set_device",
    "get_device",
    "get_device_properties",
    "stream_synchronize",
    # op modules
    "runtime",
    "norm",
    "activation",
    "attention",
    "gemm",
    "moe",
    "rope",
    "quant",
    "sampling",
    "embedding",
    "elementwise",
    "loss",
    "optimizer",
    "ssm",
    "gemm_nvfp4",
    "quantize_nvfp4",
    "repack_int4",
]
