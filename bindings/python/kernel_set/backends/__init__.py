"""Backend provider table + availability probing for the best-backend dispatch.

This subpackage is import-safe with no torch / CUDA / kernel-set shared library:
importing it only pulls in stdlib + the curated provider table. Heavy provider
libraries are imported lazily, inside the per-provider call adapters and
availability probes, never at import time.
"""

from __future__ import annotations

from ._probe import (  # noqa: F401
    GPU_SM,
    arch_ok,
    can_import,
    detect_sm,
    dtype_ok,
    gpu_to_sm,
    normalize_dtype,
    resolve_sm,
)
from ._registry import (  # noqa: F401
    KERNEL_SET,
    OP_ORDER,
    OPS,
    Op,
    Provider,
)

__all__ = [
    "OPS",
    "OP_ORDER",
    "Op",
    "Provider",
    "KERNEL_SET",
    "GPU_SM",
    "gpu_to_sm",
    "detect_sm",
    "resolve_sm",
    "normalize_dtype",
    "dtype_ok",
    "can_import",
    "arch_ok",
]
