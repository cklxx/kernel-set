"""Backend provider table + availability probing for the best-backend dispatch.

This subpackage powers **Tier 2** of kernel-set routing — runtime
*best-available-provider* selection (see ``docs/ROUTING.md``). It is the data +
gates behind ``kernel_set.dispatch``:

* ``_registry`` — the curated, rank-ordered provider table (derived from
  ``providers/registry.json``) with a lazy call adapter per provider.
* ``_probe``    — the gates: import-availability + arch/dtype support. GPU
  SM/caps and dtype aliases come from ``models/gpu_caps.json`` (the source of
  truth it shares with the Tier-1 planner ``models/select.py``).

Import-safe with no torch / CUDA / kernel-set shared library: importing only
pulls in stdlib + the curated provider table. Heavy provider libraries are
imported lazily, inside the per-provider call adapters and availability probes,
never at import time.
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
    SGL_KERNEL,
    Op,
    Provider,
)

__all__ = [
    "OPS",
    "OP_ORDER",
    "Op",
    "Provider",
    "KERNEL_SET",
    "SGL_KERNEL",
    "GPU_SM",
    "gpu_to_sm",
    "detect_sm",
    "resolve_sm",
    "normalize_dtype",
    "dtype_ok",
    "can_import",
    "arch_ok",
]
