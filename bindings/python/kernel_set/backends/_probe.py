"""Availability probing + GPU/dtype gating for the best-backend dispatcher.

Tier-2 gates (see ``docs/ROUTING.md``): a provider is *selectable* only when its
library imports AND the device meets its ``min_sm`` AND it supports the dtype.
The GPU SM/caps table and dtype aliases are loaded from the shared
``models/gpu_caps.json`` (the single source of truth, also used by the Tier-1
planner ``models/select.py``), with an inlined fallback for wheel installs.

Everything here is import-safe with no torch / CUDA / kernel-set shared library:
the probes simply report ``unavailable`` (or fall back to the kernel-set C ABI)
when those are missing. Heavy libraries are imported *lazily* — only when a
probe for that specific provider is requested — and the result is cached so the
import cost is paid at most once per process.

Two axes gate a provider:

* **import**     — can the library be imported at all (``import_check``)?
* **arch/dtype** — does the running (or hypothetically named) GPU meet the
  provider's minimum compute capability (``min_sm``) and support the requested
  dtype (``dtypes``)?

A provider is *selectable* only when both pass. The kernel-set C ABI provider is
special-cased as always selectable (it is the portable fallback and its own
runtime decides whether the device/dtype is supported).
"""

from __future__ import annotations

import importlib
import json
import os
from typing import Dict, Iterable, Optional, Tuple

# --------------------------------------------------------------------------- #
# Named-GPU -> compute capability (SM) table + dtype aliases.
#
# Canonical source of truth: ``models/gpu_caps.json`` (shared with
# ``models/select.py`` so the `ksctl backends --gpu h100` path and the
# dispatcher agree on the same SM/caps). When this binding is imported from the
# source tree that JSON is loaded directly; when the package is installed as a
# standalone wheel (no repo `models/` dir) we fall back to the inlined defaults
# below, which are kept identical to the JSON. `sm` is major*10+minor (H100 =
# sm90, L4/RTX4090 = sm89, A100 = sm80, T4 = sm75, Blackwell DC = sm100,
# consumer Blackwell = sm120).
# --------------------------------------------------------------------------- #
_GPU_SM_FALLBACK: Dict[str, int] = {
    "t4": 75,
    "a100": 80, "a100-80gb": 80, "a100-40gb": 80, "a800": 80,
    "a10": 86, "a10g": 86,
    "l4": 89, "ada": 89, "rtx4090": 89, "4090": 89, "rtx-4090": 89,
    "h100": 90, "h800": 90, "h200": 90, "hopper": 90,
    "b200": 100, "b100": 100, "blackwell": 100, "gb200": 100,
    "rtx5090": 120, "5090": 120,
}

_DTYPE_ALIASES_FALLBACK = {
    "float16": "fp16", "half": "fp16", "f16": "fp16",
    "bfloat16": "bf16", "bf16": "bf16",
    "float32": "fp32", "float": "fp32", "f32": "fp32", "tf32": "tf32",
    "float8_e4m3fn": "fp8", "float8_e4m3": "fp8", "fp8_e4m3": "fp8",
    "float8_e5m2": "fp8", "fp8_e5m2": "fp8", "fp8": "fp8",
    "int8": "int8", "i8": "int8", "w8a8": "int8",
    "int4": "int4", "i4": "int4", "w4a16": "int4", "w4a8": "int4",
    "awq": "int4", "gptq": "int4", "nf4": "int4",
    "fp4": "fp4", "nvfp4": "fp4", "mxfp4": "fp4",
}

# Repo layout: bindings/python/kernel_set/backends/_probe.py -> models/gpu_caps.json
_GPU_CAPS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "models", "gpu_caps.json")


# Minimum SM each dtype-CAPABILITY needs (named ``sm_thresholds`` in
# gpu_caps.json). Used by ``dtype_arch_ok`` so a provider whose ``min_sm`` is
# below a dtype's threshold (e.g. FlashInfer min_sm=75 with dtype "fp16, bf16,
# fp8") is STILL not selectable for that dtype on an SM that lacks it (fp8 on
# sm75). Inlined fallback kept identical to the JSON for wheel installs.
_SM_THRESHOLDS_FALLBACK: Dict[str, int] = {
    "fp8": 89, "bf16": 80, "tf32": 80, "int8": 75, "int4": 75,
    "fp4": 100, "wgmma": 90,
}

_SM_FAMILY_ALIASES_FALLBACK: Dict[int, int] = {
    120: 100,
}

# Map a normalized dtype gate token -> the capability key it requires. Tokens
# absent here (fp16/fp32/int8/int4) have no arch threshold.
_DTYPE_CAP_KEY: Dict[str, str] = {
    "bf16": "bf16", "tf32": "tf32", "fp8": "fp8", "fp4": "fp4",
}


def _load_canonical() -> Tuple[Dict[str, int], Dict[str, str], Dict[str, int],
                               Dict[int, int]]:
    """Build (GPU_SM, _DTYPE_ALIASES, SM_THRESHOLDS, SM_FAMILY_ALIASES) from the
    canonical JSON if present, else the inlined fallbacks. Never raises
    (malformed/absent -> fallback)."""
    try:
        with open(_GPU_CAPS_PATH) as f:
            caps = json.load(f)
        gpu_sm: Dict[str, int] = {}
        for gid, g in caps["gpus"].items():
            gpu_sm[gid] = g["sm"]
            for alias in g.get("aliases", []):
                gpu_sm[alias] = g["sm"]
        aliases = {k: v for k, v in caps["probe_dtype_aliases"].items()
                   if not k.startswith("_")}
        thresholds = {k: int(v) for k, v in caps["sm_thresholds"].items()
                      if not k.startswith("_")}
        sm_aliases = {int(k): int(v) for k, v in
                      caps.get("sm_family_aliases", {}).items()
                      if not k.startswith("_")}
        return gpu_sm, aliases, thresholds, sm_aliases
    except Exception:
        return (dict(_GPU_SM_FALLBACK), dict(_DTYPE_ALIASES_FALLBACK),
                dict(_SM_THRESHOLDS_FALLBACK),
                dict(_SM_FAMILY_ALIASES_FALLBACK))


GPU_SM, _DTYPE_ALIASES, _SM_THRESHOLDS, SM_FAMILY_ALIASES = _load_canonical()


def canonical_sm(sm: Optional[int]) -> Optional[int]:
    """Map raw compute capability to the optimal-table capability-family SM."""
    if sm is None:
        return None
    sm_int = int(sm)
    return SM_FAMILY_ALIASES.get(sm_int, sm_int)


def gpu_to_sm(gpu: Optional[str]) -> Optional[int]:
    """Resolve a GPU name (or an explicit ``"sm89"`` / ``"89"`` / ``89``) to an
    SM integer. Returns ``None`` when ``gpu`` is ``None`` (meaning "no arch
    gate" — auto-detect or unknown)."""
    if gpu is None:
        return None
    if isinstance(gpu, int):
        return int(gpu)
    g = str(gpu).strip().lower()
    if not g:
        return None
    if g.startswith("sm"):
        g2 = g[2:].replace("_", "")
        if g2.isdigit():
            return int(g2)
    if g.isdigit():
        return int(g)
    return GPU_SM.get(g)


# --------------------------------------------------------------------------- #
# Auto-detected device SM (when torch is importable with a visible CUDA GPU).
# Cached; safe to call with no torch (returns None).
# --------------------------------------------------------------------------- #
_DETECTED_SM: Optional[int] = None
_DETECTED_DONE = False


def detect_sm() -> Optional[int]:
    """Best-effort compute capability of the current CUDA device, or ``None``
    on a CPU/no-GPU host. Never raises; result cached for the process."""
    global _DETECTED_SM, _DETECTED_DONE
    if _DETECTED_DONE:
        return _DETECTED_SM
    _DETECTED_DONE = True
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            _DETECTED_SM = major * 10 + minor
    except Exception:
        _DETECTED_SM = None
    return _DETECTED_SM


def resolve_sm(gpu: Optional[str]) -> Optional[int]:
    """Resolve the SM to gate against: the explicit ``gpu`` if given, else the
    auto-detected device, else ``None`` (no gate)."""
    if gpu is not None:
        return canonical_sm(gpu_to_sm(gpu))
    return canonical_sm(detect_sm())


# --------------------------------------------------------------------------- #
# dtype normalization. Registry ``dtypes`` strings are free-form ("fp16, bf16,
# fp8 (e4m3/e5m2)"), so we match on substrings of a normalized request. The
# alias map (``_DTYPE_ALIASES``) is loaded above from models/gpu_caps.json
# (``probe_dtype_aliases``), with the inlined fallback for wheel installs.
# --------------------------------------------------------------------------- #
def normalize_dtype(dtype) -> Optional[str]:
    """Normalize a dtype request (str or torch.dtype) to a short token
    (fp16/bf16/fp32/tf32/fp8/int8/int4/fp4) or ``None`` if unknown/absent."""
    if dtype is None:
        return None
    name = str(getattr(dtype, "name", dtype))  # torch.dtype -> "torch.float16"
    name = name.replace("torch.", "").strip().lower()
    return _DTYPE_ALIASES.get(name, name if name else None)


def dtype_ok(requested: Optional[str], supported_str: str) -> bool:
    """Does a provider's free-form ``dtypes`` string cover the requested dtype?
    When no dtype is requested we don't gate on it."""
    req = normalize_dtype(requested)
    if req is None:
        return True
    s = supported_str.lower()
    # Map the normalized token back to substrings that appear in registry text.
    needles = {
        "fp16": ("fp16", "float16", "f16", "half"),
        "bf16": ("bf16", "bfloat16"),
        "fp32": ("fp32", "float32", "tf32", "f32"),
        "tf32": ("tf32",),
        "fp8": ("fp8", "e4m3", "e5m2", "float8"),
        "int8": ("int8", "i8", "w8a8"),
        "int4": ("int4", "i4", "w4a16", "w4a8", "uint4", "awq",
                 "gptq", "nf4"),
        "fp4": ("fp4", "nvfp4", "mxfp4"),
    }.get(req, (req,))
    return any(n in s for n in needles)


# --------------------------------------------------------------------------- #
# Import probing. ``import_check`` is a registry-style snippet like
# "from flash_attn import flash_attn_func". We import modules and, for
# ``from ... import name`` snippets, verify the named attribute/submodule exists;
# we never call provider functions.
# --------------------------------------------------------------------------- #
_IMPORT_CACHE: Dict[str, bool] = {}


def _module_names(import_check: str) -> Iterable[str]:
    """Extract the top-level importable module name(s) from a registry
    ``import_check`` snippet, without executing it."""
    mods = set()
    for stmt in import_check.split(";"):
        stmt = stmt.strip()
        if stmt.startswith("from "):
            mod = stmt[len("from "):].split(" import ")[0].strip()
            if mod:
                mods.add(mod.split(".")[0] if "." in mod else mod)
                mods.add(mod)
        elif stmt.startswith("import "):
            mod = stmt[len("import "):].split(" as ")[0].split(",")[0].strip()
            if mod:
                mods.add(mod.split(".")[0])
                mods.add(mod)
    return mods


def _import_requirements(import_check: str):
    """Parse import snippets into ``(module, attr-or-None)`` requirements.

    ``from pkg.mod import fn`` means ``pkg.mod`` must import and expose ``fn``
    either as an attribute or as an importable submodule. ``import pkg.mod`` only
    requires that module. Plain module paths are accepted for direct calls.
    """
    reqs = []
    aliases = {}
    for stmt in str(import_check).split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        if stmt.startswith("from ") and " import " in stmt:
            mod, names = stmt[len("from "):].split(" import ", 1)
            mod = mod.strip()
            for raw in names.split(","):
                parts = raw.strip().split(" as ", 1)
                name = parts[0].strip()
                if name and name != "*":
                    reqs.append((mod, name))
                    if len(parts) == 2:
                        alias = parts[1].strip()
                        if alias:
                            aliases[alias] = (mod, name)
        elif stmt.startswith("import "):
            for raw in stmt[len("import "):].split(","):
                parts = raw.strip().split(" as ", 1)
                mod = parts[0].strip()
                if mod:
                    reqs.append((mod, None))
                    aliases.setdefault(mod.split(".", 1)[0],
                                       (mod.split(".", 1)[0], None))
                    if len(parts) == 2:
                        alias = parts[1].strip()
                        if alias:
                            aliases[alias] = (mod, None)
        elif "." in stmt and stmt.split(".", 1)[0] in aliases:
            alias, attr = stmt.split(".", 1)
            mod, base_attr = aliases[alias]
            if base_attr is None:
                reqs.append((mod, attr))
            else:
                reqs.append((mod, f"{base_attr}.{attr}"))
        elif stmt:
            reqs.append((stmt, None))
    return reqs


def _has_import_requirement(mod: str, attr: Optional[str]) -> bool:
    try:
        module = importlib.import_module(mod)
    except Exception:
        return False
    if attr is None:
        return True
    cur = module
    for part in attr.split("."):
        try:
            submod_name = f"{getattr(cur, '__name__', mod)}.{part}"
            cur = importlib.import_module(submod_name)
            continue
        except Exception:
            pass
        if not hasattr(cur, part):
            return False
        cur = getattr(cur, part)
    return True


def can_import(module_or_check: str) -> bool:
    """True if the library behind a registry ``import_check`` (or a plain module
    path) can be imported and any named ``from`` attributes exist. Cached; never
    raises."""
    if module_or_check in _IMPORT_CACHE:
        return _IMPORT_CACHE[module_or_check]
    reqs = _import_requirements(module_or_check)
    ok = bool(reqs) and all(_has_import_requirement(mod, attr)
                            for mod, attr in reqs)
    _IMPORT_CACHE[module_or_check] = ok
    return ok


def arch_ok(min_sm: int, sm: Optional[int]) -> bool:
    """Does the (resolved) device SM meet the provider's minimum? When the SM is
    unknown (``None``) we *cannot* prove the gate fails, so we treat the import
    probe as the sole signal and let arch be considered satisfied. (On a real
    GPU host ``sm`` is known and gates correctly.)"""
    if sm is None:
        return True
    return sm >= min_sm


def dtype_arch_ok(requested, sm: Optional[int]) -> bool:
    """Capability-aware dtype gate: does ``sm`` actually support the requested
    *dtype itself* (independent of any provider's ``min_sm``)? fp8 needs sm89+,
    bf16/tf32 need sm80+, fp4 needs sm100+. A provider's ``dtypes`` string +
    low ``min_sm`` is NOT sufficient — e.g. FlashInfer (min_sm=75, dtypes "...,
    fp8") must NOT be selectable for fp8 on sm75 because the *hardware* has no
    fp8 Tensor Cores. When the SM is unknown (``None``) we cannot prove the gate
    fails, so we don't gate (the import/arch probes remain the signal)."""
    req = normalize_dtype(requested)
    if req is None or sm is None:
        return True
    cap = _DTYPE_CAP_KEY.get(req)
    if cap is None:
        return True
    return int(sm) >= int(_SM_THRESHOLDS.get(cap, 0))
