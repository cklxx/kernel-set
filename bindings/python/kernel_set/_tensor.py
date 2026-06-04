"""Tensor / device-pointer interop helpers shared by all op modules.

These helpers let every wrapper accept *either*:

* a ``torch.Tensor`` (CUDA tensor) — pointers come from ``.data_ptr()`` and the
  stream defaults to ``torch.cuda.current_stream().cuda_stream``; or
* a raw integer device pointer (``int``) — for callers using their own runtime
  (cupy, numba, the ``kernel_set.runtime`` allocator, plain CUDA, ...).

``torch`` is imported lazily so the package imports fine without it; only the
torch-specific code paths require it.

The public surface here is small and used internally by the op modules:

* :func:`ptr`           — resolve any accepted tensor-like to an ``int`` address.
* :func:`stream_ptr`    — resolve a stream argument to an ``int`` (0 = default).
* :func:`dtype_to_ks`   — map a ``torch.dtype`` to a ``ks_dtype_t`` int.
* :func:`ks_to_torch_dtype` — the inverse mapping.
* :func:`infer_dtype`   — pick the ``ks_dtype_t`` for a tensor-like + explicit
  override.
"""

from __future__ import annotations

from typing import Optional, Union

from . import _lib

__all__ = [
    "ptr",
    "stream_ptr",
    "dtype_to_ks",
    "ks_to_torch_dtype",
    "infer_dtype",
    "TORCH_TO_KS",
    "KS_TO_TORCH",
    "TensorLike",
]

# A value usable where the ABI wants a device pointer.
TensorLike = Union[int, "object", None]

_torch = None
_torch_checked = False


def _get_torch():
    """Import torch lazily; return the module or ``None`` if unavailable."""
    global _torch, _torch_checked
    if not _torch_checked:
        _torch_checked = True
        try:
            import torch  # type: ignore

            _torch = torch
        except Exception:  # pragma: no cover - torch optional
            _torch = None
    return _torch


def _is_torch_tensor(obj) -> bool:
    torch = _get_torch()
    return torch is not None and isinstance(obj, torch.Tensor)


# ---------------------------------------------------------------------------
# dtype mapping  (torch <-> ks_dtype_t)
# ---------------------------------------------------------------------------


def _build_dtype_maps():
    """Construct the torch<->ks dtype maps using whatever torch dtypes exist."""
    torch = _get_torch()
    t2k: dict = {}
    k2t: dict = {}
    if torch is None:
        return t2k, k2t

    # Map only dtypes that exist in the installed torch build. FP8 / int4 are
    # gated behind availability since older torch lacks them.
    pairs = [
        ("float32", _lib.KS_DTYPE_F32),
        ("float16", _lib.KS_DTYPE_F16),
        ("bfloat16", _lib.KS_DTYPE_BF16),
        ("float64", _lib.KS_DTYPE_F64),
        ("int64", _lib.KS_DTYPE_I64),
        ("int32", _lib.KS_DTYPE_I32),
        ("int8", _lib.KS_DTYPE_I8),
        ("uint8", _lib.KS_DTYPE_U8),
        # FP8 (torch>=2.1)
        ("float8_e4m3fn", _lib.KS_DTYPE_F8E4M3),
        ("float8_e5m2", _lib.KS_DTYPE_F8E5M2),
    ]
    for attr, ks in pairs:
        tdt = getattr(torch, attr, None)
        if tdt is not None:
            t2k[tdt] = ks
            # First torch dtype wins as the canonical inverse mapping.
            k2t.setdefault(ks, tdt)
    return t2k, k2t


TORCH_TO_KS, KS_TO_TORCH = _build_dtype_maps()


def _ensure_dtype_maps() -> None:
    """(Re)build the dtype maps if torch was imported after module load."""
    global TORCH_TO_KS, KS_TO_TORCH
    if not TORCH_TO_KS:
        TORCH_TO_KS, KS_TO_TORCH = _build_dtype_maps()


def dtype_to_ks(torch_dtype) -> int:
    """Map a ``torch.dtype`` to a ``ks_dtype_t`` integer constant."""
    _ensure_dtype_maps()
    try:
        return TORCH_TO_KS[torch_dtype]
    except KeyError:
        raise ValueError(
            f"torch dtype {torch_dtype!r} has no kernel_set equivalent"
        ) from None


def ks_to_torch_dtype(ks_dtype: int):
    """Map a ``ks_dtype_t`` integer back to a ``torch.dtype``."""
    _ensure_dtype_maps()
    try:
        return KS_TO_TORCH[ks_dtype]
    except KeyError:
        name = _lib.lib.ks_dtype_name(ks_dtype)
        name = name.decode() if name else str(ks_dtype)
        raise ValueError(
            f"ks_dtype {name} (={ks_dtype}) has no torch equivalent"
        ) from None


# ---------------------------------------------------------------------------
# pointer / stream resolution
# ---------------------------------------------------------------------------


def ptr(obj: TensorLike, *, name: str = "tensor", allow_none: bool = True) -> int:
    """Resolve ``obj`` to a raw integer device pointer.

    Accepts a torch CUDA tensor, a raw ``int`` address, or ``None`` (returns 0
    when ``allow_none``). For torch tensors this validates that the tensor lives
    on a CUDA device and is contiguous (kernels assume contiguous layout).
    """
    if obj is None:
        if allow_none:
            return 0
        raise ValueError(f"{name} must not be None")
    if isinstance(obj, int):
        return obj
    if _is_torch_tensor(obj):
        if not obj.is_cuda:
            raise ValueError(
                f"{name} must be a CUDA tensor (got device {obj.device}); "
                "kernel_set operates on device memory"
            )
        if not obj.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous; call .contiguous() first"
            )
        return obj.data_ptr()
    # Anything exposing data_ptr() (e.g. some array libs) is accepted too.
    dp = getattr(obj, "data_ptr", None)
    if callable(dp):
        return int(dp())
    raise TypeError(
        f"{name} must be a torch.Tensor, an int device pointer, or expose "
        f"data_ptr(); got {type(obj).__name__}"
    )


def stream_ptr(stream: TensorLike) -> int:
    """Resolve a stream argument to a raw integer (0 = default stream).

    Accepts ``None``/``0`` (default stream), a raw ``int``, a
    ``torch.cuda.Stream``, or the special string ``"current"`` to use
    ``torch.cuda.current_stream().cuda_stream``.
    """
    if stream is None:
        return 0
    if isinstance(stream, int):
        return stream
    if stream == "current":
        torch = _get_torch()
        if torch is None:
            raise RuntimeError("stream='current' requires torch to be installed")
        return torch.cuda.current_stream().cuda_stream
    torch = _get_torch()
    if torch is not None and isinstance(stream, torch.cuda.Stream):
        return stream.cuda_stream
    # Fall back to a cuda_stream attribute if present.
    cs = getattr(stream, "cuda_stream", None)
    if cs is not None:
        return int(cs)
    raise TypeError(
        f"stream must be None, an int, 'current', or a torch.cuda.Stream; "
        f"got {type(stream).__name__}"
    )


def default_stream(stream: TensorLike, reference: TensorLike = None) -> int:
    """Resolve a stream, defaulting to torch's current stream for tensors.

    If ``stream`` is explicitly provided (not ``None``) it is resolved via
    :func:`stream_ptr`. Otherwise, when ``reference`` is a torch CUDA tensor we
    default to ``torch.cuda.current_stream()`` so that kernel launches are
    correctly ordered against the caller's torch work; for raw-pointer callers
    we fall back to the default stream (0).
    """
    if stream is not None:
        return stream_ptr(stream)
    torch = _get_torch()
    if torch is not None and _is_torch_tensor(reference) and reference.is_cuda:
        return torch.cuda.current_stream(reference.device).cuda_stream
    return 0


def infer_dtype(reference: TensorLike, dtype: Optional[int]) -> int:
    """Determine the ``ks_dtype_t`` for an op.

    ``dtype`` (an explicit ``ks_dtype_t`` int) takes precedence. Otherwise the
    dtype is inferred from a torch tensor ``reference``. Raw-pointer callers
    must pass ``dtype`` explicitly.
    """
    if dtype is not None:
        return dtype
    if _is_torch_tensor(reference):
        return dtype_to_ks(reference.dtype)
    raise ValueError(
        "dtype could not be inferred from a raw pointer; pass dtype=ks.DType.*"
    )
