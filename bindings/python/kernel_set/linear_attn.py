"""Linear-attention / gated-delta / RWKV token-mixers (portable C-ABI fallback).

Thin ctypes wrappers over ``ks_gated_delta_rule`` / ``ks_gated_linear_attn`` /
``ks_rwkv_wkv7`` (see ``include/kernel_set/linear_attn.h``). These are the
correctness-first O(T) recurrent fallback; dispatch routes to FLA
(flash-linear-attention) when installed. Tensors are seq-first ``[B,T,H,D]``.
"""

from __future__ import annotations

from ctypes import POINTER, c_float, cast
from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = ["gated_delta_rule", "gated_linear_attn", "rwkv_wkv7"]

_F32P = POINTER(c_float)


def _f32(obj: TensorLike, *, name: str):
    raw = ptr(obj, name=name, allow_none=True)
    return cast(raw, _F32P) if raw else _F32P()


def gated_delta_rule(
    out: TensorLike,
    q: TensorLike,
    k: TensorLike,
    v: TensorLike,
    g: TensorLike,
    beta: TensorLike,
    batch: int,
    seqlen: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    g_is_vector: int = 0,
    use_qk_l2norm: int = 0,
    scale: float = 0.0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Gated delta-rule (gated DeltaNet / KDA). q,k ``[B,T,H,K]``; v,out
    ``[B,T,H,V]``; beta ``[B,T,H]``; g ``[B,T,H]`` (scalar) or ``[B,T,H,K]``
    (``g_is_vector=1``, KDA)."""
    dt = infer_dtype(q, dtype)
    check(
        lib.ks_gated_delta_rule(
            ptr(out, name="out"), ptr(q, name="q"), ptr(k, name="k"),
            ptr(v, name="v"), ptr(g, name="g"), ptr(beta, name="beta"),
            int(batch), int(seqlen), int(heads), int(k_dim), int(v_dim),
            int(g_is_vector), int(use_qk_l2norm), float(scale), dt,
            default_stream(stream, q)),
        "ks_gated_delta_rule",
    )
    return out


def gated_linear_attn(
    out: TensorLike,
    q: TensorLike,
    k: TensorLike,
    v: TensorLike,
    g: TensorLike,
    head_decay: TensorLike,
    batch: int,
    seqlen: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    gate_mode: int = 0,
    scale: float = 0.0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Gated linear attention (GLA / simple-GLA / lightning). ``gate_mode``: 0 =
    data-dependent diagonal ``g [B,T,H,K]``; 1 = scalar ``g [B,T,H]``; 2 = fixed
    per-head slope ``head_decay [H]`` (fp32)."""
    dt = infer_dtype(q, dtype)
    check(
        lib.ks_gated_linear_attn(
            ptr(out, name="out"), ptr(q, name="q"), ptr(k, name="k"),
            ptr(v, name="v"), ptr(g, name="g"),
            _f32(head_decay, name="head_decay"),
            int(batch), int(seqlen), int(heads), int(k_dim), int(v_dim),
            int(gate_mode), float(scale), dt, default_stream(stream, q)),
        "ks_gated_linear_attn",
    )
    return out


def rwkv_wkv7(
    out: TensorLike,
    r: TensorLike,
    w: TensorLike,
    k: TensorLike,
    v: TensorLike,
    a: TensorLike,
    b: TensorLike,
    batch: int,
    seqlen: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    scale: float = 0.0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """RWKV-7 WKV (free DPLR). r,w,k,a,b ``[B,T,H,K]``; v,out ``[B,T,H,V]``."""
    dt = infer_dtype(r, dtype)
    check(
        lib.ks_rwkv_wkv7(
            ptr(out, name="out"), ptr(r, name="r"), ptr(w, name="w"),
            ptr(k, name="k"), ptr(v, name="v"), ptr(a, name="a"),
            ptr(b, name="b"), int(batch), int(seqlen), int(heads), int(k_dim),
            int(v_dim), float(scale), dt, default_stream(stream, r)),
        "ks_rwkv_wkv7",
    )
    return out
