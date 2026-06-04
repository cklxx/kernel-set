"""Mixture-of-Experts routing and grouped expert compute (``moe.h``).

Pipeline: gate -> (sort/align) -> permute tokens by expert -> grouped GEMM ->
unpermute with routing weights. Each helper covers one stage.

Index outputs (``out_indices``, ``sorted_token_ids``, ``expert_offsets``) are
int32 device buffers; routing weights are fp32 device buffers. torch tensors
passed for these must already be the matching dtype.
"""

from __future__ import annotations

from ctypes import POINTER, c_float, c_int32, cast
from typing import Optional

from ._lib import check, lib
from ._tensor import TensorLike, default_stream, infer_dtype, ptr

__all__ = [
    "gate_softmax_topk",
    "gate_sigmoid_group_topk",
    "compute_permutation",
    "permute",
    "unpermute",
    "grouped_gemm",
]

_I32P = POINTER(c_int32)
_F32P = POINTER(c_float)


def _i32(obj, *, name, allow_none=False):
    raw = ptr(obj, name=name, allow_none=allow_none)
    return cast(raw, _I32P) if raw else _I32P()


def _f32(obj, *, name, allow_none=False):
    raw = ptr(obj, name=name, allow_none=allow_none)
    return cast(raw, _F32P) if raw else _F32P()


def gate_softmax_topk(
    out_weights: TensorLike,
    out_indices: TensorLike,
    logits: TensorLike,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    renormalize: bool = True,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """Softmax over experts then top-k select.

    logits: ``[num_tokens, num_experts]``; out_weights: ``[num_tokens, top_k]``
    fp32; out_indices: ``[num_tokens, top_k]`` int32. Returns ``(out_weights,
    out_indices)``.
    """
    dt = infer_dtype(logits, dtype)
    check(
        lib.ks_moe_gate_softmax_topk(
            _f32(out_weights, name="out_weights"),
            _i32(out_indices, name="out_indices"),
            ptr(logits, name="logits"),
            int(num_tokens), int(num_experts), int(top_k),
            1 if renormalize else 0, dt, default_stream(stream, logits),
        ),
        "ks_moe_gate_softmax_topk",
    )
    return out_weights, out_indices


def gate_sigmoid_group_topk(
    out_weights: TensorLike,
    out_indices: TensorLike,
    logits: TensorLike,
    num_tokens: int,
    num_experts: int,
    n_group: int,
    topk_group: int,
    top_k: int,
    correction_bias: TensorLike = None,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
):
    """Sigmoid + group-limited top-k gating (DeepSeek-V3 style).

    ``correction_bias`` may be ``None``. Returns ``(out_weights, out_indices)``.
    """
    dt = infer_dtype(logits, dtype)
    check(
        lib.ks_moe_gate_sigmoid_group_topk(
            _f32(out_weights, name="out_weights"),
            _i32(out_indices, name="out_indices"),
            ptr(logits, name="logits"),
            ptr(correction_bias, name="correction_bias"),
            int(num_tokens), int(num_experts),
            int(n_group), int(topk_group), int(top_k),
            1 if renormalize else 0, float(routed_scaling_factor), dt,
            default_stream(stream, logits),
        ),
        "ks_moe_gate_sigmoid_group_topk",
    )
    return out_weights, out_indices


def compute_permutation(
    sorted_token_ids: TensorLike,
    expert_offsets: TensorLike,
    topk_indices: TensorLike,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    stream: TensorLike = None,
):
    """Build the permutation that groups tokens by expert.

    topk_indices: ``[num_tokens, top_k]`` int32; sorted_token_ids:
    ``[num_tokens*top_k]`` int32; expert_offsets: ``[num_experts+1]`` int32 (CSR
    boundaries). Returns ``(sorted_token_ids, expert_offsets)``.
    """
    check(
        lib.ks_moe_compute_permutation(
            _i32(sorted_token_ids, name="sorted_token_ids"),
            _i32(expert_offsets, name="expert_offsets"),
            _i32(topk_indices, name="topk_indices"),
            int(num_tokens), int(num_experts), int(top_k),
            default_stream(stream, topk_indices),
        ),
        "ks_moe_compute_permutation",
    )
    return sorted_token_ids, expert_offsets


def permute(
    permuted: TensorLike,
    input: TensorLike,
    sorted_token_ids: TensorLike,
    num_tokens: int,
    top_k: int,
    hidden: int,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Gather rows of ``input`` ``[num_tokens, hidden]`` into ``permuted``
    ``[num_tokens*top_k, hidden]`` following ``sorted_token_ids``."""
    dt = infer_dtype(input, dtype)
    check(
        lib.ks_moe_permute(
            ptr(permuted, name="permuted"), ptr(input, name="input"),
            _i32(sorted_token_ids, name="sorted_token_ids"),
            int(num_tokens), int(top_k), int(hidden), dt,
            default_stream(stream, input),
        ),
        "ks_moe_permute",
    )
    return permuted


def unpermute(
    out: TensorLike,
    permuted: TensorLike,
    sorted_token_ids: TensorLike,
    routing_weights: TensorLike,
    num_tokens: int,
    top_k: int,
    hidden: int,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Scatter-add expert outputs back, weighted by routing weights:
    ``out[token] = sum_k weight[token,k] * permuted[pos(token,k)]``."""
    dt = infer_dtype(permuted, dtype)
    check(
        lib.ks_moe_unpermute(
            ptr(out, name="out"), ptr(permuted, name="permuted"),
            _i32(sorted_token_ids, name="sorted_token_ids"),
            _f32(routing_weights, name="routing_weights"),
            int(num_tokens), int(top_k), int(hidden), dt,
            default_stream(stream, permuted),
        ),
        "ks_moe_unpermute",
    )
    return out


def grouped_gemm(
    c: TensorLike,
    a: TensorLike,
    b: TensorLike,
    expert_offsets: TensorLike,
    num_experts: int,
    total_rows: int,
    n: int,
    k: int,
    dtype: Optional[int] = None,
    stream: TensorLike = None,
) -> TensorLike:
    """Grouped GEMM: for each expert ``e``,
    ``C[off_e:off_{e+1}] = A[off_e:off_{e+1}] @ B_e``.

    a: ``[total_rows, k]``; b: ``[num_experts, k, n]``; expert_offsets:
    ``[num_experts+1]`` int32; c: ``[total_rows, n]``.
    """
    dt = infer_dtype(a, dtype)
    check(
        lib.ks_moe_grouped_gemm(
            ptr(c, name="c"), ptr(a, name="a"), ptr(b, name="b"),
            _i32(expert_offsets, name="expert_offsets"),
            int(num_experts), int(total_rows), int(n), int(k), dt,
            default_stream(stream, a),
        ),
        "ks_moe_grouped_gemm",
    )
    return c
