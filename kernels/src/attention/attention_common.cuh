// kernel-set — shared helpers for the fused-attention kernels.
//
// These utilities centralize the conventions that every attention kernel in
// this category relies on:
//   * head_dim is the innermost (contiguous) axis;
//   * GQA/MQA maps query head h -> kv head (h / (num_heads / num_kv_heads));
//   * softmax is accumulated online in fp32 (FlashAttention-2 / FlashDecoding),
//     never materializing the full [seqlen_q, seqlen_k] score matrix;
//   * softmax_scale defaults to 1/sqrt(head_dim) when the caller passes <= 0.
//
// The kernels themselves live in the sibling .cu files and stay in
// namespace ks::attention; this header only holds device-inline helpers and a
// couple of host-side validation utilities.
#pragma once

#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/reduce.cuh"

namespace ks {
namespace attention {

// Largest head_dim we statically size shared-memory tiles for. 64 and 128 are
// the required sizes; 256 leaves head-room for MLA's kv_lora_rank + rope_dim.
constexpr int kMaxHeadDim = 256;

// Resolve the effective softmax scale (default 1/sqrt(head_dim) when <= 0).
KS_HDI float resolve_scale(float scale, int head_dim) {
  if (scale > 0.f) return scale;
  return 1.0f / sqrtf(static_cast<float>(head_dim));
}

// GQA/MQA: number of query heads sharing one kv head.
KS_HDI int gqa_group_size(int num_heads, int num_kv_heads) {
  return num_heads / num_kv_heads;
}

// ---------------------------------------------------------------------------
// Online-softmax running state (FlashAttention-2). `acc` holds the (unnormalized)
// weighted sum of V rows in fp32; `m` is the running row max; `l` the running
// denominator. All updates happen in fp32 regardless of storage dtype.
// ---------------------------------------------------------------------------
struct SoftmaxState {
  float m;  // running max of scores seen so far
  float l;  // running sum of exp(score - m)

  KS_DI void init() {
    m = -3.4028235e38f;  // -FLT_MAX sentinel (matches ks::neg_inf)
    l = 0.f;
  }
};

// Rescale-on-new-max correction factor: given the previous running max `m_prev`
// and a new max `m_new`, returns exp(m_prev - m_new) used to down-weight the
// already-accumulated partial sums.
KS_DI float rescale_factor(float m_prev, float m_new) {
  if (m_prev <= -3.4028235e38f) return 0.f;  // first tile: nothing to rescale
  return __expf(m_prev - m_new);
}

}  // namespace attention
}  // namespace ks
