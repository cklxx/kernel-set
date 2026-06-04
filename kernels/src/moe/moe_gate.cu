// kernel-set — MoE gating: softmax+top-k and DeepSeek-V3 sigmoid group top-k.
//
// One CUDA block routes one token. The block cooperatively loads the token's
// `num_experts` logits into shared memory (fp32), applies the scoring function
// (softmax or per-expert sigmoid), runs group-limited top-k selection, and
// writes the selected expert ids + routing weights. All math accumulates in
// fp32 regardless of the storage dtype, mirroring the norm reference kernels.
//
// Modeled on vLLM / SGLang fused-MoE gating (`fused_topk`,
// `biased_grouped_topk` / DeepSeek-V3 `grouped_topk`):
//   * softmax variant: softmax over experts -> top-k -> optional renorm.
//   * sigmoid group variant: per-expert sigmoid scores, an optional
//     correction_bias added ONLY for group/expert *selection* (the returned
//     weight uses the unbiased sigmoid, as in DeepSeek-V3), group-limited
//     top-k over n_group/topk_group, then top-k over the surviving experts,
//     renormalize the chosen sigmoid weights, and scale by
//     routed_scaling_factor.
#include "kernel_set/moe.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"
#include "moe/moe_common.cuh"

namespace ks {
namespace moe {

constexpr int kGateBlock = 128;

// Upper bound on experts a single block can buffer in shared memory. DeepSeek-V3
// uses 256 routed experts; classic Mixtral uses 8. 1024 covers foreseeable
// configs while keeping shared-memory use modest (a few KB of fp32).
constexpr int kMaxExperts = 1024;
// Upper bound on top_k (DeepSeek-V3 selects 8). Kept tiny for register arrays.
constexpr int kMaxTopK = 32;
// Upper bound on expert groups (DeepSeek-V3 uses 8).
constexpr int kMaxGroups = 64;

// -------------------------------------------------------------------------
// Softmax over experts, then top-k selection.
// -------------------------------------------------------------------------
template <typename scalar_t>
KS_GLOBAL void gate_softmax_topk_kernel(float* __restrict__ out_weights,
                                        int32_t* __restrict__ out_indices,
                                        const scalar_t* __restrict__ logits,
                                        int num_experts, int top_k,
                                        int renormalize) {
  const int64_t token = blockIdx.x;
  const scalar_t* row = logits + token * num_experts;

  // Shared buffer holds the fp32 softmax probabilities for this token.
  __shared__ float prob[kMaxExperts];
  __shared__ float red[kGateBlock / KS_WARP_SIZE];
  __shared__ float s_max;
  __shared__ float s_sum;

  // 1) row max (for numerically-stable softmax).
  float local_max = -3.4028235e38f;
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
    const float v = to_float(row[e]);
    prob[e] = v;  // stash raw logit; overwritten with prob below
    local_max = v > local_max ? v : local_max;
  }
  local_max = block_reduce_max(local_max, red);
  if (threadIdx.x == 0) s_max = local_max;
  __syncthreads();
  const float row_max = s_max;

  // 2) exp + sum.
  float local_sum = 0.f;
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
    const float p = __expf(prob[e] - row_max);
    prob[e] = p;
    local_sum += p;
  }
  __syncthreads();
  local_sum = block_reduce_sum(local_sum, red);
  if (threadIdx.x == 0) s_sum = local_sum;
  __syncthreads();
  const float inv_sum = 1.f / s_sum;

  // 3) normalize to probabilities.
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
    prob[e] = prob[e] * inv_sum;
  __syncthreads();

  // 4) iterative top-k argmax. Done by a single thread: top_k is tiny and the
  //    selection is inherently sequential (each pick masks the previous max).
  if (threadIdx.x == 0) {
    float chosen_w[kMaxTopK];
    int chosen_i[kMaxTopK];
    float renorm_sum = 0.f;
    for (int k = 0; k < top_k; ++k) {
      float best = -3.4028235e38f;
      int best_e = -1;
      for (int e = 0; e < num_experts; ++e) {
        const float p = prob[e];
        // Strict-greater with index tie-break keeps the lowest expert id on
        // ties, matching torch.topk's stable ordering.
        if (p > best) {
          best = p;
          best_e = e;
        }
      }
      chosen_w[k] = best;
      chosen_i[k] = best_e;
      renorm_sum += best;
      prob[best_e] = -3.4028235e38f;  // mask so it is not re-selected
    }
    const float scale = (renormalize && renorm_sum > 0.f) ? (1.f / renorm_sum)
                                                          : 1.f;
    float* ow = out_weights + token * top_k;
    int32_t* oi = out_indices + token * top_k;
    for (int k = 0; k < top_k; ++k) {
      ow[k] = chosen_w[k] * scale;
      oi[k] = chosen_i[k];
    }
  }
}

// -------------------------------------------------------------------------
// DeepSeek-V3 style: per-expert sigmoid, group-limited top-k.
// -------------------------------------------------------------------------
// experts are partitioned into n_group contiguous groups of (num_experts /
// n_group) experts. Each group's score = sum of its top-2 (biased) expert
// scores; the topk_group highest-scoring groups survive, the rest are masked.
// Then top_k experts are picked across the survivors. Returned weights use the
// *unbiased* sigmoid score, renormalized, then scaled by routed_scaling_factor.
template <typename scalar_t>
KS_GLOBAL void gate_sigmoid_group_topk_kernel(
    float* __restrict__ out_weights, int32_t* __restrict__ out_indices,
    const scalar_t* __restrict__ logits,
    const scalar_t* __restrict__ correction_bias, int num_experts, int n_group,
    int topk_group, int top_k, int renormalize, float routed_scaling_factor) {
  const int64_t token = blockIdx.x;
  const scalar_t* row = logits + token * num_experts;

  // s_score: unbiased sigmoid score (returned weight source).
  // s_sel:   biased score used for group + expert selection.
  __shared__ float s_score[kMaxExperts];
  __shared__ float s_sel[kMaxExperts];

  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
    const float p = sigmoidf(to_float(row[e]));
    s_score[e] = p;
    float sel = p;
    if (correction_bias != nullptr) sel += to_float(correction_bias[e]);
    s_sel[e] = sel;
  }
  __syncthreads();

  // Selection logic is sequential and cheap (groups/top_k are tiny), so a single
  // thread performs it after the cooperative load.
  if (threadIdx.x == 0) {
    const int experts_per_group = num_experts / n_group;

    // 1) per-group score = sum of the group's top-2 biased expert scores
    //    (DeepSeek-V3 uses top-2; falls back to top-1 if group has <2 experts).
    float group_score[kMaxGroups];
    for (int g = 0; g < n_group; ++g) {
      const int base = g * experts_per_group;
      float m1 = -3.4028235e38f, m2 = -3.4028235e38f;
      for (int j = 0; j < experts_per_group; ++j) {
        const float v = s_sel[base + j];
        if (v > m1) {
          m2 = m1;
          m1 = v;
        } else if (v > m2) {
          m2 = v;
        }
      }
      float gs = m1;
      if (experts_per_group >= 2) gs += m2;
      group_score[g] = gs;
    }

    // 2) keep the topk_group highest-scoring groups; mask experts of the rest.
    //    Build a boolean keep-mask by iteratively selecting the best group.
    bool group_keep[kMaxGroups];
    for (int g = 0; g < n_group; ++g) group_keep[g] = false;
    float tmp_gs[kMaxGroups];
    for (int g = 0; g < n_group; ++g) tmp_gs[g] = group_score[g];
    const int kg = topk_group < n_group ? topk_group : n_group;
    for (int t = 0; t < kg; ++t) {
      float best = -3.4028235e38f;
      int best_g = -1;
      for (int g = 0; g < n_group; ++g) {
        if (tmp_gs[g] > best) {
          best = tmp_gs[g];
          best_g = g;
        }
      }
      if (best_g < 0) break;
      group_keep[best_g] = true;
      tmp_gs[best_g] = -3.4028235e38f;
    }

    // 3) top_k over experts that belong to a surviving group, ranked by the
    //    biased selection score. Mask others to -inf.
    float chosen_w[kMaxTopK];  // unbiased sigmoid weight of the pick
    int chosen_i[kMaxTopK];
    float renorm_sum = 0.f;
    for (int k = 0; k < top_k; ++k) {
      float best = -3.4028235e38f;
      int best_e = -1;
      for (int e = 0; e < num_experts; ++e) {
        const int g = e / experts_per_group;
        if (!group_keep[g]) continue;
        const float v = s_sel[e];
        if (v > best) {
          best = v;
          best_e = e;
        }
      }
      if (best_e < 0) {
        // Fewer eligible experts than top_k: pad with id 0, zero weight.
        chosen_w[k] = 0.f;
        chosen_i[k] = 0;
        continue;
      }
      chosen_w[k] = s_score[best_e];  // returned weight is the UNBIASED sigmoid
      chosen_i[k] = best_e;
      renorm_sum += s_score[best_e];
      s_sel[best_e] = -3.4028235e38f;  // mask from further selection
    }

    const float norm = (renormalize && renorm_sum > 0.f) ? (1.f / renorm_sum)
                                                         : 1.f;
    float* ow = out_weights + token * top_k;
    int32_t* oi = out_indices + token * top_k;
    for (int k = 0; k < top_k; ++k) {
      ow[k] = chosen_w[k] * norm * routed_scaling_factor;
      oi[k] = chosen_i[k];
    }
  }
}

}  // namespace moe
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_moe_gate_softmax_topk(float* out_weights, int32_t* out_indices,
                                     const void* logits, int64_t num_tokens,
                                     int num_experts, int top_k,
                                     int renormalize, ks_dtype_t dtype,
                                     ks_stream_t stream) {
  KS_CHECK_PTR(out_weights);
  KS_CHECK_PTR(out_indices);
  KS_CHECK_PTR(logits);
  if (num_tokens <= 0 || num_experts <= 0 || top_k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_moe_gate_softmax_topk: bad shape");
  if (top_k > num_experts)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_moe_gate_softmax_topk: top_k > num_experts");
  if (num_experts > moe::kMaxExperts)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_moe_gate_softmax_topk: num_experts exceeds kMaxExperts");
  if (top_k > moe::kMaxTopK)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_moe_gate_softmax_topk: top_k exceeds kMaxTopK");

  const dim3 grid(static_cast<unsigned>(num_tokens));
  const dim3 block(moe::kGateBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_moe_gate_softmax_topk", {
    moe::gate_softmax_topk_kernel<scalar_t><<<grid, block, 0, s>>>(
        out_weights, out_indices, static_cast<const scalar_t*>(logits),
        num_experts, top_k, renormalize);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_moe_gate_sigmoid_group_topk(
    float* out_weights, int32_t* out_indices, const void* logits,
    const void* correction_bias, int64_t num_tokens, int num_experts,
    int n_group, int topk_group, int top_k, int renormalize,
    float routed_scaling_factor, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out_weights);
  KS_CHECK_PTR(out_indices);
  KS_CHECK_PTR(logits);
  // correction_bias may be NULL (optional).
  if (num_tokens <= 0 || num_experts <= 0 || top_k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_moe_gate_sigmoid_group_topk: bad shape");
  if (n_group <= 0 || topk_group <= 0 || topk_group > n_group)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_moe_gate_sigmoid_group_topk: bad group config");
  if (num_experts % n_group != 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_moe_gate_sigmoid_group_topk: num_experts % n_group");
  if (top_k > num_experts)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_moe_gate_sigmoid_group_topk: top_k > num_experts");
  if (num_experts > moe::kMaxExperts || top_k > moe::kMaxTopK ||
      n_group > moe::kMaxGroups)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_moe_gate_sigmoid_group_topk: config exceeds limits");

  const dim3 grid(static_cast<unsigned>(num_tokens));
  const dim3 block(moe::kGateBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_moe_gate_sigmoid_group_topk", {
    moe::gate_sigmoid_group_topk_kernel<scalar_t><<<grid, block, 0, s>>>(
        out_weights, out_indices, static_cast<const scalar_t*>(logits),
        static_cast<const scalar_t*>(correction_bias), num_experts, n_group,
        topk_group, top_k, renormalize, routed_scaling_factor);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
