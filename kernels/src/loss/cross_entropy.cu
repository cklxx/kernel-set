// kernel-set — fused cross-entropy (forward + backward in one pass).
//
// Modeled on Liger-Kernel's `liger_cross_entropy_kernel`
// (src/liger_kernel/ops/cross_entropy.py). One CUDA block owns one token row of
// the [num_tokens, vocab] logit matrix; the block makes two streaming passes
// over the row:
//
//   pass 1: online max + running denom (Algorithm-3 online softmax) -> lse,
//           and the dot/sum statistics needed for label smoothing.
//   pass 2: write grad_logits[j] = softmax[j] - (1-ls)*onehot[j] - ls/V
//           in-place (grad may alias logits).
//
// All accumulation is in fp32. `losses[]` receives the per-token loss; reduction
// (sum / mean over non-ignored tokens) is intentionally left to the caller, so
// no 1/n_non_ignore scaling is applied here (unlike Liger's "mean" path).
//
// Notes / further tuning: the row is read twice from global memory. For very
// large vocab the second pass dominates; one could cache logits in registers /
// shared memory when V is small, or use cp.async double buffering. Kept simple
// and bandwidth-correct here.
#include "kernel_set/loss.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"
#include "loss/loss_common.cuh"

namespace ks {
namespace loss {

constexpr int kCeBlock = 256;

template <typename scalar_t>
KS_GLOBAL void cross_entropy_kernel(float* __restrict__ losses,
                                    scalar_t* __restrict__ grad_logits,
                                    const scalar_t* __restrict__ logits,
                                    const void* __restrict__ targets,
                                    int targets_i64, int64_t vocab,
                                    int64_t ignore_index,
                                    float label_smoothing) {
  const int64_t row = blockIdx.x;
  const scalar_t* x = logits + row * vocab;
  scalar_t* dx = grad_logits + row * vocab;

  const int64_t y = load_target_id(targets, targets_i64, row);

  __shared__ float smem[kCeBlock / KS_WARP_SIZE];

  // Ignored token: zero the whole grad row, write 0 loss, return.
  if (y == ignore_index) {
    for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x)
      dx[i] = from_float<scalar_t>(0.f);
    if (threadIdx.x == 0) losses[row] = 0.f;
    return;
  }

  // -- Pass 1: online max + running denom, plus sum(logits) for smoothing. ----
  float thread_max = neg_inf<float>();
  float thread_denom = 0.f;  // sum exp(x - thread_max) consistent with thread_max
  float thread_sum = 0.f;    // sum of raw logits (label-smoothing term)
  for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float v = to_float(x[i]);
    thread_sum += v;
    // Online update of (max, denom) for this thread's running partition.
    if (v > thread_max) {
      thread_denom = thread_denom * __expf(thread_max - v) + 1.f;
      thread_max = v;
    } else {
      thread_denom += __expf(v - thread_max);
    }
  }

  // Block-reduce the running max first, then re-base each thread's denom to it.
  const float row_max = block_reduce_max(thread_max, smem);
  __syncthreads();
  // Rebase: denom_i was relative to thread_max; bring it to row_max.
  float rebased_denom =
      (thread_max == neg_inf<float>()) ? 0.f
                                       : thread_denom * __expf(thread_max - row_max);
  const float denom = block_reduce_sum(rebased_denom, smem);
  __syncthreads();
  const float sum_logits = block_reduce_sum(thread_sum, smem);

  // log-sum-exp and target logit.
  const float lse = row_max + __logf(denom);

  // Each thread that owns column `y` records the target logit; broadcast via
  // shared memory (cheap: a single float).
  __shared__ float s_xy;
  if (y >= 0 && y < vocab) {
    const int owner = static_cast<int>(y % blockDim.x);
    if (threadIdx.x == owner) s_xy = to_float(x[y]);
  }
  __syncthreads();
  const float x_y = (y >= 0 && y < vocab) ? s_xy : 0.f;

  // -- Loss (per Liger). nll = lse - x_y. -----------------------------------
  //   smooth_loss = label_smoothing*lse - eps*sum(logits),  eps = ls / V
  //   loss = (1 - ls)*nll + smooth_loss
  const float ls = label_smoothing;
  const float eps = (vocab > 0) ? ls / static_cast<float>(vocab) : 0.f;
  if (threadIdx.x == 0) {
    const float nll = lse - x_y;
    float loss = nll;
    if (ls > 0.f) {
      const float smooth_loss = ls * lse - eps * sum_logits;
      loss = (1.f - ls) * nll + smooth_loss;
    }
    losses[row] = loss;
  }

  // -- Pass 2: grad = softmax - (1-ls)*onehot - eps. -------------------------
  const float inv_denom = 1.f / denom;
  for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float p = __expf(to_float(x[i]) - row_max) * inv_denom;  // softmax_i
    float g = p - eps;
    if (i == y) g -= (1.f - ls);
    dx[i] = from_float<scalar_t>(g);
  }
}

}  // namespace loss
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_cross_entropy(float* losses, void* grad_logits,
                             const void* logits, const void* targets,
                             int targets_i64, int64_t num_tokens, int64_t vocab,
                             int64_t ignore_index, float label_smoothing,
                             ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(losses);
  KS_CHECK_PTR(grad_logits);
  KS_CHECK_PTR(logits);
  KS_CHECK_PTR(targets);
  if (num_tokens <= 0 || vocab <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_cross_entropy: bad shape");
  if (label_smoothing < 0.f || label_smoothing >= 1.f)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_cross_entropy: label_smoothing must be in [0,1)");

  const dim3 grid(static_cast<unsigned>(num_tokens));
  const dim3 block(loss::kCeBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_cross_entropy", {
    loss::cross_entropy_kernel<scalar_t><<<grid, block, 0, s>>>(
        losses, static_cast<scalar_t*>(grad_logits),
        static_cast<const scalar_t*>(logits), targets, targets_i64, vocab,
        ignore_index, label_smoothing);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
