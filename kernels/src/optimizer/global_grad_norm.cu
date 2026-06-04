// kernel-set — global L2 gradient norm for gradient clipping.
//
// Computes  out_norm = sqrt( sum_t sum_i grad_t[i]^2 )  over an array of
// `num_tensors` grad tensors (apex `multi_tensor_l2norm` / DeepSpeed
// `get_global_norm`). The classic two-pass form needs a workspace tensor; here
// we accumulate directly into the single fp32 `out_norm` scalar:
//
//   1. memset out_norm = 0
//   2. for each tensor: launch a block-reduce sum-of-squares kernel whose
//      block leaders atomicAdd their partial into out_norm (accumulation in
//      fp32; the storage dtype is read and converted via ks::to_float)
//   3. a 1-thread finalize kernel rewrites out_norm = sqrt(out_norm)
//
// Atomics on a single scalar are cheap because each *block* contributes once,
// not each thread. All squaring accumulates in fp32 regardless of grad dtype,
// matching the contract.
//
// Perf / further tuning: for many tiny tensors the per-tensor launch overhead
// dominates; the apex multi-tensor-apply form packs all tensors into one launch
// via a chunk descriptor. That is a worthwhile follow-up but needs a device-side
// tensor-list layout the current ABI does not provide (it passes a host array of
// device pointers), so we launch per tensor here. A vectorized 128-bit load path
// is used when a tensor's pointer + size are aligned.
#include "kernel_set/optimizer.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"
#include "common/vec.cuh"

namespace ks {
namespace optimizer {

constexpr int kNormBlock = 256;

// Block-reduce sum of squares for one tensor; each block atomicAdds its partial.
template <typename scalar_t>
KS_GLOBAL void sumsq_kernel(float* __restrict__ out_norm,
                            const scalar_t* __restrict__ grad, int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  float local = 0.f;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const float v = to_float(grad[i]);
    local += v * v;
  }
  __shared__ float smem[kNormBlock / KS_WARP_SIZE];
  const float block_sum = block_reduce_sum(local, smem);
  if (threadIdx.x == 0) atomicAdd(out_norm, block_sum);
}

// Vectorized variant: VecWidth<scalar_t> elements per step (used when aligned).
template <typename scalar_t, int VEC>
KS_GLOBAL void sumsq_kernel_vec(float* __restrict__ out_norm,
                                const scalar_t* __restrict__ grad,
                                int64_t n_vec) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  float local = 0.f;
  for (int64_t vi = blockIdx.x * blockDim.x + threadIdx.x; vi < n_vec;
       vi += stride) {
    const Vec<scalar_t, VEC> gv = load_vec<scalar_t, VEC>(grad + vi * VEC);
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      const float v = to_float(gv[k]);
      local += v * v;
    }
  }
  __shared__ float smem[kNormBlock / KS_WARP_SIZE];
  const float block_sum = block_reduce_sum(local, smem);
  if (threadIdx.x == 0) atomicAdd(out_norm, block_sum);
}

KS_GLOBAL void sqrt_inplace_kernel(float* __restrict__ out_norm) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *out_norm = sqrtf(*out_norm);
}

}  // namespace optimizer
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_global_grad_norm(float* out_norm, const void* const* grads,
                                const int64_t* sizes, int num_tensors,
                                ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out_norm);
  KS_CHECK_PTR(grads);
  KS_CHECK_PTR(sizes);
  if (num_tensors < 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_global_grad_norm: num_tensors < 0");

  auto s = to_stream(stream);
  constexpr int kBlock = optimizer::kNormBlock;

  // Zero the accumulator before any block contributes.
  ks_status_t st = ks::check(
      ks::gpuMemsetAsync(out_norm, 0, sizeof(float), s), "ks_global_grad_norm");
  if (st != KS_SUCCESS) return st;

  for (int t = 0; t < num_tensors; ++t) {
    const int64_t n = sizes[t];
    if (n < 0)
      KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                      "ks_global_grad_norm: negative tensor size");
    if (n == 0) continue;  // empty tensor contributes nothing
    KS_CHECK_PTR(grads[t]);

    KS_DISPATCH_FLOATING_TYPES(dtype, "ks_global_grad_norm", {
      const auto* g = static_cast<const scalar_t*>(grads[t]);
      constexpr int VEC = VecWidth<scalar_t>::value;

      if (is_aligned<scalar_t, VEC>(g, n)) {
        const int64_t n_vec = n / VEC;
        int blocks = static_cast<int>((n_vec + kBlock - 1) / kBlock);
        if (blocks < 1) blocks = 1;
        if (blocks > 65535) blocks = 65535;  // grid-stride covers remainder
        optimizer::sumsq_kernel_vec<scalar_t, VEC>
            <<<blocks, kBlock, 0, s>>>(out_norm, g, n_vec);
      } else {
        int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
        if (blocks < 1) blocks = 1;
        if (blocks > 65535) blocks = 65535;
        optimizer::sumsq_kernel<scalar_t>
            <<<blocks, kBlock, 0, s>>>(out_norm, g, n);
      }
    });
    KS_CHECK_LAUNCH();
  }

  // out_norm currently holds the sum of squares; take the root in place.
  optimizer::sqrt_inplace_kernel<<<1, 1, 0, s>>>(out_norm);
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
