// kernel-set — RMSNorm (forward, fused add-residual, backward).
//
// Reference implementation that establishes the house style for the library:
//   * one CUDA block per row, blockDim.x threads cooperatively reduce the row
//   * all reductions accumulate in fp32 regardless of storage dtype
//   * a single templated kernel serves f32/f16/bf16 via KS_DISPATCH_*
//   * the C ABI wrapper validates args, launches, and returns ks_status_t
#include "kernel_set/norm.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"

namespace ks {
namespace norm {

constexpr int kBlock = 256;

// out[row] = x / sqrt(mean(x^2) + eps) * weight
template <typename scalar_t>
KS_GLOBAL void rms_norm_kernel(scalar_t* __restrict__ out,
                               const scalar_t* __restrict__ input,
                               const scalar_t* __restrict__ weight,
                               int64_t cols, float eps) {
  const int64_t row = blockIdx.x;
  const scalar_t* x = input + row * cols;
  scalar_t* y = out + row * cols;

  float local_sumsq = 0.f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(x[i]);
    local_sumsq += v * v;
  }

  __shared__ float smem[kBlock / KS_WARP_SIZE];
  const float sumsq = block_reduce_sum(local_sumsq, smem);
  const float inv_rms = rsqrtf(sumsq / static_cast<float>(cols) + eps);

  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(x[i]) * inv_rms * to_float(weight[i]);
    y[i] = from_float<scalar_t>(v);
  }
}

// Fused: residual_out = input + residual; out = rms_norm(residual_out) * weight
template <typename scalar_t>
KS_GLOBAL void rms_norm_residual_kernel(scalar_t* __restrict__ out,
                                        scalar_t* __restrict__ residual_out,
                                        const scalar_t* __restrict__ input,
                                        const scalar_t* __restrict__ residual,
                                        const scalar_t* __restrict__ weight,
                                        int64_t cols, float eps) {
  const int64_t row = blockIdx.x;
  const scalar_t* xin = input + row * cols;
  const scalar_t* res = residual + row * cols;
  scalar_t* rout = residual_out + row * cols;
  scalar_t* y = out + row * cols;

  float local_sumsq = 0.f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(xin[i]) + to_float(res[i]);
    rout[i] = from_float<scalar_t>(v);
    local_sumsq += v * v;
  }

  __shared__ float smem[kBlock / KS_WARP_SIZE];
  const float sumsq = block_reduce_sum(local_sumsq, smem);
  const float inv_rms = rsqrtf(sumsq / static_cast<float>(cols) + eps);

  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(rout[i]) * inv_rms * to_float(weight[i]);
    y[i] = from_float<scalar_t>(v);
  }
}

// Backward. d_x = (1/rms) * w*dy - (x / (cols*rms^3)) * sum(w*dy*x)
// d_w accumulates (dy * x / rms) across rows in fp32.
template <typename scalar_t>
KS_GLOBAL void rms_norm_backward_kernel(scalar_t* __restrict__ grad_input,
                                        float* __restrict__ grad_weight,
                                        const scalar_t* __restrict__ grad_out,
                                        const scalar_t* __restrict__ input,
                                        const scalar_t* __restrict__ weight,
                                        int64_t cols, float eps) {
  const int64_t row = blockIdx.x;
  const scalar_t* x = input + row * cols;
  const scalar_t* dy = grad_out + row * cols;
  scalar_t* dx = grad_input + row * cols;

  float sumsq = 0.f, sum_wdyx = 0.f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float xv = to_float(x[i]);
    sumsq += xv * xv;
    sum_wdyx += to_float(weight[i]) * to_float(dy[i]) * xv;
  }
  __shared__ float smem[kBlock / KS_WARP_SIZE];
  sumsq = block_reduce_sum(sumsq, smem);
  __syncthreads();
  sum_wdyx = block_reduce_sum(sum_wdyx, smem);

  const float mean_sq = sumsq / static_cast<float>(cols);
  const float inv_rms = rsqrtf(mean_sq + eps);
  const float coef = sum_wdyx * inv_rms * inv_rms * inv_rms / cols;

  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float xv = to_float(x[i]);
    const float dyv = to_float(dy[i]);
    const float wv = to_float(weight[i]);
    dx[i] = from_float<scalar_t>(inv_rms * wv * dyv - coef * xv);
    atomicAdd(&grad_weight[i], dyv * xv * inv_rms);
  }
}

}  // namespace norm
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_rms_norm(void* out, const void* input, const void* weight,
                        int64_t rows, int64_t cols, float eps,
                        ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  KS_CHECK_PTR(weight);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_rms_norm: bad shape");

  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(norm::kBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_rms_norm", {
    norm::rms_norm_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(input),
        static_cast<const scalar_t*>(weight), cols, eps);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_rms_norm_residual(void* out, void* residual_out,
                                 const void* input, const void* residual,
                                 const void* weight, int64_t rows, int64_t cols,
                                 float eps, ks_dtype_t dtype,
                                 ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(residual_out);
  KS_CHECK_PTR(input);
  KS_CHECK_PTR(residual);
  KS_CHECK_PTR(weight);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_rms_norm_residual: shape");

  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(norm::kBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_rms_norm_residual", {
    norm::rms_norm_residual_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(out), static_cast<scalar_t*>(residual_out),
        static_cast<const scalar_t*>(input),
        static_cast<const scalar_t*>(residual),
        static_cast<const scalar_t*>(weight), cols, eps);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_rms_norm_backward(void* grad_input, void* grad_weight_fp32,
                                 const void* grad_out, const void* input,
                                 const void* weight, int64_t rows, int64_t cols,
                                 float eps, ks_dtype_t dtype,
                                 ks_stream_t stream) {
  KS_CHECK_PTR(grad_input);
  KS_CHECK_PTR(grad_weight_fp32);
  KS_CHECK_PTR(grad_out);
  KS_CHECK_PTR(input);
  KS_CHECK_PTR(weight);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_rms_norm_backward: shape");

  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(norm::kBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_rms_norm_backward", {
    norm::rms_norm_backward_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(grad_input),
        static_cast<float*>(grad_weight_fp32),
        static_cast<const scalar_t*>(grad_out),
        static_cast<const scalar_t*>(input),
        static_cast<const scalar_t*>(weight), cols, eps);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
