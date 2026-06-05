// kernel-set — LayerNorm (forward + backward), fp32 reductions.
#include "kernel_set/norm.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"

namespace ks {
namespace norm {

constexpr int kLnBlock = 256;

template <typename scalar_t>
KS_GLOBAL void layer_norm_kernel(scalar_t* __restrict__ out,
                                 const scalar_t* __restrict__ input,
                                 const scalar_t* __restrict__ weight,
                                 const scalar_t* __restrict__ bias,
                                 int64_t cols, float eps) {
  const int64_t row = blockIdx.x;
  const scalar_t* x = input + row * cols;
  scalar_t* y = out + row * cols;

  float sum = 0.f, sumsq = 0.f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(x[i]);
    sum += v;
    sumsq += v * v;
  }
  __shared__ float smem[kLnBlock / KS_WARP_SIZE];
  sum = block_reduce_sum(sum, smem);
  __syncthreads();
  sumsq = block_reduce_sum(sumsq, smem);

  const float mean = sum / static_cast<float>(cols);
  const float var = sumsq / static_cast<float>(cols) - mean * mean;
  const float inv_std = rsqrtf(var + eps);

  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    float v = (to_float(x[i]) - mean) * inv_std * to_float(weight[i]);
    if (bias) v += to_float(bias[i]);
    y[i] = from_float<scalar_t>(v);
  }
}

template <typename scalar_t>
KS_GLOBAL void layer_norm_backward_kernel(
    scalar_t* __restrict__ grad_input, float* __restrict__ grad_weight,
    float* __restrict__ grad_bias, const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ input, const scalar_t* __restrict__ weight,
    int64_t cols, float eps) {
  const int64_t row = blockIdx.x;
  const scalar_t* x = input + row * cols;
  const scalar_t* dy = grad_out + row * cols;
  scalar_t* dx = grad_input + row * cols;

  float sum = 0.f, sumsq = 0.f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(x[i]);
    sum += v;
    sumsq += v * v;
  }
  __shared__ float smem[kLnBlock / KS_WARP_SIZE];
  sum = block_reduce_sum(sum, smem);
  __syncthreads();
  sumsq = block_reduce_sum(sumsq, smem);
  const float mean = sum / cols;
  const float var = sumsq / cols - mean * mean;
  const float inv_std = rsqrtf(var + eps);

  // Reductions over normalized grads for the input gradient.
  float c1 = 0.f, c2 = 0.f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float xhat = (to_float(x[i]) - mean) * inv_std;
    const float wdy = to_float(weight[i]) * to_float(dy[i]);
    c1 += wdy;
    c2 += wdy * xhat;
  }
  __syncthreads();
  c1 = block_reduce_sum(c1, smem);
  __syncthreads();
  c2 = block_reduce_sum(c2, smem);

  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float xhat = (to_float(x[i]) - mean) * inv_std;
    const float wdy = to_float(weight[i]) * to_float(dy[i]);
    const float dxv = inv_std * (wdy - (c1 + xhat * c2) / cols);
    dx[i] = from_float<scalar_t>(dxv);
    atomicAdd(&grad_weight[i], to_float(dy[i]) * xhat);
    atomicAdd(&grad_bias[i], to_float(dy[i]));
  }
}

}  // namespace norm
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_layer_norm(void* out, const void* input, const void* weight,
                          const void* bias, int64_t rows, int64_t cols,
                          float eps, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  KS_CHECK_PTR(weight);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_layer_norm: bad shape");

  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(norm::kLnBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_layer_norm", {
    norm::layer_norm_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(input),
        static_cast<const scalar_t*>(weight),
        static_cast<const scalar_t*>(bias), cols, eps);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_layer_norm_backward(void* grad_input, void* grad_weight_fp32,
                                   void* grad_bias_fp32, const void* grad_out,
                                   const void* input, const void* weight,
                                   int64_t rows, int64_t cols, float eps,
                                   ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(grad_input);
  KS_CHECK_PTR(grad_weight_fp32);
  KS_CHECK_PTR(grad_bias_fp32);
  KS_CHECK_PTR(grad_out);
  KS_CHECK_PTR(input);
  KS_CHECK_PTR(weight);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_layer_norm_backward: shape");
  if (rows > 2147483647LL)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE, "ks_layer_norm_backward: rows > grid limit");

  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(norm::kLnBlock);
  auto s = to_stream(stream);

  // grad_weight_fp32 / grad_bias_fp32 are atomicAdd-accumulated across rows ->
  // zero them first (self-contained; caller need not pre-zero).
  {
    ks_status_t z = ks::check(
        ks::gpuMemsetAsync(grad_weight_fp32, 0, sizeof(float) * cols, s),
        "ks_layer_norm_backward");
    if (z != KS_SUCCESS) return z;
    z = ks::check(ks::gpuMemsetAsync(grad_bias_fp32, 0, sizeof(float) * cols, s),
                  "ks_layer_norm_backward");
    if (z != KS_SUCCESS) return z;
  }

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_layer_norm_backward", {
    norm::layer_norm_backward_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(grad_input),
        static_cast<float*>(grad_weight_fp32),
        static_cast<float*>(grad_bias_fp32),
        static_cast<const scalar_t*>(grad_out),
        static_cast<const scalar_t*>(input),
        static_cast<const scalar_t*>(weight), cols, eps);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
