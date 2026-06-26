// kernel-set — Group Normalization (NCHW layout).
//
// One CUDA block per (N, group) slice. Threads cooperatively compute mean and
// variance within the slice, then normalize each element. Weight and bias are
// per-channel (shape [C]).
#include "kernel_set/vision.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"

namespace ks {
namespace vision {

constexpr int kBlock = 256;

template <typename scalar_t>
KS_GLOBAL void group_norm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    int n, int c, int hw, int num_groups, float eps) {

  const int c_per_group = c / num_groups;
  const int total_groups = n * num_groups;
  const int group_idx = blockIdx.x;
  if (group_idx >= total_groups) return;

  const int batch = group_idx / num_groups;
  const int g = group_idx % num_groups;
  const int c_start = g * c_per_group;
  const int c_end = c_start + c_per_group;

  const scalar_t* x = input + batch * c * hw;
  scalar_t* y = out + batch * c * hw;

  const int elems = c_per_group * hw;

  // Pass 1: mean
  float local_sum = 0.f;
  for (int i = threadIdx.x; i < elems; i += blockDim.x) {
    const int cc = i / hw;
    const int pos = i % hw;
    local_sum += to_float(x[(c_start + cc) * hw + pos]);
  }
  __shared__ float smem[kBlock / KS_WARP_SIZE];
  const float total = block_reduce_sum(local_sum, smem);
  const float mean = total / static_cast<float>(elems);

  // Pass 2: variance
  __syncthreads();
  float local_sq = 0.f;
  for (int i = threadIdx.x; i < elems; i += blockDim.x) {
    const int cc = i / hw;
    const int pos = i % hw;
    const float v = to_float(x[(c_start + cc) * hw + pos]) - mean;
    local_sq += v * v;
  }
  __syncthreads();
  const float total_sq = block_reduce_sum(local_sq, smem);
  const float var = total_sq / static_cast<float>(elems);
  const float inv_std = rsqrtf(var + eps);

  // Pass 3: normalize + scale + shift
  __syncthreads();
  for (int i = threadIdx.x; i < elems; i += blockDim.x) {
    const int cc = i / hw;
    const int pos = i % hw;
    const int c_global = c_start + cc;
    const int idx = c_global * hw + pos;
    float v = (to_float(x[idx]) - mean) * inv_std;
    if (weight != nullptr) v *= to_float(weight[c_global]);
    if (bias != nullptr) v += to_float(bias[c_global]);
    y[idx] = from_float<scalar_t>(v);
  }
}

}  // namespace vision
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_group_norm(
    void* out, const void* input, const void* weight, const void* bias,
    int n, int c, int hw, int num_groups, float eps,
    ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  if (n <= 0 || c <= 0 || hw <= 0 || num_groups <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_group_norm: bad shape");
  if (c % num_groups != 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_group_norm: C not divisible by num_groups");

  const int total_groups = n * num_groups;
  const dim3 grid(total_groups);
  const dim3 block(vision::kBlock);
  auto st = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_group_norm", {
    vision::group_norm_kernel<scalar_t><<<grid, block, 0, st>>>(
        static_cast<scalar_t*>(out),
        static_cast<const scalar_t*>(input),
        static_cast<const scalar_t*>(weight),
        static_cast<const scalar_t*>(bias),
        n, c, hw, num_groups, eps);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"