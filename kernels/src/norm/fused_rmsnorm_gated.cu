// kernel-set — gated RMSNorm (GatedDeltaNet / GLA output norm).
//
// out = (x * rsqrt(mean(x^2) + eps)) * weight * act(gate)   [output gating]
// act: 0 = SiLU (g*sigmoid(g)), 1 = sigmoid. One block per row, fp32 reduction
// — same house style as rms_norm.cu. Matches FLA FusedRMSNormGated.
#include "kernel_set/norm.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"

namespace ks {
namespace norm {

constexpr int kGatedBlock = 256;

template <typename scalar_t>
KS_GLOBAL void rms_norm_gated_kernel(scalar_t* __restrict__ out,
                                     const scalar_t* __restrict__ input,
                                     const scalar_t* __restrict__ weight,
                                     const scalar_t* __restrict__ gate,
                                     int64_t cols, int gate_act, float eps) {
  const int64_t row = blockIdx.x;
  const scalar_t* x = input + row * cols;
  const scalar_t* g = gate + row * cols;
  scalar_t* y = out + row * cols;

  float local_sumsq = 0.f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(x[i]);
    local_sumsq += v * v;
  }
  __shared__ float smem[kGatedBlock / KS_WARP_SIZE];
  const float sumsq = block_reduce_sum(local_sumsq, smem);
  const float inv_rms = rsqrtf(sumsq / static_cast<float>(cols) + eps);

  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float gv = to_float(g[i]);
    const float act = gate_act == 1 ? (1.f / (1.f + __expf(-gv)))     // sigmoid
                                    : gv / (1.f + __expf(-gv));        // SiLU
    const float v = to_float(x[i]) * inv_rms * to_float(weight[i]) * act;
    y[i] = from_float<scalar_t>(v);
  }
}

}  // namespace norm
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_fused_rmsnorm_gated(void* out, const void* input,
                                   const void* weight, const void* gate,
                                   int64_t rows, int64_t cols, int gate_act,
                                   float eps, ks_dtype_t dtype,
                                   ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  KS_CHECK_PTR(weight);
  KS_CHECK_PTR(gate);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_fused_rmsnorm_gated: shape");
  if (gate_act != 0 && gate_act != 1)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_fused_rmsnorm_gated: gate_act must be 0 (silu) or 1 (sigmoid)");
  if (rows > 2147483647LL)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_fused_rmsnorm_gated: rows > grid limit");

  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(norm::kGatedBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_fused_rmsnorm_gated", {
    norm::rms_norm_gated_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(input),
        static_cast<const scalar_t*>(weight), static_cast<const scalar_t*>(gate),
        cols, gate_act, eps);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
