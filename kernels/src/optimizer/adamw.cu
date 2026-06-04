// kernel-set — fused AdamW step (decoupled weight decay).
//
// Mirrors apex `fused_adam`/`multi_tensor_adam` (ADAM_MODE_1 / adamw) and
// DeepSpeed's `FusedAdam`: a single grid-stride elementwise kernel that reads
// grad + fp32 state, applies bias correction and decoupled weight decay, and
// writes the updated parameter back. All math runs in fp32 regardless of the
// stored `param`/`grad` dtype; the fp32 exp_avg / exp_avg_sq moments are the
// source of truth for stability. An optional fp32 `master_param` copy keeps a
// high-precision weight that is updated and then cast down into `param`.
//
// Update (per element, with m = exp_avg, v = exp_avg_sq):
//   g  = grad / grad_scale
//   m  = beta1*m + (1-beta1)*g
//   v  = beta2*v + (1-beta2)*g*g
//   m_hat = m / (1 - beta1^step)         (bias correction, host-precomputed)
//   v_hat = v / (1 - beta2^step)
//   p  = p - lr*weight_decay*p           (decoupled decay, AdamW)
//   p  = p - lr * m_hat / (sqrt(v_hat) + eps)
//
// Bias-correction denominators are computed once on the host from `step` and
// folded into a single `step_size` plus a `bias_correction2` term so the device
// avoids per-element powf.
//
// Perf notes / further tuning: a vectorized path loads param/grad/state in
// 128-bit chunks when all base pointers and `n` are 16-byte aligned. The state
// arrays are fp32 (4 elems / 128-bit) and param/grad may be 2-byte (8 elems),
// so the vector width is taken as the min over the active buffers (= 4 here,
// VecWidth<float>). For even higher throughput one could fuse the multi-tensor
// (apex MTA) form to amortize launch overhead across many small tensors; that
// is left as a follow-up since the ABI takes one tensor per call.
#include <cmath>  // host-side pow() for bias correction

#include "kernel_set/optimizer.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/vec.cuh"

namespace ks {
namespace optimizer {

constexpr int kBlock = 256;

// Core per-element AdamW update, shared by the scalar and vector kernels.
// `inv_grad_scale` is 1/grad_scale (precomputed host-side; 1 when unused).
KS_DI void adamw_update(float& p, float g_raw, float& m, float& v,
                        float lr, float beta1, float beta2, float eps,
                        float weight_decay, float step_size,
                        float bias_correction2, float inv_grad_scale) {
  const float g = g_raw * inv_grad_scale;
  m = beta1 * m + (1.f - beta1) * g;
  v = beta2 * v + (1.f - beta2) * g * g;
  // Decoupled weight decay applied to the parameter directly (AdamW).
  p -= lr * weight_decay * p;
  // step_size = lr / bias_correction1; denom uses bias_correction2.
  const float denom = sqrtf(v * bias_correction2) + eps;
  p -= step_size * (m / denom);
}

// Scalar grid-stride kernel: handles any `n`/alignment.
template <typename scalar_t>
KS_GLOBAL void adamw_kernel(scalar_t* __restrict__ param,
                            float* __restrict__ master_param,
                            const scalar_t* __restrict__ grad,
                            float* __restrict__ exp_avg,
                            float* __restrict__ exp_avg_sq, float lr,
                            float beta1, float beta2, float eps,
                            float weight_decay, float step_size,
                            float bias_correction2, float inv_grad_scale,
                            int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    // Master fp32 copy is authoritative when present; else read from param.
    float p = master_param ? master_param[i] : to_float(param[i]);
    float m = exp_avg[i];
    float v = exp_avg_sq[i];
    adamw_update(p, to_float(grad[i]), m, v, lr, beta1, beta2, eps,
                 weight_decay, step_size, bias_correction2, inv_grad_scale);
    exp_avg[i] = m;
    exp_avg_sq[i] = v;
    if (master_param) master_param[i] = p;
    param[i] = from_float<scalar_t>(p);
  }
}

// Vectorized grid-stride kernel: 4 elements per iteration (VecWidth<float>).
// Used only when every active buffer + n is 16-byte / 4-element aligned.
template <typename scalar_t, int VEC>
KS_GLOBAL void adamw_kernel_vec(scalar_t* __restrict__ param,
                                float* __restrict__ master_param,
                                const scalar_t* __restrict__ grad,
                                float* __restrict__ exp_avg,
                                float* __restrict__ exp_avg_sq, float lr,
                                float beta1, float beta2, float eps,
                                float weight_decay, float step_size,
                                float bias_correction2, float inv_grad_scale,
                                int64_t n_vec) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t vi = blockIdx.x * blockDim.x + threadIdx.x; vi < n_vec;
       vi += stride) {
    const int64_t base = vi * VEC;
    Vec<scalar_t, VEC> pv = load_vec<scalar_t, VEC>(param + base);
    Vec<scalar_t, VEC> gv = load_vec<scalar_t, VEC>(grad + base);
    Vec<float, VEC> mv = load_vec<float, VEC>(exp_avg + base);
    Vec<float, VEC> vv = load_vec<float, VEC>(exp_avg_sq + base);
    Vec<float, VEC> mp;  // master copy (only stored if present)
    if (master_param) mp = load_vec<float, VEC>(master_param + base);

#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float p = master_param ? mp[k] : to_float(pv[k]);
      float m = mv[k];
      float v = vv[k];
      adamw_update(p, to_float(gv[k]), m, v, lr, beta1, beta2, eps,
                   weight_decay, step_size, bias_correction2, inv_grad_scale);
      mv[k] = m;
      vv[k] = v;
      if (master_param) mp[k] = p;
      pv[k] = from_float<scalar_t>(p);
    }

    store_vec<float, VEC>(exp_avg + base, mv);
    store_vec<float, VEC>(exp_avg_sq + base, vv);
    if (master_param) store_vec<float, VEC>(master_param + base, mp);
    store_vec<scalar_t, VEC>(param + base, pv);
  }
}

}  // namespace optimizer
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_adamw(void* param, void* master_param, const void* grad,
                     float* exp_avg, float* exp_avg_sq, float lr, float beta1,
                     float beta2, float eps, float weight_decay, int64_t step,
                     float grad_scale, int64_t n, ks_dtype_t dtype,
                     ks_stream_t stream) {
  KS_CHECK_PTR(param);
  KS_CHECK_PTR(grad);
  KS_CHECK_PTR(exp_avg);
  KS_CHECK_PTR(exp_avg_sq);
  // master_param is optional (may be NULL).
  if (n <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_adamw: n must be > 0");
  if (step <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_adamw: step must be >= 1");
  if (grad_scale == 0.f)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_adamw: grad_scale == 0");

  // Host-side bias correction so the device avoids per-element powf.
  //   bias_correction1 = 1 - beta1^step ; bias_correction2 = 1 - beta2^step
  //   step_size  = lr / bias_correction1
  //   denom term = sqrt(v / bias_correction2) = sqrt(v) * rsqrt(bias_correction2)
  // We pass `bias_correction2` already inverted (1/bc2) so the kernel can do a
  // single multiply before sqrt: sqrtf(v * inv_bc2).
  const double b1p =
      std::pow(static_cast<double>(beta1), static_cast<double>(step));
  const double b2p =
      std::pow(static_cast<double>(beta2), static_cast<double>(step));
  const float bias_correction1 = static_cast<float>(1.0 - b1p);
  const float bias_correction2 = static_cast<float>(1.0 - b2p);
  const float step_size = lr / bias_correction1;
  const float inv_bc2 = 1.f / bias_correction2;
  const float inv_grad_scale = 1.f / grad_scale;

  auto s = to_stream(stream);
  constexpr int kBlock = optimizer::kBlock;

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_adamw", {
    auto* p = static_cast<scalar_t*>(param);
    auto* mp = static_cast<float*>(master_param);
    const auto* g = static_cast<const scalar_t*>(grad);
    constexpr int VEC = VecWidth<float>::value;  // 4 (state is fp32-bound)

    const bool can_vec =
        is_aligned<scalar_t, VEC>(p, n) && is_aligned<scalar_t, VEC>(g, n) &&
        is_aligned<float, VEC>(exp_avg, n) &&
        is_aligned<float, VEC>(exp_avg_sq, n) &&
        (mp == nullptr || is_aligned<float, VEC>(mp, n));

    if (can_vec) {
      const int64_t n_vec = n / VEC;
      int blocks = static_cast<int>((n_vec + kBlock - 1) / kBlock);
      if (blocks < 1) blocks = 1;
      if (blocks > 65535) blocks = 65535;  // grid-stride covers the remainder
      optimizer::adamw_kernel_vec<scalar_t, VEC><<<blocks, kBlock, 0, s>>>(
          p, mp, g, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay,
          step_size, inv_bc2, inv_grad_scale, n_vec);
    } else {
      int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
      if (blocks < 1) blocks = 1;
      if (blocks > 65535) blocks = 65535;
      optimizer::adamw_kernel<scalar_t><<<blocks, kBlock, 0, s>>>(
          p, mp, g, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay,
          step_size, inv_bc2, inv_grad_scale, n);
    }
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
