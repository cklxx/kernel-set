// kernel-set — fused SGD with (optional Nesterov) momentum + weight decay.
//
// Mirrors apex `fused_sgd` / `multi_tensor_sgd` and PyTorch's SGD: a single
// grid-stride elementwise kernel. Momentum buffer is fp32 for stability; the
// stored `param`/`grad` may be lower precision and all math runs in fp32. An
// optional fp32 `master_param` copy is the authoritative weight and is cast
// back into `param`.
//
// Update (per element, buf = momentum, mu = momentum_factor):
//   g   = grad / grad_scale
//   g  += weight_decay * p          (coupled / L2 weight decay, PyTorch SGD)
//   buf = mu * buf + g              (after first step; first step seeds buf = g)
//   if nesterov: d = g + mu * buf
//   else:        d = buf
//   p  -= lr * d
//
// First-step handling: PyTorch initializes the momentum buffer to the gradient
// on the first update rather than `mu*0 + g` (identical here since buf starts
// at 0, so `mu*0 + g == g`). Callers therefore just zero the momentum buffer
// before the first step; no separate "first step" flag is needed.
//
// Perf: a vectorized path processes VecWidth<float> (=4) elements per iteration
// when all active buffers + n are 16-byte aligned. Multi-tensor fusion (apex
// MTA) would amortize launch cost across many tensors and is left as a
// follow-up since the ABI is one tensor per call.
#include "kernel_set/optimizer.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/vec.cuh"

namespace ks {
namespace optimizer {

constexpr int kSgdBlock = 256;

KS_DI void sgd_update(float& p, float g_raw, float& buf, float lr,
                      float mu, float weight_decay, bool nesterov,
                      float inv_grad_scale) {
  float g = g_raw * inv_grad_scale;
  if (weight_decay != 0.f) g += weight_decay * p;  // L2 (coupled) decay
  buf = mu * buf + g;
  const float d = nesterov ? (g + mu * buf) : buf;
  p -= lr * d;
}

template <typename scalar_t>
KS_GLOBAL void sgd_kernel(scalar_t* __restrict__ param,
                          float* __restrict__ master_param,
                          const scalar_t* __restrict__ grad,
                          float* __restrict__ momentum, float lr, float mu,
                          float weight_decay, bool nesterov,
                          float inv_grad_scale, int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    float p = master_param ? master_param[i] : to_float(param[i]);
    float buf = momentum[i];
    sgd_update(p, to_float(grad[i]), buf, lr, mu, weight_decay, nesterov,
               inv_grad_scale);
    momentum[i] = buf;
    if (master_param) master_param[i] = p;
    param[i] = from_float<scalar_t>(p);
  }
}

template <typename scalar_t, int VEC>
KS_GLOBAL void sgd_kernel_vec(scalar_t* __restrict__ param,
                              float* __restrict__ master_param,
                              const scalar_t* __restrict__ grad,
                              float* __restrict__ momentum, float lr, float mu,
                              float weight_decay, bool nesterov,
                              float inv_grad_scale, int64_t n_vec) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t vi = blockIdx.x * blockDim.x + threadIdx.x; vi < n_vec;
       vi += stride) {
    const int64_t base = vi * VEC;
    Vec<scalar_t, VEC> pv = load_vec<scalar_t, VEC>(param + base);
    Vec<scalar_t, VEC> gv = load_vec<scalar_t, VEC>(grad + base);
    Vec<float, VEC> bv = load_vec<float, VEC>(momentum + base);
    Vec<float, VEC> mp;
    if (master_param) mp = load_vec<float, VEC>(master_param + base);

#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float p = master_param ? mp[k] : to_float(pv[k]);
      float buf = bv[k];
      sgd_update(p, to_float(gv[k]), buf, lr, mu, weight_decay, nesterov,
                 inv_grad_scale);
      bv[k] = buf;
      if (master_param) mp[k] = p;
      pv[k] = from_float<scalar_t>(p);
    }

    store_vec<float, VEC>(momentum + base, bv);
    if (master_param) store_vec<float, VEC>(master_param + base, mp);
    store_vec<scalar_t, VEC>(param + base, pv);
  }
}

}  // namespace optimizer
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_sgd_momentum(void* param, void* master_param, const void* grad,
                            float* momentum, float lr, float momentum_factor,
                            float weight_decay, int nesterov, float grad_scale,
                            int64_t n, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(param);
  KS_CHECK_PTR(grad);
  KS_CHECK_PTR(momentum);
  // master_param is optional (may be NULL).
  if (n <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_sgd_momentum: n must be > 0");
  if (grad_scale == 0.f)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_sgd_momentum: grad_scale==0");
  // Nesterov momentum is only well-defined with a positive momentum factor.
  if (nesterov && momentum_factor <= 0.f)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_sgd_momentum: nesterov needs momentum_factor > 0");

  const float inv_grad_scale = 1.f / grad_scale;
  const bool nest = nesterov != 0;
  auto s = to_stream(stream);
  constexpr int kBlock = optimizer::kSgdBlock;

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_sgd_momentum", {
    auto* p = static_cast<scalar_t*>(param);
    auto* mp = static_cast<float*>(master_param);
    const auto* g = static_cast<const scalar_t*>(grad);
    constexpr int VEC = VecWidth<float>::value;  // 4

    const bool can_vec =
        is_aligned<scalar_t, VEC>(p, n) && is_aligned<scalar_t, VEC>(g, n) &&
        is_aligned<float, VEC>(momentum, n) &&
        (mp == nullptr || is_aligned<float, VEC>(mp, n));

    if (can_vec) {
      const int64_t n_vec = n / VEC;
      int blocks = static_cast<int>((n_vec + kBlock - 1) / kBlock);
      if (blocks < 1) blocks = 1;
      if (blocks > 65535) blocks = 65535;
      optimizer::sgd_kernel_vec<scalar_t, VEC><<<blocks, kBlock, 0, s>>>(
          p, mp, g, momentum, lr, momentum_factor, weight_decay, nest,
          inv_grad_scale, n_vec);
    } else {
      int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
      if (blocks < 1) blocks = 1;
      if (blocks > 65535) blocks = 65535;
      optimizer::sgd_kernel<scalar_t><<<blocks, kBlock, 0, s>>>(
          p, mp, g, momentum, lr, momentum_factor, weight_decay, nest,
          inv_grad_scale, n);
    }
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
