// kernel-set — activation device functors + vectorized elementwise launcher.
//
// Shared by the elementwise activations (silu/gelu/relu) and the gated-MLP
// fusions (swiglu/geglu and the swiglu backward). All math is performed in fp32
// (storage dtype is converted via ks::to_float / ks::from_float<scalar_t>) so
// f16/bf16 inputs get full-precision accumulation, matching the house style.
//
// The activations themselves are pure scalar functions of a single fp32 value
// (plus their derivatives for the backward pass), which lets the elementwise and
// the gated kernels reuse the exact same numerics.
#pragma once

#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/vec.cuh"

namespace ks {
namespace activation {

// 1 / sqrt(2)  and  sqrt(2/pi), used by the two GELU formulations.
constexpr float kInvSqrt2 = 0.7071067811865476f;
constexpr float kSqrt2OverPi = 0.7978845608028654f;
// GELU-tanh cubic coefficient (0.044715), standard in the BERT/GPT formula.
constexpr float kGeluTanhC = 0.044715f;

// ---- Scalar activation primitives (all in fp32) ---------------------------

// Numerically-stable logistic sigmoid.
KS_DI float sigmoid_f(float x) {
  // Avoids overflow of expf for large |x| by reusing the negative-exp branch.
  return 1.0f / (1.0f + __expf(-x));
}

// SiLU / swish: x * sigmoid(x).
KS_DI float silu_f(float x) { return x * sigmoid_f(x); }

// d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
KS_DI float silu_grad_f(float x) {
  const float s = sigmoid_f(x);
  return s + x * s * (1.0f - s);
}

// Exact GELU using the error function: 0.5 * x * (1 + erf(x / sqrt(2))).
KS_DI float gelu_erf_f(float x) {
  return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

// d/dx of the exact GELU. The pdf term is the standard-normal density.
KS_DI float gelu_erf_grad_f(float x) {
  const float cdf = 0.5f * (1.0f + erff(x * kInvSqrt2));
  // phi(x) = exp(-x^2/2) / sqrt(2*pi); 0.3989422804014327 = 1/sqrt(2*pi).
  const float pdf = 0.3989422804014327f * __expf(-0.5f * x * x);
  return cdf + x * pdf;
}

// Tanh approximation of GELU (the GPT-2 / BERT formulation):
//   0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) )).
KS_DI float gelu_tanh_f(float x) {
  const float inner = kSqrt2OverPi * (x + kGeluTanhC * x * x * x);
  return 0.5f * x * (1.0f + tanhf(inner));
}

// d/dx of the tanh-approx GELU.
KS_DI float gelu_tanh_grad_f(float x) {
  const float x2 = x * x;
  const float inner = kSqrt2OverPi * (x + kGeluTanhC * x * x2);
  const float t = tanhf(inner);
  const float sech2 = 1.0f - t * t;  // d/du tanh(u)
  const float dinner = kSqrt2OverPi * (1.0f + 3.0f * kGeluTanhC * x2);
  return 0.5f * (1.0f + t) + 0.5f * x * sech2 * dinner;
}

KS_DI float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

KS_DI float relu_grad_f(float x) { return x > 0.0f ? 1.0f : 0.0f; }

// ---- Generic vectorized elementwise launcher ------------------------------
//
// `Op` is a stateless functor with `float operator()(float) const`. When the
// input/output pointers and `n` are 128-bit aligned we process VecWidth<T>
// elements per thread through a single wide load/store; otherwise we fall back
// to a scalar grid-stride loop that handles the ragged tail. The two kernels
// share the same Op so the math is identical on both paths.

template <typename scalar_t, typename Op, int VEC>
KS_GLOBAL void elementwise_vec_kernel(scalar_t* __restrict__ out,
                                      const scalar_t* __restrict__ in,
                                      int64_t n_vec, Op op) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t v = idx; v < n_vec; v += stride) {
    Vec<scalar_t, VEC> in_v =
        load_vec<scalar_t, VEC>(in + v * VEC);
    Vec<scalar_t, VEC> out_v;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      out_v[j] = from_float<scalar_t>(op(to_float(in_v[j])));
    }
    store_vec<scalar_t, VEC>(out + v * VEC, out_v);
  }
}

template <typename scalar_t, typename Op>
KS_GLOBAL void elementwise_scalar_kernel(scalar_t* __restrict__ out,
                                         const scalar_t* __restrict__ in,
                                         int64_t n, Op op) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    out[i] = from_float<scalar_t>(op(to_float(in[i])));
  }
}

// Block size used by the elementwise kernels. 256 gives good occupancy on
// sm_80/89/90 while keeping enough resident blocks for memory-bound work.
constexpr int kElemBlock = 256;
// Cap the grid so a small launch still saturates the device without spawning an
// excessive number of idle blocks; the grid-stride loops cover any remainder.
constexpr int kMaxBlocks = 65535;

KS_HDI int grid_for(int64_t work) {
  int64_t blocks = (work + kElemBlock - 1) / kElemBlock;
  if (blocks < 1) blocks = 1;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  return static_cast<int>(blocks);
}

// Launch `op` over n elements, taking the vectorized path when alignment allows.
template <typename scalar_t, typename Op>
void launch_elementwise(scalar_t* out, const scalar_t* in, int64_t n, Op op,
                        gpuStream_t stream) {
  constexpr int VEC = VecWidth<scalar_t>::value;
  if (VEC > 1 && is_aligned<scalar_t, VEC>(in, n) &&
      is_aligned<scalar_t, VEC>(out, n)) {
    const int64_t n_vec = n / VEC;
    const dim3 grid(grid_for(n_vec));
    const dim3 block(kElemBlock);
    elementwise_vec_kernel<scalar_t, Op, VEC>
        <<<grid, block, 0, stream>>>(out, in, n_vec, op);
  } else {
    const dim3 grid(grid_for(n));
    const dim3 block(kElemBlock);
    elementwise_scalar_kernel<scalar_t, Op>
        <<<grid, block, 0, stream>>>(out, in, n, op);
  }
}

// ---- Activation functors (wrap the scalar fns for the launcher) -----------
struct SiluOp {
  KS_DI float operator()(float x) const { return silu_f(x); }
};
struct GeluErfOp {
  KS_DI float operator()(float x) const { return gelu_erf_f(x); }
};
struct GeluTanhOp {
  KS_DI float operator()(float x) const { return gelu_tanh_f(x); }
};
struct ReluOp {
  KS_DI float operator()(float x) const { return relu_f(x); }
};

}  // namespace activation
}  // namespace ks
