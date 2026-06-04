// kernel-set — shared GEMM helpers (activation epilogue, index math, WMMA tags).
//
// House-style note: production-grade GEMM in this library is intended to defer to
// CUTLASS / DeepGEMM (Hopper FP8) / Marlin (W4A16) for peak throughput. The
// kernels in this category are *correct, compile-ready reference paths* that use
// nvcuda::wmma tensor-core tiles (16x16x16, fp32 accumulate) with shared-memory
// tiling for f16/bf16 and a tiled SIMT path for f32. They are deliberately
// straightforward so they are easy to audit on a machine without a GPU; the
// comments flag where the production replacement (cp.async multistage pipeline,
// ldmatrix + mma.sync, swizzled smem, split-K) would slot in.
#pragma once

#include "common/platform.cuh"
#include "common/dtype.cuh"

namespace ks {
namespace gemm {

// ---------------------------------------------------------------------------
// Activation epilogue (fp32 math; matches ks_activation_t in types.h).
// ---------------------------------------------------------------------------
KS_DI float gelu_erf(float x) {
  // 0.5 * x * (1 + erf(x / sqrt(2)))
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

KS_DI float gelu_tanh(float x) {
  // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
  const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
  const float k1 = 0.044715f;
  const float inner = k0 * (x + k1 * x * x * x);
  return 0.5f * x * (1.0f + tanhf(inner));
}

KS_DI float silu(float x) { return x / (1.0f + __expf(-x)); }

// Apply ks_activation_t (passed as int to keep this header CUDA-only-clean).
KS_DI float apply_activation(float x, int act) {
  switch (act) {
    case 1:  // KS_ACT_RELU
      return x > 0.0f ? x : 0.0f;
    case 2:  // KS_ACT_GELU (exact)
      return gelu_erf(x);
    case 3:  // KS_ACT_GELU_TANH
      return gelu_tanh(x);
    case 4:  // KS_ACT_SILU
      return silu(x);
    case 0:  // KS_ACT_NONE
    default:
      return x;
  }
}

// ---------------------------------------------------------------------------
// Row-major element addressing honoring trans + leading dimension.
//
// op(A) is [M,K]. If trans_a == 0, A is stored [M,K] row-major with row stride
// lda, so element (row=m, col=k) lives at a[m*lda + k]. If trans_a != 0, the
// stored matrix is [K,M] row-major with row stride lda, so op(A)(m,k) reads from
// a[k*lda + m]. Same logic for B with op(B) being [K,N].
// ---------------------------------------------------------------------------
template <typename T>
KS_DI float load_a(const T* a, int64_t m, int64_t k, int64_t lda, int trans_a) {
  const int64_t idx = trans_a ? (k * lda + m) : (m * lda + k);
  return to_float(a[idx]);
}

template <typename T>
KS_DI float load_b(const T* b, int64_t k, int64_t n, int64_t ldb, int trans_b) {
  const int64_t idx = trans_b ? (n * ldb + k) : (k * ldb + n);
  return to_float(b[idx]);
}

}  // namespace gemm
}  // namespace ks
