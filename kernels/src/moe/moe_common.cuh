// kernel-set — shared helpers for the Mixture-of-Experts kernels.
//
// Small device-side utilities used across the MoE gating / permute / grouped
// GEMM kernels. Kept header-only so each .cu in this category can include it.
#pragma once

#include "common/platform.cuh"
#include "common/dtype.cuh"

namespace ks {
namespace moe {

// Numerically-stable sigmoid in fp32 (used by the DeepSeek-V3 gate).
KS_DI float sigmoidf(float x) {
  // 1 / (1 + e^-x), branch on sign to avoid overflow of expf for large |x|.
  if (x >= 0.f) {
    const float z = __expf(-x);
    return 1.f / (1.f + z);
  }
  const float z = __expf(x);
  return z / (1.f + z);
}

}  // namespace moe
}  // namespace ks
