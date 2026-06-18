// kernel-set — quant category shared helpers.
//
// Houses the constants and small device/host utilities reused by the FP8 / INT8
// dynamic quantizers and the INT4 weight dequantizer. Everything here mirrors
// the house style established by kernels/src/norm/rms_norm.cu:
//   * one CUDA block per row for per-token paths; grid-stride for per-tensor
//   * amax/scale reductions accumulate in fp32 via ks::to_float
//   * block reductions go through ks::block_reduce_max
#pragma once

#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"

namespace ks {
namespace quant {

// Saturation thresholds (|max representable|) per quantized target. Used as the
// denominator when deriving a dynamic scale: scale = amax / qmax so that the
// largest-magnitude element maps onto the extreme of the target range.
//   e4m3 : OCP FP8 E4M3, finite max = 448
//   e5m2 : OCP FP8 E5M2, finite max = 57344
//   int8 : symmetric range, |max| = 127
constexpr float kFp8E4M3Max = 448.0f;
constexpr float kFp8E5M2Max = 57344.0f;
constexpr float kInt8Max = 127.0f;

// Block size for the cooperative per-row reductions. Matches the norm kernels.
constexpr int kBlock = 256;

// Avoid a zero scale (and thus a divide-by-zero / NaN) for all-zero rows.
constexpr float kEpsScale = 1e-12f;

// Clamp a float into [lo, hi] (used for INT8 round-to-nearest with saturation).
KS_DI float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Round-to-nearest-even to an int (matches PyTorch's quantize semantics).
KS_DI int round_to_int(float v) {
  // __float2int_rn = round to nearest even, the IEEE default.
  return __float2int_rn(v);
}

// ---------------------------------------------------------------------------
// Host helper: are FP8 conversion types available in this build?
// Quant/dequant kernels only cast through cuda_fp8 types; <cuda_fp8.h> provides
// software conversions on pre-sm89 devices. Tensor-core FP8 GEMM is still gated
// separately by the GEMM/provider path.
// ---------------------------------------------------------------------------
inline bool device_has_fp8() {
#if defined(KS_HAS_FP8_TYPES)
  return true;
#else
  return false;
#endif
}

}  // namespace quant
}  // namespace ks
