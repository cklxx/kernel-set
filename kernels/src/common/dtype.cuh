// kernel-set — dtype traits, accumulation types, and scalar conversions.
//
// Maps the ABI ks_dtype_t enum onto concrete device scalar types and provides
// fp32 <-> low-precision conversions plus per-type numeric limits used by the
// online-softmax and reduction kernels.
#pragma once

#include "kernel_set/types.h"
#include "common/platform.cuh"

namespace ks {

// ---- Element <-> accumulator selection ------------------------------------
template <typename T>
struct AccumType {
  using type = float;
};
template <>
struct AccumType<double> {
  using type = double;
};

template <typename T>
using accum_t = typename AccumType<T>::type;

// ---- fp32 <-> scalar conversions (overloaded; uniform call site) ----------
KS_DI float to_float(float x) { return x; }
KS_DI float to_float(__half x) { return __half2float(x); }
KS_DI float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
KS_DI T from_float(float x);
template <>
KS_DI float from_float<float>(float x) { return x; }
template <>
KS_DI __half from_float<__half>(float x) { return __float2half_rn(x); }
template <>
KS_DI __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

#if defined(KS_HAS_FP8) || (defined(__CUDACC__) && !defined(__CUDA_ARCH__))
KS_DI float to_float(__nv_fp8_e4m3 x) { return float(x); }
KS_DI float to_float(__nv_fp8_e5m2 x) { return float(x); }
template <>
KS_DI __nv_fp8_e4m3 from_float<__nv_fp8_e4m3>(float x) {
  return __nv_fp8_e4m3(x);
}
template <>
KS_DI __nv_fp8_e5m2 from_float<__nv_fp8_e5m2>(float x) {
  return __nv_fp8_e5m2(x);
}
#endif

// ---- Numeric limits used by softmax/reduction -----------------------------
template <typename T>
KS_DI float neg_inf() {
  return -3.4028235e38f;  // -FLT_MAX is a safe sentinel for masked scores
}

// ---- Compile-time dtype tag <-> C++ type ----------------------------------
template <ks_dtype_t D>
struct DTypeTraits;
template <>
struct DTypeTraits<KS_DTYPE_F32> {
  using type = float;
  static constexpr int bits = 32;
};
template <>
struct DTypeTraits<KS_DTYPE_F16> {
  using type = __half;
  static constexpr int bits = 16;
};
template <>
struct DTypeTraits<KS_DTYPE_BF16> {
  using type = __nv_bfloat16;
  static constexpr int bits = 16;
};

}  // namespace ks
