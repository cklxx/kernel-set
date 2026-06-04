// kernel-set — dtype conversion (ks_cast).
//
// Converts a contiguous buffer of `n` elements from src_dtype to dst_dtype. The
// ABI dtype set spans float (f32/f16/bf16/f64/fp8) and integer (i64/i32/i8/u8)
// storage; this kernel supports the cross product of those scalar types.
//
// Design:
//   * A `double` accumulator is the universal intermediate so int64/f64 keep
//     full range/precision; narrower conversions degrade gracefully. Each
//     storage type provides cast_in() (T -> double) and cast_out<T>() (double ->
//     T) so a single templated kernel serves any (src, dst) pair.
//   * Float->int truncates toward zero (C semantics), matching torch.to / numpy
//     astype for the common cases; we clamp to the int range to avoid UB on
//     overflow (round-trip-safe for in-range values).
//   * IO is a scalar grid-stride loop. Because src and dst element widths differ
//     in the general case (e.g. f32->i8), a single 128-bit Vec<T,N> struct
//     cannot cover both buffers at the same element granularity, so a correct
//     scalar path is used; cast is rarely the bandwidth bottleneck.
//     Future optimization (documented, not applied): for same-width conversions
//     (f16<->bf16, i8<->u8) vectorize load+store with Vec<src,N>/Vec<dst,N>;
//     for same-dtype requests fall back to gpuMemcpyAsync(DeviceToDevice).
//
// I4 (packed int4) is intentionally unsupported here: it is a sub-byte packed
// format whose (de)quantization belongs in the quant category, not a plain
// elementwise cast. Requesting it returns KS_ERROR_UNSUPPORTED_DTYPE.
#include <cstdint>

#include "kernel_set/elementwise.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace elementwise {

constexpr int kCastBlock = 256;

// --------------------------------------------------------------------------
// Universal intermediate: every storage type <-> double.
// --------------------------------------------------------------------------
KS_DI double cast_in(float x) { return static_cast<double>(x); }
KS_DI double cast_in(double x) { return x; }
KS_DI double cast_in(__half x) { return static_cast<double>(__half2float(x)); }
KS_DI double cast_in(__nv_bfloat16 x) {
  return static_cast<double>(__bfloat162float(x));
}
KS_DI double cast_in(int64_t x) { return static_cast<double>(x); }
KS_DI double cast_in(int32_t x) { return static_cast<double>(x); }
KS_DI double cast_in(int8_t x) { return static_cast<double>(x); }
KS_DI double cast_in(uint8_t x) { return static_cast<double>(x); }
#if defined(KS_HAS_FP8) || (defined(__CUDACC__) && !defined(__CUDA_ARCH__))
KS_DI double cast_in(__nv_fp8_e4m3 x) {
  return static_cast<double>(static_cast<float>(x));
}
KS_DI double cast_in(__nv_fp8_e5m2 x) {
  return static_cast<double>(static_cast<float>(x));
}
#endif

// Per-type clamp range for float->int conversions, expressed as double literals
// so there is never a UB shift (e.g. 1LL<<63). int64's bounds are not exactly
// representable in double, but clamping with the nearest double sentinel keeps
// in-range conversions exact and out-of-range ones saturating (not UB).
template <typename IntT>
struct IntRange;
template <>
struct IntRange<int64_t> {
  static constexpr double lo = -9223372036854775808.0;   // -2^63
  static constexpr double hi = 9223372036854774784.0;    // largest double < 2^63
};
template <>
struct IntRange<int32_t> {
  static constexpr double lo = -2147483648.0;            // INT32_MIN
  static constexpr double hi = 2147483647.0;             // INT32_MAX
};
template <>
struct IntRange<int8_t> {
  static constexpr double lo = -128.0;
  static constexpr double hi = 127.0;
};
template <>
struct IntRange<uint8_t> {
  static constexpr double lo = 0.0;
  static constexpr double hi = 255.0;
};

// Clamp then truncate toward zero (C cast semantics). NaN maps to 0.
template <typename IntT>
KS_DI IntT clamp_to_int(double v) {
  if (!(v == v)) return static_cast<IntT>(0);  // NaN guard
  if (v < IntRange<IntT>::lo) v = IntRange<IntT>::lo;
  if (v > IntRange<IntT>::hi) v = IntRange<IntT>::hi;
  return static_cast<IntT>(v);
}

// double -> destination storage type.
template <typename Dst>
KS_DI Dst cast_out(double v);
template <>
KS_DI float cast_out<float>(double v) { return static_cast<float>(v); }
template <>
KS_DI double cast_out<double>(double v) { return v; }
template <>
KS_DI __half cast_out<__half>(double v) {
  return __float2half_rn(static_cast<float>(v));
}
template <>
KS_DI __nv_bfloat16 cast_out<__nv_bfloat16>(double v) {
  return __float2bfloat16_rn(static_cast<float>(v));
}
template <>
KS_DI int64_t cast_out<int64_t>(double v) { return clamp_to_int<int64_t>(v); }
template <>
KS_DI int32_t cast_out<int32_t>(double v) { return clamp_to_int<int32_t>(v); }
template <>
KS_DI int8_t cast_out<int8_t>(double v) { return clamp_to_int<int8_t>(v); }
template <>
KS_DI uint8_t cast_out<uint8_t>(double v) { return clamp_to_int<uint8_t>(v); }
#if defined(KS_HAS_FP8) || (defined(__CUDACC__) && !defined(__CUDA_ARCH__))
template <>
KS_DI __nv_fp8_e4m3 cast_out<__nv_fp8_e4m3>(double v) {
  return __nv_fp8_e4m3(static_cast<float>(v));
}
template <>
KS_DI __nv_fp8_e5m2 cast_out<__nv_fp8_e5m2>(double v) {
  return __nv_fp8_e5m2(static_cast<float>(v));
}
#endif

template <typename SrcT, typename DstT>
KS_GLOBAL void cast_kernel(DstT* __restrict__ out, const SrcT* __restrict__ in,
                           int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = cast_out<DstT>(cast_in(in[i]));
  }
}

inline dim3 cast_grid(int64_t n) {
  int64_t blocks = (n + kCastBlock - 1) / kCastBlock;
  if (blocks < 1) blocks = 1;
  if (blocks > 65535) blocks = 65535;
  return dim3(static_cast<unsigned>(blocks));
}

template <typename SrcT, typename DstT>
inline void launch_cast(DstT* out, const SrcT* in, int64_t n,
                        gpuStream_t stream) {
  cast_kernel<SrcT, DstT><<<cast_grid(n), kCastBlock, 0, stream>>>(out, in, n);
}

// Resolve the DST dtype to a concrete type and launch (SrcT is already fixed).
// Returns KS_SUCCESS on launch, or an error status for unsupported dtypes. The
// template-comma in launch_cast<SrcT, dst_t> is the reason this is a function
// rather than a token-pasting macro (commas in <...> break macro arg parsing).
template <typename SrcT>
inline ks_status_t cast_dispatch_dst(void* out, ks_dtype_t dst_dtype,
                                     const SrcT* in, int64_t n,
                                     gpuStream_t stream) {
  switch (dst_dtype) {
    case KS_DTYPE_F32:
      launch_cast<SrcT, float>(static_cast<float*>(out), in, n, stream);
      return KS_SUCCESS;
    case KS_DTYPE_F16:
      launch_cast<SrcT, __half>(static_cast<__half*>(out), in, n, stream);
      return KS_SUCCESS;
    case KS_DTYPE_BF16:
      launch_cast<SrcT, __nv_bfloat16>(static_cast<__nv_bfloat16*>(out), in, n,
                                       stream);
      return KS_SUCCESS;
    case KS_DTYPE_F64:
      launch_cast<SrcT, double>(static_cast<double*>(out), in, n, stream);
      return KS_SUCCESS;
    case KS_DTYPE_I64:
      launch_cast<SrcT, int64_t>(static_cast<int64_t*>(out), in, n, stream);
      return KS_SUCCESS;
    case KS_DTYPE_I32:
      launch_cast<SrcT, int32_t>(static_cast<int32_t*>(out), in, n, stream);
      return KS_SUCCESS;
    case KS_DTYPE_I8:
      launch_cast<SrcT, int8_t>(static_cast<int8_t*>(out), in, n, stream);
      return KS_SUCCESS;
    case KS_DTYPE_U8:
      launch_cast<SrcT, uint8_t>(static_cast<uint8_t*>(out), in, n, stream);
      return KS_SUCCESS;
#if defined(KS_HAS_FP8) || (defined(__CUDACC__) && !defined(__CUDA_ARCH__))
    case KS_DTYPE_F8E4M3:
      launch_cast<SrcT, __nv_fp8_e4m3>(static_cast<__nv_fp8_e4m3*>(out), in, n,
                                       stream);
      return KS_SUCCESS;
    case KS_DTYPE_F8E5M2:
      launch_cast<SrcT, __nv_fp8_e5m2>(static_cast<__nv_fp8_e5m2*>(out), in, n,
                                       stream);
      return KS_SUCCESS;
#else
    case KS_DTYPE_F8E4M3:
    case KS_DTYPE_F8E5M2:
      KS_RETURN_ERROR(KS_ERROR_ARCH_UNSUPPORTED, "ks_cast: fp8 needs sm_89+");
#endif
    default:
      KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                      "ks_cast: unsupported dst dtype");
  }
}

}  // namespace elementwise
}  // namespace ks

using namespace ks;

namespace {

// Element size (bytes) for each supported ABI dtype; 0 => unsupported here.
inline int dtype_bytes(ks_dtype_t d) {
  switch (d) {
    case KS_DTYPE_F64:
    case KS_DTYPE_I64:
      return 8;
    case KS_DTYPE_F32:
    case KS_DTYPE_I32:
      return 4;
    case KS_DTYPE_F16:
    case KS_DTYPE_BF16:
      return 2;
    case KS_DTYPE_F8E4M3:
    case KS_DTYPE_F8E5M2:
    case KS_DTYPE_I8:
    case KS_DTYPE_U8:
      return 1;
    default:
      return 0;  // I4 (packed) / unknown
  }
}

}  // namespace

extern "C" {

ks_status_t ks_cast(void* out, ks_dtype_t dst_dtype, const void* in,
                    ks_dtype_t src_dtype, int64_t n, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(in);
  if (n < 0) KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_cast: n < 0");
  if (n == 0) return KS_SUCCESS;
  if (dtype_bytes(src_dtype) == 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE, "ks_cast: unsupported src dtype");
  if (dtype_bytes(dst_dtype) == 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE, "ks_cast: unsupported dst dtype");

  auto s = to_stream(stream);
  ks_status_t st = KS_ERROR_UNSUPPORTED_DTYPE;

  // Dispatch SRC dtype, then SRC-typed dispatch resolves DST and launches.
  switch (src_dtype) {
    case KS_DTYPE_F32:
      st = elementwise::cast_dispatch_dst<float>(
          out, dst_dtype, static_cast<const float*>(in), n, s);
      break;
    case KS_DTYPE_F16:
      st = elementwise::cast_dispatch_dst<__half>(
          out, dst_dtype, static_cast<const __half*>(in), n, s);
      break;
    case KS_DTYPE_BF16:
      st = elementwise::cast_dispatch_dst<__nv_bfloat16>(
          out, dst_dtype, static_cast<const __nv_bfloat16*>(in), n, s);
      break;
    case KS_DTYPE_F64:
      st = elementwise::cast_dispatch_dst<double>(
          out, dst_dtype, static_cast<const double*>(in), n, s);
      break;
    case KS_DTYPE_I64:
      st = elementwise::cast_dispatch_dst<int64_t>(
          out, dst_dtype, static_cast<const int64_t*>(in), n, s);
      break;
    case KS_DTYPE_I32:
      st = elementwise::cast_dispatch_dst<int32_t>(
          out, dst_dtype, static_cast<const int32_t*>(in), n, s);
      break;
    case KS_DTYPE_I8:
      st = elementwise::cast_dispatch_dst<int8_t>(
          out, dst_dtype, static_cast<const int8_t*>(in), n, s);
      break;
    case KS_DTYPE_U8:
      st = elementwise::cast_dispatch_dst<uint8_t>(
          out, dst_dtype, static_cast<const uint8_t*>(in), n, s);
      break;
#if defined(KS_HAS_FP8) || (defined(__CUDACC__) && !defined(__CUDA_ARCH__))
    case KS_DTYPE_F8E4M3:
      st = elementwise::cast_dispatch_dst<__nv_fp8_e4m3>(
          out, dst_dtype, static_cast<const __nv_fp8_e4m3*>(in), n, s);
      break;
    case KS_DTYPE_F8E5M2:
      st = elementwise::cast_dispatch_dst<__nv_fp8_e5m2>(
          out, dst_dtype, static_cast<const __nv_fp8_e5m2*>(in), n, s);
      break;
#else
    case KS_DTYPE_F8E4M3:
    case KS_DTYPE_F8E5M2:
      KS_RETURN_ERROR(KS_ERROR_ARCH_UNSUPPORTED, "ks_cast: fp8 needs sm_89+");
#endif
    default:
      KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                      "ks_cast: unsupported src dtype");
  }

  if (st != KS_SUCCESS) return st;
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
