// kernel-set — dynamic FP8 quantize / dequantize (E4M3 / E5M2).
//
// FP8 hardware conversions require sm_89+ (Ada) / sm_90 (Hopper). The kernels
// here are arch-gated: the FP8 conversion paths only compile where the
// __nv_fp8_* intrinsics exist, and the public ABI returns
// KS_ERROR_ARCH_UNSUPPORTED at runtime on older GPUs (checked via the device's
// compute capability) so a sm_80 build links and runs cleanly.
//
// quantize:  out_fp8 = cast_to_fp8(in / scale),  scale = amax / qmax
//   qmax = 448 (E4M3) or 57344 (E5M2).
//   * KS_QUANT_PER_TOKEN : one scale/row, one block/row reduces row amax.
//   * KS_QUANT_PER_TENSOR: global amax via atomicMax (fp32 bits), then a
//     single-thread amax->scale pass, then the cast pass.
// dequantize: out = float(in_fp8) * scale.
//
// Further tuning (best-effort note): the cast loops are scalar; for large
// tensors a vectorized __nv_fp8x4_* path (4 elements/transaction) and
// cp.async-staged loads would raise bandwidth. The current form is chosen for
// correctness and clarity, consistent with the norm reference kernels.
#include "kernel_set/quant.h"
#include "quant/quant_common.cuh"

// True wherever the FP8 conversion intrinsics are usable: every device pass
// that targets sm_89+ (KS_HAS_FP8) and the single nvcc host pass (which pulls
// in <cuda_fp8.h> for size/dispatch logic). Mirrors the guard in dtype.cuh.
// Define on every nvcc pass (host + all device arches): the fp8 conversion types
// are available everywhere via <cuda_fp8.h> (software conversion on pre-sm89), so
// the kernel symbols must exist for every targeted arch or the .so won't dlopen
// on A100/T4/V100. (Hardware fp8 tensor-core ops, if any, stay under KS_HAS_FP8.)
#if defined(KS_HAS_FP8_TYPES)
#define KS_QUANT_FP8_AVAILABLE 1
#endif

namespace ks {
namespace quant {

#if defined(KS_QUANT_FP8_AVAILABLE)

KS_DI void atomic_max_nonneg_fp8(float* addr, float v) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

// ---- Per-token quantize: one block per row (fp8_t = __nv_fp8_e4m3/e5m2) ----
template <typename scalar_t, typename fp8_t>
KS_GLOBAL void fp8_quant_per_token_kernel(fp8_t* __restrict__ out,
                                          float* __restrict__ scale,
                                          const scalar_t* __restrict__ input,
                                          int64_t cols, float qmax) {
  const int64_t row = blockIdx.x;
  const scalar_t* x = input + row * cols;
  fp8_t* y = out + row * cols;

  float local_amax = 0.f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    local_amax = fmaxf(local_amax, fabsf(to_float(x[i])));
  }
  __shared__ float smem[kBlock / KS_WARP_SIZE];
  const float amax = block_reduce_max(local_amax, smem);
  const float sc = fmaxf(amax / qmax, kEpsScale);
  if (threadIdx.x == 0) scale[row] = sc;
  const float inv_sc = 1.0f / sc;

  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    // The FP8 cast itself saturates to the finite max, so an explicit clamp is
    // unnecessary; we still divide by the scale so amax maps onto qmax.
    y[i] = from_float<fp8_t>(to_float(x[i]) * inv_sc);
  }
}

// ---- Per-tensor pass 1: global amax via atomicMax on fp32 bits ----
template <typename scalar_t>
KS_GLOBAL void fp8_amax_kernel(float* __restrict__ amax_out,
                               const scalar_t* __restrict__ input,
                               int64_t numel) {
  float local_amax = 0.f;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < numel;
       i += (int64_t)gridDim.x * blockDim.x) {
    local_amax = fmaxf(local_amax, fabsf(to_float(input[i])));
  }
  __shared__ float smem[kBlock / KS_WARP_SIZE];
  local_amax = block_reduce_max(local_amax, smem);
  if (threadIdx.x == 0) atomic_max_nonneg_fp8(amax_out, local_amax);
}

// ---- Per-tensor pass 2: amax -> scale (single thread) ----
KS_GLOBAL void fp8_amax_to_scale_kernel(float* __restrict__ scale, float qmax) {
  const float amax = *scale;
  *scale = fmaxf(amax / qmax, kEpsScale);
}

// ---- Per-tensor pass 3: cast with the finalized scale ----
template <typename scalar_t, typename fp8_t>
KS_GLOBAL void fp8_quant_per_tensor_kernel(fp8_t* __restrict__ out,
                                           const float* __restrict__ scale,
                                           const scalar_t* __restrict__ input,
                                           int64_t numel) {
  const float inv_sc = 1.0f / (*scale);
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < numel;
       i += (int64_t)gridDim.x * blockDim.x) {
    out[i] = from_float<fp8_t>(to_float(input[i]) * inv_sc);
  }
}

// ---- Dequantize: out = float(in_fp8) * scale ----
template <typename scalar_t, typename fp8_t>
KS_GLOBAL void fp8_dequant_kernel(scalar_t* __restrict__ out,
                                  const fp8_t* __restrict__ input,
                                  const float* __restrict__ scale,
                                  int64_t rows, int64_t cols, bool per_token) {
  const int64_t numel = rows * cols;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < numel;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t row = i / cols;
    const float sc = per_token ? scale[row] : scale[0];
    out[i] = from_float<scalar_t>(to_float(input[i]) * sc);
  }
}

#endif  // KS_QUANT_FP8_AVAILABLE

}  // namespace quant
}  // namespace ks

using namespace ks;

namespace {
inline unsigned grid_for_fp8(int64_t numel, int block) {
  int64_t g = (numel + block - 1) / block;
  if (g < 1) g = 1;
  if (g > 65535) g = 65535;
  return static_cast<unsigned>(g);
}

// Validate that fp8_dtype names an FP8 target and return its saturation max.
inline bool fp8_qmax(ks_dtype_t fp8_dtype, float* qmax) {
  if (fp8_dtype == KS_DTYPE_F8E4M3) {
    *qmax = ks::quant::kFp8E4M3Max;
    return true;
  }
  if (fp8_dtype == KS_DTYPE_F8E5M2) {
    *qmax = ks::quant::kFp8E5M2Max;
    return true;
  }
  return false;
}
}  // namespace

extern "C" {

ks_status_t ks_quantize_fp8(void* out, float* scale, const void* input,
                            int64_t rows, int64_t cols, ks_dtype_t in_dtype,
                            ks_dtype_t fp8_dtype, ks_quant_mode_t mode,
                            ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(scale);
  KS_CHECK_PTR(input);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_quantize_fp8: bad shape");
  if (mode != KS_QUANT_PER_TENSOR && mode != KS_QUANT_PER_TOKEN)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_quantize_fp8: mode must be per-tensor or per-token");
  float qmax = 0.f;
  if (!fp8_qmax(fp8_dtype, &qmax))
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                    "ks_quantize_fp8: fp8_dtype must be e4m3 or e5m2");
  if (!ks::quant::device_has_fp8())
    KS_RETURN_ERROR(KS_ERROR_ARCH_UNSUPPORTED,
                    "ks_quantize_fp8: FP8 requires sm_89+ (Ada/Hopper)");

#if defined(KS_QUANT_FP8_AVAILABLE)
  const int64_t numel = rows * cols;
  auto s = to_stream(stream);
  const dim3 block(quant::kBlock);

  if (mode == KS_QUANT_PER_TOKEN) {
    const dim3 grid(static_cast<unsigned>(rows));
    if (fp8_dtype == KS_DTYPE_F8E4M3) {
      auto* o = static_cast<__nv_fp8_e4m3*>(out);
      KS_DISPATCH_FLOATING_TYPES(in_dtype, "ks_quantize_fp8", {
        quant::fp8_quant_per_token_kernel<scalar_t, __nv_fp8_e4m3>
            <<<grid, block, 0, s>>>(o, scale,
                                    static_cast<const scalar_t*>(input), cols,
                                    qmax);
      });
    } else {
      auto* o = static_cast<__nv_fp8_e5m2*>(out);
      KS_DISPATCH_FLOATING_TYPES(in_dtype, "ks_quantize_fp8", {
        quant::fp8_quant_per_token_kernel<scalar_t, __nv_fp8_e5m2>
            <<<grid, block, 0, s>>>(o, scale,
                                    static_cast<const scalar_t*>(input), cols,
                                    qmax);
      });
    }
    KS_CHECK_LAUNCH();
    return KS_SUCCESS;
  }

  // Per-tensor: amax accumulator lives in scale[0].
  ks_status_t z =
      ks::check(ks::gpuMemsetAsync(scale, 0, sizeof(float), s), "ks_quantize_fp8");
  if (z != KS_SUCCESS) return z;

  const dim3 grid(grid_for_fp8(numel, quant::kBlock));
  KS_DISPATCH_FLOATING_TYPES(in_dtype, "ks_quantize_fp8", {
    quant::fp8_amax_kernel<scalar_t><<<grid, block, 0, s>>>(
        scale, static_cast<const scalar_t*>(input), numel);
  });
  KS_CHECK_LAUNCH();

  quant::fp8_amax_to_scale_kernel<<<1, 1, 0, s>>>(scale, qmax);
  KS_CHECK_LAUNCH();

  if (fp8_dtype == KS_DTYPE_F8E4M3) {
    auto* o = static_cast<__nv_fp8_e4m3*>(out);
    KS_DISPATCH_FLOATING_TYPES(in_dtype, "ks_quantize_fp8", {
      quant::fp8_quant_per_tensor_kernel<scalar_t, __nv_fp8_e4m3>
          <<<grid, block, 0, s>>>(o, scale,
                                  static_cast<const scalar_t*>(input), numel);
    });
  } else {
    auto* o = static_cast<__nv_fp8_e5m2*>(out);
    KS_DISPATCH_FLOATING_TYPES(in_dtype, "ks_quantize_fp8", {
      quant::fp8_quant_per_tensor_kernel<scalar_t, __nv_fp8_e5m2>
          <<<grid, block, 0, s>>>(o, scale,
                                  static_cast<const scalar_t*>(input), numel);
    });
  }
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
#else
  // FP8 intrinsics unavailable in this build (no sm_89+ device pass): the
  // runtime arch check above already returns before reaching here on real
  // hardware, but keep a defensive status for completeness.
  (void)qmax;
  KS_RETURN_ERROR(KS_ERROR_ARCH_UNSUPPORTED,
                  "ks_quantize_fp8: built without FP8 support");
#endif
}

ks_status_t ks_dequantize_fp8(void* out, const void* input, const float* scale,
                              int64_t rows, int64_t cols, ks_dtype_t out_dtype,
                              ks_dtype_t fp8_dtype, ks_quant_mode_t mode,
                              ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  KS_CHECK_PTR(scale);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_dequantize_fp8: bad shape");
  if (mode != KS_QUANT_PER_TENSOR && mode != KS_QUANT_PER_TOKEN)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_dequantize_fp8: mode must be per-tensor or per-token");
  float qmax = 0.f;
  if (!fp8_qmax(fp8_dtype, &qmax))
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                    "ks_dequantize_fp8: fp8_dtype must be e4m3 or e5m2");
  if (!ks::quant::device_has_fp8())
    KS_RETURN_ERROR(KS_ERROR_ARCH_UNSUPPORTED,
                    "ks_dequantize_fp8: FP8 requires sm_89+ (Ada/Hopper)");

#if defined(KS_QUANT_FP8_AVAILABLE)
  // qmax is only needed to validate fp8_dtype above; dequant rescales by the
  // stored scale and does not reference the saturation max.
  (void)qmax;
  const int64_t numel = rows * cols;
  const bool per_token = (mode == KS_QUANT_PER_TOKEN);
  auto s = to_stream(stream);
  const dim3 block(quant::kBlock);
  const dim3 grid(grid_for_fp8(numel, quant::kBlock));

  if (fp8_dtype == KS_DTYPE_F8E4M3) {
    auto* in8 = static_cast<const __nv_fp8_e4m3*>(input);
    KS_DISPATCH_FLOATING_TYPES(out_dtype, "ks_dequantize_fp8", {
      quant::fp8_dequant_kernel<scalar_t, __nv_fp8_e4m3><<<grid, block, 0, s>>>(
          static_cast<scalar_t*>(out), in8, scale, rows, cols, per_token);
    });
  } else {
    auto* in8 = static_cast<const __nv_fp8_e5m2*>(input);
    KS_DISPATCH_FLOATING_TYPES(out_dtype, "ks_dequantize_fp8", {
      quant::fp8_dequant_kernel<scalar_t, __nv_fp8_e5m2><<<grid, block, 0, s>>>(
          static_cast<scalar_t*>(out), in8, scale, rows, cols, per_token);
    });
  }
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
#else
  (void)qmax;
  KS_RETURN_ERROR(KS_ERROR_ARCH_UNSUPPORTED,
                  "ks_dequantize_fp8: built without FP8 support");
#endif
}

}  // extern "C"
