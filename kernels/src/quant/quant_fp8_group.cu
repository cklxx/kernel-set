// kernel-set — per-token-GROUP dynamic FP8 quantization (1 x group_size tiles).
//
// This is the activation format the DeepSeek blockwise FP8 GEMM
// (ks_gemm_fp8_blockwise / DeepGEMM) consumes. For each row r and each
// contiguous group g of `group_size` columns (the last group may be short):
//   amax  = max |input[r, c]|  over the group
//   scale = max(amax / qmax, kEpsScale)        (qmax = 448 e4m3 / 57344 e5m2)
//   out[r, c] = fp8(input[r, c] / scale)
//   scale[r, g] = scale                        (scale layout [rows, ceil(cols/gs)])
//
// FP8 quant uses cuda_fp8 conversion types. CUDA provides software conversions
// on pre-sm89 devices, so activation quantization is usable on A100-class
// hardware; tensor-core FP8 GEMM/provider selection stays separately gated.
//
// One CUDA block per (row, group); the block reduces the group amax via
// ks::block_reduce_max, then casts. Correctness and clarity over speed,
// consistent with quant_fp8.cu and the norm reference kernels.
#include "kernel_set/quant.h"
#include "quant/quant_common.cuh"

// True wherever the FP8 conversion intrinsics are usable: defined on every nvcc
// pass (host + all device arches) via <cuda_fp8.h>, so the kernel symbols exist
// for every targeted arch or the .so won't dlopen on A100/T4/V100. Mirrors the
// guard in quant_fp8.cu / gemm_fp8.cu.
#if defined(KS_HAS_FP8_TYPES)
#define KS_QUANT_FP8_GROUP_AVAILABLE 1
#endif

namespace ks {
namespace quant {

#if defined(KS_QUANT_FP8_GROUP_AVAILABLE)

// ---- Per-token-group quantize: one block per (row, group) ----
// scalar_t = f16/bf16/f32 input; fp8_t = __nv_fp8_e4m3/e5m2 output.
// num_groups = ceil(cols / group_size); scale is [rows, num_groups] row-major.
template <typename scalar_t, typename fp8_t>
KS_GLOBAL void fp8_quant_per_group_kernel(fp8_t* __restrict__ out,
                                          float* __restrict__ scale,
                                          const scalar_t* __restrict__ input,
                                          int64_t rows, int64_t cols,
                                          int group_size, int64_t num_groups,
                                          float qmax) {
  const int64_t row = blockIdx.y;
  const int64_t grp = blockIdx.x;
  if (row >= rows || grp >= num_groups) return;

  const int64_t col0 = grp * group_size;             // first col of this group
  const int64_t col_end = min(col0 + group_size, cols);  // exclusive, clamps tail
  const scalar_t* x = input + row * cols;
  fp8_t* y = out + row * cols;

  // Group absmax (fp32) via a block reduction over the group's columns.
  float local_amax = 0.f;
  for (int64_t c = col0 + threadIdx.x; c < col_end; c += blockDim.x) {
    local_amax = fmaxf(local_amax, fabsf(to_float(x[c])));
  }
  __shared__ float smem[kBlock / KS_WARP_SIZE];
  const float amax = block_reduce_max(local_amax, smem);
  const float sc = fmaxf(amax / qmax, kEpsScale);
  if (threadIdx.x == 0) scale[row * num_groups + grp] = sc;
  const float inv_sc = 1.0f / sc;

  // Cast pass: out = fp8(x / scale). The FP8 cast saturates to the finite max,
  // so dividing by the scale maps amax onto qmax (no explicit clamp needed).
  for (int64_t c = col0 + threadIdx.x; c < col_end; c += blockDim.x) {
    y[c] = from_float<fp8_t>(to_float(x[c]) * inv_sc);
  }
}

#endif  // KS_QUANT_FP8_GROUP_AVAILABLE

}  // namespace quant
}  // namespace ks

using namespace ks;

namespace {
// Validate that fp8_dtype names an FP8 target and return its saturation max.
inline bool fp8_group_qmax(ks_dtype_t fp8_dtype, float* qmax) {
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

ks_status_t ks_quantize_fp8_group(void* out, float* scale, const void* input,
                                  int64_t rows, int64_t cols, int group_size,
                                  ks_dtype_t in_dtype, ks_dtype_t fp8_dtype,
                                  ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(scale);
  KS_CHECK_PTR(input);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_quantize_fp8_group: bad shape");
  if (group_size <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_quantize_fp8_group: group_size must be > 0");
  float qmax = 0.f;
  if (!fp8_group_qmax(fp8_dtype, &qmax))
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                    "ks_quantize_fp8_group: fp8_dtype must be e4m3 or e5m2");
  if (!ks::quant::device_has_fp8())
    KS_RETURN_ERROR(KS_ERROR_ARCH_UNSUPPORTED,
                    "ks_quantize_fp8_group: built without cuda_fp8 conversion types");

#if defined(KS_QUANT_FP8_GROUP_AVAILABLE)
  // num_groups = ceil(cols / group_size); group_size > 0 so this is >= 1 and
  // cannot overflow (num_groups <= cols).
  const int64_t num_groups = (cols + group_size - 1) / group_size;

  // grid = (num_groups, rows); guard both dims against the CUDA grid limits
  // before the dim3 cast (grid.x <= 2^31-1, grid.y <= 65535).
  constexpr uint64_t kMaxGridX = 2147483647ULL;
  constexpr uint64_t kMaxGridY = 65535ULL;
  if (static_cast<uint64_t>(num_groups) > kMaxGridX ||
      static_cast<uint64_t>(rows) > kMaxGridY)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_quantize_fp8_group: grid exceeds CUDA grid limits");

  auto s = to_stream(stream);
  const dim3 block(quant::kBlock);
  const dim3 grid(static_cast<unsigned>(num_groups),
                  static_cast<unsigned>(rows));

  if (fp8_dtype == KS_DTYPE_F8E4M3) {
    auto* o = static_cast<__nv_fp8_e4m3*>(out);
    KS_DISPATCH_FLOATING_TYPES(in_dtype, "ks_quantize_fp8_group", {
      quant::fp8_quant_per_group_kernel<scalar_t, __nv_fp8_e4m3>
          <<<grid, block, 0, s>>>(o, scale,
                                  static_cast<const scalar_t*>(input), rows,
                                  cols, group_size, num_groups, qmax);
    });
  } else {
    auto* o = static_cast<__nv_fp8_e5m2*>(out);
    KS_DISPATCH_FLOATING_TYPES(in_dtype, "ks_quantize_fp8_group", {
      quant::fp8_quant_per_group_kernel<scalar_t, __nv_fp8_e5m2>
          <<<grid, block, 0, s>>>(o, scale,
                                  static_cast<const scalar_t*>(input), rows,
                                  cols, group_size, num_groups, qmax);
    });
  }
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
#else
  // FP8 intrinsics unavailable in this build: the runtime arch check above
  // returns before reaching here on real hardware; keep a defensive status.
  (void)qmax;
  KS_RETURN_ERROR(KS_ERROR_ARCH_UNSUPPORTED,
                  "ks_quantize_fp8_group: built without FP8 support");
#endif
}

}  // extern "C"
