#include "kernel_set/quant.h"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include <cmath>

namespace ks {
namespace quant {

// NVFP4 e2m1 representable magnitudes: {0, .5, 1, 1.5, 2, 3, 4, 6}.
// Encode a float to the 4-bit nibble (S EE M) used by decode_e2m1 in the GEMM.
KS_DI uint8_t encode_e2m1(float x) {
  uint8_t sign = (x < 0.0f) ? 0x8 : 0x0;
  float v = fabsf(x);
  uint8_t mag;
  // round-to-nearest using midpoints between adjacent levels
  if (v < 0.25f)      mag = 0;  // 0.0
  else if (v < 0.75f) mag = 1;  // 0.5
  else if (v < 1.25f) mag = 2;  // 1.0
  else if (v < 1.75f) mag = 3;  // 1.5
  else if (v < 2.5f)  mag = 4;  // 2.0
  else if (v < 3.5f)  mag = 5;  // 3.0
  else if (v < 5.0f)  mag = 6;  // 4.0
  else                mag = 7;  // 6.0
  return sign | mag;
}

// Portable NVFP4 quantizer. One thread per 1x16 block.
//   out_fp4:    [rows, cols/2]  packed e2m1 (low nibble = even col, high = odd)
//   out_scales: [rows, cols/16] fp8 e4m3 block scale
//   global_scale: per-tensor fp32 amplitude applied to the input.
template <typename scalar_t>
KS_GLOBAL void quantize_nvfp4_kernel(uint8_t* __restrict__ out_fp4,
                                     __nv_fp8_e4m3* __restrict__ out_scales,
                                     const scalar_t* __restrict__ input,
                                     float global_scale,
                                     int64_t rows, int64_t cols) {
  const int64_t blocks_per_row = cols / 16;
  const int64_t block_id = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (block_id >= rows * blocks_per_row) return;

  const int64_t row = block_id / blocks_per_row;
  const int64_t blk = block_id % blocks_per_row;
  const int64_t col0 = blk * 16;
  const int64_t in_base = row * cols + col0;

  // 1) block absmax (after applying the global amplitude scale).
  float amax = 0.0f;
  for (int j = 0; j < 16; ++j) {
    float v = fabsf(to_float(input[in_base + j]) * global_scale);
    amax = fmaxf(amax, v);
  }

  // 2) block scale = amax / 6 (6 is the max e2m1 magnitude). Store as e4m3.
  float scale = amax / 6.0f;
  if (scale <= 0.0f) scale = 1.0f;  // all-zero block: avoid div-by-zero
  out_scales[row * blocks_per_row + blk] = __nv_fp8_e4m3(scale);

  // 3) requantize using the (rounded) stored scale for round-trip consistency.
  float inv = 1.0f / float(out_scales[row * blocks_per_row + blk]);

  // 4) encode 16 elements -> 8 packed bytes.
  const int64_t out_base = row * (cols / 2) + col0 / 2;
  for (int p = 0; p < 8; ++p) {
    float ve = to_float(input[in_base + 2 * p])     * global_scale * inv;
    float vo = to_float(input[in_base + 2 * p + 1]) * global_scale * inv;
    uint8_t lo = encode_e2m1(ve);
    uint8_t hi = encode_e2m1(vo);
    out_fp4[out_base + p] = static_cast<uint8_t>((hi << 4) | (lo & 0xF));
  }
}

} // namespace quant
} // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_quantize_nvfp4(void* out_fp4, void* out_scales,
                              const void* input, float global_scale,
                              int64_t rows, int64_t cols,
                              ks_dtype_t in_dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out_fp4);
  KS_CHECK_PTR(out_scales);
  KS_CHECK_PTR(input);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_quantize_nvfp4: bad shape");
  if (cols % 16 != 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_quantize_nvfp4: cols must be a multiple of 16");
  if (global_scale == 0.0f) global_scale = 1.0f;

#if defined(KS_HAS_FP8_TYPES)
  auto s = to_stream(stream);
  const int64_t total_blocks = rows * (cols / 16);
  const int threads = 256;
  int64_t g = (total_blocks + threads - 1) / threads;
  if (g < 1) g = 1;
  if (g > 2147483647LL) g = 2147483647LL;
  const dim3 grid(static_cast<unsigned>(g));
  const dim3 block(threads);

  KS_DISPATCH_FLOATING_TYPES(in_dtype, "ks_quantize_nvfp4", {
    quant::quantize_nvfp4_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<uint8_t*>(out_fp4),
        static_cast<__nv_fp8_e4m3*>(out_scales),
        static_cast<const scalar_t*>(input),
        global_scale, rows, cols);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
#else
  (void)rows; (void)cols; (void)in_dtype; (void)stream;
  KS_RETURN_ERROR(KS_ERROR_ARCH_UNSUPPORTED,
                  "ks_quantize_nvfp4: requires FP8 (e4m3) support for block scales");
#endif
}

} // extern "C"
