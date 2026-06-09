#include "kernel_set/gemm.h"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace gemm {

KS_DI float decode_e2m1(uint8_t bits) {
  // S E E M
  int s = (bits >> 3) & 1;
  int e = (bits >> 1) & 3;
  int m = bits & 1;
  float val = 0.0f;
  if (e == 0) val = m ? 0.5f : 0.0f;
  else if (e == 1) val = m ? 1.5f : 1.0f;
  else if (e == 2) val = m ? 3.0f : 2.0f;
  else if (e == 3) val = m ? 6.0f : 4.0f;
  return s ? -val : val;
}

// Naive correctness fallback
template <typename scalar_t>
KS_GLOBAL void gemm_nvfp4_kernel(scalar_t* __restrict__ c,
                                 const uint8_t* __restrict__ a_fp4,
                                 const uint8_t* __restrict__ b_fp4,
                                 const void* __restrict__ a_scale,
                                 const void* __restrict__ b_scale,
                                 float alpha, int64_t m, int64_t n, int64_t k) {
  const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;

#if defined(KS_HAS_FP8_TYPES)
  const __nv_fp8_e4m3* a_s = static_cast<const __nv_fp8_e4m3*>(a_scale);
  const __nv_fp8_e4m3* b_s = static_cast<const __nv_fp8_e4m3*>(b_scale);
#else
  const uint8_t* a_s = static_cast<const uint8_t*>(a_scale);
  const uint8_t* b_s = static_cast<const uint8_t*>(b_scale);
#endif

  float acc = 0.0f;
  for (int64_t idx = 0; idx < k; ++idx) {
    // a_fp4 is [M, K/2]. 2 elements per byte.
    int64_t a_idx = row * (k / 2) + (idx / 2);
    uint8_t a_byte = a_fp4[a_idx];
    uint8_t a_bits = (idx % 2 == 0) ? (a_byte & 0xF) : (a_byte >> 4);
    float a_val = decode_e2m1(a_bits);

    // a_scale is [M, K/16].
    int64_t a_scale_idx = row * (k / 16) + (idx / 16);
#if defined(KS_HAS_FP8_TYPES)
    float a_sc = float(a_s[a_scale_idx]);
#else
    float a_sc = 1.0f; // Mock if no fp8
#endif

    // b_fp4 is [N, K/2].
    int64_t b_idx = col * (k / 2) + (idx / 2);
    uint8_t b_byte = b_fp4[b_idx];
    uint8_t b_bits = (idx % 2 == 0) ? (b_byte & 0xF) : (b_byte >> 4);
    float b_val = decode_e2m1(b_bits);

    // b_scale is [N, K/16].
    int64_t b_scale_idx = col * (k / 16) + (idx / 16);
#if defined(KS_HAS_FP8_TYPES)
    float b_sc = float(b_s[b_scale_idx]);
#else
    float b_sc = 1.0f; // Mock if no fp8
#endif

    acc += (a_val * a_sc) * (b_val * b_sc);
  }
  c[row * n + col] = from_float<scalar_t>(acc * alpha);
}

} // namespace gemm
} // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_gemm_nvfp4(void* c, const void* a_fp4, const void* b_fp4,
                          const void* a_scale, const void* b_scale,
                          float alpha, int64_t m, int64_t n, int64_t k,
                          ks_dtype_t out_dtype, ks_stream_t stream) {
  KS_CHECK_PTR(c); KS_CHECK_PTR(a_fp4); KS_CHECK_PTR(b_fp4);
  KS_CHECK_PTR(a_scale); KS_CHECK_PTR(b_scale);
  if (k % 16 != 0) return KS_ERROR_UNSUPPORTED_SHAPE;
  auto s = to_stream(stream);
  dim3 block(16, 16);
  dim3 grid((m + 15) / 16, (n + 15) / 16);
  KS_DISPATCH_FLOATING_AND_FP8_TYPES(out_dtype, "ks_gemm_nvfp4", {
    gemm::gemm_nvfp4_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(c), static_cast<const uint8_t*>(a_fp4),
        static_cast<const uint8_t*>(b_fp4), a_scale, b_scale, alpha, m, n, k);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

} // extern "C"
