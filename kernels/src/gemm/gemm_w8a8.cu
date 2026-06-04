// kernel-set — W8A8 GEMM: int8 A [M,K] x int8 B [K,N] -> int32 acc -> dequant.
//
//   C = (a_scale (x) b_scale) * (A_i8 @ B_i8) + bias
//     a_scale: per-token  [M]  (KS_QUANT_PER_TOKEN)  or [1] (PER_TENSOR)
//     b_scale: per-channel [N] (KS_QUANT_PER_CHANNEL) or [1] (PER_TENSOR)
//     bias:    [N] in out_dtype (added after dequant) or NULL
//
// Row-major, non-transposed (A row stride K, B row stride N) — the standard
// quantized-linear layout (B is the transposed-and-quantized weight, already
// stored [K,N]). int32 accumulation uses dp4a (__dp4a) which maps to the IMMA
// pipeline on sm_61+; sm_80+ would prefer mma.sync.s8 via CUTLASS for peak
// throughput (the documented production path).
//
// Tiling mirrors the SIMT f32 GEMM: 64x64 block tile, 4x4 register micro-tile,
// K streamed in 16-wide slabs through shared memory. We stage int8 into shared
// memory as int8 and use __dp4a over 4-element groups for the inner product.
#include "kernel_set/gemm.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace gemm {

constexpr int kI8TileM = 64;
constexpr int kI8TileN = 64;
constexpr int kI8TileK = 16;  // multiple of 4 for dp4a packing
constexpr int kI8ThrM = 4;
constexpr int kI8ThrN = 4;
constexpr int kI8BlockThreads = (kI8TileM / kI8ThrM) * (kI8TileN / kI8ThrN);

// Portable 4x int8 dot-product with int32 accumulate. Uses the hardware __dp4a
// on supported arches; falls back to a scalar loop so host/older-arch passes of
// nvcc still compile.
KS_DI int dp4a(int a4, int b4, int acc) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
  return __dp4a(a4, b4, acc);
#else
  const signed char* pa = reinterpret_cast<const signed char*>(&a4);
  const signed char* pb = reinterpret_cast<const signed char*>(&b4);
  int s = acc;
#pragma unroll
  for (int i = 0; i < 4; ++i) s += int(pa[i]) * int(pb[i]);
  return s;
#endif
}

template <typename out_t>
KS_GLOBAL void gemm_w8a8_kernel(out_t* __restrict__ c,
                                const int8_t* __restrict__ a,
                                const int8_t* __restrict__ b,
                                const float* __restrict__ a_scale,
                                const float* __restrict__ b_scale,
                                const out_t* __restrict__ bias, int64_t m,
                                int64_t n, int64_t k, int a_per_token,
                                int b_per_channel) {
  // Shared int8 tiles. A: [kI8TileM][kI8TileK], B: [kI8TileK][kI8TileN].
  __shared__ int8_t as[kI8TileM][kI8TileK];
  __shared__ int8_t bs[kI8TileK][kI8TileN];

  const int block_row = blockIdx.y * kI8TileM;
  const int block_col = blockIdx.x * kI8TileN;

  const int tid = threadIdx.x;
  const int tr = tid / (kI8TileN / kI8ThrN);  // 0..15
  const int tc = tid % (kI8TileN / kI8ThrN);  // 0..15

  int acc[kI8ThrM][kI8ThrN];
#pragma unroll
  for (int i = 0; i < kI8ThrM; ++i)
#pragma unroll
    for (int j = 0; j < kI8ThrN; ++j) acc[i][j] = 0;

  for (int64_t k0 = 0; k0 < k; k0 += kI8TileK) {
    for (int idx = tid; idx < kI8TileM * kI8TileK; idx += kI8BlockThreads) {
      const int r = idx / kI8TileK;
      const int cc = idx % kI8TileK;
      const int64_t gm = block_row + r;
      const int64_t gk = k0 + cc;
      as[r][cc] =
          (gm < m && gk < k) ? a[gm * k + gk] : (int8_t)0;  // A [M,K] row-major
    }
    for (int idx = tid; idx < kI8TileK * kI8TileN; idx += kI8BlockThreads) {
      const int r = idx / kI8TileN;
      const int cc = idx % kI8TileN;
      const int64_t gk = k0 + r;
      const int64_t gn = block_col + cc;
      bs[r][cc] =
          (gk < k && gn < n) ? b[gk * n + gn] : (int8_t)0;  // B [K,N] row-major
    }
    __syncthreads();

    // Inner product over the K slab, 4 elements at a time via dp4a.
#pragma unroll
    for (int kk = 0; kk < kI8TileK; kk += 4) {
      // Pack 4 consecutive int8 of each A row / B col into a 32-bit word.
      int areg[kI8ThrM], breg[kI8ThrN];
#pragma unroll
      for (int i = 0; i < kI8ThrM; ++i) {
        const int row = tr * kI8ThrM + i;
        int w = 0;
        signed char* p = reinterpret_cast<signed char*>(&w);
        p[0] = as[row][kk + 0];
        p[1] = as[row][kk + 1];
        p[2] = as[row][kk + 2];
        p[3] = as[row][kk + 3];
        areg[i] = w;
      }
#pragma unroll
      for (int j = 0; j < kI8ThrN; ++j) {
        const int col = tc * kI8ThrN + j;
        int w = 0;
        signed char* p = reinterpret_cast<signed char*>(&w);
        p[0] = bs[kk + 0][col];
        p[1] = bs[kk + 1][col];
        p[2] = bs[kk + 2][col];
        p[3] = bs[kk + 3][col];
        breg[j] = w;
      }
#pragma unroll
      for (int i = 0; i < kI8ThrM; ++i)
#pragma unroll
        for (int j = 0; j < kI8ThrN; ++j)
          acc[i][j] = dp4a(areg[i], breg[j], acc[i][j]);
    }
    __syncthreads();
  }

  // Dequant epilogue.
#pragma unroll
  for (int i = 0; i < kI8ThrM; ++i) {
    const int64_t gm = block_row + tr * kI8ThrM + i;
    if (gm >= m) continue;
    const float as_scale = a_per_token ? a_scale[gm] : a_scale[0];
#pragma unroll
    for (int j = 0; j < kI8ThrN; ++j) {
      const int64_t gn = block_col + tc * kI8ThrN + j;
      if (gn >= n) continue;
      const float bs_scale = b_per_channel ? b_scale[gn] : b_scale[0];
      float v = static_cast<float>(acc[i][j]) * as_scale * bs_scale;
      if (bias) v += to_float(bias[gn]);
      c[gm * n + gn] = from_float<out_t>(v);
    }
  }
}

template <typename out_t>
void launch_w8a8(out_t* c, const int8_t* a, const int8_t* b,
                 const float* a_scale, const float* b_scale, const out_t* bias,
                 int64_t m, int64_t n, int64_t k, int a_per_token,
                 int b_per_channel, gpuStream_t s) {
  dim3 grid(static_cast<unsigned>((n + kI8TileN - 1) / kI8TileN),
            static_cast<unsigned>((m + kI8TileM - 1) / kI8TileM));
  dim3 block(kI8BlockThreads);
  gemm_w8a8_kernel<out_t><<<grid, block, 0, s>>>(
      c, a, b, a_scale, b_scale, bias, m, n, k, a_per_token, b_per_channel);
}

}  // namespace gemm
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_gemm_w8a8(void* c, const void* a_i8, const void* b_i8,
                         const float* a_scale, const float* b_scale,
                         const void* bias, int64_t m, int64_t n, int64_t k,
                         ks_quant_mode_t a_mode, ks_quant_mode_t b_mode,
                         ks_dtype_t out_dtype, ks_stream_t stream) {
  KS_CHECK_PTR(c);
  KS_CHECK_PTR(a_i8);
  KS_CHECK_PTR(b_i8);
  KS_CHECK_PTR(a_scale);
  KS_CHECK_PTR(b_scale);
  if (m <= 0 || n <= 0 || k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_gemm_w8a8: bad shape");

  // a_scale is per-token [M] for PER_TOKEN, else [1]; b_scale is per-channel [N]
  // for PER_CHANNEL, else [1]. Reject the granularities this kernel can't honor.
  if (a_mode != KS_QUANT_PER_TENSOR && a_mode != KS_QUANT_PER_TOKEN)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_gemm_w8a8: a_mode must be per-tensor or per-token");
  if (b_mode != KS_QUANT_PER_TENSOR && b_mode != KS_QUANT_PER_CHANNEL)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_gemm_w8a8: b_mode must be per-tensor or per-channel");
  const int a_per_token = (a_mode == KS_QUANT_PER_TOKEN) ? 1 : 0;
  const int b_per_channel = (b_mode == KS_QUANT_PER_CHANNEL) ? 1 : 0;

  auto s = to_stream(stream);
  const auto* a = static_cast<const int8_t*>(a_i8);
  const auto* b = static_cast<const int8_t*>(b_i8);

  switch (out_dtype) {
    case KS_DTYPE_F32:
      gemm::launch_w8a8<float>(static_cast<float*>(c), a, b, a_scale, b_scale,
                               static_cast<const float*>(bias), m, n, k,
                               a_per_token, b_per_channel, s);
      break;
    case KS_DTYPE_F16:
      gemm::launch_w8a8<__half>(static_cast<__half*>(c), a, b, a_scale, b_scale,
                                static_cast<const __half*>(bias), m, n, k,
                                a_per_token, b_per_channel, s);
      break;
    case KS_DTYPE_BF16:
      gemm::launch_w8a8<__nv_bfloat16>(
          static_cast<__nv_bfloat16*>(c), a, b, a_scale, b_scale,
          static_cast<const __nv_bfloat16*>(bias), m, n, k, a_per_token,
          b_per_channel, s);
      break;
    default:
      KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                      "ks_gemm_w8a8: out_dtype must be f32/f16/bf16");
  }
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
