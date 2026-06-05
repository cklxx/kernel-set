// kernel-set — FP8 BLOCKWISE GEMM (DeepSeek-V3 recipe): fp8 A [M,K] x fp8 B
// [K,N] -> out [M,N] with fine-grained fp32 scales and TWO-LEVEL fp32
// accumulation.
//
//   C = scaled(A_fp8 @ B_fp8)
//     a_scale: [M, ceil(K/block_k)]               — per 1 x block_k activation
//                                                   tile (per-token-group)
//     b_scale: [ceil(K/block_k), ceil(N/block_n)] — per block_k x block_n weight
//                                                   block
//
// Row-major, non-transposed (A row stride K, B row stride N) — the standard
// quantized-linear layout (B is the transposed-and-quantized weight stored
// [K,N]), identical to gemm_fp8.cu.
//
// ===========================================================================
// THE POINT: two-level fp32 accumulation
// ===========================================================================
// FP8 tensor cores accumulate inner products in reduced precision, which drifts
// over long K. DeepGEMM's fix (and DeepSeek-V3's blockwise recipe) is to
// PROMOTE: accumulate the raw fp8->fp32 inner products within ONE block_k slab
// into a fp32 partial sum, then at the K-block boundary multiply that partial by
// (a_scale[gm, kblk] * b_scale[kblk, gn-block]) and ADD it into a SEPARATE fp32
// output accumulator. The partial is then reset for the next K-block. The
// per-block scale therefore folds in at full fp32 precision and accumulation
// error cannot compound across blocks.
//
// This file ships the portable, auditable SIMT software-dequant path only:
//   * load fp8 (__nv_fp8_e4m3 / __nv_fp8_e5m2) operands,
//   * convert each element to fp32 with ks::to_float (software on pre-sm89,
//     hardware on Ada/Hopper, but ALWAYS available via <cuda_fp8.h> under
//     KS_HAS_FP8_TYPES),
//   * accumulate the inner product in fp32 within a K-block, fold the per-block
//     scale at the block boundary, write out_dtype.
// Because the fp8->fp32 conversion is available on every targeted arch, this
// compiles and runs on sm_80 (A100, no hardware fp8) as well as sm_89/sm_90.
// There is intentionally NO hardware-mma path here; performance comes from
// DeepGEMM/CUTLASS via dispatch above this layer.
#include "kernel_set/gemm.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

// FP8 conversion TYPES are available wherever <cuda_fp8.h> was pulled in — i.e.
// on every nvcc pass (host + all device arches), guarded by KS_HAS_FP8_TYPES in
// platform.cuh. The kernel symbols must exist for EVERY targeted arch or an
// sm_80-only build would leave them undefined and the .so would fail to dlopen
// on A100/T4. Mirror the guard used by gemm_fp8.cu / quant_fp8.cu so the gating
// is consistent across categories.
#if defined(KS_HAS_FP8_TYPES)
#define KS_GEMM_FP8_BLOCKWISE_AVAILABLE 1
#endif

namespace ks {
namespace gemm {

#if defined(KS_GEMM_FP8_BLOCKWISE_AVAILABLE)

// 16x16 thread block; each thread computes a 4x4 register micro-tile, so one
// block produces a 64x64 output tile — the same footprint as gemm_fp8.cu. We
// keep the inner product in registers (no shared staging) for maximum clarity:
// this is the portable correctness fallback, not the perf path. Each thread
// reloads/reconverts its operands every K step; readability over speed.
constexpr int kBwTileM = 64;
constexpr int kBwTileN = 64;
constexpr int kBwThrM = 4;
constexpr int kBwThrN = 4;
constexpr int kBwBlockThreads = (kBwTileM / kBwThrM) * (kBwTileN / kBwThrN);  // 256

// fp8_t is __nv_fp8_e4m3 or __nv_fp8_e5m2; out_t is float / __half / bf16.
//
// num_kblocks = ceil(K/block_k), num_nblocks = ceil(N/block_n). These index the
// row-major scale tensors:
//   a_scale[gm * num_kblocks + kblk]                  (activation 1 x block_k)
//   b_scale[kblk * num_nblocks + (gn / block_n)]      (weight block_k x block_n)
template <typename fp8_t, typename out_t>
KS_GLOBAL void gemm_fp8_blockwise_kernel(
    out_t* __restrict__ c, const fp8_t* __restrict__ a,
    const fp8_t* __restrict__ b, const float* __restrict__ a_scale,
    const float* __restrict__ b_scale, int64_t m, int64_t n, int64_t k,
    int block_n, int block_k, int64_t num_kblocks, int64_t num_nblocks) {
  const int block_row = blockIdx.y * kBwTileM;  // first output row of this tile
  const int block_col = blockIdx.x * kBwTileN;  // first output col of this tile

  // Linear thread id -> (thread row, thread col) inside the 16x16 thread grid.
  const int tid = threadIdx.x;
  const int tr = tid / (kBwTileN / kBwThrN);  // 0..15
  const int tc = tid % (kBwTileN / kBwThrN);  // 0..15

  // Global output coordinates of this thread's 4x4 micro-tile.
  int64_t gm[kBwThrM], gn[kBwThrN];
#pragma unroll
  for (int i = 0; i < kBwThrM; ++i) gm[i] = block_row + tr * kBwThrM + i;
#pragma unroll
  for (int j = 0; j < kBwThrN; ++j) gn[j] = block_col + tc * kBwThrN + j;

  // out_acc: the final, scale-promoted fp32 output accumulator (level 2). It
  // never holds an unscaled partial — each K-block's contribution is folded in
  // at full fp32 precision at the block boundary.
  float out_acc[kBwThrM][kBwThrN];
#pragma unroll
  for (int i = 0; i < kBwThrM; ++i)
#pragma unroll
    for (int j = 0; j < kBwThrN; ++j) out_acc[i][j] = 0.0f;

  // Iterate K in chunks aligned to block_k. Each chunk is exactly one scale
  // K-block (the last may be short when K is not a multiple of block_k).
  for (int64_t k0 = 0; k0 < k; k0 += block_k) {
    const int64_t kblk = k0 / block_k;            // scale K-block index
    const int64_t k_end = min(k0 + block_k, k);   // exclusive, clamps remainder

    // part: the raw fp8->fp32 inner-product partial for THIS K-block (level 1).
    float part[kBwThrM][kBwThrN];
#pragma unroll
    for (int i = 0; i < kBwThrM; ++i)
#pragma unroll
      for (int j = 0; j < kBwThrN; ++j) part[i][j] = 0.0f;

    // Accumulate the unscaled inner product over this K-block in fp32.
    for (int64_t kk = k0; kk < k_end; ++kk) {
      float areg[kBwThrM], breg[kBwThrN];
#pragma unroll
      for (int i = 0; i < kBwThrM; ++i) {
        areg[i] = (gm[i] < m) ? to_float(a[gm[i] * k + kk]) : 0.0f;  // A [M,K]
      }
#pragma unroll
      for (int j = 0; j < kBwThrN; ++j) {
        breg[j] = (gn[j] < n) ? to_float(b[kk * n + gn[j]]) : 0.0f;  // B [K,N]
      }
#pragma unroll
      for (int i = 0; i < kBwThrM; ++i)
#pragma unroll
        for (int j = 0; j < kBwThrN; ++j) part[i][j] += areg[i] * breg[j];
    }

    // K-block boundary: promote. Multiply the partial by this block's
    // a_scale (per (row, kblk)) and b_scale (per (kblk, n-block)) and ADD into
    // the fp32 output accumulator, then the partial resets next iteration.
#pragma unroll
    for (int i = 0; i < kBwThrM; ++i) {
      if (gm[i] >= m) continue;
      const float as = a_scale[gm[i] * num_kblocks + kblk];
#pragma unroll
      for (int j = 0; j < kBwThrN; ++j) {
        if (gn[j] >= n) continue;
        const int64_t nblk = gn[j] / block_n;  // scale N-block index
        const float bs = b_scale[kblk * num_nblocks + nblk];
        out_acc[i][j] += part[i][j] * (as * bs);
      }
    }
  }

  // Epilogue: write the promoted fp32 accumulator as out_dtype.
#pragma unroll
  for (int i = 0; i < kBwThrM; ++i) {
    if (gm[i] >= m) continue;
#pragma unroll
    for (int j = 0; j < kBwThrN; ++j) {
      if (gn[j] >= n) continue;
      c[gm[i] * n + gn[j]] = from_float<out_t>(out_acc[i][j]);
    }
  }
}

template <typename fp8_t, typename out_t>
void launch_fp8_blockwise(out_t* c, const fp8_t* a, const fp8_t* b,
                          const float* a_scale, const float* b_scale, int64_t m,
                          int64_t n, int64_t k, int block_n, int block_k,
                          int64_t num_kblocks, int64_t num_nblocks,
                          gpuStream_t s) {
  dim3 grid(static_cast<unsigned>((n + kBwTileN - 1) / kBwTileN),
            static_cast<unsigned>((m + kBwTileM - 1) / kBwTileM));
  dim3 block(kBwBlockThreads);
  gemm_fp8_blockwise_kernel<fp8_t, out_t><<<grid, block, 0, s>>>(
      c, a, b, a_scale, b_scale, m, n, k, block_n, block_k, num_kblocks,
      num_nblocks);
}

// Dispatch over the output type. Keeps the enum switch out of the public
// function body so the structure matches gemm_fp8.cu's dispatch_out.
template <typename fp8_t>
ks_status_t dispatch_out_blockwise(void* c, const fp8_t* a, const fp8_t* b,
                                   const float* a_scale, const float* b_scale,
                                   int64_t m, int64_t n, int64_t k, int block_n,
                                   int block_k, int64_t num_kblocks,
                                   int64_t num_nblocks, ks_dtype_t out_dtype,
                                   gpuStream_t s) {
  switch (out_dtype) {
    case KS_DTYPE_F32:
      launch_fp8_blockwise<fp8_t, float>(static_cast<float*>(c), a, b, a_scale,
                                         b_scale, m, n, k, block_n, block_k,
                                         num_kblocks, num_nblocks, s);
      return KS_SUCCESS;
    case KS_DTYPE_F16:
      launch_fp8_blockwise<fp8_t, __half>(static_cast<__half*>(c), a, b, a_scale,
                                          b_scale, m, n, k, block_n, block_k,
                                          num_kblocks, num_nblocks, s);
      return KS_SUCCESS;
    case KS_DTYPE_BF16:
      launch_fp8_blockwise<fp8_t, __nv_bfloat16>(
          static_cast<__nv_bfloat16*>(c), a, b, a_scale, b_scale, m, n, k,
          block_n, block_k, num_kblocks, num_nblocks, s);
      return KS_SUCCESS;
    default:
      KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                      "ks_gemm_fp8_blockwise: out_dtype must be f32/f16/bf16");
  }
}

#endif  // KS_GEMM_FP8_BLOCKWISE_AVAILABLE

}  // namespace gemm
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_gemm_fp8_blockwise(void* out, const void* a_fp8,
                                  const void* b_fp8, const float* a_scale,
                                  const float* b_scale, int64_t m, int64_t n,
                                  int64_t k, int block_n, int block_k,
                                  ks_dtype_t fp8_dtype, ks_dtype_t out_dtype,
                                  ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(a_fp8);
  KS_CHECK_PTR(b_fp8);
  KS_CHECK_PTR(a_scale);
  KS_CHECK_PTR(b_scale);
  if (m <= 0 || n <= 0 || k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_gemm_fp8_blockwise: bad shape");
  if (block_n <= 0 || block_k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_gemm_fp8_blockwise: block_n/block_k must be > 0");

  // The kernel indexes A as gm*k+kk, B as kk*n+gn, C as gm*n+gn (all int64). For
  // an int64 ABI those products can signed-overflow on pathological shapes,
  // which would silently turn a valid-looking call into OOB access. Reject any
  // shape whose index products cannot be represented in int64. (m,n,k all > 0.)
  constexpr int64_t kI64Max = 9223372036854775807LL;  // INT64_MAX
  if (m > kI64Max / k || k > kI64Max / n || m > kI64Max / n)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_gemm_fp8_blockwise: M*K / K*N / M*N exceeds int64 range");

  if (fp8_dtype != KS_DTYPE_F8E4M3 && fp8_dtype != KS_DTYPE_F8E5M2)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                    "ks_gemm_fp8_blockwise: fp8_dtype must be e4m3 or e5m2");
  if (out_dtype != KS_DTYPE_F32 && out_dtype != KS_DTYPE_F16 &&
      out_dtype != KS_DTYPE_BF16)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                    "ks_gemm_fp8_blockwise: out_dtype must be f32/f16/bf16");

  // Scale-tensor block counts. ceil division in int64; block_n/block_k > 0 so
  // these are >= 1 and cannot overflow (num_*blocks <= K resp. N).
  const int64_t num_kblocks = (k + block_k - 1) / block_k;
  const int64_t num_nblocks = (n + block_n - 1) / block_n;

  // Tile counts feed the CUDA grid (grid.x = N tiles, grid.y = M tiles). Compute
  // them in uint64 (so m+63 / n+63 cannot overflow) and reject anything past the
  // conservative grid limits before the dim3 cast in launch_fp8_blockwise.
  constexpr uint64_t kMaxGridX = 2147483647ULL;  // grid.x bound (conservative)
  constexpr uint64_t kMaxGridY = 65535ULL;       // grid.y bound
  const uint64_t tiles_n = (static_cast<uint64_t>(n) + 63) / 64;
  const uint64_t tiles_m = (static_cast<uint64_t>(m) + 63) / 64;
  if (tiles_n > kMaxGridX || tiles_m > kMaxGridY)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_gemm_fp8_blockwise: tile grid exceeds CUDA grid limits");

#if defined(KS_GEMM_FP8_BLOCKWISE_AVAILABLE)
  // NOTE on the runtime arch gate: the SIMT dequant path below converts fp8 in
  // SOFTWARE and works on every device (sm_80+), so it deliberately does NOT
  // return KS_ERROR_ARCH_UNSUPPORTED. The hardware fp8-mma upgrade (DeepGEMM /
  // CUTLASS) lives above this layer via dispatch and would gate on
  // device_has_fp8() were it wired in here.
  auto s = to_stream(stream);
  ks_status_t st;
  if (fp8_dtype == KS_DTYPE_F8E4M3) {
    st = gemm::dispatch_out_blockwise<__nv_fp8_e4m3>(
        out, static_cast<const __nv_fp8_e4m3*>(a_fp8),
        static_cast<const __nv_fp8_e4m3*>(b_fp8), a_scale, b_scale, m, n, k,
        block_n, block_k, num_kblocks, num_nblocks, out_dtype, s);
  } else {
    st = gemm::dispatch_out_blockwise<__nv_fp8_e5m2>(
        out, static_cast<const __nv_fp8_e5m2*>(a_fp8),
        static_cast<const __nv_fp8_e5m2*>(b_fp8), a_scale, b_scale, m, n, k,
        block_n, block_k, num_kblocks, num_nblocks, out_dtype, s);
  }
  if (st != KS_SUCCESS) return st;
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
#else
  // FP8 conversion types unavailable in this build (no nvcc / <cuda_fp8.h>): the
  // .so should still link, so report the missing capability defensively rather
  // than fail to compile. On a normal CUDA 12.x build
  // KS_GEMM_FP8_BLOCKWISE_AVAILABLE is always defined (the host pass pulls in
  // <cuda_fp8.h>).
  (void)num_kblocks;
  (void)num_nblocks;
  KS_RETURN_ERROR(KS_ERROR_ARCH_UNSUPPORTED,
                  "ks_gemm_fp8_blockwise: built without FP8 type support");
#endif
}

}  // extern "C"
