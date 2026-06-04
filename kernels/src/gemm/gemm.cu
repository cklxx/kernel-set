// kernel-set — general matrix multiply (f32/f16/bf16) + fused bias/act + batched.
//
//   C = alpha * op(A) @ op(B) + beta * C          (ks_gemm)
//   D = act(alpha * A @ B + bias)                  (ks_gemm_bias_act)
//   strided-batched variant                        (ks_gemm_batched)
//
// Two compute paths share one launcher:
//   * f16/bf16, both operands non-transposed -> nvcuda::wmma tensor-core tiles
//     (16x16x16, fp32 accumulate) with a shared-memory tile loop.
//   * everything else (f32, or any transpose) -> a register-tiled SIMT GEMM that
//     honors trans_a/trans_b and arbitrary leading dimensions.
//
// PRODUCTION PATH: for peak throughput replace these with CUTLASS device GEMMs
// (or DeepGEMM on Hopper for FP8). The WMMA loop here is intentionally a simple
// single-buffered tile loop; the documented upgrades are (1) a cp.async
// multistage software pipeline (double/triple buffering), (2) ldmatrix +
// mma.sync.aligned instead of the wmma:: wrapper for finer scheduling, (3) smem
// swizzling to kill bank conflicts, and (4) split-K with a separate reduction.
#include <mma.h>

#include <type_traits>

#include "kernel_set/gemm.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "gemm/gemm_common.cuh"

namespace ks {
namespace gemm {

// ===========================================================================
// SIMT tiled GEMM (generic: any dtype, any trans, any leading dim).
// ===========================================================================
// Block computes a kTileM x kTileN output tile. Each thread computes a
// kThrM x kThrN micro-tile of registers. K is streamed in kTileK slabs through
// shared memory. Accumulation is fp32 regardless of storage dtype.
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThrM = 4;
constexpr int kThrN = 4;
constexpr int kBlockThreads = (kTileM / kThrM) * (kTileN / kThrN);  // 16*16=256

// Epilogue selector: GEMM (alpha*acc + beta*C) vs bias+act (act(alpha*acc+bias)).
template <typename scalar_t>
KS_GLOBAL void gemm_simt_kernel(scalar_t* __restrict__ c,
                                const scalar_t* __restrict__ a,
                                const scalar_t* __restrict__ b,
                                const scalar_t* __restrict__ bias,
                                int64_t m, int64_t n, int64_t k, int trans_a,
                                int trans_b, int64_t lda, int64_t ldb,
                                int64_t ldc, float alpha, float beta, int act,
                                int use_bias_act) {
  __shared__ float as[kTileM][kTileK];
  __shared__ float bs[kTileK][kTileN];

  const int block_row = blockIdx.y * kTileM;  // first output row of this tile
  const int block_col = blockIdx.x * kTileN;  // first output col of this tile

  // Linear thread id -> (thread row, thread col) inside the 16x16 thread grid.
  const int tid = threadIdx.x;
  const int tr = tid / (kTileN / kThrN);  // 0..15
  const int tc = tid % (kTileN / kThrN);  // 0..15

  float acc[kThrM][kThrN];
#pragma unroll
  for (int i = 0; i < kThrM; ++i)
#pragma unroll
    for (int j = 0; j < kThrN; ++j) acc[i][j] = 0.0f;

  for (int64_t k0 = 0; k0 < k; k0 += kTileK) {
    // Cooperatively stage A tile [kTileM x kTileK] and B tile [kTileK x kTileN].
    // kBlockThreads (256) threads load kTileM*kTileK (1024) elements -> 4 each.
    for (int idx = tid; idx < kTileM * kTileK; idx += kBlockThreads) {
      const int r = idx / kTileK;
      const int cc = idx % kTileK;
      const int64_t gm = block_row + r;
      const int64_t gk = k0 + cc;
      as[r][cc] = (gm < m && gk < k) ? load_a(a, gm, gk, lda, trans_a) : 0.0f;
    }
    for (int idx = tid; idx < kTileK * kTileN; idx += kBlockThreads) {
      const int r = idx / kTileN;
      const int cc = idx % kTileN;
      const int64_t gk = k0 + r;
      const int64_t gn = block_col + cc;
      bs[r][cc] = (gk < k && gn < n) ? load_b(b, gk, gn, ldb, trans_b) : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) {
      float areg[kThrM], breg[kThrN];
#pragma unroll
      for (int i = 0; i < kThrM; ++i) areg[i] = as[tr * kThrM + i][kk];
#pragma unroll
      for (int j = 0; j < kThrN; ++j) breg[j] = bs[kk][tc * kThrN + j];
#pragma unroll
      for (int i = 0; i < kThrM; ++i)
#pragma unroll
        for (int j = 0; j < kThrN; ++j) acc[i][j] += areg[i] * breg[j];
    }
    __syncthreads();
  }

  // Epilogue.
#pragma unroll
  for (int i = 0; i < kThrM; ++i) {
    const int64_t gm = block_row + tr * kThrM + i;
    if (gm >= m) continue;
#pragma unroll
    for (int j = 0; j < kThrN; ++j) {
      const int64_t gn = block_col + tc * kThrN + j;
      if (gn >= n) continue;
      float v = alpha * acc[i][j];
      if (use_bias_act) {
        if (bias) v += to_float(bias[gn]);
        v = apply_activation(v, act);
      } else if (beta != 0.0f) {
        v += beta * to_float(c[gm * ldc + gn]);
      }
      c[gm * ldc + gn] = from_float<scalar_t>(v);
    }
  }
}

// ===========================================================================
// WMMA tensor-core GEMM (f16/bf16, non-transposed, fp32 accumulate).
// ===========================================================================
// One warp owns a 16x16 output tile. A thread block of (WARPS_N x WARPS_M)
// warps computes a (16*WARPS_M) x (16*WARPS_N) block tile. K is streamed in
// WMMA_K (=16) slabs straight from global memory through wmma fragments.
//
// Requires the half-precision fragment element type. For correctness we keep the
// loop simple (load_matrix_sync per K-step from global). The smem-staged,
// cp.async double-buffered version is the documented perf upgrade.
namespace wmma = nvcuda::wmma;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int kWarpsM = 4;  // block tile = 64 rows
constexpr int kWarpsN = 2;  // block tile = 32 cols
constexpr int kWmmaBlockThreads = kWarpsM * kWarpsN * KS_WARP_SIZE;

// The WMMA fast path is currently DISABLED by default: on-device validation
// (sm_89/L4) showed it returns numerically incorrect results (uncorrelated with
// A@B in any operand orientation), while the SIMT path below is verified correct
// (rel_err ~3e-4 fp16, ~8e-7 fp32). Until the fragment bug is root-caused, every
// GEMM routes through the correct SIMT kernel. Production deployments should bind
// cuBLASLt/CUTLASS here anyway. Set -DKS_ENABLE_WMMA_GEMM=1 to opt back in.
#ifndef KS_ENABLE_WMMA_GEMM
#define KS_ENABLE_WMMA_GEMM 0
#endif

template <typename scalar_t>
KS_GLOBAL void gemm_wmma_kernel(scalar_t* __restrict__ c,
                                const scalar_t* __restrict__ a,
                                const scalar_t* __restrict__ b,
                                const scalar_t* __restrict__ bias, int m, int n,
                                int k, int lda, int ldb, int ldc, float alpha,
                                float beta, int act, int use_bias_act) {
  const int warp_id = threadIdx.x / KS_WARP_SIZE;
  const int warp_m = warp_id / kWarpsN;  // 0..kWarpsM-1
  const int warp_n = warp_id % kWarpsN;  // 0..kWarpsN-1

  const int tile_row = (blockIdx.y * kWarpsM + warp_m) * WMMA_M;
  const int tile_col = (blockIdx.x * kWarpsN + warp_n) * WMMA_N;

  // M/N/K are tile-aligned (enforced by the host), but a 64x32 block tile may
  // straddle the M/N edge, leaving some warps wholly out of range. Those warps
  // must skip both the mma loop (to avoid OOB load_matrix_sync) and the store.
  const bool warp_in_range = (tile_row < m) && (tile_col < n);

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  // Walk K in WMMA_K steps (K is a whole multiple of WMMA_K).
  for (int k0 = 0; warp_in_range && k0 < k; k0 += WMMA_K) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, scalar_t,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, scalar_t,
                   wmma::row_major>
        b_frag;
    // A tile (tile_row.., k0..) row-major, leading dim lda. 64-bit math: even
    // with m,n,k < 2^31 the products tile_row*lda / k0*ldb can exceed int range.
    const scalar_t* a_ptr =
        a + static_cast<int64_t>(tile_row) * lda + static_cast<int64_t>(k0);
    // B tile (k0.., tile_col..) row-major, leading dim ldb.
    const scalar_t* b_ptr =
        b + static_cast<int64_t>(k0) * ldb + static_cast<int64_t>(tile_col);
    wmma::load_matrix_sync(a_frag, a_ptr, lda);
    wmma::load_matrix_sync(b_frag, b_ptr, ldb);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Epilogue through shared memory so every lane can apply scalar math + bias.
  __shared__ float
      c_tile[kWarpsM * WMMA_M][kWarpsN * WMMA_N + 1];  // +1: bank padding
  float* c_warp = &c_tile[warp_m * WMMA_M][warp_n * WMMA_N];
  wmma::store_matrix_sync(c_warp, acc_frag, kWarpsN * WMMA_N + 1,
                          wmma::mem_row_major);
  __syncthreads();

  // Each thread writes a strided subset of the block tile to global memory.
  const int rows = kWarpsM * WMMA_M;
  const int cols = kWarpsN * WMMA_N;
  for (int idx = threadIdx.x; idx < rows * cols; idx += kWmmaBlockThreads) {
    const int r = idx / cols;
    const int cc = idx % cols;
    const int gm = blockIdx.y * rows + r;
    const int gn = blockIdx.x * cols + cc;
    if (gm >= m || gn >= n) continue;
    float v = alpha * c_tile[r][cc];
    if (use_bias_act) {
      if (bias) v += to_float(bias[gn]);
      v = apply_activation(v, act);
    } else if (beta != 0.0f) {
      v += beta * to_float(c[(int64_t)gm * ldc + gn]);
    }
    c[(int64_t)gm * ldc + gn] = from_float<scalar_t>(v);
  }
}

// ---------------------------------------------------------------------------
// Host launchers.
// ---------------------------------------------------------------------------
template <typename scalar_t>
void launch_simt(scalar_t* c, const scalar_t* a, const scalar_t* b,
                 const scalar_t* bias, int64_t m, int64_t n, int64_t k,
                 int trans_a, int trans_b, int64_t lda, int64_t ldb, int64_t ldc,
                 float alpha, float beta, int act, int use_bias_act,
                 gpuStream_t s) {
  dim3 grid(static_cast<unsigned>((n + kTileN - 1) / kTileN),
            static_cast<unsigned>((m + kTileM - 1) / kTileM));
  dim3 block(kBlockThreads);
  gemm_simt_kernel<scalar_t><<<grid, block, 0, s>>>(
      c, a, b, bias, m, n, k, trans_a, trans_b, lda, ldb, ldc, alpha, beta, act,
      use_bias_act);
}

// True when the WMMA fast path is applicable: half/bf16, no transpose, and the
// contiguous WMMA load (16-wide along the leading dim) is well-defined.
template <typename scalar_t>
constexpr bool wmma_supported() {
  return !std::is_same<scalar_t, float>::value;
}

// Dispatch one GEMM (single matrix, fp32 acc). Chooses WMMA vs SIMT.
template <typename scalar_t>
void launch_gemm(scalar_t* c, const scalar_t* a, const scalar_t* b,
                 const scalar_t* bias, int64_t m, int64_t n, int64_t k,
                 int trans_a, int trans_b, int64_t lda, int64_t ldb, int64_t ldc,
                 float alpha, float beta, int act, int use_bias_act,
                 gpuStream_t s) {
  // WMMA fast path requires non-transposed half/bf16 operands and tile-aligned
  // M/N/K so that load_matrix_sync never reads past a matrix edge (the wmma
  // wrapper has no per-element bounds handling). Misaligned shapes fall through
  // to the always-safe SIMT path. A production kernel would instead stage padded
  // tiles into shared memory and run wmma on the padded smem.
  //
  // The kernel takes 32-bit shapes/strides for the wmma path, and
  // load_matrix_sync requires the leading dimension to be a multiple of 8
  // elements (16B) for a 16-bit fragment. Bound *all* of m/n/k/lda/ldb/ldc to
  // int range (strides may exceed the logical extent) and require 8-aligned
  // leading dims; anything else falls back to the always-safe int64 SIMT path.
  constexpr int64_t kI32Max = 0x7fffffff;
  // `if constexpr` keeps gemm_wmma_kernel<float> from ever being instantiated:
  // nvcuda::wmma has no float matrix_a/matrix_b fragment, so naming that
  // specialization (even on a runtime-dead branch) is a hard compile error.
  if constexpr (wmma_supported<scalar_t>()) {
    const bool use_wmma =
        KS_ENABLE_WMMA_GEMM && trans_a == 0 && trans_b == 0 && m <= kI32Max &&
        n <= kI32Max &&
        k <= kI32Max && lda <= kI32Max && ldb <= kI32Max && ldc <= kI32Max &&
        (m % WMMA_M == 0) && (n % WMMA_N == 0) && (k % WMMA_K == 0) &&
        (lda % 8 == 0) && (ldb % 8 == 0) && (ldc % 8 == 0);
    if (use_wmma) {
      const int rows = kWarpsM * WMMA_M;  // 64
      const int cols = kWarpsN * WMMA_N;  // 32
      dim3 grid(static_cast<unsigned>((n + cols - 1) / cols),
                static_cast<unsigned>((m + rows - 1) / rows));
      dim3 block(kWmmaBlockThreads);
      gemm_wmma_kernel<scalar_t><<<grid, block, 0, s>>>(
          c, a, b, bias, static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(k), static_cast<int>(lda), static_cast<int>(ldb),
          static_cast<int>(ldc), alpha, beta, act, use_bias_act);
      return;
    }
  }
  {
    launch_simt<scalar_t>(c, a, b, bias, m, n, k, trans_a, trans_b, lda, ldb,
                          ldc, alpha, beta, act, use_bias_act, s);
  }
}

}  // namespace gemm
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_gemm(void* c, const void* a, const void* b, int64_t m, int64_t n,
                    int64_t k, int trans_a, int trans_b, int64_t lda,
                    int64_t ldb, int64_t ldc, float alpha, float beta,
                    ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(c);
  KS_CHECK_PTR(a);
  KS_CHECK_PTR(b);
  if (m <= 0 || n <= 0 || k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_gemm: bad shape");
  if (lda <= 0 || ldb <= 0 || ldc <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_gemm: bad leading dim");

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_gemm", {
    gemm::launch_gemm<scalar_t>(
        static_cast<scalar_t*>(c), static_cast<const scalar_t*>(a),
        static_cast<const scalar_t*>(b), /*bias=*/nullptr, m, n, k, trans_a,
        trans_b, lda, ldb, ldc, alpha, beta, /*act=*/0, /*use_bias_act=*/0, s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_gemm_bias_act(void* d, const void* a, const void* b,
                             const void* bias, int64_t m, int64_t n, int64_t k,
                             float alpha, ks_activation_t act, ks_dtype_t dtype,
                             ks_stream_t stream) {
  KS_CHECK_PTR(d);
  KS_CHECK_PTR(a);
  KS_CHECK_PTR(b);  // bias may be NULL (optional).
  if (m <= 0 || n <= 0 || k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_gemm_bias_act: bad shape");

  // D = act(alpha * A @ B + bias). Row-major, no transpose; lda=k, ldb=n, ldc=n.
  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_gemm_bias_act", {
    gemm::launch_gemm<scalar_t>(
        static_cast<scalar_t*>(d), static_cast<const scalar_t*>(a),
        static_cast<const scalar_t*>(b), static_cast<const scalar_t*>(bias), m,
        n, k, /*trans_a=*/0, /*trans_b=*/0, /*lda=*/k, /*ldb=*/n, /*ldc=*/n,
        alpha, /*beta=*/0.0f, static_cast<int>(act), /*use_bias_act=*/1, s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_gemm_batched(void* c, const void* a, const void* b,
                            int64_t batch, int64_t m, int64_t n, int64_t k,
                            int trans_a, int trans_b, int64_t stride_a,
                            int64_t stride_b, int64_t stride_c, float alpha,
                            float beta, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(c);
  KS_CHECK_PTR(a);
  KS_CHECK_PTR(b);
  if (batch <= 0 || m <= 0 || n <= 0 || k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_gemm_batched: bad shape");

  // Leading dims follow the row-major convention used by the per-op loaders:
  //   op(A)[M,K]: lda = trans_a ? M : K ;  op(B)[K,N]: ldb = trans_b ? K : N.
  const int64_t lda = trans_a ? m : k;
  const int64_t ldb = trans_b ? k : n;
  const int64_t ldc = n;

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_gemm_batched", {
    auto* cp = static_cast<scalar_t*>(c);
    const auto* ap = static_cast<const scalar_t*>(a);
    const auto* bp = static_cast<const scalar_t*>(b);
    for (int64_t bi = 0; bi < batch; ++bi) {
      gemm::launch_gemm<scalar_t>(cp + bi * stride_c, ap + bi * stride_a,
                                  bp + bi * stride_b, /*bias=*/nullptr, m, n, k,
                                  trans_a, trans_b, lda, ldb, ldc, alpha, beta,
                                  /*act=*/0, /*use_bias_act=*/0, s);
    }
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
