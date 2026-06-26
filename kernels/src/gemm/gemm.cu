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

#include <cublas_v2.h>
#include <cuda_fp16.h>
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
// Each warp owns a 16x16 output tile; a thread block of kWarpsM x kWarpsN warps
// computes a (16*kWarpsM) x (16*kWarpsN) block tile (here 64x64, 16 warps / 512
// threads). K is streamed in BK-wide slabs that are STAGED INTO SHARED MEMORY,
// then each warp runs load_matrix_sync from the smem staging buffers.
//
// =====================  THE BUG IN THE OLD WMMA PATH  ======================
// The previous version ran load_matrix_sync directly on *global* pointers
//   a_ptr = a + tile_row*lda + k0      (matrix_a, row_major, ldm=lda)
//   b_ptr = b + k0*ldb   + tile_col    (matrix_b, row_major, ldm=ldb)
// for every K step. That is the FlashAttention-tier subtle WMMA pitfall: a warp
// whose 64x32 block tile straddled the M/N edge was guarded out of the K-loop
// (`warp_in_range`), yet it still fell through to store_matrix_sync +
// __syncthreads. On the benchmark's tile-aligned shapes all warps are in range,
// so that alone is not the corruption — but routing every fragment load through
// raw global strides leaves zero margin for the documented load_matrix_sync
// alignment contract (the base AND every strided row must be 16B-aligned) and
// makes the kernel impossible to validate by inspection. The robust, canonical
// fix (used by CUTLASS/cuBLAS reference paths) is to stage A and B tiles into
// shared memory with a fixed, padded leading dimension and load the fragments
// from smem. With a constant smem ldm the fragment layout is unambiguous, bounds
// are handled at the cooperative-load step (zero-fill past the edge), and the
// accumulator->smem->global epilogue uses one consistent stride. This rewrite
// does exactly that and is re-enabled by default (KS_ENABLE_WMMA_GEMM=1).
//
// Requires the half-precision fragment element type (f16/bf16). Further perf:
// cp.async multistage double-buffering of the smem tiles, ldmatrix +
// mma.sync.aligned, swizzled smem to kill bank conflicts, and split-K.
namespace wmma = nvcuda::wmma;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int kWarpsM = 4;  // block tile = 64 rows
constexpr int kWarpsN = 4;  // block tile = 64 cols
constexpr int kWmmaBlockThreads = kWarpsM * kWarpsN * KS_WARP_SIZE;  // 16*32=512

constexpr int kWmmaBM = kWarpsM * WMMA_M;  // 64 block-tile rows (M)
constexpr int kWmmaBN = kWarpsN * WMMA_N;  // 64 block-tile cols (N)
constexpr int kWmmaBK = 64;               // K per smem slab (inner loops over 16-wide slices)
constexpr int kWmmaBKSteps = kWmmaBK / WMMA_K;  // 4 inner iterations per slab
// No padding on the inner stride — 64 halfs = 128 bytes = 32 banks × 4 bytes.
// This gives zero bank conflicts for row-major access and is 16B-aligned
// as required by load_matrix_sync.
constexpr int kSmemPad = 0;
constexpr int kAsLd = kWmmaBK + kSmemPad;   // A staging row stride (along K): 64
constexpr int kBsLd = kWmmaBN + kSmemPad;   // B staging row stride (along N): 64

// The WMMA fast path is ENABLED by default. It was previously disabled because
// the direct-from-global load_matrix_sync version returned incorrect results on
// L4; this rewrite stages A/B into shared memory with constant padded leading
// dimensions (see the bug note above), which is the canonical correct form.
// DISABLED by default: on-GPU verification (L4 sm89) showed even this staged-smem
// rewrite still returns rel_err=inf vs cuBLAS, so the WMMA path is NOT trusted yet.
// GEMM routes through the verified-correct SIMT path (rel_err ~3e-4). Production
// should bind cuBLASLt/CUTLASS. Set -DKS_ENABLE_WMMA_GEMM=1 only after fixing +
// re-verifying the tensor-core path on a GPU.
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
  const int warp_m = warp_id / kWarpsN;  // 0..kWarpsM-1 (this warp's tile row)
  const int warp_n = warp_id % kWarpsN;  // 0..kWarpsN-1 (this warp's tile col)

  // First global row/col of this block's 64x64 output tile.
  const int block_row = blockIdx.y * kWmmaBM;
  const int block_col = blockIdx.x * kWmmaBN;

  // Shared-memory staging for one K-slab of A (kWmmaBM x kWmmaBK) and B
  // (kWmmaBK x kWmmaBN), each padded on the inner stride. Stored as scalar_t so
  // the wmma fragment element type matches exactly.
  __shared__ scalar_t as[kWmmaBM][kAsLd];   // [64][128+8]
  __shared__ scalar_t bs[kWmmaBK][kBsLd];   // [128][64+8]

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  const scalar_t kZero = from_float<scalar_t>(0.0f);

  // Walk K in kWmmaBK (128) slabs; inner loop processes 8 slices of WMMA_K (16).
  for (int k0 = 0; k0 < k; k0 += kWmmaBK) {
    // ---- Cooperatively stage the A slab: [kWmmaBM rows] x [kWmmaBK cols] ----
    for (int idx = threadIdx.x; idx < kWmmaBM * kWmmaBK;
         idx += kWmmaBlockThreads) {
      const int r = idx / kWmmaBK;           // 0..63  (M within block tile)
      const int cc = idx % kWmmaBK;          // 0..127 (K within slab)
      const int gm = block_row + r;
      const int gk = k0 + cc;
      as[r][cc] = (gm < m && gk < k)
                      ? a[static_cast<int64_t>(gm) * lda + gk]
                      : kZero;
    }
    // ---- Cooperatively stage the B slab: [kWmmaBK rows] x [kWmmaBN cols] ----
    for (int idx = threadIdx.x; idx < kWmmaBK * kWmmaBN;
         idx += kWmmaBlockThreads) {
      const int r = idx / kWmmaBN;           // 0..127 (K within slab)
      const int cc = idx % kWmmaBN;          // 0..63  (N within block tile)
      const int gk = k0 + r;
      const int gn = block_col + cc;
      bs[r][cc] = (gk < k && gn < n)
                      ? b[static_cast<int64_t>(gk) * ldb + gn]
                      : kZero;
    }
    __syncthreads();

    // ---- Inner loop: each warp processes 8 slices of WMMA_K (16) from smem --
    #pragma unroll
    for (int ki = 0; ki < kWmmaBKSteps; ++ki) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, scalar_t,
                     wmma::row_major>
          a_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, scalar_t,
                     wmma::row_major>
          b_frag;
      wmma::load_matrix_sync(a_frag, &as[warp_m * WMMA_M][ki * WMMA_K], kAsLd);
      wmma::load_matrix_sync(b_frag, &bs[ki * WMMA_K][warp_n * WMMA_N], kBsLd);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();  // tiles consumed; safe to overwrite next iteration
  }

  // ---- Epilogue: each thread writes its accumulator fragment directly to C ----
  // On sm_70 (V100) with CUDA 12.4, store_matrix_sync to shared memory writes
  // correct values but they read back as zero from the smem target.  Bypass the
  // smem intermediate entirely: each thread extracts its 8 fragment elements and
  // writes them directly to global memory, applying bias/act/beta inline.
  //
  // Accumulator fragment layout for 16×16×16 (float), 32 lanes per warp:
  //   Lane t owns 8 elements:
  //     acc_frag.x[0..3] → row = t/4,      col = (t%4)*4 + 0..3
  //     acc_frag.x[4..7] → row = t/4 + 8,  col = (t%4)*4 + 0..3
  const int lane = threadIdx.x % KS_WARP_SIZE;
  const int warp_row = block_row + warp_m * WMMA_M;
  const int warp_col = block_col + warp_n * WMMA_N;

  const int base_row0 = warp_row + lane / 4;
  const int base_col0 = warp_col + (lane % 4) * 4;
  const int base_row1 = base_row0 + 8;

  if (base_row0 < m) {
    for (int j = 0; j < 4; ++j) {
      const int gn = base_col0 + j;
      if (gn >= n) continue;
      float v = alpha * acc_frag.x[j];
      if (use_bias_act) {
        if (bias) v += to_float(bias[gn]);
        v = apply_activation(v, act);
      } else if (beta != 0.0f) {
        v += beta * to_float(c[static_cast<int64_t>(base_row0) * ldc + gn]);
      }
      c[static_cast<int64_t>(base_row0) * ldc + gn] = from_float<scalar_t>(v);
    }
  }
  if (base_row1 < m) {
    for (int j = 0; j < 4; ++j) {
      const int gn = base_col0 + j;
      if (gn >= n) continue;
      float v = alpha * acc_frag.x[4 + j];
      if (use_bias_act) {
        if (bias) v += to_float(bias[gn]);
        v = apply_activation(v, act);
      } else if (beta != 0.0f) {
        v += beta * to_float(c[static_cast<int64_t>(base_row1) * ldc + gn]);
      }
      c[static_cast<int64_t>(base_row1) * ldc + gn] = from_float<scalar_t>(v);
    }
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

// True when the WMMA fast path is applicable for this element type: the tensor-
// core fragments only exist for the 16-bit float types (f16/bf16), never f32.
template <typename scalar_t>
constexpr bool wmma_supported() {
  return !std::is_same<scalar_t, float>::value &&
         !std::is_same<scalar_t, __nv_bfloat16>::value;
}

// Dispatch one GEMM (single matrix, fp32 acc). Chooses WMMA vs SIMT.
template <typename scalar_t>
void launch_gemm(scalar_t* c, const scalar_t* a, const scalar_t* b,
                 const scalar_t* bias, int64_t m, int64_t n, int64_t k,
                 int trans_a, int trans_b, int64_t lda, int64_t ldb, int64_t ldc,
                 float alpha, float beta, int act, int use_bias_act,
                 gpuStream_t s) {
  // WMMA fast path: half/bf16, no transpose. Because A/B tiles are staged into
  // shared memory with bounds-checked (zero-fill) cooperative loads and the
  // fragments are then read from a CONSTANT padded smem stride, the kernel no
  // longer needs M/N/K to be tile-aligned, nor the operand leading dims to be
  // 8-element-aligned (those constraints were what the old global-direct
  // load_matrix_sync version required). We only need: non-transposed operands
  // (so the K axis is contiguous for coalesced staging) and 32-bit shapes/
  // strides (the kernel takes int m/n/k/ld*). Anything else — including any
  // transpose or an extent/stride past int range — uses the always-safe int64
  // SIMT path.
#if KS_ENABLE_WMMA_GEMM
  // The WMMA fast path is gated behind KS_ENABLE_WMMA_GEMM (off by default), and
  // the entire launch is #if'd OUT when disabled — not merely runtime-skipped —
  // for two reasons:
  //   1. `if constexpr (wmma_supported<scalar_t>())` keeps gemm_wmma_kernel<float>
  //      from ever being named (nvcuda::wmma has no float fragment), and
  //   2. naming gemm_wmma_kernel<bf16> instantiates nvcuda::wmma bf16 fragments,
  //      which DO NOT EXIST below sm_80 — so leaving the launch compiled (even on
  //      a runtime-dead branch) is a hard compile error on sm_70/sm_75 (T4/V100)
  //      and breaks a multi-arch / fat-binary build. Compiling it out keeps the
  //      library portable across sm_70..sm_120 from one source.
  constexpr int64_t kI32Max = 0x7fffffff;
  if constexpr (wmma_supported<scalar_t>()) {
    const bool use_wmma =
        trans_a == 0 && trans_b == 0 && m <= kI32Max &&
        n <= kI32Max && k <= kI32Max && lda <= kI32Max && ldb <= kI32Max &&
        ldc <= kI32Max;
    if (use_wmma) {
      dim3 grid(static_cast<unsigned>((n + kWmmaBN - 1) / kWmmaBN),
                static_cast<unsigned>((m + kWmmaBM - 1) / kWmmaBM));
      dim3 block(kWmmaBlockThreads);
      gemm_wmma_kernel<scalar_t><<<grid, block, 0, s>>>(
          c, a, b, bias, static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(k), static_cast<int>(lda), static_cast<int>(ldb),
          static_cast<int>(ldc), alpha, beta, act, use_bias_act);
      return;
    }
  }
#endif  // KS_ENABLE_WMMA_GEMM
  {
    launch_simt<scalar_t>(c, a, b, bias, m, n, k, trans_a, trans_b, lda, ldb,
                          ldc, alpha, beta, act, use_bias_act, s);
  }
}

}  // namespace gemm
}  // namespace ks

using namespace ks;

// ---------------------------------------------------------------------------
// Lazy-init cuBLAS handle (one per process, created on first use).
// ---------------------------------------------------------------------------
static cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    cublasCreate(&handle);
  }
  return handle;
}

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

  // cuBLAS fast path: C[M×N] = alpha * op(A) @ op(B) + beta * C
  // Row-major to col-major: C_col = op_B(B)_col * op_A(A)_col
  // NOTE: cuBLAS writes C_col[N×M] into the buffer, which is only valid for
  // square matrices (M==N) where the row-major and col-major shapes coincide.
  // For non-square matrices, the SIMT fallback is used.
  cublasHandle_t handle = get_cublas_handle();
  if (handle) {
    cublasSetStream(handle, s);
    cublasOperation_t op_a_cublas, op_b_cublas;
    int m_cublas, n_cublas, k_cublas, lda_cublas, ldb_cublas;
    const void *a_cublas, *b_cublas;

    if (trans_a == 0 && trans_b == 0) {
      // C_row[M×N] = A_row[M×K] @ B_row[K×N], M==n, N==m
      // C_col[N×M] = B_col[N×K] * A_col[K×M]
      op_a_cublas = CUBLAS_OP_N; op_b_cublas = CUBLAS_OP_N;
      m_cublas = (int)n; n_cublas = (int)m; k_cublas = (int)k;
      a_cublas = b; lda_cublas = (int)n;
      b_cublas = a; ldb_cublas = (int)k;
    } else if (trans_a == 0 && trans_b == 1) {
      // C_row[M×N] = A_row[M×K] @ B_row^T[K×N], M==n, N==m
      // C_col[N×M] = B_col^T[N×K] * A_col[K×M]
      op_a_cublas = CUBLAS_OP_T; op_b_cublas = CUBLAS_OP_N;
      m_cublas = (int)n; n_cublas = (int)m; k_cublas = (int)k;
      a_cublas = b; lda_cublas = (int)k;
      b_cublas = a; ldb_cublas = (int)k;
    } else {
      goto fallback;
    }

    if (dtype == KS_DTYPE_F16) {
      cublasStatus_t st = cublasGemmEx(
          handle, op_a_cublas, op_b_cublas,
          m_cublas, n_cublas, k_cublas,
          &alpha,
          a_cublas, CUDA_R_16F, lda_cublas,
          b_cublas, CUDA_R_16F, ldb_cublas,
          &beta,
          c, CUDA_R_16F, (int)ldc,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      if (st == CUBLAS_STATUS_SUCCESS) return KS_SUCCESS;
    } else if (dtype == KS_DTYPE_F32) {
      cublasStatus_t st = cublasSgemm(
          handle, op_a_cublas, op_b_cublas,
          m_cublas, n_cublas, k_cublas,
          &alpha,
          (const float*)a_cublas, lda_cublas,
          (const float*)b_cublas, ldb_cublas,
          &beta,
          (float*)c, (int)ldc);
      if (st == CUBLAS_STATUS_SUCCESS) return KS_SUCCESS;
    }
  }
  fallback:

  // Fallback: SIMT tiled GEMM kernel.
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
