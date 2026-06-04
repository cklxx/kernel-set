// kernel-set — MoE grouped GEMM (per-expert C = A @ B_e over CSR row groups).
//
// For each expert e the rows [off_e, off_{e+1}) of the permuted activation
// matrix A [total_rows, k] are multiplied by that expert's weight slab
// B_e = b[e] [k, n], writing C [total_rows, n]. This is the compute core of a
// fused MoE layer; it is modeled on Megatron's grouped GEMM and vLLM/SGLang's
// `grouped_gemm` (Triton) tile scheduler.
//
// Design
// ------
// A classic shared-memory tiled GEMM (TILE_M x TILE_N output tile, TILE_K deep)
// with fp32 accumulation regardless of storage dtype. The difficulty unique to
// grouped GEMM is that each expert's row count is data-dependent and only known
// on the device (in `expert_offsets`). We therefore:
//
//   1. Build a *tile schedule* on device: a flat list mapping a linear tile id
//      to (expert, m_tile_start). The number of m-tiles for expert e is
//      ceil(rows_e / TILE_M); summed over experts this is <= ceil(total_rows /
//      TILE_M) + num_experts, which bounds the launch grid with only ~num_experts
//      slack blocks (no per-expert over-launch).
//   2. Launch the GEMM with grid.x = that upper bound and grid.y = n-tiles.
//      Each block looks up its (expert, m_start) from the schedule and computes
//      its TILE_M x TILE_N output tile; blocks past the real tile count exit.
//
// Further tuning (documented, not implemented to stay correct + portable): swap
// the inner loop for wmma / mma.sync tensor-core tiles on the half paths, use
// cp.async double-buffering for the K loop, or hand off to CUTLASS grouped GEMM.
#include "kernel_set/moe.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace moe {

// Output tile + K-depth. 16x16 with a 16-wide K step keeps shared memory small
// (2 * 16 * 16 floats) and the block at 256 threads (one thread per C element).
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 16;
constexpr int kGemmThreads = kTileM * kTileN;  // 256

// -------------------------------------------------------------------------
// Schedule builder: fill (tile_expert, tile_mstart) and the real tile count.
// Single thread; num_experts is small so a serial scan is fine and avoids a
// device-wide prefix-sum dependency.
// -------------------------------------------------------------------------
KS_GLOBAL void gemm_build_schedule_kernel(
    int32_t* __restrict__ tile_expert, int32_t* __restrict__ tile_mstart,
    int32_t* __restrict__ num_tiles, const int32_t* __restrict__ expert_offsets,
    int num_experts) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int32_t t = 0;
  for (int e = 0; e < num_experts; ++e) {
    const int32_t start = expert_offsets[e];
    const int32_t end = expert_offsets[e + 1];
    for (int32_t m = start; m < end; m += kTileM) {
      tile_expert[t] = e;
      tile_mstart[t] = m;
      ++t;
    }
  }
  *num_tiles = t;
}

// -------------------------------------------------------------------------
// Tiled grouped GEMM. Block (x = linear tile id, y = n-tile). Each thread owns
// one C element of the TILE_M x TILE_N tile; the K dimension is streamed through
// shared memory in TILE_K chunks. fp32 accumulation.
// -------------------------------------------------------------------------
template <typename scalar_t>
KS_GLOBAL void grouped_gemm_kernel(
    scalar_t* __restrict__ c, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, const int32_t* __restrict__ tile_expert,
    const int32_t* __restrict__ tile_mstart,
    const int32_t* __restrict__ num_tiles,
    const int32_t* __restrict__ expert_offsets, int64_t n, int64_t k) {
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles) return;  // slack block

  const int expert = tile_expert[tile_id];
  const int64_t m_start = tile_mstart[tile_id];
  const int64_t m_end = expert_offsets[expert + 1];
  const int64_t n_start = static_cast<int64_t>(blockIdx.y) * kTileN;

  // This thread's output element within the tile.
  const int tr = threadIdx.x / kTileN;  // row in tile [0, kTileM)
  const int tc = threadIdx.x % kTileN;  // col in tile [0, kTileN)
  const int64_t row = m_start + tr;     // global A/C row
  const int64_t col = n_start + tc;     // global C/B col

  // Per-expert weight slab B_e = b + expert * (k * n), row-major [k, n].
  const scalar_t* be = b + static_cast<int64_t>(expert) * k * n;

  __shared__ float as[kTileM][kTileK];
  __shared__ float bs[kTileK][kTileN];

  float acc = 0.f;
  for (int64_t k0 = 0; k0 < k; k0 += kTileK) {
    // Cooperative load of the A and B sub-tiles into shared memory (fp32).
    // 256 threads load a 16x16 A tile and a 16x16 B tile (one element each).
    const int a_r = threadIdx.x / kTileK;  // [0,16)
    const int a_c = threadIdx.x % kTileK;  // [0,16)
    const int64_t a_row = m_start + a_r;
    const int64_t a_col = k0 + a_c;
    as[a_r][a_c] =
        (a_row < m_end && a_col < k)
            ? to_float(a[a_row * k + a_col])
            : 0.f;

    const int b_r = threadIdx.x / kTileN;  // [0,16) -> k index
    const int b_c = threadIdx.x % kTileN;  // [0,16) -> n index
    const int64_t b_row = k0 + b_r;
    const int64_t b_col = n_start + b_c;
    bs[b_r][b_c] =
        (b_row < k && b_col < n) ? to_float(be[b_row * n + b_col]) : 0.f;

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) acc += as[tr][kk] * bs[kk][tc];

    __syncthreads();
  }

  if (row < m_end && col < n) c[row * n + col] = from_float<scalar_t>(acc);
}

}  // namespace moe
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_moe_grouped_gemm(void* c, const void* a, const void* b,
                                const int32_t* expert_offsets, int num_experts,
                                int64_t total_rows, int64_t n, int64_t k,
                                ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(c);
  KS_CHECK_PTR(a);
  KS_CHECK_PTR(b);
  KS_CHECK_PTR(expert_offsets);
  if (num_experts <= 0 || total_rows < 0 || n <= 0 || k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_moe_grouped_gemm: bad shape");
  if (total_rows == 0) return KS_SUCCESS;  // nothing routed
  // Validate the dtype up front so we never leak the schedule scratch on the
  // unsupported-dtype path of KS_DISPATCH_FLOATING_TYPES (which returns early).
  if (dtype != KS_DTYPE_F32 && dtype != KS_DTYPE_F16 && dtype != KS_DTYPE_BF16)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                    "ks_moe_grouped_gemm: unsupported dtype");

  auto s = to_stream(stream);

  // Upper bound on m-tiles across all experts: sum_e ceil(rows_e/TILE_M)
  // <= ceil(total_rows/TILE_M) + num_experts.
  const int64_t max_tiles =
      (total_rows + moe::kTileM - 1) / moe::kTileM + num_experts;

  // Internal scratch: tile_expert | tile_mstart | num_tiles.
  int32_t* sched = nullptr;
  const size_t sched_bytes =
      (static_cast<size_t>(2) * max_tiles + 1) * sizeof(int32_t);
  ks::gpuError_t me =
      ks::gpuMalloc(reinterpret_cast<void**>(&sched), sched_bytes);
  if (me != ks::gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_OUT_OF_MEMORY,
                    "ks_moe_grouped_gemm: schedule alloc");
  int32_t* tile_expert = sched;
  int32_t* tile_mstart = sched + max_tiles;
  int32_t* num_tiles = sched + 2 * max_tiles;

  moe::gemm_build_schedule_kernel<<<1, 1, 0, s>>>(
      tile_expert, tile_mstart, num_tiles, expert_offsets, num_experts);

  const int64_t n_tiles = (n + moe::kTileN - 1) / moe::kTileN;
  const dim3 grid(static_cast<unsigned>(max_tiles),
                  static_cast<unsigned>(n_tiles));
  const dim3 block(moe::kGemmThreads);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_moe_grouped_gemm", {
    moe::grouped_gemm_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(c), static_cast<const scalar_t*>(a),
        static_cast<const scalar_t*>(b), tile_expert, tile_mstart, num_tiles,
        expert_offsets, n, k);
  });

  // Schedule buffer feeds the GEMM kernel, so it must outlive the launch.
  // Synchronize this stream before freeing to avoid a use-after-free race.
  ks::gpuError_t se = ks::gpuStreamSynchronize(s);
  ks::gpuFree(sched);
  if (se != ks::gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_CUDA, "ks_moe_grouped_gemm: sync");

  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
