// kernel-set — FlashAttention-2 forward (dense + variable-length prefill).
//
// =====================  WHY THE OLD KERNEL WAS ~260x SLOW  ==================
// The previous version launched ONE CUDA block per (batch, head, query-ROW)
// with blockDim.x == head_dim (64-128 threads) and computed every key's score
// with a *full block reduction* (block_reduce_sum + __syncthreads) one key at a
// time. That is catastrophic:
//   * blocks = batch*num_heads*seqlen_q, but each does almost no parallel work
//     per score — a head_dim-wide reduction per key, fully serialized;
//   * a __syncthreads() per key (thousands per row) → the SMs stall;
//   * K/V were re-read from global memory for every query row (no reuse across
//     rows), so the kernel is both latency- and bandwidth-bound.
// Net effect: ~0.2 TFLOP/s on an L4.
//
// =====================  NEW DESIGN (FlashAttention-2)  =====================
// Grid  : (ceil(seqlen_q / BLOCK_M), num_heads, batch).
// Block : BLOCK_M warps (one warp per query row) → BLOCK_M query rows / block.
// A Q-block of BLOCK_M rows is loaded ONCE (into per-lane registers). The K/V
// sequence is streamed in BLOCK_N-key tiles that are staged in shared memory by
// the *whole* block and therefore reused by all BLOCK_M warps. Each warp then:
//   1. dot(q_row, k_j) as a warp reduction (no __syncthreads, no block reduce);
//   2. online-softmax update in fp32 (running max m, denom l, acc[]), rescaling
//      acc/l whenever a larger score appears (Dao 2023, FA2);
//   3. acc += p * v_j accumulated per-lane across the head_dim it owns.
// Each lane owns EPL = head_dim / 32 contiguous dims of its warp's query row,
// its Q components, and its output accumulator. The score is a warp_reduce_sum
// over those EPL partial products → one fast shuffle reduction per key.
//
// This keeps everything in fp32, materializes no [seqlen_q, seqlen_k] matrix,
// handles GQA/MQA (num_kv_heads < num_heads), right-aligned causal masking,
// the 1/sqrt(head_dim) default scale, and head_dim 64/128 (up to kMaxHeadDim).
// softmax_lse = m + log(l) is emitted in scaled-score space, matching the
// backward kernel's recovery P_ij = exp(scale*q.k - lse_i).
//
// Parallelism now scales as batch*num_heads*ceil(seqlen_q/BLOCK_M) blocks, each
// doing BLOCK_M*BLOCK_N*head_dim useful FLOPs with warp-local reductions and
// shared-memory K/V reuse — a large constant-factor and occupancy win over the
// one-row-per-block design. Expected: easily >10-30x the old throughput.
//
// FURTHER TUNING (documented, intentionally not done to stay correctness-first):
//   * Use cp.async (memcpy_async) to double-buffer the K/V tiles so the next
//     tile loads while the current tile is consumed.
//   * Replace the warp-reduce dot product / V accumulation with wmma/mma.sync
//     register tiles (Q,Kᵀ → S then S,V) for tensor-core throughput, à la the
//     CUTLASS FA2 kernel. That requires a 2-D warp tiling over query rows.
//   * Vectorize the global K/V loads with 128-bit (Vec<scalar_t, N>) transfers.
#include <cstdlib>  // malloc/free for the dense-path prefix-sum scratch

#include "kernel_set/attention.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"
#include "attention_common.cuh"

namespace ks {
namespace attention {

// Query rows handled per block (== warps per block). 4 warps = 128 threads is a
// good occupancy point and keeps the shared-memory K/V tile modest.
constexpr int kBlockM = 4;
// Keys staged into shared memory per streaming iteration. 64 keys * head_dim
// (<=256) * 2 (K+V) * 4B = up to 128 KB at head_dim 256, 64 KB at head_dim 128,
// 32 KB at head_dim 64. We size shared dynamically and only require the default
// 48 KB carveout for head_dim<=64 with BLOCK_N=64; for head_dim 128/256 we shrink
// BLOCK_N on the host so the footprint stays within the 48 KB default (no opt-in
// cudaFuncSetAttribute needed). See fwd_kv_tile()/fwd_smem_bytes() below.
constexpr int kBlockNMax = 64;

// Resolve the K/V tile width for a head_dim so the shared K+V tile stays within
// the 48 KB default dynamic-smem limit (K tile + V tile = 2*BLOCK_N*head_dim
// floats). 48 KB / (2*4B*head_dim) keys; clamp to [16, kBlockNMax] and to a
// warp-friendly value. Must match what the kernel iterates with (passed in).
KS_HDI int fwd_kv_tile(int head_dim) {
  // Bytes available for the two fp32 tiles (leave a little slack < 48 KB).
  constexpr int kSmemBudget = 46 * 1024;
  int by_budget = kSmemBudget / (2 * static_cast<int>(sizeof(float)) * head_dim);
  int bn = by_budget < kBlockNMax ? by_budget : kBlockNMax;
  if (bn > 32) bn = 32 * (bn / 32);  // round down to a multiple of the warp size
  if (bn < 16) bn = 16;              // a sane floor
  return bn;
}

// One block == one (batch, head, query-BLOCK of kBlockM rows).
// Warp w (0..kBlockM-1) owns query row (q_block_start + w). Lane `lane` owns the
// head-dim slice [lane*EPL, lane*EPL+EPL) of that row (EPL = head_dim/32).
//
// q     : [total_q, num_heads, head_dim]               (varlen-packed)
// k,v   : [total_kv, num_kv_heads, head_dim]
// out   : [total_q, num_heads, head_dim]
// lse   : [num_heads, total_q] fp32 (optional; lse_i = m_i + log l_i, scaled).
template <typename scalar_t>
KS_GLOBAL void flash_attn_fwd_kernel(
    scalar_t* __restrict__ out, float* __restrict__ softmax_lse,
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ cu_seqlens_k, int num_heads, int num_kv_heads,
    int head_dim, float scale, int causal, int total_q, int block_n) {
  const int warp = threadIdx.x / KS_WARP_SIZE;   // 0..kBlockM-1 (query row slot)
  const int lane = threadIdx.x % KS_WARP_SIZE;   // 0..31
  const int head = blockIdx.y;                   // query head
  const int batch = blockIdx.z;                  // sequence index

  // Per-lane element count along head_dim (head_dim is a multiple of 32).
  const int epl = head_dim / KS_WARP_SIZE;       // EPL: 2 (hd64) / 4 (hd128) ...
  const int d0 = lane * epl;                     // first dim this lane owns

  // Resolve this sequence's [q_start, q_end) and [k_start, k_end).
  const int q_start = cu_seqlens_q[batch];
  const int q_end = cu_seqlens_q[batch + 1];
  const int k_start = cu_seqlens_k[batch];
  const int k_end = cu_seqlens_k[batch + 1];
  const int seqlen_q = q_end - q_start;
  const int seqlen_k = k_end - k_start;

  const int qi = blockIdx.x * kBlockM + warp;    // query row within this seq
  const bool row_valid = (qi < seqlen_q);

  const int kv_head = head / gqa_group_size(num_heads, num_kv_heads);
  // Right-aligned causal masking (matches FlashAttention when seqlen_q !=
  // seqlen_k): query position qi attends to keys [0 .. causal_offset + qi].
  const int causal_offset = seqlen_k - seqlen_q;
  // Per-row last key index attended (exclusive). Differs per warp under causal.
  const int kv_max = causal ? min(seqlen_k, causal_offset + qi + 1) : seqlen_k;

  // The K/V tile loop (and its __syncthreads()) MUST be uniform across the whole
  // block — every warp has to reach the same barriers — so it is bounded by the
  // *block-wide* maximum kv_max (the largest valid query row in this block).
  // Each warp still applies its own per-key causal mask via `kv_max` above.
  const int block_last_qi =
      min(blockIdx.x * kBlockM + (kBlockM - 1), seqlen_q - 1);
  const int block_kv_max =
      (block_last_qi < 0)
          ? 0
          : (causal ? min(seqlen_k, causal_offset + block_last_qi + 1)
                    : seqlen_k);

  // Load this warp's Q row into registers (per-lane slice). Up to kMaxHeadDim/32.
  constexpr int kMaxEpl = kMaxHeadDim / 32;      // 8 at head_dim 256
  float q_reg[kMaxEpl];
  float acc[kMaxEpl];
#pragma unroll
  for (int e = 0; e < kMaxEpl; ++e) {
    q_reg[e] = 0.f;
    acc[e] = 0.f;
  }
  if (row_valid) {
    const int64_t q_row = static_cast<int64_t>(q_start + qi);
    const scalar_t* q_ptr = q + (q_row * num_heads + head) * head_dim;
#pragma unroll
    for (int e = 0; e < kMaxEpl; ++e) {
      if (e < epl) q_reg[e] = to_float(q_ptr[d0 + e]);
    }
  }

  float m = -3.4028235e38f;  // running max of scaled scores
  float l = 0.f;             // running sum of exp(scaled_score - m)

  // Shared K/V tiles, staged by the whole block and reused by all kBlockM warps.
  // Layout: [block_n * head_dim] K | [block_n * head_dim] V (row-major).
  extern __shared__ float smem_raw[];
  float* k_tile = smem_raw;                       // block_n * head_dim
  float* v_tile = k_tile + block_n * head_dim;    // block_n * head_dim

  const int nthreads = blockDim.x;

  for (int tile0 = 0; tile0 < block_kv_max; tile0 += block_n) {
    const int tile_n = min(block_n, block_kv_max - tile0);

    // Cooperatively stage the K/V tile into shared memory. All threads in the
    // block load (coalesced over head_dim) so the tile is shared by every warp.
    for (int idx = threadIdx.x; idx < tile_n * head_dim; idx += nthreads) {
      const int j = idx / head_dim;             // key within tile
      const int dd = idx % head_dim;            // dim
      const int64_t kv_row = static_cast<int64_t>(k_start + tile0 + j);
      const scalar_t* k_ptr = k + (kv_row * num_kv_heads + kv_head) * head_dim;
      const scalar_t* v_ptr = v + (kv_row * num_kv_heads + kv_head) * head_dim;
      k_tile[idx] = to_float(k_ptr[dd]);
      v_tile[idx] = to_float(v_ptr[dd]);
    }
    __syncthreads();

    if (row_valid) {
      // Number of keys in this tile that THIS warp's query row attends to. The
      // tile loop runs to the block-wide max, so a tile may contain keys beyond
      // this row's own causal limit (kv_max); skip those here.
      const int row_tile_n = min(tile_n, kv_max - tile0);
      // Score every attended key in the tile against this warp's query row, then
      // do the online-softmax accumulation. Each key is a warp-reduced dot
      // product over the EPL dims each lane owns — one shuffle reduction per key.
      for (int j = 0; j < row_tile_n; ++j) {
        const float* kj = k_tile + j * head_dim;
        float partial = 0.f;
#pragma unroll
        for (int e = 0; e < kMaxEpl; ++e) {
          if (e < epl) partial += q_reg[e] * kj[d0 + e];
        }
        const float dot = warp_reduce_sum(partial);  // broadcast to all lanes
        const float s = dot * scale;

        // Online softmax: update running max, rescale acc/l, add this key.
        const float m_new = fmaxf(m, s);
        const float correction = rescale_factor(m, m_new);  // exp(m-m_new) or 0
        const float p = __expf(s - m_new);
#pragma unroll
        for (int e = 0; e < kMaxEpl; ++e) {
          if (e < epl) acc[e] = acc[e] * correction + p * v_tile[j * head_dim + d0 + e];
        }
        l = l * correction + p;
        m = m_new;
      }
    }
    __syncthreads();  // all warps done reading the tile before it is overwritten
  }

  if (!row_valid) return;

  // Normalize and write the output row. l == 0 only if every key was masked,
  // which cannot happen here (qi always attends to key 0), but guard anyway.
  const float inv_l = (l > 0.f) ? (1.0f / l) : 0.f;
  const int64_t q_row = static_cast<int64_t>(q_start + qi);
  scalar_t* out_ptr = out + (q_row * num_heads + head) * head_dim;
#pragma unroll
  for (int e = 0; e < kMaxEpl; ++e) {
    if (e < epl) out_ptr[d0 + e] = from_float<scalar_t>(acc[e] * inv_l);
  }

  if (softmax_lse != nullptr && lane == 0) {
    // log-sum-exp in scaled-score space; index [head, total_q] row-major.
    const float lse = (l > 0.f) ? (m + logf(l)) : (-3.4028235e38f);
    softmax_lse[static_cast<int64_t>(head) * total_q + q_row] = lse;
  }
}

// Shared-memory bytes required by the forward kernel for a given head_dim/tile.
inline size_t fwd_smem_bytes(int head_dim, int block_n) {
  return sizeof(float) * (2u * static_cast<size_t>(block_n) * head_dim);
}

// Common launcher used by both the dense and varlen entry points. The caller
// supplies device prefix-sum arrays describing the q/k sequence boundaries.
template <typename scalar_t>
ks_status_t launch_flash_fwd(scalar_t* out, float* lse, const scalar_t* q,
                             const scalar_t* k, const scalar_t* v,
                             const int32_t* cu_q, const int32_t* cu_k,
                             int batch, int max_seqlen_q, int num_heads,
                             int num_kv_heads, int head_dim, float scale,
                             int causal, int total_q, gpuStream_t s) {
  const int block_n = fwd_kv_tile(head_dim);
  const unsigned q_blocks =
      static_cast<unsigned>((max_seqlen_q + kBlockM - 1) / kBlockM);
  const dim3 grid(q_blocks, static_cast<unsigned>(num_heads),
                  static_cast<unsigned>(batch));
  const dim3 block(static_cast<unsigned>(kBlockM * KS_WARP_SIZE));
  const size_t smem = fwd_smem_bytes(head_dim, block_n);
  flash_attn_fwd_kernel<scalar_t><<<grid, block, smem, s>>>(
      out, lse, q, k, v, cu_q, cu_k, num_heads, num_kv_heads, head_dim, scale,
      causal, total_q, block_n);
  return KS_SUCCESS;
}

}  // namespace attention
}  // namespace ks

using namespace ks;

namespace {

// Only f32/f16/bf16 are dispatched by KS_DISPATCH_FLOATING_TYPES. Reject other
// dtypes up front so allocation paths never leak scratch on the default case.
inline ks_status_t check_float_dtype(ks_dtype_t dtype) {
  if (dtype != KS_DTYPE_F32 && dtype != KS_DTYPE_F16 && dtype != KS_DTYPE_BF16)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE, "flash_attn: needs f32/f16/bf16");
  return KS_SUCCESS;
}

// Validate the fields shared by both forward entry points.
inline ks_status_t check_fwd_common(int num_heads, int num_kv_heads,
                                    int head_dim) {
  if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "flash_attn: bad head config");
  if (num_kv_heads > num_heads || (num_heads % num_kv_heads) != 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "flash_attn: num_heads must be a multiple of num_kv_heads");
  if (head_dim > ks::attention::kMaxHeadDim)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE, "flash_attn: head_dim too large");
  // Each warp reduces a head-dim dot product with the lanes splitting head_dim
  // evenly, so head_dim must be a multiple of the warp size (32). The supported
  // configs being 64/128 (and up to kMaxHeadDim) all satisfy this.
  if ((head_dim & 31) != 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "flash_attn: head_dim must be a multiple of 32 (use 64/128)");
  return KS_SUCCESS;
}

}  // namespace

extern "C" {

ks_status_t ks_flash_attn_varlen(
    void* out, void* softmax_lse, const void* q, const void* k, const void* v,
    const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_k, int batch,
    int max_seqlen_q, int max_seqlen_k, int num_heads, int num_kv_heads,
    int head_dim, float softmax_scale, int causal, ks_dtype_t dtype,
    ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(q);
  KS_CHECK_PTR(k);
  KS_CHECK_PTR(v);
  KS_CHECK_PTR(cu_seqlens_q);
  KS_CHECK_PTR(cu_seqlens_k);
  if (batch <= 0 || max_seqlen_q <= 0 || max_seqlen_k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_flash_attn_varlen: shape");
  ks_status_t st = check_fwd_common(num_heads, num_kv_heads, head_dim);
  if (st != KS_SUCCESS) return st;
  st = check_float_dtype(dtype);
  if (st != KS_SUCCESS) return st;

  const float scale = attention::resolve_scale(softmax_scale, head_dim);
  auto s = to_stream(stream);
  (void)max_seqlen_k;

  // softmax_lse is [num_heads, total_q] where total_q == cu_seqlens_q[batch]
  // (the packed sum of query lengths). The kernel writes lse at the absolute
  // row index (q_start + qi), so it needs total_q as the row stride. That value
  // only exists on the device; read just the final prefix-sum element to host.
  // This is skipped entirely when lse is not requested (inference path).
  int total_q = 0;
  if (softmax_lse != nullptr) {
    gpuError_t e = gpuMemcpyAsync(&total_q, cu_seqlens_q + batch, sizeof(int),
                                  gpuMemcpyDeviceToHost, s);
    if (e != gpuSuccess)
      KS_RETURN_ERROR(KS_ERROR_CUDA, "ks_flash_attn_varlen: cu_seqlens copy");
    e = gpuStreamSynchronize(s);
    if (e != gpuSuccess)
      KS_RETURN_ERROR(KS_ERROR_CUDA, "ks_flash_attn_varlen: sync");
  }

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_flash_attn_varlen", {
    attention::launch_flash_fwd<scalar_t>(
        static_cast<scalar_t*>(out), static_cast<float*>(softmax_lse),
        static_cast<const scalar_t*>(q), static_cast<const scalar_t*>(k),
        static_cast<const scalar_t*>(v), cu_seqlens_q, cu_seqlens_k, batch,
        max_seqlen_q, num_heads, num_kv_heads, head_dim, scale, causal, total_q,
        s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_flash_attn(void* out, void* softmax_lse, const void* q,
                          const void* k, const void* v, int batch,
                          int seqlen_q, int seqlen_k, int num_heads,
                          int num_kv_heads, int head_dim, float softmax_scale,
                          int causal, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(q);
  KS_CHECK_PTR(k);
  KS_CHECK_PTR(v);
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_flash_attn: bad shape");
  ks_status_t st = check_fwd_common(num_heads, num_kv_heads, head_dim);
  if (st != KS_SUCCESS) return st;
  st = check_float_dtype(dtype);
  if (st != KS_SUCCESS) return st;

  const float scale = attention::resolve_scale(softmax_scale, head_dim);
  auto s = to_stream(stream);

  // Build uniform prefix-sum arrays on device so the dense path reuses the
  // varlen kernel verbatim: cu_q[i] = i*seqlen_q, cu_k[i] = i*seqlen_k.
  // total_q for the lse stride is batch*seqlen_q.
  const int total_q = batch * seqlen_q;
  int32_t* cu_q = nullptr;
  int32_t* cu_k = nullptr;
  const size_t n = static_cast<size_t>(batch + 1) * sizeof(int32_t);
  gpuError_t e = gpuMalloc(reinterpret_cast<void**>(&cu_q), n);
  if (e != gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_OUT_OF_MEMORY, "ks_flash_attn: cu_q alloc");
  e = gpuMalloc(reinterpret_cast<void**>(&cu_k), n);
  if (e != gpuSuccess) {
    gpuFree(cu_q);
    KS_RETURN_ERROR(KS_ERROR_OUT_OF_MEMORY, "ks_flash_attn: cu_k alloc");
  }

  // Fill prefix sums on host then upload (batch is small; one tiny copy each).
  // Use a stack buffer to avoid extra allocations for the common small batch.
  {
    constexpr int kStackMax = 1024;
    int32_t stack_q[kStackMax + 1];
    int32_t stack_k[kStackMax + 1];
    int32_t* hq = stack_q;
    int32_t* hk = stack_k;
    int32_t* heap_q = nullptr;
    int32_t* heap_k = nullptr;
    if (batch > kStackMax) {
      heap_q = static_cast<int32_t*>(malloc(n));
      heap_k = static_cast<int32_t*>(malloc(n));
      if (heap_q == nullptr || heap_k == nullptr) {
        free(heap_q);
        free(heap_k);
        gpuFree(cu_q);
        gpuFree(cu_k);
        KS_RETURN_ERROR(KS_ERROR_OUT_OF_MEMORY, "ks_flash_attn: host buf");
      }
      hq = heap_q;
      hk = heap_k;
    }
    for (int i = 0; i <= batch; ++i) {
      hq[i] = i * seqlen_q;
      hk[i] = i * seqlen_k;
    }
    e = gpuMemcpyAsync(cu_q, hq, n, gpuMemcpyHostToDevice, s);
    if (e == gpuSuccess)
      e = gpuMemcpyAsync(cu_k, hk, n, gpuMemcpyHostToDevice, s);
    if (e == gpuSuccess) e = gpuStreamSynchronize(s);  // hq/hk freed below
    free(heap_q);
    free(heap_k);
    if (e != gpuSuccess) {
      gpuFree(cu_q);
      gpuFree(cu_k);
      KS_RETURN_ERROR(KS_ERROR_CUDA, "ks_flash_attn: cu_seqlens upload");
    }
  }

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_flash_attn", {
    attention::launch_flash_fwd<scalar_t>(
        static_cast<scalar_t*>(out), static_cast<float*>(softmax_lse),
        static_cast<const scalar_t*>(q), static_cast<const scalar_t*>(k),
        static_cast<const scalar_t*>(v), cu_q, cu_k, batch, seqlen_q, num_heads,
        num_kv_heads, head_dim, scale, causal, total_q, s);
  });

  // Capture launch-config errors immediately, then sync to keep the prefix-sum
  // scratch alive until the kernel has consumed it.
  const gpuError_t launch_err = gpuGetLastError();
  e = gpuStreamSynchronize(s);
  gpuFree(cu_q);
  gpuFree(cu_k);
  if (launch_err != gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_CUDA,
                    std::string("ks_flash_attn: launch ") +
                        gpuGetErrorString(launch_err));
  if (e != gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_CUDA, "ks_flash_attn: sync");
  return KS_SUCCESS;
}

}  // extern "C"
