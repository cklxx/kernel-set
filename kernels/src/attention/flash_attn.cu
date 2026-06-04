// kernel-set — FlashAttention-2 forward (dense + variable-length prefill).
//
// Modeled on FlashAttention-2 (Dao, 2023): the [seqlen_q, seqlen_k] score
// matrix is never materialized. Each CUDA block owns one (batch, query-head,
// query-row) triple and streams the K/V sequence in tiles that are staged in
// shared memory. A single fp32 online-softmax pass keeps the running max `m`,
// denominator `l`, and the (unnormalized) output accumulator `acc`, rescaling
// `acc` whenever a larger score is seen. GQA/MQA, causal masking, and the
// 1/sqrt(head_dim) default scale are all handled here.
//
// Layout: blockDim.x == head_dim. Thread `d` owns Q[d], acc[d], and one column
// of every staged K/V tile. Each key's score is a head_dim-wide dot product
// reduced across the block (block_reduce_sum). This keeps every accumulation in
// fp32 and is robust for head_dim 64/128 (and up to kMaxHeadDim).
//
// Further tuning (documented, not implemented here to stay correctness-first):
// process BLOCK_M query rows per block with a register tile + wmma/cp.async to
// hit tensor-core throughput, à la the CUTLASS FA2 kernel. The current
// one-row-per-block scheme is bandwidth-bound but numerically identical.
#include <cstdlib>  // malloc/free for the dense-path prefix-sum scratch

#include "kernel_set/attention.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"
#include "attention_common.cuh"

namespace ks {
namespace attention {

// Keys staged in shared memory per iteration. Sized so the dynamic smem
// footprint (red + K tile + V tile + scores) stays under the 48 KB default
// limit even at head_dim == 128 (33 KB), so no cudaFuncSetAttribute opt-in is
// needed. Larger tiles + the >48 KB carveout are a documented perf follow-up.
constexpr int kKvTile = 32;

// One block == one (batch, head, query-row). blockDim.x == head_dim.
//
// q     : [total_q, num_heads, head_dim]               (varlen-packed)
// k,v   : [total_kv, num_kv_heads, head_dim]
// out   : [total_q, num_heads, head_dim]
// lse   : [num_heads, total_q] fp32 (optional)
// q_base/k_base hold the per-sequence row offsets for varlen; for dense launch
// the caller bakes the (batch*seqlen) offsets into the same formula.
template <typename scalar_t>
KS_GLOBAL void flash_attn_fwd_kernel(
    scalar_t* __restrict__ out, float* __restrict__ softmax_lse,
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ cu_seqlens_k, int num_heads, int num_kv_heads,
    int head_dim, float scale, int causal, int total_q) {
  const int d = threadIdx.x;            // dim this thread owns (0..head_dim-1)
  const int head = blockIdx.y;          // query head
  const int batch = blockIdx.z;         // sequence index

  // Resolve this sequence's [q_start, q_end) and [k_start, k_end).
  const int q_start = cu_seqlens_q[batch];
  const int q_end = cu_seqlens_q[batch + 1];
  const int k_start = cu_seqlens_k[batch];
  const int k_end = cu_seqlens_k[batch + 1];
  const int seqlen_q = q_end - q_start;
  const int seqlen_k = k_end - k_start;

  const int qi = blockIdx.x;            // query row within this sequence
  if (qi >= seqlen_q) return;

  const int kv_head = head / gqa_group_size(num_heads, num_kv_heads);
  // When causal, query position qi (0-based from the end-aligned convention)
  // attends to keys [0 .. (seqlen_k - seqlen_q) + qi]. This matches
  // FlashAttention's right-aligned causal masking for seqlen_q != seqlen_k.
  const int causal_offset = seqlen_k - seqlen_q;

  const int64_t q_row = static_cast<int64_t>(q_start + qi);
  const scalar_t* q_ptr =
      q + (q_row * num_heads + head) * head_dim;

  // Per-thread registers: this thread's Q component and output accumulator.
  const float q_reg = to_float(q_ptr[d]);

  extern __shared__ float smem_raw[];
  // Layout: [head_dim] reduction scratch | [kKvTile * head_dim] K tile |
  //         [kKvTile * head_dim] V tile  | [kKvTile] scores
  float* red = smem_raw;                                  // head_dim (>= warps)
  float* k_tile = red + head_dim;                         // kKvTile*head_dim
  float* v_tile = k_tile + kKvTile * head_dim;            // kKvTile*head_dim
  float* scores = v_tile + kKvTile * head_dim;            // kKvTile

  float acc = 0.f;       // unnormalized output accumulator for dim d
  float m = -3.4028235e38f;
  float l = 0.f;

  const int kv_max = causal ? (causal_offset + qi + 1) : seqlen_k;

  for (int tile0 = 0; tile0 < kv_max; tile0 += kKvTile) {
    const int tile_n = min(kKvTile, kv_max - tile0);

    // Stage K/V tile into shared memory. Thread `d` loads column d of each row.
    for (int j = 0; j < tile_n; ++j) {
      const int64_t kv_row = static_cast<int64_t>(k_start + tile0 + j);
      const scalar_t* k_ptr = k + (kv_row * num_kv_heads + kv_head) * head_dim;
      const scalar_t* v_ptr = v + (kv_row * num_kv_heads + kv_head) * head_dim;
      k_tile[j * head_dim + d] = to_float(k_ptr[d]);
      v_tile[j * head_dim + d] = to_float(v_ptr[d]);
    }
    __syncthreads();

    // Compute the tile_n scores (one block reduction per key).
    for (int j = 0; j < tile_n; ++j) {
      const float prod = q_reg * k_tile[j * head_dim + d];
      const float dot = block_reduce_sum(prod, red);
      if (d == 0) {
        const int key_pos = tile0 + j;
        float s = dot * scale;
        if (causal && key_pos > causal_offset + qi) s = neg_inf<float>();
        scores[j] = s;
      }
      __syncthreads();  // red[] reused next key; scores[j] visible below
    }

    // Online-softmax update for this tile.
    float tile_max = -3.4028235e38f;
    for (int j = 0; j < tile_n; ++j) tile_max = fmaxf(tile_max, scores[j]);
    const float m_new = fmaxf(m, tile_max);
    const float correction = rescale_factor(m, m_new);
    acc *= correction;
    l *= correction;

    for (int j = 0; j < tile_n; ++j) {
      const float p = __expf(scores[j] - m_new);
      l += p;  // every thread adds the same p -> l stays consistent
      acc += p * v_tile[j * head_dim + d];
    }
    m = m_new;
    __syncthreads();
  }

  // Normalize and write out. l == 0 only when every key was masked (causal with
  // an all-masked row never happens because qi attends to >=1 key), but guard.
  const float inv_l = (l > 0.f) ? (1.0f / l) : 0.f;
  scalar_t* out_ptr = out + (q_row * num_heads + head) * head_dim;
  out_ptr[d] = from_float<scalar_t>(acc * inv_l);

  if (softmax_lse != nullptr && d == 0) {
    // log-sum-exp = m + log(l); index [head, total_q] row-major.
    const float lse = (l > 0.f) ? (m + logf(l)) : (-3.4028235e38f);
    softmax_lse[static_cast<int64_t>(head) * total_q + q_row] = lse;
  }
}

// Shared-memory bytes required by the forward kernel for a given head_dim.
inline size_t fwd_smem_bytes(int head_dim) {
  return sizeof(float) *
         (static_cast<size_t>(head_dim) + 2u * kKvTile * head_dim + kKvTile);
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
  const dim3 grid(static_cast<unsigned>(max_seqlen_q),
                  static_cast<unsigned>(num_heads),
                  static_cast<unsigned>(batch));
  const dim3 block(static_cast<unsigned>(head_dim));
  const size_t smem = fwd_smem_bytes(head_dim);
  flash_attn_fwd_kernel<scalar_t><<<grid, block, smem, s>>>(
      out, lse, q, k, v, cu_q, cu_k, num_heads, num_kv_heads, head_dim, scale,
      causal, total_q);
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
  // blockDim.x == head_dim and the score dot-product uses block_reduce over the
  // block's warps, so head_dim must tile evenly into warps (multiple of 32 on
  // CUDA, the supported configs being 64/128).
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
