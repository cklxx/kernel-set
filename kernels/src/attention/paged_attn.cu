// kernel-set — paged KV-cache decode (FlashDecoding + vLLM PagedAttention).
//
// Decode attends a single query position per sequence against a paged KV cache
// whose blocks are indexed indirectly through `block_tables`. Following
// FlashDecoding, attention over the KV history is an online fp32 softmax; the
// query head reads K/V one cache block at a time via the per-sequence block
// table (vLLM's PagedAttention layout). GQA/MQA is handled by mapping each
// query head to its kv head.
//
// Parallelization: one CUDA block per (sequence, query-head); blockDim.x ==
// head_dim. Thread `d` owns Q[d] and the output accumulator acc[d]; per-key
// scores are head_dim-wide dot products reduced across the block. Because decode
// already exposes (num_seqs * num_heads) blocks of parallelism, this single-pass
// scheme is usually enough. The classic FlashDecoding split-K (one block per
// (seq, head, kv-chunk) + a combine pass) is the documented next step when a
// few long sequences underfill the GPU — it would need a caller workspace for
// the partial (out, m, l), which this self-contained ABI does not expose.
#include "kernel_set/attention.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"
#include "attention_common.cuh"

namespace ks {
namespace attention {

// k_cache,v_cache: [num_blocks, num_kv_heads, block_size, head_dim]
// block_tables   : [num_seqs, max_blocks_per_seq]
// seq_lens       : [num_seqs]
// q,out          : [num_seqs, num_heads, head_dim]
template <typename scalar_t>
KS_GLOBAL void paged_attn_decode_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache, const scalar_t* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_blocks_per_seq, float scale) {
  const int d = threadIdx.x;        // head-dim component owned by this thread
  const int head = blockIdx.x;      // query head
  const int seq = blockIdx.y;       // sequence

  const int seq_len = seq_lens[seq];
  const int kv_head = head / gqa_group_size(num_heads, num_kv_heads);

  const scalar_t* q_ptr =
      q + (static_cast<int64_t>(seq) * num_heads + head) * head_dim;
  const float q_reg = to_float(q_ptr[d]);

  extern __shared__ float smem_raw[];
  float* red = smem_raw;  // block_reduce scratch (>= num_warps)

  float acc = 0.f;
  float m = -3.4028235e38f;
  float l = 0.f;

  // Strides into the paged cache (element counts).
  const int64_t kv_head_stride = static_cast<int64_t>(block_size) * head_dim;
  const int64_t block_stride =
      static_cast<int64_t>(num_kv_heads) * block_size * head_dim;

  const int num_blocks =
      (seq_len + block_size - 1) / block_size;  // populated blocks for this seq
  const int32_t* seq_block_table =
      block_tables + static_cast<int64_t>(seq) * max_blocks_per_seq;

  for (int b = 0; b < num_blocks; ++b) {
    const int32_t phys_block = seq_block_table[b];
    const int tokens_in_block = min(block_size, seq_len - b * block_size);

    const scalar_t* k_block =
        k_cache + phys_block * block_stride + kv_head * kv_head_stride;
    const scalar_t* v_block =
        v_cache + phys_block * block_stride + kv_head * kv_head_stride;

    for (int t = 0; t < tokens_in_block; ++t) {
      // Score = scale * dot(Q, K[t]); reduced across the block in fp32.
      const float prod = q_reg * to_float(k_block[t * head_dim + d]);
      const float dot = block_reduce_sum(prod, red);
      const float s = dot * scale;

      // Online-softmax update (all threads see the same s, m, l).
      const float m_new = fmaxf(m, s);
      const float correction = rescale_factor(m, m_new);
      const float p = __expf(s - m_new);
      acc = acc * correction + p * to_float(v_block[t * head_dim + d]);
      l = l * correction + p;
      m = m_new;
      __syncthreads();  // red[] reused by the next key's reduction
    }
  }

  const float inv_l = (l > 0.f) ? (1.0f / l) : 0.f;
  scalar_t* out_ptr =
      out + (static_cast<int64_t>(seq) * num_heads + head) * head_dim;
  out_ptr[d] = from_float<scalar_t>(acc * inv_l);
}

template <typename scalar_t>
KS_GLOBAL void paged_attn_decode_split_k_partial_kernel(
    scalar_t* __restrict__ partial_out, float* __restrict__ partial_lse,
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens, int num_seqs, int num_heads,
    int num_kv_heads, int head_dim, int block_size, int max_blocks_per_seq,
    int num_splits, float scale) {
  const int d = threadIdx.x;
  const int head = blockIdx.x;
  const int seq = blockIdx.y;
  const int split = blockIdx.z;

  const int seq_len = seq_lens[seq];
  const int chunk_begin =
      static_cast<int>((static_cast<int64_t>(seq_len) * split) / num_splits);
  const int chunk_end = static_cast<int>(
      (static_cast<int64_t>(seq_len) * (split + 1)) / num_splits);
  const int kv_head = head / gqa_group_size(num_heads, num_kv_heads);

  const scalar_t* q_ptr =
      q + (static_cast<int64_t>(seq) * num_heads + head) * head_dim;
  const float q_reg = to_float(q_ptr[d]);

  extern __shared__ float smem_raw[];
  float* red = smem_raw;

  float acc = 0.f;
  float m = -3.4028235e38f;
  float l = 0.f;

  const int64_t kv_head_stride = static_cast<int64_t>(block_size) * head_dim;
  const int64_t block_stride =
      static_cast<int64_t>(num_kv_heads) * block_size * head_dim;
  const int32_t* seq_block_table =
      block_tables + static_cast<int64_t>(seq) * max_blocks_per_seq;

  int pos = chunk_begin;
  while (pos < chunk_end) {
    const int b = pos / block_size;
    const int off = pos - b * block_size;
    const int tokens = min(block_size - off, chunk_end - pos);
    const int32_t phys_block = seq_block_table[b];
    const scalar_t* k_block =
        k_cache + phys_block * block_stride + kv_head * kv_head_stride;
    const scalar_t* v_block =
        v_cache + phys_block * block_stride + kv_head * kv_head_stride;

    for (int i = 0; i < tokens; ++i) {
      const int t = off + i;
      const float prod = q_reg * to_float(k_block[t * head_dim + d]);
      const float dot = block_reduce_sum(prod, red);
      const float s = dot * scale;

      const float m_new = fmaxf(m, s);
      const float correction = rescale_factor(m, m_new);
      const float p = __expf(s - m_new);
      acc = acc * correction + p * to_float(v_block[t * head_dim + d]);
      l = l * correction + p;
      m = m_new;
      __syncthreads();
    }
    pos += tokens;
  }

  const int64_t row =
      (static_cast<int64_t>(split) * num_seqs + seq) * num_heads + head;
  scalar_t* pout = partial_out + row * head_dim;
  if (l > 0.f) {
    pout[d] = from_float<scalar_t>(acc / l);
    if (d == 0) partial_lse[row] = m + __logf(l);
  } else {
    pout[d] = from_float<scalar_t>(0.f);
    if (d == 0) partial_lse[row] = -INFINITY;
  }
}

template <typename scalar_t>
KS_GLOBAL void paged_attn_decode_split_k_combine_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ partial_out,
    const float* __restrict__ partial_lse, int num_seqs, int num_heads,
    int head_dim, int num_splits) {
  const int d = threadIdx.x;
  const int head = blockIdx.x;
  const int seq = blockIdx.y;

  float m = -INFINITY;
  for (int split = 0; split < num_splits; ++split) {
    const int64_t row =
        (static_cast<int64_t>(split) * num_seqs + seq) * num_heads + head;
    m = fmaxf(m, partial_lse[row]);
  }

  float denom = 0.f;
  float acc = 0.f;
  for (int split = 0; split < num_splits; ++split) {
    const int64_t row =
        (static_cast<int64_t>(split) * num_seqs + seq) * num_heads + head;
    const float lse = partial_lse[row];
    const float w = isfinite(lse) ? __expf(lse - m) : 0.f;
    denom += w;
    acc += w * to_float(partial_out[row * head_dim + d]);
  }

  scalar_t* out_ptr =
      out + (static_cast<int64_t>(seq) * num_heads + head) * head_dim;
  out_ptr[d] = from_float<scalar_t>(denom > 0.f ? acc / denom : 0.f);
}

}  // namespace attention
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_paged_attn_decode(
    void* out, const void* q, const void* k_cache, const void* v_cache,
    const int32_t* block_tables, const int32_t* seq_lens, int num_seqs,
    int num_heads, int num_kv_heads, int head_dim, int block_size,
    int max_blocks_per_seq, float softmax_scale, ks_dtype_t dtype,
    ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(q);
  KS_CHECK_PTR(k_cache);
  KS_CHECK_PTR(v_cache);
  KS_CHECK_PTR(block_tables);
  KS_CHECK_PTR(seq_lens);
  if (num_seqs <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      block_size <= 0 || max_blocks_per_seq <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_paged_attn_decode: shape");
  if (num_kv_heads > num_heads || (num_heads % num_kv_heads) != 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_paged_attn_decode: num_heads % num_kv_heads != 0");
  if (head_dim > attention::kMaxHeadDim)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_paged_attn_decode: head_dim too large");
  if ((head_dim & 31) != 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_paged_attn_decode: head_dim must be a multiple of 32");

  const float scale = attention::resolve_scale(softmax_scale, head_dim);
  auto s = to_stream(stream);
  const dim3 grid(static_cast<unsigned>(num_heads),
                  static_cast<unsigned>(num_seqs));
  const dim3 block(static_cast<unsigned>(head_dim));
  // block_reduce scratch: one float per warp (head_dim/warp). head_dim floats
  // is a safe upper bound and keeps the layout uniform with the prefill kernel.
  const size_t smem = sizeof(float) * static_cast<size_t>(head_dim);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_paged_attn_decode", {
    attention::paged_attn_decode_kernel<scalar_t><<<grid, block, smem, s>>>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(q),
        static_cast<const scalar_t*>(k_cache),
        static_cast<const scalar_t*>(v_cache), block_tables, seq_lens,
        num_heads, num_kv_heads, head_dim, block_size, max_blocks_per_seq,
        scale);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_paged_attn_decode_split_k(
    void* out, void* partial_out, float* partial_lse, const void* q,
    const void* k_cache, const void* v_cache, const int32_t* block_tables,
    const int32_t* seq_lens, int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_blocks_per_seq, int num_splits,
    float softmax_scale, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(partial_out);
  KS_CHECK_PTR(partial_lse);
  KS_CHECK_PTR(q);
  KS_CHECK_PTR(k_cache);
  KS_CHECK_PTR(v_cache);
  KS_CHECK_PTR(block_tables);
  KS_CHECK_PTR(seq_lens);
  if (num_seqs <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      block_size <= 0 || max_blocks_per_seq <= 0 || num_splits <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_paged_attn_decode_split_k: shape");
  if (num_kv_heads > num_heads || (num_heads % num_kv_heads) != 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_paged_attn_decode_split_k: num_heads % num_kv_heads != 0");
  if (head_dim > attention::kMaxHeadDim)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_paged_attn_decode_split_k: head_dim too large");
  if ((head_dim & 31) != 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_paged_attn_decode_split_k: head_dim must be a multiple of 32");
  if (num_splits > 65535)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_paged_attn_decode_split_k: num_splits exceeds grid.z");

  const float scale = attention::resolve_scale(softmax_scale, head_dim);
  auto s = to_stream(stream);
  const dim3 grid(static_cast<unsigned>(num_heads),
                  static_cast<unsigned>(num_seqs),
                  static_cast<unsigned>(num_splits));
  const dim3 combine_grid(static_cast<unsigned>(num_heads),
                          static_cast<unsigned>(num_seqs));
  const dim3 block(static_cast<unsigned>(head_dim));
  const size_t smem = sizeof(float) * static_cast<size_t>(head_dim);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_paged_attn_decode_split_k", {
    attention::paged_attn_decode_split_k_partial_kernel<scalar_t>
        <<<grid, block, smem, s>>>(
            static_cast<scalar_t*>(partial_out), partial_lse,
            static_cast<const scalar_t*>(q),
            static_cast<const scalar_t*>(k_cache),
            static_cast<const scalar_t*>(v_cache), block_tables, seq_lens,
            num_seqs, num_heads, num_kv_heads, head_dim, block_size,
            max_blocks_per_seq, num_splits, scale);
    attention::paged_attn_decode_split_k_combine_kernel<scalar_t>
        <<<combine_grid, block, 0, s>>>(
            static_cast<scalar_t*>(out),
            static_cast<const scalar_t*>(partial_out), partial_lse, num_seqs,
            num_heads, head_dim, num_splits);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
