// kernel-set — DeepSeek Multi-head Latent Attention (MLA) decode.
//
// MLA compresses the KV cache to a single low-rank latent vector per token that
// is shared across all query heads (so the cache is MQA-shaped), plus a small
// decoupled RoPE part. In the "weight-absorbed" decode form the query has
// already been projected into the latent space:
//   q_nope : [num_seqs, num_heads, kv_lora_rank]   (latent-space query)
//   q_pe   : [num_seqs, num_heads, rope_dim]        (decoupled RoPE query)
//   kv_cache (per token, shared by all heads):
//            [num_blocks, block_size, kv_lora_rank + rope_dim]
//            = [ latent (kv_lora_rank) | pe (rope_dim) ]
// Per head h and cached token t the score is
//   s = scale * ( q_nope[h] . latent[t] + q_pe[h] . pe[t] )
// and the output is the softmax-weighted sum of the latents:
//   out[h] = sum_t softmax(s)_t * latent[t]   ->  [num_seqs, num_heads,
//   kv_lora_rank]
// (the model up-projects this latent output afterwards).
//
// Online fp32 softmax over the paged latent cache, exactly like the paged decode
// kernel; the only differences are the shared MQA latent and the split
// nope/pe score. One block per (seq, head); blockDim.x == kv_lora_rank, so
// thread `d` owns latent dim d, q_nope[d], and the output accumulator. Threads
// with d < rope_dim additionally fold in the q_pe . pe contribution to the
// score's block reduction.
#include "kernel_set/attention.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"
#include "attention_common.cuh"

namespace ks {
namespace attention {

template <typename scalar_t>
KS_GLOBAL void mla_decode_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q_nope,
    const scalar_t* __restrict__ q_pe, const scalar_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens, int num_heads, int kv_lora_rank,
    int rope_dim, int block_size, int max_blocks_per_seq, float scale) {
  const int d = threadIdx.x;     // latent dim owned by this thread (< kv_lora_rank)
  const int head = blockIdx.x;   // query head
  const int seq = blockIdx.y;    // sequence

  const int seq_len = seq_lens[seq];
  const int entry_dim = kv_lora_rank + rope_dim;  // per-token cache width

  // This thread's latent-query component and (for d < rope_dim) pe-query comp.
  const scalar_t* qn_ptr =
      q_nope + (static_cast<int64_t>(seq) * num_heads + head) * kv_lora_rank;
  const scalar_t* qp_ptr =
      q_pe + (static_cast<int64_t>(seq) * num_heads + head) * rope_dim;
  const float qn_reg = to_float(qn_ptr[d]);
  const float qp_reg = (d < rope_dim) ? to_float(qp_ptr[d]) : 0.f;

  extern __shared__ float smem_raw[];
  float* red = smem_raw;  // block_reduce scratch (>= num_warps)

  float acc = 0.f;  // unnormalized latent output accumulator for dim d
  float m = -3.4028235e38f;
  float l = 0.f;

  const int num_blocks = (seq_len + block_size - 1) / block_size;
  const int32_t* seq_block_table =
      block_tables + static_cast<int64_t>(seq) * max_blocks_per_seq;

  for (int b = 0; b < num_blocks; ++b) {
    const int32_t phys_block = seq_block_table[b];
    const int tokens_in_block = min(block_size, seq_len - b * block_size);
    const scalar_t* blk =
        kv_cache + static_cast<int64_t>(phys_block) * block_size * entry_dim;

    for (int t = 0; t < tokens_in_block; ++t) {
      const scalar_t* entry = blk + static_cast<int64_t>(t) * entry_dim;
      const float latent_d = to_float(entry[d]);  // latent[t][d]
      // Per-thread contribution to the score: nope term for every thread, plus
      // the pe term for the first rope_dim threads.
      float prod = qn_reg * latent_d;
      if (d < rope_dim) prod += qp_reg * to_float(entry[kv_lora_rank + d]);
      const float dot = block_reduce_sum(prod, red);
      const float s = dot * scale;

      // Online-softmax update (uniform s, m, l across the block).
      const float m_new = fmaxf(m, s);
      const float correction = rescale_factor(m, m_new);
      const float p = __expf(s - m_new);
      acc = acc * correction + p * latent_d;  // output lives in latent space
      l = l * correction + p;
      m = m_new;
      __syncthreads();  // red[] reused next token
    }
  }

  const float inv_l = (l > 0.f) ? (1.0f / l) : 0.f;
  scalar_t* out_ptr =
      out + (static_cast<int64_t>(seq) * num_heads + head) * kv_lora_rank;
  out_ptr[d] = from_float<scalar_t>(acc * inv_l);
}

template <typename scalar_t>
KS_GLOBAL void mla_decode_split_k_partial_kernel(
    scalar_t* __restrict__ partial_out, float* __restrict__ partial_lse,
    const scalar_t* __restrict__ q_nope, const scalar_t* __restrict__ q_pe,
    const scalar_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens, int num_seqs, int num_heads,
    int kv_lora_rank, int rope_dim, int block_size, int max_blocks_per_seq,
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
  const int entry_dim = kv_lora_rank + rope_dim;

  const scalar_t* qn_ptr =
      q_nope + (static_cast<int64_t>(seq) * num_heads + head) * kv_lora_rank;
  const scalar_t* qp_ptr =
      q_pe + (static_cast<int64_t>(seq) * num_heads + head) * rope_dim;
  const float qn_reg = to_float(qn_ptr[d]);
  const float qp_reg = (d < rope_dim) ? to_float(qp_ptr[d]) : 0.f;

  extern __shared__ float smem_raw[];
  float* red = smem_raw;

  float acc = 0.f;
  float m = -3.4028235e38f;
  float l = 0.f;

  const int32_t* seq_block_table =
      block_tables + static_cast<int64_t>(seq) * max_blocks_per_seq;

  int pos = chunk_begin;
  while (pos < chunk_end) {
    const int b = pos / block_size;
    const int off = pos - b * block_size;
    const int tokens = min(block_size - off, chunk_end - pos);
    const int32_t phys_block = seq_block_table[b];
    const scalar_t* blk =
        kv_cache + static_cast<int64_t>(phys_block) * block_size * entry_dim;

    for (int i = 0; i < tokens; ++i) {
      const int t = off + i;
      const scalar_t* entry = blk + static_cast<int64_t>(t) * entry_dim;
      const float latent_d = to_float(entry[d]);
      float prod = qn_reg * latent_d;
      if (d < rope_dim) prod += qp_reg * to_float(entry[kv_lora_rank + d]);
      const float dot = block_reduce_sum(prod, red);
      const float s = dot * scale;

      const float m_new = fmaxf(m, s);
      const float correction = rescale_factor(m, m_new);
      const float p = __expf(s - m_new);
      acc = acc * correction + p * latent_d;
      l = l * correction + p;
      m = m_new;
      __syncthreads();
    }
    pos += tokens;
  }

  const int64_t row =
      (static_cast<int64_t>(split) * num_seqs + seq) * num_heads + head;
  scalar_t* pout = partial_out + row * kv_lora_rank;
  if (l > 0.f) {
    pout[d] = from_float<scalar_t>(acc / l);
    if (d == 0) partial_lse[row] = m + __logf(l);
  } else {
    pout[d] = from_float<scalar_t>(0.f);
    if (d == 0) partial_lse[row] = -INFINITY;
  }
}

template <typename scalar_t>
KS_GLOBAL void mla_decode_split_k_combine_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ partial_out,
    const float* __restrict__ partial_lse, int num_seqs, int num_heads,
    int kv_lora_rank, int num_splits) {
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
    acc += w * to_float(partial_out[row * kv_lora_rank + d]);
  }

  scalar_t* out_ptr =
      out + (static_cast<int64_t>(seq) * num_heads + head) * kv_lora_rank;
  out_ptr[d] = from_float<scalar_t>(denom > 0.f ? acc / denom : 0.f);
}

}  // namespace attention
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_mla_decode(
    void* out, const void* q_nope, const void* q_pe, const void* kv_cache,
    const int32_t* block_tables, const int32_t* seq_lens, int num_seqs,
    int num_heads, int kv_lora_rank, int rope_dim, int block_size,
    int max_blocks_per_seq, float softmax_scale, ks_dtype_t dtype,
    ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(q_nope);
  KS_CHECK_PTR(q_pe);
  KS_CHECK_PTR(kv_cache);
  KS_CHECK_PTR(block_tables);
  KS_CHECK_PTR(seq_lens);
  if (num_seqs <= 0 || num_heads <= 0 || kv_lora_rank <= 0 || rope_dim <= 0 ||
      block_size <= 0 || max_blocks_per_seq <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_mla_decode: shape");
  if (rope_dim > kv_lora_rank)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_mla_decode: rope_dim must be <= kv_lora_rank");
  if (kv_lora_rank > 1024)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_mla_decode: kv_lora_rank exceeds max block threads");
  // blockDim.x == kv_lora_rank and scores use a block_reduce over the warps, so
  // the latent rank must tile evenly into warps (multiple of 32; DeepSeek uses
  // kv_lora_rank == 512).
  if ((kv_lora_rank & 31) != 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_mla_decode: kv_lora_rank must be a multiple of 32");

  // MLA's effective per-head score dim is (kv_lora_rank + rope_dim); the scale
  // default follows the same 1/sqrt(d) convention applied to that combined dim.
  const float scale = attention::resolve_scale(softmax_scale,
                                               kv_lora_rank + rope_dim);
  auto s = to_stream(stream);
  const dim3 grid(static_cast<unsigned>(num_heads),
                  static_cast<unsigned>(num_seqs));
  const dim3 block(static_cast<unsigned>(kv_lora_rank));
  const size_t smem = sizeof(float) * static_cast<size_t>(kv_lora_rank);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_mla_decode", {
    attention::mla_decode_kernel<scalar_t><<<grid, block, smem, s>>>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(q_nope),
        static_cast<const scalar_t*>(q_pe),
        static_cast<const scalar_t*>(kv_cache), block_tables, seq_lens,
        num_heads, kv_lora_rank, rope_dim, block_size, max_blocks_per_seq,
        scale);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_mla_decode_split_k(
    void* out, void* partial_out, float* partial_lse, const void* q_nope,
    const void* q_pe, const void* kv_cache, const int32_t* block_tables,
    const int32_t* seq_lens, int num_seqs, int num_heads, int kv_lora_rank,
    int rope_dim, int block_size, int max_blocks_per_seq, int num_splits,
    float softmax_scale, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(partial_out);
  KS_CHECK_PTR(partial_lse);
  KS_CHECK_PTR(q_nope);
  KS_CHECK_PTR(q_pe);
  KS_CHECK_PTR(kv_cache);
  KS_CHECK_PTR(block_tables);
  KS_CHECK_PTR(seq_lens);
  if (num_seqs <= 0 || num_heads <= 0 || kv_lora_rank <= 0 || rope_dim <= 0 ||
      block_size <= 0 || max_blocks_per_seq <= 0 || num_splits <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_mla_decode_split_k: shape");
  if (rope_dim > kv_lora_rank)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_mla_decode_split_k: rope_dim must be <= kv_lora_rank");
  if (kv_lora_rank > 1024)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_mla_decode_split_k: kv_lora_rank exceeds max block threads");
  if ((kv_lora_rank & 31) != 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_mla_decode_split_k: kv_lora_rank must be a multiple of 32");
  if (num_splits > 65535)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_mla_decode_split_k: num_splits exceeds grid.z");

  const float scale =
      attention::resolve_scale(softmax_scale, kv_lora_rank + rope_dim);
  auto s = to_stream(stream);
  const dim3 grid(static_cast<unsigned>(num_heads),
                  static_cast<unsigned>(num_seqs),
                  static_cast<unsigned>(num_splits));
  const dim3 combine_grid(static_cast<unsigned>(num_heads),
                          static_cast<unsigned>(num_seqs));
  const dim3 block(static_cast<unsigned>(kv_lora_rank));
  const size_t smem = sizeof(float) * static_cast<size_t>(kv_lora_rank);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_mla_decode_split_k", {
    attention::mla_decode_split_k_partial_kernel<scalar_t>
        <<<grid, block, smem, s>>>(
            static_cast<scalar_t*>(partial_out), partial_lse,
            static_cast<const scalar_t*>(q_nope),
            static_cast<const scalar_t*>(q_pe),
            static_cast<const scalar_t*>(kv_cache), block_tables, seq_lens,
            num_seqs, num_heads, kv_lora_rank, rope_dim, block_size,
            max_blocks_per_seq, num_splits, scale);
    attention::mla_decode_split_k_combine_kernel<scalar_t>
        <<<combine_grid, block, 0, s>>>(
            static_cast<scalar_t*>(out),
            static_cast<const scalar_t*>(partial_out), partial_lse, num_seqs,
            num_heads, kv_lora_rank, num_splits);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
