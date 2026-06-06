/*
 * kernel-set — fused attention (FlashAttention-style prefill, paged decode, MLA).
 *
 * Conventions:
 *   - head_dim is contiguous (innermost).
 *   - GQA/MQA: num_kv_heads may be < num_heads; each kv head is shared by
 *     num_heads/num_kv_heads query heads.
 *   - softmax is computed online in fp32 (no full score matrix materialized).
 *   - softmax_scale defaults to 1/sqrt(head_dim) when <= 0.
 */
#ifndef KERNEL_SET_ATTENTION_H_
#define KERNEL_SET_ATTENTION_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* Variable-length packed prefill (FlashAttention-2 forward).
 *   q: [total_q, num_heads, head_dim]
 *   k,v: [total_kv, num_kv_heads, head_dim]
 *   cu_seqlens_q / cu_seqlens_k: [batch+1] int32 prefix sums of sequence lens.
 *   out: [total_q, num_heads, head_dim]
 *   softmax_lse: [num_heads, total_q] fp32 (may be NULL if not training). */
KS_API ks_status_t ks_flash_attn_varlen(
    void* out, void* softmax_lse, const void* q, const void* k, const void* v,
    const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_k, int batch,
    int max_seqlen_q, int max_seqlen_k, int num_heads, int num_kv_heads,
    int head_dim, float softmax_scale, int causal, ks_dtype_t dtype,
    ks_stream_t stream);

/* Dense (non-varlen) prefill for a uniform [batch, seqlen] shape.
 *   q: [batch, seqlen_q, num_heads, head_dim]
 *   k,v: [batch, seqlen_k, num_kv_heads, head_dim] */
KS_API ks_status_t ks_flash_attn(void* out, void* softmax_lse, const void* q,
                                 const void* k, const void* v, int batch,
                                 int seqlen_q, int seqlen_k, int num_heads,
                                 int num_kv_heads, int head_dim,
                                 float softmax_scale, int causal,
                                 ks_dtype_t dtype, ks_stream_t stream);

/* Paged KV-cache decode (single query position per sequence; FlashDecoding).
 *   q: [num_seqs, num_heads, head_dim]
 *   k_cache,v_cache: [num_blocks, num_kv_heads, block_size, head_dim]
 *   block_tables: [num_seqs, max_blocks_per_seq] int32 (block ids per seq)
 *   seq_lens: [num_seqs] int32 (current context length per seq)
 *   out: [num_seqs, num_heads, head_dim] */
KS_API ks_status_t ks_paged_attn_decode(
    void* out, const void* q, const void* k_cache, const void* v_cache,
    const int32_t* block_tables, const int32_t* seq_lens, int num_seqs,
    int num_heads, int num_kv_heads, int head_dim, int block_size,
    int max_blocks_per_seq, float softmax_scale, ks_dtype_t dtype,
    ks_stream_t stream);

/* Write new K/V into a paged cache at the given slots (prefill & decode append).
 *   key,value: [num_tokens, num_kv_heads, head_dim]
 *   slot_mapping: [num_tokens] int32 -> flat slot = block_id*block_size + off */
KS_API ks_status_t ks_reshape_and_cache(
    void* k_cache, void* v_cache, const void* key, const void* value,
    const int32_t* slot_mapping, int num_tokens, int num_kv_heads,
    int head_dim, int block_size, ks_dtype_t dtype, ks_stream_t stream);

/* Merge two partial attention states by log-sum-exp (cascade / chunked / ring
 * attention glue): for each row,
 *   out = (out_a*exp(lse_a) + out_b*exp(lse_b)) / (exp(lse_a)+exp(lse_b))
 *   lse = log(exp(lse_a) + exp(lse_b))           (numerically stable)
 * out/out_a/out_b: [n_rows, v_dim] (model dtype); lse/lse_a/lse_b: [n_rows] fp32.
 * `out` may alias out_a and `lse` may alias lse_a (in-place accumulate). */
KS_API ks_status_t ks_attention_state_merge(
    void* out, float* lse, const void* out_a, const float* lse_a,
    const void* out_b, const float* lse_b, int64_t n_rows, int64_t v_dim,
    ks_dtype_t dtype, ks_stream_t stream);

/* DeepSeek Multi-head Latent Attention decode over a compressed KV cache.
 *   q_nope: [num_seqs, num_heads, kv_lora_rank]
 *   q_pe:   [num_seqs, num_heads, rope_dim]
 *   kv_cache (compressed latent + pe): [num_blocks, block_size,
 *            kv_lora_rank + rope_dim] */
KS_API ks_status_t ks_mla_decode(
    void* out, const void* q_nope, const void* q_pe, const void* kv_cache,
    const int32_t* block_tables, const int32_t* seq_lens, int num_seqs,
    int num_heads, int kv_lora_rank, int rope_dim, int block_size,
    int max_blocks_per_seq, float softmax_scale, ks_dtype_t dtype,
    ks_stream_t stream);

/* FlashAttention backward (training). Requires the forward `out` and
 * `softmax_lse`. grad_q/grad_k/grad_v match q/k/v shapes. */
KS_API ks_status_t ks_flash_attn_backward(
    void* grad_q, void* grad_k, void* grad_v, const void* grad_out,
    const void* q, const void* k, const void* v, const void* out,
    const void* softmax_lse, int batch, int seqlen_q, int seqlen_k,
    int num_heads, int num_kv_heads, int head_dim, float softmax_scale,
    int causal, ks_dtype_t dtype, ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_ATTENTION_H_ */
