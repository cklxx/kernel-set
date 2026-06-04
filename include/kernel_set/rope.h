/*
 * kernel-set — rotary position embeddings (RoPE).
 *
 * Tensors are laid out [num_tokens, num_heads, head_dim] (row-major, contiguous
 * over head_dim). `cos`/`sin` are [num_tokens, head_dim/2] (or [max_pos,
 * head_dim/2] gathered by `positions`). Two conventions are supported:
 *   - NeoX / "rotate_half": pairs are (i, i + head_dim/2)
 *   - GPT-J / interleaved:  pairs are (2i, 2i+1)
 */
#ifndef KERNEL_SET_ROPE_H_
#define KERNEL_SET_ROPE_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* In-place rotary embedding applied to q and k.
 *   q: [num_tokens, num_q_heads, head_dim]
 *   k: [num_tokens, num_kv_heads, head_dim]   (k may be NULL to skip)
 *   cos/sin: [num_tokens, head_dim/2]
 *   interleaved != 0 selects the GPT-J convention. */
KS_API ks_status_t ks_rope_inplace(void* q, void* k, const void* cos,
                                   const void* sin, int64_t num_tokens,
                                   int num_q_heads, int num_kv_heads,
                                   int head_dim, int interleaved,
                                   ks_dtype_t dtype, ks_stream_t stream);

/* Out-of-place variant writing to q_out/k_out. */
KS_API ks_status_t ks_rope(void* q_out, void* k_out, const void* q,
                           const void* k, const void* cos, const void* sin,
                           int64_t num_tokens, int num_q_heads, int num_kv_heads,
                           int head_dim, int interleaved, ks_dtype_t dtype,
                           ks_stream_t stream);

/* Gathered variant: cos/sin are caches [max_pos, head_dim/2] indexed by
 * `positions` (int32, [num_tokens]). */
KS_API ks_status_t ks_rope_gather(void* q, void* k, const void* cos_cache,
                                  const void* sin_cache, const int32_t* positions,
                                  int64_t num_tokens, int num_q_heads,
                                  int num_kv_heads, int head_dim, int interleaved,
                                  ks_dtype_t dtype, ks_stream_t stream);

/* Backward: rotate gradients by the conjugate rotation (sin -> -sin). */
KS_API ks_status_t ks_rope_backward(void* grad_q, void* grad_k,
                                    const void* cos, const void* sin,
                                    int64_t num_tokens, int num_q_heads,
                                    int num_kv_heads, int head_dim,
                                    int interleaved, ks_dtype_t dtype,
                                    ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_ROPE_H_ */
