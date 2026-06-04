/*
 * kernel-set — training loss kernels (memory-efficient cross-entropy).
 */
#ifndef KERNEL_SET_LOSS_H_
#define KERNEL_SET_LOSS_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* Fused cross-entropy forward+backward in one pass.
 *   logits: [num_tokens, vocab] of `dtype` (overwritten with grad if
 *           grad_logits == logits; otherwise grad written to grad_logits)
 *   targets: [num_tokens] int32/int64 class ids; `ignore_index` masked out
 *   losses:  [num_tokens] fp32 per-token loss
 *   label_smoothing in [0,1). Reduction is left to the caller. */
KS_API ks_status_t ks_cross_entropy(float* losses, void* grad_logits,
                                    const void* logits, const void* targets,
                                    int targets_i64, int64_t num_tokens,
                                    int64_t vocab, int64_t ignore_index,
                                    float label_smoothing, ks_dtype_t dtype,
                                    ks_stream_t stream);

/* Fused-linear-cross-entropy: computes CE from hidden states and the LM head
 * weight in chunks WITHOUT materializing the full [num_tokens, vocab] logits,
 * and writes grad_hidden / grad_weight. (Chunked over the token dim.)
 *   hidden: [num_tokens, hidden_dim]; weight: [vocab, hidden_dim]
 *   grad_hidden: [num_tokens, hidden_dim]; grad_weight_fp32: [vocab, hidden_dim]
 */
KS_API ks_status_t ks_fused_linear_cross_entropy(
    float* losses, void* grad_hidden, void* grad_weight_fp32,
    const void* hidden, const void* weight, const void* targets,
    int targets_i64, int64_t num_tokens, int64_t hidden_dim, int64_t vocab,
    int64_t ignore_index, float label_smoothing, int chunk_size,
    ks_dtype_t dtype, ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_LOSS_H_ */
