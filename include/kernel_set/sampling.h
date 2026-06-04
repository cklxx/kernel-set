/*
 * kernel-set — softmax and token sampling for autoregressive decoding.
 *
 * logits are [num_seqs, vocab_size]. Sampling kernels select one token per
 * sequence into out_tokens [num_seqs] int32. Randomness uses a counter-based
 * RNG seeded by (seed, philox_offset) for reproducibility across batches.
 */
#ifndef KERNEL_SET_SAMPLING_H_
#define KERNEL_SET_SAMPLING_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* out = softmax(input / temperature) along the last dim. temperature<=0 => 1. */
KS_API ks_status_t ks_softmax(void* out, const void* input, int64_t rows,
                              int64_t cols, float temperature, ks_dtype_t dtype,
                              ks_stream_t stream);

/* log_softmax along the last dim (fp32 out recommended for stability). */
KS_API ks_status_t ks_log_softmax(void* out, const void* input, int64_t rows,
                                  int64_t cols, ks_dtype_t dtype,
                                  ks_stream_t stream);

/* Greedy: out_tokens[s] = argmax_v logits[s, v]. */
KS_API ks_status_t ks_argmax(int32_t* out_tokens, const void* logits,
                             int64_t num_seqs, int64_t vocab_size,
                             ks_dtype_t dtype, ks_stream_t stream);

/* Combined temperature + top-k + top-p sampling.
 *   temperatures: [num_seqs] fp32 or NULL (=> 1.0)
 *   top_ks:  [num_seqs] int32 or NULL (<=0 => disabled)
 *   top_ps:  [num_seqs] fp32 or NULL (<=0 => disabled)
 *   out_tokens: [num_seqs] int32
 *   out_probs:  [num_seqs] fp32 sampled-token probability or NULL. */
KS_API ks_status_t ks_sample(int32_t* out_tokens, float* out_probs,
                             const void* logits, const float* temperatures,
                             const int32_t* top_ks, const float* top_ps,
                             int64_t num_seqs, int64_t vocab_size,
                             uint64_t seed, uint64_t philox_offset,
                             ks_dtype_t dtype, ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_SAMPLING_H_ */
