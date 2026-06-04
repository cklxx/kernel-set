/*
 * kernel-set — Mixture-of-Experts routing and grouped expert compute.
 *
 * Pipeline: gate -> (sort/align) -> permute tokens by expert -> grouped GEMM ->
 * unpermute with routing weights. Helpers below cover each stage so a backend
 * can compose them or fuse as needed.
 */
#ifndef KERNEL_SET_MOE_H_
#define KERNEL_SET_MOE_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* Softmax over experts then top-k select.
 *   logits: [num_tokens, num_experts] of `dtype`
 *   out_weights: [num_tokens, top_k] fp32 (routing weights; renormalized if set)
 *   out_indices: [num_tokens, top_k] int32 (selected expert ids) */
KS_API ks_status_t ks_moe_gate_softmax_topk(
    float* out_weights, int32_t* out_indices, const void* logits,
    int64_t num_tokens, int num_experts, int top_k, int renormalize,
    ks_dtype_t dtype, ks_stream_t stream);

/* Sigmoid + group-limited top-k gating (DeepSeek-V3 style). `n_group`,
 * `topk_group`, and `routed_scaling_factor` follow the DeepSeek config. */
KS_API ks_status_t ks_moe_gate_sigmoid_group_topk(
    float* out_weights, int32_t* out_indices, const void* logits,
    const void* correction_bias, int64_t num_tokens, int num_experts,
    int n_group, int topk_group, int top_k, int renormalize,
    float routed_scaling_factor, ks_dtype_t dtype, ks_stream_t stream);

/* Build the permutation that groups tokens by expert.
 *   topk_indices: [num_tokens, top_k] int32
 *   out: sorted_token_ids [num_tokens*top_k] int32,
 *        expert_offsets [num_experts+1] int32 (CSR-style group boundaries) */
KS_API ks_status_t ks_moe_compute_permutation(
    int32_t* sorted_token_ids, int32_t* expert_offsets,
    const int32_t* topk_indices, int64_t num_tokens, int num_experts,
    int top_k, ks_stream_t stream);

/* Gather rows of `input` [num_tokens, hidden] into `permuted`
 * [num_tokens*top_k, hidden] following sorted_token_ids (row = id / top_k). */
KS_API ks_status_t ks_moe_permute(void* permuted, const void* input,
                                  const int32_t* sorted_token_ids,
                                  int64_t num_tokens, int top_k, int64_t hidden,
                                  ks_dtype_t dtype, ks_stream_t stream);

/* Scatter-add expert outputs back, weighted by routing weights:
 *   out[token] = sum_k weight[token,k] * permuted[pos(token,k)]. */
KS_API ks_status_t ks_moe_unpermute(void* out, const void* permuted,
                                    const int32_t* sorted_token_ids,
                                    const float* routing_weights,
                                    int64_t num_tokens, int top_k,
                                    int64_t hidden, ks_dtype_t dtype,
                                    ks_stream_t stream);

/* Grouped GEMM: for each expert e, C[off_e:off_{e+1}] = A[off_e:off_{e+1}] @ B_e.
 *   a: [total_rows, k]   (permuted tokens)
 *   b: [num_experts, k, n] (per-expert weights, contiguous)
 *   expert_offsets: [num_experts+1] int32 row boundaries
 *   c: [total_rows, n] */
KS_API ks_status_t ks_moe_grouped_gemm(
    void* c, const void* a, const void* b, const int32_t* expert_offsets,
    int num_experts, int64_t total_rows, int64_t n, int64_t k,
    ks_dtype_t dtype, ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_MOE_H_ */
