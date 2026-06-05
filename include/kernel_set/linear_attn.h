/*
 * kernel-set — linear-attention / gated-delta / RWKV token-mixers.
 *
 * The "linear attention" family that backs the non-softmax sequence mixers of
 * modern hybrid LLMs (gated DeltaNet, GLA, lightning/TransNormer, Kimi-Delta,
 * RWKV-7). They are all special cases of one general gated diagonal-plus-low-rank
 * (DPLR) linear recurrence over a per-head state S in R^{K x V}:
 *
 *     S_t = S_{t-1} * M_t + v_t k_t^T ,   o_t = q_t^T S_t        (k_t outer v_t)
 *
 * where the transition M_t specializes per variant:
 *   - gated_linear_attn : M_t = Diag(alpha_t)            (diagonal gate only)
 *   - gated_delta_rule  : M_t = Diag(g_t) - beta_t k_t k_t^T   (gate + tied rank-1)
 *   - rwkv_wkv7         : M_t = Diag(exp(w_t)) + b_t a_t^T  (free diagonal-plus-rank-1)
 *
 * kernel-set's strategy: dispatch routes these to the industry-best provider
 * (flash-linear-attention / FLA, Triton, sm80+); these C-ABI kernels are the
 * always-available PORTABLE FALLBACK — a correctness-first O(T) sequential
 * recurrence (fp32 state accumulation), not the chunked-parallel fast path.
 *
 * Layout (row-major, seq-first, NON-head-first — matches FLA's default):
 *   q,k        : [batch, seqlen, heads, k_dim]
 *   v, out     : [batch, seqlen, heads, v_dim]
 *   per-head state S is [k_dim, v_dim], accumulated in fp32.
 * Tensors are the model dtype (fp16/bf16/fp32) except scalar gates noted fp32.
 */
#ifndef KERNEL_SET_LINEAR_ATTN_H_
#define KERNEL_SET_LINEAR_ATTN_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* GATED DELTA RULE (DPLR/delta tier): gated DeltaNet (scalar decay) + Kimi-Delta
 * (KDA, diagonal decay) + JetBlock. Per step, per head:
 *     decay = exp(g_t)                       (g_is_vector ? per-k diagonal : scalar)
 *     S = Diag(decay) * S                    (row-scale the state over k)
 *     S = S + beta_t * k_t (v_t - S^T k_t)^T (the delta write, tied-key rank-1)
 *     o_t = q_t^T S
 *   q,k: [B,T,H,K]; v,out: [B,T,H,V]; beta: [B,T,H] (model dtype);
 *   g: [B,T,H] scalar (g_is_vector=0, GDN) or [B,T,H,K] diagonal (g_is_vector=1, KDA).
 *   use_qk_l2norm: L2-normalize q,k per (token,head) before use (1) or not (0).
 *   scale: query scale (e.g. 1/sqrt(K)); pass 0 for no extra scale (1.0). */
KS_API ks_status_t ks_gated_delta_rule(
    void* out, const void* q, const void* k, const void* v, const void* g,
    const void* beta, int64_t batch, int64_t seqlen, int64_t heads,
    int64_t k_dim, int64_t v_dim, int g_is_vector, int use_qk_l2norm,
    float scale, ks_dtype_t dtype, ks_stream_t stream);

/* GATED LINEAR ATTENTION (diagonal tier): GLA (data-dependent diagonal gate),
 * simple-GLA (scalar head decay), lightning/TransNormer (fixed per-head slope).
 * No rank-1 removal term. Per step, per head:
 *     S = Diag(alpha_t) * S + v_t k_t^T ;  o_t = q_t^T S
 *   alpha_t source by gate_mode:
 *     0 = data-dependent diagonal: g is [B,T,H,K] log-gate, alpha = exp(g)
 *     1 = scalar per-head data-dependent: g is [B,T,H] log-gate
 *     2 = fixed per-head slope: head_decay is [heads] (alpha = exp(-slope), g ignored)
 *   q,k: [B,T,H,K]; v,out: [B,T,H,V]; g model dtype or null; head_decay fp32 or null. */
KS_API ks_status_t ks_gated_linear_attn(
    void* out, const void* q, const void* k, const void* v, const void* g,
    const float* head_decay, int64_t batch, int64_t seqlen, int64_t heads,
    int64_t k_dim, int64_t v_dim, int gate_mode, float scale, ks_dtype_t dtype,
    ks_stream_t stream);

/* RWKV-7 "Goose" WKV (free DPLR / generalized delta rule). Per step, per head:
 *     S = Diag(exp(w_t)) S + b_t (a_t^T S) + k_t v_t^T ;  o_t = r_t^T S
 *   (matches FLA fused_recurrent_rwkv7 exactly, so the ks fallback and the FLA
 *   provider are interchangeable; RWKV-7 supplies a = -kk, b = kk * icl_rate.)
 *   r (receptance = query), w (log-decay -> Diag(exp(w))), k, a, b: [B,T,H,K];
 *   v, out: [B,T,H,V]. All model dtype. scale: query scale (0 => 1.0). */
KS_API ks_status_t ks_rwkv_wkv7(
    void* out, const void* r, const void* w, const void* k, const void* v,
    const void* a, const void* b, int64_t batch, int64_t seqlen, int64_t heads,
    int64_t k_dim, int64_t v_dim, float scale, ks_dtype_t dtype,
    ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_LINEAR_ATTN_H_ */
