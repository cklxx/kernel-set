/*
 * stub_kernel_set.c — CPU stub implementation of the kernel-set C ABI.
 *
 * Purpose: exercise the full FFI plumbing of every language binding WITHOUT a
 * GPU. Every KS_API entry point declared in the kernel_set headers is defined
 * here with the EXACT signature from the headers, as a trivial no-op:
 *   - functions returning ks_status_t set benign out-params and return
 *     KS_SUCCESS;
 *   - string/int introspection functions return fixed benign values.
 *
 * Device pointers are NEVER dereferenced, so callers may pass dummy/NULL
 * pointers safely. The math is intentionally meaningless — real kernel math is
 * verified separately on GPU. This proves the *interfaces* link, resolve, and
 * marshal correctly.
 *
 * Build:
 *   cc -shared -fPIC -I include -DKERNEL_SET_BUILD \
 *      bindings/verify/stub_kernel_set.c -o bindings/verify/libkernel_set.dylib
 */
#include "kernel_set/kernel_set.h"

/* ===================================================================== */
/* runtime.h — introspection                                              */
/* ===================================================================== */

const char* ks_version(void) { return "0.0.0-stub"; }

const char* ks_status_string(ks_status_t status) {
  (void)status;
  return "stub-status";
}

int ks_dtype_size_bits(ks_dtype_t dtype) {
  (void)dtype;
  return 16;
}

const char* ks_dtype_name(ks_dtype_t dtype) {
  (void)dtype;
  return "stub-dtype";
}

const char* ks_backend_name(void) { return "stub"; }

/* ===================================================================== */
/* runtime.h — device queries                                             */
/* ===================================================================== */

ks_status_t ks_device_count(int* out_count) {
  if (out_count) *out_count = 1;
  return KS_SUCCESS;
}

ks_status_t ks_set_device(int device) {
  (void)device;
  return KS_SUCCESS;
}

ks_status_t ks_get_device(int* out_device) {
  if (out_device) *out_device = 0;
  return KS_SUCCESS;
}

ks_status_t ks_get_device_properties(int device,
                                     ks_device_properties_t* out_props) {
  (void)device;
  if (out_props) {
    /* Fill a benign, internally-consistent set of properties. */
    const char* nm = "stub-device";
    int i = 0;
    for (; i < 255 && nm[i]; ++i) out_props->name[i] = nm[i];
    out_props->name[i] = '\0';
    out_props->compute_major = 0;
    out_props->compute_minor = 0;
    out_props->multiprocessor_count = 1;
    out_props->max_threads_per_block = 1024;
    out_props->max_shared_memory_per_block = 49152;
    out_props->warp_size = 32;
    out_props->total_global_memory = (size_t)0;
    out_props->supports_bf16 = 0;
    out_props->supports_fp8 = 0;
    out_props->supports_tf32 = 0;
  }
  return KS_SUCCESS;
}

const char* ks_last_error_string(void) { return ""; }

/* ===================================================================== */
/* runtime.h — streams                                                    */
/* ===================================================================== */

ks_status_t ks_stream_create(ks_stream_t* out_stream) {
  /* Hand back a non-NULL but bogus handle that we never dereference. */
  if (out_stream) *out_stream = (ks_stream_t)0;
  return KS_SUCCESS;
}

ks_status_t ks_stream_destroy(ks_stream_t stream) {
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_stream_synchronize(ks_stream_t stream) {
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* runtime.h — device memory                                              */
/* ===================================================================== */

ks_status_t ks_malloc_device(void** out_ptr, size_t bytes) {
  (void)bytes;
  /* Do NOT actually allocate; hand back a benign non-NULL sentinel that the
   * stub never dereferences. Callers must not read/write it. */
  if (out_ptr) *out_ptr = (void*)0;
  return KS_SUCCESS;
}

ks_status_t ks_free_device(void* ptr) {
  (void)ptr;
  return KS_SUCCESS;
}

ks_status_t ks_memcpy(void* dst, const void* src, size_t bytes,
                      ks_memcpy_kind_t kind, ks_stream_t stream) {
  (void)dst;
  (void)src;
  (void)bytes;
  (void)kind;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_memset_device(void* dst, int value, size_t bytes,
                             ks_stream_t stream) {
  (void)dst;
  (void)value;
  (void)bytes;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* activation.h                                                           */
/* ===================================================================== */

ks_status_t ks_silu(void* out, const void* input, int64_t n, ks_dtype_t dtype,
                    ks_stream_t stream) {
  (void)out;
  (void)input;
  (void)n;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_gelu(void* out, const void* input, int64_t n, int tanh_approx,
                    ks_dtype_t dtype, ks_stream_t stream) {
  (void)out;
  (void)input;
  (void)n;
  (void)tanh_approx;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_relu(void* out, const void* input, int64_t n, ks_dtype_t dtype,
                    ks_stream_t stream) {
  (void)out;
  (void)input;
  (void)n;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_swiglu(void* out, const void* gate, const void* up, int64_t rows,
                      int64_t inter, ks_dtype_t dtype, ks_stream_t stream) {
  (void)out;
  (void)gate;
  (void)up;
  (void)rows;
  (void)inter;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_swiglu_packed(void* out, const void* input, int64_t rows,
                             int64_t inter, ks_dtype_t dtype,
                             ks_stream_t stream) {
  (void)out;
  (void)input;
  (void)rows;
  (void)inter;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_geglu(void* out, const void* gate, const void* up, int64_t rows,
                     int64_t inter, int tanh_approx, ks_dtype_t dtype,
                     ks_stream_t stream) {
  (void)out;
  (void)gate;
  (void)up;
  (void)rows;
  (void)inter;
  (void)tanh_approx;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_swiglu_backward(void* grad_gate, void* grad_up,
                               const void* grad_out, const void* gate,
                               const void* up, int64_t rows, int64_t inter,
                               ks_dtype_t dtype, ks_stream_t stream) {
  (void)grad_gate;
  (void)grad_up;
  (void)grad_out;
  (void)gate;
  (void)up;
  (void)rows;
  (void)inter;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* attention.h                                                            */
/* ===================================================================== */

ks_status_t ks_flash_attn_varlen(void* out, void* softmax_lse, const void* q,
                                 const void* k, const void* v,
                                 const int32_t* cu_seqlens_q,
                                 const int32_t* cu_seqlens_k, int batch,
                                 int max_seqlen_q, int max_seqlen_k,
                                 int num_heads, int num_kv_heads, int head_dim,
                                 float softmax_scale, int causal,
                                 ks_dtype_t dtype, ks_stream_t stream) {
  (void)out;
  (void)softmax_lse;
  (void)q;
  (void)k;
  (void)v;
  (void)cu_seqlens_q;
  (void)cu_seqlens_k;
  (void)batch;
  (void)max_seqlen_q;
  (void)max_seqlen_k;
  (void)num_heads;
  (void)num_kv_heads;
  (void)head_dim;
  (void)softmax_scale;
  (void)causal;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_flash_attn(void* out, void* softmax_lse, const void* q,
                          const void* k, const void* v, int batch, int seqlen_q,
                          int seqlen_k, int num_heads, int num_kv_heads,
                          int head_dim, float softmax_scale, int causal,
                          ks_dtype_t dtype, ks_stream_t stream) {
  (void)out;
  (void)softmax_lse;
  (void)q;
  (void)k;
  (void)v;
  (void)batch;
  (void)seqlen_q;
  (void)seqlen_k;
  (void)num_heads;
  (void)num_kv_heads;
  (void)head_dim;
  (void)softmax_scale;
  (void)causal;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_paged_attn_decode(void* out, const void* q, const void* k_cache,
                                 const void* v_cache,
                                 const int32_t* block_tables,
                                 const int32_t* seq_lens, int num_seqs,
                                 int num_heads, int num_kv_heads, int head_dim,
                                 int block_size, int max_blocks_per_seq,
                                 float softmax_scale, ks_dtype_t dtype,
                                 ks_stream_t stream) {
  (void)out;
  (void)q;
  (void)k_cache;
  (void)v_cache;
  (void)block_tables;
  (void)seq_lens;
  (void)num_seqs;
  (void)num_heads;
  (void)num_kv_heads;
  (void)head_dim;
  (void)block_size;
  (void)max_blocks_per_seq;
  (void)softmax_scale;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_reshape_and_cache(void* k_cache, void* v_cache, const void* key,
                                 const void* value,
                                 const int32_t* slot_mapping, int num_tokens,
                                 int num_kv_heads, int head_dim, int block_size,
                                 ks_dtype_t dtype, ks_stream_t stream) {
  (void)k_cache;
  (void)v_cache;
  (void)key;
  (void)value;
  (void)slot_mapping;
  (void)num_tokens;
  (void)num_kv_heads;
  (void)head_dim;
  (void)block_size;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_mla_decode(void* out, const void* q_nope, const void* q_pe,
                          const void* kv_cache, const int32_t* block_tables,
                          const int32_t* seq_lens, int num_seqs, int num_heads,
                          int kv_lora_rank, int rope_dim, int block_size,
                          int max_blocks_per_seq, float softmax_scale,
                          ks_dtype_t dtype, ks_stream_t stream) {
  (void)out;
  (void)q_nope;
  (void)q_pe;
  (void)kv_cache;
  (void)block_tables;
  (void)seq_lens;
  (void)num_seqs;
  (void)num_heads;
  (void)kv_lora_rank;
  (void)rope_dim;
  (void)block_size;
  (void)max_blocks_per_seq;
  (void)softmax_scale;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_flash_attn_backward(void* grad_q, void* grad_k, void* grad_v,
                                   const void* grad_out, const void* q,
                                   const void* k, const void* v,
                                   const void* out, const void* softmax_lse,
                                   int batch, int seqlen_q, int seqlen_k,
                                   int num_heads, int num_kv_heads, int head_dim,
                                   float softmax_scale, int causal,
                                   ks_dtype_t dtype, ks_stream_t stream) {
  (void)grad_q;
  (void)grad_k;
  (void)grad_v;
  (void)grad_out;
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  (void)softmax_lse;
  (void)batch;
  (void)seqlen_q;
  (void)seqlen_k;
  (void)num_heads;
  (void)num_kv_heads;
  (void)head_dim;
  (void)softmax_scale;
  (void)causal;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* elementwise.h                                                          */
/* ===================================================================== */

ks_status_t ks_add(void* out, const void* a, const void* b, int64_t n,
                   ks_dtype_t dtype, ks_stream_t stream) {
  (void)out;
  (void)a;
  (void)b;
  (void)n;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_mul(void* out, const void* a, const void* b, int64_t n,
                   ks_dtype_t dtype, ks_stream_t stream) {
  (void)out;
  (void)a;
  (void)b;
  (void)n;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_add_residual(void* residual, const void* x, int64_t n,
                            ks_dtype_t dtype, ks_stream_t stream) {
  (void)residual;
  (void)x;
  (void)n;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_scale(void* out, const void* x, float scale, int64_t n,
                     ks_dtype_t dtype, ks_stream_t stream) {
  (void)out;
  (void)x;
  (void)scale;
  (void)n;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_cast(void* out, ks_dtype_t dst_dtype, const void* in,
                    ks_dtype_t src_dtype, int64_t n, ks_stream_t stream) {
  (void)out;
  (void)dst_dtype;
  (void)in;
  (void)src_dtype;
  (void)n;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_axpby(void* out, const void* a, float alpha, const void* b,
                     float beta, int64_t n, ks_dtype_t dtype,
                     ks_stream_t stream) {
  (void)out;
  (void)a;
  (void)alpha;
  (void)b;
  (void)beta;
  (void)n;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* embedding.h                                                            */
/* ===================================================================== */

ks_status_t ks_embedding_lookup(void* out, const void* table,
                                const void* indices, int indices_i64,
                                int64_t num_tokens, int64_t embed_dim,
                                ks_dtype_t dtype, ks_stream_t stream) {
  (void)out;
  (void)table;
  (void)indices;
  (void)indices_i64;
  (void)num_tokens;
  (void)embed_dim;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_embedding_backward(void* grad_table_fp32, const void* grad_out,
                                  const void* indices, int indices_i64,
                                  int64_t num_tokens, int64_t embed_dim,
                                  ks_dtype_t dtype, ks_stream_t stream) {
  (void)grad_table_fp32;
  (void)grad_out;
  (void)indices;
  (void)indices_i64;
  (void)num_tokens;
  (void)embed_dim;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* gemm.h                                                                 */
/* ===================================================================== */

ks_status_t ks_gemm(void* c, const void* a, const void* b, int64_t m, int64_t n,
                    int64_t k, int trans_a, int trans_b, int64_t lda,
                    int64_t ldb, int64_t ldc, float alpha, float beta,
                    ks_dtype_t dtype, ks_stream_t stream) {
  (void)c;
  (void)a;
  (void)b;
  (void)m;
  (void)n;
  (void)k;
  (void)trans_a;
  (void)trans_b;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)alpha;
  (void)beta;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_gemm_bias_act(void* d, const void* a, const void* b,
                             const void* bias, int64_t m, int64_t n, int64_t k,
                             float alpha, ks_activation_t act, ks_dtype_t dtype,
                             ks_stream_t stream) {
  (void)d;
  (void)a;
  (void)b;
  (void)bias;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)act;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_gemm_batched(void* c, const void* a, const void* b,
                            int64_t batch, int64_t m, int64_t n, int64_t k,
                            int trans_a, int trans_b, int64_t stride_a,
                            int64_t stride_b, int64_t stride_c, float alpha,
                            float beta, ks_dtype_t dtype, ks_stream_t stream) {
  (void)c;
  (void)a;
  (void)b;
  (void)batch;
  (void)m;
  (void)n;
  (void)k;
  (void)trans_a;
  (void)trans_b;
  (void)stride_a;
  (void)stride_b;
  (void)stride_c;
  (void)alpha;
  (void)beta;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_gemm_w8a8(void* c, const void* a_i8, const void* b_i8,
                         const float* a_scale, const float* b_scale,
                         const void* bias, int64_t m, int64_t n, int64_t k,
                         ks_quant_mode_t a_mode, ks_quant_mode_t b_mode,
                         ks_dtype_t out_dtype, ks_stream_t stream) {
  (void)c;
  (void)a_i8;
  (void)b_i8;
  (void)a_scale;
  (void)b_scale;
  (void)bias;
  (void)m;
  (void)n;
  (void)k;
  (void)a_mode;
  (void)b_mode;
  (void)out_dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_gemm_w4a16(void* c, const void* a, const void* b_packed,
                          const void* scales, const void* zeros,
                          const void* bias, int64_t m, int64_t n, int64_t k,
                          int group_size, ks_dtype_t dtype,
                          ks_stream_t stream) {
  (void)c;
  (void)a;
  (void)b_packed;
  (void)scales;
  (void)zeros;
  (void)bias;
  (void)m;
  (void)n;
  (void)k;
  (void)group_size;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* loss.h                                                                 */
/* ===================================================================== */

ks_status_t ks_cross_entropy(float* losses, void* grad_logits,
                             const void* logits, const void* targets,
                             int targets_i64, int64_t num_tokens, int64_t vocab,
                             int64_t ignore_index, float label_smoothing,
                             ks_dtype_t dtype, ks_stream_t stream) {
  (void)losses;
  (void)grad_logits;
  (void)logits;
  (void)targets;
  (void)targets_i64;
  (void)num_tokens;
  (void)vocab;
  (void)ignore_index;
  (void)label_smoothing;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_fused_linear_cross_entropy(
    float* losses, void* grad_hidden, void* grad_weight_fp32,
    const void* hidden, const void* weight, const void* targets,
    int targets_i64, int64_t num_tokens, int64_t hidden_dim, int64_t vocab,
    int64_t ignore_index, float label_smoothing, int chunk_size,
    ks_dtype_t dtype, ks_stream_t stream) {
  (void)losses;
  (void)grad_hidden;
  (void)grad_weight_fp32;
  (void)hidden;
  (void)weight;
  (void)targets;
  (void)targets_i64;
  (void)num_tokens;
  (void)hidden_dim;
  (void)vocab;
  (void)ignore_index;
  (void)label_smoothing;
  (void)chunk_size;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* moe.h                                                                  */
/* ===================================================================== */

ks_status_t ks_moe_gate_softmax_topk(float* out_weights, int32_t* out_indices,
                                     const void* logits, int64_t num_tokens,
                                     int num_experts, int top_k, int renormalize,
                                     ks_dtype_t dtype, ks_stream_t stream) {
  (void)out_weights;
  (void)out_indices;
  (void)logits;
  (void)num_tokens;
  (void)num_experts;
  (void)top_k;
  (void)renormalize;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_moe_gate_sigmoid_group_topk(
    float* out_weights, int32_t* out_indices, const void* logits,
    const void* correction_bias, int64_t num_tokens, int num_experts,
    int n_group, int topk_group, int top_k, int renormalize,
    float routed_scaling_factor, ks_dtype_t dtype, ks_stream_t stream) {
  (void)out_weights;
  (void)out_indices;
  (void)logits;
  (void)correction_bias;
  (void)num_tokens;
  (void)num_experts;
  (void)n_group;
  (void)topk_group;
  (void)top_k;
  (void)renormalize;
  (void)routed_scaling_factor;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_moe_compute_permutation(int32_t* sorted_token_ids,
                                       int32_t* expert_offsets,
                                       const int32_t* topk_indices,
                                       int64_t num_tokens, int num_experts,
                                       int top_k, ks_stream_t stream) {
  (void)sorted_token_ids;
  (void)expert_offsets;
  (void)topk_indices;
  (void)num_tokens;
  (void)num_experts;
  (void)top_k;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_moe_permute(void* permuted, const void* input,
                           const int32_t* sorted_token_ids, int64_t num_tokens,
                           int top_k, int64_t hidden, ks_dtype_t dtype,
                           ks_stream_t stream) {
  (void)permuted;
  (void)input;
  (void)sorted_token_ids;
  (void)num_tokens;
  (void)top_k;
  (void)hidden;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_moe_unpermute(void* out, const void* permuted,
                             const int32_t* sorted_token_ids,
                             const float* routing_weights, int64_t num_tokens,
                             int top_k, int64_t hidden, ks_dtype_t dtype,
                             ks_stream_t stream) {
  (void)out;
  (void)permuted;
  (void)sorted_token_ids;
  (void)routing_weights;
  (void)num_tokens;
  (void)top_k;
  (void)hidden;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_moe_grouped_gemm(void* c, const void* a, const void* b,
                                const int32_t* expert_offsets, int num_experts,
                                int64_t total_rows, int64_t n, int64_t k,
                                ks_dtype_t dtype, ks_stream_t stream) {
  (void)c;
  (void)a;
  (void)b;
  (void)expert_offsets;
  (void)num_experts;
  (void)total_rows;
  (void)n;
  (void)k;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* norm.h                                                                 */
/* ===================================================================== */

ks_status_t ks_rms_norm(void* out, const void* input, const void* weight,
                        int64_t rows, int64_t cols, float eps, ks_dtype_t dtype,
                        ks_stream_t stream) {
  (void)out;
  (void)input;
  (void)weight;
  (void)rows;
  (void)cols;
  (void)eps;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_fused_rmsnorm_gated(void* out, const void* input,
                                   const void* weight, const void* gate,
                                   int64_t rows, int64_t cols, int gate_act,
                                   float eps, ks_dtype_t dtype,
                                   ks_stream_t stream) {
  (void)out; (void)input; (void)weight; (void)gate; (void)rows; (void)cols;
  (void)gate_act; (void)eps; (void)dtype; (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_attention_state_merge(void* out, float* lse, const void* out_a,
                                     const float* lse_a, const void* out_b,
                                     const float* lse_b, int64_t n_rows,
                                     int64_t v_dim, ks_dtype_t dtype,
                                     ks_stream_t stream) {
  (void)out; (void)lse; (void)out_a; (void)lse_a; (void)out_b; (void)lse_b;
  (void)n_rows; (void)v_dim; (void)dtype; (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_dsa_topk_select(int32_t* indices, const void* scores,
                               int64_t n_rows, int64_t n_cols, int topk,
                               ks_dtype_t dtype, ks_stream_t stream) {
  (void)indices; (void)scores; (void)n_rows; (void)n_cols; (void)topk;
  (void)dtype; (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_rms_norm_residual(void* out, void* residual_out,
                                 const void* input, const void* residual,
                                 const void* weight, int64_t rows, int64_t cols,
                                 float eps, ks_dtype_t dtype,
                                 ks_stream_t stream) {
  (void)out;
  (void)residual_out;
  (void)input;
  (void)residual;
  (void)weight;
  (void)rows;
  (void)cols;
  (void)eps;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_layer_norm(void* out, const void* input, const void* weight,
                          const void* bias, int64_t rows, int64_t cols,
                          float eps, ks_dtype_t dtype, ks_stream_t stream) {
  (void)out;
  (void)input;
  (void)weight;
  (void)bias;
  (void)rows;
  (void)cols;
  (void)eps;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_rms_norm_backward(void* grad_input, void* grad_weight_fp32,
                                 const void* grad_out, const void* input,
                                 const void* weight, int64_t rows, int64_t cols,
                                 float eps, ks_dtype_t dtype,
                                 ks_stream_t stream) {
  (void)grad_input;
  (void)grad_weight_fp32;
  (void)grad_out;
  (void)input;
  (void)weight;
  (void)rows;
  (void)cols;
  (void)eps;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_layer_norm_backward(void* grad_input, void* grad_weight_fp32,
                                   void* grad_bias_fp32, const void* grad_out,
                                   const void* input, const void* weight,
                                   int64_t rows, int64_t cols, float eps,
                                   ks_dtype_t dtype, ks_stream_t stream) {
  (void)grad_input;
  (void)grad_weight_fp32;
  (void)grad_bias_fp32;
  (void)grad_out;
  (void)input;
  (void)weight;
  (void)rows;
  (void)cols;
  (void)eps;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* optimizer.h                                                            */
/* ===================================================================== */

ks_status_t ks_adamw(void* param, void* master_param, const void* grad,
                     float* exp_avg, float* exp_avg_sq, float lr, float beta1,
                     float beta2, float eps, float weight_decay, int64_t step,
                     float grad_scale, int64_t n, ks_dtype_t dtype,
                     ks_stream_t stream) {
  (void)param;
  (void)master_param;
  (void)grad;
  (void)exp_avg;
  (void)exp_avg_sq;
  (void)lr;
  (void)beta1;
  (void)beta2;
  (void)eps;
  (void)weight_decay;
  (void)step;
  (void)grad_scale;
  (void)n;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_sgd_momentum(void* param, void* master_param, const void* grad,
                            float* momentum, float lr, float momentum_factor,
                            float weight_decay, int nesterov, float grad_scale,
                            int64_t n, ks_dtype_t dtype, ks_stream_t stream) {
  (void)param;
  (void)master_param;
  (void)grad;
  (void)momentum;
  (void)lr;
  (void)momentum_factor;
  (void)weight_decay;
  (void)nesterov;
  (void)grad_scale;
  (void)n;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_global_grad_norm(float* out_norm, const void* const* grads,
                                const int64_t* sizes, int num_tensors,
                                ks_dtype_t dtype, ks_stream_t stream) {
  if (out_norm) *out_norm = 0.0f;
  (void)grads;
  (void)sizes;
  (void)num_tensors;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* quant.h                                                                */
/* ===================================================================== */

ks_status_t ks_quantize_fp8(void* out, float* scale, const void* input,
                            int64_t rows, int64_t cols, ks_dtype_t in_dtype,
                            ks_dtype_t fp8_dtype, ks_quant_mode_t mode,
                            ks_stream_t stream) {
  (void)out;
  (void)scale;
  (void)input;
  (void)rows;
  (void)cols;
  (void)in_dtype;
  (void)fp8_dtype;
  (void)mode;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_quantize_fp8_group(void* out, float* scale, const void* input,
                                  int64_t rows, int64_t cols, int group_size,
                                  ks_dtype_t in_dtype, ks_dtype_t fp8_dtype,
                                  ks_stream_t stream) {
  (void)out;
  (void)scale;
  (void)input;
  (void)rows;
  (void)cols;
  (void)group_size;
  (void)in_dtype;
  (void)fp8_dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_dequantize_fp8(void* out, const void* input, const float* scale,
                              int64_t rows, int64_t cols, ks_dtype_t out_dtype,
                              ks_dtype_t fp8_dtype, ks_quant_mode_t mode,
                              ks_stream_t stream) {
  (void)out;
  (void)input;
  (void)scale;
  (void)rows;
  (void)cols;
  (void)out_dtype;
  (void)fp8_dtype;
  (void)mode;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_quantize_int8(void* out, float* scale, const void* input,
                             int64_t rows, int64_t cols, ks_dtype_t in_dtype,
                             ks_quant_mode_t mode, ks_stream_t stream) {
  (void)out;
  (void)scale;
  (void)input;
  (void)rows;
  (void)cols;
  (void)in_dtype;
  (void)mode;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_dequantize_int8(void* out, const void* input, const float* scale,
                               int64_t rows, int64_t cols, ks_dtype_t out_dtype,
                               ks_quant_mode_t mode, ks_stream_t stream) {
  (void)out;
  (void)input;
  (void)scale;
  (void)rows;
  (void)cols;
  (void)out_dtype;
  (void)mode;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_dequantize_int4(void* out, const void* qweight_packed,
                               const void* scales, const void* zeros, int64_t k,
                               int64_t n, int group_size, ks_dtype_t out_dtype,
                               ks_stream_t stream) {
  (void)out;
  (void)qweight_packed;
  (void)scales;
  (void)zeros;
  (void)k;
  (void)n;
  (void)group_size;
  (void)out_dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* rope.h                                                                 */
/* ===================================================================== */

ks_status_t ks_rope_inplace(void* q, void* k, const void* cos, const void* sin,
                            int64_t num_tokens, int num_q_heads,
                            int num_kv_heads, int head_dim, int interleaved,
                            ks_dtype_t dtype, ks_stream_t stream) {
  (void)q;
  (void)k;
  (void)cos;
  (void)sin;
  (void)num_tokens;
  (void)num_q_heads;
  (void)num_kv_heads;
  (void)head_dim;
  (void)interleaved;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_rope(void* q_out, void* k_out, const void* q, const void* k,
                    const void* cos, const void* sin, int64_t num_tokens,
                    int num_q_heads, int num_kv_heads, int head_dim,
                    int interleaved, ks_dtype_t dtype, ks_stream_t stream) {
  (void)q_out;
  (void)k_out;
  (void)q;
  (void)k;
  (void)cos;
  (void)sin;
  (void)num_tokens;
  (void)num_q_heads;
  (void)num_kv_heads;
  (void)head_dim;
  (void)interleaved;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_rope_gather(void* q, void* k, const void* cos_cache,
                           const void* sin_cache, const int32_t* positions,
                           int64_t num_tokens, int num_q_heads,
                           int num_kv_heads, int head_dim, int interleaved,
                           ks_dtype_t dtype, ks_stream_t stream) {
  (void)q;
  (void)k;
  (void)cos_cache;
  (void)sin_cache;
  (void)positions;
  (void)num_tokens;
  (void)num_q_heads;
  (void)num_kv_heads;
  (void)head_dim;
  (void)interleaved;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_rope_backward(void* grad_q, void* grad_k, const void* cos,
                             const void* sin, int64_t num_tokens,
                             int num_q_heads, int num_kv_heads, int head_dim,
                             int interleaved, ks_dtype_t dtype,
                             ks_stream_t stream) {
  (void)grad_q;
  (void)grad_k;
  (void)cos;
  (void)sin;
  (void)num_tokens;
  (void)num_q_heads;
  (void)num_kv_heads;
  (void)head_dim;
  (void)interleaved;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* sampling.h                                                             */
/* ===================================================================== */

ks_status_t ks_softmax(void* out, const void* input, int64_t rows, int64_t cols,
                       float temperature, ks_dtype_t dtype, ks_stream_t stream) {
  (void)out;
  (void)input;
  (void)rows;
  (void)cols;
  (void)temperature;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_log_softmax(void* out, const void* input, int64_t rows,
                           int64_t cols, ks_dtype_t dtype, ks_stream_t stream) {
  (void)out;
  (void)input;
  (void)rows;
  (void)cols;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_argmax(int32_t* out_tokens, const void* logits, int64_t num_seqs,
                      int64_t vocab_size, ks_dtype_t dtype, ks_stream_t stream) {
  (void)out_tokens;
  (void)logits;
  (void)num_seqs;
  (void)vocab_size;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_sample(int32_t* out_tokens, float* out_probs, const void* logits,
                      const float* temperatures, const int32_t* top_ks,
                      const float* top_ps, int64_t num_seqs, int64_t vocab_size,
                      uint64_t seed, uint64_t philox_offset, ks_dtype_t dtype,
                      ks_stream_t stream) {
  (void)out_tokens;
  (void)out_probs;
  (void)logits;
  (void)temperatures;
  (void)top_ks;
  (void)top_ps;
  (void)num_seqs;
  (void)vocab_size;
  (void)seed;
  (void)philox_offset;
  (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* ssm.h — Mamba state-space (added with the SSM ABI)                     */
/* ===================================================================== */

ks_status_t ks_causal_conv1d(void* out, const void* x, const void* weight,
                             const void* bias, int batch, int dim, int seqlen,
                             int width, int silu, ks_dtype_t dtype,
                             ks_stream_t stream) {
  (void)out; (void)x; (void)weight; (void)bias; (void)batch; (void)dim;
  (void)seqlen; (void)width; (void)silu; (void)dtype; (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_selective_scan(void* out, const void* x, const void* dt,
                              const void* A, const void* B, const void* C,
                              const void* D, const void* z, const void* dt_bias,
                              int delta_softplus, int batch, int dim, int seqlen,
                              int dstate, ks_dtype_t dtype, ks_stream_t stream) {
  (void)out; (void)x; (void)dt; (void)A; (void)B; (void)C; (void)D; (void)z;
  (void)dt_bias; (void)delta_softplus; (void)batch; (void)dim; (void)seqlen;
  (void)dstate; (void)dtype; (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_selective_scan_update(void* state, void* out, const void* x,
                                     const void* dt, const void* A,
                                     const void* B, const void* C, const void* D,
                                     const void* z, const void* dt_bias,
                                     int delta_softplus, int batch, int dim,
                                     int dstate, ks_dtype_t dtype,
                                     ks_stream_t stream) {
  (void)state; (void)out; (void)x; (void)dt; (void)A; (void)B; (void)C; (void)D;
  (void)z; (void)dt_bias; (void)delta_softplus; (void)batch; (void)dim;
  (void)dstate; (void)dtype; (void)stream;
  return KS_SUCCESS;
}

/* ---- linear_attn.h -------------------------------------------------- */
ks_status_t ks_gated_delta_rule(void* out, const void* q, const void* k,
                                const void* v, const void* g, const void* beta,
                                int64_t batch, int64_t seqlen, int64_t heads,
                                int64_t k_dim, int64_t v_dim, int g_is_vector,
                                int use_qk_l2norm, float scale, ks_dtype_t dtype,
                                ks_stream_t stream) {
  (void)out; (void)q; (void)k; (void)v; (void)g; (void)beta; (void)batch;
  (void)seqlen; (void)heads; (void)k_dim; (void)v_dim; (void)g_is_vector;
  (void)use_qk_l2norm; (void)scale; (void)dtype; (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_gated_linear_attn(void* out, const void* q, const void* k,
                                 const void* v, const void* g,
                                 const float* head_decay, int64_t batch,
                                 int64_t seqlen, int64_t heads, int64_t k_dim,
                                 int64_t v_dim, int gate_mode, float scale,
                                 ks_dtype_t dtype, ks_stream_t stream) {
  (void)out; (void)q; (void)k; (void)v; (void)g; (void)head_decay; (void)batch;
  (void)seqlen; (void)heads; (void)k_dim; (void)v_dim; (void)gate_mode;
  (void)scale; (void)dtype; (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_rwkv_wkv7(void* out, const void* r, const void* w, const void* k,
                         const void* v, const void* a, const void* b,
                         int64_t batch, int64_t seqlen, int64_t heads,
                         int64_t k_dim, int64_t v_dim, float scale,
                         ks_dtype_t dtype, ks_stream_t stream) {
  (void)out; (void)r; (void)w; (void)k; (void)v; (void)a; (void)b; (void)batch;
  (void)seqlen; (void)heads; (void)k_dim; (void)v_dim; (void)scale; (void)dtype;
  (void)stream;
  return KS_SUCCESS;
}

/* ===================================================================== */
/* gemm.h — native FP8 GEMM (added with the fp8 ABI)                      */
/* ===================================================================== */

ks_status_t ks_gemm_fp8(void* out, const void* a_fp8, const void* b_fp8,
                        const float* a_scale, const float* b_scale, int64_t m,
                        int64_t n, int64_t k, ks_quant_mode_t a_mode,
                        ks_quant_mode_t b_mode, ks_dtype_t fp8_dtype,
                        ks_dtype_t out_dtype, ks_stream_t stream) {
  (void)out; (void)a_fp8; (void)b_fp8; (void)a_scale; (void)b_scale; (void)m;
  (void)n; (void)k; (void)a_mode; (void)b_mode; (void)fp8_dtype; (void)out_dtype;
  (void)stream;
  return KS_SUCCESS;
}

ks_status_t ks_gemm_fp8_blockwise(void* out, const void* a_fp8,
                                  const void* b_fp8, const float* a_scale,
                                  const float* b_scale, int64_t m, int64_t n,
                                  int64_t k, int block_n, int block_k,
                                  ks_dtype_t fp8_dtype, ks_dtype_t out_dtype,
                                  ks_stream_t stream) {
  (void)out; (void)a_fp8; (void)b_fp8; (void)a_scale; (void)b_scale; (void)m;
  (void)n; (void)k; (void)block_n; (void)block_k; (void)fp8_dtype;
  (void)out_dtype; (void)stream;
  return KS_SUCCESS;
}
