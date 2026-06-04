/*
 * kernel-set — token embedding lookup / scatter.
 */
#ifndef KERNEL_SET_EMBEDDING_H_
#define KERNEL_SET_EMBEDDING_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* out[i, :] = table[indices[i], :].
 *   table:   [vocab_size, embed_dim] of `dtype`
 *   indices: [num_tokens] int32 (or int64 if indices_i64 != 0)
 *   out:     [num_tokens, embed_dim] of `dtype` */
KS_API ks_status_t ks_embedding_lookup(void* out, const void* table,
                                       const void* indices, int indices_i64,
                                       int64_t num_tokens, int64_t embed_dim,
                                       ks_dtype_t dtype, ks_stream_t stream);

/* grad_table[indices[i], :] += grad_out[i, :]  (atomic scatter-add).
 * grad_table is fp32 [vocab_size, embed_dim]. */
KS_API ks_status_t ks_embedding_backward(void* grad_table_fp32,
                                         const void* grad_out,
                                         const void* indices, int indices_i64,
                                         int64_t num_tokens, int64_t embed_dim,
                                         ks_dtype_t dtype, ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_EMBEDDING_H_ */
