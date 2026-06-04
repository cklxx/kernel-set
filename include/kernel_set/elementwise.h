/*
 * kernel-set — elementwise & casting kernels.
 *
 * All ops run over `n` contiguous elements. Vectorized 128-bit loads/stores are
 * used when alignment permits.
 */
#ifndef KERNEL_SET_ELEMENTWISE_H_
#define KERNEL_SET_ELEMENTWISE_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* out = a + b. */
KS_API ks_status_t ks_add(void* out, const void* a, const void* b, int64_t n,
                          ks_dtype_t dtype, ks_stream_t stream);

/* out = a * b (Hadamard). */
KS_API ks_status_t ks_mul(void* out, const void* a, const void* b, int64_t n,
                          ks_dtype_t dtype, ks_stream_t stream);

/* in-place residual accumulation: residual <- residual + x. */
KS_API ks_status_t ks_add_residual(void* residual, const void* x, int64_t n,
                                   ks_dtype_t dtype, ks_stream_t stream);

/* out = x * scale (scalar). */
KS_API ks_status_t ks_scale(void* out, const void* x, float scale, int64_t n,
                            ks_dtype_t dtype, ks_stream_t stream);

/* out[i] = cast<dst>(in[i]). dtype conversion (e.g. bf16 -> f32). */
KS_API ks_status_t ks_cast(void* out, ks_dtype_t dst_dtype, const void* in,
                           ks_dtype_t src_dtype, int64_t n, ks_stream_t stream);

/* out = a*alpha + b*beta  (fused AXPBY). */
KS_API ks_status_t ks_axpby(void* out, const void* a, float alpha,
                            const void* b, float beta, int64_t n,
                            ks_dtype_t dtype, ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_ELEMENTWISE_H_ */
