/*
 * kernel-set — normalization kernels (RMSNorm, LayerNorm) + fused residual.
 *
 * Layout convention: 2-D row-major [rows, cols]. Each row of `cols` elements is
 * normalized independently. `weight`/`bias` have `cols` elements. All float
 * pointers are device pointers of element type `dtype` unless noted; reductions
 * are accumulated in fp32 internally for numerical stability.
 */
#ifndef KERNEL_SET_NORM_H_
#define KERNEL_SET_NORM_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* out = (x / rms(x)) * weight,  rms(x) = sqrt(mean(x^2) + eps). */
KS_API ks_status_t ks_rms_norm(void* out, const void* input, const void* weight,
                               int64_t rows, int64_t cols, float eps,
                               ks_dtype_t dtype, ks_stream_t stream);

/* Fused add-RMSNorm used between transformer sub-layers:
 *   x      <- input + residual      (the new residual, written to residual_out)
 *   out    <- rms_norm(x) * weight
 * `residual_out` may alias `residual` for in-place accumulation. */
KS_API ks_status_t ks_rms_norm_residual(void* out, void* residual_out,
                                        const void* input, const void* residual,
                                        const void* weight, int64_t rows,
                                        int64_t cols, float eps,
                                        ks_dtype_t dtype, ks_stream_t stream);

/* out = (x - mean) / sqrt(var + eps) * weight + bias. `bias` may be NULL. */
KS_API ks_status_t ks_layer_norm(void* out, const void* input,
                                 const void* weight, const void* bias,
                                 int64_t rows, int64_t cols, float eps,
                                 ks_dtype_t dtype, ks_stream_t stream);

/* ---- Training backward passes ------------------------------------------- */

/* Given grad_out and the forward inputs, produce grad_input and grad_weight.
 * grad_weight is accumulated in fp32; pass a [cols] fp32 buffer. */
KS_API ks_status_t ks_rms_norm_backward(void* grad_input, void* grad_weight_fp32,
                                        const void* grad_out, const void* input,
                                        const void* weight, int64_t rows,
                                        int64_t cols, float eps,
                                        ks_dtype_t dtype, ks_stream_t stream);

KS_API ks_status_t ks_layer_norm_backward(
    void* grad_input, void* grad_weight_fp32, void* grad_bias_fp32,
    const void* grad_out, const void* input, const void* weight,
    int64_t rows, int64_t cols, float eps, ks_dtype_t dtype,
    ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_NORM_H_ */
