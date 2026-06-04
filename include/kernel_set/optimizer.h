/*
 * kernel-set — fused optimizer step kernels.
 *
 * Each kernel updates a single parameter tensor of `n` elements in place. State
 * tensors (exp_avg, exp_avg_sq, momentum) are fp32 for stability; `param` and
 * `grad` may be lower precision (`dtype`). Master fp32 weights are supported via
 * the optional `master_param` pointer.
 */
#ifndef KERNEL_SET_OPTIMIZER_H_
#define KERNEL_SET_OPTIMIZER_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* AdamW (decoupled weight decay). `step` is the 1-based iteration for bias
 * correction. If master_param != NULL it holds the fp32 master copy that is
 * updated and then cast back into `param`. */
KS_API ks_status_t ks_adamw(void* param, void* master_param, const void* grad,
                            float* exp_avg, float* exp_avg_sq, float lr,
                            float beta1, float beta2, float eps,
                            float weight_decay, int64_t step, float grad_scale,
                            int64_t n, ks_dtype_t dtype, ks_stream_t stream);

/* SGD with (optional Nesterov) momentum and weight decay. */
KS_API ks_status_t ks_sgd_momentum(void* param, void* master_param,
                                   const void* grad, float* momentum, float lr,
                                   float momentum_factor, float weight_decay,
                                   int nesterov, float grad_scale, int64_t n,
                                   ks_dtype_t dtype, ks_stream_t stream);

/* Compute the global L2 norm of a set of grad tensors (for clipping).
 *   grads: array of `num_tensors` device pointers; sizes: their element counts.
 *   out_norm: fp32 [1] device scalar = sqrt(sum ||g||^2). */
KS_API ks_status_t ks_global_grad_norm(float* out_norm, const void* const* grads,
                                       const int64_t* sizes, int num_tensors,
                                       ks_dtype_t dtype, ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_OPTIMIZER_H_ */
