/*
 * kernel-set — activation kernels and gated-MLP fusions.
 *
 * Elementwise ops run over `n` contiguous elements. Gated variants implement
 * the SwiGLU / GeGLU pattern used by Llama-family MLPs.
 */
#ifndef KERNEL_SET_ACTIVATION_H_
#define KERNEL_SET_ACTIVATION_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* out = silu(x) = x * sigmoid(x). */
KS_API ks_status_t ks_silu(void* out, const void* input, int64_t n,
                           ks_dtype_t dtype, ks_stream_t stream);

/* out = gelu(x). `tanh_approx != 0` selects the tanh approximation. */
KS_API ks_status_t ks_gelu(void* out, const void* input, int64_t n,
                           int tanh_approx, ks_dtype_t dtype,
                           ks_stream_t stream);

KS_API ks_status_t ks_relu(void* out, const void* input, int64_t n,
                           ks_dtype_t dtype, ks_stream_t stream);

/* out = act(gate) * up. `gate` and `up` are each [rows, inter]. */
KS_API ks_status_t ks_swiglu(void* out, const void* gate, const void* up,
                             int64_t rows, int64_t inter, ks_dtype_t dtype,
                             ks_stream_t stream);

/* Packed variant: input is [rows, 2*inter] with gate = input[:, :inter] and
 * up = input[:, inter:]. out is [rows, inter]. */
KS_API ks_status_t ks_swiglu_packed(void* out, const void* input, int64_t rows,
                                    int64_t inter, ks_dtype_t dtype,
                                    ks_stream_t stream);

/* out = gelu(gate) * up (GeGLU). `tanh_approx` selects the approximation. */
KS_API ks_status_t ks_geglu(void* out, const void* gate, const void* up,
                            int64_t rows, int64_t inter, int tanh_approx,
                            ks_dtype_t dtype, ks_stream_t stream);

/* ---- Training backward passes ------------------------------------------- */

/* d_gate, d_up from d_out given the forward gate/up tensors. */
KS_API ks_status_t ks_swiglu_backward(void* grad_gate, void* grad_up,
                                      const void* grad_out, const void* gate,
                                      const void* up, int64_t rows,
                                      int64_t inter, ks_dtype_t dtype,
                                      ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_ACTIVATION_H_ */
