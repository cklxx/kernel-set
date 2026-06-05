/*
 * kernel-set — state-space model (Mamba / SSM) kernels.
 *
 * Two primitives back the Mamba block:
 *   1. ks_causal_conv1d      — depthwise causal 1-D convolution (+ optional SiLU)
 *   2. ks_selective_scan     — the data-dependent (selective) SSM scan
 *   3. ks_selective_scan_update — single-step decode update of the SSM state
 *
 * Layout conventions (row-major, channel-major like the reference Mamba CUDA):
 *   - Sequence tensors are [batch, dim, seqlen]: the sequence axis is innermost
 *     (contiguous), the channel/feature axis `dim` is the middle stride, and
 *     `batch` is outermost. This is the (B, D, L) layout produced after the
 *     in-projection + transpose in a Mamba block.
 *   - B and C carry the *selective* (input-dependent) projections and are shared
 *     across all `dim` channels: [batch, dstate, seqlen]. (This is the n_groups=1
 *     case; every channel reads the same B/C, broadcast over `dim`.)
 *
 * Dtype policy. The bulk activation tensors (x, dt, B, C, z, out, conv weight,
 * conv bias) use the dispatched `dtype` (f32 / f16 / bf16) and are converted to
 * fp32 for all arithmetic, matching the house rule of fp32 accumulation. The
 * recurrence *parameters* A, D and dt_bias are ALWAYS fp32 device buffers
 * regardless of `dtype` — they are tiny ([dim,dstate] / [dim] / [dim]), are kept
 * in fp32 by the reference implementation, and keeping them fp32 avoids any
 * precision loss in the exp(dt*A) state transition. Pass them as float*.
 *
 * All pointers are device pointers; `stream` is a ks_stream_t (NULL = default).
 * Every entry point returns ks_status_t (KS_SUCCESS == 0).
 */
#ifndef KERNEL_SET_SSM_H_
#define KERNEL_SET_SSM_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/*
 * Depthwise causal 1-D convolution.
 *
 *   x:      [batch, dim, seqlen]   input  (dtype)
 *   weight: [dim, width]           per-channel kernel (dtype), width is small (~4)
 *   bias:   [dim]                  optional per-channel bias (fp32), or NULL
 *   out:    [batch, dim, seqlen]   output (dtype)
 *
 * For each (batch b, channel d, position t):
 *   out[b,d,t] = bias[d] + sum_{k=0..width-1} weight[d,k] * x[b,d, t - (width-1) + k]
 * with implicit zero left-padding (x at negative positions reads 0) so the
 * convolution is causal: out[t] depends only on inputs at positions <= t.
 * If `silu != 0`, a SiLU activation (x * sigmoid(x)) is applied to the result.
 *
 * `bias` is fp32 (cast to float* by the caller) to match the recurrence-param
 * dtype policy above; pass NULL to skip the bias add.
 */
KS_API ks_status_t ks_causal_conv1d(void* out, const void* x, const void* weight,
                                    const void* bias, int batch, int dim,
                                    int seqlen, int width, int silu,
                                    ks_dtype_t dtype, ks_stream_t stream);

/*
 * Mamba selective scan (forward).
 *
 *   x:        [batch, dim, seqlen]      input sequence      (dtype)
 *   dt:       [batch, dim, seqlen]      per-(channel,time) timestep delta (dtype)
 *   A:        [dim, dstate]             state transition (fp32, typically < 0)
 *   B:        [batch, dstate, seqlen]   input projection    (dtype, shared over dim)
 *   C:        [batch, dstate, seqlen]   output projection   (dtype, shared over dim)
 *   D:        [dim]                     skip / residual (fp32) or NULL
 *   z:        [batch, dim, seqlen]      SiLU gate (dtype) or NULL
 *   dt_bias:  [dim]                     bias added to dt (fp32) or NULL
 *   out:      [batch, dim, seqlen]      output (dtype)
 *
 * Per (batch, channel) the fp32 recurrence over t = 0..seqlen-1 is:
 *   dt'_t   = dt[t] + dt_bias            (dt_bias only if provided)
 *   dt'_t   = softplus(dt'_t)            (only if delta_softplus != 0)
 *   dA_t,n  = exp(dt'_t * A[d,n])        per state n
 *   h_t,n   = dA_t,n * h_{t-1},n + dt'_t * B[t,n] * x[t]
 *   y_t     = sum_n( C[t,n] * h_t,n ) + D[d] * x[t]      (D term only if D != NULL)
 *   if z:  y_t *= silu(z[t])
 * with the initial state h_{-1} = 0. B and C are broadcast across all `dim`
 * channels (single-group selective scan).
 *
 * The softplus guard mirrors the reference: softplus(v) = v for v > 20 to avoid
 * exp overflow, else log1p(exp(v)).
 */
KS_API ks_status_t ks_selective_scan(void* out, const void* x, const void* dt,
                                     const void* A, const void* B, const void* C,
                                     const void* D, const void* z,
                                     const void* dt_bias, int delta_softplus,
                                     int batch, int dim, int seqlen, int dstate,
                                     ks_dtype_t dtype, ks_stream_t stream);

/*
 * Single-step selective-scan decode update.
 *
 * Advances the SSM state by exactly one position (seqlen == 1) and updates the
 * recurrent state in place. Used by autoregressive decoding where one token is
 * processed at a time.
 *
 *   state:   [batch, dim, dstate]   recurrent state h, READ AND WRITTEN (fp32)
 *   x:       [batch, dim]           input at this step      (dtype)
 *   dt:      [batch, dim]           timestep delta          (dtype)
 *   A:       [dim, dstate]          state transition (fp32)
 *   B:       [batch, dstate]        input projection (dtype, shared over dim)
 *   C:       [batch, dstate]        output projection (dtype, shared over dim)
 *   D:       [dim]                  skip (fp32) or NULL
 *   z:       [batch, dim]           SiLU gate (dtype) or NULL
 *   dt_bias: [dim]                  bias added to dt (fp32) or NULL
 *   out:     [batch, dim]           output (dtype)
 *
 * Computes, per (batch, channel), the same recurrence as ks_selective_scan for a
 * single step, reading h from `state` and writing the new h back to `state`:
 *   dt' = softplus(dt + dt_bias) if delta_softplus
 *   h_n = exp(dt' * A[d,n]) * h_n + dt' * B[n] * x
 *   y   = sum_n( C[n] * h_n ) + D[d] * x ;  if z: y *= silu(z)
 * `state` is kept in fp32 across decode steps for stability.
 */
KS_API ks_status_t ks_selective_scan_update(
    void* state, void* out, const void* x, const void* dt, const void* A,
    const void* B, const void* C, const void* D, const void* z,
    const void* dt_bias, int delta_softplus, int batch, int dim, int dstate,
    ks_dtype_t dtype, ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_SSM_H_ */
