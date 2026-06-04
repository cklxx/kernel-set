/*
 * kernel-set — general matrix multiply and fused/quantized variants.
 *
 * Row-major convention. For op(A) [M,K] x op(B) [K,N] = C [M,N]:
 *   - `trans_a`/`trans_b` transpose the respective operand.
 *   - leading dimensions are the row stride (elements) of the *stored* matrix.
 * FP16/BF16 paths target tensor cores (mma) with fp32 accumulation; TF32 is
 * used for the F32 path on Ampere+ when available.
 */
#ifndef KERNEL_SET_GEMM_H_
#define KERNEL_SET_GEMM_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* C = alpha * op(A) @ op(B) + beta * C. A/B/C all `dtype`; acc in fp32. */
KS_API ks_status_t ks_gemm(void* c, const void* a, const void* b, int64_t m,
                           int64_t n, int64_t k, int trans_a, int trans_b,
                           int64_t lda, int64_t ldb, int64_t ldc, float alpha,
                           float beta, ks_dtype_t dtype, ks_stream_t stream);

/* D = act(alpha * A @ B + bias). `bias` is [N] (broadcast over rows) or NULL.
 * Fuses the linear epilogue used by MLP/QKV projections. */
KS_API ks_status_t ks_gemm_bias_act(void* d, const void* a, const void* b,
                                    const void* bias, int64_t m, int64_t n,
                                    int64_t k, float alpha,
                                    ks_activation_t act, ks_dtype_t dtype,
                                    ks_stream_t stream);

/* Batched / strided-batched GEMM. Each batch b uses base + b*stride_{a,b,c}. */
KS_API ks_status_t ks_gemm_batched(void* c, const void* a, const void* b,
                                   int64_t batch, int64_t m, int64_t n,
                                   int64_t k, int trans_a, int trans_b,
                                   int64_t stride_a, int64_t stride_b,
                                   int64_t stride_c, float alpha, float beta,
                                   ks_dtype_t dtype, ks_stream_t stream);

/* W8A8 GEMM: int8 A [M,K] and int8 B [K,N] -> `out_dtype` C [M,N].
 * Dequant: C = (A_scale (x) B_scale) * (A_i8 @ B_i8) + bias.
 *   a_scale: per-token  [M]  (or [1] for per-tensor)
 *   b_scale: per-channel [N] (or [1] for per-tensor)
 *   bias:    [N] in out_dtype or NULL */
KS_API ks_status_t ks_gemm_w8a8(void* c, const void* a_i8, const void* b_i8,
                                const float* a_scale, const float* b_scale,
                                const void* bias, int64_t m, int64_t n,
                                int64_t k, ks_quant_mode_t a_mode,
                                ks_quant_mode_t b_mode, ks_dtype_t out_dtype,
                                ks_stream_t stream);

/* W4A16 GEMM: fp16/bf16 activations A [M,K] x int4 weights B [K,N] with
 * group-wise scales/zeros (AWQ/GPTQ layout). group_size columns of K share a
 * scale.  b_packed holds two int4 per byte; scales/zeros are [K/group_size, N].
 */
KS_API ks_status_t ks_gemm_w4a16(void* c, const void* a, const void* b_packed,
                                 const void* scales, const void* zeros,
                                 const void* bias, int64_t m, int64_t n,
                                 int64_t k, int group_size, ks_dtype_t dtype,
                                 ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_GEMM_H_ */
