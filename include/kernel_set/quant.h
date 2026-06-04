/*
 * kernel-set — quantization / dequantization kernels.
 *
 * Scales are fp32. "Dynamic" quantizers compute the scale from the data and
 * write it out; "static" quantizers take a precomputed scale.
 */
#ifndef KERNEL_SET_QUANT_H_
#define KERNEL_SET_QUANT_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* Dynamic FP8 quantization. in: [rows, cols] of `in_dtype` ->
 *   out: [rows, cols] FP8 (e4m3 or e5m2 per `fp8_dtype`)
 *   scale: per-tensor [1] or per-token [rows] (by `mode`). */
KS_API ks_status_t ks_quantize_fp8(void* out, float* scale, const void* input,
                                   int64_t rows, int64_t cols,
                                   ks_dtype_t in_dtype, ks_dtype_t fp8_dtype,
                                   ks_quant_mode_t mode, ks_stream_t stream);

KS_API ks_status_t ks_dequantize_fp8(void* out, const void* input,
                                     const float* scale, int64_t rows,
                                     int64_t cols, ks_dtype_t out_dtype,
                                     ks_dtype_t fp8_dtype, ks_quant_mode_t mode,
                                     ks_stream_t stream);

/* Dynamic INT8 quantization (symmetric). out int8 + scale (per-token/tensor). */
KS_API ks_status_t ks_quantize_int8(void* out, float* scale, const void* input,
                                    int64_t rows, int64_t cols,
                                    ks_dtype_t in_dtype, ks_quant_mode_t mode,
                                    ks_stream_t stream);

KS_API ks_status_t ks_dequantize_int8(void* out, const void* input,
                                      const float* scale, int64_t rows,
                                      int64_t cols, ks_dtype_t out_dtype,
                                      ks_quant_mode_t mode, ks_stream_t stream);

/* Dequantize group-wise INT4 weights (AWQ/GPTQ): packed [K/8, N] int32 (8 per
 * word) -> [K, N] of `out_dtype`, using scales/zeros [K/group_size, N]. */
KS_API ks_status_t ks_dequantize_int4(void* out, const void* qweight_packed,
                                      const void* scales, const void* zeros,
                                      int64_t k, int64_t n, int group_size,
                                      ks_dtype_t out_dtype, ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_QUANT_H_ */
