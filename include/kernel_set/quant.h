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

/* Per-token-GROUP dynamic FP8 quantization (1 x group_size tiles): for each row
 * and each contiguous group of `group_size` columns, compute the group absmax,
 * derive an fp8 scale, and emit fp8. This is the activation format the DeepSeek
 * blockwise FP8 GEMM (ks_gemm_fp8_blockwise / DeepGEMM) consumes (group_size
 * typically 128).
 *   out:   [rows, cols] FP8 (e4m3 or e5m2 per `fp8_dtype`)
 *   scale: [rows, ceil(cols/group_size)] fp32 (one scale per (row, col-group)). */
KS_API ks_status_t ks_quantize_fp8_group(void* out, float* scale,
                                         const void* input, int64_t rows,
                                         int64_t cols, int group_size,
                                         ks_dtype_t in_dtype,
                                         ks_dtype_t fp8_dtype,
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

/* Repack group-wise INT4 weights (GPTQ/AWQ -> Marlin layout).
 * This is a correctness stub / fallback that allows loading INT4 checkpoints
 * without Marlin being installed. (Actually returns KS_ERROR_NOT_IMPLEMENTED
 * if called directly for now, acting as a terminal dead-end stub). */
KS_API ks_status_t ks_repack_int4(void* out_packed, const void* qweight,
                                  const void* perm, int64_t size_k, int64_t size_n,
                                  int num_bits, ks_stream_t stream);

/* NVFP4 quantization (portable correctness fallback).
 *   out_fp4: packed 2/byte e2m1 [rows, cols/2]
 *   out_scales: fp8 e4m3 1x16 block scales [rows, cols/16]
 *   global_scale: per-tensor fp32 input global scale
 * Computes a per-1x16-block absmax scale, stores it as e4m3, and encodes each
 * element to e2m1 (round-trip consistent with the ks_gemm_nvfp4 decode path).
 * `cols` must be a multiple of 16. Requires FP8 (e4m3) support at compile time;
 * otherwise returns KS_ERROR_ARCH_UNSUPPORTED. */
KS_API ks_status_t ks_quantize_nvfp4(void* out_fp4, void* out_scales,
                                     const void* input, float global_scale,
                                     int64_t rows, int64_t cols,
                                     ks_dtype_t in_dtype, ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_QUANT_H_ */
