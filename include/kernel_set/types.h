/*
 * kernel-set — core ABI types shared by every kernel category.
 *
 * This header is pure C and intentionally free of any CUDA/HIP includes so it
 * can be parsed by every language binding (ctypes, cgo, bindgen, node-ffi)
 * without a GPU toolchain present. Device pointers are passed as `void*` and
 * streams as `ks_stream_t` (an opaque `void*` that is really a cudaStream_t /
 * hipStream_t under the hood).
 */
#ifndef KERNEL_SET_TYPES_H_
#define KERNEL_SET_TYPES_H_

#include <stddef.h>
#include <stdint.h>

#include "kernel_set/export.h"

KS_BEGIN_EXTERN_C

/* Opaque GPU stream handle (cudaStream_t / hipStream_t). NULL == default. */
typedef void* ks_stream_t;

/* Status / error codes returned by every public entry point. */
typedef enum ks_status_t {
  KS_SUCCESS = 0,
  KS_ERROR_INVALID_ARGUMENT = 1,   /* a NULL pointer or out-of-range value     */
  KS_ERROR_UNSUPPORTED_DTYPE = 2,  /* dtype not implemented for this kernel    */
  KS_ERROR_UNSUPPORTED_SHAPE = 3,  /* shape/alignment not supported            */
  KS_ERROR_CUDA = 4,               /* underlying CUDA/HIP runtime failure      */
  KS_ERROR_NOT_IMPLEMENTED = 5,    /* declared but no backend yet              */
  KS_ERROR_OUT_OF_MEMORY = 6,      /* device or workspace allocation failed    */
  KS_ERROR_ARCH_UNSUPPORTED = 7,   /* GPU SM/arch lacks a required feature     */
  KS_ERROR_INTERNAL = 8,           /* invariant violated; please file a bug    */
} ks_status_t;

/* Numeric element types understood across the ABI. */
typedef enum ks_dtype_t {
  KS_DTYPE_F32 = 0,     /* IEEE float32                                        */
  KS_DTYPE_F16 = 1,     /* IEEE float16 (half)                                 */
  KS_DTYPE_BF16 = 2,    /* bfloat16                                            */
  KS_DTYPE_F8E4M3 = 3,  /* FP8 e4m3 (OCP)                                      */
  KS_DTYPE_F8E5M2 = 4,  /* FP8 e5m2 (OCP)                                      */
  KS_DTYPE_F64 = 5,     /* IEEE float64                                        */
  KS_DTYPE_I64 = 6,     /* int64 (token ids, indices)                         */
  KS_DTYPE_I32 = 7,     /* int32                                              */
  KS_DTYPE_I8 = 8,      /* int8 (quantized weights/acts)                      */
  KS_DTYPE_U8 = 9,      /* uint8                                             */
  KS_DTYPE_I4 = 10,     /* packed int4 (two per byte)                        */
} ks_dtype_t;

/* Activation functions referenced by fused epilogues. */
typedef enum ks_activation_t {
  KS_ACT_NONE = 0,
  KS_ACT_RELU = 1,
  KS_ACT_GELU = 2,       /* exact (erf) GELU      */
  KS_ACT_GELU_TANH = 3,  /* tanh approximation    */
  KS_ACT_SILU = 4,       /* x * sigmoid(x)        */
} ks_activation_t;

/* Quantization granularity for quant.h / gemm.h. */
typedef enum ks_quant_mode_t {
  KS_QUANT_PER_TENSOR = 0,
  KS_QUANT_PER_TOKEN = 1,
  KS_QUANT_PER_CHANNEL = 2,
  KS_QUANT_GROUPWISE = 3,
} ks_quant_mode_t;

KS_END_EXTERN_C

#endif /* KERNEL_SET_TYPES_H_ */
