package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// QuantizeFP8 dynamically quantizes input [rows, cols] of inDtype to FP8
// (e4m3 or e5m2 per fp8Dtype). scale is a fp32 device pointer: per-tensor [1]
// or per-token [rows] depending on mode. Wraps ks_quantize_fp8.
func QuantizeFP8(out, scale, input unsafe.Pointer, rows, cols int64,
	inDtype, fp8Dtype Dtype, mode QuantMode, stream Stream) error {
	st := Status(C.ks_quantize_fp8(out, (*C.float)(scale), input,
		C.int64_t(rows), C.int64_t(cols),
		C.ks_dtype_t(inDtype), C.ks_dtype_t(fp8Dtype),
		C.ks_quant_mode_t(mode), stream.c()))
	return statusError(st, "ks_quantize_fp8")
}

// DequantizeFP8 dequantizes FP8 input [rows, cols] to outDtype using a fp32
// scale (per mode). Wraps ks_dequantize_fp8.
func DequantizeFP8(out, input, scale unsafe.Pointer, rows, cols int64,
	outDtype, fp8Dtype Dtype, mode QuantMode, stream Stream) error {
	st := Status(C.ks_dequantize_fp8(out, input, (*C.float)(scale),
		C.int64_t(rows), C.int64_t(cols),
		C.ks_dtype_t(outDtype), C.ks_dtype_t(fp8Dtype),
		C.ks_quant_mode_t(mode), stream.c()))
	return statusError(st, "ks_dequantize_fp8")
}

// QuantizeInt8 dynamically (symmetric) quantizes input [rows, cols] of inDtype
// to int8, writing a fp32 scale (per-token/tensor by mode). Wraps
// ks_quantize_int8.
func QuantizeInt8(out, scale, input unsafe.Pointer, rows, cols int64,
	inDtype Dtype, mode QuantMode, stream Stream) error {
	st := Status(C.ks_quantize_int8(out, (*C.float)(scale), input,
		C.int64_t(rows), C.int64_t(cols),
		C.ks_dtype_t(inDtype), C.ks_quant_mode_t(mode), stream.c()))
	return statusError(st, "ks_quantize_int8")
}

// DequantizeInt8 dequantizes int8 input [rows, cols] to outDtype using a fp32
// scale (per mode). Wraps ks_dequantize_int8.
func DequantizeInt8(out, input, scale unsafe.Pointer, rows, cols int64,
	outDtype Dtype, mode QuantMode, stream Stream) error {
	st := Status(C.ks_dequantize_int8(out, input, (*C.float)(scale),
		C.int64_t(rows), C.int64_t(cols),
		C.ks_dtype_t(outDtype), C.ks_quant_mode_t(mode), stream.c()))
	return statusError(st, "ks_dequantize_int8")
}

// DequantizeInt4 dequantizes group-wise INT4 weights (AWQ/GPTQ): packed
// [K/8, N] int32 (8 per word) -> [K, N] of outDtype, using scales/zeros
// [K/groupSize, N]. Wraps ks_dequantize_int4.
func DequantizeInt4(out, qweightPacked, scales, zeros unsafe.Pointer,
	k, n int64, groupSize int, outDtype Dtype, stream Stream) error {
	st := Status(C.ks_dequantize_int4(out, qweightPacked, scales, zeros,
		C.int64_t(k), C.int64_t(n), C.int(groupSize),
		C.ks_dtype_t(outDtype), stream.c()))
	return statusError(st, "ks_dequantize_int4")
}
