package kernelset

/*
#include "kernel_set/permute.h"
*/
import "C"

import "unsafe"

// Transpose2D transposes a 2D tensor [M, K] -> [K, M] (fp16/bf16/fp32).
func Transpose2D(out, in unsafe.Pointer, M, K int, dtype Dtype, stream Stream) error {
	st := Status(C.ks_transpose_2d(out, in, C.int(M), C.int(K), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_transpose_2d")
}

// NCHWToNHWC permutes [N, C, H, W] -> [N, H, W, C].
func NCHWToNHWC(out, in unsafe.Pointer, N, C, H, W int, dtype Dtype, stream Stream) error {
	st := Status(C.ks_nchw_to_nhwc(out, in, C.int(N), C.int(C), C.int(H), C.int(W), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_nchw_to_nhwc")
}

// NHWCToNCHW permutes [N, H, W, C] -> [N, C, H, W].
func NHWCToNCHW(out, in unsafe.Pointer, N, H, W, C int, dtype Dtype, stream Stream) error {
	st := Status(C.ks_nhwc_to_nchw(out, in, C.int(N), C.int(H), C.int(W), C.int(C), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_nhwc_to_nchw")
}

// UpsampleNearest2x performs nearest-neighbor 2x upsampling.
// in: [N, C, H, W], out: [N, C, 2H, 2W].
func UpsampleNearest2x(out, in unsafe.Pointer, N, C, H, W int, dtype Dtype, stream Stream) error {
	st := Status(C.ks_upsample_nearest_2x(out, in, C.int(N), C.int(C), C.int(H), C.int(W), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_upsample_nearest_2x")
}