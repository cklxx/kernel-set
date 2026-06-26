package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// Conv2D runs a 2D convolution (NCHW layout).
//
//	input:  [N, C, H, W]
//	weight: [K, C/groups, R, S]
//	bias:   [K] (may be nil)
//	out:    [N, K, OH, OW]
//
// Wraps ks_conv2d.
func Conv2D(out, input, weight, bias unsafe.Pointer,
	n, c, h, w, k, r, s int,
	strideH, strideW, paddingH, paddingW, dilationH, dilationW, groups int,
	dtype Dtype, stream Stream) error {
	st := Status(C.ks_conv2d(out, input, weight, bias,
		C.int(n), C.int(c), C.int(h), C.int(w),
		C.int(k), C.int(r), C.int(s),
		C.int(strideH), C.int(strideW),
		C.int(paddingH), C.int(paddingW),
		C.int(dilationH), C.int(dilationW),
		C.int(groups),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_conv2d")
}

// GroupNorm computes Group Normalization (NCHW layout).
//
//	input:      [N, C, H, W]
//	weight:     [C] (may be nil, defaults to 1)
//	bias:       [C] (may be nil, defaults to 0)
//	out:        [N, C, H, W]
//	hw:         H * W (flattened spatial dims)
//	numGroups:  number of groups (must divide C evenly)
//
// Wraps ks_group_norm.
func GroupNorm(out, input, weight, bias unsafe.Pointer,
	n, c, hw, numGroups int, eps float32, dtype Dtype, stream Stream) error {
	st := Status(C.ks_group_norm(out, input, weight, bias,
		C.int(n), C.int(c), C.int(hw), C.int(numGroups),
		C.float(eps), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_group_norm")
}