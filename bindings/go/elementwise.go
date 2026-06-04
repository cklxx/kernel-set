package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// Add computes out = a + b over n contiguous elements. Wraps ks_add.
func Add(out, a, b unsafe.Pointer, n int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_add(out, a, b, C.int64_t(n), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_add")
}

// Mul computes out = a * b (Hadamard) over n elements. Wraps ks_mul.
func Mul(out, a, b unsafe.Pointer, n int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_mul(out, a, b, C.int64_t(n), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_mul")
}

// AddResidual performs in-place residual accumulation: residual <- residual + x.
// Wraps ks_add_residual.
func AddResidual(residual, x unsafe.Pointer, n int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_add_residual(residual, x, C.int64_t(n), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_add_residual")
}

// Scale computes out = x * scale (scalar) over n elements. Wraps ks_scale.
func Scale(out, x unsafe.Pointer, scale float32, n int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_scale(out, x, C.float(scale), C.int64_t(n), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_scale")
}

// Cast converts n elements from srcDtype to dstDtype (e.g. bf16 -> f32). Wraps
// ks_cast.
func Cast(out unsafe.Pointer, dstDtype Dtype, in unsafe.Pointer, srcDtype Dtype,
	n int64, stream Stream) error {
	st := Status(C.ks_cast(out, C.ks_dtype_t(dstDtype), in, C.ks_dtype_t(srcDtype),
		C.int64_t(n), stream.c()))
	return statusError(st, "ks_cast")
}

// AXPBY computes out = a*alpha + b*beta (fused) over n elements. Wraps ks_axpby.
func AXPBY(out, a unsafe.Pointer, alpha float32, b unsafe.Pointer, beta float32,
	n int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_axpby(out, a, C.float(alpha), b, C.float(beta),
		C.int64_t(n), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_axpby")
}
