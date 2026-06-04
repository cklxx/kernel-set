package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// SiLU computes out = silu(x) = x * sigmoid(x) over n contiguous elements.
// Wraps ks_silu.
func SiLU(out, input unsafe.Pointer, n int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_silu(out, input, C.int64_t(n), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_silu")
}

// GELU computes out = gelu(x) over n elements. tanhApprox selects the tanh
// approximation. Wraps ks_gelu.
func GELU(out, input unsafe.Pointer, n int64, tanhApprox bool, dtype Dtype, stream Stream) error {
	st := Status(C.ks_gelu(out, input, C.int64_t(n), boolToCInt(tanhApprox),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_gelu")
}

// ReLU computes out = relu(x) over n elements. Wraps ks_relu.
func ReLU(out, input unsafe.Pointer, n int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_relu(out, input, C.int64_t(n), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_relu")
}

// SwiGLU computes out = silu(gate) * up. gate and up are each [rows, inter];
// out is [rows, inter]. Wraps ks_swiglu.
func SwiGLU(out, gate, up unsafe.Pointer, rows, inter int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_swiglu(out, gate, up,
		C.int64_t(rows), C.int64_t(inter), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_swiglu")
}

// SwiGLUPacked computes out = silu(gate) * up where input is [rows, 2*inter]
// with gate = input[:, :inter] and up = input[:, inter:]. out is [rows, inter].
// Wraps ks_swiglu_packed.
func SwiGLUPacked(out, input unsafe.Pointer, rows, inter int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_swiglu_packed(out, input,
		C.int64_t(rows), C.int64_t(inter), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_swiglu_packed")
}

// GeGLU computes out = gelu(gate) * up. tanhApprox selects the approximation.
// Wraps ks_geglu.
func GeGLU(out, gate, up unsafe.Pointer, rows, inter int64, tanhApprox bool,
	dtype Dtype, stream Stream) error {
	st := Status(C.ks_geglu(out, gate, up,
		C.int64_t(rows), C.int64_t(inter), boolToCInt(tanhApprox),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_geglu")
}

// SwiGLUBackward computes grad_gate and grad_up from grad_out and the forward
// gate/up tensors. Wraps ks_swiglu_backward.
func SwiGLUBackward(gradGate, gradUp, gradOut, gate, up unsafe.Pointer,
	rows, inter int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_swiglu_backward(gradGate, gradUp, gradOut, gate, up,
		C.int64_t(rows), C.int64_t(inter), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_swiglu_backward")
}
