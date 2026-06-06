package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// RMSNorm computes out = (x / rms(x)) * weight per row, with rms(x) =
// sqrt(mean(x^2) + eps). out, input and weight are device pointers; input is
// [rows, cols], weight is [cols]. Wraps ks_rms_norm.
func RMSNorm(out, input, weight unsafe.Pointer, rows, cols int64, eps float32,
	dtype Dtype, stream Stream) error {
	st := Status(C.ks_rms_norm(out, input, weight,
		C.int64_t(rows), C.int64_t(cols), C.float(eps),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_rms_norm")
}

// FusedRMSNormGated computes out = (x/rms(x))*weight*act(gate) — GatedDeltaNet /
// GLA output norm (gate applied after norm). gateAct: 0 = SiLU, 1 = sigmoid.
// out, input, weight, gate are device pointers [rows, cols] (weight [cols]).
// Wraps ks_fused_rmsnorm_gated.
func FusedRMSNormGated(out, input, weight, gate unsafe.Pointer, rows, cols int64,
	gateAct int, eps float32, dtype Dtype, stream Stream) error {
	st := Status(C.ks_fused_rmsnorm_gated(out, input, weight, gate,
		C.int64_t(rows), C.int64_t(cols), C.int(gateAct), C.float(eps),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_fused_rmsnorm_gated")
}

// RMSNormResidual fuses add + RMSNorm between transformer sub-layers:
//
//	x   = input + residual   (the new residual, written to residualOut)
//	out = rms_norm(x) * weight
//
// residualOut may alias residual for in-place accumulation. Wraps
// ks_rms_norm_residual.
func RMSNormResidual(out, residualOut, input, residual, weight unsafe.Pointer,
	rows, cols int64, eps float32, dtype Dtype, stream Stream) error {
	st := Status(C.ks_rms_norm_residual(out, residualOut, input, residual, weight,
		C.int64_t(rows), C.int64_t(cols), C.float(eps),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_rms_norm_residual")
}

// LayerNorm computes out = (x - mean) / sqrt(var + eps) * weight + bias per row.
// bias may be nil. Wraps ks_layer_norm.
func LayerNorm(out, input, weight, bias unsafe.Pointer, rows, cols int64,
	eps float32, dtype Dtype, stream Stream) error {
	st := Status(C.ks_layer_norm(out, input, weight, bias,
		C.int64_t(rows), C.int64_t(cols), C.float(eps),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_layer_norm")
}

// RMSNormBackward computes grad_input and grad_weight (accumulated in fp32; pass
// a [cols] fp32 buffer for gradWeightFP32) from grad_out and the forward inputs.
// Wraps ks_rms_norm_backward.
func RMSNormBackward(gradInput, gradWeightFP32, gradOut, input, weight unsafe.Pointer,
	rows, cols int64, eps float32, dtype Dtype, stream Stream) error {
	st := Status(C.ks_rms_norm_backward(gradInput, gradWeightFP32, gradOut, input, weight,
		C.int64_t(rows), C.int64_t(cols), C.float(eps),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_rms_norm_backward")
}

// LayerNormBackward computes grad_input, grad_weight and grad_bias (the latter
// two accumulated in fp32, [cols] each) from grad_out and the forward inputs.
// Wraps ks_layer_norm_backward.
func LayerNormBackward(gradInput, gradWeightFP32, gradBiasFP32, gradOut, input, weight unsafe.Pointer,
	rows, cols int64, eps float32, dtype Dtype, stream Stream) error {
	st := Status(C.ks_layer_norm_backward(gradInput, gradWeightFP32, gradBiasFP32,
		gradOut, input, weight,
		C.int64_t(rows), C.int64_t(cols), C.float(eps),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_layer_norm_backward")
}
