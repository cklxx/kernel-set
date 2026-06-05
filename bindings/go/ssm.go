package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// CausalConv1d runs a depthwise causal 1-D convolution (+ optional SiLU).
// x and out are [batch, dim, seqlen]; weight is [dim, width] (dtype); bias is a
// per-channel [dim] fp32 device pointer (or nil to skip the bias add). When silu
// is true a SiLU activation (x * sigmoid(x)) is applied to the result.
// Wraps ks_causal_conv1d.
func CausalConv1d(out, x, weight, bias unsafe.Pointer, batch, dim, seqlen, width int,
	silu bool, dtype Dtype, stream Stream) error {
	st := Status(C.ks_causal_conv1d(out, x, weight, bias,
		C.int(batch), C.int(dim), C.int(seqlen), C.int(width),
		boolToCInt(silu), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_causal_conv1d")
}

// SelectiveScan runs the Mamba selective scan (forward) over a full sequence.
// x, dt, z and out are [batch, dim, seqlen]; B and C are [batch, dstate, seqlen]
// (shared over dim). A is [dim, dstate], D and dtBias are [dim] — all three are
// fp32 device pointers; D, z and dtBias may be nil. When deltaSoftplus is true,
// softplus is applied to dt (+ dtBias). Wraps ks_selective_scan.
func SelectiveScan(out, x, dt, a, b, c, d, z, dtBias unsafe.Pointer,
	deltaSoftplus bool, batch, dim, seqlen, dstate int, dtype Dtype, stream Stream) error {
	st := Status(C.ks_selective_scan(out, x, dt, a, b, c, d, z, dtBias,
		boolToCInt(deltaSoftplus),
		C.int(batch), C.int(dim), C.int(seqlen), C.int(dstate),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_selective_scan")
}

// SelectiveScanUpdate advances the SSM state by exactly one decode step,
// updating state in place. state is [batch, dim, dstate] (fp32, read and
// written); x, dt, z and out are [batch, dim]; B and C are [batch, dstate]
// (shared over dim). A is [dim, dstate], D and dtBias are [dim] — all three are
// fp32 device pointers; D, z and dtBias may be nil. When deltaSoftplus is true,
// softplus is applied to dt (+ dtBias). Wraps ks_selective_scan_update.
func SelectiveScanUpdate(state, out, x, dt, a, b, c, d, z, dtBias unsafe.Pointer,
	deltaSoftplus bool, batch, dim, dstate int, dtype Dtype, stream Stream) error {
	st := Status(C.ks_selective_scan_update(state, out, x, dt, a, b, c, d, z, dtBias,
		boolToCInt(deltaSoftplus),
		C.int(batch), C.int(dim), C.int(dstate),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_selective_scan_update")
}
