package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// GatedDeltaRule runs the gated delta-rule linear attention (gated DeltaNet /
// Kimi-Delta) recurrent fallback. q,k are [B,T,H,K]; v,out are [B,T,H,V];
// beta is [B,T,H]; g is [B,T,H] (gIsVector=0) or [B,T,H,K] (gIsVector=1).
// Wraps ks_gated_delta_rule.
func GatedDeltaRule(out, q, k, v, g, beta unsafe.Pointer,
	batch, seqlen, heads, kDim, vDim int, gIsVector, useQKL2Norm int,
	scale float32, dtype Dtype, stream Stream) error {
	st := Status(C.ks_gated_delta_rule(out, q, k, v, g, beta,
		C.int64_t(batch), C.int64_t(seqlen), C.int64_t(heads), C.int64_t(kDim),
		C.int64_t(vDim), C.int(gIsVector), C.int(useQKL2Norm), C.float(scale),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_gated_delta_rule")
}

// GatedLinearAttn runs gated linear attention (GLA / simple-GLA / lightning).
// gateMode: 0 = data-dependent diagonal g [B,T,H,K]; 1 = scalar g [B,T,H];
// 2 = fixed per-head slope headDecay [H] (fp32 device ptr). q,k [B,T,H,K];
// v,out [B,T,H,V]. Wraps ks_gated_linear_attn.
func GatedLinearAttn(out, q, k, v, g, headDecay unsafe.Pointer,
	batch, seqlen, heads, kDim, vDim, gateMode int, scale float32,
	dtype Dtype, stream Stream) error {
	st := Status(C.ks_gated_linear_attn(out, q, k, v, g,
		(*C.float)(headDecay), C.int64_t(batch), C.int64_t(seqlen),
		C.int64_t(heads), C.int64_t(kDim), C.int64_t(vDim), C.int(gateMode),
		C.float(scale), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_gated_linear_attn")
}

// RwkvWkv7 runs the RWKV-7 WKV (free DPLR generalized delta rule) recurrent
// fallback. r,w,k,a,b are [B,T,H,K]; v,out are [B,T,H,V]. Wraps ks_rwkv_wkv7.
func RwkvWkv7(out, r, w, k, v, a, b unsafe.Pointer,
	batch, seqlen, heads, kDim, vDim int, scale float32,
	dtype Dtype, stream Stream) error {
	st := Status(C.ks_rwkv_wkv7(out, r, w, k, v, a, b,
		C.int64_t(batch), C.int64_t(seqlen), C.int64_t(heads), C.int64_t(kDim),
		C.int64_t(vDim), C.float(scale), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_rwkv_wkv7")
}
