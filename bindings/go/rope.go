package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// RoPEInplace applies rotary position embedding in place to q and k.
//
//	q:       [num_tokens, num_q_heads, head_dim]
//	k:       [num_tokens, num_kv_heads, head_dim] (may be nil to skip)
//	cos/sin: [num_tokens, head_dim/2]
//
// interleaved selects the GPT-J convention (vs NeoX rotate-half). Wraps
// ks_rope_inplace.
func RoPEInplace(q, k, cos, sin unsafe.Pointer, numTokens int64,
	numQHeads, numKVHeads, headDim int, interleaved bool, dtype Dtype, stream Stream) error {
	st := Status(C.ks_rope_inplace(q, k, cos, sin,
		C.int64_t(numTokens), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		boolToCInt(interleaved), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_rope_inplace")
}

// RoPE applies rotary position embedding out of place, writing to qOut/kOut.
// Wraps ks_rope.
func RoPE(qOut, kOut, q, k, cos, sin unsafe.Pointer, numTokens int64,
	numQHeads, numKVHeads, headDim int, interleaved bool, dtype Dtype, stream Stream) error {
	st := Status(C.ks_rope(qOut, kOut, q, k, cos, sin,
		C.int64_t(numTokens), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		boolToCInt(interleaved), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_rope")
}

// RoPEGather applies RoPE in place using cos/sin caches [max_pos, head_dim/2]
// indexed by positions (int32, [num_tokens] device ptr). Wraps ks_rope_gather.
func RoPEGather(q, k, cosCache, sinCache, positions unsafe.Pointer, numTokens int64,
	numQHeads, numKVHeads, headDim int, interleaved bool, dtype Dtype, stream Stream) error {
	st := Status(C.ks_rope_gather(q, k, cosCache, sinCache, (*C.int32_t)(positions),
		C.int64_t(numTokens), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		boolToCInt(interleaved), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_rope_gather")
}

// RoPEBackward rotates gradients by the conjugate rotation (sin -> -sin). Wraps
// ks_rope_backward.
func RoPEBackward(gradQ, gradK, cos, sin unsafe.Pointer, numTokens int64,
	numQHeads, numKVHeads, headDim int, interleaved bool, dtype Dtype, stream Stream) error {
	st := Status(C.ks_rope_backward(gradQ, gradK, cos, sin,
		C.int64_t(numTokens), C.int(numQHeads), C.int(numKVHeads), C.int(headDim),
		boolToCInt(interleaved), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_rope_backward")
}
