package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// EmbeddingLookup computes out[i, :] = table[indices[i], :].
//
//	table:   [vocab_size, embed_dim] of dtype
//	indices: [num_tokens] int32 (or int64 if indicesI64 is true)
//	out:     [num_tokens, embed_dim] of dtype
//
// Wraps ks_embedding_lookup.
func EmbeddingLookup(out, table, indices unsafe.Pointer, indicesI64 bool,
	numTokens, embedDim int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_embedding_lookup(out, table, indices, boolToCInt(indicesI64),
		C.int64_t(numTokens), C.int64_t(embedDim), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_embedding_lookup")
}

// EmbeddingBackward performs the atomic scatter-add:
// grad_table[indices[i], :] += grad_out[i, :]. gradTableFP32 is a fp32
// [vocab_size, embed_dim] device pointer. Wraps ks_embedding_backward.
func EmbeddingBackward(gradTableFP32, gradOut, indices unsafe.Pointer, indicesI64 bool,
	numTokens, embedDim int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_embedding_backward(gradTableFP32, gradOut, indices, boolToCInt(indicesI64),
		C.int64_t(numTokens), C.int64_t(embedDim), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_embedding_backward")
}
