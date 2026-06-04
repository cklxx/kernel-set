package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// FlashAttnVarlen runs variable-length packed FlashAttention-2 forward.
//
//	q:            [total_q, num_heads, head_dim]
//	k, v:         [total_kv, num_kv_heads, head_dim]
//	cuSeqlensQ/K: [batch+1] int32 prefix sums of sequence lengths (device ptrs)
//	out:          [total_q, num_heads, head_dim]
//	softmaxLSE:   [num_heads, total_q] fp32, may be nil if not training
//
// softmaxScale defaults to 1/sqrt(head_dim) when <= 0. Wraps ks_flash_attn_varlen.
func FlashAttnVarlen(out, softmaxLSE, q, k, v unsafe.Pointer,
	cuSeqlensQ, cuSeqlensK unsafe.Pointer, batch, maxSeqlenQ, maxSeqlenK,
	numHeads, numKVHeads, headDim int, softmaxScale float32, causal bool,
	dtype Dtype, stream Stream) error {
	st := Status(C.ks_flash_attn_varlen(out, softmaxLSE, q, k, v,
		(*C.int32_t)(cuSeqlensQ), (*C.int32_t)(cuSeqlensK),
		C.int(batch), C.int(maxSeqlenQ), C.int(maxSeqlenK),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim),
		C.float(softmaxScale), boolToCInt(causal),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_flash_attn_varlen")
}

// FlashAttn runs dense (non-varlen) FlashAttention-2 prefill for a uniform
// [batch, seqlen] shape.
//
//	q:    [batch, seqlen_q, num_heads, head_dim]
//	k, v: [batch, seqlen_k, num_kv_heads, head_dim]
//
// softmaxLSE may be nil. softmaxScale <= 0 defaults to 1/sqrt(head_dim).
// Wraps ks_flash_attn.
func FlashAttn(out, softmaxLSE, q, k, v unsafe.Pointer,
	batch, seqlenQ, seqlenK, numHeads, numKVHeads, headDim int,
	softmaxScale float32, causal bool, dtype Dtype, stream Stream) error {
	st := Status(C.ks_flash_attn(out, softmaxLSE, q, k, v,
		C.int(batch), C.int(seqlenQ), C.int(seqlenK),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim),
		C.float(softmaxScale), boolToCInt(causal),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_flash_attn")
}

// PagedAttnDecode runs paged KV-cache decode (one query position per sequence;
// FlashDecoding).
//
//	q:               [num_seqs, num_heads, head_dim]
//	kCache, vCache:  [num_blocks, num_kv_heads, block_size, head_dim]
//	blockTables:     [num_seqs, max_blocks_per_seq] int32 (device ptr)
//	seqLens:         [num_seqs] int32 (device ptr)
//	out:             [num_seqs, num_heads, head_dim]
//
// Wraps ks_paged_attn_decode.
func PagedAttnDecode(out, q, kCache, vCache unsafe.Pointer,
	blockTables, seqLens unsafe.Pointer, numSeqs, numHeads, numKVHeads, headDim,
	blockSize, maxBlocksPerSeq int, softmaxScale float32, dtype Dtype, stream Stream) error {
	st := Status(C.ks_paged_attn_decode(out, q, kCache, vCache,
		(*C.int32_t)(blockTables), (*C.int32_t)(seqLens),
		C.int(numSeqs), C.int(numHeads), C.int(numKVHeads), C.int(headDim),
		C.int(blockSize), C.int(maxBlocksPerSeq), C.float(softmaxScale),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_paged_attn_decode")
}

// ReshapeAndCache writes new K/V into a paged cache at the given slots (prefill
// & decode append).
//
//	key, value:  [num_tokens, num_kv_heads, head_dim]
//	slotMapping: [num_tokens] int32 -> flat slot = block_id*block_size + offset
//
// Wraps ks_reshape_and_cache.
func ReshapeAndCache(kCache, vCache, key, value unsafe.Pointer,
	slotMapping unsafe.Pointer, numTokens, numKVHeads, headDim, blockSize int,
	dtype Dtype, stream Stream) error {
	st := Status(C.ks_reshape_and_cache(kCache, vCache, key, value,
		(*C.int32_t)(slotMapping),
		C.int(numTokens), C.int(numKVHeads), C.int(headDim), C.int(blockSize),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_reshape_and_cache")
}

// MLADecode runs DeepSeek Multi-head Latent Attention decode over a compressed
// KV cache.
//
//	qNope:   [num_seqs, num_heads, kv_lora_rank]
//	qPE:     [num_seqs, num_heads, rope_dim]
//	kvCache: [num_blocks, block_size, kv_lora_rank + rope_dim]
//
// Wraps ks_mla_decode.
func MLADecode(out, qNope, qPE, kvCache unsafe.Pointer,
	blockTables, seqLens unsafe.Pointer, numSeqs, numHeads, kvLoraRank, ropeDim,
	blockSize, maxBlocksPerSeq int, softmaxScale float32, dtype Dtype, stream Stream) error {
	st := Status(C.ks_mla_decode(out, qNope, qPE, kvCache,
		(*C.int32_t)(blockTables), (*C.int32_t)(seqLens),
		C.int(numSeqs), C.int(numHeads), C.int(kvLoraRank), C.int(ropeDim),
		C.int(blockSize), C.int(maxBlocksPerSeq), C.float(softmaxScale),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_mla_decode")
}

// FlashAttnBackward runs FlashAttention backward (training). Requires the
// forward out and softmaxLSE. grad_q/grad_k/grad_v match q/k/v shapes. Wraps
// ks_flash_attn_backward.
func FlashAttnBackward(gradQ, gradK, gradV, gradOut, q, k, v, out, softmaxLSE unsafe.Pointer,
	batch, seqlenQ, seqlenK, numHeads, numKVHeads, headDim int,
	softmaxScale float32, causal bool, dtype Dtype, stream Stream) error {
	st := Status(C.ks_flash_attn_backward(gradQ, gradK, gradV, gradOut,
		q, k, v, out, softmaxLSE,
		C.int(batch), C.int(seqlenQ), C.int(seqlenK),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim),
		C.float(softmaxScale), boolToCInt(causal),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_flash_attn_backward")
}
