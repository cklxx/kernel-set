package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// Softmax computes out = softmax(input / temperature) along the last dim.
// temperature <= 0 is treated as 1. Wraps ks_softmax.
func Softmax(out, input unsafe.Pointer, rows, cols int64, temperature float32,
	dtype Dtype, stream Stream) error {
	st := Status(C.ks_softmax(out, input, C.int64_t(rows), C.int64_t(cols),
		C.float(temperature), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_softmax")
}

// LogSoftmax computes log_softmax along the last dim (fp32 out recommended for
// stability). Wraps ks_log_softmax.
func LogSoftmax(out, input unsafe.Pointer, rows, cols int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_log_softmax(out, input, C.int64_t(rows), C.int64_t(cols),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_log_softmax")
}

// Argmax computes greedy decoding: outTokens[s] = argmax_v logits[s, v].
// outTokens is an int32 device pointer [num_seqs]. Wraps ks_argmax.
func Argmax(outTokens, logits unsafe.Pointer, numSeqs, vocabSize int64,
	dtype Dtype, stream Stream) error {
	st := Status(C.ks_argmax((*C.int32_t)(outTokens), logits,
		C.int64_t(numSeqs), C.int64_t(vocabSize), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_argmax")
}

// Sample runs combined temperature + top-k + top-p sampling.
//
//	temperatures: [num_seqs] fp32 device ptr or nil (=> 1.0)
//	topKs:        [num_seqs] int32 device ptr or nil (<=0 => disabled)
//	topPs:        [num_seqs] fp32 device ptr or nil (<=0 => disabled)
//	outTokens:    [num_seqs] int32 device ptr
//	outProbs:     [num_seqs] fp32 sampled-token probability or nil
//
// Randomness uses a counter-based RNG seeded by (seed, philoxOffset). Wraps
// ks_sample.
func Sample(outTokens, outProbs, logits unsafe.Pointer,
	temperatures, topKs, topPs unsafe.Pointer, numSeqs, vocabSize int64,
	seed, philoxOffset uint64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_sample((*C.int32_t)(outTokens), (*C.float)(outProbs), logits,
		(*C.float)(temperatures), (*C.int32_t)(topKs), (*C.float)(topPs),
		C.int64_t(numSeqs), C.int64_t(vocabSize),
		C.uint64_t(seed), C.uint64_t(philoxOffset),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_sample")
}
