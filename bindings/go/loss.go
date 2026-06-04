package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// CrossEntropy runs fused cross-entropy forward+backward in one pass.
//
//	logits:     [num_tokens, vocab] of dtype (overwritten with grad if
//	            gradLogits == logits; otherwise grad written to gradLogits)
//	targets:    [num_tokens] int32/int64 class ids (targetsI64 selects int64);
//	            ignoreIndex is masked out
//	losses:     [num_tokens] fp32 per-token loss (device ptr)
//
// labelSmoothing is in [0,1). Reduction is left to the caller. Wraps
// ks_cross_entropy.
func CrossEntropy(losses, gradLogits, logits, targets unsafe.Pointer, targetsI64 bool,
	numTokens, vocab, ignoreIndex int64, labelSmoothing float32,
	dtype Dtype, stream Stream) error {
	st := Status(C.ks_cross_entropy((*C.float)(losses), gradLogits, logits, targets,
		boolToCInt(targetsI64), C.int64_t(numTokens), C.int64_t(vocab),
		C.int64_t(ignoreIndex), C.float(labelSmoothing),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_cross_entropy")
}

// FusedLinearCrossEntropy computes CE from hidden states and the LM head weight
// in chunks WITHOUT materializing the full [num_tokens, vocab] logits, writing
// gradHidden / gradWeightFP32.
//
//	hidden:          [num_tokens, hidden_dim]
//	weight:          [vocab, hidden_dim]
//	gradHidden:      [num_tokens, hidden_dim]
//	gradWeightFP32:  [vocab, hidden_dim] fp32
//	targets:         [num_tokens] int32/int64 (targetsI64 selects int64)
//	losses:          [num_tokens] fp32 (device ptr)
//
// chunkSize controls the token-dim chunking. Wraps ks_fused_linear_cross_entropy.
func FusedLinearCrossEntropy(losses, gradHidden, gradWeightFP32, hidden, weight, targets unsafe.Pointer,
	targetsI64 bool, numTokens, hiddenDim, vocab, ignoreIndex int64,
	labelSmoothing float32, chunkSize int, dtype Dtype, stream Stream) error {
	st := Status(C.ks_fused_linear_cross_entropy((*C.float)(losses),
		gradHidden, gradWeightFP32, hidden, weight, targets,
		boolToCInt(targetsI64), C.int64_t(numTokens), C.int64_t(hiddenDim),
		C.int64_t(vocab), C.int64_t(ignoreIndex), C.float(labelSmoothing),
		C.int(chunkSize), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_fused_linear_cross_entropy")
}
