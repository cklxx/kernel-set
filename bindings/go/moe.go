package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// MoEGateSoftmaxTopK runs softmax over experts then top-k select.
//
//	logits:     [num_tokens, num_experts] of dtype
//	outWeights: [num_tokens, top_k] fp32 routing weights (renormalized if set)
//	outIndices: [num_tokens, top_k] int32 selected expert ids
//
// All pointers are device pointers. Wraps ks_moe_gate_softmax_topk.
func MoEGateSoftmaxTopK(outWeights, outIndices, logits unsafe.Pointer,
	numTokens int64, numExperts, topK int, renormalize bool,
	dtype Dtype, stream Stream) error {
	st := Status(C.ks_moe_gate_softmax_topk(
		(*C.float)(outWeights), (*C.int32_t)(outIndices), logits,
		C.int64_t(numTokens), C.int(numExperts), C.int(topK),
		boolToCInt(renormalize), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_moe_gate_softmax_topk")
}

// MoEGateSigmoidGroupTopK runs sigmoid + group-limited top-k gating
// (DeepSeek-V3 style). correctionBias may be nil. nGroup, topkGroup and
// routedScalingFactor follow the DeepSeek config. Wraps
// ks_moe_gate_sigmoid_group_topk.
func MoEGateSigmoidGroupTopK(outWeights, outIndices, logits, correctionBias unsafe.Pointer,
	numTokens int64, numExperts, nGroup, topkGroup, topK int, renormalize bool,
	routedScalingFactor float32, dtype Dtype, stream Stream) error {
	st := Status(C.ks_moe_gate_sigmoid_group_topk(
		(*C.float)(outWeights), (*C.int32_t)(outIndices), logits, correctionBias,
		C.int64_t(numTokens), C.int(numExperts), C.int(nGroup), C.int(topkGroup),
		C.int(topK), boolToCInt(renormalize), C.float(routedScalingFactor),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_moe_gate_sigmoid_group_topk")
}

// MoEComputePermutation builds the permutation that groups tokens by expert.
//
//	topkIndices:    [num_tokens, top_k] int32 (device ptr)
//	sortedTokenIDs: [num_tokens*top_k] int32 (out, device ptr)
//	expertOffsets:  [num_experts+1] int32 CSR-style group boundaries (out)
//
// Wraps ks_moe_compute_permutation.
func MoEComputePermutation(sortedTokenIDs, expertOffsets, topkIndices unsafe.Pointer,
	numTokens int64, numExperts, topK int, stream Stream) error {
	st := Status(C.ks_moe_compute_permutation(
		(*C.int32_t)(sortedTokenIDs), (*C.int32_t)(expertOffsets),
		(*C.int32_t)(topkIndices),
		C.int64_t(numTokens), C.int(numExperts), C.int(topK), stream.c()))
	return statusError(st, "ks_moe_compute_permutation")
}

// MoEPermute gathers rows of input [num_tokens, hidden] into permuted
// [num_tokens*top_k, hidden] following sortedTokenIDs (row = id / top_k). Wraps
// ks_moe_permute.
func MoEPermute(permuted, input, sortedTokenIDs unsafe.Pointer,
	numTokens int64, topK int, hidden int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_moe_permute(permuted, input, (*C.int32_t)(sortedTokenIDs),
		C.int64_t(numTokens), C.int(topK), C.int64_t(hidden),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_moe_permute")
}

// MoEUnpermute scatter-adds expert outputs back, weighted by routing weights:
//
//	out[token] = sum_k weight[token,k] * permuted[pos(token,k)]
//
// routingWeights is a fp32 device pointer. Wraps ks_moe_unpermute.
func MoEUnpermute(out, permuted, sortedTokenIDs, routingWeights unsafe.Pointer,
	numTokens int64, topK int, hidden int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_moe_unpermute(out, permuted, (*C.int32_t)(sortedTokenIDs),
		(*C.float)(routingWeights),
		C.int64_t(numTokens), C.int(topK), C.int64_t(hidden),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_moe_unpermute")
}

// MoEGroupedGEMM runs grouped GEMM: for each expert e,
// C[off_e:off_{e+1}] = A[off_e:off_{e+1}] @ B_e.
//
//	a:             [total_rows, k] permuted tokens
//	b:             [num_experts, k, n] per-expert weights, contiguous
//	expertOffsets: [num_experts+1] int32 row boundaries (device ptr)
//	c:             [total_rows, n]
//
// Wraps ks_moe_grouped_gemm.
func MoEGroupedGEMM(c, a, b, expertOffsets unsafe.Pointer,
	numExperts int, totalRows, n, k int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_moe_grouped_gemm(c, a, b, (*C.int32_t)(expertOffsets),
		C.int(numExperts), C.int64_t(totalRows), C.int64_t(n), C.int64_t(k),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_moe_grouped_gemm")
}
