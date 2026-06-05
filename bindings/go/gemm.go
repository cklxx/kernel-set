package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// GEMM computes C = alpha * op(A) @ op(B) + beta * C in row-major layout.
// transA/transB transpose the respective operand; lda/ldb/ldc are the row
// strides (in elements) of the stored matrices. A/B/C all share dtype; the
// accumulation is in fp32. Wraps ks_gemm.
func GEMM(c, a, b unsafe.Pointer, m, n, k int64, transA, transB bool,
	lda, ldb, ldc int64, alpha, beta float32, dtype Dtype, stream Stream) error {
	st := Status(C.ks_gemm(c, a, b,
		C.int64_t(m), C.int64_t(n), C.int64_t(k),
		boolToCInt(transA), boolToCInt(transB),
		C.int64_t(lda), C.int64_t(ldb), C.int64_t(ldc),
		C.float(alpha), C.float(beta), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_gemm")
}

// GEMMBiasAct computes D = act(alpha * A @ B + bias). bias is [N] (broadcast
// over rows) or nil. Fuses the linear epilogue used by MLP/QKV projections.
// Wraps ks_gemm_bias_act.
func GEMMBiasAct(d, a, b, bias unsafe.Pointer, m, n, k int64, alpha float32,
	act Activation, dtype Dtype, stream Stream) error {
	st := Status(C.ks_gemm_bias_act(d, a, b, bias,
		C.int64_t(m), C.int64_t(n), C.int64_t(k), C.float(alpha),
		C.ks_activation_t(act), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_gemm_bias_act")
}

// GEMMBatched runs strided-batched GEMM. Each batch b uses base + b*stride_{a,b,c}.
// Wraps ks_gemm_batched.
func GEMMBatched(c, a, b unsafe.Pointer, batch, m, n, k int64, transA, transB bool,
	strideA, strideB, strideC int64, alpha, beta float32, dtype Dtype, stream Stream) error {
	st := Status(C.ks_gemm_batched(c, a, b,
		C.int64_t(batch), C.int64_t(m), C.int64_t(n), C.int64_t(k),
		boolToCInt(transA), boolToCInt(transB),
		C.int64_t(strideA), C.int64_t(strideB), C.int64_t(strideC),
		C.float(alpha), C.float(beta), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_gemm_batched")
}

// GEMMW8A8 runs a W8A8 GEMM: int8 A [M,K] and int8 B [K,N] -> outDtype C [M,N].
// Dequant: C = (aScale (x) bScale) * (A_i8 @ B_i8) + bias.
//
//	aScale: per-token [M] (or [1] for per-tensor) fp32 device ptr
//	bScale: per-channel [N] (or [1] for per-tensor) fp32 device ptr
//	bias:   [N] in outDtype or nil
//
// Wraps ks_gemm_w8a8.
func GEMMW8A8(c, aI8, bI8 unsafe.Pointer, aScale, bScale unsafe.Pointer,
	bias unsafe.Pointer, m, n, k int64, aMode, bMode QuantMode,
	outDtype Dtype, stream Stream) error {
	st := Status(C.ks_gemm_w8a8(c, aI8, bI8,
		(*C.float)(aScale), (*C.float)(bScale), bias,
		C.int64_t(m), C.int64_t(n), C.int64_t(k),
		C.ks_quant_mode_t(aMode), C.ks_quant_mode_t(bMode),
		C.ks_dtype_t(outDtype), stream.c()))
	return statusError(st, "ks_gemm_w8a8")
}

// GEMMW4A16 runs a W4A16 GEMM: fp16/bf16 activations A [M,K] x int4 weights
// B [K,N] with group-wise scales/zeros (AWQ/GPTQ layout). groupSize columns of
// K share a scale; bPacked holds two int4 per byte; scales/zeros are
// [K/groupSize, N]. bias may be nil. Wraps ks_gemm_w4a16.
func GEMMW4A16(c, a, bPacked, scales, zeros, bias unsafe.Pointer,
	m, n, k int64, groupSize int, dtype Dtype, stream Stream) error {
	st := Status(C.ks_gemm_w4a16(c, a, bPacked, scales, zeros, bias,
		C.int64_t(m), C.int64_t(n), C.int64_t(k), C.int(groupSize),
		C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_gemm_w4a16")
}

// GemmFP8 runs an FP8 GEMM: fp8 A [M,K] x fp8 B [K,N] -> outDtype C [M,N]
// (bf16/fp16/fp32). Both operands are FP8 (e4m3 or e5m2, selected by fp8Dtype),
// dequantized to fp32 by the supplied scales before the matmul:
// C = (aScale (x) bScale) * (A_fp8 @ B_fp8).
//
//	aScale: per-token [M] (KS_QUANT_PER_TOKEN) or [1] (PER_TENSOR) fp32 device ptr
//	bScale: per-channel [N] (KS_QUANT_PER_CHANNEL) or [1] (PER_TENSOR) fp32 device ptr
//
// Row-major, non-transposed (A row stride K, B row stride N). Wraps ks_gemm_fp8.
func GemmFP8(out, aFP8, bFP8 unsafe.Pointer, aScale, bScale unsafe.Pointer,
	m, n, k int64, aMode, bMode QuantMode, fp8Dtype, outDtype Dtype, stream Stream) error {
	st := Status(C.ks_gemm_fp8(out, aFP8, bFP8,
		(*C.float)(aScale), (*C.float)(bScale),
		C.int64_t(m), C.int64_t(n), C.int64_t(k),
		C.ks_quant_mode_t(aMode), C.ks_quant_mode_t(bMode),
		C.ks_dtype_t(fp8Dtype), C.ks_dtype_t(outDtype), stream.c()))
	return statusError(st, "ks_gemm_fp8")
}

// GemmFP8Blockwise runs an FP8 blockwise GEMM (DeepSeek-V3 recipe): fp8 A [M,K]
// x fp8 B [K,N] -> outDtype C [M,N] with fine-grained fp32 scales and two-level
// fp32 accumulation.
//
//	aScale: [M, ceil(K/blockK)] fp32 device ptr (per 1 x blockK activation tile)
//	bScale: [ceil(K/blockK), ceil(N/blockN)] fp32 device ptr (per blockK x blockN block)
//
// blockK/blockN are typically 128. Row-major, non-transposed (A row stride K,
// B row stride N). Wraps ks_gemm_fp8_blockwise.
func GemmFP8Blockwise(out, aFP8, bFP8 unsafe.Pointer, aScale, bScale unsafe.Pointer,
	m, n, k int64, blockN, blockK int, fp8Dtype, outDtype Dtype, stream Stream) error {
	st := Status(C.ks_gemm_fp8_blockwise(out, aFP8, bFP8,
		(*C.float)(aScale), (*C.float)(bScale),
		C.int64_t(m), C.int64_t(n), C.int64_t(k),
		C.int(blockN), C.int(blockK),
		C.ks_dtype_t(fp8Dtype), C.ks_dtype_t(outDtype), stream.c()))
	return statusError(st, "ks_gemm_fp8_blockwise")
}
