package main

import (
	"fmt"
	"math"
	"unsafe"

	ks "github.com/kernel-set/go"
)

// LLMEngine implements the DeepSeek-LLM-7b forward pass.
type LLMEngine struct {
	jp *JanusPro
}

// Prefill runs the LLM over input embeddings [seqLen, hidden] and writes KV cache.
// Returns the last hidden state [1, hidden] for next-token prediction.
// The caller owns hiddenStates and must free it.
func (lm *LLMEngine) Prefill(hiddenStates *deviceTensor, startPos int) (*deviceTensor, error) {
	cfg := lm.jp.config
	seqLen := hiddenStates.shape[0]
	hidden := cfg.llmHidden

	x := hiddenStates
	for i := 0; i < cfg.llmLayers; i++ {
		var err error
		x, err = lm.transformerLayer(x, i, seqLen, startPos, true)
		if err != nil {
			return nil, fmt.Errorf("llm layer %d: %w", i, err)
		}
	}

	// Final RMSNorm
	normW, _ := lm.jp.getWeight("model.norm.weight")
	normOut, _ := lm.jp.allocTensor(seqLen, hidden)
	ks.RMSNorm(normOut.ptr, x.ptr, normW.ptr, int64(seqLen), int64(hidden), 1e-6, cfg.dtype, lm.jp.stream)

	// Return last token only
	last, _ := lm.jp.allocTensor(1, hidden)
	lm.copySlice(last, normOut, 0, seqLen-1, 1)
	normOut.free()
	return last, nil
}

// DecodeOne runs one decode step for a single token [1, hidden].
// The caller owns x and must free it.
func (lm *LLMEngine) DecodeOne(x *deviceTensor, pos int) (*deviceTensor, error) {
	cfg := lm.jp.config
	hidden := cfg.llmHidden

	for i := 0; i < cfg.llmLayers; i++ {
		var err error
		x, err = lm.transformerLayer(x, i, 1, pos, false)
		if err != nil {
			return nil, fmt.Errorf("llm layer %d: %w", i, err)
		}
	}

	normW, _ := lm.jp.getWeight("model.norm.weight")
	normOut, _ := lm.jp.allocTensor(1, hidden)
	ks.RMSNorm(normOut.ptr, x.ptr, normW.ptr, 1, int64(hidden), 1e-6, cfg.dtype, lm.jp.stream)

	return normOut, nil
}

func (lm *LLMEngine) transformerLayer(x *deviceTensor, layerIdx, seqLen, startPos int, isPrefill bool) (*deviceTensor, error) {
	cfg := lm.jp.config
	hidden := cfg.llmHidden
	heads := cfg.llmHeads
	kvHeads := cfg.llmKVHeads
	headDim := hidden / heads

	prefix := fmt.Sprintf("model.layers.%d", layerIdx)

	// RMSNorm
	lnW, _ := lm.jp.getWeight(prefix + ".input_layernorm.weight")
	lnOut, _ := lm.jp.allocTensor(seqLen, hidden)
	ks.RMSNorm(lnOut.ptr, x.ptr, lnW.ptr, int64(seqLen), int64(hidden), 1e-6, cfg.dtype, lm.jp.stream)

	// Q, K, V projections
	qW, _ := lm.jp.getWeight(prefix + ".self_attn.q_proj.weight")
	kW, _ := lm.jp.getWeight(prefix + ".self_attn.k_proj.weight")
	vW, _ := lm.jp.getWeight(prefix + ".self_attn.v_proj.weight")

	q, _ := lm.jp.allocTensor(seqLen, hidden)
	k, _ := lm.jp.allocTensor(seqLen, hidden)
	v, _ := lm.jp.allocTensor(seqLen, hidden)

	ks.GEMM(q.ptr, lnOut.ptr, qW.ptr, int64(seqLen), int64(hidden), int64(hidden),
		false, false, int64(hidden), int64(hidden), int64(hidden), 1.0, 0.0, cfg.dtype, lm.jp.stream)
	ks.GEMM(k.ptr, lnOut.ptr, kW.ptr, int64(seqLen), int64(hidden), int64(hidden),
		false, false, int64(hidden), int64(hidden), int64(hidden), 1.0, 0.0, cfg.dtype, lm.jp.stream)
	ks.GEMM(v.ptr, lnOut.ptr, vW.ptr, int64(seqLen), int64(hidden), int64(hidden),
		false, false, int64(hidden), int64(hidden), int64(hidden), 1.0, 0.0, cfg.dtype, lm.jp.stream)
	lnOut.free()

	// RoPE: extract cos/sin for current positions
	cosSlice, _ := lm.jp.allocTensor(seqLen, headDim/2)
	sinSlice, _ := lm.jp.allocTensor(seqLen, headDim/2)
	lm.copySlice(cosSlice, lm.jp.cos, 0, startPos, seqLen)
	lm.copySlice(sinSlice, lm.jp.sin, 0, startPos, seqLen)

	ks.RoPEInplace(q.ptr, k.ptr, cosSlice.ptr, sinSlice.ptr,
		int64(seqLen), heads, kvHeads, headDim, false, cfg.dtype, lm.jp.stream)
	cosSlice.free()
	sinSlice.free()

	// Write K/V to cache
	kvCache := lm.jp.kvCache[layerIdx]
	lm.copySlice(kvCache.k, k, startPos, 0, seqLen)
	lm.copySlice(kvCache.v, v, startPos, 0, seqLen)

	// Attention: use full context from cache directly (no copy).
	totalLen := startPos + seqLen
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	attnOut, _ := lm.jp.allocTensor(seqLen, hidden)
	ks.FlashAttn(attnOut.ptr, nil, q.ptr, kvCache.k.ptr, kvCache.v.ptr,
		1, seqLen, totalLen, heads, kvHeads, headDim,
		scale, isPrefill, cfg.dtype, lm.jp.stream)
	q.free()
	k.free()
	v.free()

	// Output projection
	outW, _ := lm.jp.getWeight(prefix + ".self_attn.o_proj.weight")
	projOut, _ := lm.jp.allocTensor(seqLen, hidden)
	ks.GEMM(projOut.ptr, attnOut.ptr, outW.ptr, int64(seqLen), int64(hidden), int64(hidden),
		false, false, int64(hidden), int64(hidden), int64(hidden), 1.0, 0.0, cfg.dtype, lm.jp.stream)
	attnOut.free()

	// Residual 1
	ks.Add(x.ptr, x.ptr, projOut.ptr, int64(seqLen*hidden), cfg.dtype, lm.jp.stream)
	projOut.free()

	// RMSNorm (post-attention)
	ln2W, _ := lm.jp.getWeight(prefix + ".post_attention_layernorm.weight")
	ln2Out, _ := lm.jp.allocTensor(seqLen, hidden)
	ks.RMSNorm(ln2Out.ptr, x.ptr, ln2W.ptr, int64(seqLen), int64(hidden), 1e-6, cfg.dtype, lm.jp.stream)

	// SwiGLU MLP
	intermediate := cfg.llmIntermediate
	gateW, _ := lm.jp.getWeight(prefix + ".mlp.gate_proj.weight")
	upW, _ := lm.jp.getWeight(prefix + ".mlp.up_proj.weight")
	downW, _ := lm.jp.getWeight(prefix + ".mlp.down_proj.weight")

	gate, _ := lm.jp.allocTensor(seqLen, intermediate)
	up, _ := lm.jp.allocTensor(seqLen, intermediate)
	ks.GEMM(gate.ptr, ln2Out.ptr, gateW.ptr, int64(seqLen), int64(intermediate), int64(hidden),
		false, false, int64(hidden), int64(hidden), int64(intermediate), 1.0, 0.0, cfg.dtype, lm.jp.stream)
	ks.GEMM(up.ptr, ln2Out.ptr, upW.ptr, int64(seqLen), int64(intermediate), int64(hidden),
		false, false, int64(hidden), int64(hidden), int64(intermediate), 1.0, 0.0, cfg.dtype, lm.jp.stream)
	ln2Out.free()

	swiOut, _ := lm.jp.allocTensor(seqLen, intermediate)
	ks.SwiGLU(swiOut.ptr, gate.ptr, up.ptr, int64(seqLen), int64(intermediate), cfg.dtype, lm.jp.stream)
	gate.free()
	up.free()

	downOut, _ := lm.jp.allocTensor(seqLen, hidden)
	ks.GEMM(downOut.ptr, swiOut.ptr, downW.ptr, int64(seqLen), int64(hidden), int64(intermediate),
		false, false, int64(intermediate), int64(intermediate), int64(hidden), 1.0, 0.0, cfg.dtype, lm.jp.stream)
	swiOut.free()

	// Residual 2
	ks.Add(x.ptr, x.ptr, downOut.ptr, int64(seqLen*hidden), cfg.dtype, lm.jp.stream)
	downOut.free()

	return x, nil
}

// copySlice copies seqLen rows from src[srcStartRow:] to dst[dstStartRow:].
func (lm *LLMEngine) copySlice(dst, src *deviceTensor, dstStartRow, srcStartRow, seqLen int) {
	rowSize := 1
	for _, s := range src.shape[1:] {
		rowSize *= s
	}
	elemBytes := 2 // fp16
	dstOffset := dstStartRow * rowSize * elemBytes
	srcOffset := srcStartRow * rowSize * elemBytes
	bytes := uintptr(seqLen * rowSize * elemBytes)

	ks.Memcpy(
		unsafe.Pointer(uintptr(dst.ptr)+uintptr(dstOffset)),
		unsafe.Pointer(uintptr(src.ptr)+uintptr(srcOffset)),
		bytes,
		ks.MemcpyDeviceToDevice, lm.jp.stream)
}

// Logits computes logits from the final hidden state via lm_head.
func (lm *LLMEngine) Logits(x *deviceTensor) (*deviceTensor, error) {
	cfg := lm.jp.config
	seqLen := x.shape[0]
	headW, _ := lm.jp.getWeight("lm_head.weight")

	logits, _ := lm.jp.allocTensor(seqLen, cfg.llmVocab)
	ks.GEMM(logits.ptr, x.ptr, headW.ptr, int64(seqLen), int64(cfg.llmVocab), int64(cfg.llmHidden),
		false, false, int64(cfg.llmHidden), int64(cfg.llmHidden), int64(cfg.llmVocab),
		1.0, 0.0, cfg.dtype, lm.jp.stream)
	return logits, nil
}

// ArgmaxToken returns the greedy token id from logits [1, vocab].
func (lm *LLMEngine) ArgmaxToken(logits *deviceTensor) (int32, error) {
	token := make([]int32, 1)
	tokPtr, err := ks.MallocDevice(4) // int32
	if err != nil {
		return 0, err
	}
	defer ks.FreeDevice(tokPtr)

	ks.Argmax(tokPtr, logits.ptr, 1, int64(lm.jp.config.llmVocab), lm.jp.config.dtype, lm.jp.stream)
	if err := ks.CopyFromDevice(
		(*[4]byte)(unsafe.Pointer(&token[0]))[:],
		tokPtr, lm.jp.stream); err != nil {
		return 0, err
	}
	if err := lm.jp.stream.Synchronize(); err != nil {
		return 0, err
	}
	return token[0], nil
}