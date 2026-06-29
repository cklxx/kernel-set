package main

import (
	"fmt"
	"math"
	"os"
	"unsafe"

	ks "github.com/kernel-set/go"
)

// ViTEncoder implements the SigLIP-L Vision Transformer (ViT-L/16, 384x384)
// with a Perceiver Resampler (attn_pool) that compresses 576 patches into 1 token.
type ViTEncoder struct {
	jp *JanusPro
}

// Forward runs the full ViT encoder including the attn_pool, returning a single
// vision token [1, 1, hidden].
func (v *ViTEncoder) Forward(pixels []float32) (*deviceTensor, error) {
	cfg := v.jp.config
	B := 1
	H := cfg.vitImageSize
	W := cfg.vitImageSize
	C := 3
	P := cfg.vitPatch
	NP := (H / P) * (W / P)
	hidden := cfg.vitHidden

	// 1. Patch embedding: Conv2d 3 -> hidden, kernel=patch, stride=patch
	pixelsFP16 := make([]uint16, len(pixels))
	for i, f := range pixels {
		pixelsFP16[i] = float32ToFP16(f)
	}
	imgTensor, err := v.jp.hostToDevice(pixelsFP16)
	if err != nil {
		return nil, fmt.Errorf("vit: upload image: %w", err)
	}
	defer imgTensor.free()
	imgTensor.shape = []int{B, C, H, W}

	// Debug: dump preprocessed image
	fmt.Fprintf(os.Stderr, "[vit] image: min=%v max=%v\n", pixels[0], pixels[len(pixels)-1])

	patchWeight, err := v.jp.getWeight("vision_model.embeddings.patch_embedding.weight")
	if err != nil {
		return nil, fmt.Errorf("vit: patch weight: %w", err)
	}
	var patchBias *deviceTensor
	patchBias, _ = v.jp.getWeight("vision_model.embeddings.patch_embedding.bias")

	out, err := v.jp.allocTensor(B, hidden, NP)
	if err != nil {
		return nil, err
	}
	if err := ks.Conv2D(out.ptr, imgTensor.ptr, patchWeight.ptr, unsafePtr(patchBias),
		B, C, H, W, hidden, P, P, P, P, 0, 0, 1, 1, 1,
		cfg.dtype, v.jp.stream); err != nil {
		out.free()
		return nil, fmt.Errorf("vit: patch conv: %w", err)
	}

	// Debug: dump Conv2D output for comparison
	rawBytes, err := v.jp.deviceToHost(out)
	if err == nil {
		os.WriteFile("/tmp/go_conv2d_red_fp16.bin", rawBytes, 0644)
	}

	// 2. Transpose [B, hidden, NP] -> [B, NP, hidden] and add position embedding
	x, err := v.jp.allocTensor(B, NP, hidden)
	if err != nil {
		out.free()
		return nil, err
	}
	outHost, err := v.jp.deviceToHost(out)
	out.free()
	if err != nil {
		x.free()
		return nil, err
	}
	xHost := make([]byte, B*NP*hidden*2)
	for b := 0; b < B; b++ {
		for p := 0; p < NP; p++ {
			for d := 0; d < hidden; d++ {
				src := (b*hidden*NP + d*NP + p) * 2
				dst := (b*NP*hidden + p*hidden + d) * 2
				xHost[dst] = outHost[src]
				xHost[dst+1] = outHost[src+1]
			}
		}
	}
	if err := ks.CopyToDevice(x.ptr, xHost, v.jp.stream); err != nil {
		x.free()
		return nil, err
	}

	posEmb, err := v.jp.getWeight("vision_model.embeddings.position_embedding.weight")
	if err != nil {
		x.free()
		return nil, fmt.Errorf("vit: pos emb: %w", err)
	}
	if err := ks.Add(x.ptr, x.ptr, posEmb.ptr, int64(B*NP*hidden), cfg.dtype, v.jp.stream); err != nil {
		x.free()
		return nil, err
	}
	v.jp.stream.Synchronize()

	// 3. Transformer blocks
	for i := 0; i < cfg.vitLayers; i++ {
		var err error
		x, err = v.transformerBlock(x, i, B*NP, hidden)
		if err != nil {
			return nil, fmt.Errorf("vit: block %d: %w", i, err)
		}
	}

	// 4. Final LayerNorm (post_layernorm)
	lnWeight, err := v.jp.getWeight("vision_model.post_layernorm.weight")
	if err != nil {
		return nil, fmt.Errorf("vit: final ln weight: %w", err)
	}
	lnBias, _ := v.jp.getWeight("vision_model.post_layernorm.bias")

	lnOut, err := v.jp.allocTensor(B, NP, hidden)
	if err != nil {
		return nil, err
	}
	if err := ks.LayerNorm(lnOut.ptr, x.ptr, lnWeight.ptr, unsafePtr(lnBias),
		int64(B*NP), int64(hidden), 1e-6, cfg.dtype, v.jp.stream); err != nil {
		lnOut.free()
		return nil, err
	}
	x.free()

	// 5. Attention Pooling (Perceiver Resampler): compress 576 patches -> 1 token
	pooled, err := v.attnPool(lnOut)
	if err != nil {
		lnOut.free()
		return nil, fmt.Errorf("vit: attn_pool: %w", err)
	}
	lnOut.free()

	return pooled, nil
}

// attnPool implements the Perceiver Resampler that compresses the ViT output
// into a single token using cross-attention with a learned latent query.
// Input: x [1, 576, 1024], Output: [1, 1, 1024]
func (v *ViTEncoder) attnPool(x *deviceTensor) (*deviceTensor, error) {
	cfg := v.jp.config
	hidden := cfg.vitHidden
	heads := cfg.vitHeads
	headDim := hidden / heads

	// Get latent query [1, 1, hidden]
	latent, err := v.jp.getWeight("vision_model.vision_tower.attn_pool.latent")
	if err != nil {
		return nil, fmt.Errorf("attn_pool latent: %w", err)
	}
	// latent shape from safetensors is [1, 1, 1024]

	// Copy latent to a writable tensor
	latentCopy, err := v.jp.allocTensor(1, hidden)
	if err != nil {
		return nil, err
	}
	ks.Memcpy(latentCopy.ptr, latent.ptr, uintptr(hidden*2), ks.MemcpyDeviceToDevice, v.jp.stream)

	// Cross-attention: latent attends to x
	qW, _ := v.jp.getWeight("vision_model.vision_tower.attn_pool.q.weight")
	qB, _ := v.jp.getWeight("vision_model.vision_tower.attn_pool.q.bias")
	kvW, _ := v.jp.getWeight("vision_model.vision_tower.attn_pool.kv.weight")
	kvB, _ := v.jp.getWeight("vision_model.vision_tower.attn_pool.kv.bias")

	// q = latent @ qW + qB  -> [1, hidden]
	q, _ := v.jp.allocTensor(1, hidden)
	ks.GEMMBiasAct(q.ptr, latentCopy.ptr, qW.ptr, unsafePtr(qB),
		1, int64(hidden), int64(hidden), 1.0, ks.ActNone, cfg.dtype, v.jp.stream)

	// kv = x @ kvW + kvB  -> [576, hidden*2]; each row is [K, V] interleaved
	seqLen := x.shape[1]
	kv, _ := v.jp.allocTensor(seqLen, hidden*2)
	ks.GEMMBiasAct(kv.ptr, x.ptr, kvW.ptr, unsafePtr(kvB),
		int64(seqLen), int64(hidden*2), int64(hidden), 1.0, ks.ActNone, cfg.dtype, v.jp.stream)

	// De-interleave kv into separate k and v tensors.
	// FlashAttn expects k, v as [1, seqLen, hidden] (contiguous).
	k, _ := v.jp.allocTensor(seqLen, hidden)
	vVal, _ := v.jp.allocTensor(seqLen, hidden)
	// Copy alternating rows: for each row i, copy first hidden elems to k, last hidden to v.
	for i := 0; i < seqLen; i++ {
		rowBytes := uintptr(hidden * 2) // fp16 bytes per half-row
		ks.Memcpy(
			unsafePtrOffset(k.ptr, i*hidden*2),
			unsafePtrOffset(kv.ptr, i*hidden*2*2),
			rowBytes, ks.MemcpyDeviceToDevice, v.jp.stream)
		ks.Memcpy(
			unsafePtrOffset(vVal.ptr, i*hidden*2),
			unsafePtrOffset(kv.ptr, i*hidden*2*2+hidden*2),
			rowBytes, ks.MemcpyDeviceToDevice, v.jp.stream)
	}
	kv.free()

	// FlashAttention for cross-attention
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	attnOut, err := v.jp.allocTensor(1, hidden)
	if err != nil {
		q.free(); k.free(); vVal.free()
		return nil, err
	}
	if err := ks.FlashAttn(attnOut.ptr, nil, q.ptr, k.ptr, vVal.ptr,
		1, 1, seqLen, heads, heads, headDim,
		scale, false, cfg.dtype, v.jp.stream); err != nil {
		q.free(); k.free(); vVal.free(); attnOut.free()
		return nil, fmt.Errorf("attn_pool flash_attn: %w", err)
	}
	q.free()
	k.free()
	vVal.free()

	// Residual: latent = latent + attnOut
	ks.Add(latentCopy.ptr, latentCopy.ptr, attnOut.ptr, int64(hidden), cfg.dtype, v.jp.stream)
	attnOut.free()

	// LayerNorm
	normW, err := v.jp.getWeight("vision_model.vision_tower.attn_pool.norm.weight")
	if err != nil {
		return nil, fmt.Errorf("attn_pool norm weight: %w", err)
	}
	normB, _ := v.jp.getWeight("vision_model.vision_tower.attn_pool.norm.bias")
	lnOut, err := v.jp.allocTensor(1, hidden)
	if err != nil {
		latentCopy.free()
		return nil, fmt.Errorf("attn_pool ln alloc: %w", err)
	}
	ks.LayerNorm(lnOut.ptr, latentCopy.ptr, normW.ptr, unsafePtr(normB),
		1, int64(hidden), 1e-6, cfg.dtype, v.jp.stream)

	// MLP: fc1 -> GELU -> fc2
	intermediate := cfg.vitIntermediate
	fc1W, _ := v.jp.getWeight("vision_model.vision_tower.attn_pool.mlp.fc1.weight")
	fc1B, _ := v.jp.getWeight("vision_model.vision_tower.attn_pool.mlp.fc1.bias")
	fc1Out, _ := v.jp.allocTensor(1, intermediate)
	ks.GEMMBiasAct(fc1Out.ptr, lnOut.ptr, fc1W.ptr, unsafePtr(fc1B),
		1, int64(intermediate), int64(hidden), 1.0, ks.ActNone, cfg.dtype, v.jp.stream)
	ks.GELU(fc1Out.ptr, fc1Out.ptr, int64(intermediate), true, cfg.dtype, v.jp.stream)

	fc2W, _ := v.jp.getWeight("vision_model.vision_tower.attn_pool.mlp.fc2.weight")
	fc2B, _ := v.jp.getWeight("vision_model.vision_tower.attn_pool.mlp.fc2.bias")
	fc2Out, _ := v.jp.allocTensor(1, hidden)
	ks.GEMMBiasAct(fc2Out.ptr, fc1Out.ptr, fc2W.ptr, unsafePtr(fc2B),
		1, int64(hidden), int64(intermediate), 1.0, ks.ActNone, cfg.dtype, v.jp.stream)
	lnOut.free()
	fc1Out.free()

	// Residual: latent = latent + fc2Out
	ks.Add(latentCopy.ptr, latentCopy.ptr, fc2Out.ptr, int64(hidden), cfg.dtype, v.jp.stream)
	fc2Out.free()

	// Final projection
	projW, _ := v.jp.getWeight("vision_model.vision_tower.attn_pool.proj.weight")
	projB, _ := v.jp.getWeight("vision_model.vision_tower.attn_pool.proj.bias")
	result, _ := v.jp.allocTensor(1, hidden)
	ks.GEMMBiasAct(result.ptr, latentCopy.ptr, projW.ptr, unsafePtr(projB),
		1, int64(hidden), int64(hidden), 1.0, ks.ActNone, cfg.dtype, v.jp.stream)
	latentCopy.free()

	return result, nil
}

func (v *ViTEncoder) transformerBlock(x *deviceTensor, layerIdx, seqLen, hidden int) (*deviceTensor, error) {
	cfg := v.jp.config
	heads := cfg.vitHeads
	headDim := hidden / heads

	prefix := fmt.Sprintf("vision_model.encoder.layers.%d", layerIdx)

	// LayerNorm 1
	ln1W, _ := v.jp.getWeight(prefix + ".layer_norm1.weight")
	ln1B, _ := v.jp.getWeight(prefix + ".layer_norm1.bias")

	lnOut, err := v.jp.allocTensor(seqLen, hidden)
	if err != nil {
		return nil, fmt.Errorf("vit: ln1 alloc: %w", err)
	}
	if err := ks.LayerNorm(lnOut.ptr, x.ptr, ln1W.ptr, unsafePtr(ln1B),
		int64(seqLen), int64(hidden), 1e-6, cfg.dtype, v.jp.stream); err != nil {
		lnOut.free()
		return nil, fmt.Errorf("vit: ln1: %w", err)
	}

	// Q, K, V projections (separate)
	qW, _ := v.jp.getWeight(prefix + ".self_attn.q_proj.weight")
	qB, _ := v.jp.getWeight(prefix + ".self_attn.q_proj.bias")
	kW, _ := v.jp.getWeight(prefix + ".self_attn.k_proj.weight")
	kB, _ := v.jp.getWeight(prefix + ".self_attn.k_proj.bias")
	vW, _ := v.jp.getWeight(prefix + ".self_attn.v_proj.weight")
	vB, _ := v.jp.getWeight(prefix + ".self_attn.v_proj.bias")

	q, err := v.jp.allocTensor(seqLen, hidden)
	if err != nil {
		lnOut.free()
		return nil, fmt.Errorf("vit: q alloc: %w", err)
	}
	k, err := v.jp.allocTensor(seqLen, hidden)
	if err != nil {
		lnOut.free(); q.free()
		return nil, fmt.Errorf("vit: k alloc: %w", err)
	}
	vVal, err := v.jp.allocTensor(seqLen, hidden)
	if err != nil {
		lnOut.free(); q.free(); k.free()
		return nil, fmt.Errorf("vit: v alloc: %w", err)
	}

	ks.GEMMBiasAct(q.ptr, lnOut.ptr, qW.ptr, unsafePtr(qB), int64(seqLen), int64(hidden), int64(hidden), 1.0, ks.ActNone, cfg.dtype, v.jp.stream)
	ks.GEMMBiasAct(k.ptr, lnOut.ptr, kW.ptr, unsafePtr(kB), int64(seqLen), int64(hidden), int64(hidden), 1.0, ks.ActNone, cfg.dtype, v.jp.stream)
	ks.GEMMBiasAct(vVal.ptr, lnOut.ptr, vW.ptr, unsafePtr(vB), int64(seqLen), int64(hidden), int64(hidden), 1.0, ks.ActNone, cfg.dtype, v.jp.stream)
	lnOut.free()

	// Flash attention (non-causal for ViT)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	attnOut, _ := v.jp.allocTensor(seqLen, hidden)
	ks.FlashAttn(attnOut.ptr, nil, q.ptr, k.ptr, vVal.ptr,
		1, seqLen, seqLen, heads, heads, headDim,
		scale, false, cfg.dtype, v.jp.stream)
	q.free()
	k.free()
	vVal.free()

	// Output projection
	outW, _ := v.jp.getWeight(prefix + ".self_attn.out_proj.weight")
	outB, _ := v.jp.getWeight(prefix + ".self_attn.out_proj.bias")

	projOut, _ := v.jp.allocTensor(seqLen, hidden)
	ks.GEMMBiasAct(projOut.ptr, attnOut.ptr, outW.ptr, unsafePtr(outB), int64(seqLen), int64(hidden), int64(hidden), 1.0, ks.ActNone, cfg.dtype, v.jp.stream)
	attnOut.free()

	// Residual 1
	ks.Add(x.ptr, x.ptr, projOut.ptr, int64(seqLen*hidden), cfg.dtype, v.jp.stream)
	projOut.free()

	// LayerNorm 2
	ln2W, _ := v.jp.getWeight(prefix + ".layer_norm2.weight")
	ln2B, _ := v.jp.getWeight(prefix + ".layer_norm2.bias")

	ln2Out, err := v.jp.allocTensor(seqLen, hidden)
	if err != nil {
		return nil, fmt.Errorf("vit: ln2 alloc: %w", err)
	}
	if err := ks.LayerNorm(ln2Out.ptr, x.ptr, ln2W.ptr, unsafePtr(ln2B),
		int64(seqLen), int64(hidden), 1e-6, cfg.dtype, v.jp.stream); err != nil {
		ln2Out.free()
		return nil, fmt.Errorf("vit: ln2: %w", err)
	}

	// MLP: fc1 -> GELU -> fc2
	intermediate := cfg.vitIntermediate
	fc1W, _ := v.jp.getWeight(prefix + ".mlp.fc1.weight")
	fc1B, _ := v.jp.getWeight(prefix + ".mlp.fc1.bias")

	fc1Out, _ := v.jp.allocTensor(seqLen, intermediate)
	ks.GEMMBiasAct(fc1Out.ptr, ln2Out.ptr, fc1W.ptr, unsafePtr(fc1B), int64(seqLen), int64(intermediate), int64(hidden), 1.0, ks.ActNone, cfg.dtype, v.jp.stream)

	ks.GELU(fc1Out.ptr, fc1Out.ptr, int64(seqLen*intermediate), true, cfg.dtype, v.jp.stream)

	fc2W, _ := v.jp.getWeight(prefix + ".mlp.fc2.weight")
	fc2B, _ := v.jp.getWeight(prefix + ".mlp.fc2.bias")

	fc2Out, _ := v.jp.allocTensor(seqLen, hidden)
	ks.GEMMBiasAct(fc2Out.ptr, fc1Out.ptr, fc2W.ptr, unsafePtr(fc2B), int64(seqLen), int64(hidden), int64(intermediate), 1.0, ks.ActNone, cfg.dtype, v.jp.stream)
	ln2Out.free()
	fc1Out.free()

	// Residual 2
	ks.Add(x.ptr, x.ptr, fc2Out.ptr, int64(seqLen*hidden), cfg.dtype, v.jp.stream)
	if err := v.jp.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("vit: layer %d resid2: %w", layerIdx, err)
	}
	fc2Out.free()

	return x, nil
}

func unsafePtr(t *deviceTensor) unsafe.Pointer {
	if t == nil {
		return nil
	}
	return t.ptr
}

func fp16ToFloat32(bits uint16) float32 {
	sign := float32(1.0)
	if bits&0x8000 != 0 {
		sign = -1.0
	}
	exp := int((bits >> 10) & 0x1f)
	mant := float32(bits&0x3ff) / 1024.0
	if exp == 0 {
		return sign * mant * 6.103515625e-05 // 2^-14
	}
	// Avoid left-shift by negative amount (Go UB).
	if exp >= 15 {
		return sign * (1.0 + mant) * float32(int32(1)<<uint32(exp-15))
	}
	return sign * (1.0 + mant) / float32(int32(1)<<uint32(15-exp))
}

func toUint16(raw []byte) []uint16 {
	out := make([]uint16, len(raw)/2)
	for i := range out {
		out[i] = uint16(raw[i*2]) | uint16(raw[i*2+1])<<8
	}
	return out
}

func meanToBytes(mean []uint16) []byte {
	raw := make([]byte, len(mean)*2)
	for i, v := range mean {
		raw[i*2] = byte(v)
		raw[i*2+1] = byte(v >> 8)
	}
	return raw
}