package main

import (
	"fmt"
	"math"
	"unsafe"

	ks "github.com/kernel-set/go"
)

// ViTEncoder implements the SigLIP-L Vision Transformer (ViT-L/16, 384x384).
type ViTEncoder struct {
	jp *JanusPro
}

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
	// Convert float32 pixels to fp16 for the GPU.
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

	return lnOut, nil
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
	kW, _ := v.jp.getWeight(prefix + ".self_attn.k_proj.weight")
	vW, _ := v.jp.getWeight(prefix + ".self_attn.v_proj.weight")

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

	ks.GEMM(q.ptr, lnOut.ptr, qW.ptr, int64(seqLen), int64(hidden), int64(hidden),
		false, false, int64(hidden), int64(hidden), int64(hidden), 1.0, 0.0, cfg.dtype, v.jp.stream)
	if err := v.jp.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("vit: layer %d q_proj: %w", layerIdx, err)
	}
	ks.GEMM(k.ptr, lnOut.ptr, kW.ptr, int64(seqLen), int64(hidden), int64(hidden),
		false, false, int64(hidden), int64(hidden), int64(hidden), 1.0, 0.0, cfg.dtype, v.jp.stream)
	if err := v.jp.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("vit: layer %d k_proj: %w", layerIdx, err)
	}
	ks.GEMM(vVal.ptr, lnOut.ptr, vW.ptr, int64(seqLen), int64(hidden), int64(hidden),
		false, false, int64(hidden), int64(hidden), int64(hidden), 1.0, 0.0, cfg.dtype, v.jp.stream)
	if err := v.jp.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("vit: layer %d v_proj: %w", layerIdx, err)
	}
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

	projOut, _ := v.jp.allocTensor(seqLen, hidden)
	ks.GEMM(projOut.ptr, attnOut.ptr, outW.ptr, int64(seqLen), int64(hidden), int64(hidden),
		false, false, int64(hidden), int64(hidden), int64(hidden), 1.0, 0.0, cfg.dtype, v.jp.stream)
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

	fc1Out, _ := v.jp.allocTensor(seqLen, intermediate)
	ks.GEMM(fc1Out.ptr, ln2Out.ptr, fc1W.ptr, int64(seqLen), int64(intermediate), int64(hidden),
		false, false, int64(hidden), int64(hidden), int64(intermediate), 1.0, 0.0, cfg.dtype, v.jp.stream)

	ks.GELU(fc1Out.ptr, fc1Out.ptr, int64(seqLen*intermediate), false, cfg.dtype, v.jp.stream)

	fc2W, _ := v.jp.getWeight(prefix + ".mlp.fc2.weight")
	fc2B, _ := v.jp.getWeight(prefix + ".mlp.fc2.bias")

	fc2Out, _ := v.jp.allocTensor(seqLen, hidden)
	ks.GEMM(fc2Out.ptr, fc1Out.ptr, fc2W.ptr, int64(seqLen), int64(hidden), int64(intermediate),
		false, false, int64(intermediate), int64(intermediate), int64(hidden), 1.0, 0.0, cfg.dtype, v.jp.stream)
	if err := v.jp.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("vit: layer %d fc2: %w", layerIdx, err)
	}
	ln2Out.free()
	fc1Out.free()
	_ = fc2B // bias not supported by ks.Add (element-wise, no broadcast)

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