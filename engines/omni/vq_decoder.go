package main

import (
	"fmt"
	"strconv"

	ks "github.com/kernel-set/go"
)

// VQDecoder implements the VQGAN CNN decoder.
type VQDecoder struct {
	jp *JanusPro
}

func (vd *VQDecoder) Decode(tokens []int32, height, width int) (*deviceTensor, error) {
	cfg := vd.jp.config
	latentDim := cfg.vqLatentDim
	h := height / cfg.vqDownsample
	w := width / cfg.vqDownsample
	if h*w != len(tokens) {
		return nil, fmt.Errorf("vq: token count %d != %d*%d", len(tokens), h, w)
	}

	// 1. Codebook lookup: tokens -> embeddings [seqLen, latentDim]
	codebookW, _ := vd.jp.getWeight("gen_vision_model.codebook.weight")
	tokPtr, _ := ks.MallocDevice(uintptr(len(tokens) * 4))
	defer ks.FreeDevice(tokPtr)
	tokBytes := make([]byte, len(tokens)*4)
	for i, t := range tokens {
		tokBytes[i*4] = byte(t)
		tokBytes[i*4+1] = byte(t >> 8)
		tokBytes[i*4+2] = byte(t >> 16)
		tokBytes[i*4+3] = byte(t >> 24)
	}
	ks.CopyToDevice(tokPtr, tokBytes, vd.jp.stream)

	emb, _ := vd.jp.allocTensor(len(tokens), latentDim)
	ks.EmbeddingLookup(emb.ptr, codebookW.ptr, tokPtr, false,
		int64(len(tokens)), int64(latentDim), cfg.dtype, vd.jp.stream)

	// 2. Reshape: [seqLen, latentDim] -> [latentDim, h, w]
	// CPU roundtrip: only 24*24*8 = 4608 elements, negligible.
	embHost, _ := vd.jp.deviceToHost(emb)
	emb.free()
	spatial := make([]byte, latentDim*h*w*2)
	for p := 0; p < h*w; p++ {
		py := p / w
		px := p % w
		for d := 0; d < latentDim; d++ {
			src := (p*latentDim + d) * 2
			dst := (d*h*w + py*w + px) * 2
			spatial[dst] = embHost[src]
			spatial[dst+1] = embHost[src+1]
		}
	}
	x, _ := vd.jp.allocTensor(1, latentDim, h, w)
	ks.CopyToDevice(x.ptr, spatial, vd.jp.stream)

	// 3. post_quant_conv: [8, 256] 1x1 conv -> [256, h, w]
	postQW, _ := vd.jp.getWeight("gen_vision_model.post_quant_conv.weight")
	postQB, _ := vd.jp.getWeight("gen_vision_model.post_quant_conv.bias")
	x, err := vd.conv2D(x, postQW, postQB, 256, 1)
	if err != nil {
		return nil, fmt.Errorf("vq: post_quant_conv: %w", err)
	}

	// 4. conv_in: [256, 512] 3x3 conv -> [512, h, w]
	convInW, _ := vd.jp.getWeight("gen_vision_model.decoder.conv_in.weight")
	convInB, _ := vd.jp.getWeight("gen_vision_model.decoder.conv_in.bias")
	x, err = vd.conv2D(x, convInW, convInB, 512, 1)
	if err != nil {
		return nil, fmt.Errorf("vq: conv_in: %w", err)
	}

	// 5. mid: res -> attn -> res
	prefix := "gen_vision_model.decoder.mid"
	x, err = vd.resBlock(x, prefix+".0", 512, 512, false)
	if err != nil {
		return nil, fmt.Errorf("vq: mid.0: %w", err)
	}
	x, err = vd.attentionBlock(x, prefix+".1", 512)
	if err != nil {
		return nil, fmt.Errorf("vq: mid.1: %w", err)
	}
	x, err = vd.resBlock(x, prefix+".2", 512, 512, false)
	if err != nil {
		return nil, fmt.Errorf("vq: mid.2: %w", err)
	}

	// 6. conv_blocks: 5 stages
	channels := 512
	hasAttn := []bool{true, false, false, false, false}
	nextCh := []int{512, 256, 256, 128, 128}
	for b := 0; b < 5; b++ {
		prefix = fmt.Sprintf("gen_vision_model.decoder.conv_blocks.%d", b)
		for r := 0; r < 3; r++ {
			outCh := channels
			if r == 0 {
				outCh = nextCh[b]
			}
			hasShortcut := (r == 0 && channels != outCh)
			x, err = vd.resBlock(x, prefix+".res."+strconv.Itoa(r), channels, outCh, hasShortcut)
			if err != nil {
				return nil, fmt.Errorf("vq: block %d res %d: %w", b, r, err)
			}
			channels = outCh
		}
		if hasAttn[b] {
			for a := 0; a < 3; a++ {
				x, err = vd.attentionBlock(x, prefix+".attn."+strconv.Itoa(a), channels)
				if err != nil {
					return nil, fmt.Errorf("vq: block %d attn %d: %w", b, a, err)
				}
			}
		}
		if b < 4 {
			upW, _ := vd.jp.getWeight(prefix + ".upsample.conv.weight")
			upB, _ := vd.jp.getWeight(prefix + ".upsample.conv.bias")
			x, err = vd.upsample(x, upW, upB, channels)
			if err != nil {
				return nil, fmt.Errorf("vq: block %d upsample: %w", b, err)
			}
			h *= 2
			w *= 2
		}
	}

	// 7. norm_out
	normW, _ := vd.jp.getWeight("gen_vision_model.decoder.norm_out.weight")
	normB, _ := vd.jp.getWeight("gen_vision_model.decoder.norm_out.bias")
	normOut, _ := vd.jp.allocTensor(1, channels, h, w)
	ks.GroupNorm(normOut.ptr, x.ptr, unsafePtr(normW), unsafePtr(normB),
		1, channels, h*w, 32, 1e-6, cfg.dtype, vd.jp.stream)
	ks.SiLU(normOut.ptr, normOut.ptr, int64(channels*h*w), cfg.dtype, vd.jp.stream)
	x.free()
	x = normOut

	// 8. final conv: [128, 3] 3x3 conv -> [3, h, w]
	finalW, _ := vd.jp.getWeight("gen_vision_model.decoder.final.weight")
	finalB, _ := vd.jp.getWeight("gen_vision_model.decoder.final.bias")
	x, err = vd.conv2D(x, finalW, finalB, 3, 1)
	if err != nil {
		return nil, fmt.Errorf("vq: final: %w", err)
	}

	return x, nil
}

func (vd *VQDecoder) conv2D(x *deviceTensor, weight, bias *deviceTensor, outChannels, groups int) (*deviceTensor, error) {
	cfg := vd.jp.config
	inC := x.shape[1]
	h := x.shape[2]
	w := x.shape[3]
	kH, kW := 1, 1
	if len(weight.shape) >= 4 {
		kH, kW = weight.shape[2], weight.shape[3]
	}
	out, _ := vd.jp.allocTensor(1, outChannels, h, w)
	ks.Conv2D(out.ptr, x.ptr, weight.ptr, unsafePtr(bias),
		1, inC, h, w, outChannels, kH, kW, 1, 1, kH/2, kW/2, 1, 1, groups, cfg.dtype, vd.jp.stream)
	x.free()
	return out, nil
}

func (vd *VQDecoder) resBlock(x *deviceTensor, prefix string, inCh, outCh int, hasShortcut bool) (*deviceTensor, error) {
	cfg := vd.jp.config
	h := x.shape[2]
	w := x.shape[3]

	// norm1 -> silu -> conv1
	norm1W, _ := vd.jp.getWeight(prefix + ".norm1.weight")
	norm1B, _ := vd.jp.getWeight(prefix + ".norm1.bias")
	norm1, _ := vd.jp.allocTensor(1, inCh, h, w)
	ks.GroupNorm(norm1.ptr, x.ptr, unsafePtr(norm1W), unsafePtr(norm1B),
		1, inCh, h*w, 32, 1e-6, cfg.dtype, vd.jp.stream)
	ks.SiLU(norm1.ptr, norm1.ptr, int64(inCh*h*w), cfg.dtype, vd.jp.stream)

	conv1W, _ := vd.jp.getWeight(prefix + ".conv1.weight")
	conv1B, _ := vd.jp.getWeight(prefix + ".conv1.bias")
	conv1, _ := vd.jp.allocTensor(1, outCh, h, w)
	ks.Conv2D(conv1.ptr, norm1.ptr, conv1W.ptr, unsafePtr(conv1B),
		1, inCh, h, w, outCh, 3, 3, 1, 1, 1, 1, 1, 1, 1, cfg.dtype, vd.jp.stream)
	norm1.free()

	// norm2 -> silu -> conv2
	norm2W, _ := vd.jp.getWeight(prefix + ".norm2.weight")
	norm2B, _ := vd.jp.getWeight(prefix + ".norm2.bias")
	norm2, _ := vd.jp.allocTensor(1, outCh, h, w)
	ks.GroupNorm(norm2.ptr, conv1.ptr, unsafePtr(norm2W), unsafePtr(norm2B),
		1, outCh, h*w, 32, 1e-6, cfg.dtype, vd.jp.stream)
	ks.SiLU(norm2.ptr, norm2.ptr, int64(outCh*h*w), cfg.dtype, vd.jp.stream)

	conv2W, _ := vd.jp.getWeight(prefix + ".conv2.weight")
	conv2B, _ := vd.jp.getWeight(prefix + ".conv2.bias")
	conv2, _ := vd.jp.allocTensor(1, outCh, h, w)
	ks.Conv2D(conv2.ptr, norm2.ptr, conv2W.ptr, unsafePtr(conv2B),
		1, outCh, h, w, outCh, 3, 3, 1, 1, 1, 1, 1, 1, 1, cfg.dtype, vd.jp.stream)
	norm2.free()
	conv1.free()

	if hasShortcut {
		scW, _ := vd.jp.getWeight(prefix + ".nin_shortcut.weight")
		scB, _ := vd.jp.getWeight(prefix + ".nin_shortcut.bias")
		sc, _ := vd.jp.allocTensor(1, outCh, h, w)
		ks.Conv2D(sc.ptr, x.ptr, scW.ptr, unsafePtr(scB),
			1, inCh, h, w, outCh, 1, 1, 1, 0, 0, 0, 1, 1, 1, cfg.dtype, vd.jp.stream)
		ks.Add(conv2.ptr, conv2.ptr, sc.ptr, int64(outCh*h*w), cfg.dtype, vd.jp.stream)
		sc.free()
	} else {
		ks.Add(conv2.ptr, conv2.ptr, x.ptr, int64(outCh*h*w), cfg.dtype, vd.jp.stream)
	}
	x.free()
	return conv2, nil
}

func (vd *VQDecoder) attentionBlock(x *deviceTensor, prefix string, channels int) (*deviceTensor, error) {
	cfg := vd.jp.config
	h := x.shape[2]
	w := x.shape[3]
	n := h * w

	// norm
	normW, _ := vd.jp.getWeight(prefix + ".norm.weight")
	normB, _ := vd.jp.getWeight(prefix + ".norm.bias")
	norm, _ := vd.jp.allocTensor(1, channels, h, w)
	ks.GroupNorm(norm.ptr, x.ptr, unsafePtr(normW), unsafePtr(normB),
		1, channels, n, 32, 1e-6, cfg.dtype, vd.jp.stream)

	// Q, K, V: 1x1 convs
	qW, _ := vd.jp.getWeight(prefix + ".q.weight")
	qB, _ := vd.jp.getWeight(prefix + ".q.bias")
	q, _ := vd.jp.allocTensor(1, channels, h, w)
	ks.Conv2D(q.ptr, norm.ptr, qW.ptr, unsafePtr(qB),
		1, channels, h, w, channels, 1, 1, 1, 0, 0, 0, 1, 1, 1, cfg.dtype, vd.jp.stream)

	kW, _ := vd.jp.getWeight(prefix + ".k.weight")
	kB, _ := vd.jp.getWeight(prefix + ".k.bias")
	k, _ := vd.jp.allocTensor(1, channels, h, w)
	ks.Conv2D(k.ptr, norm.ptr, kW.ptr, unsafePtr(kB),
		1, channels, h, w, channels, 1, 1, 1, 0, 0, 0, 1, 1, 1, cfg.dtype, vd.jp.stream)

	vW, _ := vd.jp.getWeight(prefix + ".v.weight")
	vB, _ := vd.jp.getWeight(prefix + ".v.bias")
	vVal, _ := vd.jp.allocTensor(1, channels, h, w)
	ks.Conv2D(vVal.ptr, norm.ptr, vW.ptr, unsafePtr(vB),
		1, channels, h, w, channels, 1, 1, 1, 0, 0, 0, 1, 1, 1, cfg.dtype, vd.jp.stream)
	norm.free()

	// NCHW -> NHWC (= seq layout) via GPU kernel
	qFlat, _ := vd.jp.allocTensor(1, n, channels)
	kFlat, _ := vd.jp.allocTensor(1, n, channels)
	vFlat, _ := vd.jp.allocTensor(1, n, channels)
	ks.NCHWToNHWC(qFlat.ptr, q.ptr, 1, channels, h, w, cfg.dtype, vd.jp.stream)
	ks.NCHWToNHWC(kFlat.ptr, k.ptr, 1, channels, h, w, cfg.dtype, vd.jp.stream)
	ks.NCHWToNHWC(vFlat.ptr, vVal.ptr, 1, channels, h, w, cfg.dtype, vd.jp.stream)
	q.free()
	k.free()
	vVal.free()

	scale := float32(1.0)
	attnOut, _ := vd.jp.allocTensor(1, n, channels)
	ks.FlashAttn(attnOut.ptr, nil, qFlat.ptr, kFlat.ptr, vFlat.ptr,
		1, n, n, 1, 1, channels, scale, false, cfg.dtype, vd.jp.stream)
	qFlat.free()
	kFlat.free()
	vFlat.free()

	// NHWC -> NCHW via GPU kernel
	attnSpatial, _ := vd.jp.allocTensor(1, channels, h, w)
	ks.NHWCToNCHW(attnSpatial.ptr, attnOut.ptr, 1, h, w, channels, cfg.dtype, vd.jp.stream)
	attnOut.free()

	projW, _ := vd.jp.getWeight(prefix + ".proj_out.weight")
	projB, _ := vd.jp.getWeight(prefix + ".proj_out.bias")
	proj, _ := vd.jp.allocTensor(1, channels, h, w)
	ks.Conv2D(proj.ptr, attnSpatial.ptr, projW.ptr, unsafePtr(projB),
		1, channels, h, w, channels, 1, 1, 1, 0, 0, 0, 1, 1, 1, cfg.dtype, vd.jp.stream)
	attnSpatial.free()

	ks.Add(proj.ptr, proj.ptr, x.ptr, int64(channels*n), cfg.dtype, vd.jp.stream)
	x.free()
	return proj, nil
}

func (vd *VQDecoder) upsample(x *deviceTensor, weight, bias *deviceTensor, outCh int) (*deviceTensor, error) {
	cfg := vd.jp.config
	inCh := x.shape[1]
	h := x.shape[2]
	w := x.shape[3]
	oh := h * 2
	ow := w * 2

	// GPU nearest-neighbor upsample
	up, _ := vd.jp.allocTensor(1, inCh, oh, ow)
	ks.UpsampleNearest2x(up.ptr, x.ptr, 1, inCh, h, w, cfg.dtype, vd.jp.stream)

	out, _ := vd.jp.allocTensor(1, outCh, oh, ow)
	ks.Conv2D(out.ptr, up.ptr, weight.ptr, unsafePtr(bias),
		1, inCh, oh, ow, outCh, 3, 3, 1, 1, 1, 1, 1, 1, 1, cfg.dtype, vd.jp.stream)
	up.free()
	x.free()
	return out, nil
}