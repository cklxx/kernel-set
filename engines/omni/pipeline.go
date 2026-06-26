package main

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"unsafe"

	ks "github.com/kernel-set/go"
)

// Understand runs the understanding pipeline: image -> ViT -> adaptor -> LLM -> text.
func (jp *JanusPro) Understand(img *image.RGBA, prompt string, maxNewTokens int) (string, error) {
	cfg := jp.config

	// 1. Preprocess image: resize to 384x384, normalize
	pixels := preprocessImage(img, cfg.vitImageSize)

	// 2. ViT encode
	vit := &ViTEncoder{jp: jp}
	vitOut, err := vit.Forward(pixels)
	if err != nil {
		return "", fmt.Errorf("vit: %w", err)
	}
	// vitOut shape: [1, NP, vitHidden] where NP = (384/16)^2 = 576

	// 3. Adaptor: 2-layer MLP [NP, vitHidden] -> [NP, llmHidden]
	NP := vitOut.shape[1]
	adaptorW1, err := jp.getWeight("aligner.layers.0.weight")
	if err != nil {
		vitOut.free()
		return "", fmt.Errorf("adaptor weight: %w", err)
	}
	adaptorW2, _ := jp.getWeight("aligner.layers.2.weight")

	// Layer 1: GEMM + GELU
	adaptorHid, err := jp.allocTensor(NP, cfg.llmHidden)
	if err != nil {
		vitOut.free()
		return "", err
	}
	ks.GEMM(adaptorHid.ptr, vitOut.ptr, adaptorW1.ptr,
		int64(NP), int64(cfg.llmHidden), int64(cfg.vitHidden),
		false, false, int64(cfg.vitHidden), int64(cfg.vitHidden), int64(cfg.llmHidden),
		1.0, 0.0, cfg.dtype, jp.stream)
	ks.GELU(adaptorHid.ptr, adaptorHid.ptr, int64(NP*cfg.llmHidden), false, cfg.dtype, jp.stream)

	// Layer 2: GEMM
	adaptorOut, err := jp.allocTensor(NP, cfg.llmHidden)
	if err != nil {
		adaptorHid.free()
		vitOut.free()
		return "", err
	}
	ks.GEMM(adaptorOut.ptr, adaptorHid.ptr, adaptorW2.ptr,
		int64(NP), int64(cfg.llmHidden), int64(cfg.llmHidden),
		false, false, int64(cfg.llmHidden), int64(cfg.llmHidden), int64(cfg.llmHidden),
		1.0, 0.0, cfg.dtype, jp.stream)
	adaptorHid.free()
	vitOut.free()

	// 4. Tokenize prompt
	promptTokens := tokenize(prompt)

	// 5. Embed token ids
	embedW, err := jp.getWeight("model.embed_tokens.weight")
	if err != nil {
		adaptorOut.free()
		return "", fmt.Errorf("embed weight: %w", err)
	}
	textEmb, err := jp.allocTensor(len(promptTokens), cfg.llmHidden)
	if err != nil {
		adaptorOut.free()
		return "", err
	}
	tokPtr := hostToDevice32(jp, promptTokens)
	defer ks.FreeDevice(tokPtr)
	ks.EmbeddingLookup(textEmb.ptr, embedW.ptr, tokPtr, false,
		int64(len(promptTokens)), int64(cfg.llmHidden), cfg.dtype, jp.stream)

	// 6. Concatenate: [vision_embeds, text_embeds]
	totalLen := NP + len(promptTokens)
	combined, err := jp.allocTensor(totalLen, cfg.llmHidden)
	if err != nil {
		adaptorOut.free()
		textEmb.free()
		return "", err
	}
	// Copy vision embeds
	ks.Memcpy(combined.ptr, adaptorOut.ptr, uintptr(NP*cfg.llmHidden*2), ks.MemcpyDeviceToDevice, jp.stream)
	adaptorOut.free()
	// Copy text embeds
	ks.Memcpy(
		unsafePtrOffset(combined.ptr, NP*cfg.llmHidden*2),
		textEmb.ptr, uintptr(len(promptTokens)*cfg.llmHidden*2),
		ks.MemcpyDeviceToDevice, jp.stream)
	textEmb.free()

	// 7. LLM prefill
	llm := &LLMEngine{jp: jp}
	hidden, err := llm.Prefill(combined, 0)
	combined.free()
	if err != nil {
		return "", fmt.Errorf("llm prefill: %w", err)
	}

	// 8. Autoregressive decode
	generated := make([]int32, 0, maxNewTokens)
	currentPos := totalLen
	for i := 0; i < maxNewTokens; i++ {
		logits, err := llm.Logits(hidden)
		if err != nil {
			hidden.free()
			return "", fmt.Errorf("logits: %w", err)
		}
		token, err := llm.ArgmaxToken(logits)
		logits.free()
		if err != nil {
			hidden.free()
			return "", fmt.Errorf("argmax: %w", err)
		}
		generated = append(generated, token)

		// Embed next token
		nextEmb, err := jp.allocTensor(1, cfg.llmHidden)
		if err != nil {
			hidden.free()
			return "", err
		}
		nextTokPtr := hostToDevice32(jp, []int32{token})
		ks.EmbeddingLookup(nextEmb.ptr, embedW.ptr, nextTokPtr, false,
			1, int64(cfg.llmHidden), cfg.dtype, jp.stream)
		ks.FreeDevice(nextTokPtr)

		hidden.free()
		hidden, err = llm.DecodeOne(nextEmb, currentPos)
		nextEmb.free()
		currentPos++
		if err != nil {
			return "", fmt.Errorf("decode step %d: %w", i, err)
		}
	}

	hidden.free()
	return detokenize(generated), nil
}

// Generate runs the generation pipeline: text -> LLM -> adaptor -> VQ decoder -> image.
func (jp *JanusPro) Generate(prompt string, maxNewTokens int) (image.Image, error) {
	cfg := jp.config

	// 1. Tokenize prompt
	promptTokens := tokenize(prompt)

	// 2. Embed
	embedW, err := jp.getWeight("model.embed_tokens.weight")
	if err != nil {
		return nil, fmt.Errorf("embed weight: %w", err)
	}
	textEmb, err := jp.allocTensor(len(promptTokens), cfg.llmHidden)
	if err != nil {
		return nil, err
	}
	tokPtr := hostToDevice32(jp, promptTokens)
	defer ks.FreeDevice(tokPtr)
	ks.EmbeddingLookup(textEmb.ptr, embedW.ptr, tokPtr, false,
		int64(len(promptTokens)), int64(cfg.llmHidden), cfg.dtype, jp.stream)

	// 3. LLM prefill
	llm := &LLMEngine{jp: jp}
	hidden, err := llm.Prefill(textEmb, 0)
	textEmb.free()
	if err != nil {
		return nil, fmt.Errorf("llm prefill: %w", err)
	}

	// 4. Autoregressive decode (generate image tokens)
	imageTokens := make([]int32, 0, maxNewTokens)
	currentPos := len(promptTokens)
	for i := 0; i < maxNewTokens; i++ {
		logits, err := llm.Logits(hidden)
		if err != nil {
			hidden.free()
			return nil, fmt.Errorf("logits: %w", err)
		}
		token, err := llm.ArgmaxToken(logits)
		logits.free()
		if err != nil {
			hidden.free()
			return nil, err
		}
		imageTokens = append(imageTokens, token)

		nextEmb, err := jp.allocTensor(1, cfg.llmHidden)
		if err != nil {
			hidden.free()
			return nil, err
		}
		nextTokPtr := hostToDevice32(jp, []int32{token})
		ks.EmbeddingLookup(nextEmb.ptr, embedW.ptr, nextTokPtr, false,
			1, int64(cfg.llmHidden), cfg.dtype, jp.stream)
		ks.FreeDevice(nextTokPtr)

		hidden.free()
		hidden, err = llm.DecodeOne(nextEmb, currentPos)
		nextEmb.free()
		currentPos++
		if err != nil {
			return nil, fmt.Errorf("decode step %d: %w", i, err)
		}
	}
	hidden.free()

	// 5. VQ Decoder: image tokens -> image
	vd := &VQDecoder{jp: jp}
	imgTensor, err := vd.Decode(imageTokens, cfg.vitImageSize, cfg.vitImageSize)
	if err != nil {
		return nil, fmt.Errorf("vq decode: %w", err)
	}
	defer imgTensor.free()

	// 6. Convert to Go image
	imgHost, err := jp.deviceToHost(imgTensor)
	if err != nil {
		return nil, fmt.Errorf("read image: %w", err)
	}

	H := cfg.vitImageSize
	W := cfg.vitImageSize
	rgba := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			r := fp16ToU8(imgHost[2*(0*H*W+y*W+x):])
			g := fp16ToU8(imgHost[2*(1*H*W+y*W+x):])
			b := fp16ToU8(imgHost[2*(2*H*W+y*W+x):])
			rgba.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}
	return rgba, nil
}

// preprocessImage resizes to targetSize x targetSize and normalizes to [-1, 1]
// returning [3, H, W] float32 in NCHW layout.
func preprocessImage(img *image.RGBA, targetSize int) []float32 {
	bounds := img.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()

	out := make([]float32, 3*targetSize*targetSize)
	for y := 0; y < targetSize; y++ {
		srcY := y * srcH / targetSize
		for x := 0; x < targetSize; x++ {
			srcX := x * srcW / targetSize
			r, g, b, _ := img.At(srcX, srcY).RGBA()
			// Normalize: (pixel/255 - 0.5) / 0.5 = 2*(pixel/255) - 1
			out[0*targetSize*targetSize+y*targetSize+x] = float32(r>>8)/127.5 - 1.0
			out[1*targetSize*targetSize+y*targetSize+x] = float32(g>>8)/127.5 - 1.0
			out[2*targetSize*targetSize+y*targetSize+x] = float32(b>>8)/127.5 - 1.0
		}
	}
	return out
}

func hostToDevice32(jp *JanusPro, data []int32) unsafe.Pointer {
	raw := make([]byte, len(data)*4)
	for i, v := range data {
		raw[i*4] = byte(v)
		raw[i*4+1] = byte(v >> 8)
		raw[i*4+2] = byte(v >> 16)
		raw[i*4+3] = byte(v >> 24)
	}
	ptr, _ := ks.MallocDevice(uintptr(len(raw)))
	ks.CopyToDevice(ptr, raw, jp.stream)
	return ptr
}

func unsafePtrOffset(base unsafe.Pointer, offset int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(base) + uintptr(offset))
}

func fp16ToU8(b []byte) uint8 {
	if len(b) < 2 {
		return 0
	}
	bits := uint16(b[0]) | uint16(b[1])<<8
	sign := bits >> 15
	exp := (bits >> 10) & 0x1f
	mant := bits & 0x3ff
	if exp == 0 {
		return 0
	}
	val := float32(1.0 + float32(mant)/1024.0)
	if exp >= 16 {
		val *= float32(int32(1) << uint32(exp-16))
	} else {
		val /= float32(int32(1) << uint32(16-exp))
	}
	if sign != 0 {
		val = -val
	}
	val = val*0.5 + 0.5
	if val < 0 {
		val = 0
	}
	if val > 1 {
		val = 1
	}
	return uint8(val * 255)
}

// ---- tokenizer via subprocess ----

var tokenizerCmd *exec.Cmd
var tokenizerStdin io.WriteCloser
var tokenizerStdout *bufio.Scanner

func initTokenizer() error {
	tokenizerCmd = exec.Command("python3", "tokenizer.py")
	tokenizerCmd.Dir = "." // run from same directory as binary
	var err error
	tokenizerStdin, err = tokenizerCmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("tokenizer stdin: %w", err)
	}
	stdout, err := tokenizerCmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("tokenizer stdout: %w", err)
	}
	tokenizerStdout = bufio.NewScanner(stdout)
	if err := tokenizerCmd.Start(); err != nil {
		return fmt.Errorf("start tokenizer: %w", err)
	}
	return nil
}

func closeTokenizer() {
	if tokenizerCmd != nil {
		tokenizerStdin.Close()
		tokenizerCmd.Wait()
	}
}

func tokenize(text string) []int32 {
	if tokenizerCmd == nil {
		// fallback: raw code points
		tokens := make([]int32, len(text))
		for i, c := range text {
			tokens[i] = int32(c)
		}
		return tokens
	}
	fmt.Fprintf(tokenizerStdin, "E:%s\n", text)
	if !tokenizerStdout.Scan() {
		return nil
	}
	line := tokenizerStdout.Text()
	parts := strings.Fields(line)
	tokens := make([]int32, len(parts))
	for i, p := range parts {
		v, _ := strconv.Atoi(p)
		tokens[i] = int32(v)
	}
	return tokens
}

func detokenize(tokens []int32) string {
	if tokenizerCmd == nil {
		runes := make([]rune, len(tokens))
		for i, t := range tokens {
			if t >= 0 && t < 128 {
				runes[i] = rune(t)
			} else {
				runes[i] = '?'
			}
		}
		return string(runes)
	}
	strs := make([]string, len(tokens))
	for i, t := range tokens {
		strs[i] = strconv.Itoa(int(t))
	}
	fmt.Fprintf(tokenizerStdin, "D:%s\n", strings.Join(strs, " "))
	if !tokenizerStdout.Scan() {
		return ""
	}
	return tokenizerStdout.Text()
}