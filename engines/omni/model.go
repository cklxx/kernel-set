package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"unsafe"

	ks "github.com/kernel-set/go"
)

// JanusPro holds the model weights and configuration for Janus-Pro-7B.
type JanusPro struct {
	stream  ks.Stream
	weights map[string]*deviceTensor
	config  janusConfig

	// pre-allocated device buffers
	kvCache []kvCacheLayer
	cos, sin *deviceTensor
}

type janusConfig struct {
	// ViT (SigLIP-L)
	vitHidden     int // 1024
	vitLayers     int // 24
	vitHeads      int // 16
	vitPatch      int // 16
	vitImageSize  int // 384
	vitIntermediate int // 4096

	// LLM (DeepSeek-LLM-7b)
	llmHidden     int // 4096
	llmLayers     int // 30
	llmHeads      int // 32
	llmKVHeads    int // 32
	llmIntermediate int // 11008
	llmVocab      int // 102400
	llmMaxPos     int // 4096
	ropeTheta     float32 // 10000.0

	// VQ Decoder
	vqCodebook    int // 16384
	vqLatentDim   int // 8
	vqDownsample  int // 16

	dtype ks.Dtype // F16
}

type kvCacheLayer struct {
	k, v *deviceTensor
}

// deviceTensor is a GPU buffer with shape metadata.
type deviceTensor struct {
	ptr   unsafe.Pointer
	shape []int
	size  int // bytes
}

func (t *deviceTensor) free() error {
	if t.ptr != nil {
		return ks.FreeDevice(t.ptr)
	}
	return nil
}

func (t *deviceTensor) numel() int {
	n := 1
	for _, s := range t.shape {
		n *= s
	}
	return n
}

func (t *deviceTensor) elemSize() int {
	return t.size / t.numel()
}

// NewJanusPro loads the model from a directory of safetensors files.
// NewJanusPro loads the model from a directory of safetensors files.
// Expects pre-transposed 2D weight matrices: model_fp16_t.safetensors.
// Embedding/codebook weights are loaded from the plain file to avoid transposition.
func NewJanusPro(dir string) (*JanusPro, error) {
	paths, err := findSafetensors(dir)
	if err != nil {
		return nil, fmt.Errorf("find safetensors: %w", err)
	}

	// Separate transposed and plain files.
	var transposedPaths, plainPaths []string
	for _, p := range paths {
		if strings.HasSuffix(p, "_t.safetensors") {
			transposedPaths = append(transposedPaths, p)
		} else if strings.HasSuffix(p, ".safetensors") {
			plainPaths = append(plainPaths, p)
		}
	}

	// Load transposed files first (for all weights).
	if len(transposedPaths) == 0 {
		// Fall back to plain files if no transposed files.
		transposedPaths = plainPaths
		plainPaths = nil
	}
	if len(transposedPaths) == 0 {
		return nil, fmt.Errorf("no .safetensors files in %s", dir)
	}

	stream, err := ks.NewStream()
	if err != nil {
		return nil, fmt.Errorf("create stream: %w", err)
	}

	jp := &JanusPro{
		stream:  stream,
		weights: make(map[string]*deviceTensor),
		config: janusConfig{
			vitHidden:      1024,
			vitLayers:      24,
			vitHeads:       16,
			vitPatch:       16,
			vitImageSize:   384,
			vitIntermediate: 4096,
			llmHidden:      4096,
			llmLayers:      30,
			llmHeads:       32,
			llmKVHeads:     32,
			llmIntermediate: 11008,
			llmVocab:       102400,
			llmMaxPos:      4096,
			ropeTheta:      10000.0,
			vqCodebook:     16384,
			vqLatentDim:    8,
			vqDownsample:   16,
			dtype:          ks.F16,
		},
	}

	// Load all safetensors files.
	// 1. Load plain file first for embedding/codebook weights (not transposed).
	for _, p := range plainPaths {
		if err := jp.loadSafetensorsFiltered(p, nil); err != nil {
			jp.Close()
			return nil, fmt.Errorf("load %s: %w", p, err)
		}
	}
	// 2. Load transposed file, overwriting non-embedding weights.
	embedKeys := map[string]bool{"embed_tokens.weight": true, "gen_embed.weight": true, "gen_vision_model.codebook.weight": true}
	for _, p := range transposedPaths {
		if err := jp.loadSafetensorsFiltered(p, embedKeys); err != nil {
			jp.Close()
			return nil, fmt.Errorf("load %s: %w", p, err)
		}
	}

	fmt.Printf("Loaded %d weights from %d safetensors file(s)\n", len(jp.weights), len(transposedPaths)+len(plainPaths))

	// Allocate KV cache.
	if err := jp.allocKVCache(); err != nil {
		jp.Close()
		return nil, fmt.Errorf("alloc kv cache: %w", err)
	}

	// Precompute RoPE cache.
	if err := jp.makeRopeCache(); err != nil {
		jp.Close()
		return nil, fmt.Errorf("rope cache: %w", err)
	}

	return jp, nil
}

func (jp *JanusPro) Close() {
	for _, t := range jp.weights {
		t.free()
	}
	for _, l := range jp.kvCache {
		l.k.free()
		l.v.free()
	}
	if jp.cos != nil {
		jp.cos.free()
	}
	if jp.sin != nil {
		jp.sin.free()
	}
	jp.stream.Destroy()
}

func (jp *JanusPro) allocKVCache() error {
	cfg := jp.config
	headDim := cfg.llmHidden / cfg.llmHeads
	for i := 0; i < cfg.llmLayers; i++ {
		k, err := jp.allocTensor(cfg.llmMaxPos, cfg.llmKVHeads, headDim)
		if err != nil {
			return err
		}
		v, err := jp.allocTensor(cfg.llmMaxPos, cfg.llmKVHeads, headDim)
		if err != nil {
			k.free()
			return err
		}
		jp.kvCache = append(jp.kvCache, kvCacheLayer{k: k, v: v})
	}
	return nil
}

func (jp *JanusPro) makeRopeCache() error {
	cfg := jp.config
	headDim := cfg.llmHidden / cfg.llmHeads
	maxPos := cfg.llmMaxPos
	halfDim := headDim / 2

	cosHost := make([]uint16, maxPos*halfDim)
	sinHost := make([]uint16, maxPos*halfDim)

	for pos := 0; pos < maxPos; pos++ {
		for i := 0; i < halfDim; i++ {
			theta := 1.0 / math.Pow(float64(cfg.ropeTheta), float64(2*i)/float64(headDim))
			freq := float32(float64(pos) * theta)
			cosHost[pos*halfDim+i] = float32ToFP16(cos32(freq))
			sinHost[pos*halfDim+i] = float32ToFP16(sin32(freq))
		}
	}

	cos, err := jp.hostToDevice(cosHost)
	if err != nil {
		return err
	}
	cos.shape = []int{maxPos, halfDim}
	jp.cos = cos

	sin, err := jp.hostToDevice(sinHost)
	if err != nil {
		return err
	}
	sin.shape = []int{maxPos, halfDim}
	jp.sin = sin

	return nil
}

// ---- safetensors loading ----

type safetensorsHeader map[string]struct {
	Dtype   string  `json:"dtype"`
	Shape   []int   `json:"shape"`
	Offsets []int64 `json:"data_offsets"`
}

func (jp *JanusPro) loadSafetensors(path string) error {
	return jp.loadSafetensorsFiltered(path, nil)
}

func (jp *JanusPro) loadSafetensorsFiltered(path string, skipKeys map[string]bool) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Read header length (8 bytes, little-endian u64).
	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		return fmt.Errorf("read header len: %w", err)
	}

	// Read header JSON.
	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return fmt.Errorf("read header: %w", err)
	}

	var header safetensorsHeader
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return fmt.Errorf("parse header: %w", err)
	}

	headerOffset := int64(8 + headerLen)

	for name, info := range header {
		if len(info.Offsets) != 2 {
			continue
		}
		// Skip keys already loaded from the plain file.
		if skipKeys != nil && skipKeys[name] {
			continue
		}
		start := headerOffset + info.Offsets[0]
		end := headerOffset + info.Offsets[1]
		dataLen := end - start

		// Allocate device memory.
		ptr, err := ks.MallocDevice(uintptr(dataLen))
		if err != nil {
			return fmt.Errorf("alloc %s: %w", name, err)
		}

		// Read data from file.
		data := make([]byte, dataLen)
		if _, err := f.ReadAt(data, start); err != nil {
			ks.FreeDevice(ptr)
			return fmt.Errorf("read %s: %w", name, err)
		}

		// Copy to device.
		if err := ks.CopyToDevice(ptr, data, jp.stream); err != nil {
			ks.FreeDevice(ptr)
			return fmt.Errorf("copy %s: %w", name, err)
		}

		jp.weights[name] = &deviceTensor{
			ptr:   ptr,
			shape: info.Shape,
			size:  int(dataLen),
		}
	}

	return jp.stream.Synchronize()
}

// ---- device memory helpers ----

func (jp *JanusPro) allocTensor(shape ...int) (*deviceTensor, error) {
	n := 1
	for _, s := range shape {
		n *= s
	}
	elemSize := 2 // fp16
	if jp.config.dtype == ks.F32 {
		elemSize = 4
	}
	bytes := n * elemSize
	ptr, err := ks.MallocDevice(uintptr(bytes))
	if err != nil {
		return nil, err
	}
	return &deviceTensor{ptr: ptr, shape: shape, size: bytes}, nil
}

func (jp *JanusPro) hostToDevice(data interface{}) (*deviceTensor, error) {
	var raw []byte
	var shape []int
	switch v := data.(type) {
	case []float32:
		shape = []int{len(v)}
		raw = make([]byte, len(v)*4)
		for i, f := range v {
			binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(f))
		}
	case []uint16:
		shape = []int{len(v)}
		raw = make([]byte, len(v)*2)
		for i, u := range v {
			binary.LittleEndian.PutUint16(raw[i*2:], u)
		}
	default:
		return nil, fmt.Errorf("unsupported host type")
	}
	ptr, err := ks.MallocDevice(uintptr(len(raw)))
	if err != nil {
		return nil, err
	}
	if err := ks.CopyToDevice(ptr, raw, jp.stream); err != nil {
		ks.FreeDevice(ptr)
		return nil, err
	}
	return &deviceTensor{ptr: ptr, shape: shape, size: len(raw)}, nil
}

func (jp *JanusPro) deviceToHost(t *deviceTensor) ([]byte, error) {
	out := make([]byte, t.size)
	if err := ks.CopyFromDevice(out, t.ptr, jp.stream); err != nil {
		return nil, err
	}
	if err := jp.stream.Synchronize(); err != nil {
		return nil, err
	}
	return out, nil
}

func (jp *JanusPro) getWeight(name string) (*deviceTensor, error) {
	w, ok := jp.weights[name]
	if !ok {
		// Try alternate naming conventions.
		for k, v := range jp.weights {
			if k == name || endsWith(k, "."+name) {
				return v, nil
			}
		}
		return nil, fmt.Errorf("weight not found: %s", name)
	}
	return w, nil
}

func endsWith(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}

// ---- math helpers ----

func cos32(x float32) float32 { return float32(math.Cos(float64(x))) }
func sin32(x float32) float32 { return float32(math.Sin(float64(x))) }

func float32ToFP16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff) - 127
	mant := uint16((bits >> 13) & 0x3ff)
	if exp > 15 {
		exp = 15
		mant = 0x3ff
	}
	if exp < -14 {
		exp = -14
		mant = 0
	}
	return sign | uint16((exp+15)<<10) | mant
}