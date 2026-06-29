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

type JanusPro struct {
	stream  ks.Stream
	weights map[string]*deviceTensor
	config  janusConfig

	kvCache []kvCacheLayer
	cos, sin *deviceTensor
}

type janusConfig struct {
	vitHidden       int
	vitLayers       int
	vitHeads        int
	vitPatch        int
	vitImageSize    int
	vitIntermediate int

	llmHidden       int
	llmLayers       int
	llmHeads        int
	llmKVHeads      int
	llmIntermediate int
	llmVocab        int
	llmMaxPos       int
	ropeTheta       float32

	vqCodebook  int
	vqLatentDim int
	vqDownsample int

	dtype ks.Dtype
}

type kvCacheLayer struct {
	k, v *deviceTensor
}

type deviceTensor struct {
	ptr   unsafe.Pointer
	shape []int
	size  int
}

func (t *deviceTensor) free() error {
	if t.ptr != nil {
		ptr := t.ptr
		t.ptr = nil
		return ks.FreeDevice(ptr)
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

func NewJanusPro(dir string) (*JanusPro, error) {
	paths, err := findSafetensors(dir)
	if err != nil {
		return nil, fmt.Errorf("find safetensors: %w", err)
	}

	var transposedPaths, plainPaths []string
	for _, p := range paths {
		if strings.HasSuffix(p, "_t.safetensors") {
			transposedPaths = append(transposedPaths, p)
		} else if strings.HasSuffix(p, ".safetensors") {
			plainPaths = append(plainPaths, p)
		}
	}

	if len(transposedPaths) == 0 {
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

	for _, p := range plainPaths {
		if err := jp.loadSafetensorsFiltered(p, nil); err != nil {
			jp.Close()
			return nil, fmt.Errorf("load %s: %w", p, err)
		}
	}
	embedSkip := func(name string, shape []int) bool {
		return strings.HasSuffix(name, "embed_tokens.weight") || strings.HasSuffix(name, "gen_embed.weight") || strings.HasSuffix(name, "gen_vision_model.codebook.weight")
	}
	for _, p := range transposedPaths {
		if err := jp.loadSafetensorsFiltered(p, embedSkip); err != nil {
			jp.Close()
			return nil, fmt.Errorf("load %s: %w", p, err)
		}
	}

	fmt.Printf("Loaded %d weights from %d safetensors file(s)\n", len(jp.weights), len(transposedPaths)+len(plainPaths))

	if err := jp.allocKVCache(); err != nil {
		jp.Close()
		return nil, fmt.Errorf("alloc kv cache: %w", err)
	}

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

type safetensorsHeader map[string]struct {
	Dtype   string  `json:"dtype"`
	Shape   []int   `json:"shape"`
	Offsets []int64 `json:"data_offsets"`
}

func (jp *JanusPro) loadSafetensorsFiltered(path string, skipFn func(name string, shape []int) bool) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		return fmt.Errorf("read header len: %w", err)
	}

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
		if skipFn != nil && skipFn(name, info.Shape) {
			continue
		}
		start := headerOffset + info.Offsets[0]
		end := headerOffset + info.Offsets[1]
		dataLen := end - start

		ptr, err := ks.MallocDevice(uintptr(dataLen))
		if err != nil {
			return fmt.Errorf("alloc %s: %w", name, err)
		}

		data := make([]byte, dataLen)
		if _, err := f.ReadAt(data, start); err != nil {
			ks.FreeDevice(ptr)
			return fmt.Errorf("read %s: %w", name, err)
		}

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

func (jp *JanusPro) allocTensor(shape ...int) (*deviceTensor, error) {
	n := 1
	for _, s := range shape {
		n *= s
	}
	elemSize := 2
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

// tensorPool holds reusable per-layer tensors to avoid per-step alloc/free.
type tensorPool struct {
	lnOut    *deviceTensor
	q        *deviceTensor
	k        *deviceTensor
	v        *deviceTensor
	attnOut  *deviceTensor
	projOut  *deviceTensor
	ln2Out   *deviceTensor
	gate     *deviceTensor
	up       *deviceTensor
	swiOut   *deviceTensor
	downOut  *deviceTensor
	cosSlice *deviceTensor
	sinSlice *deviceTensor
	allocMaxSeq int
}

func (jp *JanusPro) newPool(maxSeqLen int) (*tensorPool, error) {
	cfg := jp.config
	hidden := cfg.llmHidden
	intermediate := cfg.llmIntermediate
	headDim := hidden / cfg.llmHeads

	p := &tensorPool{allocMaxSeq: maxSeqLen}
	var err error

	p.lnOut, err = jp.allocTensor(maxSeqLen, hidden)
	if err != nil {
		return nil, err
	}
	p.q, err = jp.allocTensor(maxSeqLen, hidden)
	if err != nil {
		p.free(); return nil, err
	}
	p.k, err = jp.allocTensor(maxSeqLen, hidden)
	if err != nil {
		p.free(); return nil, err
	}
	p.v, err = jp.allocTensor(maxSeqLen, hidden)
	if err != nil {
		p.free(); return nil, err
	}
	p.attnOut, err = jp.allocTensor(maxSeqLen, hidden)
	if err != nil {
		p.free(); return nil, err
	}
	p.projOut, err = jp.allocTensor(maxSeqLen, hidden)
	if err != nil {
		p.free(); return nil, err
	}
	p.ln2Out, err = jp.allocTensor(maxSeqLen, hidden)
	if err != nil {
		p.free(); return nil, err
	}
	p.gate, err = jp.allocTensor(maxSeqLen, intermediate)
	if err != nil {
		p.free(); return nil, err
	}
	p.up, err = jp.allocTensor(maxSeqLen, intermediate)
	if err != nil {
		p.free(); return nil, err
	}
	p.swiOut, err = jp.allocTensor(maxSeqLen, intermediate)
	if err != nil {
		p.free(); return nil, err
	}
	p.downOut, err = jp.allocTensor(maxSeqLen, hidden)
	if err != nil {
		p.free(); return nil, err
	}
	p.cosSlice, err = jp.allocTensor(maxSeqLen, headDim/2)
	if err != nil {
		p.free(); return nil, err
	}
	p.sinSlice, err = jp.allocTensor(maxSeqLen, headDim/2)
	if err != nil {
		p.free(); return nil, err
	}
	return p, nil
}

func (p *tensorPool) free() {
	if p == nil {
		return
	}
	for _, t := range []*deviceTensor{p.lnOut, p.q, p.k, p.v, p.attnOut, p.projOut, p.ln2Out, p.gate, p.up, p.swiOut, p.downOut, p.cosSlice, p.sinSlice} {
		if t != nil {
			t.free()
		}
	}
}

func (jp *JanusPro) getWeight(name string) (*deviceTensor, error) {
	w, ok := jp.weights[name]
	if !ok {
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