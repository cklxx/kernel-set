// kernel-set Go example: RMSNorm end to end.
//
// Allocates device buffers, uploads f32 input + weight, calls ks.RMSNorm
// (ks_rms_norm), copies the result back, and prints row 0.
//
// Build & run (cgo needs the headers + shared library):
//
//	cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j
//	export KERNEL_SET_LIB="$PWD/build/libkernel_set.so"
//	export CGO_CFLAGS="-I$PWD/include"
//	export CGO_LDFLAGS="-L$(dirname "$KERNEL_SET_LIB")"
//	export LD_LIBRARY_PATH="$(dirname "$KERNEL_SET_LIB")"   # DYLD_LIBRARY_PATH on macOS
//	go run ./examples/go
package main

import (
	"encoding/binary"
	"log"
	"math"

	ks "github.com/kernel-set/go"
)

const (
	rows = 4
	cols = 8
	eps  = float32(1e-5)
)

// f32sliceToBytes packs an []float32 into little-endian bytes for upload.
func f32sliceToBytes(xs []float32) []byte {
	b := make([]byte, 4*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

// bytesToF32slice unpacks little-endian bytes back into []float32.
func bytesToF32slice(b []byte) []float32 {
	xs := make([]float32, len(b)/4)
	for i := range xs {
		xs[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return xs
}

func main() {
	log.Printf("kernel-set %s (backend %s)", ks.Version(), ks.BackendName())
	if path, ok := ks.LibPath(); ok {
		log.Printf("library: %s", path)
	}

	stream, err := ks.NewStream()
	if err != nil {
		log.Fatal(err)
	}
	defer stream.Destroy()

	const n = rows * cols
	const elemBytes = 4 // f32

	// Host data: a ramp for x, all-ones weight.
	xHost := make([]float32, n)
	for i := range xHost {
		xHost[i] = float32(i) * 0.1
	}
	wHost := make([]float32, cols)
	for i := range wHost {
		wHost[i] = 1.0
	}

	x, err := ks.MallocDevice(uintptr(n * elemBytes))
	if err != nil {
		log.Fatal(err)
	}
	defer ks.FreeDevice(x)

	w, err := ks.MallocDevice(uintptr(cols * elemBytes))
	if err != nil {
		log.Fatal(err)
	}
	defer ks.FreeDevice(w)

	out, err := ks.MallocDevice(uintptr(n * elemBytes))
	if err != nil {
		log.Fatal(err)
	}
	defer ks.FreeDevice(out)

	if err := ks.CopyToDevice(x, f32sliceToBytes(xHost), stream); err != nil {
		log.Fatal(err)
	}
	if err := ks.CopyToDevice(w, f32sliceToBytes(wHost), stream); err != nil {
		log.Fatal(err)
	}

	// The kernel call.
	if err := ks.RMSNorm(out, x, w, rows, cols, eps, ks.F32, stream); err != nil {
		log.Fatal(err) // *kernelset.Error
	}
	if err := stream.Synchronize(); err != nil {
		log.Fatal(err)
	}

	outBytes := make([]byte, n*elemBytes)
	if err := ks.CopyFromDevice(outBytes, out, stream); err != nil {
		log.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		log.Fatal(err)
	}

	outF32 := bytesToF32slice(outBytes)
	log.Printf("rms_norm(out)[0] = %v", outF32[:cols])
}
