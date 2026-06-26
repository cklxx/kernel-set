// +build ignore

package main

import (
	"fmt"
	"os"
	"unsafe"

	ks "github.com/kernel-set/go"
)

func main() {
	// 1. Device info
	n, err := ks.DeviceCount()
	if err != nil {
		fmt.Fprintf(os.Stderr, "device count: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("devices: %d\n", n)

	props, err := ks.GetDeviceProperties(0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "props: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("GPU: %s (sm_%d%d)\n", props.Name, props.ComputeMajor, props.ComputeMinor)
	fmt.Printf("BF16: %v, FP8: %v, TF32: %v\n", props.SupportsBF16, props.SupportsFP8, props.SupportsTF32)

	// 2. Create stream
	stream, err := ks.NewStream()
	if err != nil {
		fmt.Fprintf(os.Stderr, "stream: %v\n", err)
		os.Exit(1)
	}
	defer stream.Destroy()

	// 3. Test RMSNorm (smoke test)
	const N = 4
	const D = 256
	bytes := N * D * 2 // fp16
	input, err := ks.MallocDevice(uintptr(bytes))
	if err != nil {
		fmt.Fprintf(os.Stderr, "malloc: %v\n", err)
		os.Exit(1)
	}
	defer ks.FreeDevice(input)

	weight, err := ks.MallocDevice(uintptr(D * 2))
	if err != nil {
		fmt.Fprintf(os.Stderr, "malloc w: %v\n", err)
		os.Exit(1)
	}
	defer ks.FreeDevice(weight)

	out, err := ks.MallocDevice(uintptr(bytes))
	if err != nil {
		fmt.Fprintf(os.Stderr, "malloc out: %v\n", err)
		os.Exit(1)
	}
	defer ks.FreeDevice(out)

	// Init with ones
	hostIn := make([]byte, bytes)
	for i := 0; i < N*D; i++ {
		hostIn[i*2] = 0x00 // 1.0 in fp16: 0x3C00
		hostIn[i*2+1] = 0x3C
	}
	hostW := make([]byte, D*2)
	for i := 0; i < D; i++ {
		hostW[i*2] = 0x00
		hostW[i*2+1] = 0x3C
	}

	ks.CopyToDevice(input, hostIn, stream)
	ks.CopyToDevice(weight, hostW, stream)

	err = ks.RMSNorm(out, input, weight, N, D, 1e-6, ks.F16, stream)
	if err != nil {
		fmt.Fprintf(os.Stderr, "rmsnorm: %v\n", err)
		os.Exit(1)
	}

	stream.Synchronize()

	// 4. Test Conv2D (our new vision kernel)
	const B = 1
	const C = 3
	const H = 16
	const W = 16
	const K = 8
	const R = 3
	const S = 3
	imgBytes := B * C * H * W * 2
	img, _ := ks.MallocDevice(uintptr(imgBytes))
	defer ks.FreeDevice(img)
	hostImg := make([]byte, imgBytes)
	for i := 0; i < len(hostImg); i += 2 {
		hostImg[i] = 0x00
		hostImg[i+1] = 0x3C
	}
	ks.CopyToDevice(img, hostImg, stream)

	wBytes := K * C * R * S * 2
	convW, _ := ks.MallocDevice(uintptr(wBytes))
	defer ks.FreeDevice(convW)
	hostCW := make([]byte, wBytes)
	for i := 0; i < len(hostCW); i += 2 {
		hostCW[i] = 0x00
		hostCW[i+1] = 0x3C
	}
	ks.CopyToDevice(convW, hostCW, stream)

	OH := H
	OW := W
	convOut, _ := ks.MallocDevice(uintptr(B * K * OH * OW * 2))
	defer ks.FreeDevice(convOut)

	err = ks.Conv2D(convOut, img, convW, nil,
		B, C, H, W, K, R, S, 1, 1, 1, 1, 1, 1, 1, ks.F16, stream)
	if err != nil {
		fmt.Fprintf(os.Stderr, "conv2d: %v\n", err)
		os.Exit(1)
	}
	stream.Synchronize()
	fmt.Println("conv2d: OK")

	// 5. Test GroupNorm
	gnOut, _ := ks.MallocDevice(uintptr(imgBytes))
	defer ks.FreeDevice(gnOut)
	err = ks.GroupNorm(gnOut, img, nil, nil, B, C, H*W, 1, 1e-6, ks.F16, stream)
	if err != nil {
		fmt.Fprintf(os.Stderr, "group_norm: %v\n", err)
		os.Exit(1)
	}
	stream.Synchronize()
	fmt.Println("group_norm: OK")

	// 6. Test GEMM
	const M = 16
	const NK = 64
	gemmOut, _ := ks.MallocDevice(uintptr(M * NK * 2))
	defer ks.FreeDevice(gemmOut)
	gemmA, _ := ks.MallocDevice(uintptr(M * NK * 2))
	defer ks.FreeDevice(gemmA)
	gemmB, _ := ks.MallocDevice(uintptr(NK * NK * 2))
	defer ks.FreeDevice(gemmB)
	hostA := make([]byte, M*NK*2)
	hostB := make([]byte, NK*NK*2)
	for i := 0; i < len(hostA); i += 2 {
		hostA[i] = 0x00
		hostA[i+1] = 0x3C
		hostB[i] = 0x00
		hostB[i+1] = 0x3C
	}
	ks.CopyToDevice(gemmA, hostA, stream)
	ks.CopyToDevice(gemmB, hostB, stream)

	err = ks.GEMM(gemmOut, gemmA, gemmB, M, NK, NK, false, false,
		NK, NK, NK, 1.0, 0.0, ks.F16, stream)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gemm: %v\n", err)
		os.Exit(1)
	}
	stream.Synchronize()
	fmt.Println("gemm: OK")

	// 7. Test FlashAttn
	const seqLen = 8
	const numHeads = 4
	const headDim = 64
	qBytes := 1 * seqLen * numHeads * headDim * 2
	q, _ := ks.MallocDevice(uintptr(qBytes))
	defer ks.FreeDevice(q)
	k, _ := ks.MallocDevice(uintptr(qBytes))
	defer ks.FreeDevice(k)
	v, _ := ks.MallocDevice(uintptr(qBytes))
	defer ks.FreeDevice(v)
	attnOut, _ := ks.MallocDevice(uintptr(qBytes))
	defer ks.FreeDevice(attnOut)

	hostQ := make([]byte, qBytes)
	for i := 0; i < len(hostQ); i += 2 {
		hostQ[i] = 0x00
		hostQ[i+1] = 0x3C
	}
	ks.CopyToDevice(q, hostQ, stream)
	ks.CopyToDevice(k, hostQ, stream)
	ks.CopyToDevice(v, hostQ, stream)

	err = ks.FlashAttn(attnOut, nil, q, k, v, 1, seqLen, seqLen, numHeads, numHeads, headDim,
		1.0/8.0, true, ks.F16, stream)
	if err != nil {
		fmt.Fprintf(os.Stderr, "flash_attn: %v\n", err)
		os.Exit(1)
	}
	stream.Synchronize()
	fmt.Println("flash_attn: OK")

	fmt.Println("\nAll kernel smoke tests passed on V100!")
}