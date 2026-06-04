// Command verify exercises the kernel-set Go (cgo) binding against the CPU stub.
//
// It confirms the package builds (cgo compiles every C.ks_* call against the
// real header, which is itself a signature check), links the shared library,
// resolves symbols, and that a ks_status_t round-trips correctly — all without
// a GPU. Device pointers are dummy/nil and never dereferenced by the stub.
//
// Build & run (from bindings/go):
//
//	CGO_CFLAGS="-I ../../include" \
//	CGO_LDFLAGS="-L ../../bindings/verify -lkernel_set" \
//	DYLD_LIBRARY_PATH=../../bindings/verify \
//	go run ./verify
package main

import (
	"fmt"
	"os"
	"unsafe"

	ks "github.com/kernel-set/go"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "[go] FAIL:", err)
		os.Exit(1)
	}
	fmt.Println("[go] PASS")
}

func run() error {
	// ---- introspection --------------------------------------------------
	if p, ok := ks.LibPath(); ok {
		fmt.Printf("[go] LibPath() = %s\n", p)
	} else {
		fmt.Println("[go] LibPath() = <not found via discovery> (linked at build time)")
	}

	v := ks.Version()
	fmt.Printf("[go] Version()      = %q\n", v)
	if v != "0.0.0-stub" {
		return fmt.Errorf("unexpected version: %q", v)
	}

	bn := ks.BackendName()
	fmt.Printf("[go] BackendName()  = %q\n", bn)
	if bn != "stub" {
		return fmt.Errorf("unexpected backend: %q", bn)
	}

	fmt.Printf("[go] BF16.Name()    = %q\n", ks.BF16.Name())
	fmt.Printf("[go] BF16.SizeBits()= %d\n", ks.BF16.SizeBits())
	if ks.BF16.SizeBits() != 16 {
		return fmt.Errorf("unexpected size bits: %d", ks.BF16.SizeBits())
	}
	fmt.Printf("[go] Status(SUCCESS).String() = %q\n", ks.StatusSuccess.String())
	fmt.Printf("[go] LastErrorString() = %q\n", ks.LastErrorString())

	// ---- device queries (out-params) ------------------------------------
	n, err := ks.DeviceCount()
	if err != nil {
		return fmt.Errorf("DeviceCount: %w", err)
	}
	fmt.Printf("[go] DeviceCount()  = %d\n", n)
	if n != 1 {
		return fmt.Errorf("stub should report 1 device, got %d", n)
	}

	if err := ks.SetDevice(0); err != nil {
		return fmt.Errorf("SetDevice: %w", err)
	}
	d, err := ks.GetDevice()
	if err != nil {
		return fmt.Errorf("GetDevice: %w", err)
	}
	fmt.Printf("[go] GetDevice()    = %d\n", d)

	props, err := ks.GetDeviceProperties(0)
	if err != nil {
		return fmt.Errorf("GetDeviceProperties: %w", err)
	}
	fmt.Printf("[go] device 0: name=%q warp=%d maxThreads=%d\n",
		props.Name, props.WarpSize, props.MaxThreadsPerBlock)
	if props.Name != "stub-device" {
		return fmt.Errorf("unexpected device name: %q", props.Name)
	}
	if props.WarpSize != 32 {
		return fmt.Errorf("unexpected warp size: %d", props.WarpSize)
	}

	// ---- streams --------------------------------------------------------
	s, err := ks.NewStream()
	if err != nil {
		return fmt.Errorf("NewStream: %w", err)
	}
	if err := s.Synchronize(); err != nil {
		return fmt.Errorf("Synchronize: %w", err)
	}
	if err := s.Destroy(); err != nil {
		return fmt.Errorf("Destroy: %w", err)
	}
	fmt.Println("[go] stream create/sync/destroy OK")

	// ---- op wrappers reaching the C call with dummy (nil) pointers -------
	// The stub never dereferences pointers, so nil is safe. These confirm cgo
	// marshalling + symbol resolution + ks_status_t handling.
	if err := ks.Add(nil, nil, nil, 16, ks.F16, ks.DefaultStream); err != nil {
		return fmt.Errorf("Add: %w", err)
	}
	fmt.Println("[go] Add(nil...) -> success")

	if err := ks.RMSNorm(nil, nil, nil, 2, 8, 1e-6, ks.F16, ks.DefaultStream); err != nil {
		return fmt.Errorf("RMSNorm: %w", err)
	}
	fmt.Println("[go] RMSNorm(nil...) -> success")

	if err := ks.GEMM(nil, nil, nil, 4, 4, 4, false, false, 4, 4, 4, 1.0, 0.0,
		ks.F16, ks.DefaultStream); err != nil {
		return fmt.Errorf("GEMM: %w", err)
	}
	fmt.Println("[go] GEMM(nil...) -> success")

	// Sample exercises uint64 seed/offset marshalling. Use the wrapper if it
	// matches; otherwise this still validates via the kernelset package.
	if err := ks.Softmax(nil, nil, 4, 32000, 1.0, ks.F32, ks.DefaultStream); err != nil {
		return fmt.Errorf("Softmax: %w", err)
	}
	fmt.Println("[go] Softmax(nil...) -> success")

	// Memcpy with nil + zero bytes (stub ignores).
	if err := ks.Memcpy(unsafe.Pointer(nil), unsafe.Pointer(nil), 0,
		ks.MemcpyHostToDevice, ks.DefaultStream); err != nil {
		return fmt.Errorf("Memcpy: %w", err)
	}
	fmt.Println("[go] Memcpy(nil, 0 bytes) -> success")

	return nil
}
