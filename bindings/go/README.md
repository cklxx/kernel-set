# kernel-set — Go bindings (`kernelset`)

Idiomatic Go (cgo) bindings for [**kernel-set**](../../README.md), a collection of
high-performance CUDA/HIP kernels for LLM inference and training exposed through a
stable, pure-C ABI (`include/kernel_set/*.h`).

- **Module:** `github.com/kernel-set/go`
- **Package:** `kernelset`
- **Links against:** `libkernel_set.so` / `libkernel_set.dylib` / `kernel_set.dll`

Every kernel is wrapped by a Go function that takes device pointers as
`unsafe.Pointer`, an optional `Stream`, and returns an `error` built from the C
status code and the thread-local backend message (`ks_status_string` /
`ks_last_error_string`). The full ABI is covered: runtime, norm, activation,
attention, gemm, moe, rope, quant, sampling, embedding, elementwise, loss, and
optimizer.

## Install

```bash
go get github.com/kernel-set/go@latest
```

cgo and a C compiler are required (the package uses `import "C"`).

## Building & linking against the shared library

The package's cgo directives are:

```go
// #cgo CFLAGS: -I${SRCDIR}/../../include
// #cgo LDFLAGS: -lkernel_set
```

`${SRCDIR}/../../include` points at the in-tree headers when the bindings live
inside the `kernel-set` repo. When you consume the module from elsewhere (via
`go get`), point the compiler and linker at your installed library using the
standard cgo / loader environment variables:

```bash
# Headers (kernel_set/*.h) — needed at compile time:
export CGO_CFLAGS="-I/path/to/kernel-set/include"

# The .so/.dylib/.dll — needed at link time:
export CGO_LDFLAGS="-L/path/to/lib"

# The shared library — needed by the dynamic loader at run time:
export LD_LIBRARY_PATH=/path/to/lib      # Linux
export DYLD_LIBRARY_PATH=/path/to/lib    # macOS
# (on Windows, put kernel_set.dll on the PATH)
```

If you built kernel-set from source per the top-level README, the library is at
`build/libkernel_set.so`:

```bash
export KERNEL_SET_LIB=$PWD/build/libkernel_set.so
export CGO_LDFLAGS="-L$(dirname "$KERNEL_SET_LIB")"
export LD_LIBRARY_PATH="$(dirname "$KERNEL_SET_LIB")"
go build ./...
```

### Locating the library at run time

`kernelset.LibPath()` reports where the shared library is expected to resolve
from, for logging/diagnostics. Discovery order:

1. `$KERNEL_SET_LIB` — a file path, or a directory containing the library.
2. The platform loader search paths (`LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` /
   `PATH`) and common install prefixes (`/usr/local/lib`, `/usr/lib`, ...).

```go
if path, ok := kernelset.LibPath(); ok {
    log.Printf("kernel-set library: %s", path)
}
```

> Note: cgo links the library by name (`-lkernel_set`); the dynamic linker
> performs the real resolution at run time. `LibPath()` is a helper so your
> program can log/verify the expected location — it does not itself `dlopen`.

## Usage

Device pointers are raw addresses passed as `unsafe.Pointer`. You can either use
the built-in runtime helpers (`MallocDevice`/`Memcpy`/...) when you have no
tensor library, or pass pointers you already own from a GPU tensor framework
(see *Interop* below). The zero-value `Stream{}` (a.k.a. `kernelset.DefaultStream`)
is the default GPU stream (`NULL`).

```go
package main

import (
	"log"

	ks "github.com/kernel-set/go"
)

func main() {
	log.Printf("kernel-set %s (backend %s)", ks.Version(), ks.BackendName())

	if n, err := ks.DeviceCount(); err == nil && n > 0 {
		props, _ := ks.GetDeviceProperties(0)
		log.Printf("device 0: %s (sm_%d%d, bf16=%v, fp8=%v)",
			props.Name, props.ComputeMajor, props.ComputeMinor,
			props.SupportsBF16, props.SupportsFP8)
	}

	// Create a stream (or use ks.DefaultStream).
	stream, err := ks.NewStream()
	if err != nil {
		log.Fatal(err)
	}
	defer stream.Destroy()

	// RMSNorm a [rows, cols] bf16 activation tensor by a [cols] weight.
	const rows, cols = 4096, 4096
	const elemBytes = 2 // bf16

	x, err := ks.MallocDevice(uintptr(rows * cols * elemBytes))
	if err != nil {
		log.Fatal(err)
	}
	defer ks.FreeDevice(x)

	w, err := ks.MallocDevice(uintptr(cols * elemBytes))
	if err != nil {
		log.Fatal(err)
	}
	defer ks.FreeDevice(w)

	out, err := ks.MallocDevice(uintptr(rows * cols * elemBytes))
	if err != nil {
		log.Fatal(err)
	}
	defer ks.FreeDevice(out)

	// ... fill x and w on the device (e.g. via ks.CopyToDevice) ...

	if err := ks.RMSNorm(out, x, w, rows, cols, 1e-6, ks.BF16, stream); err != nil {
		log.Fatal(err) // e.g. "kernelset: ks_rms_norm: ... (KS_ERROR_...)"
	}
	if err := stream.Synchronize(); err != nil {
		log.Fatal(err)
	}
}
```

## Types & enums

The Go enum types mirror the C ABI exactly (same integer values):

| Go type      | C enum             | Values |
|--------------|--------------------|--------|
| `Status`     | `ks_status_t`      | `StatusSuccess`, `StatusInvalidArgument`, `StatusUnsupportedDtype`, `StatusUnsupportedShape`, `StatusCUDA`, `StatusNotImplemented`, `StatusOutOfMemory`, `StatusArchUnsupported`, `StatusInternal` |
| `Dtype`      | `ks_dtype_t`       | `F32`, `F16`, `BF16`, `F8E4M3`, `F8E5M2`, `F64`, `I64`, `I32`, `I8`, `U8`, `I4` |
| `Activation` | `ks_activation_t`  | `ActNone`, `ActReLU`, `ActGELU`, `ActGELUTanh`, `ActSiLU` |
| `QuantMode`  | `ks_quant_mode_t`  | `QuantPerTensor`, `QuantPerToken`, `QuantPerChannel`, `QuantGroupwise` |
| `MemcpyKind` | `ks_memcpy_kind_t` | `MemcpyHostToDevice`, `MemcpyDeviceToHost`, `MemcpyDeviceToDevice` |

`Dtype` exposes `.Name()` (e.g. `"bf16"`) and `.SizeBits()`; `Status` exposes
`.String()`. Errors returned by kernels are `*kernelset.Error`, which carries the
failing op name, the `Status`, and the backend message, and supports
`errors.Is` matching by status.

## Error handling

```go
err := ks.GEMM(c, a, b, m, n, k, false, false, k, n, n, 1.0, 0.0, ks.BF16, stream)
if err != nil {
	var kerr *ks.Error
	if errors.As(err, &kerr) {
		switch kerr.Status {
		case ks.StatusUnsupportedDtype:
			// fall back to a supported dtype
		case ks.StatusOutOfMemory:
			// shrink the batch
		}
	}
	log.Fatal(err)
}
```

## Interop: passing GPU pointers from a tensor library

kernel-set kernels operate directly on device memory you already own — you do
**not** have to allocate through `MallocDevice`. Any GPU buffer's device address
can be handed to a wrapper as an `unsafe.Pointer`, and any stream handle
(`cudaStream_t` / `hipStream_t`) can be wrapped as a `Stream`.

The ecosystem's Go tensor libraries expose device pointers and stream handles as
integer addresses (`uintptr`). Wrap them like so:

```go
// A device buffer address obtained from your tensor lib (e.g. tensor.DataPtr()
// returning a uintptr, or a CUDA driver allocation):
devPtr := unsafe.Pointer(uintptr(tensor.DataPtr()))

// A CUDA/HIP stream handle as an integer address (0 == default stream):
stream := ks.StreamFromUintptr(uintptr(streamHandle))
// or, if you already hold an unsafe.Pointer:
stream := ks.StreamFromPtr(rawStreamPtr)

ks.RMSNorm(outPtr, devPtr, wPtr, rows, cols, 1e-6, ks.BF16, stream)
```

Guidelines for safe interop:

- **Ownership stays with the tensor library.** kernel-set never frees buffers it
  did not allocate; only call `FreeDevice` on pointers from `MallocDevice`.
- **Match the dtype.** Pass the `Dtype` corresponding to the tensor's element
  type; element-size/stride are your responsibility (the ABI is shape-driven, not
  tensor-object-driven).
- **Lifetime.** Keep the source tensor alive (so the GC/allocator doesn't reclaim
  it) for the duration of the kernel launch; the device pointer is opaque to Go's
  GC.
- **Streams.** Pass the same stream your tensor library schedules work on to keep
  ordering correct, or `ks.DefaultStream` for the default stream. Use
  `stream.Synchronize()` before reading results back to the host.
- **Layout.** Tensors must be contiguous in the layout each header documents
  (row-major; `head_dim` innermost for attention/RoPE; see
  `include/kernel_set/*.h`).

When passing a Go-side `uintptr` from a tensor framework, route it through
`unsafe.Pointer(uintptr(...))` in a single expression as above — device addresses
are not Go-managed memory, so they are stable across GC.

## ABI coverage

Runtime: `Version`, `BackendName`, `DeviceCount`, `SetDevice`, `GetDevice`,
`GetDeviceProperties`, `MallocDevice`, `FreeDevice`, `Memcpy`, `MemsetDevice`,
`CopyToDevice`, `CopyFromDevice`, `NewStream`/`Stream.{Create,Destroy,Synchronize}`,
`LastErrorString`, `LibPath`.

| Category    | File             | Functions |
|-------------|------------------|-----------|
| Norm        | `norm.go`        | `RMSNorm`, `RMSNormResidual`, `LayerNorm`, `RMSNormBackward`, `LayerNormBackward` |
| Activation  | `activation.go`  | `SiLU`, `GELU`, `ReLU`, `SwiGLU`, `SwiGLUPacked`, `GeGLU`, `SwiGLUBackward` |
| Attention   | `attention.go`   | `FlashAttnVarlen`, `FlashAttn`, `PagedAttnDecode`, `ReshapeAndCache`, `MLADecode`, `FlashAttnBackward` |
| GEMM        | `gemm.go`        | `GEMM`, `GEMMBiasAct`, `GEMMBatched`, `GEMMW8A8`, `GEMMW4A16` |
| MoE         | `moe.go`         | `MoEGateSoftmaxTopK`, `MoEGateSigmoidGroupTopK`, `MoEComputePermutation`, `MoEPermute`, `MoEUnpermute`, `MoEGroupedGEMM` |
| RoPE        | `rope.go`        | `RoPEInplace`, `RoPE`, `RoPEGather`, `RoPEBackward` |
| Quant       | `quant.go`       | `QuantizeFP8`, `DequantizeFP8`, `QuantizeInt8`, `DequantizeInt8`, `DequantizeInt4` |
| Sampling    | `sampling.go`    | `Softmax`, `LogSoftmax`, `Argmax`, `Sample` |
| Embedding   | `embedding.go`   | `EmbeddingLookup`, `EmbeddingBackward` |
| Elementwise | `elementwise.go` | `Add`, `Mul`, `AddResidual`, `Scale`, `Cast`, `AXPBY` |
| Loss        | `loss.go`        | `CrossEntropy`, `FusedLinearCrossEntropy` |
| Optimizer   | `optimizer.go`   | `AdamW`, `SGDMomentum`, `GlobalGradNorm` |

Core types and helpers (`Status`, `Dtype`, `Activation`, `QuantMode`, `Stream`,
`Error`) live in `kernelset.go`; library discovery in `lib.go`.

## License

Same as the parent project — see the repository root [`LICENSE`](../../LICENSE).
