# kernel-set — Usage

This guide covers building the shared library, then a quickstart for each
language binding (Python, Rust, Go, TypeScript) with a real `ks_rms_norm` call,
and how to pass device pointers and streams across the ABI.

For the design behind all of this, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
Runnable starting points live in [`../examples/`](../examples/).

---

## 1. Build the library

kernel-set compiles to a single shared library that every binding loads at run
time: `libkernel_set.so` (Linux) / `libkernel_set.dylib` (macOS) /
`kernel_set.dll` (Windows). You need CMake ≥ 3.24 and the CUDA toolkit
(12.x).

```sh
# From the repo root.
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89   # 89=L4/4090, 80=A100, 90=H100
cmake --build build -j
# -> build/libkernel_set.so (or .dylib / .dll)
```

Useful options (all default-on for CUDA):

| Option | Default | Effect |
|--------|---------|--------|
| `-DCMAKE_CUDA_ARCHITECTURES=...` | `75;80;86;89;90` | Which SMs to target. Narrow it to your GPU for faster builds. |
| `-DKS_ENABLE_CUDA=ON` | ON | Build the CUDA backend. |
| `-DKS_ENABLE_HIP=ON -DKS_ENABLE_CUDA=OFF` | OFF | Build the HIP/ROCm backend (reserved). |
| `-DKS_FAST_MATH=ON` | OFF | `--use_fast_math` (lower-precision transcendentals). |

Adding a new `.cu` kernel requires **no** CMake change — sources are globbed
(`kernels/src/**/*.cu`). See `ARCHITECTURE.md` §4.

### Pointing the bindings at the library

Every binding finds the library through environment variables. The common one is
`KERNEL_SET_LIB` — set it to the directory containing the library *or* to the
full path of the file:

```sh
export KERNEL_SET_LIB="$PWD/build/libkernel_set.so"   # full path
# or a directory:
export KERNEL_SET_LIB="$PWD/build"
```

Per-language search order is detailed in each binding's README
(`bindings/<lang>/README.md`).

---

## 2. The call you'll see everywhere: `ks_rms_norm`

The C ABI signature (`include/kernel_set/norm.h`) is:

```c
ks_status_t ks_rms_norm(void* out, const void* input, const void* weight,
                        int64_t rows, int64_t cols, float eps,
                        ks_dtype_t dtype, ks_stream_t stream);
```

`input` is a row-major `[rows, cols]` device tensor, `weight` is `[cols]`, `out`
is `[rows, cols]`. `dtype` tags the element type of all three pointers; reductions
run in fp32 internally. Returns `KS_SUCCESS` (0) or a `ks_status_t` error.

Every binding below wraps exactly this.

---

## 3. Python (`kernel_set`, ctypes — torch-friendly)

Install the binding (no compile step; it `dlopen`s the prebuilt library):

```sh
pip install ./bindings/python          # or: pip install -e ./bindings/python
pip install "kernel_set[torch]"        # optional: ergonomic tensor paths
export KERNEL_SET_LIB="$PWD/build/libkernel_set.so"
```

### With torch (dtype, shape, and stream inferred from the tensors)

```python
import torch
import kernel_set as ks

print(ks.version(), ks.backend_name())          # "0.1.0" "cuda"

x   = torch.randn(8, 4096, device="cuda", dtype=torch.float16)
w   = torch.ones(4096,     device="cuda", dtype=torch.float16)
out = torch.empty_like(x)

ks.norm.rms_norm(out, x, w, eps=1e-6)           # rows/cols/dtype/stream inferred
torch.cuda.synchronize()
print(out.shape)
```

The wrapper pulls the device pointer from `tensor.data_ptr()`, infers the dtype
from `tensor.dtype`, and defaults the stream to
`torch.cuda.current_stream().cuda_stream` so launches order against your torch
work.

### Raw device pointers (no torch)

```python
import kernel_set as ks
from kernel_set import DType, MemcpyKind

rows, cols = 8, 4096
nbytes = rows * cols * 2                         # float16 = 2 bytes

x   = ks.runtime.malloc_device(nbytes)          # int device address
w   = ks.runtime.malloc_device(cols * 2)
out = ks.runtime.malloc_device(nbytes)
# ... ks.runtime.memcpy(x, host_buf, nbytes, MemcpyKind.HOST_TO_DEVICE) ...

ks.norm.rms_norm(out, x, w, rows=rows, cols=cols, eps=1e-6,
                 dtype=DType.F16, stream=0)      # 0 == default stream
ks.runtime.stream_synchronize(0)

for p in (x, w, out):
    ks.runtime.free_device(p)
```

Errors raise `ks.KernelSetError` (carries `.status`, `.status_name`,
`.backend_message`). Full surface in `bindings/python/README.md`.

---

## 4. Rust (`kernel-set` crate — safe wrappers)

The crate links the system shared library. Tell `build.rs` where it lives via
`KERNEL_SET_LIB` (a directory or the full path); it adds the link search path and
bakes an rpath so the binary loads the library at run time.

```toml
# Cargo.toml
[dependencies]
kernel-set = { path = "bindings/rust" }
```

```rust
use kernel_set::{self as ks, Dtype, Stream, DeviceBuffer};

fn main() -> Result<(), ks::Error> {
    println!("kernel-set {} ({} backend)", ks::version(), ks::backend_name());

    let (rows, cols) = (8i64, 4096i64);
    let bytes = (rows * cols) as usize * Dtype::Bf16.size_bytes();

    // RAII device buffers from the built-in allocator.
    let x   = DeviceBuffer::new(bytes)?;
    let w   = DeviceBuffer::for_elems(cols as usize, Dtype::Bf16)?;
    let out = DeviceBuffer::new(bytes)?;
    // ... fill x / w (copy_from_host, another kernel, etc.) ...

    let stream = Stream::DEFAULT;               // null == default stream
    ks::rms_norm(out.ptr_mut(), x.ptr(), w.ptr(), rows, cols, 1e-5,
                 Dtype::Bf16, stream)?;
    ks::stream_synchronize(stream)?;
    Ok(())
}
```

```sh
KERNEL_SET_LIB="$PWD/build" cargo run --example info   # bundled example
```

Every kernel returns `Result<(), kernel_set::Error>`; the `Error` carries the
typed `Status` and the backend's thread-local message. Pointer args accept
`DevicePtr`, `DevicePtrMut`, a `DeviceBuffer` pointer, or a raw `usize` address
interchangeably. Full details in `bindings/rust/README.md`.

---

## 5. Go (`kernelset` — cgo)

cgo and a C compiler are required. Point cgo at the headers and library:

```sh
export KERNEL_SET_LIB="$PWD/build/libkernel_set.so"
export CGO_CFLAGS="-I$PWD/include"
export CGO_LDFLAGS="-L$(dirname "$KERNEL_SET_LIB")"
export LD_LIBRARY_PATH="$(dirname "$KERNEL_SET_LIB")"   # DYLD_LIBRARY_PATH on macOS
```

```go
package main

import (
	"log"

	ks "github.com/kernel-set/go"
)

func main() {
	log.Printf("kernel-set %s (backend %s)", ks.Version(), ks.BackendName())

	stream, err := ks.NewStream()             // or ks.DefaultStream
	if err != nil {
		log.Fatal(err)
	}
	defer stream.Destroy()

	const rows, cols, elemBytes = 4096, 4096, 2 // bf16
	x, _ := ks.MallocDevice(uintptr(rows * cols * elemBytes))
	w, _ := ks.MallocDevice(uintptr(cols * elemBytes))
	out, _ := ks.MallocDevice(uintptr(rows * cols * elemBytes))
	defer ks.FreeDevice(x)
	defer ks.FreeDevice(w)
	defer ks.FreeDevice(out)
	// ... fill x and w on device (e.g. ks.CopyToDevice(x, hostBytes, stream)) ...

	if err := ks.RMSNorm(out, x, w, rows, cols, 1e-6, ks.BF16, stream); err != nil {
		log.Fatal(err)                        // *kernelset.Error, supports errors.As
	}
	if err := stream.Synchronize(); err != nil {
		log.Fatal(err)
	}
}
```

Device pointers are `unsafe.Pointer`; the zero-value `ks.Stream{}` /
`ks.DefaultStream` is the default stream. Errors are `*kernelset.Error` (op name
+ `Status` + backend message), matchable with `errors.Is`. Full details in
`bindings/go/README.md`.

---

## 6. TypeScript / Node (`kernel-set` — koffi FFI)

No native compile; `npm install` only fetches the prebuilt koffi addon, and the
binding `dlopen`s the kernel-set library at run time.

```sh
cd bindings/ts && npm install && npm run build
export KERNEL_SET_LIB="$PWD/../../build/libkernel_set.so"
```

```ts
import * as ks from 'kernel-set';

console.log(ks.version(), ks.backendName());       // "0.1.0" "cuda"

const stream = ks.streamCreate();
const rows = 16, cols = 4096, elemBytes = 2;       // bf16
const x = ks.mallocDevice(rows * cols * elemBytes); // bigint device address
const w = ks.mallocDevice(cols * elemBytes);
const y = ks.mallocDevice(rows * cols * elemBytes);
// ... ks.memcpy(x, hostBuf, rows*cols*elemBytes, ks.MemcpyKind.HostToDevice, stream) ...

ks.rmsNorm(y, x, w, rows, cols, 1e-5, ks.Dtype.BF16, stream);

ks.streamSynchronize(stream);
ks.freeDevice(x); ks.freeDevice(w); ks.freeDevice(y);
ks.streamDestroy(stream);
```

Device pointers are raw GPU addresses as `bigint` (recommended) or `number`;
`0`/`null` is the default stream. Errors throw `KernelSetError` (`.status`,
`.fn`, `.backendMessage`). Multi-argument kernels (attention, gemm, …) take a
typed options object. Full details in `bindings/ts/README.md`.

---

## 7. Passing device pointers and streams (all languages)

kernel-set **never copies your data** — it operates on raw device addresses you
already own. The contract is the same in every language:

- **Device pointers** are `void*` GPU addresses. Source them from:
  - the built-in runtime allocator (`malloc_device` / `DeviceBuffer` /
    `MallocDevice` / `mallocDevice`), **or**
  - your tensor library — PyTorch `tensor.data_ptr()`, a CuPy/Numba array, a
    `tch`/`cust` buffer, a CUDA driver allocation. The integer value of the
    `void*`/`CUdeviceptr` is the pointer.
- **Tensors must be contiguous** in the layout each header documents (row-major;
  `head_dim` innermost for attention/RoPE — see `include/kernel_set/*.h`). Call
  `.contiguous()` first if unsure. Python validates CUDA + contiguous and raises
  a clear error.
- **dtype must match** the buffer's element type — pass the corresponding
  `ks_dtype_t`. The ABI is shape-driven, so element size and strides are your
  responsibility.
- **Streams** are also just `void*` handles. Reuse the `cudaStream_t` your
  tensor library schedules on (pass its integer address) to keep ordering
  correct, or use the default stream (`NULL`/`0`). Always **synchronize the
  stream** before reading results back to the host.
- **Ownership** stays with whoever allocated the buffer. Only free pointers you
  obtained from kernel-set's own allocator. Keep externally-owned tensors alive
  for the duration of the launch.

| Concept | C ABI | Python | Rust | Go | TS |
|---------|-------|--------|------|----|----|
| device pointer | `void*` | `tensor.data_ptr()` or `int` | `DevicePtr` / `usize` | `unsafe.Pointer` | `bigint`/`number` |
| default stream | `NULL` | `0` / `None` | `Stream::DEFAULT` | `ks.DefaultStream` | `0` / `null` |
| wrap external stream | `(ks_stream_t)handle` | `stream=<int>` | `Stream::from_raw(p)` | `ks.StreamFromUintptr(addr)` | pass int address |
| status check | return value | raises `KernelSetError` | `Result<_, Error>` | `*kernelset.Error` | throws `KernelSetError` |
