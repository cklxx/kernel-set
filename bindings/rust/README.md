# kernel-set (Rust)

Idiomatic Rust bindings for **kernel-set**, a high-performance LLM inference &
training kernel library exposed as a frozen C ABI
(`include/kernel_set/*.h`, umbrella `kernel_set.h`). Covers the full ABI —
runtime, norm, activation, attention, gemm, moe, rope, quant, sampling,
embedding, elementwise, loss, optimizer.

The crate has two layers:

| Layer | Module | What you get |
|-------|--------|--------------|
| **Raw FFI** | [`kernel_set::sys`](src/sys.rs) | Hand-written `extern "C"` declarations + `#[repr(C)]` enums/structs, 1:1 with the headers. Everything `unsafe`, raw `*mut c_void` pointers. |
| **Safe API** | crate root ([`src/lib.rs`](src/lib.rs)) | `Result<(), kernel_set::Error>`-returning wrappers, a `Dtype`/`Activation`/`QuantMode` enum, `Stream`/`OwnedStream`, `DevicePtr`/`DevicePtrMut`/`DeviceBuffer` helpers. |

Device pointers are passed across the ABI as **raw integer addresses**
(`void*`); the library never dereferences them on the host. Streams are `void*`
(null = the default stream).

## Layout

```
bindings/rust/
├── Cargo.toml        # crate "kernel-set", libc dep, links = "kernel_set"
├── build.rs          # emits the link directive + search path / rpath
├── README.md         # this file
├── examples/
│   └── info.rs       # prints version/backend/device props
└── src/
    ├── sys.rs        # raw FFI (extern "C", repr(C) enums)
    └── lib.rs        # safe wrappers
```

## Building & locating the shared library

The crate **links** the system shared library `kernel_set`
(`libkernel_set.so` / `libkernel_set.dylib` / `kernel_set.dll`). Tell the build
where it lives with the **`KERNEL_SET_LIB`** environment variable — either a
directory or the full path to the shared object:

```sh
# A directory containing libkernel_set.{so,dylib} / kernel_set.dll:
export KERNEL_SET_LIB=/opt/kernel-set/lib
# ...or the full path to the file itself (parent dir is used):
export KERNEL_SET_LIB=/opt/kernel-set/lib/libkernel_set.so

cargo build
```

`build.rs` then:

1. adds `KERNEL_SET_LIB` to the linker search path
   (`cargo:rustc-link-search=native=…`),
2. bakes an **rpath** (Unix/macOS) so the produced binary loads the library at
   run time without needing `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`,
3. emits `cargo:rustc-link-lib=dylib=kernel_set`.

If `KERNEL_SET_LIB` is unset, the linker falls back to the standard system
search paths (`/usr/lib`, `/usr/local/lib`, `ldconfig` cache, etc.).

**Build knobs (env vars):**

| Variable | Effect |
|----------|--------|
| `KERNEL_SET_LIB` | Directory or full path to the shared lib (link + rpath). |
| `KERNEL_SET_NO_LINK` | Skip emitting the link directive (compile the `sys` types only, or when you `dlopen` the lib yourself). Used by `docs.rs`. |

On **Windows** there is no rpath: put `kernel_set.dll` on `PATH` or next to the
final `.exe`, and ensure the import lib (`kernel_set.lib`) is reachable via
`KERNEL_SET_LIB`.

## Usage

```rust,no_run
use kernel_set::{self as ks, Dtype, Stream};

fn main() -> Result<(), ks::Error> {
    println!("kernel-set {} ({} backend)", ks::version(), ks::backend_name());

    let rows = 8i64;
    let cols = 4096i64;
    let bytes = (rows * cols) as usize * Dtype::Bf16.size_bytes();

    // Allocate device buffers via the built-in runtime allocator (RAII).
    let x   = ks::DeviceBuffer::new(bytes)?;
    let w   = ks::DeviceBuffer::for_elems(cols as usize, Dtype::Bf16)?;
    let out = ks::DeviceBuffer::new(bytes)?;

    // ... fill x / w on the device (copy_from_host, another kernel, etc.) ...
    // let host: Vec<u16> = ...;  x.copy_from_host(&host, Stream::DEFAULT)?;

    // RMSNorm on the default stream.
    let stream = Stream::default(); // null == default stream
    ks::rms_norm(out.ptr_mut(), x.ptr(), w.ptr(), rows, cols, 1e-5, Dtype::Bf16, stream)?;
    ks::stream_synchronize(stream)?;

    Ok(())
}
```

Run the bundled example:

```sh
KERNEL_SET_LIB=/opt/kernel-set/lib cargo run --example info
```

### Conventions in the safe API

- Every kernel returns `Result<(), kernel_set::Error>`. On error, `Error`
  carries the typed `Status` (mapped from `ks_status_t`) **and** the backend's
  thread-local message (`ks_last_error_string`), captured at failure time.
- C `int` boolean flags (`causal`, `interleaved`, `tanh_approx`, `trans_a`,
  `targets_i64`, …) are exposed as Rust `bool`.
- Optional pointers (e.g. `bias`, `softmax_lse`, `master_param`,
  `correction_bias`) accept `DevicePtr::NULL` / `DevicePtrMut::NULL`.
- Pointer arguments take `impl AsDevicePtr` / `impl AsDevicePtrMut`, so you can
  pass a `DevicePtr`, a `DevicePtrMut`, a `DeviceBuffer` pointer, or a raw
  `usize` device address interchangeably.
- `ks_global_grad_norm` takes a `&[DevicePtr]` host slice (it is
  `#[repr(transparent)]` over `*const c_void`, so it is layout-compatible with
  the `const void* const*` the ABI expects).

## Obtaining device pointers

Three ways:

1. **Built-in allocator** — `DeviceBuffer::new(bytes)` /
   `DeviceBuffer::for_elems(count, dtype)`; frees on drop. Or the raw
   `malloc_device` / `free_device`.
2. **From a raw address you already hold** —
   `DevicePtr::from_addr(addr)` / `DevicePtrMut::from_raw(ptr)`.
3. **From an external GPU/tensor stack** — see Interop below.

## Interop with the ecosystem's tensor lib

The kernels take raw device addresses, so any tensor library that can hand you a
GPU pointer + a stream works. The crate intentionally has **no hard dependency**
on a GPU stack so the base crate builds without a CUDA toolchain; the optional
`cust` / `tch` features are documentation/feature-gate hooks you enable in *your*
crate.

### `tch` (libtorch / PyTorch tensors)

```toml
[dependencies]
kernel-set = { path = "…/bindings/rust", features = ["tch"] }
tch = "0.16"
```

```rust,ignore
use kernel_set::{self as ks, Dtype, DevicePtr, DevicePtrMut, Stream};

// A CUDA, contiguous bf16 tensor [rows, cols].
let x: tch::Tensor = /* ... .to_device(Device::Cuda(0)) ... */;
let w: tch::Tensor = /* ... */;
let out = x.empty_like();

// `data_ptr()` returns the raw device address as *mut c_void.
let xp = DevicePtr::from_raw(x.data_ptr() as *const _);
let wp = DevicePtr::from_raw(w.data_ptr() as *const _);
let op = DevicePtrMut::from_raw(out.data_ptr());

let (rows, cols) = (x.size()[0], x.size()[1]);
ks::rms_norm(op, xp, wp, rows, cols, 1e-5, Dtype::Bf16, Stream::DEFAULT)?;
// Sync libtorch's current CUDA stream before reading `out`, or pass that
// stream explicitly via `Stream::from_raw(...)`.
# Ok::<(), ks::Error>(())
```

Notes:
- Ensure the tensor is **contiguous** and on the expected CUDA device; match the
  `Dtype` to the tensor's dtype.
- To run on libtorch's stream rather than the default, obtain the current
  `cudaStream_t` and wrap it: `Stream::from_raw(stream as *mut c_void)`.

### `cust` (CUDA driver API)

```toml
[dependencies]
kernel-set = { path = "…/bindings/rust", features = ["cust"] }
cust = "0.3"
```

```rust,ignore
use kernel_set::{self as ks, DevicePtrMut, Stream};

let buf: cust::memory::DeviceBuffer<u16> = /* bf16 stored as u16 */;
let ptr = DevicePtrMut::from_addr(buf.as_device_ptr().as_raw() as usize);

// Wrap a cust stream: cust `Stream` -> raw `CUstream` -> *mut c_void.
let stream = ks::Stream::from_raw(/* cu_stream */ std::ptr::null_mut());
// ... call kernels with `ptr` and `stream` ...
let _ = (ptr, stream);
```

The exact accessor names differ across `cust` versions; the pattern is always
"get the integer/`CUdeviceptr` address, hand it to `DevicePtr*::from_addr`/`from_raw`,
and wrap the stream handle with `Stream::from_raw`."

## Regenerating `sys.rs` with bindgen (alternative)

`src/sys.rs` is **hand-written** for deterministic, review-friendly output and
zero build-time dependency on libclang. If you prefer generation, add bindgen
and replace the module body:

```toml
# Cargo.toml
[build-dependencies]
bindgen = "0.69"
```

```rust,ignore
// build.rs
let bindings = bindgen::Builder::default()
    .header("../../include/kernel_set/kernel_set.h")
    .clang_arg("-I../../include")
    .allowlist_function("ks_.*")
    .allowlist_type("ks_.*")
    .default_enum_style(bindgen::EnumVariation::Rust { non_exhaustive: false })
    .generate()
    .expect("bindgen");
bindings
    .write_to_file(std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("bindings.rs"))
    .unwrap();
```

```rust,ignore
// src/sys.rs
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
```

The hand-written version is the maintained default.

## Testing

Host-only unit tests (enum/status mapping, pointer helpers) compile and run
without a GPU or the linked library:

```sh
KERNEL_SET_NO_LINK=1 cargo test --lib
```

## License

Apache-2.0.
