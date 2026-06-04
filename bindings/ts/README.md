# kernel-set — TypeScript / Node binding

Pure-FFI bindings (via [koffi](https://koffi.dev)) for **kernel-set**, the
high-performance LLM inference & training kernel library. No `node-gyp`, no
native compile step — at install time you only fetch the prebuilt `koffi`
addon; at runtime we `dlopen` the kernel-set shared library directly.

- `src/lib.ts` — the low-level layer. Locates/loads the shared library and
  declares **every** C ABI function (~72 entry points across runtime, norm,
  activation, attention, gemm, moe, rope, quant, sampling, embedding,
  elementwise, loss, optimizer) with exact koffi signatures.
- `src/index.ts` — the ergonomic layer. Typed wrappers that take device
  pointers as `bigint | number`, take the stream as a pointer (`0`/`null` =
  default), check the returned `ks_status_t`, and **throw `KernelSetError`** on
  any non-zero status. Plus `Dtype`, `Status`, `Activation`, `QuantMode`,
  `MemcpyKind` enums.

## Install & build

```bash
npm install          # pulls koffi (prebuilt) + typescript + @types/node
npm run build        # tsc -> dist/
```

Requires Node >= 16.

## Locating the shared library

At first use the binding loads `libkernel_set.{so,dylib}` / `kernel_set.dll`:

1. **`KERNEL_SET_LIB`** — if set and it points at an existing file, it is used
   verbatim. Otherwise it is treated as a library *name* and searched on the
   directories below (then handed to the OS loader).
2. **`KERNEL_SET_LIB_DIR`** — extra search directories (`PATH`-style separated).
3. Sibling build trees of this repo (`build/`, `build/lib`, `build/Release`, …).
4. Standard system paths (`/usr/local/lib`, `/usr/lib`, `/opt/homebrew/lib`, …;
   `%SystemRoot%\System32` on Windows).
5. The bare platform default name, letting the OS loader use
   `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` / `PATH` / rpath.

```bash
export KERNEL_SET_LIB=/path/to/build/libkernel_set.so
```

## Usage

```ts
import * as ks from 'kernel-set';

// --- introspection (no GPU work) -----------------------------------------
console.log(ks.version(), ks.backendName());      // "0.1.0" "cuda"
console.log(ks.dtypeName(ks.Dtype.BF16));         // "bf16"

if (ks.deviceCount() > 0) {
  ks.setDevice(0);
  const p = ks.getDeviceProperties(0);
  console.log(p.name, `sm${p.computeMajor}${p.computeMinor}`, p.supportsBf16);
}

// --- a kernel call -------------------------------------------------------
// Device pointers are raw GPU addresses (bigint recommended). Get them from a
// tensor lib (see "Interop") or from the runtime helpers below.
const stream = ks.streamCreate();

const rows = 16, cols = 4096, elemBytes = 2; // bf16
const x = ks.mallocDevice(rows * cols * elemBytes);
const w = ks.mallocDevice(cols * elemBytes);
const y = ks.mallocDevice(rows * cols * elemBytes);
// ... upload x, w via ks.memcpy(..., ks.MemcpyKind.HostToDevice, stream) ...

ks.rmsNorm(y, x, w, rows, cols, 1e-5, ks.Dtype.BF16, stream);

ks.streamSynchronize(stream);
ks.freeDevice(x); ks.freeDevice(w); ks.freeDevice(y);
ks.streamDestroy(stream);

// --- error handling ------------------------------------------------------
import { KernelSetError } from 'kernel-set';
try {
  ks.gemm({ c: y, a: x, b: w, m: 16, n: 16, k: 16, lda: 16, ldb: 16, ldc: 16,
            dtype: ks.Dtype.BF16 });
} catch (e) {
  if (e instanceof KernelSetError) {
    console.error('kernel failed:', e.status, e.fn, e.backendMessage);
  }
}
```

Multi-argument kernels (attention, gemm, moe, rope, quant, sampling, loss,
optimizer) take a single typed **options object** so call sites stay readable:

```ts
ks.flashAttn({
  out, q, k, v,
  batch: 1, seqlenQ: 1024, seqlenK: 1024,
  numHeads: 32, numKvHeads: 8, headDim: 128,
  causal: true, dtype: ks.Dtype.BF16, stream,
});
```

Simple elementwise/norm/activation kernels keep positional args.

## Interop — passing GPU pointers from a tensor library

kernel-set never copies your data; it operates on raw device addresses. Pass an
address as a `bigint` (full 64-bit, recommended) or a `number`.

- **PyTorch (via a Python sidecar / IPC):** `tensor.data_ptr()` returns the
  device address as a Python int — forward it to Node as a string and parse with
  `BigInt(addr)`.
- **Any CUDA allocation you already have:** the integer value of the
  `void*`/`CUdeviceptr` is the pointer. `BigInt('0x7f...')` it.
- **No tensor lib?** Use the built-in runtime helpers, which return addresses as
  `bigint`:

  ```ts
  const dptr = ks.mallocDevice(nBytes);                       // bigint address
  ks.memcpy(dptr, hostBuffer, nBytes, ks.MemcpyKind.HostToDevice);
  // ... kernels ...
  ks.memcpy(hostBuffer, dptr, nBytes, ks.MemcpyKind.DeviceToHost);
  ks.freeDevice(dptr);
  ```

Streams are also just pointers. Reuse a `cudaStream_t` from your tensor lib by
passing its integer address as the `stream` argument; `0` / `null` selects the
default stream. `ks.streamCreate()` / `ks.streamDestroy()` manage kernel-set's
own streams.

### Pointer & integer notes

- `int64_t` dimensions accept `number` (safe below 2^53) or `bigint`.
- Out-parameter helpers (`deviceCount`, `mallocDevice`, `streamCreate`,
  `getDeviceProperties`) return decoded JS values; you never marshal C structs
  by hand.
- `KernelSetError` carries `.status` (the `Status` enum value), `.fn` (the C
  function name), and `.backendMessage` (the backend's thread-local
  `ks_last_error_string()`).

## Advanced: raw FFI access

```ts
import { ffi } from 'kernel-set';        // every L.func handle, unwrapped
ffi.ks_rms_norm(out, x, w, 16n, 4096n, 1e-5, ks.Dtype.BF16, 0);
import { resolveLibraryPath } from 'kernel-set';
console.log(resolveLibraryPath());
```

## ABI coverage

runtime · norm · activation · attention · gemm · moe · rope · quant ·
sampling · embedding · elementwise · loss · optimizer — all entry points from
`include/kernel_set/*.h` are bound.
