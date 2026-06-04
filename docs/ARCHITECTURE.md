# kernel-set — Architecture

kernel-set is a library of high-performance GPU kernels for LLM **inference and
training** (norm, activation, attention, gemm, MoE, RoPE, quant, sampling,
embedding, elementwise, loss, optimizer). It is built around one idea: a single,
frozen **pure-C ABI** that every language and every GPU backend agrees on, so
that kernels, language bindings, and the model-mapping layer can be authored
independently and still compose into one coherent library.

This document describes the layered design, the CUDA→HIP abstraction strategy,
the dtype/dispatch model, and the exact recipe for adding a new kernel. The
authoritative, normative rules live in [`CONTRACT.md`](../CONTRACT.md); this is
the explanatory companion to it.

---

## 1. Layered design

```
┌──────────────────────────────────────────────────────────────────────────┐
│ models/            Model → kernel mapping + GPU/dtype auto-selection (CLI) │
├──────────────────────────────────────────────────────────────────────────┤
│ bindings/<lang>/   One FFI wrapper per language over the C ABI             │
│   python (ctypes) · rust (bindgen-style) · go (cgo) · ts (koffi)           │
├──────────────────────────────────────────────────────────────────────────┤
│ include/kernel_set/*.h    Pure-C ABI — the single source of truth (NO CUDA)│
├──────────────────────────────────────────────────────────────────────────┤
│ kernels/src/<category>/*.cu   Kernel implementations of the matching header│
│ kernels/src/common/*.cuh      Shared device utilities (platform/dtype/...) │
└──────────────────────────────────────────────────────────────────────────┘
                         CMake → libkernel_set.{so,dylib,dll}
```

The layers are deliberately decoupled: the only contract that crosses a layer
boundary is the C ABI in `include/`. Everything above it (bindings, models)
depends *only* on those headers; everything below it (kernels, common) *implements*
them. No binding ever sees a CUDA type, and no kernel ever sees a Python/Go/Rust
type.

### 1.1 The C ABI (`include/kernel_set/`) — the contract

This is the heart of the project. The headers are **pure C**, free of any
CUDA/HIP includes, so they can be parsed by every FFI host (ctypes, cgo,
bindgen, koffi) on a machine that has no GPU toolchain at all.

| Header | Responsibility |
|--------|----------------|
| `types.h` | `ks_status_t`, `ks_dtype_t`, `ks_activation_t`, `ks_quant_mode_t` enums + opaque `ks_stream_t`. |
| `export.h` | `KS_API` visibility macro (dllexport/dllimport/`visibility("default")`) and `KS_BEGIN/END_EXTERN_C`. |
| `runtime.h` | Vendor-neutral device/stream/memory helpers + introspection (`ks_version`, `ks_backend_name`, device properties). |
| `norm.h`, `activation.h`, `attention.h`, `gemm.h`, `moe.h`, `rope.h`, `quant.h`, `sampling.h`, `embedding.h`, `elementwise.h`, `loss.h`, `optimizer.h` | One category each. |
| `kernel_set.h` | Umbrella include pulling in all of the above. |

The headers are **frozen contracts**. Implementations must match the existing
declarations exactly — a kernel author may not change a signature in `include/`.
`scripts/check_abi.py` verifies that the declared symbols and the compiled/bound
surface stay in sync.

ABI conventions (see `CONTRACT.md` §"C ABI conventions"):

- Every entry point returns `ks_status_t` (`KS_SUCCESS == 0`).
- Tensors are **device pointers** passed as `void*`, typed at runtime by a
  `ks_dtype_t` argument. The library never dereferences a pointer on the host.
- Streams are `ks_stream_t` — an opaque `void*` that is really a
  `cudaStream_t` / `hipStream_t`. `NULL` means the default stream.
- Shapes/strides are passed as explicit `int64_t`/`int` arguments; the ABI is
  **shape-driven, not tensor-object-driven**. This is what lets a Go `unsafe.Pointer`
  and a PyTorch `data_ptr()` be treated identically.
- A thread-local error string (`ks_last_error_string()`) carries the backend
  message behind a non-zero status.

### 1.2 Shared device utilities (`kernels/src/common/`)

Header-only `.cuh` utilities every kernel builds on. These are where the
portability and the house performance idioms live:

| File | Provides |
|------|----------|
| `platform.cuh` | Backend selection (CUDA vs HIP), function qualifiers (`KS_GLOBAL`, `KS_DEVICE`, `KS_DI`), `KS_WARP_SIZE`, host runtime aliases (`ks::gpuMalloc`, `ks::gpuStreamSynchronize`, …), and mask-correct warp primitives (`ks::shfl_xor/down/...`). |
| `dtype.cuh` | `ks_dtype_t` → concrete scalar type mapping, fp32↔low-precision conversions (`ks::to_float`, `ks::from_float<T>`), accumulator selection, and numeric limits. |
| `dispatch.cuh` | The `KS_DISPATCH_FLOATING_TYPES` / `KS_DISPATCH_HALF_TYPES` macros and the status/error plumbing macros (`KS_CHECK_PTR`, `KS_RETURN_ERROR`, `KS_CHECK_LAUNCH`). |
| `reduce.cuh` | `ks::block_reduce_sum` / `block_reduce_max` (warp-shuffle + shared-memory tree). |
| `vec.cuh` | 128-bit vectorized load/store helpers for bandwidth-bound kernels. |
| `runtime.cu` | The single `.cu` that implements `runtime.h` (device queries, streams, malloc/memcpy, version/backend strings, the thread-local error buffer). |

`kernels/src/common/`, `include/`, and `kernels/src/norm/` are **owned by the
orchestrator** — kernel authors do not modify them. They are the stable
foundation everyone else writes against.

### 1.3 Kernels (`kernels/src/<category>/`)

Each ABI category maps to a directory of `.cu` files plus an optional
`<category>_common.cuh` for category-private helpers (e.g.
`attention/attention_common.cuh`, `gemm/gemm_common.cuh`). A `.cu` file:

1. includes its ABI header (`#include "kernel_set/<category>.h"`) and the common
   utilities it needs;
2. puts device code in `namespace ks { namespace <category> { ... } }`;
3. exposes the public functions in an `extern "C" { ... }` block whose bodies
   validate args, dispatch the dtype, launch, and return `ks_status_t`.

`kernels/src/norm/rms_norm.cu` is the canonical reference; every other kernel
mirrors its structure.

### 1.4 Bindings (`bindings/<lang>/`)

One FFI wrapper per language, each a thin, idiomatic layer over the same C ABI:

| Language | Mechanism | Pointer type | Error model |
|----------|-----------|--------------|-------------|
| **Python** | `ctypes`, loads a prebuilt `.so` | `torch.Tensor` *or* raw int address (`data_ptr()`) | raises `KernelSetError` |
| **Rust** | hand-written `extern "C"` (`sys.rs`) + safe wrappers | `DevicePtr`/`DeviceBuffer`/`usize` | returns `Result<(), Error>` |
| **Go** | cgo (`import "C"`) | `unsafe.Pointer` | returns `*kernelset.Error` |
| **TypeScript** | `koffi` FFI, `dlopen`s the lib | `bigint`/`number` address | throws `KernelSetError` |

All four follow the same pattern: a low-level layer that mirrors the ABI 1:1,
plus an ergonomic layer that infers/checks dtype, handles the stream, checks the
returned status, and surfaces the backend message. None of them require a CUDA
toolchain to build the *bindings* themselves — they load/link a prebuilt library.

A binding author writes only under `bindings/<lang>/`; no two agents touch the
same file (see `CONTRACT.md` §"File ownership").

### 1.5 Models (`models/`)

The top layer maps a real model + GPU + dtype + mode (prefill/decode/train) to
the *strongest* ABI entry points and the dtype/quant scheme to feed them:

- `registry.json` / `registry.yaml` — machine-readable architecture facts and
  the default op→fn map per model (generated by `_gen_registry.py`).
- `select.py` — the auto-selection engine that layers GPU-capability and dtype
  rules on top of the registry (e.g. FP8 only on sm89+, bf16 needs sm80+, MLA
  models route decode to `ks_mla_decode`, MoE models add gate + grouped-GEMM).
- `ksctl` — the CLI front end.

This layer consumes the ABI as data; it does not link the library.

---

## 2. CUDA → HIP abstraction strategy

The design goal (CONTRACT.md): **porting to ROCm is a build-flag change, not a
source rewrite.** This is achieved with one rule and one header.

**The rule:** kernel code never uses raw CUDA spellings. No `#include
<cuda_runtime.h>`, no `__global__`, no literal `32` for the warp size, no
`__shfl_xor_sync`. Everything goes through `common/platform.cuh`.

**The header:** `platform.cuh` selects the backend and provides a uniform vocabulary.

1. **Backend selection.** `__HIPCC__` or `-DKS_PLATFORM_HIP` switches to the HIP
   include set (`hip/hip_runtime.h`, `hip_fp16.h`, `hip_bf16.h`); otherwise CUDA
   (`cuda_runtime.h`, `cuda_fp16.h`, `cuda_bf16.h`, and `cuda_fp8.h` when
   `__CUDA_ARCH__ >= 890`).

2. **Function qualifiers.** Kernels write `KS_GLOBAL` (→ `__global__`),
   `KS_DEVICE`, `KS_HOST_DEVICE`, `KS_DI` (device + forceinline) instead of the
   raw attributes, so the same tokens compile under both `nvcc` and `hipcc`.

3. **Warp width.** `KS_WARP_SIZE` is `32` on CUDA but resolves to `warpSize` on
   HIP — correctly handling the 64-wide wavefronts on CDNA GPUs. Kernels size
   their shared-memory reduction buffers with `kBlock / KS_WARP_SIZE`, so they
   adapt automatically.

4. **Warp primitives.** `ks::shfl_xor/shfl_down/shfl` wrap the *sync* variants
   with the full mask on CUDA (`__shfl_xor_sync(0xffffffff, …)`) and the
   plain variants on HIP. Kernels call the `ks::` version and stay mask-correct
   on both platforms.

5. **Host runtime aliases.** Host-side code (the `runtime.cu` implementation and
   any kernel that needs to allocate scratch) calls `ks::gpuMalloc`,
   `ks::gpuStreamSynchronize`, `ks::gpuMemcpyAsync`, etc., which alias the
   `cuda*` or `hip*` functions, and `ks::to_stream(void*)` to turn the opaque
   `ks_stream_t` into the backend stream type.

6. **Build wiring.** `CMakeLists.txt` has two branches behind `KS_ENABLE_CUDA`
   (default ON) and `KS_ENABLE_HIP`. The CUDA branch `enable_language(CUDA)` and
   targets `CMAKE_CUDA_ARCHITECTURES` (default `75;80;86;89;90` — T4, A100,
   A10/3090, L4/4090, H100). The HIP branch `enable_language(HIP)`, marks the
   globbed `.cu` files as `LANGUAGE HIP`, and defines `KS_PLATFORM_HIP=1`. Both
   branches glob `kernels/src/**/*.cu`, so **the same sources feed both
   backends**. The HIP path is wired but reserved; CUDA is the active backend
   today.

Because every CUDA-specific token is funneled through these abstractions, a HIP
build is `cmake -DKS_ENABLE_CUDA=OFF -DKS_ENABLE_HIP=ON` (plus, optionally,
`hipify` for any vendor library calls) rather than a source fork.

---

## 3. dtype & dispatch model

The ABI passes element types as a **runtime** `ks_dtype_t` enum; the kernels need
**compile-time** scalar types to instantiate templates. The dispatch macros
bridge the two, and the conversion helpers keep precision sane.

### 3.1 The dtype enum

`ks_dtype_t` (in `types.h`) spans the LLM numeric zoo: `F32`, `F16`, `BF16`,
`F8E4M3`, `F8E5M2`, `F64`, `I64`, `I32`, `I8`, `U8`, and packed `I4` (two per
byte). `ks_dtype_size_bits()` / `ks_dtype_name()` expose size and a short token
(`"bf16"`, `"i4"`, …) for every binding.

### 3.2 Runtime → compile-time dispatch

A kernel's C wrapper turns the runtime dtype into a bound `scalar_t` with one of
two macros from `dispatch.cuh`:

- `KS_DISPATCH_FLOATING_TYPES(dtype, "name", { ... })` — binds `scalar_t` to
  `float` / `__half` / `__nv_bfloat16` and runs the body. This is the default
  for most kernels.
- `KS_DISPATCH_HALF_TYPES(dtype, "name", { ... })` — `__half` / `__nv_bfloat16`
  only, for tensor-core-only paths.

Anything outside the supported set returns `KS_ERROR_UNSUPPORTED_DTYPE` with a
descriptive message. So the body is written **once** as a template and the macro
stamps out the per-dtype instantiations:

```cpp
KS_DISPATCH_FLOATING_TYPES(dtype, "ks_rms_norm", {
  norm::rms_norm_kernel<scalar_t><<<grid, block, 0, s>>>(
      static_cast<scalar_t*>(out), static_cast<const scalar_t*>(input),
      static_cast<const scalar_t*>(weight), cols, eps);
});
```

### 3.3 Accumulate in fp32

The numerical rule (CONTRACT.md): **accumulate in fp32 regardless of storage
dtype.** Kernels load a stored value with `ks::to_float(x)`, do all math in
`float` (reductions, softmax, normalization), and store back with
`ks::from_float<scalar_t>(v)`. `dtype.cuh` provides overloaded `to_float` for
each scalar type and a specialized `from_float<T>` with round-to-nearest-even.
`AccumType<T>` selects `double` for `double` inputs and `float` otherwise.
fp32-accumulated training gradients (`grad_weight_fp32`, `grad_table_fp32`, …)
are passed as explicit fp32 buffers in the ABI for stable accumulation across
rows.

### 3.4 Reductions & bandwidth

Row/block reductions go through `ks::block_reduce_sum/max` (`reduce.cuh`), which
combine a warp-shuffle reduction (via the portable `ks::shfl_*`) with a
shared-memory tree sized `kBlock / KS_WARP_SIZE`. Bandwidth-bound elementwise
and IO loops use the 128-bit vectorized loads/stores in `vec.cuh` when alignment
permits.

---

## 4. How to add a new kernel

This is the end-to-end recipe. It assumes the ABI declaration already exists in
`include/` (the headers are frozen and orchestrator-owned). If a brand-new entry
point is needed, the orchestrator adds the declaration first; a kernel author
then implements it.

**Reference to mirror:** `kernels/src/norm/rms_norm.cu`.

1. **Find the declaration.** Locate the `KS_API ks_status_t ks_<name>(...)` in
   the relevant `include/kernel_set/<category>.h`. Implement it *exactly* —
   do not alter the signature.

2. **Create the source file** under your category directory:
   `kernels/src/<category>/<name>.cu`. (A new category gets a new directory and,
   optionally, a `<category>_common.cuh` for shared device helpers.) **No CMake
   edit is required** — `CMakeLists.txt` globs `kernels/src/**/*.cu` with
   `CONFIGURE_DEPENDS`.

3. **Include what you use.** `#include "kernel_set/<category>.h"` for the ABI and
   `#include "common/<x>.cuh"` for utilities (`kernels/src` is on the private
   include path). Never include `<cuda_runtime.h>` directly.

4. **Write device code in the right namespace:**
   `namespace ks { namespace <category> { ... } }`. Use `KS_GLOBAL` for kernels
   and `KS_DEVICE`/`KS_DI` for helpers; never raw `__global__`/`__device__`.

5. **Follow the numeric rules.** Accumulate in fp32 via `ks::to_float` /
   `ks::from_float<scalar_t>`. Use `ks::block_reduce_sum/max` for reductions,
   `ks::shfl_*` for warp shuffles (so the HIP port stays a flag), `KS_WARP_SIZE`
   for warp width, and `vec.cuh` vectorized IO where it helps bandwidth.

6. **Write the C wrapper** in an `extern "C" { ... }` block:
   - validate every pointer with `KS_CHECK_PTR(p)`;
   - validate shapes/ranges, returning `KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "...")`;
   - get the stream with `auto s = to_stream(stream);`;
   - dispatch the dtype with `KS_DISPATCH_FLOATING_TYPES(dtype, "ks_<name>", { launch<scalar_t>...; })`;
   - call `KS_CHECK_LAUNCH();` then `return KS_SUCCESS;`.

7. **Build.** Configure once (so the glob re-runs) and build:
   ```sh
   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89   # e.g. L4
   cmake --build build -j
   ```
   The code must be **compile-ready for `nvcc` (CUDA 12.x)** targeting
   sm_80/89/90 — this repo is authored without a local GPU, so favor clear,
   correct kernels and note further tuning in a comment.

8. **Verify the ABI surface** with `scripts/check_abi.py`, and confirm every
   binding can reach the new symbol (the bindings already enumerate the full ABI;
   a new entry point is added in lockstep by the binding owners).

**What you must NOT touch** (orchestrator-owned): `include/`,
`kernels/src/common/`, `kernels/src/norm/`, the top-level `CMakeLists.txt`, and
`CONTRACT.md`.

---

## 5. Build & artifacts

A single shared library is produced: `libkernel_set.so` (Linux) /
`libkernel_set.dylib` (macOS) / `kernel_set.dll` (Windows). It exports only the
`ks_*` C symbols (the rest is hidden via `-fvisibility=hidden` +
`CXX_VISIBILITY_PRESET hidden`, with `KS_API` re-exporting the public surface).
All four bindings then locate this one artifact at run time via
`KERNEL_SET_LIB` (and friends). See [`USAGE.md`](USAGE.md) for building and
per-language quickstarts.
