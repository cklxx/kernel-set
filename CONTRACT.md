# kernel-set — implementation contract

Every kernel and binding in this repo follows this contract so that independently
authored pieces compose into one coherent library. Read it before adding code.

## Layered design

```
include/kernel_set/*.h      Pure-C ABI. The single source of truth. NO CUDA here.
kernels/src/common/*.cuh    Shared device utilities (platform, dtype, reduce, ...).
kernels/src/<category>/*.cu Kernel implementations of the matching header.
bindings/<lang>/            One FFI wrapper per language over the C ABI.
models/                     Model -> kernel mapping + auto-selection CLI.
```

The C ABI headers are frozen contracts. **Implementations must match the existing
declarations exactly** — do not change a signature in `include/`.

## C ABI conventions

- Every entry point returns `ks_status_t` (`KS_SUCCESS` == 0).
- Pointers are **device pointers** passed as `void*` (typed by `ks_dtype_t`).
- Streams are `ks_stream_t` (an opaque `void*` == `cudaStream_t`); `NULL` = default.
- Validate args with `KS_CHECK_PTR(p)` and shape checks returning
  `KS_ERROR_INVALID_ARGUMENT`; set messages via `KS_RETURN_ERROR(status, msg)`.
- After a launch call `KS_CHECK_LAUNCH();` then `return KS_SUCCESS;`.
- Wrap the public function bodies in `extern "C" { ... }`.

## Kernel style (mirror `kernels/src/norm/rms_norm.cu`)

- Put device code in `namespace ks { namespace <category> { ... } }`.
- **Accumulate in fp32** regardless of storage dtype. Convert with
  `ks::to_float(x)` / `ks::from_float<scalar_t>(v)` from `common/dtype.cuh`.
- Dispatch runtime dtype with `KS_DISPATCH_FLOATING_TYPES(dtype, "name", { ... })`
  (binds `scalar_t`) or `KS_DISPATCH_HALF_TYPES` for tensor-core-only paths.
- Use `ks::block_reduce_sum/max` (`common/reduce.cuh`) and vectorized IO
  (`common/vec.cuh`) where it helps bandwidth.
- Use warp primitives `ks::shfl_xor/down/...` (mask-correct on CUDA, HIP-safe).
- Use `KS_GLOBAL/KS_DEVICE/KS_DI` qualifiers and `KS_WARP_SIZE` — never raw
  `__global__`/`32`/`<cuda_runtime.h>`. This is what keeps the HIP port a flag.

## Includes inside `kernels/src`

`#include "kernel_set/<category>.h"` for the ABI, and
`#include "common/<x>.cuh"` for utilities (the `kernels/src` dir is on the
private include path).

## Build

CMake globs `kernels/src/**/*.cu`; **adding a `.cu` requires no CMake change.**
Never edit the top-level `CMakeLists.txt` from a kernel agent. Target arches
include L4 (sm_89) and A100 (sm_80) for Colab benchmarking.

## File ownership (no two agents touch the same file)

- A kernel agent writes only under its own `kernels/src/<category>/`.
- A binding agent writes only under its own `bindings/<lang>/`.
- `include/`, `kernels/src/common/`, `kernels/src/norm/`, `CMakeLists.txt`,
  and this file are owned by the orchestrator — do not modify them.

## Correctness without a local GPU

This repo is authored on a machine without CUDA. Code must be **compile-ready**
for `nvcc` (CUDA 12.x) targeting sm_80/89/90. Prefer clear, correct kernels with
the documented optimizations; note any further tuning in a comment.
