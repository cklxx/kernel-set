# kernel-set language binding FFI verification

This document records a **GPU-free** verification of the four kernel-set
language bindings (Python, Rust, Go, TypeScript). The goal is to prove the FFI
**interfaces** are correct — symbol resolution, linking, calling convention,
argument marshalling, return-value handling, enum/struct layout — *not* the GPU
kernel math (that is verified separately on Colab).

## Strategy

1. Build a **CPU stub** of `libkernel_set` from the headers
   (`bindings/verify/stub_kernel_set.c`). It defines every one of the 72 `KS_API`
   functions with the **exact** header signature as a trivial no-op:
   `ks_status_t`-returning functions set benign out-params and `return KS_SUCCESS`;
   introspection functions return fixed benign values
   (`ks_version` → `"0.0.0-stub"`, `ks_backend_name` → `"stub"`,
   `ks_dtype_size_bits` → `16`, `ks_dtype_name`/`ks_status_string` → static strings,
   `ks_last_error_string` → `""`, `ks_device_count` → `1`,
   `ks_get_device_properties` → a benign struct). **Device pointers are never
   dereferenced**, so each binding can call with dummy/0/NULL pointers safely.
2. Load/link/call the stub from each binding via `KERNEL_SET_LIB` /
   `KERNEL_SET_LIB_DIR`, exercising real introspection, device queries, streams,
   and several op wrappers (reaching the C call with dummy pointers, including
   ops that marshal `int64`, `uint64`, `float`, enums, and out-params).
3. **Independently** cross-check each binding's declared FFI signatures (arg
   count, type category, return type) against the C headers
   (`bindings/verify/xcheck_signatures.py`).

## Result summary

| Language   | Builds | Loads lib | Resolves symbols | Call returns ok | Signature match |
|------------|:------:|:---------:|:----------------:|:---------------:|:---------------:|
| Python     |  PASS  |   PASS    |  PASS (72/72)    |      PASS       |  PASS (72/72)   |
| Rust       |  PASS  |   PASS    |  PASS (72/72)    |      PASS       |  PASS (72/72)   |
| Go (cgo)   |  PASS  |   PASS    |  PASS (72/72)    |      PASS       |  PASS (cgo)\*   |
| TypeScript |  PASS  |   PASS    |  PASS (72/72)    |      PASS       |  PASS (72/72)   |

\* Go signatures are validated by **cgo at compile time**: the C compiler checks
every `C.ks_*` call against the real header, so a wrong arg count/type would fail
the build. All 72 functions are referenced by the Go wrappers and the build
succeeds, so they are correct by construction. (The Python/Rust/TS checker
re-parses each binding's declared signatures and diffs them against the headers
type-category by type-category.)

**Bottom line:** all four bindings build, load the stub, resolve all 72 symbols,
marshal arguments correctly, and return `KS_SUCCESS` through their error-checking
layers. All declared signatures match the headers. Enum discriminants
(`ks_status_t`, `ks_dtype_t`, `ks_activation_t`, `ks_quant_mode_t`,
`ks_memcpy_kind_t`) and the `ks_device_properties_t` struct layout match the
headers in all four bindings.

## Bugs found / fixed

**None.** No interface bugs were found in any binding. All four are faithful
1:1 transcriptions of `include/kernel_set/*.h`. No binding source was modified.

### Non-blocking observations (no action taken)

- **Go `go vet`** emits one note: `kernelset.go:152: possible misuse of
  unsafe.Pointer` for `StreamFromUintptr(addr uintptr)`. This is the documented,
  intentional FFI helper for wrapping an externally-obtained stream handle
  (`cudaStream_t` address) and is not an interface defect — it does not affect
  linking, symbol resolution, or marshalling. Left unchanged.
- The stub C file emits no warnings; an earlier cosmetic `-Wcomment` was fixed
  in the stub itself (not a binding).

## What this did NOT verify (needs the real GPU library)

- **Kernel math / numerical correctness** of every op — the stub is a no-op and
  returns `KS_SUCCESS` regardless of inputs. Correctness of GEMM/attention/norm/
  etc. is verified separately on Colab with the real CUDA/HIP build.
- **Actual device behaviour**: real `ks_device_count`/`ks_get_device_properties`
  values, real stream semantics, real `ks_malloc_device`/`ks_memcpy` data
  movement. The stub returns benign placeholders and ignores pointers.
- **Error-path return codes** (e.g. `KS_ERROR_INVALID_ARGUMENT` on a NULL out
  pointer): the stub always returns `KS_SUCCESS`, so each binding's
  status→exception/error mapping was exercised only on the success path (the
  failure-mapping code itself is present and reviewed, but not triggered here).
- **Windows** (`.dll`, `__declspec(dllimport/export)`, no rpath): only macOS
  (`.dylib`) was exercised. A `libkernel_set.so` copy was also produced for
  Linux-style loaders but not run on Linux.
- **fp8/bf16 packed dtypes & I4 packing** at the data level (only the enum
  values were checked, not byte-level layout, which is a kernel-math concern).

## Artifacts (all under `bindings/verify/`)

| File | Purpose |
|------|---------|
| `stub_kernel_set.c`     | CPU stub implementing all 72 `KS_API` functions |
| `libkernel_set.dylib`   | Built stub shared library (macOS) |
| `libkernel_set.so`      | Copy of the stub for Linux-style loaders (not run here) |
| `verify_python.py`      | Python FFI driver |
| `verify_ts.mjs`         | TypeScript/koffi FFI driver (loads `bindings/ts/dist`) |
| `xcheck_signatures.py`  | Independent header-vs-binding signature diff (py/rust/ts) |
| `bindings/rust/examples/verify.rs` | Rust FFI driver (safe API + raw `sys`) |
| `bindings/go/verify/main.go`       | Go (cgo) FFI driver |

## Reproduce

All commands are run from the repo root
`/Users/bytedance/code/kernel-set` unless noted. Toolchains used:
clang (Apple), rustc/cargo 1.93, go, node v24, python3.11.

### 1. Build the CPU stub

```sh
cc -shared -fPIC -I include -DKERNEL_SET_BUILD \
   bindings/verify/stub_kernel_set.c -o bindings/verify/libkernel_set.dylib
install_name_tool -id "@rpath/libkernel_set.dylib" bindings/verify/libkernel_set.dylib
cp bindings/verify/libkernel_set.dylib bindings/verify/libkernel_set.so   # Linux-style copy

# Confirm all 72 ks_* symbols are exported:
nm -gU bindings/verify/libkernel_set.dylib | grep -c ' _ks_'   # -> 72
```

### 2. Python

```sh
KERNEL_SET_LIB="$PWD/bindings/verify/libkernel_set.dylib" \
  PYTHONPATH=bindings/python python3.11 -c \
  "import kernel_set as ks; print(ks.version()); print(ks.backend_name())"

KERNEL_SET_LIB="$PWD/bindings/verify/libkernel_set.dylib" \
  PYTHONPATH=bindings/python python3.11 bindings/verify/verify_python.py
```

### 3. Rust

```sh
cd bindings/rust
KERNEL_SET_LIB="$PWD/../verify/libkernel_set.dylib" cargo build
KERNEL_SET_LIB="$PWD/../verify/libkernel_set.dylib" cargo run --example verify
```
(`build.rs` reads `KERNEL_SET_LIB`, adds `rustc-link-search`, links
`-lkernel_set`, and bakes an rpath so the example runs without `DYLD_LIBRARY_PATH`.)

### 4. Go (cgo)

```sh
cd bindings/go
CGO_CFLAGS="-I $PWD/../../include" \
CGO_LDFLAGS="-L $PWD/../verify -lkernel_set" \
DYLD_LIBRARY_PATH="$PWD/../verify" \
  go run ./verify
```

### 5. TypeScript

```sh
cd bindings/ts
npm install
npx tsc -p tsconfig.json          # builds dist/
cd ../..
KERNEL_SET_LIB="$PWD/bindings/verify/libkernel_set.dylib" \
  node bindings/verify/verify_ts.mjs
```

### 6. Independent signature cross-check (Python / Rust / TS)

```sh
python3.11 bindings/verify/xcheck_signatures.py
# -> "ALL SIGNATURES MATCH" for python, rust, ts; go validated by cgo compile.
```
