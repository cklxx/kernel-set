# Contributing to kernel-set

Thanks for your interest in contributing. This project is a clean-room kernel
library with a frozen C ABI and four language bindings; a few conventions keep
independently authored pieces composing correctly. Please read
[`CONTRACT.md`](CONTRACT.md) before writing code — it is the authoritative spec.

## Project layout

```
include/kernel_set/*.h      Pure-C ABI — the single source of truth (no CUDA).
kernels/src/<category>/*.cu  Kernel implementations of the matching header.
kernels/src/common/*.cuh     Shared device utilities.
kernels/tests/test_*.cu      C++ correctness tests (one .cu per kernel category).
bindings/<lang>/             One FFI wrapper per language over the C ABI.
models/, providers/          Model→kernel mapping + auto-selection data/CLI.
benchmarks/                  Bench harnesses + recorded results.
```

## Ground rules

- **Never change a signature in `include/`.** The ABI headers are frozen
  contracts; implementations must match the existing declarations exactly.
- **Add a `.cu`, not a CMake edit.** `CMakeLists.txt` globs `kernels/src/**/*.cu`
  and `kernels/tests/test_*.cu`, so new kernels and tests need no build changes.
- **Accumulate in fp32**, dispatch dtypes via the `KS_DISPATCH_*` macros, and use
  the `KS_GLOBAL/KS_DEVICE/KS_DI` qualifiers — this is what keeps the HIP port a
  build flag. See `kernels/src/norm/rms_norm.cu` for the canonical style.
- **Keep facts in one place.** GPU caps / dtype aliases live in
  `models/gpu_caps.json`; don't reintroduce duplicate tables.

## Local checks (no GPU required)

All of these run in CI and can be run locally without a CUDA device:

```bash
python scripts/check_abi.py                       # every declared ks_* is defined
python scripts/gen_baselines.py --check           # baselines.yaml in sync
python scripts/gen_optimal.py --check             # optimal.json in sync + invariants
PYTHONPATH=bindings/python python -m pytest \
  bindings/python/tests/test_dispatch.py bindings/python/tests/test_optimal.py -q
python models/ksctl list                          # CLI smoke
```

## Building + testing with a GPU toolchain

```bash
cmake -B build -DKS_ENABLE_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
ctest --test-dir build --output-on-failure        # cases skip cleanly w/o a device
```

When you add a kernel, add a matching `kernels/tests/test_<category>.cu` that
compares the kernel against a plain host reference (mirror an existing test, e.g.
`test_norm.cu` / `test_gemm.cu`). Tests must compile for `nvcc` (CUDA 12.x) and
**skip** at runtime when no GPU is present.

## Pull requests

1. Fork and branch from `main`.
2. Keep the change focused; follow the file-ownership rules in `CONTRACT.md`
   (don't touch `include/`, `kernels/src/common/`, or `CMakeLists.txt` from a
   kernel change unless that *is* the change).
3. Ensure the no-GPU checks above pass.
4. Add a `CHANGELOG.md` entry under "Unreleased".
5. Open the PR using the template; describe what was verified and on which arch.

## License

By contributing you agree your contributions are licensed under
[Apache-2.0](LICENSE). Vendored third-party sources keep their own licenses
([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).
