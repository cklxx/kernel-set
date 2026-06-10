# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- C++ correctness tests for GEMM (`ks_gemm`, `ks_gemm_bias_act`), RoPE (all four
  entry points incl. gather + backward), optimizers (`ks_adamw`,
  `ks_sgd_momentum`, `ks_global_grad_norm`), fused cross-entropy
  (`ks_cross_entropy`), and INT8 quant/dequant. Each compiles for `nvcc` and
  skips at runtime without a GPU.
- `ctest` step in the `build-cuda` CI matrix (`-DKS_ENABLE_TESTS=ON`).
- `Release` workflow: builds the multi-arch `libkernel_set.so`, vendors it into a
  Python wheel, and publishes to PyPI (Trusted Publishing) / crates.io / npm on a
  `vX.Y.Z` tag.
- Project governance files: `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, and GitHub issue/PR templates.

### Changed
- Unified the package version to `0.2.1` across `CMakeLists.txt`, the Python
  package, the Rust crate, and the npm package (previously drifted 0.1.0/0.2.1).

## [0.2.1]
- Python bindings and dispatch layer (baseline of this changelog).

[Unreleased]: https://github.com/cklxx/kernel-set/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/cklxx/kernel-set/releases/tag/v0.2.1
