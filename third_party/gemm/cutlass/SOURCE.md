# cutlass

- upstream: https://github.com/NVIDIA/cutlass
- commit: 2599f29
- license: BSD-3-Clause
- vendored: 2026-06-04 (sparse: include/cute include/cutlass)
- files: 784

Note: header-only. Vendored full include/cute (CuTe) + include/cutlass (~27MB total, well under 120MB so no trim needed). Includes the complete cutlass/gemm kernel tree (collective, device, kernel, warp, threadblock). shallow+sparse, no .git.
