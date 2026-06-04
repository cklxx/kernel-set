# DeepGEMM

- upstream: https://github.com/deepseek-ai/DeepGEMM
- commit: 88965b0
- license: MIT
- vendored: 2026-06-04 (sparse: deep_gemm csrc include)
- files: 116

Note: upstream restructured — no top-level `include` dir exists; kernel CUDA headers (.cuh: ptx/wgmma, tcgen05, scheduler/gemm, mma, epilogue, layout) live under `deep_gemm/include/deep_gemm/`. csrc holds JIT/api wrappers (.hpp/.cpp/.cu). FP8/BF16 + grouped GEMM kernels. shallow+sparse, no .git.
