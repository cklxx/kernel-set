# vllm

- upstream: https://github.com/vllm-project/vllm
- commit: 06ee2d8
- license: Apache-2.0
- vendored: 2026-06-04 (sparse: csrc)
- files: 288
- size: 4.8M
- lang: CUDA/C++

## Contents
Preserved the full csrc/ subtree: attention, moe, quantization (machete/marlin/w8a8/fp8/cutlass), core, cpu, rocm, quickreduce, cutlass_extensions, libtorch_stable (layernorm/activation/pos_encoding/cache kernels), and top-level custom_all_reduce / cumem_allocator / type_convert sources.

## Trims
csrc was 4.8MB after sparse-checkout (well under 120MB), so no kernel pruning was needed. Only non-source cruft removed: 2 .md docs (machete/Readme.md, w8a8/cutlass/Epilogues.md) and 2 .gitignore files. Kept all .cu/.cuh/.h/.hpp/.cpp/.inc/.inl/.py source. The .inl include (moe_permute_unpermute_kernel.inl) was explicitly preserved.

Note: in this commit the canonical CUDA layernorm/activation/pos_encoding/cache kernels reside under csrc/libtorch_stable/ (upstream restructure); CPU variants under csrc/cpu/.
