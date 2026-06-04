# Third-Party Vendored Kernel Libraries

This directory vendors upstream GPU/accelerator kernel libraries that **kernel-set wraps**
behind its stable C ABI (`ks_*`). Each library lives under
`third_party/<category>/<lib>/` and ships with its own `LICENSE` and a `SOURCE.md`
describing exactly what was copied (sparse-checkout scope, upstream commit, drops).

These libraries are vendored **verbatim under their own upstream licenses**. kernel-set
does **not** fork or modify their kernels — it only provides thin adapter/wrapper code
that exposes them through the kernel-set operator ABI. For license attribution and
redistribution compliance see [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

- **Total vendored size:** ~72 MB across **21 libraries** in **11 categories**.
- **Cross-references:** the "kernel-set op(s)" column is derived from
  [`benchmarks/baselines.yaml`](../benchmarks/baselines.yaml) (`maps_to:` field) and
  [`docs/MODEL_KERNEL_MAP.md`](../docs/MODEL_KERNEL_MAP.md). (There is no
  `docs/OPERATOR_CATALOG.md` in this tree; the op catalog lives in `baselines.yaml` +
  the `providers/_frag_*.json` fragments.)
- **`# source files`** counts kernel/source files only
  (`.cu .cuh .h .hpp .cpp .cc .c .py .pyi .inc .inl .metal .mm`); it excludes
  `LICENSE`, `NOTICE`, `SOURCE.md`, and `THIRDPARTYNOTICES.txt`.

> **LICENSE check:** every vendored library below has a `LICENSE` file present. None
> are missing. (NOTICE files are present where upstream ships one: `flashinfer`,
> `bitsandbytes`, `liger-kernel`, and `tilelang`'s `THIRDPARTYNOTICES.txt`.)

---

## attention — `third_party/attention/`

| Lib | Language | Upstream | Commit | License | Size | # src | Backs kernel-set op(s) |
|-----|----------|----------|--------|---------|------|-------|------------------------|
| **flash-attention** | CUDA/C++ (.cu/.cuh/.h/.hpp/.cpp + codegen .py) | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `d80a771` | BSD-3-Clause | 4.3M | 675 | `ks_flash_attn`, `ks_flash_attn_varlen`, `ks_flash_attn_backward` |
| **flashinfer** | CUDA/C++ (.cu/.cuh/.h/.hpp/.cpp/.cc/.inc) | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | `e00c306` | Apache-2.0 | 11M | 618 | `ks_paged_attn_decode`, `ks_reshape_and_cache`, `ks_mla_decode`, `ks_gemm`, `ks_gemm_w8a8`, `ks_gemm_w4a16`, `ks_quantize_fp8`, `ks_moe_grouped_gemm`, `ks_moe_gate_softmax_topk`, `ks_moe_gate_sigmoid_group_topk`, `ks_swiglu`, `ks_sample` |
| **FlashMLA** | CUDA/C++ (.cu/.cuh/.h/.hpp/.cpp) | [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | `9241ae3` | MIT | 1.2M | 103 | `ks_mla_decode` (incl. DeepSeek sparse/NSA MLA decode) |
| **SageAttention** | CUDA/C++ (.cu/.cuh/.h/.cpp) | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) | `d1a57a5` | Apache-2.0 | 404K | 29 | `ks_flash_attn`, `ks_flash_attn_varlen` (INT8/FP8/FP4 quantized attention) |

*Notes:* flash-attention is a sparse checkout of `csrc` (FA2 sm80 in `csrc/flash_attn/src`, FA3 sm90 in `hopper/`; bundles AMD CK path, layer_norm, fused_dense_lib). `csrc/cutlass` and `csrc/composable_kernel` are upstream submodules — empty in the sparse checkout and dropped. flashinfer kept all `.cu/.cuh/.h/.hpp` under `csrc` and `include/flashinfer/`; `.jinja` codegen templates were dropped by the extension allowlist. FlashMLA covers sm90/sm100/smxx + `kerutils` headers; its `csrc/cutlass` submodule was empty and dropped.

## gemm — `third_party/gemm/`

| Lib | Language | Upstream | Commit | License | Size | # src | Backs kernel-set op(s) |
|-----|----------|----------|--------|---------|------|-------|------------------------|
| **DeepGEMM** | CUDA/C++/Python | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | `88965b0` | MIT | 1.3M | 115 | `ks_gemm`, `ks_gemm_w8a8`, `ks_moe_grouped_gemm` (block-scaled FP8/BF16 dense + grouped MoE GEMM) |
| **cutlass** | CUDA/C++ (header-only) | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | `2599f29` | BSD-3-Clause | 27M | 783 | `ks_gemm`, `ks_gemm_bias_act`, `ks_gemm_batched`, `ks_gemm_w8a8`, `ks_gemm_w4a16` |

*Notes:* DeepGEMM vendors `deep_gemm` + `csrc` (48 CUDA `.cuh` kernel headers under `deep_gemm/include/deep_gemm/`: ptx/wgmma, tcgen05, scheduler, mma, epilogue, layout; JIT/API wrappers in `csrc`). cutlass is header-only — full `include/cute` (289 `.hpp`) + `include/cutlass` (494 `.h`) kept (no trim needed under the 120 MB cap).

## quant — `third_party/quant/`

| Lib | Language | Upstream | Commit | License | Size | # src | Backs kernel-set op(s) |
|-----|----------|----------|--------|---------|------|-------|------------------------|
| **marlin** | CUDA/C++/Python | [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) | `1f25790` | Apache-2.0 | 64K | 3 | `ks_gemm_w4a16` (W4A16 Marlin GEMM) |
| **llm-awq** | CUDA/C++ | [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) | `d6e797a` | MIT | 472K | 40 | `ks_gemm_w4a16`, `ks_dequantize_int4` (AWQ W4A16 GEMM/GEMV + w8a8, layernorm, rope) |
| **exllamav2** | CUDA/C++ | [turboderp-org/exllamav2](https://github.com/turboderp-org/exllamav2) | `7dc12af` | MIT | 724K | 104 | `ks_gemm_w4a16`, `ks_dequantize_int4` (GPTQ/EXL2 W4A16) |
| **bitsandbytes** | CUDA/C++/Metal/ObjC++ | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | `3343bac` | MIT | 380K | 25 | `ks_quantize_int8`, `ks_dequantize_int8`, `ks_gemm_w8a8`, `ks_dequantize_int4` (LLM.int8 / NF4 / FP4, incl. Apple Metal/MPS backend) |

*Notes:* bitsandbytes vendors `csrc` incl. `gemm_4bit` (simt/sm75/sm80), CUDA `kernels.cu`/`ops.cu`, CPU/XPU ops, and the Apple `mps_kernels.metal` + `mps_ops.mm`. It ships its own `NOTICE.md`.

## cuda-kernels — `third_party/vllm/`

| Lib | Language | Upstream | Commit | License | Size | # src | Backs kernel-set op(s) |
|-----|----------|----------|--------|---------|------|-------|------------------------|
| **vllm** | CUDA/C++ | [vllm-project/vllm](https://github.com/vllm-project/vllm) | `06ee2d8` | Apache-2.0 | 4.8M | 287 | `ks_moe_gate_softmax_topk`, `ks_moe_gate_sigmoid_group_topk`, `ks_moe_compute_permutation`, `ks_moe_permute`, `ks_moe_unpermute`, `ks_moe_grouped_gemm`, `ks_gemm_w8a8`, `ks_gemm_w4a16`, `ks_paged_attn_decode`, `ks_reshape_and_cache`, `ks_rms_norm`, `ks_layer_norm`, `ks_rope`, plus activation/pos-encoding/cache kernels |

*Notes:* vllm is vendored directly at `third_party/vllm/` (not inside a `<category>/` dir). Sparse checkout of `csrc/`: attention, moe (topk/marlin/align/permute), quantization (machete/marlin/w8a8/fp8/cutlass/fused), core, cpu, rocm, quickreduce, cutlass_extensions, and `libtorch_stable` (canonical layernorm/activation/pos_encoding/cache kernels after an upstream restructure).

## ssm — `third_party/ssm/`

| Lib | Language | Upstream | Commit | License | Size | # src | Backs kernel-set op(s) |
|-----|----------|----------|--------|---------|------|-------|------------------------|
| **mamba** | CUDA/C++ | [state-spaces/mamba](https://github.com/state-spaces/mamba) | `6ff8ad1` | Apache-2.0 | 184K | 17 | *(no exact `ks_*` ABI op — selective-scan / SSM, attention substitute in hybrid models; benched as `mamba-ssm`)* |
| **causal-conv1d** | CUDA/C++ | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | `4f6ae4e` | BSD-3-Clause | 116K | 7 | *(no exact `ks_*` ABI op — fused short causal depthwise conv1d for Mamba/SSM blocks)* |

*Notes:* mamba vendors `csrc` selective-scan fwd/bwd kernels (fp16/bf16/fp32 real+complex). causal-conv1d vendors `csrc` fwd/bwd/update `.cu` kernels.

## linear_attn — `third_party/linear_attn/`

| Lib | Language | Upstream | Commit | License | Size | # src | Backs kernel-set op(s) |
|-----|----------|----------|--------|---------|------|-------|------------------------|
| **flash-linear-attention** | Python (Triton) | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | `7378dfe` | MIT | 2.6M | 230 | `ks_flash_attn` *(linear/SSM-attention substitute; no exact `ks_*` ABI op)* |

*Notes:* vendors `fla/ops/` Triton kernels: gla, gsa, hgrn, rwkv6/7, retention, delta_rule, gated_delta_rule, based, simple_gla, mesa_net, nsa, lightning_attn and ~35 other linear-attention op families.

## triton/tilelang/training — `third_party/triton/`, `third_party/tilelang/`, `third_party/training/`

| Lib | Language | Upstream | Commit | License | Size | # src | Backs kernel-set op(s) |
|-----|----------|----------|--------|---------|------|-------|------------------------|
| **liger-kernel** | Triton | [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | `94236ea` | BSD-2-Clause | 1.1M | 68 | `ks_rms_norm`, `ks_layer_norm`, `ks_rope`, `ks_swiglu`, `ks_geglu`, `ks_cross_entropy`, `ks_fused_linear_cross_entropy` |
| **tilelang** | TileLang DSL (.py/.pyi + 1 .cu/1 .cpp) | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | `550e25d` | MIT | 5.9M | 534 | `ks_gemm`, `ks_quantize_fp8`, `ks_dequantize_fp8` *(DSL + examples; GDN chunked / FP8 quant-cast backends)* |
| **cut-cross-entropy** | Triton | [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy) | `b7a0279` | Apple Sample Code License (custom permissive) | 184K | 24 | `ks_fused_linear_cross_entropy`, `ks_cross_entropy` |

*Notes:* liger-kernel vendors `src/liger_kernel/ops` (incl. `_ascend` and cutile backends) + its `NOTICE`. tilelang vendors `tilelang` core lib + DSL `examples` (incl. 1 `.cu` + 1 `.cpp` from examples/minference) + `THIRDPARTYNOTICES.txt`. cut-cross-entropy uses a **custom Apple permissive license** (not a standard SPDX id) that explicitly permits source redistribution with notice retention — its root `LICENSE` was re-fetched with `--no-cone` to materialize it.

## megakernel — `third_party/megakernel/`

| Lib | Language | Upstream | Commit | License | Size | # src | Backs kernel-set op(s) |
|-----|----------|----------|--------|---------|------|-------|------------------------|
| **thunderkittens** | CUDA C++ (tile DSL, header-only .cuh) | [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | `34b15f7` | MIT | 2.5M | 226 | `ks_flash_attn`, `ks_flash_attn_varlen`, `ks_gemm` (tile-DSL attn/gemm/mamba2/based/fftconv) |
| **mirage** | CUDA + C++ (Python) | [mirage-project/mirage](https://github.com/mirage-project/mirage) | `b293bb6` | Apache-2.0 | 3.9M | 338 | *(Mirage Persistent Kernel / MPK — compiles LLMs into a persistent megakernel; spans attn/gemm/moe-class ops, no single `ks_*` ABI op)* |
| **hazy-megakernels** | CUDA (.cu/.cuh) + Python | [HazyResearch/Megakernels](https://github.com/HazyResearch/Megakernels) | `7309cec` | MIT | 416K | 52 | *(low-latency LLM-inference megakernel; demos/low-latency-llama: attention_partial, rms_matvec_rope_append, llama, matvec_pipeline)* |

*Notes:* thunderkittens vendors `kernels/`, `include/`, `prototype/` (175 `.cu/.cuh` tile-DSL kernels + 51 `.py` harnesses). mirage is from the `mpk` branch (MPK lives there); vendors `src/` (kernel/cuda, transpiler, triton_transpiler, nki_transpiler, threadblock, search, base, utils) + `include/`. hazy-megakernels excludes its nested upstream ThunderKittens submodule (vendored separately above).

---

## sglang (sgl-kernel) — `third_party/sglang/`

| Lib | Language | Upstream | Commit | License | Size | # src | Backs kernel-set op(s) |
|-----|----------|----------|--------|---------|------|-------|------------------------|
| **sgl-kernel** | CUDA/C++ (.cu/.cuh/.h/.hpp/.cpp) + Python | [sgl-project/sglang](https://github.com/sgl-project/sglang) | `8e836e7` | Apache-2.0 | 3.9M | 291 | `ks_moe_gate_softmax_topk`, `ks_moe_gate_sigmoid_group_topk`, `ks_moe_compute_permutation`, `ks_moe_grouped_gemm`, `ks_moe_unpermute`, `ks_rmsnorm`, `ks_fused_add_rmsnorm`, `ks_gemma_rmsnorm`, `ks_rope`, `ks_silu_and_mul`, `ks_gemm_w8a8`/FP8, `ks_sample`, `ks_flash_attn`/`ks_flash_attn_varlen`, `ks_paged_attn_decode`, `ks_mla_decode` |

*Notes:* **The hard-op alignment target for kernel-set.** Vendored sparse from `sgl-project/sglang` (the `sgl-kernel/` subtree only); the exact Python API is in `sgl-kernel/python/sgl_kernel/`. sgl-kernel is wired as a first-class provider in `kernel_set.dispatch` and `benchmarks/bench_sota.py` — **rank #1** for the MoE gate ops (`topk_softmax`, `moe_fused_gate`) and grouped-MoE (its specialty), competitive for sampling, RMSNorm/fused-add/Gemma, RoPE, SiLU-mul, FP8/INT8 scaled-mm, and FA3 attention + FlashMLA.

---

## Tree summary

```
third_party/                                       ~72M total, 21 libs, 11 categories
├── attention/        4.3M+11M+1.2M+404K ≈ 17M     flash-attention, flashinfer, FlashMLA, SageAttention
├── gemm/             1.3M+27M           ≈ 28M     DeepGEMM, cutlass
├── quant/            64K+472K+724K+380K ≈ 1.6M    marlin, llm-awq, exllamav2, bitsandbytes
├── vllm/  (cuda-kernels, top-level)     ≈ 4.8M    vllm
├── ssm/              184K+116K          ≈ 300K    mamba, causal-conv1d
├── linear_attn/      2.6M               ≈ 2.6M    flash-linear-attention
├── triton/           1.1M               ≈ 1.1M    liger-kernel
├── tilelang/         5.9M               ≈ 5.9M    tilelang
├── training/         184K               ≈ 184K    cut-cross-entropy
├── megakernel/       2.5M+3.9M+416K     ≈ 6.8M    thunderkittens, mirage, hazy-megakernels
└── sglang/           3.9M               ≈ 3.9M    sgl-kernel (hard-op alignment target)
```

**LICENSE coverage:** 21/21 libraries have a `LICENSE` file — **none missing**.
