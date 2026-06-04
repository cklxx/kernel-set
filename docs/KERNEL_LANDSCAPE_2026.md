# Kernel Landscape — 2026 H1 High-Performance Kernel Libraries

A freshly-researched catalog of the **2026 H1** SOTA kernel libraries that
`kernel-set` benchmarks against. This is the provenance + sourcing reference for
[`benchmarks/baselines.yaml`](../benchmarks/baselines.yaml): the machine-readable
config the bench uses to install real upstream kernels and compare them, op-for-op,
against kernel-set's clean-room implementations.

For the standing, slower-moving survey (best-in-class per op by arch/dtype, model
family classification) see [`KERNEL_LANDSCAPE.md`](KERNEL_LANDSCAPE.md). **This**
document is the "what's new and fast right now" layer on top of it.

Only libraries with **high / medium confidence** are included. As of **2026-06-04**.

## GPU / arch shorthand

| Tag | GPU | Arch | Notes |
|---|---|---|---|
| **sm75** | T4 | Turing | FP16 MMA only (no BF16/FP8) — Colab target |
| **sm80** | A100 / A30 | Ampere | FP16/BF16/TF32 MMA, INT8; **no FP8 tensor cores** |
| **sm89** | L4 / RTX-4090 | Ada | adds FP8 (e4m3/e5m2) tensor cores, INT8/INT4 |
| **sm90(a)** | H100 / H200 / H800 | **Hopper** | WGMMA + TMA, FP8 tensor cores, warpgroup pipelines |
| **sm100/sm103** | B200 / GB200 / GB300 | **Blackwell (datacenter)** | 5th-gen tensor cores (tcgen05/TMEM), FP8 + **FP4 (nvfp4/mxfp4)** |
| **sm120/sm121** | RTX 50 / RTX PRO 6000 / DGX Spark GB10 | **Blackwell (consumer/Thor)** | consumer FP4 tensor cores |

> **Blackwell-only** = needs sm100+ (and sometimes only sm100, not consumer sm120).
> **Hopper-only** = needs sm90 (WGMMA/TMA); will not build/run on sm80/sm89.
> Both are called out per-library below and flagged in `baselines.yaml`.

---

## 1. Attention / SSM

### FlashAttention-4 (FA4) — `Dao-AILab/flash-attention`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** `fa4-v4.0.0.beta16` (latest beta as of 2026-06-02); FA4 blog 2026-03-05.
- **Ops:** dense prefill `flash_attn` (fwd + bwd), `flash_attn_varlen`, causal/non-causal,
  MLA forward (sparse top-k, sm100), head_dim up to 256.
- **Hardware:** **Blackwell sm100 (B200) primary**; Hopper sm90 path; ROCm/Triton path for AMD. CUDA 12.8+/12.9+ (built with up to CUDA 13.2).
- **dtypes:** BF16, FP16; FP8 (e4m3/e5m2) paths in betas (beta10+).
- **License:** BSD-3-Clause.
- **Install:** written in CuTe-DSL (CUTLASS Python kernel DSL). Install from FA4 beta wheels / build from source; FA2/FA3 still via `pip install flash-attn`.
- **Claimed perf:** up to ~1605–1613 TFLOP/s BF16 on B200 (~71% util); 1.1–1.3× vs cuDNN 9.13 fwd; 2.1–2.7× vs Triton on B200 (head_dim=128).
- **Baselines:** `ks_flash_attn` / `ks_flash_attn_varlen` (prefill fwd), `ks_flash_attn_backward` (training bwd). Best-in-class attention prefill/training on **Blackwell**.
- **Source:** <https://tridao.me/blog/2026/flash4/>

### FlashAttention-3 (FA3) — `Dao-AILab/flash-attention`  ⟶ confidence: **high**
- **Version / date:** FA3 (Hopper); shipped in the flash-attn 2.x/3.x line, maintained into 2026 (superseded by FA4 betas for Blackwell). *(not a 2026 H1 release; included as the standing sm90 baseline.)*
- **Ops:** dense prefill `flash_attn` (fwd + bwd), `flash_attn_varlen`, GQA/MQA, causal, FP8 prefill.
- **Hardware:** **Hopper sm90 (H100/H200) only.**
- **dtypes:** FP16, BF16, FP8 (e4m3 with block-quant + incoherent processing).
- **License:** BSD-3-Clause.
- **Install:** `pip install flash-attn` (build from source for the FA3/Hopper kernels).
- **Claimed perf:** 1.5–2.0× over FA2 in FP16 (~740 TFLOP/s, ~75% H100 peak); FP8 ~1.2 PFLOP/s with ~2.6× lower error than naive FP8.
- **Baselines:** `ks_flash_attn` / `ks_flash_attn_varlen` (prefill fwd), `ks_flash_attn_backward` (bwd). The **sm90** best-in-class baseline (FA2 remains baseline for sm80/sm89).
- **Source:** <https://github.com/Dao-AILab/flash-attention/releases>

### FlashInfer — `flashinfer-ai/flashinfer`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** v0.6.12 (2026-05-29, current stable; nightlies through 2026-06-04). Blackwell support landed in v0.4.0 (2025-10-08).
- **Ops:** paged KV-cache decode, ragged/prefill attention, MLA decode+prefill (matrix absorption), DeepSeek V4 sparse MLA (trtllm-gen), Kimi K2.5 H64 CuTe-DSL MLA decode, CUTLASS MLA paged attention (FP8 out), trtllm-gen GQA decode, FP8 KV-cache, prefix-cascade, RMSNorm/RoPE/SiLU-mul, sampling (top-k/top-p/min-p), Mamba SSU kernel.
- **Hardware:** sm80 (A100) → sm90 (Hopper) → sm100/sm103 (Blackwell, auto trtllm-gen) → sm120/sm12x (consumer Blackwell, b12x kernels). **Broadest arch coverage of any baseline here.**
- **dtypes:** FP16, BF16, FP8 (e4m3/e5m2) + FP8 KV-cache; NVFP4/MXFP4 (Blackwell); per-token NVFP4 quant.
- **License:** Apache-2.0.
- **Install:** `pip install flashinfer-python` (AOT wheels incl. SM120 fmha_v2; JIT path available).
- **Claimed perf:** 28–30% latency reduction for long-context inference (MLSys'25 best paper); 3-stage plan/run API for CUDA-graph serving. NVIDIA-backed; sources newest TRT-LLM kernels.
- **Baselines:** `ks_paged_attn_decode` (paged decode), `ks_reshape_and_cache` (KV append), `ks_mla_decode` (MLA decode). The **portable** best-in-class decode + MLA baseline across all archs; backend for vLLM/SGLang. *(Also a GEMM/MoE baseline — see §2/§3.)*
- **Source:** <https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.12>

### FlashMLA — `deepseek-ai/FlashMLA`  ⟶ confidence: **medium**, 2026 H1
- **Version / date:** updated through 2026 H1 (Blackwell sm100 + DeepSeek V4 support landed ~Jan–Feb 2026).
- **Ops:** MLA decode (paged, varlen), DeepSeek V3.2/V4 sparse attention kernel, 512-dim attention heads.
- **Hardware:** Hopper sm90 (H800/H100) originally; **sm100 (Blackwell) added 2026**.
- **dtypes:** BF16; FP8 (KV-cache, with V3.2/V4 FP8 format changes).
- **License:** MIT.
- **Install:** build from source (`pip install -v .`).
- **Claimed perf:** very high memory-bandwidth utilization on H800/H100 for DeepSeek MLA decode; restructured KV-cache layout + new sparsity handling for V4.
- **Baselines:** `ks_mla_decode` (DeepSeek MLA decode). The Hopper/Blackwell best-in-class MLA baseline (vs FlashInfer MLA for portability).
- **Source:** <https://github.com/deepseek-ai/FlashMLA>

### SageAttention3 — `thu-ml/SageAttention`  ⟶ confidence: **high**
- **Version / date:** SageAttention3 code released 2025-09-27 (`sageattention3_blackwell`); NeurIPS 2025 Spotlight; arXiv:2505.11594. *(not strictly 2026 H1; current best FP4-attention baseline on Blackwell.)*
- **Ops:** quantized attention (prefill/inference fwd), microscaling FP4 attention, 8-bit attention training (explored).
- **Hardware:** **Blackwell** (RTX 5090 / sm120, B200 sm100) FP4 tensor cores.
- **dtypes:** microscaling FP4 (mxfp4) for QK and PV; FP16/FP32 accumulate.
- **License:** Apache-2.0 (BSD-style components).
- **Install:** `git clone` + `cd sageattention3_blackwell` + `python setup.py install` (build from source).
- **Claimed perf:** 1038 TOPS on RTX 5090, ~5× over fastest FlashAttention on RTX 5090; FP4-core inference acceleration.
- **Baselines:** `ks_flash_attn` / `ks_flash_attn_varlen` (prefill fwd) — the **FP4 quantized attention** variant on Blackwell.
- **Source:** <https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell>

### SageAttention2++ — `thu-ml/SageAttention`  ⟶ confidence: **high**
- **Version / date:** SageAttention2++ code released 2025-07; arXiv:2505.21136; SageAttention2 (ICML 2025). *(not 2026 H1; current best INT8/FP8-attention baseline on Ada/Hopper.)*
- **Ops:** quantized attention (prefill/inference fwd), INT8 QK matmul, FP8 PV matmul (FP16 accumulator).
- **Hardware:** Ada sm89, Hopper sm90 (also Ampere sm80). H100/H800/H20 match FA3-FP8 speed with better accuracy.
- **dtypes:** INT8 (QK^T), FP8 e4m3 (PV) w/ FP16 accumulator (`mma.f16.f8.f8.f16`); per-thread INT4 variant in SA2.
- **License:** Apache-2.0.
- **Install:** `pip install sageattention==2.2.0 --no-build-isolation` (or build from source).
- **Claimed perf:** ~4× over FP16 for the PV matmul instruction; 2–5× vs FlashAttention end-to-end; ~3.9× on RTX 4090 over FA2.
- **Baselines:** `ks_flash_attn` / `ks_flash_attn_varlen` (prefill fwd) — INT8/FP8 quantized attention on Ada/Hopper.
- **Source:** <https://arxiv.org/pdf/2505.21136>

### ThunderKittens 2.0 — `HazyResearch/ThunderKittens`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** TK 2.0 (released ~Jan–Feb 2026; blog 2026-02-19).
- **Ops:** tile-primitive attention kernels (fwd + bwd), GEMM/matmul, linear-attention / SSM kernels, megakernels, multi-GPU kernels.
- **Hardware:** **Blackwell B200 sm100** full support (TMEM controllability, CLC scheduling); Hopper sm90; Ada/Ampere.
- **dtypes:** BF16, FP16, FP8, MXFP8, NVFP4 (Blackwell).
- **License:** MIT.
- **Install:** build from source (C++/CUDA tile DSL); header-library + Python bindings.
- **Claimed perf:** production use at Together AI / Jump Trading / Cursor; B200 attention/matmul with reduced memory instructions; competitive with hand-tuned CUTLASS.
- **Baselines:** `ks_flash_attn` / `ks_flash_attn_varlen` (prefill) and GEMM ops — a DSL/substrate baseline for the attention+gemm building blocks.
- **Source:** <https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2>

### xformers — `facebookresearch/xformers`  ⟶ confidence: **medium**, 2026 H1
- **Version / date:** active 2026 dev; `cutlass_blackwell` module added; full sm120 Blackwell support in progress (issue #1395, 2026-04-29).
- **Ops:** `memory_efficient_attention` (Cutlass/Flash/Triton backends), block-diagonal/varlen masks, FA3 default backend when available.
- **Hardware:** sm70+ broad coverage (Volta→Hopper); Blackwell sm100/sm120 partial/in-progress.
- **dtypes:** FP16, BF16, FP8 (via FA3 backend).
- **License:** BSD-3-Clause.
- **Install:** `pip install xformers` (matched to torch/CUDA).
- **Claimed perf:** uses FA3 by default in `memory_efficient_attention` when available (~10% faster end-to-end training on H100 vs FA2).
- **Baselines:** `ks_flash_attn` / `ks_flash_attn_varlen` — the **portable fallback** attention baseline (esp. older archs / odd head-dims).
- **Source:** <https://github.com/facebookresearch/xformers/blob/main/CHANGELOG.md>

### flash-linear-attention (fla) — `fla-org/flash-linear-attention`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** active 2026 H1; [2026-04] added TileLang backend for GDN/KDA/parallel-attn + attention-sink; FLA 0.5.0 referenced as the baseline FlashQLA reports 2–3× over.
- **Ops:** `chunk_gla`, `chunk_gdn` (Gated DeltaNet), `chunk_retention`, KDA, NSA (native sparse attention), PaTH attention, RWKV-7 kernels, fused LayerNormGated, attention-sink (GPT-OSS-style).
- **Hardware:** arch-portable (Triton; new TileLang backend); Hopper sm90+ for TileLang kernels.
- **dtypes:** BF16, FP16 (fp32 accumulate).
- **License:** MIT.
- **Install:** `pip install flash-linear-attention` (or `fla-core`); `rwkv-fla` variant on PyPI.
- **Claimed perf:** ~1.1× speedup for (Gated)DeltaNet from fused LayerNormGated; the Triton baseline FlashQLA reports 2–3× over.
- **Baselines:** **linear/SSM attention class** — substitute for `ks_flash_attn` in hybrid models (Qwen3-Next gated-DeltaNet, RWKV). No direct kernel-set C ABI op yet.
- **Source:** <https://github.com/fla-org/flash-linear-attention/releases>

### FlashQLA — `QwenLM/FlashQLA`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** new in 2026 (Qwen team; ~19 commits, 2026-dated); blog `qwen.ai/blog?id=flashqla`.
- **Ops:** GDN (Gated DeltaNet) chunked prefill (fwd + bwd), context-parallel (CP) + backward-friendly linear attention, parallel attention.
- **Hardware:** **Hopper sm90 or above** (TileLang-built).
- **dtypes:** BF16/FP16 (linear-attention chunked).
- **License:** MIT.
- **Install:** `git clone` + `pip install -v .`.
- **Claimed perf:** 2–3× forward and 2× backward over the FLA Triton kernel on Hopper (vs FLA 0.5.0 and FlashInfer 0.6.9).
- **Baselines:** **linear/SSM attention class** — substitute for `ks_flash_attn` in Qwen3.5 / Qwen3-Next gated-DeltaNet. No direct kernel-set C ABI op; baselines CP-friendly GDN linear attention.
- **Source:** <https://github.com/QwenLM/FlashQLA>

### mamba-ssm (Mamba / Mamba-2 / Mamba-3) — `state-spaces/mamba`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** mamba-ssm 2.3.2.post1 (2026-05-09); Mamba-3 (arXiv:2603.15569) from source.
- **Ops:** `selective_scan`, chunked SSD scan (Mamba-2), Mamba-3 selective scan + MIMO variant.
- **Hardware:** Ampere sm80 → Hopper sm90 (CUDA fwd+bwd); arch-portable Triton paths.
- **dtypes:** BF16, FP16, FP32.
- **License:** Apache-2.0.
- **Install:** `pip install mamba-ssm[causal-conv1d] --no-build-isolation`; Mamba-3: `MAMBA_FORCE_BUILD=TRUE pip install git+... --no-build-isolation`.
- **Claimed perf:** Mamba-3 1.5B: +0.6pp avg downstream acc over Gated DeltaNet (+1.2pp for MIMO); comparable PPL to Mamba-2 at half the state size.
- **Baselines:** **SSM/selective-scan class** — substitute for attention in hybrid models (Jamba/Zamba/Nemotron-H). No direct kernel-set C ABI op.
- **Source:** <https://github.com/state-spaces/mamba/releases>

### causal-conv1d — `Dao-AILab/causal-conv1d`  ⟶ confidence: **medium**, 2026 H1
- **Version / date:** ≥1.4.0 (tracked with mamba-ssm 2.3.x in 2026 H1).
- **Ops:** fused short causal depthwise conv1d (fwd + bwd), conv update for decode.
- **Hardware:** Ampere sm80 → Hopper sm90; arch-portable.
- **dtypes:** BF16, FP16, FP32.
- **License:** BSD-3-Clause.
- **Install:** `pip install causal-conv1d --no-build-isolation` (or via `mamba-ssm[causal-conv1d]`).
- **Claimed perf:** fused short causal conv used inside every Mamba/SSM block; replaces slow PyTorch conv1d.
- **Baselines:** **causal-conv1d class** (conv-mixer for Mamba/SSM/conv blocks). No direct kernel-set C ABI op.
- **Source:** <https://github.com/state-spaces/mamba/blob/main/README.md>

### DeepSeek Sparse Attention (DSA) / NSA / FSA — `deepseek-ai/DeepSeek-V3.2-Exp`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** DSA: V3.2-Exp (2025-09), hardened V3.2 (2025-12), V4 (2026-04); NSA arXiv:2502.11089; FSA arXiv:2508.18224.
- **Ops:** fine-grained sparse attention (DSA), native sparse attention (NSA), Flash Sparse Attention (FSA) kernel, long-context prefill + decode.
- **Hardware:** Hopper sm90, Blackwell sm100 (via FlashMLA sparse kernel + DeepGEMM indexer).
- **dtypes:** BF16, FP8.
- **License:** MIT (DeepSeek repos); various for NSA/FSA reimpls.
- **Install:** via vLLM/SGLang day-0 support; FlashMLA + DeepGEMM build from source.
- **Claimed perf:** substantial long-context train+infer efficiency at near-identical quality (DSA); FSA up to 3.5× kernel-latency reduction vs vanilla NSA.
- **Baselines:** sparse variant of `ks_mla_decode` / attention prefill for long-context (DeepSeek V3.2/V4) — the emerging **sparse-attention op class**.
- **Source:** <https://vllm-project.github.io/2026/04/24/deepseek-v4.html>

### Ring / Context-Parallel Attention (vLLM / SGLang) — `sgl-project/sglang`  ⟶ confidence: **medium**, 2026 H1
- **Version / date:** active 2026 H1: SGLang PCP (issue #22223), vLLM CP RFC (#26133); builds on FA/FlashInfer attention backends.
- **Ops:** zigzag ring attention, pass-KV / pass-Q ring attention, prefill context parallel (PCP), decode context parallel, fully-sharded KV-cache CP.
- **Hardware:** multi-GPU Hopper/Blackwell (H100/B200) over NVLink/RDMA.
- **dtypes:** BF16, FP16, FP8 (KV).
- **License:** Apache-2.0.
- **Install:** via SGLang / vLLM (pip).
- **Claimed perf:** near-linear scaling for long-context prefill to 128 H100s (1M-ctx Llama3-405B prefill in 77s); exact/lossless ring variants.
- **Baselines:** distributed wrapper over `ks_flash_attn_varlen` / `ks_paged_attn_decode` — the **context-parallel orchestration** layer (kernel-set provides the per-rank op; **multi-GPU, not single-GPU benchable**).
- **Source:** <https://github.com/sgl-project/sglang/issues/22223>

---

## 2. GEMM / quantization

### CUTLASS — `NVIDIA/cutlass`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** 4.5.2 (latest); 4.5.0 (2026-05-01), 4.4.0 (2026-02-14, CUDA 13.1 + GB300), 4.3.0 (2025-11-21).
- **Ops:** dense GEMM (FP16/BF16/TF32/FP8/INT8), block-scaled GEMM (NVFP4/MXFP4/MXFP8/MXFP6), grouped GEMM (MoE), mixed-input GEMM (W4A16/W8A16), epilogue fusion (bias/act/scale), CuTe DSL Python kernels.
- **Hardware:** sm80 / sm89 / sm90 (WGMMA+TMA) / sm100,sm103 (Blackwell, tcgen05) / sm120 (RTX Blackwell / Spark).
- **dtypes:** FP16, BF16, TF32, FP8 (e4m3/e5m2), INT8, FP4 (nvfp4/mxfp4), FP6 (mxfp6), block-scaled MX.
- **License:** BSD-3-Clause.
- **Install:** `git clone` + CMake (header-only templates); Python DSL: `pip install nvidia-cutlass-dsl`.
- **Claimed perf:** tcgen05 MMA 2×–4× faster than Hopper WGMMA on Blackwell sm100; near-peak tensor-core utilization across archs.
- **Baselines:** `ks_gemm` / `ks_gemm_bias_act` / `ks_gemm_batched` (dense + epilogue), `ks_gemm_w8a8` (INT8 GEMM), `ks_gemm_w4a16` (mixed-input). The **substrate baseline** for kernel-set GEMM ops.
- **Source:** <https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md>

### DeepGEMM — `deepseek-ai/DeepGEMM`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** `public-release-260416` / PR #304 "Mega MoE, FP4 Indexer" (2026-04-16/17); further Mega-MoE opts in PR #316 (2026-04-24), #347 (2026-06-01).
- **Ops:** FP8 dense GEMM (fine-grained block/tile scaling, two-level CUDA-core accumulation), FP8 grouped GEMM (MoE contiguous + masked), FP8×FP4 GEMM, FP4 indexer, lightning indexer (weighted ReLU MQA logits, DeepSeek sparse attn), `paged_mqa_logits`, Mega MoE mega-kernel (dispatch+linear+SwiGLU+combine).
- **Hardware:** **Hopper sm90** (H800/H100; NT layout only) and **Blackwell sm100** (B200/GB200; NT/TN/NN/TT); partial sm120. Mega MoE needs NVLink + PyTorch ≥ 2.9.
- **dtypes:** FP8 (e4m3 block-scaled), FP4 (nvfp4/mxfp4), FP8×FP4, BF16 (k-grouped) accumulate.
- **License:** MIT.
- **Install:** build from source (`git clone --recursive`; `./develop.sh` then `./install.sh`); JIT CPP module, low CPU overhead. Needs CUDA 12.3+ (FP4 path PyTorch ≥ 2.9), C++20.
- **Claimed perf:** ~1550 TFLOP/s FP8 on H800; on B200 FP4 tensor cores ~2× FP8 TFLOP/s; lightning-indexer kernels power DeepSeek V3.2/V4 sparse attention in vLLM/SGLang.
- **Baselines:** `ks_gemm_w8a8`-analog for **block-scaled FP8 dense GEMM** (`ks_gemm` FP8 path); **FP8/FP4 grouped GEMM** is the MoE-expert analog of `ks_moe_grouped_gemm`; pairs with `ks_quantize_fp8` block-scale. *(Also central to DeepSeek sparse-attention serving — see §1.)*
- **Source:** <https://github.com/deepseek-ai/DeepGEMM/pull/304>

### TileKernels (DeepSeek) — `deepseek-ai/TileKernels`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** open-sourced ~2026-04-23 (no tagged release yet); written in TileLang ≥ 0.1.9.
- **Ops:** MoE gating / top-k routing, quantization casting (per-token / per-block / per-channel FP8 / FP4 / E5M6), batched transpose, Engram gating (fused RMSNorm), Manifold HyperConnection (Sinkhorn norm).
- **Hardware:** **sm90 (H100/H200) and sm100 (Blackwell)**; CUDA Toolkit 13.1+.
- **dtypes:** FP8, FP4, E5M6 quantization casting (per-token/per-block/per-channel).
- **License:** MIT.
- **Install:** `pip install tile-kernels`; needs Python 3.10+, PyTorch 2.10+, TileLang 0.1.9+, CUDA 13.1+.
- **Claimed perf:** authors state kernels approach hardware limit on compute intensity and memory bandwidth (used internally; not claimed best-practice).
- **Baselines:** `ks_quantize_fp8` (per-token/per-tensor/block FP8 cast) and a per-block/per-channel FP4/E5M6 extension; MoE gating maps to the MoE routing path (not a GEMM op).
- **Source:** <https://github.com/deepseek-ai/TileKernels>

### Marlin / Machete (vLLM) — `vllm-project/vllm`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** ongoing in vLLM 2026 releases; NVFP4 W4A16 MoE Marlin support added (PR #30906, 2026).
- **Ops:** W4A16 mixed-input GEMM (Marlin, INT4 weight × FP16/BF16 act), W8A16 / W4A8 (GPTQ-Marlin), W4A16 on Hopper WGMMA (Machete, weight prepack), NVFP4 W4A16 GEMM + NVFP4 W4A16 MoE (Marlin, 2026).
- **Hardware:** Marlin sm80/sm86/sm89 sweet spot (also sm90/sm120); Machete sm90a (Hopper); NVFP4 Marlin needs native FP4 (Blackwell) or emulated.
- **dtypes:** INT4 weight × FP16/BF16 act; W8A16; W4A8; NVFP4 (E2M1 weight) W4A16.
- **License:** Apache-2.0.
- **Install:** `pip install vllm` (kernels bundled); GPTQ/AWQ checkpoints auto-dispatch to Marlin/Machete.
- **Claimed perf:** Marlin near-ideal ~4× speedup for W4A16 vs FP16; Machete best W4A16 on H100.
- **Baselines:** `ks_gemm_w4a16` (group-wise INT4 weight × fp16/bf16 act, AWQ/GPTQ layout) is the direct analog; NVFP4 Marlin extends it to FP4 weights.
- **Source:** <https://github.com/vllm-project/vllm/pull/30906>

### TransformerEngine (NVIDIA) — `NVIDIA/TransformerEngine`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** 2.15.0 (2026-05-11); 2.14 (2026-04), 2.13 (2026-03), 2.12 (2026-02), 2.11 (2026-01).
- **Ops:** FP8 Linear / LayerNormLinear / LayerNormMLP (cuBLASLt-backed), FP8/MXFP8/NVFP4 quantize+cast (delayed/current scaling, amax), FP8 + NVFP4 GEMM (train+infer), fused MoE router, cuDNN FMHA.
- **Hardware:** sm89 (Ada, FP8) / sm90 (Hopper) / sm100 (Blackwell, MXFP8 + NVFP4).
- **dtypes:** FP8 e4m3/e5m2 (per-tensor/block), MXFP8 (block-32 e4m3), NVFP4 (E2M1, block-16 scale), BF16.
- **License:** Apache-2.0.
- **Install:** `pip install transformer-engine[pytorch]`.
- **Claimed perf:** standard FP8 training stack; NVFP4 training matches ~16-bit precision at 4-bit speed (used to train Nemotron); improved Blackwell MoE router + NVFP4 amax kernel.
- **Baselines:** `ks_quantize_fp8` / `ks_dequantize_fp8` (FP8 cast + scaling) and FP8 GEMM epilogue (`ks_gemm_bias_act` FP8 analog); NVFP4/MXFP8 extend the quant-cast ops.
- **Source:** <https://pypi.org/project/transformer-engine/>

### torchao (pytorch/ao) — `pytorch/ao`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** 0.17.0 (2026-03-30); 0.16.0 (2026-02-10), 0.15.0 (2025-12-18).
- **Ops:** float8 dynamic/rowwise quant + GEMM, MXFP8 / MXFP4 / NVFP4 quantize + GEMM (Blackwell), int4 weight-only GEMM (tinygemm/Marlin), int8 dynamic-act / weight-only quant, NVFP4 QAT, CuteDSL MXFP8 MoE kernels, 8-bit/4-bit AdamW optimizers.
- **Hardware:** sm89/sm90 (FP8) and sm100+ (Blackwell, MXFP8/MXFP4/NVFP4); int4/int8 broadly sm75+.
- **dtypes:** float8 (e4m3/e5m2), MXFP8 (e8m0 scale), MXFP4/NVFP4 (float4_e2m1fn_x2 + e4m3 block scale), int4, int8.
- **License:** BSD-3-Clause.
- **Install:** `pip install torchao`.
- **Claimed perf:** B200: MXFP8 cuBLAS ~1.75–1.97× GEMM roofline, NVFP4 ~2.36–3.82×; end-to-end up to 1.26× (MXFP8) / 1.68× (NVFP4) on diffusion; CuteDSL MXFP8 MoE ~12% over prior 2-kernel approach. Llama-3.1-70B pretrain ~1.5× faster with float8.
- **Baselines:** `ks_quantize_fp8`/`ks_dequantize_fp8` (float8 cast), `ks_quantize_int8`/`ks_dequantize_int8` (int8 dynamic), `ks_dequantize_int4` + `ks_gemm_w4a16` (int4 weight-only); `ks_adamw` (8/4-bit optimizer); NVFP4/MXFP4 extend FP-quant ops.
- **Source:** <https://pypi.org/project/torchao/>

### FlashInfer (GEMM / NVFP4) — `flashinfer-ai/flashinfer`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** v0.6.12 (2026-05-29); nightly 0.6.12-20260604.
- **Ops:** NVFP4 GEMM + fused NVFP4 MoE, FP8 CUTLASS GEMM / MLA, trtllm-gen GEMM & MoE, grouped-GEMM + combine fusion (CuTe DSL), `fp4_quantize` torch.compile custom op, SM120 W4A16 kernel.
- **Hardware:** sm80 → sm100 (B200); SM120/SM110 (RTX Blackwell, Thor) kernels added; FP8 V-scratch on sm90.
- **dtypes:** NVFP4/FP4 (E2M1), FP8 (e4m3), MXFP8, MXINT4, BF16.
- **License:** Apache-2.0.
- **Install:** `pip install flashinfer-python`.
- **Claimed perf:** ~1225 TFLOP/s NVFP4 GEMM on Blackwell at BS=4096 (within ~3% of SGLang); NVIDIA ships newest TRT-LLM kernels here.
- **Baselines:** `ks_gemm_w8a8` (FP8 GEMM analog), `ks_gemm_w4a16` (W4A16 / mxint4), an NVFP4 GEMM extension; `fp4_quantize` maps to a quant-cast op analogous to `ks_quantize_fp8`. *(Same package as the §1 attention entry — one install, multiple op mappings.)*
- **Source:** <https://github.com/flashinfer-ai/flashinfer/releases>

### GPTQModel (ModelCloud) — `ModelCloud/GPTQModel`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** 7.0.0 (2026-04-28); 6.1.0 (2026-04-16, fully JIT CUDA), 6.0.3 (2026-04-03), 5.8.0 (2026-03-19).
- **Ops:** GPTQ / AWQ W4A16 quant + GEMM (Marlin), FP8 quant, EXL3 trellis quant, ParoQuant (pairwise rotation), QQQ / GGUF / FOEM / GPTAQ, JIT-compiled CUDA kernels.
- **Hardware:** NVIDIA Turing+ (sm75+) via GPTQ-Marlin; AMD, Intel GPU, Intel/AMD/Apple CPU, Huawei Ascend NPU.
- **dtypes:** INT4 (GPTQ/AWQ W4A16), INT8, FP8, EXL3 (1–8 bpw trellis), 1-bit (Bonsai).
- **License:** Apache-2.0.
- **Install:** `pip install gptqmodel`.
- **Claimed perf:** faster ParoQuant/AWQ kernels; fully JIT-compiled CUDA kernels (6.1.0); new fast HF CPU kernels.
- **Baselines:** `ks_gemm_w4a16` + `ks_dequantize_int4` (GPTQ/AWQ W4A16 group-wise scales/zeros); FP8 path → `ks_quantize_fp8`.
- **Source:** <https://pypi.org/project/GPTQModel/>

### ExLlamaV3 (turboderp) — `turboderp-org/exllamav3`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** 0.0.38 (2026-05-29).
- **Ops:** EXL3 trellis quantization (QTIP-based, 1–8 bpw), fused W4A16-style mixed-input GEMM on 16×16 tiles, per-layer mixed bit-rate inference.
- **Hardware:** consumer NVIDIA GPUs; 16×16 tensor-core tiles (sm75+); Ampere/Ada/Blackwell consumer cards.
- **dtypes:** EXL3 (procedural-codebook trellis, ~1.6–8 bpw weight) × FP16/BF16 act.
- **License:** MIT.
- **Install:** `pip install exllamav3` (or build from source).
- **Claimed perf:** QTIP-quality trellis quant to low bpw with fused kernels; consumer-GPU throughput focus.
- **Baselines:** `ks_gemm_w4a16` analog (low-bit weight × fp16 act mixed-input GEMM); trellis-coded weights are a non-uniform variant of the int4 path (`ks_dequantize_int4`).
- **Source:** <https://github.com/turboderp-org/exllamav3/releases>

### bitsandbytes — `bitsandbytes-foundation/bitsandbytes`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** 0.49.2 (2026-02-16); 0.49.1 (2026-01-08), 0.49.0 (2025-12-11).
- **Ops:** LLM.int8() W8A8 GEMM (outlier fp16 path), NF4 / FP4 4-bit blockwise quant + dequant (QLoRA), branchless NF4/FP4 dequant kernel, 8-bit optimizers.
- **Hardware:** Turing+ (sm75+) incl. A100/H100/B200; CUDA 13 builds (Linux x86-64/aarch64, Windows); Thor (sm110) on aarch64; AMD ROCm experimental; Intel Gaudi2/3.
- **dtypes:** INT8 (LLM.int8()), NF4, FP4 (4-bit blockwise, fp32/bf16 dequant). **Note: NF4 ≠ NVIDIA native NVFP4.**
- **License:** MIT.
- **Install:** `pip install bitsandbytes`.
- **Claimed perf:** branchless NF4/FP4 dequant speedups on A100/H100/B200 for prefill, batch decode, and training.
- **Baselines:** `ks_quantize_int8`/`ks_dequantize_int8` (LLM.int8 W8A8 → `ks_gemm_w8a8`) and `ks_dequantize_int4` (NF4/FP4 blockwise weight dequant for QLoRA-style W4A16).
- **Source:** <https://pypi.org/project/bitsandbytes/>

### Quack (Dao-AILab) — `Dao-AILab/quack`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** `quack-kernels` (PyPI, actively updated 2026; GEMM kernels added beyond the original 2025 memory-bound set).
- **Ops:** Hopper GEMM + epilogue, Blackwell GEMM + epilogue, Blackwell GeForce GEMM + epilogue, RMSNorm/LayerNorm/Softmax/CrossEntropy fwd+bwd (memory-bound, CuTe DSL).
- **Hardware:** sm90 (H100), sm100 (B200/B300), sm120 (RTX 50 GeForce Blackwell); CUDA 12.9+ / 13.x.
- **dtypes:** BF16 (benchmarked); FP16; written in CuTe-DSL (Python).
- **License:** BSD-3-Clause.
- **Install:** `pip install quack-kernels` (CUDA 12.9); `pip install 'quack-kernels[cu13]' --extra-index-url https://download.pytorch.org/whl/cu130` (CUDA 13.x).
- **Claimed perf:** memory-bound kernels reach near speed-of-light bandwidth; new Hopper/Blackwell GEMM+epilogue kernels.
- **Baselines:** `ks_gemm` / `ks_gemm_bias_act` (GEMM + epilogue, Hopper & Blackwell) and the memory-bound `ks_rms_norm`/`ks_layer_norm`/`ks_softmax`/`ks_cross_entropy`. No FP8/INT quant-GEMM, so no W8A8/W4A16 mapping.
- **Source:** <https://github.com/Dao-AILab/quack/blob/main/README.md>

### AutoAWQ — `casper-hansen/AutoAWQ`  ⟶ confidence: **high** (provenance only)
- **Version / date:** **DEPRECATED / archived May 2025** (last tested Torch 2.6.0, Transformers 4.51.3); superseded by vLLM `llm-compressor`. *(not a 2026 release; included for AWQ provenance.)*
- **Ops:** AWQ W4A16 quantization + fused dequant-GEMM / GEMV (legacy).
- **Hardware:** sm75+ (legacy AWQ kernels).
- **dtypes:** INT4 weight × FP16 act (W4A16, group-wise scales/zeros).
- **License:** MIT.
- **Install:** `pip install autoawq` (unmaintained); **use `vllm-project/llm-compressor` instead**.
- **Claimed perf:** ~2× inference speedup vs FP16 (legacy claim).
- **Baselines:** `ks_gemm_w4a16` + `ks_dequantize_int4` (AWQ layout). Provenance/completeness only — the AWQ algorithm now lives in `llm-compressor`.
- **Source:** <https://github.com/casper-hansen/AutoAWQ>

---

## 3. MoE / comm / training

### DeepEP (EPv2) — `deepseek-ai/DeepEP`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** EPv2 (main, public release 2026-04; merged 2026-04-29, active through 2026-05-26); last tag v1.2.1.
- **Ops:** MoE all-to-all dispatch (FP8/BF16), all-to-all combine (BF16), group-limited gating-aware NVLink→RDMA forwarding, low-latency decode dispatch/combine, 0-SM Engram / PP / CP comm paths.
- **Hardware:** **NVIDIA Hopper sm90 + Blackwell sm100**; CX7 RDMA + NVLink; up to EP2048. **Multi-GPU.**
- **dtypes:** FP8 dispatch, BF16 combine.
- **License:** MIT.
- **Install:** `git clone` + `python setup.py install` (NCCL Gin backend; reuses NCCL communicators; no NVSHMEM required in EPv2).
- **Claimed perf:** 1.3× peak vs V1 while saving up to ~4× SM count (24 → 4–6 SMs); e.g. SM90 EP8×2 CX7: dispatch 90 GB/s, combine 81 GB/s.
- **Baselines:** **no direct C ABI op (kernel-set is single-GPU).** The EP comm layer wrapping `ks_moe_permute`/`ks_moe_unpermute` — the cross-GPU all-to-all feeding `ks_moe_grouped_gemm`. Comm counterpart to the MoE pipeline.
- **Source:** <https://github.com/deepseek-ai/DeepEP/pull/605>

### DeepGEMM (Mega MoE) — `deepseek-ai/DeepGEMM`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** main, public release 2026-04 (PR #304 merged 2026-04-17); Mega-MoE opts in PR #316 (2026-04-24), #347 (2026-06-01).
- **Ops:** FP8 dense GEMM (fine-grained block scaling), MoE grouped GEMM (contiguous M-grouped + masked), FP8×FP4 GEMM, Mega MoE fused mega-kernel (dispatch+linear1+SwiGLU+linear2+combine), FP4 indexer / MQA logits (MTP).
- **Hardware:** **Hopper sm90 + Blackwell sm100**; WGMMA/TMA, tcgen05/TMEM. Mega MoE needs NVLink + PyTorch ≥ 2.9.
- **dtypes:** FP8 (e4m3 block-scaled), FP4 (mxfp4/nvfp4), FP8×FP4, BF16 (k-grouped).
- **License:** MIT.
- **Install:** build from source (`python setup.py develop/install`; low-CPU-overhead JIT CPP module).
- **Claimed perf:** ~1550 TFLOP/s FP8 on H800 (dense); Mega MoE overlaps NVLink comm with tensor-core compute in a single kernel; two-level CUDA-core accumulation fixes FP8 accumulation error.
- **Baselines:** `ks_moe_grouped_gemm` (primary: FP8/FP4 grouped experts, contiguous + masked); FP8 dense → `ks_gemm`/`ks_gemm_w8a8`. Mega MoE fuses the whole `ks_moe_permute` → `ks_moe_grouped_gemm` (×2 w/ SwiGLU) → `ks_moe_unpermute` pipeline + DeepEP comm.
- **Source:** <https://github.com/deepseek-ai/DeepGEMM/pull/304>

### NCCL EP (`ncclEpDispatch`/`ncclEpCombine`) — `NVIDIA/nccl`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** built on the NCCL Device API (NCCL 2.30.x, 2026); paper arXiv:2603.13606 (submitted 2026-03-13, v2 2026-04-02).
- **Ops:** MoE all-to-all dispatch (`ncclEpDispatch`), combine (`ncclEpCombine`), low-latency (LL) decode all-to-all (1–128 tokens), high-throughput (HT) hierarchical all-to-all (4096+ tokens).
- **Hardware:** NVIDIA H100 GPU clusters; RDMA + NVLink. **Multi-GPU.**
- **dtypes:** BF16 / FP8 MoE tokens (fp32 routing).
- **License:** BSD-3-Clause (NCCL).
- **Install:** NCCL ≥ 2.30 (vendor lib); C and Python `ncclEp*` interfaces; vLLM integration path.
- **Claimed perf:** competitive LL kernel perf vs DeepEP/pplx on H100; unified EP API across LL (decode) and HT (train/prefill) modes.
- **Baselines:** **no direct C ABI op (cross-GPU comm).** Vendor-unified EP all-to-all surrounding `ks_moe_permute`/`ks_moe_grouped_gemm`/`ks_moe_unpermute`. Direct alternative to DeepEP/pplx.
- **Source:** <https://arxiv.org/abs/2603.13606>

### Perplexity pplx-garden — `perplexityai/pplx-garden`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** pplx-garden (initial release 2025-11-05, active through 2026-05-27, incl. pplx-unigram); supersedes the deprecated NVSHMEM-based pplx-kernels.
- **Ops:** MoE all-to-all dispatch (IBGDA GPU-initiated), all-to-all combine, split dispatch/combine for comm-compute overlap, fabric-lib portable EP transport (NVLink/CX7/EFA).
- **Hardware:** NVIDIA Hopper/Blackwell; NVLink, ConnectX-7, AWS EFA; trillion-param scale. **Multi-GPU.**
- **dtypes:** FP8 dispatch, BF16 combine.
- **License:** MIT.
- **Install:** `git clone github.com/perplexityai/pplx-garden` + build (CUDA 12/13 bindings).
- **Claimed perf:** 2.5× lower latency than prior fastest on single node; up to ~10× faster than standard all-to-all multi-node; portable across NVSHMEM versions and NIC/transport.
- **Baselines:** **no direct C ABI op (cross-GPU comm).** Portable EP all-to-all around the MoE pipeline feeding `ks_moe_grouped_gemm`; alternative to DeepEP/NCCL-EP.
- **Source:** <https://www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication>

### sgl-kernel (SGLang) — `sgl-project/sglang`  ⟶ confidence: **high**, 2026 H1
- **Status:** **The hard-op alignment target for kernel-set.** sgl-kernel is wired as a first-class provider in the dispatcher (`kernel_set.dispatch`) and the SOTA bench (`benchmarks/bench_sota.py`): kernel-set is benched/aligned against it for the MoE gating + grouped-MoE path (where sgl-kernel is **rank #1**, its specialty), and competitively for sampling, RMSNorm / fused-add-RMSNorm / Gemma-RMSNorm, RoPE, SiLU-mul, FP8/INT8 scaled-mm, and FA3 attention (prefill/decode) + FlashMLA.
- **Version / date:** sgl-kernel 0.3.21 on PyPI (2026-01-15); now in-tree in the sglang monorepo; MoE gate extended to 256 experts (MiMo V2) 2026-06-01. Vendored at `third_party/sglang/sgl-kernel/`.
- **Ops:** MoE fused gate (softmax/sigmoid top-k via `topk_softmax`/`topk_sigmoid`), DeepSeek group-limited top-k gating (`moe_fused_gate`, up to 256 experts; `kimi_k2_moe_fused_gate`), `moe_align_block_size`, fused/grouped MoE (CUTLASS grouped FP8/FP4, `fp8_blockwise_scaled_grouped_mm`, `expert_specialization`), `rmsnorm`/`fused_add_rmsnorm`/`gemma_rmsnorm`, `rotary_embedding`, `silu_and_mul`/`gelu_and_mul`, `fp8_scaled_mm`/`fp8_blockwise_scaled_mm`/`int8_scaled_mm`/`bmm_fp8`/`awq_dequantize`, FA3 attention (`flash_attn_varlen_func`/`flash_attn_with_kvcache`), FlashMLA (`flash_mla_with_kvcache`, `get_mla_metadata`, `cutlass_mla_decode`), sampling renorm (`top_k_renorm_prob`/`top_p_renorm_prob`; MUSA build also `top_k_top_p_sampling_from_probs`/`min_p_sampling_from_probs`), speculative verify (`tree_speculative_sampling_target_only`/`verify_tree_greedy`).
- **Hardware:** NVIDIA sm80/89/90/100 (+ CPU AVX512/AMX, HIP/ROCm); FlashInfer mxfp8 MoE/GEMM integration. FP8 scaled-mm + FlashMLA + FA3 are sm90 paths.
- **dtypes:** BF16/FP16, FP8 (incl. mxfp8 microscaling), INT8, FP4/NVFP4.
- **License:** Apache-2.0.
- **Install:** `pip install sgl-kernel` (or build from sglang/sgl-kernel). Added to `benchmarks/install_baselines.sh` (`INSTALL_SGL=1`, guarded; needs torch + a recent CUDA).
- **Claimed perf:** production serving kernels; `moe_fused_gate` is a single-kernel DeepSeek-V3 biased group-topk gate (best-in-class routing); FlashInfer mxfp8 path for higher-accuracy FP8 MoE; dispatcher picks per shape/dtype/arch.
- **Baselines:** `ks_moe_gate_softmax_topk` + `ks_moe_gate_sigmoid_group_topk` (DeepSeek group-limited, **rank #1**); `ks_moe_compute_permutation`/`ks_moe_permute`/`ks_moe_grouped_gemm`/`ks_moe_unpermute` (fused_moe); plus `ks_rmsnorm`/`ks_fused_add_rmsnorm`/`ks_gemma_rmsnorm`, `ks_rope`, `ks_silu_and_mul`, `ks_gemm_w8a8`/FP8, `ks_sample`, and the attention/MLA decode ABIs.
- **Source:** <https://github.com/sgl-project/sglang> (MoE gate: <https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/moe/moe_fused_gate.cu>)

### vLLM kernels (fused MoE / FP4) — `vllm-project/vllm`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** main, 2026 H1 (MXFP4 W4A4 CUTLASS MoE SM100 PR #37463; FlashInfer b12x MoE+FP4 GEMM SM120/121 PR #40082); active through 2026-06.
- **Ops:** Triton `fused_moe` (gate+permute+grouped GEMM+unpermute), CUTLASS grouped FP8 MoE (`cutlass_moe_fp8`), MXFP4/NVFP4 W4A4 CUTLASS MoE (SM100), FlashInfer b12x fused MoE (NVFP4, SM120/121), W8A8/W4A16/FP8 GEMM dispatch (Marlin/Machete/CUTLASS), paged attention, RMSNorm, RoPE, SiLU-mul.
- **Hardware:** NVIDIA sm80→sm100 + sm120/121 (RTX Pro 6000, DGX Spark GB10).
- **dtypes:** BF16/FP16, FP8, INT8, INT4 (W4A16), MXFP4/NVFP4 (W4A4).
- **License:** Apache-2.0.
- **Install:** `pip install vllm` (kernels in `vllm._custom_ops` / `fused_moe`).
- **Claimed perf:** b12x NVFP4 fused MoE on SM120/121 fuses dispatch+W1 GEMM+SwiGLU+W2 GEMM into one call; +1.8% to +6.0% throughput vs flashinfer-cutlass on DGX Spark (Qwen3-30B-A3B-NVFP4).
- **Baselines:** full MoE pipeline — `ks_moe_gate_*`, `ks_moe_compute_permutation`, `ks_moe_permute`, `ks_moe_grouped_gemm` (FP8/FP4), `ks_moe_unpermute`; plus `ks_gemm_w8a8` / `ks_gemm_w4a16` / FP8 `ks_gemm`.
- **Source:** <https://docs.vllm.ai/en/latest/design/moe_kernel_features/>

### FlashInfer (fused MoE / activation / sampling) — `flashinfer-ai/flashinfer`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** v0.6.12 (2026-05-29); nightlies through 2026-06-04.
- **Ops:** fused MoE (`b12x_fused_moe`, NVFP4), `mm_fp4` / FP4 GEMM (block-scale), mxfp8 MoE/GEMM, paged decode/prefill attention + MLA, fused RMSNorm (+residual), RoPE, SiLU-mul, fused sampling (top-k/top-p/min-p, rejection).
- **Hardware:** NVIDIA sm80→sm100 + sm120/121 (b12x: DGX Spark GB10, RTX Pro 6000).
- **dtypes:** FP16/BF16, FP8 (e4m3/e5m2, mxfp8), FP4 (NVFP4/MXFP4), FP8 KV-cache.
- **License:** Apache-2.0.
- **Install:** `pip install flashinfer-python` (JIT plan/run API).
- **Claimed perf:** b12x GEMM +1.8%/+6.0% throughput vs flashinfer-cutlass on DGX Spark; NVIDIA ships newest TRT-LLM kernels here; backend for vLLM and SGLang.
- **Baselines:** `ks_moe_grouped_gemm` + `ks_moe_gate_*` (fused MoE); `ks_gemm` FP4/FP8; `ks_rms_norm`, `ks_rope`, `ks_swiglu` (activation); `ks_sample` (sampling). *(One install covers attention §1, GEMM §2, and MoE/activation/sampling §3.)*
- **Source:** <https://github.com/flashinfer-ai/flashinfer/releases>

### Megatron-Core (MoE) — `NVIDIA/Megatron-LM`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** megatron-core 0.17.1 (PyPI 2026-05-28); 0.17.0 2026-04-16; MoE tech report arXiv:2603.07685 (2026-03-11).
- **Ops:** MoE GroupedGEMM (FP8/MXFP8), DeepEP token dispatcher (all-to-all), HybridEP token dispatcher, router / top-k + group-limited gating, kernel fusion + sync-free dropless MoE, fused inference MoE (`core.inference.moe.fused_moe`), FusedAdam, distributed optimizer.
- **Hardware:** NVIDIA Hopper sm90 + Blackwell sm100 (native MXFP8 ~2× BF16 TFLOP/s); multi-node EP/TP/PP/CP.
- **dtypes:** BF16, FP8, MXFP8 (GroupedGEMM), FP32 master.
- **License:** Apache-2.0-derived (custom NVIDIA license / NOASSERTION).
- **Install:** `pip install megatron-core`.
- **Claimed perf:** with MXFP8+DeepEP up to 41% faster DeepSeek-V3 pre-training on B200; DeepEP dispatcher + comm-compute overlap + GroupedGEMM + CUDA Graphs.
- **Baselines:** `ks_moe_grouped_gemm` (GroupedGEMM FP8/MXFP8); `ks_moe_gate_softmax_topk` / `ks_moe_gate_sigmoid_group_topk` (router); `ks_moe_permute`/`ks_moe_unpermute` via dispatcher; `ks_adamw` + `ks_global_grad_norm` (FusedAdam + grad clip).
- **Source:** <https://arxiv.org/abs/2603.07685>

### Liger-Kernel — `linkedin/Liger-Kernel`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** v0.8.0 (2026-04-30); v0.7.0 (2026-02-12); v0.6.5 (2026-02-04).
- **Ops:** RMSNorm / LayerNorm, RoPE (incl. Llama4 rope, NPU freq fusion), SwiGLU / GeGLU, CrossEntropy / FusedLinearCrossEntropy (chunked), fused-linear post-training losses (DPO/ORPO/SimPO/GRPO; CISPO/SAPO), fused MoE / grouped GEMM helpers.
- **Hardware:** arch-portable via Triton (NVIDIA sm80→sm100, AMD, Intel XPU, NPU).
- **dtypes:** FP16/BF16 with FP32 accumulate.
- **License:** BSD-2-Clause.
- **Install:** `pip install liger-kernel`.
- **Claimed perf:** ~20% training throughput gain, ~60% memory reduction; FusedLinearCrossEntropy avoids materializing full logits.
- **Baselines:** `ks_rms_norm` / `ks_layer_norm`; `ks_rope`; `ks_swiglu` / `ks_geglu` (activation); `ks_cross_entropy` + `ks_fused_linear_cross_entropy` — Liger is the **direct reference** for the chunked CE kernels.
- **Source:** <https://github.com/linkedin/Liger-Kernel/releases>

### torchao (training/optimizer) — `pytorch/ao`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** v0.17.0 (2026-03-30); v0.16.0 (2026-02-10).
- **Ops:** float8 rowwise training (`nn.Linear`), MXFP8 MoE training (prototype), 8-bit / 4-bit AdamW optimizers, INT4/INT8/FP8/NVFP4 quant (inference), QAT / QLoRA.
- **Hardware:** NVIDIA sm89/sm90/sm100 (float8 Ada+, MXFP8 Blackwell); torch.compile + FSDP2.
- **dtypes:** FP8 (rowwise), MXFP8, INT8/INT4, NVFP4; 8-bit & 4-bit optimizer states.
- **License:** BSD-3-Clause (NOASSERTION on GitHub).
- **Install:** `pip install torchao`.
- **Claimed perf:** Llama-3.1-70B pretrain ~1.5× faster with float8; MXFP8 MoE ~1.45× (Llama4 Scout MoE), ~1.25× (DeepSeek-V3 671B MoE) vs BF16 with comparable numerics.
- **Baselines:** `ks_adamw` (8/4-bit + master-weight optimizer); FP8/MXFP8 GEMM → `ks_gemm` FP8 path and `ks_moe_grouped_gemm` (MXFP8 MoE training); `ks_gemm_w8a8` / `ks_gemm_w4a16`. *(Same package as §2 torchao — quant inference vs training overlap.)*
- **Source:** <https://github.com/pytorch/ao/releases>

### Unsloth MoE kernels — `unslothai/unsloth`  ⟶ confidence: **high**, 2026 H1
- **Version / date:** "12× Faster MoE Training" announced 2026-02-10 (first 2026 release); active through 2026-06-04.
- **Ops:** fused MoE grouped GEMM (Triton), fused MoE expert layer (gate+up+down), fused cross-entropy (`unsloth_fused_ce_loss`), fused RoPE (variable-length), gradient-checkpointing patch.
- **Hardware:** NVIDIA T4, H100, B200, RTX 6000 Pro (Triton).
- **dtypes:** FP16, BF16 (16-bit and 4-bit LoRA).
- **License:** Apache-2.0.
- **Install:** `pip install unsloth`.
- **Claimed perf:** 12× faster MoE training, 35% less VRAM, 6× longer context, no accuracy loss (DeepSeek R1/V3, Qwen3 30B/235B, GLM 4.7/Flash, GPT-OSS); RoPE 1.9–2.3× faster.
- **Baselines:** `ks_moe_grouped_gemm` + `ks_moe_permute`/`ks_moe_unpermute` (fused experts); `ks_cross_entropy` / `ks_fused_linear_cross_entropy`; `ks_rope`; `ks_swiglu` (SwiGLU expert MLP).
- **Source:** <https://github.com/unslothai/unsloth/discussions/4020>

### cut-cross-entropy (CCE) — `apple/ml-cross-entropy`  ⟶ confidence: **medium**
- **Version / date:** Apple repo last pushed 2025-09-23 (no 2026 H1 release); Unsloth fork pushed 2025-01-19. Algorithm vendored into Liger/Unsloth in 2026. *(not 2026 H1; canonical reference for the chunked CE op.)*
- **Ops:** fused linear cross-entropy (no logit materialization), log-sum-exp reduction in SRAM, gradient-filtered CE backward.
- **Hardware:** NVIDIA GPUs via Triton; torch.compile fallback on MacOS/unsupported Triton.
- **dtypes:** BF16/FP16 logits, FP32 accumulate.
- **License:** Apple sample-code license (NOASSERTION).
- **Install:** `pip install cut-cross-entropy` (patches Llama/Mistral/Phi3/Gemma2).
- **Claimed perf:** computes linear-CE without materializing the `[tokens, vocab]` logits; large memory reduction on large-vocab LM heads.
- **Baselines:** `ks_fused_linear_cross_entropy` (loss.h) — CCE is the canonical reference for this chunked, no-logit-materialization kernel; also `ks_cross_entropy`.
- **Source:** <https://github.com/apple/ml-cross-entropy>

### NVIDIA Apex — `NVIDIA/apex`  ⟶ confidence: **high**
- **Version / date:** continuously updated (no semver releases); classic CUDA training fusions; superseded for FP8 by TransformerEngine. *(not a 2026 H1 release; standing optimizer/norm baseline.)*
- **Ops:** FusedLayerNorm / FusedRMSNorm, FusedAdam / FusedAdamW / FusedLAMB, fused scaled-masked softmax, multi-tensor global grad-norm / L2 clip.
- **Hardware:** NVIDIA sm70+ → sm100 (CUDA).
- **dtypes:** FP16/BF16/FP32 (fp32 accumulate).
- **License:** BSD-3-Clause.
- **Install:** build from source (`pip install -v --no-build-isolation .`, CUDA extensions).
- **Claimed perf:** reference fused CUDA training kernels (LayerNorm/RMSNorm, Adam/LAMB, grad-norm); baseline for many training stacks.
- **Baselines:** `ks_adamw` (FusedAdamW), `ks_global_grad_norm` (multi-tensor L2 norm); `ks_layer_norm` / `ks_rms_norm` (FusedLayerNorm/RMSNorm).
- **Source:** <https://github.com/NVIDIA/apex>

---

## Notes on benchability

- **Single-GPU only.** kernel-set is a single-GPU C-ABI library. The four EP comm
  libraries (**DeepEP, NCCL-EP, pplx-garden**) and **Ring/Context-Parallel attention**
  are **multi-GPU** — they wrap, not replace, the per-rank kernel-set ops. They are
  cataloged for completeness but `baselines.yaml` flags them `benchable: false`.
- **Blackwell-only / Hopper-only.** Many 2026 H1 kernels (FA4, SageAttention3, FP4
  GEMM/MoE paths) need sm100+ or sm90. On the project's Colab targets (T4/L4/A100)
  these will `skip`. `baselines.yaml` records `gpu_arch_required` so the bench can
  skip-with-reason instead of erroring.
- **One package, many ops.** FlashInfer, torchao, and DeepGEMM each appear in
  multiple domains above (and as multiple `maps_to` entries in `baselines.yaml`):
  install once, compare against several kernel-set ops.
