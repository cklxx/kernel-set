# Kernel Landscape — High-Performance LLM Kernels (≈2026)

This document surveys the real-world landscape of high-performance kernels for LLM
inference and training, names the **best-in-class kernel per operator** (varying by
GPU and dtype), and classifies the major **model families** by their architectural
operators. It is the provenance reference for `kernel-set`: every operator we expose
is modeled on the algorithms documented here (clean-room implementations).

GPU shorthand used throughout:

| Tag | GPU | Arch | Key tensor-core / dtype capability |
|---|---|---|---|
| **sm80** | A100 / A30 | Ampere | FP16/BF16/TF32 MMA, INT8; **no FP8 tensor cores**, no WGMMA/TMA |
| **sm86/sm89** | A10/A40 (86), L4/RTX-4090 (89) | Ada | adds **FP8 (e4m3/e5m2) tensor cores**, INT8/INT4; no WGMMA/TMA |
| **sm90(a)** | H100 / H200 | Hopper | **WGMMA + TMA**, FP8 tensor cores, async warpgroup pipelines |
| **sm100** | B200 / GB200 | Blackwell | **5th-gen tensor cores**, FP8 + **FP4 (mxfp4/nvfp4)**, TMEM, tcgen05 |

> sm75 (T4, Turing) is FP16-MMA only (no BF16/FP8) — listed where relevant for
> the project's T4 Colab target.

---

## 1. Library survey — ops provided + hardware notes

### Attention engines

- **FlashAttention-2** (Dao-AILab). Tiled, IO-aware fused attention; causal/dense,
  varlen, GQA/MQA, ALiBi, sliding window; forward + backward (training). The workhorse
  on **sm80** (A100) and **sm89** (Ada). Head dims up to 256. FP16/BF16. Backward is
  the default training-attention everywhere outside Hopper.
- **FlashAttention-3** (Dao-AILab + Colfax + NVIDIA). Hopper-only (**sm90**) rewrite
  using **WGMMA** async tensor cores, **TMA**, warp-specialization (producer/consumer),
  and softmax/matmul interleave. 1.5–2.0× over FA2 in FP16 (~740 TFLOPS, ~75% of H100
  peak); **FP8 path** ~1.2 PFLOPS with block-quant + incoherent processing (2.6× lower
  error than naive FP8). Best prefill+training attention on H100. FP16/BF16/FP8.
- **FlashInfer** (flashinfer-ai; MLSys'25 best paper, now NVIDIA-backed). KV-cache-centric
  serving kernels: **paged decode**, ragged/prefill, **MLA** (DeepSeek) decode+prefill
  with matrix absorption, FP8 KV-cache, prefix-cascade/shared-prefix, speculative decode.
  Three-stage **plan/run** API (JIT-specialize → schedule/load-balance → CUDA-graph run).
  NVIDIA ships its newest TRT-LLM kernels here. Backends for vLLM and SGLang. sm80→sm100.
- **FlashMLA** (DeepSeek). Hopper-tuned MLA decode kernel (paged, varlen) for DeepSeek
  V2/V3/R1; very high BW utilization on H800/H100. sm90.
- **xFormers** (Meta). `memory_efficient_attention` (Cutlass/Flash/Triton backends),
  block-diagonal/varlen masks, broad dtype + arch coverage incl. older GPUs (sm70+).
  Good fallback when FA isn't available for an arch/head-dim.
- **PyTorch SDPA**. `scaled_dot_product_attention` dispatches to FA2/cuDNN/mem-efficient/
  math; the universal portable baseline. cuDNN backend is strong on Hopper.

### GEMM / linear

- **CUTLASS** (NVIDIA). Template GEMM/conv library; the substrate under most custom
  kernels. CUTLASS 3.x exposes **WGMMA+TMA** (sm90) and **tcgen05** (sm100); grouped
  GEMM, epilogue fusion (bias/act/scale), FP16/BF16/TF32/FP8/INT8/FP4. **CuTe** DSL.
- **cuBLAS / cuBLASLt**. Vendor GEMM; LtMatmul with epilogue fusion + FP8 scaling. The
  default fp16/bf16 dense GEMM on every arch; **TransformerEngine** drives it for FP8.
- **DeepGEMM** (DeepSeek). Clean **FP8 GEMM with fine-grained (block/tile) scaling** +
  **two-level CUDA-core accumulation** to fix FP8 accumulation error. Dense **and MoE
  grouped** (contiguous + masked) GEMM. JIT, single core kernel. **Hopper-only** (~1550
  TFLOPS on H800); Blackwell support added later. Powers V3/R1 train+infer.
- **Marlin / Machete** (IST-DASLab → Neural Magic/vLLM). **Marlin**: FP16×INT4 (**W4A16**)
  mixed-input GEMM, near-ideal ~4× speedup, GPTQ/AWQ; **sm80/sm86/sm89** sweet spot.
  **GPTQ-Marlin** also does W8A16/W4A8. **Machete**: CUTLASS-3 successor for **Hopper
  (sm90a)** using WGMMA; W4A16/W8A16, weight prepacking. Best W4A16 on H100.
- **TransformerEngine** (NVIDIA). FP8 (and FP4 on Blackwell) **training+inference**:
  fp8 Linear/LayerNormLinear/LayerNormMLP, **delayed scaling** + amax history, cuDNN FMHA.
  sm89/sm90/sm100. The standard FP8 training stack.
- **TensorRT-LLM** (NVIDIA). Production inference engine: custom fused attention (context
  + generation FMHA, FP8 context FMHA), paged-KV, in-flight batching, fused GEMM+SwiGLU
  (`low_latency_gemm_swiglu`), W4A16/W4A8/W8A8/FP8/FP4 quant. Increasingly sources kernels
  from FlashInfer. sm80→sm100.

### Quantization kernels

- **AWQ** (MIT-HAN-Lab). Activation-aware W4A16; salient-channel scaling + fused
  dequant-GEMM kernels (GEMM/GEMV). sm75+; integrated in vLLM/TRT-LLM/SGLang.
- **GPTQ + ExLlamaV2/V3** (AutoGPTQ; turboderp). GPTQ second-order PTQ to 2–4 bit.
  ExLlamaV2 (EXL2): per-layer mixed bit-rate + fast fused W4A16 kernels for consumer GPUs.
  **EXL3**: trellis quant coherent to ~1.6 bpw with fused kernels. sm75+ (consumer focus).
- **bitsandbytes**. LLM.int8() (W8A8 with outlier fp16 path), NF4/FP4 (QLoRA dequant).
  Broad arch coverage; the training-time 4-bit/8-bit standard.
- **vLLM / compressed-tensors quant kernels**. Runtime FP8 (per-tensor/channel/block),
  INT8 W8A8 (SmoothQuant-style, CUTLASS), FP8/INT8 KV-cache quant, FP4 (Blackwell).
  Dispatches to Marlin/Machete/CUTLASS by shape+arch.

### Norm / RoPE / activation / fused training kernels

- **Liger-Kernel** (LinkedIn). Triton training kernels, HF-compatible: **RMSNorm,
  LayerNorm, RoPE, SwiGLU/GeGLU, CrossEntropy, FusedLinearCrossEntropy** (chunked, avoids
  materializing logits → big memory save), plus post-training losses (DPO/ORPO/SimPO/JSD).
  ~20% throughput / ~60% memory wins; arch-portable (Triton). The go-to fused training kernels.
- **Apex** (NVIDIA). FusedLayerNorm/FusedRMSNorm, **FusedAdam/FusedLAMB**, fused softmax;
  classic CUDA training fusions.
- **DeepSpeed**. CPU/fused Adam, ZeRO, inference kernels; training infra + some fused ops.
- **FlashInfer / vLLM / SGLang norm+rope+activation**. Each ships fast inference RMSNorm
  (+fused add-residual), RoPE (NeoX/GPT-J/Llama3/YaRN/long-rope), SiLU-mul (SwiGLU) kernels.
- **TransformerEngine norm**. Fused LayerNorm/RMSNorm with FP8 cast for FP8 pipelines.

### MoE kernels

- **DeepGEMM grouped GEMM** — FP8 grouped (contiguous/masked) experts; best FP8 MoE on Hopper.
- **vLLM fused MoE** — Triton `fused_moe` (gate+permute+grouped-GEMM+unpermute), **CUTLASS
  grouped FP8 (`cutlass_moe_fp8`)**, and DeepGEMM-backed path; dispatcher picks by shape/dtype/arch.
- **SGLang / EP MoE** — expert-parallel MoE, DeepSeek group-limited top-k gating, fused routing.
- **DeepEP** (DeepSeek) — expert-parallel all-to-all dispatch/combine comm kernels (NVLink/RDMA).
- **MegaBlocks / ScatterMoE / Triton grouped GEMM** — dropless block-sparse MoE for training.

### Sampling / loss / optimizers

- **FlashInfer sampling** — fused top-k/top-p/min-p, **rejection-sampling** (sorting-free),
  temperature; for speculative decoding & serving.
- **vLLM / SGLang sampling** — fused logits-processing + categorical sampling kernels.
- **Liger / Apex cross-entropy** — fused CE and **fused-linear-CE** (chunked) for training.
- **Apex / DeepSpeed / bnb optimizers** — FusedAdamW, 8-bit Adam (bnb), SGD-momentum,
  global grad-norm clipping.

### State-space / conv (non-attention)

- **mamba-ssm** (state-spaces). Mamba/**Mamba-2** selective-scan (`selective_scan`,
  **chunked SSD** scan), Mamba-3 (2026). Used by hybrid models (Jamba, Zamba, Nemotron-H).
- **causal-conv1d** — fused short causal depthwise conv1d (fwd+bwd) for Mamba/SSM/conv-mixers.

---

## 2. Best-in-class kernel per operator

Notation: pick depends on **(GPU arch, dtype)**. "FA2/FA3/FI" = FlashAttention-2/3 / FlashInfer.

| Op | Best-in-class | By GPU | By dtype |
|---|---|---|---|
| **Attention — prefill / training fwd+bwd** | **FA3** (sm90), **FA2** (sm80/sm89) | A100→FA2; L4→FA2; H100→FA3 (cuDNN/TE alt) | FP16/BF16 everywhere; FP8 prefill only FA3 on H100 |
| **Attention — decode (paged)** | **FlashInfer** paged decode | all archs; FA2-decode/vLLM PagedAttn as fallback on sm80 | FP16/BF16 + **FP8 KV-cache** (sm89/sm90) |
| **MLA (DeepSeek)** | **FlashMLA** (H100) / **FlashInfer MLA** (portable) | H100→FlashMLA; A100/L4→FlashInfer MLA; TRT-LLM MLA in TRT | BF16; FP8 KV on Hopper |
| **GEMM fp16/bf16** | **cuBLASLt** (baseline) / **CUTLASS** (fused epilogue) | all archs (WGMMA path auto on sm90) | FP16/BF16/TF32 |
| **FP8 GEMM (dense)** | **DeepGEMM** (sm90) / **TransformerEngine→cuBLASLt** (sm89/sm90) | H100→DeepGEMM; L4→TE/cuBLASLt FP8; A100→**not supported** (use bf16) | e4m3 (fwd) / e5m2; block-scaled |
| **W4A16 GEMM** | **Machete** (sm90) / **Marlin** (sm80/86/89) | H100→Machete; A100/L4→Marlin; consumer→ExLlamaV2/V3 | INT4 weight × FP16/BF16 act |
| **W8A8 GEMM** | **CUTLASS INT8** (SmoothQuant) / **TRT-LLM** | all archs w/ INT8 tensor cores (sm75+) | INT8×INT8→INT32, fp32 dequant |
| **RMSNorm** | **Liger** (train) / **FlashInfer-vLLM** (infer, +fused residual) | all archs | any; **fp32 accumulate** |
| **LayerNorm** | **Apex FusedLayerNorm** / **Liger** (train), vLLM (infer) | all archs | any; fp32 accumulate |
| **RoPE** | **Liger** (train) / **FlashInfer-vLLM** (infer) | all archs | NeoX/GPT-J/Llama3/YaRN/long-rope |
| **SwiGLU / GeGLU** | **Liger** (train) / vLLM SiLU-mul (infer); **fused GEMM+SwiGLU** on TRT-LLM FP8 | H100 gets fused-GEMM+act; all archs for elementwise | any |
| **MoE gating (top-k / group)** | **vLLM/SGLang fused gate** (softmax/sigmoid top-k + **DeepSeek group-limited**) | all archs | fp32 routing math |
| **MoE grouped GEMM** | **DeepGEMM grouped** (FP8, sm90) / **CUTLASS grouped** / **vLLM Triton fused_moe** | H100→DeepGEMM/CUTLASS FP8; A100/L4→Triton fused_moe / CUTLASS | FP8 (Hopper) else BF16/INT4 |
| **FP8 quant (cast/scale)** | **TransformerEngine** (delayed scaling) / **DeepGEMM** (block scale) | sm89/sm90/sm100; A100 N/A | e4m3/e5m2, per-tensor/channel/block |
| **INT8 quant** | **CUTLASS / vLLM** (per-tensor & per-token dynamic) | sm75+ | INT8 |
| **Sampling** | **FlashInfer** (top-k/top-p/min-p + rejection) | all archs | fp32 probs |
| **Cross-entropy** | **Liger FusedLinearCrossEntropy** (chunked) | all archs | fp32 accumulate; bf16 logits |
| **Optimizers (AdamW/…)** | **Apex FusedAdamW** / **bnb 8-bit Adam** (memory) / DeepSpeed CPUAdam (offload) | all archs | fp32 master / bf16 / 8-bit states |

**Arch cheat-sheet for selection:**
- **A100 (sm80):** no FP8 — use **BF16 GEMM (cuBLASLt/CUTLASS)**, **Marlin W4A16**, **FA2** attention, FlashInfer paged decode.
- **L4 (sm89):** FP8 tensor cores exist → FP8 GEMM via **TE/cuBLASLt**, **Marlin W4A16**, **FA2** attention, FP8 KV-cache decode. No WGMMA → DeepGEMM/Machete/FA3 not applicable.
- **H100 (sm90):** full stack — **FA3**, **DeepGEMM FP8**, **Machete W4A16**, **FlashMLA**, fused GEMM+SwiGLU, FP8 KV-cache, CUTLASS/DeepGEMM grouped MoE.

---

## 3. Model family classification

Columns: **Attention** (variant + rope), **Norm**, **MLP/MoE**, **Quant** commonly used.
RoPE θ note: classic = 10k; "high-θ" (1e6) used by long-context Llama-3/Qwen2.5+.

| Family | Attention + RoPE | Norm | MLP / MoE | Quant |
|---|---|---|---|---|
| **Llama 2** | MHA (7/13B), **GQA** (34/70B); RoPE θ=10k | **RMSNorm** (pre) | dense **SwiGLU** | GPTQ/AWQ W4A16, FP8 |
| **Llama 3 / 3.1** | **GQA**; RoPE θ=500k, **Llama3 rope scaling** (128K) | RMSNorm | dense SwiGLU | FP8, W4A16 (Marlin/Machete), W8A8 |
| **Llama 4** | **GQA + iRoPE** (interleaved RoPE / **NoPE** blocks), some layers no positional enc | RMSNorm | **MoE** (top-k, shared+routed) + dense layers; SwiGLU; early-fusion multimodal | FP8, W4A16 |
| **Qwen 1.5 / 2** | GQA; RoPE θ=1e6; QKV **bias** | RMSNorm | dense SwiGLU; Qwen2-MoE variants | GPTQ/AWQ W4A16, FP8 |
| **Qwen 2.5** | GQA; high-θ RoPE + **YaRN** long-ctx | RMSNorm | dense SwiGLU; MoE (A14B etc.) | GPTQ/AWQ/GGUF, FP8, W8A8 |
| **Qwen 3 / 3-Next** | GQA + **QK-norm**; RoPE θ=1e6, YaRN/DCA; 3-Next **hybrid (gated-DeltaNet + attention)** | RMSNorm | dense + **MoE (30B-A3B, 235B-A22B)**, ultra-sparse; SwiGLU | FP8, AWQ/GPTQ W4A16 |
| **Mistral 7B** | GQA + **sliding-window attention** (4096); RoPE θ=1e6 | RMSNorm | dense SwiGLU | AWQ/GPTQ, FP8 |
| **Mixtral 8x7B/8x22B** | GQA (+SWA on 8x7B); RoPE | RMSNorm | **MoE** top-2 of 8, SwiGLU experts | AWQ/GPTQ W4A16, FP8 |
| **DeepSeek V2** | **MLA** (low-rank KV + **decoupled RoPE** key); θ via YaRN | RMSNorm | **DeepSeekMoE**: fine-grained routed + **shared** experts | FP8, AWQ |
| **DeepSeek V3 / R1** | **MLA** (decoupled-rope key, matrix absorption) | RMSNorm | DeepSeekMoE (256 routed +1 shared, top-8) + **group-limited, aux-loss-free sigmoid gating**; **MTP**; SwiGLU | **native FP8 (block-scaled, DeepGEMM)**, W4A16 |
| **Gemma 2** | GQA, **local/global** interleaved SWA; **logit soft-cap**; RoPE | RMSNorm **pre+post** | **GeGLU** (gelu-tanh) | int8/int4, FP8 |
| **Gemma 3** | GQA, **5 local : 1 global** SWA (win 1024), **QK-norm** (replaces soft-cap); RoPE (global θ=1e6) | RMSNorm pre+post | GeGLU | int4/int8, FP8 |
| **Phi-3** | MHA/GQA; RoPE (+long-rope/SU-scaling for 128K) | RMSNorm | dense SwiGLU | int4 (AWQ/GPTQ), FP8 |
| **Phi-4** | GQA; RoPE | RMSNorm | dense SwiGLU | int4, FP8 |
| **GPT-OSS (20B/120B)** | **GQA (group 8) + learned attention sinks**; **alternating full / 128-tok SWA**; RoPE 128K | RMSNorm | **MoE** (32 or 128 experts, top-4, **no shared expert**), SwiGLU | **native MXFP4** experts (Blackwell/Hopper), FP8 |
| **Yi** | GQA (Llama-like); RoPE θ=1e6 (200K ctx variants) | RMSNorm | dense SwiGLU | GPTQ/AWQ, FP8 |
| **Baichuan 2** | MHA (7B) / **ALiBi** (13B); 7B uses RoPE | RMSNorm | dense SwiGLU | GPTQ/AWQ |
| **ChatGLM / GLM-4** | **MQA/GQA**; **rotary on half-dim (2D/partial RoPE)**; QKV bias | RMSNorm (post-style in older GLM) | SwiGLU; GLM-4.5/4.6 **MoE** | GPTQ/AWQ W4A16, FP8 |
| **Falcon** | **MQA** (40B/180B) / MHA (7B); **parallel attn+MLP**; RoPE | **LayerNorm** | dense GeLU MLP | GPTQ, bnb |
| **Command-R / R+** | GQA; RoPE; **no bias**, tied embeddings | LayerNorm | dense SwiGLU | bnb/GPTQ, FP8 |
| **StableLM** | MHA/GQA; **partial RoPE** (rotary_pct); QK-LayerNorm (2 variants) | LayerNorm / RMSNorm | dense SwiGLU | GPTQ/AWQ |
| **MPT** | MHA + **ALiBi** (no RoPE) | LayerNorm (no bias) | dense GeLU | GPTQ, bnb |
| **GPT-NeoX** | MHA; **partial/GPT-J-style RoPE**; **parallel attn+MLP** | LayerNorm | dense GeLU | GPTQ, bnb |
| **OLMo / OLMo-2** | MHA/GQA; RoPE; OLMo-2 **QK-norm + reordered (post) norm** | LayerNorm (OLMo-1, non-param) / RMSNorm (OLMo-2) | dense SwiGLU | FP8, GPTQ |

**Cross-cutting notes**
- **Norm:** RMSNorm dominates modern decoders; LayerNorm persists in older/“GPT-lineage”
  families (Falcon, MPT, NeoX, GPT-2-style). Gemma uses **pre+post** RMSNorm; Gemma3/Qwen3/
  OLMo-2 add **QK-norm** for stability.
- **Attention:** trend is MHA → **GQA** (KV-cache savings) → **MLA** (DeepSeek, max KV
  compression) and **hybrid SSM/linear-attention** (Qwen3-Next, Jamba, Nemotron-H).
  Local/global **sliding-window** interleave (Mistral, Gemma2/3, GPT-OSS) cuts long-ctx cost.
- **RoPE:** classic θ=10k → **high-θ (1e6/5e5)** + **YaRN / Llama3-scaling / long-rope /
  DCA** for 128K+ contexts; **partial RoPE** (GPT-J/NeoX/StableLM/GLM); **NoPE/iRoPE** (Llama 4).
- **MLP/MoE:** **SwiGLU** is the default dense MLP; **GeGLU** for Gemma. MoE is now mainstream
  (Mixtral, DeepSeek, Qwen3, Llama 4, GPT-OSS, GLM-4.5) with **top-k routing**, often a
  **shared expert** (DeepSeek/Qwen) and **group-limited / aux-loss-free** gating (DeepSeek V3).
- **Quant:** inference defaults — **W4A16** (AWQ/GPTQ via Marlin/Machete) for memory-bound
  decode; **FP8** (Hopper/Ada/Blackwell) for compute-bound; **W8A8** (SmoothQuant) for INT8-only
  paths; **MXFP4/NVFP4** emerging (GPT-OSS native, Blackwell). KV-cache quant: FP8 (Hopper/Ada),
  INT8.

---

## Sources

- FlashAttention-3: <https://tridao.me/blog/2024/flash3/>, <https://arxiv.org/abs/2407.08608>, <https://pytorch.org/blog/flashattention-3/>
- FlashInfer: <https://arxiv.org/pdf/2501.01005>, <https://docs.flashinfer.ai/api/attention.html>, <https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html>
- SGLang attention backends: <https://docs.sglang.io/advanced_features/attention_backend.html>
- DeepGEMM: <https://github.com/deepseek-ai/DeepGEMM>, <https://www.marktechpost.com/2025/02/25/deepseek-ai-releases-deepgemm-an-fp8-gemm-library-that-supports-both-dense-and-moe-gemms-powering-v3-r1-training-and-inference/>
- Marlin / Machete: <https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel>, <https://github.com/vllm-project/vllm/pull/7174>
- Liger-Kernel: <https://github.com/linkedin/Liger-Kernel>, <https://arxiv.org/pdf/2410.10989>
- TransformerEngine / TensorRT-LLM: <https://github.com/NVIDIA/TransformerEngine>, <https://developer.nvidia.com/tensorrt-llm>
- AWQ/GPTQ/ExLlama: <https://pytorch.org/blog/accelerating-triton/>, <https://www.johal.in/exllamav3-quantizers-gptq-awq-for-consumer-gpus-2025/>
- vLLM fused MoE / CUTLASS FP8 MoE: <https://docs.vllm.ai/en/latest/design/moe_kernel_features/>, <https://github.com/vllm-project/vllm/pull/13972>
- DeepSeek-V3: <https://arxiv.org/html/2412.19437v1>
- Gemma 3: <https://arxiv.org/html/2503.19786v1>
- GPT-OSS: <https://huggingface.co/blog/welcome-openai-gpt-oss>, <https://blog.vllm.ai/2025/08/05/gpt-oss.html>
- Llama 4 / Qwen3: <https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison>, <https://arxiv.org/pdf/2505.09388>
- Mamba-2 / causal-conv1d: <https://github.com/state-spaces/mamba>
