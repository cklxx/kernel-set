<h1 align="center">kernel-set</h1>

<p align="center"><b>Fast LLM inference & training kernels behind one C ABI — call them from Python / Rust / Go / TypeScript, and let it auto-pick the strongest kernel for your op + GPU.</b></p>

<p align="center">
  <a href="https://pypi.org/project/kernel-set/"><img src="https://img.shields.io/pypi/v/kernel-set?color=blue" alt="PyPI"></a>
  <a href="https://github.com/cklxx/kernel-set/actions/workflows/ci.yml"><img src="https://github.com/cklxx/kernel-set/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

---

## What it is

You write `ks.dispatch.rms_norm(x, w)`. Under the hood it runs the **best kernel
installed on your machine** — FlashAttention, FlashInfer, vLLM, SGLang, DeepGEMM,
Marlin, … — and falls back to kernel-set's own portable kernel when nothing else
is there. Same call, every GPU, every language.

- **One C ABI, four languages.** 84 operators (attention, GEMM, norm, RoPE,
  gated-MLP, MoE, quant, sampling, loss, optimizer, **Mamba SSM**, **linear
  attention** — gated-DeltaNet / GLA / RWKV-7) behind one `libkernel_set` —
  Python / Rust / Go / TypeScript bind the *same* library, no GPU toolchain of
  their own.
- **Auto best-kernel selection.** **50 logical ops** route per-`(op, GPU, dtype)`
  to the strongest of **28 backends** (FlashAttention-3, FlashInfer, FlashMLA,
  DeepGEMM, Marlin/Machete, SGLang, vLLM, FLA, mamba-ssm, FBGEMM, GemLite, Quack,
  BitBLAS, …); kernel-set's clean-room kernels are the always-there fallback.
  ([how it decides](docs/OPTIMAL_SELECTION.md))

## Install

```bash
pip install kernel-set     # Linux x86_64 · NVIDIA driver (CUDA 12.x) · torch optional
```

The wheel bundles a prebuilt `libkernel_set.so` (sm75–sm120, static CUDA runtime)
— nothing to compile. Other platforms/archs build from source (below).

## 30-second example

```python
import torch, kernel_set as ks

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.ones(4096, device="cuda", dtype=torch.bfloat16)

y = ks.dispatch.rms_norm(x, w)          # runs the best RMSNorm available
ks.dispatch.which("rmsnorm")            # who got picked? -> 'flashinfer' or 'kernel-set'
ks.dispatch.which("fp8_gemm_blockwise", gpu="h100", dtype="fp8")   # -> 'deep_gemm'
```

No GPU handy? `ks.dispatch.available()` still prints the routing table.

## The strategy (honest)

- **Production defaults → best installed provider.** FlashAttention, FlashInfer,
  Liger, vLLM, SGLang, DeepGEMM, cuBLAS, etc. lead the route when they are the
  right production kernel; kernel-set is the portable fallback.
- **kernel-set wins are promoted only with enough evidence.** Memory-bound
  kernels such as RMSNorm, fused-add-RMSNorm, RoPE, SwiGLU/GeGLU, elementwise and
  AdamW are competitive, but shape-sensitive or provider-incomplete rows stay as
  benchmark evidence until a shape gate or full SOTA comparison justifies a
  default change.

> **Measured on real GPUs:** checked-in benchmark runs cover **L4 (sm89)**,
> **A100 (sm80)**, **H20 (sm90, Hopper)**, and **RTX PRO 6000 Blackwell
> (sm120)**. The durable data lives in `benchmarks/results/runs/`; the summary
> below is generated from those JSON runs. CUDA release builds cover sm75–sm120;
> HIP/ROCm remains behind a build flag.

<!-- BENCHMARK_SUMMARY:START -->
## Benchmarks

Canonical benchmark data is checked in under [`benchmarks/results/runs/`](benchmarks/results/runs/) and summarized in [`benchmarks/results/README.md`](benchmarks/results/README.md). Current coverage: **24 runs**, **1125/1284 ok rows**, GPUs: NVIDIA A100-SXM4-40GB sm80, NVIDIA H20 sm90, NVIDIA L4 sm89, NVIDIA RTX PRO 6000 Blackwell Server Edition sm120.

Rows are scoped by their suite: `sota` rows compare installed production providers; `kernel_set` rows are diagnostic kernel-set/reference runs and are not promoted to default routing by themselves.

Representative large-kernel rows:

| GPU | op | shape | measured impl | latency |
|---|---|---|---|---:|
| NVIDIA H20 sm90 | `attn_prefill` | `b=1,seq=2048,qh=32,kvh=8,hd=128` | `flashinfer` | 335.7 us |
| NVIDIA H20 sm90 | `attn_decode` | `seqs=64,ctx=2048,qh=32,kvh=8,hd=128` | `flashinfer` | 340.8 us |
| NVIDIA H20 sm90 | `mla_decode` | `seqs=64,ctx=2048,h=128,lora=512,rope=64` | `flash-mla` | 303.2 us |
| NVIDIA H20 sm90 | `gemm` | `M=4096,N=4096,K=4096` | `torch-cublas` | 1039.8 us |
| NVIDIA H20 sm90 | `fp8_gemm` | `M=4096,N=4096,K=4096` | `torch-scaled-mm` | 536.7 us |
| NVIDIA A100-SXM4-40GB sm80 | `fp8_gemm_blockwise` | `M=4096,N=4096,K=4096,bn=128,bk=128` | `kernel-set` | 79622.7 us |
| NVIDIA H20 sm90 | `w8a8` | `M=4096,N=4096,K=4096` | `kernel-set` | 10002.8 us |
| NVIDIA H20 sm90 | `w4a16` | `M=4096,N=4096,K=4096` | `kernel-set` | 19182.0 us |

Memory-bound provider highlights:

| GPU | op | fastest measured impl | runner-up | ratio |
|---|---|---|---|---:|
| NVIDIA H20 sm90 | `fused_add_rmsnorm` | `flashinfer-norm` 15.6 us | `sgl-norm` 17.2 us | 1.10x |
| NVIDIA L4 sm89 | `fused_add_rmsnorm` | `kernel-set` 2333.7 us | `flashinfer-norm` 4622.8 us | 1.98x |
| NVIDIA H20 sm90 | `swiglu` | `kernel-set` 12.1 us | `flashinfer-act` 12.5 us | 1.03x |
| NVIDIA L4 sm89 | `swiglu` | `kernel-set` 12.3 us | `flashinfer-act` 13.3 us | 1.08x |
| NVIDIA A100-SXM4-40GB sm80 | `swiglu` | `kernel-set` 8.2 us | `eager` 33.8 us | 4.12x |
| NVIDIA RTX PRO 6000 Blackwell Server Edition sm120 | `swiglu` | `kernel-set` 6.1 us | `eager` 12.3 us | 2.02x |

Engine smoke:

Kernel-coverage rows prove call-path coverage and token parity; checked-in kernel benchmark tables provide provider-selection evidence.

| model / GPU | engine | scope | new tok/s | token match | notes | source |
|---|---|---|---:|---|---|---|
| Qwen/Qwen3-8B / NVIDIA L4 (sm89, bf16) | `vllm` | vLLM LLM.generate | 16.24 | yes | historical baseline from 20260618-qwen3-8b-l4-long-greedy-vllm | [20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json](benchmarks/results/inference/20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json) |
| Qwen/Qwen3-8B / NVIDIA L4 (sm89, bf16) | `sglang` | SGLang Engine.generate | 16.40 | no (104/197) | Colab L4 SGLang Engine.generate; greedy output diverged from HF reference after prompt + 3 generated tokens | [20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json](benchmarks/results/inference/20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json) |
| Qwen/Qwen3-8B / NVIDIA L4 (sm89, bf16) | `kernel_set_engine` | single-request Python engine; torch/cuBLAS linears + ks RMSNorm/RoPE/KV write/short-decode/SwiGLU + shape-aware embedding/attention | 15.65 | no (136/197) | shape-aware kernel-set engine composition from Qwen3 kernel microbench; Python loop/allocation and unfused Q/K/V + gate/up remain; historical baseline from 20260618-qwen3-8b-l4-long-greedy-vllm | [20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json](benchmarks/results/inference/20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json) |

Kernel coverage for the composed kernel-set engine:

| engine | covered kernel-set kernels | remaining torch/Python path | counted calls |
|---|---|---|---|
| `kernel_set_engine` | `ks_embedding_lookup(auto multi-token)`<br>`ks_rms_norm`<br>`ks_rope_gather`<br>`ks_paged_attn_decode(auto short-context)`<br>`ks_reshape_and_cache`<br>`ks_swiglu` | embedding single-token=torch<br>linear=torch/cuBLAS<br>attention prefill/long-context=torch SDPA<br>argmax=torch<br>residual add<br>tensor reshape/view/allocation<br>Python request/decode loop<br>paged block scheduler | `embedding_lookup`=1<br>`paged_attn_decode`=3420<br>`reshape_and_cache`=3456<br>`rmsnorm`=13920<br>`rope`=3456<br>`swiglu`=3456<br>`torch_argmax`=96<br>`torch_attention_prefill`=36<br>`torch_embedding`=95<br>`torch_linear`=24288 |

Quantized serving-engine comparison:

No vLLM/SGLang quantized serving baseline is checked in yet. HF/Transformers rows in JSON are correctness references only, not README performance baselines.

<!-- BENCHMARK_SUMMARY:END -->

## Kernel Optimization Notes

Latest kernel-only A/B used Colab L4 (sm89), fp16, L2-flushed CUDA-event timing
with `bench.py --ops quant,reshape_and_cache --target-ms 100`. The baseline was
the pre-patch `HEAD`; the final checked-in run is
[`20260618-l4-kernel-opt-final-nvidia-l4-fp16.json`](benchmarks/results/runs/20260618-l4-kernel-opt-final-nvidia-l4-fp16.json).

| kernel | shape | baseline | current | result |
|---|---|---:|---:|---:|
| `quantize_fp8_group` | `rows=4096,cols=4096,g=128` | 487.4 us / 103.3 GB/s | 474.1 us / 106.2 GB/s | 1.03x |
| `quantize_fp8_group` | `rows=8192,cols=8192,g=128` | 2234.4 us / 90.1 GB/s | 2128.9 us / 94.6 GB/s | 1.05x |
| `reshape_and_cache` | `tokens=4096,kvh=8,hd=128,blk=16` | 188.4 us / 178.1 GB/s | 188.4 us / 178.1 GB/s | no measured win |

`quantize_fp8_group` now caches each <=256-column group in shared memory before
the fp8 cast pass, avoiding one global reread for the common group-128 path.
`reshape_and_cache` now performs direct same-dtype K/V copies; it is kept as a
simple instruction cleanup, not a claimed performance improvement.

Systematic attention triage added explicit split-K workspace decode APIs and
bench rows. Colab L4 fp16 (`bench_sota.py --ops attention_decode,mla_decode
--target-ms 80`) shows split-K is only a small paged-decode win at the current
multi-sequence shapes and is a loss for the scalar MLA implementation:

| kernel | shape | baseline | split-K | result |
|---|---|---:|---:|---:|
| `paged_attn_decode` | `seqs=64,ctx=2048,qh=32,kvh=8,hd=128` | 7501.3 us | 7240.2 us | 1.04x |
| `paged_attn_decode` | `seqs=256,ctx=1024,qh=32,kvh=8,hd=128` | 14669.8 us | 14616.6 us | 1.00x |
| `mla_decode` | `seqs=64,ctx=2048,h=128,lora=512,rope=64` | 78373.4 us | 103387.1 us | 0.76x |

Conclusion: decode attention still needs GQA KV reuse / token-head tiling, and
MLA needs a tensor-core/FlashMLA-class algorithm; split-K alone is diagnostic,
not a production default.

## Try it on a real model

```bash
python examples/eval_model.py --model Qwen/Qwen2.5-0.5B-Instruct   # or a Gemma / Llama id
```

Hot-swaps kernel-set's RMSNorm/RoPE/SwiGLU·GeGLU into a stock HuggingFace model
and checks it against the original: **bit-identical** output on Gemma-2-2B (64/64
greedy tokens), top-1-correct on Qwen2.5, **3–9× faster** per op vs eager torch.
([results](benchmarks/results/real_model_eval.md))

## Operators

| Category | Header | Highlights |
|---|---|---|
| Attention | `attention.h` | FlashAttention-2 prefill (dense+varlen), paged decode, **MLA**, KV-cache, backward |
| GEMM | `gemm.h` | tensor-core fp16/bf16, bias+act, batched, **W8A8 / W4A16 / FP8 / FP8-blockwise** |
| Norm | `norm.h` | RMSNorm (+fused residual), LayerNorm, backward |
| RoPE | `rope.h` | NeoX & interleaved, gathered, GQA, backward |
| Activation | `activation.h` | SiLU/GeLU/ReLU, **SwiGLU/GeGLU** (+backward) |
| Quant | `quant.h` | FP8 e4m3/e5m2 (+ per-token-group), INT8, INT4 dequant; via dispatch: **W4A8, W8A16-fp8, NVFP4/MXFP4, 2:4-sparse, BitNet ternary** |
| MoE | `moe.h` | softmax & **DeepSeek group** gating, permute, grouped GEMM |
| Sampling | `sampling.h` | softmax, argmax, temp+top-k+top-p (Philox) |
| SSM · linear-attn | `ssm.h` · `linear_attn.h` | **Mamba** selective-scan + causal-conv1d; **gated-DeltaNet / GLA / RWKV-7** (FLA-matched recurrent fallbacks) |
| Loss · Optimizer · Embedding · Elementwise | … | fused CE / FLCE · AdamW/SGD · lookup+bwd · add/mul/cast/… |

Every entry point returns `ks_status_t`, takes device pointers + a `ks_stream_t`.
See [`CONTRACT.md`](CONTRACT.md).

## Build from source

```bash
# CUDA 12.x, CMake ≥3.24 — globs kernels/src/**/*.cu, no per-kernel edits
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j   # L4; 80=A100, 90=H100
export KERNEL_SET_LIB=$PWD/build/libkernel_set.so

python3 models/ksctl plan --model deepseek-v3 --gpu h100 --dtype fp8   # best kernel per op
```

## Docs

| | |
|---|---|
| [`OPTIMAL_SELECTION.md`](docs/OPTIMAL_SELECTION.md) · [`ROUTING.md`](docs/ROUTING.md) | how a kernel gets picked (table + 3-tier routing) |
| [`EVOLUTION_PLAN.md`](docs/EVOLUTION_PLAN.md) | measured-data roadmap: benchmark persistence, sm90/sm100 promotion, release gates |
| [`QUANT_OPERATORS.md`](docs/QUANT_OPERATORS.md) | quant ops: what ships, what dispatches, the gaps |
| [`OPERATOR_CATALOG.md`](docs/OPERATOR_CATALOG.md) · [`ATOMIC_OPERATORS.md`](docs/ATOMIC_OPERATORS.md) | provider-ranked logical ops and atomic `sgl.*`/`flashinfer.*`/`vllm.*` catalogs |
| [`MODEL_KERNEL_MAP.md`](docs/MODEL_KERNEL_MAP.md) | 178 models → kernels (DeepSeek-V4, GLM-5, Kimi-2.6, Gemma-4, Llama 4, MiMo-V2.5, LongCat, Ling-2.5, … + Mamba/RWKV/DeltaNet) |
| [`USAGE.md`](docs/USAGE.md) · [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md) | usage · architecture · bench methodology |

## License

[Apache-2.0](LICENSE). kernel-set kernels are clean-room; vendored third-party
sources keep their own licenses ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).
