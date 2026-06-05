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

- **One C ABI, four languages.** 78 operators (attention, GEMM, norm, RoPE,
  gated-MLP, MoE, quant, sampling, loss, optimizer, **Mamba SSM**) behind one
  `libkernel_set` — Python / Rust / Go / TypeScript bind the *same* library, no
  GPU toolchain of their own.
- **Auto best-kernel selection.** A per-`(op, GPU, dtype)` table routes each call
  to the strongest available backend; kernel-set's clean-room kernels are the
  always-there fallback. ([how it decides](docs/OPTIMAL_SELECTION.md))

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

- **Memory-bound ops → kernel-set's own kernels.** RMSNorm, SwiGLU/GeGLU, RoPE,
  elementwise, AdamW hit **84–87 % of A100 peak bandwidth** — on par with or
  beating FlashInfer/Liger.
- **Compute-bound ops → the industry best.** Our clean-room GEMM / attention /
  MoE can't beat cuBLAS / FlashAttention / DeepGEMM, so the dispatcher routes to
  them; kernel-set stays as the correct portable fallback.

> **Verified on real GPUs:** every kernel-set op is correctness-checked on **L4
> (sm89)**, **A100 (sm80)**, and **RTX PRO 6000 Blackwell (sm120)** —
> `correct=100, incorrect=0` each (`benchmarks/results/`). Builds + loads across
> **sm70–sm120** (T4/V100 → Blackwell). HIP/ROCm behind a build flag.

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
| Quant | `quant.h` | FP8 e4m3/e5m2 (+ per-token-group), INT8, INT4 dequant; NVFP4/MXFP4 via dispatch |
| MoE | `moe.h` | softmax & **DeepSeek group** gating, permute, grouped GEMM |
| Sampling | `sampling.h` | softmax, argmax, temp+top-k+top-p (Philox) |
| SSM | `ssm.h` | **Mamba** selective-scan + causal-conv1d |
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
| [`QUANT_OPERATORS.md`](docs/QUANT_OPERATORS.md) | quant ops: what ships, what dispatches, the gaps |
| [`OPERATOR_CATALOG.md`](docs/OPERATOR_CATALOG.md) · [`ATOMIC_OPERATORS.md`](docs/ATOMIC_OPERATORS.md) | 127 logical ops · 476 atomic ops (`sgl.*`/`flashinfer.*`/`vllm.*`) |
| [`MODEL_KERNEL_MAP.md`](docs/MODEL_KERNEL_MAP.md) | 157 models → kernels (DeepSeek-V4, GLM-5, Kimi-2.6, Gemma-4, Llama 4, … + Mamba/RWKV) |
| [`USAGE.md`](docs/USAGE.md) · [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md) | usage · architecture · bench methodology |

## License

[Apache-2.0](LICENSE). kernel-set kernels are clean-room; vendored third-party
sources keep their own licenses ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).
