<h1 align="center">kernel-set</h1>

<p align="center"><b>High-performance LLM inference & training kernels — one C ABI, callable from Python / Rust / Go / TypeScript — that auto-selects the strongest kernel per model & GPU.</b></p>

<p align="center">
  <a href="https://github.com/cklxx/kernel-set/actions/workflows/ci.yml"><img src="https://github.com/cklxx/kernel-set/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

---

`kernel-set` is two things in one:

1. **A portable kernel library** — the operators that dominate transformer
   inference/training (attention, GEMM, norm, RoPE, gated-MLP, MoE, quant,
   sampling, loss, optimizer, **Mamba SSM**) behind **one pure-C ABI**, so any
   language binds the *same* `libkernel_set` with no GPU toolchain of its own.
2. **A best-kernel selector** — a catalog of the real-world SOTA kernels
   (FlashAttention/FlashInfer/vLLM/SGLang/CUTLASS/Marlin/DeepGEMM/…) and a
   dispatcher that routes each op to the **strongest available** implementation,
   falling back to kernel-set's own kernels for portability.

> **Verified:** every kernel-set op is GPU-checked for correctness on **L4
> (sm89)** and **A100 (sm80)** (`benchmarks/results/`); the library loads on
> **sm70–sm90** (T4/V100/A100/L4/H100). HIP/ROCm is wired behind a build flag.

## The strategy (honest)

- **Memory-bound ops are SOTA-class self-developed kernels** — RMSNorm, SwiGLU,
  RoPE, elementwise, AdamW hit **84–87 % of A100 peak bandwidth** (AdamW 87 %,
  SwiGLU/GeGLU 84 %), on par with or beating FlashInfer/Liger.
- **Compute-bound ops dispatch to the industry best** — our clean-room GEMM /
  attention / MoE are ~0.01–0.1× cuBLAS/FlashAttention, so the dispatcher routes
  them to cuBLAS / FlashAttention / FlashInfer / Marlin / DeepGEMM / sgl-kernel;
  kernel-set stays as the correct portable fallback. See
  [`docs/OPTIMAL_SELECTION.md`](docs/OPTIMAL_SELECTION.md).

## Operators

| Category | Header | Highlights |
|---|---|---|
| Attention | `attention.h` | FlashAttention-2 prefill (dense+varlen), paged decode, **MLA**, KV-cache, backward |
| GEMM | `gemm.h` | tensor-core fp16/bf16, bias+act, batched, **W8A8 / W4A16 / FP8** |
| Norm | `norm.h` | RMSNorm (+fused residual), LayerNorm, backward |
| RoPE | `rope.h` | NeoX & interleaved, gathered, GQA, backward |
| Activation | `activation.h` | SiLU/GeLU/ReLU, **SwiGLU/GeGLU** (+backward) |
| Quant | `quant.h` | FP8 e4m3/e5m2, INT8, INT4 dequant |
| MoE | `moe.h` | softmax & **DeepSeek group** gating, permute, grouped GEMM |
| Sampling | `sampling.h` | softmax, argmax, temp+top-k+top-p (Philox) |
| SSM | `ssm.h` | **Mamba** selective-scan + causal-conv1d |
| Loss · Optimizer · Embedding · Elementwise | … | fused CE / FLCE · AdamW/SGD · lookup+bwd · add/mul/cast/… |

Every entry point returns `ks_status_t`, takes device pointers + a `ks_stream_t`.
See [`CONTRACT.md`](CONTRACT.md).

## Quickstart

```bash
# build (CUDA 12.x, CMake ≥3.24) — globs kernels/src/**/*.cu, no per-kernel edits
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j   # L4; 80=A100, 90=H100
export KERNEL_SET_LIB=$PWD/build/libkernel_set.so

# which kernel is best for my model+GPU?
python3 models/ksctl plan     --model deepseek-v3 --gpu a100 --dtype bf16
python3 models/ksctl backends --gpu h100        # the runtime best-backend chain per op

# call it (auto-routes to the best installed backend; ks kernel as fallback)
python3 - <<'PY'
import torch, kernel_set as ks
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print(ks.dispatch.rms_norm(x, torch.ones(4096, device="cuda", dtype=torch.bfloat16)))
PY

# benchmark every op vs PyTorch/SOTA (Colab L4/A100)
benchmarks/build_and_bench.sh      # or open benchmarks/colab_bench.ipynb
```

## Layout

```
include/kernel_set/   pure-C ABI (the contract)        bindings/{python,rust,go,ts}/  FFI wrappers
kernels/src/          CUDA kernels (+common/ platform)  models/   registry + ksctl (model→kernel plan)
providers/            SOTA catalog + 476-op atomic index third_party/  vendored SOTA kernel sources
benchmarks/           harness + results/{l4,a100}.md     docs/     architecture / routing / selection
```

## Docs

| | |
|---|---|
| [`docs/ROUTING.md`](docs/ROUTING.md) | three-tier routing: static plan → runtime dispatch → C-ABI fallback |
| [`docs/OPTIMAL_SELECTION.md`](docs/OPTIMAL_SELECTION.md) | per-op optimal provider + adopt-vs-self-develop decision |
| [`docs/OPERATOR_CATALOG.md`](docs/OPERATOR_CATALOG.md) · [`ATOMIC_OPERATORS.md`](docs/ATOMIC_OPERATORS.md) | 127 logical ops · 476 atomic ops (`sgl.*`/`flashinfer.*`/`vllm.*`) flattened |
| [`docs/MODEL_KERNEL_MAP.md`](docs/MODEL_KERNEL_MAP.md) | 100+ models → kernels (Llama 4, Qwen3, DeepSeek-V3, Kimi-K2, …) |
| [`docs/KERNEL_LANDSCAPE.md`](docs/KERNEL_LANDSCAPE.md) · [`BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md) · [`USAGE.md`](docs/USAGE.md) · [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | landscape · bench methodology · usage · architecture |

## License

[Apache-2.0](LICENSE). kernel-set kernels are clean-room; vendored third-party
sources keep their own licenses ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).
