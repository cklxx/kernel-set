<h1 align="center">kernel-set</h1>

<p align="center">
  <b>High-performance CUDA kernels for LLM inference & training — callable from Python, Rust, Go, and TypeScript through one stable C ABI.</b>
</p>

---

`kernel-set` is a single, coherent collection of the operators that dominate
transformer inference and training — fused attention, GEMM, normalization,
rotary embeddings, gated MLPs, MoE routing, quantization, sampling, losses, and
optimizers — exposed behind **one pure-C ABI** so any language can call them
without a GPU toolchain of its own. A model→kernel registry and the `ksctl` CLI
then **auto-select the strongest kernel per operator** for a given model and GPU.

> **Status:** CUDA backend. A HIP/ROCm backend is wired into the build and the
> platform abstraction layer (`KS_PLATFORM_HIP`) so porting to ROCm is a flag,
> not a rewrite — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Why

- **Write once, call everywhere.** The ABI in [`include/kernel_set/`](include/kernel_set)
  is plain C with no CUDA headers — so Python (ctypes), Go (cgo), Rust (FFI), and
  Node (koffi) all bind to the *same* `libkernel_set`.
- **Best-in-class, mapped to models.** Operators are surveyed against the
  real-world landscape (FlashAttention/FlashInfer/vLLM/SGLang/CUTLASS/Marlin/
  AWQ/Liger/…) and mapped per model family. `ksctl plan` picks the strongest
  kernel + dtype for your `(model, gpu)`.
- **Built for the GPUs you have.** Targets T4/A100/A10/L4/H100 by default; the
  Colab benchmark harness builds for L4 (sm_89) and A100 (sm_80) and benchmarks
  every op against a PyTorch reference.

## Repository layout

```
include/kernel_set/     Pure-C ABI — the single source of truth (12 categories + runtime)
kernels/src/common/     Platform abstraction (CUDA/HIP), dtype traits, reductions, vec IO, dispatch
kernels/src/<category>/ CUDA kernel implementations (.cu)
kernels/tests/          C++ correctness tests (compare vs CPU reference)
bindings/python|rust|go|ts/   One FFI wrapper per language over the C ABI
models/                 Model→kernel registry (registry.json/.yaml) + select engine + `ksctl` CLI
benchmarks/             Colab-ready benchmark harness (bench.py, colab_bench.ipynb)
docs/                   Architecture, usage, kernel landscape, model→kernel table
examples/               Minimal end-to-end example per language
```

## Operator categories

| Category | Header | Highlights |
|---|---|---|
| Attention | `attention.h` | FlashAttention-2 prefill (dense + varlen), paged decode, KV-cache append, **MLA** decode, backward |
| GEMM | `gemm.h` | tensor-core fp16/bf16, fused bias+act, batched, **W8A8**, **W4A16** (AWQ/GPTQ) |
| Norm | `norm.h` | RMSNorm (+fused add-residual), LayerNorm, both backward |
| RoPE | `rope.h` | NeoX & interleaved, gathered, GQA, backward |
| Activation | `activation.h` | SiLU/GeLU/ReLU, **SwiGLU**/GeGLU (+packed, +backward) |
| Quant | `quant.h` | **FP8** e4m3/e5m2, INT8 (per-tensor/token), INT4 dequant |
| MoE | `moe.h` | softmax & **DeepSeek group** top-k gating, permute/unpermute, **grouped GEMM** |
| Sampling | `sampling.h` | softmax, argmax, fused temperature+top-k+top-p (Philox RNG) |
| Embedding | `embedding.h` | lookup + scatter-add backward |
| Elementwise | `elementwise.h` | add/mul/residual/scale/cast/axpby (vectorized) |
| Loss | `loss.h` | fused cross-entropy, **fused-linear-cross-entropy** (chunked) |
| Optimizer | `optimizer.h` | fused AdamW, SGD-momentum, global grad-norm |

Every entry point returns `ks_status_t`, takes device pointers (`void*`), a
dtype tag, and a `ks_stream_t` (`NULL` = default stream). See
[`CONTRACT.md`](CONTRACT.md) for the full convention.

## Build

Requires CUDA 12.x + CMake ≥ 3.24.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89   # L4; use 80 for A100
cmake --build build -j
# -> build/libkernel_set.so
export KERNEL_SET_LIB=$PWD/build/libkernel_set.so   # bindings discover the lib here
```

The build globs `kernels/src/**/*.cu`, so adding a kernel needs no CMake edit.

## Auto-select kernels for a model

```bash
python3 models/ksctl list
python3 models/ksctl plan --model deepseek-v3 --gpu a100 --dtype bf16 --mode inference
python3 models/ksctl table --gpu l4 --md docs/MODEL_KERNEL_MAP.md
```

`plan` reports, per logical op, the strongest `kernel-set` function + dtype/quant
scheme + rationale given the GPU's capabilities (e.g. FP8 on L4/H100 but not
A100) and the model's architecture (MLA→`ks_mla_decode`, MoE→`ks_moe_*`).

## Call it from your language

```python
import torch, kernel_set as ks
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
out = ks.rms_norm(x, w, eps=1e-6)
```

See [`docs/USAGE.md`](docs/USAGE.md) and each `bindings/<lang>/README.md` for
Rust, Go, and TypeScript snippets.

## Benchmark on Colab (L4 / A100)

Open [`benchmarks/colab_bench.ipynb`](benchmarks/colab_bench.ipynb) on a GPU
runtime — it detects the GPU, builds for the right arch, and benchmarks every op
against PyTorch. Locally: `benchmarks/build_and_bench.sh`.

## License

See [`LICENSE`](LICENSE). Kernels are clean-room implementations; provenance of
the algorithms they are modeled on is documented in
[`docs/KERNEL_LANDSCAPE.md`](docs/KERNEL_LANDSCAPE.md).
