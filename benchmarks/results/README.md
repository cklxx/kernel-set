# kernel-set benchmark results

This directory keeps human-readable benchmark reports plus canonical
JSON runs under `runs/`. Use the JSON files as the durable data
source; Markdown files are display artifacts.

- **Canonical runs:** 21
- **GPU coverage:** NVIDIA A100-SXM4-40GB (sm80), NVIDIA H20 (sm90), NVIDIA L4 (sm89), NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120)
- **Suites:** kernel_set=17, sota=4
- **Index:** [`index.json`](index.json)

## Latest Runs

| run | GPU | suite | dtype | timing | rows | ok / skip / import-fail / error | data |
|---|---|---|---|---|---:|---:|---|
| `20260616t133654z-colab-kernel-set-swiglu-bwd-plus-4-nvidia-l4-fp16-kernel_set` | NVIDIA L4 (sm89) | kernel_set | fp16 | events-l2-flush | 47 | 47 / 0 / 0 / 0 | [20260616t133654z-colab-kernel-set-swiglu-bwd-plus-4-nvidia-l4-fp16.json](runs/20260616t133654z-colab-kernel-set-swiglu-bwd-plus-4-nvidia-l4-fp16.json) |
| `20260616t133654z-colab-kernel-set-rmsnorm-plus-4-nvidia-l4-fp16-kernel_set` | NVIDIA L4 (sm89) | kernel_set | fp16 | events-l2-flush | 30 | 30 / 0 / 0 / 0 | [20260616t133654z-colab-kernel-set-rmsnorm-plus-4-nvidia-l4-fp16.json](runs/20260616t133654z-colab-kernel-set-rmsnorm-plus-4-nvidia-l4-fp16.json) |
| `20260616t133654z-colab-kernel-set-quant-plus-4-nvidia-l4-fp16-kernel_set` | NVIDIA L4 (sm89) | kernel_set | fp16 | events-l2-flush | 18 | 18 / 0 / 0 / 0 | [20260616t133654z-colab-kernel-set-quant-plus-4-nvidia-l4-fp16.json](runs/20260616t133654z-colab-kernel-set-quant-plus-4-nvidia-l4-fp16.json) |
| `20260616t133654z-colab-kernel-set-gemm-plus-4-nvidia-l4-fp16-kernel_set` | NVIDIA L4 (sm89) | kernel_set | fp16 | events-l2-flush | 22 | 22 / 0 / 0 / 0 | [20260616t133654z-colab-kernel-set-gemm-plus-4-nvidia-l4-fp16.json](runs/20260616t133654z-colab-kernel-set-gemm-plus-4-nvidia-l4-fp16.json) |
| `20260616t133654z-colab-kernel-set-fused-linear-ce-plus-4-nvidia-l4-fp16-kernel_set` | NVIDIA L4 (sm89) | kernel_set | fp16 | events-l2-flush | 14 | 14 / 0 / 0 / 0 | [20260616t133654z-colab-kernel-set-fused-linear-ce-plus-4-nvidia-l4-fp16.json](runs/20260616t133654z-colab-kernel-set-fused-linear-ce-plus-4-nvidia-l4-fp16.json) |
| `20260616t133654z-colab-kernel-set-cross-entropy-plus-4-nvidia-l4-fp16-kernel_set` | NVIDIA L4 (sm89) | kernel_set | fp16 | events-l2-flush | 20 | 20 / 0 / 0 / 0 | [20260616t133654z-colab-kernel-set-cross-entropy-plus-4-nvidia-l4-fp16.json](runs/20260616t133654z-colab-kernel-set-cross-entropy-plus-4-nvidia-l4-fp16.json) |
| `20260616-pro6000-full-kernel-set-swiglu-bwd-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-kernel_set` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120) | kernel_set | bf16 | events-l2-flush | 47 | 47 / 0 / 0 / 0 | [20260616-pro6000-full-kernel-set-swiglu-bwd-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json](runs/20260616-pro6000-full-kernel-set-swiglu-bwd-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json) |
| `20260616-pro6000-full-kernel-set-rmsnorm-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-kernel_set` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120) | kernel_set | bf16 | events-l2-flush | 30 | 30 / 0 / 0 / 0 | [20260616-pro6000-full-kernel-set-rmsnorm-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json](runs/20260616-pro6000-full-kernel-set-rmsnorm-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json) |
| `20260616-pro6000-full-kernel-set-quant-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-kernel_set` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120) | kernel_set | bf16 | events-l2-flush | 18 | 18 / 0 / 0 / 0 | [20260616-pro6000-full-kernel-set-quant-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json](runs/20260616-pro6000-full-kernel-set-quant-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json) |
| `20260616-pro6000-full-kernel-set-gemm-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-kernel_set` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120) | kernel_set | bf16 | events-l2-flush | 22 | 22 / 0 / 0 / 0 | [20260616-pro6000-full-kernel-set-gemm-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json](runs/20260616-pro6000-full-kernel-set-gemm-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json) |
| `20260616-pro6000-full-kernel-set-fused-linear-ce-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-kernel_set` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120) | kernel_set | bf16 | events-l2-flush | 14 | 14 / 0 / 0 / 0 | [20260616-pro6000-full-kernel-set-fused-linear-ce-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json](runs/20260616-pro6000-full-kernel-set-fused-linear-ce-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json) |
| `20260616-pro6000-full-kernel-set-cross-entropy-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-kernel_set` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120) | kernel_set | bf16 | events-l2-flush | 20 | 20 / 0 / 0 / 0 | [20260616-pro6000-full-kernel-set-cross-entropy-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json](runs/20260616-pro6000-full-kernel-set-cross-entropy-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json) |

## Model-Part Coverage

| model part | position | suite | ops | shapes | impls | GPUs | ok / total |
|---|---|---|---:|---:|---:|---:|---:|
| `attention` | `decode` | kernel_set | 2 | 3 | 1 | 4 | 18 / 18 |
| `attention` | `decode` | sota | 2 | 3 | 6 | 2 | 15 / 28 |
| `attention` | `prefill` | kernel_set | 2 | 3 | 2 | 4 | 30 / 30 |
| `attention` | `prefill` | sota | 1 | 2 | 5 | 3 | 11 / 38 |
| `attention` | `training` | kernel_set | 1 | 2 | 1 | 4 | 12 / 12 |
| `elementwise` | `bulk` | kernel_set | 6 | 2 | 3 | 4 | 132 / 132 |
| `elementwise` | `decode` | kernel_set | 6 | 1 | 3 | 4 | 66 / 66 |
| `embedding` | `decode` | kernel_set | 1 | 1 | 2 | 4 | 12 / 12 |
| `embedding` | `prefill` | kernel_set | 1 | 2 | 2 | 4 | 24 / 24 |
| `embedding` | `training` | kernel_set | 1 | 3 | 1 | 4 | 18 / 18 |
| `linear` | `prefill` | kernel_set | 6 | 13 | 4 | 4 | 99 / 102 |
| `linear` | `prefill` | sota | 3 | 3 | 9 | 3 | 48 / 68 |
| `loss` | `training` | kernel_set | 2 | 4 | 2 | 4 | 36 / 36 |
| `loss` | `training` | sota | 2 | 2 | 4 | 3 | 20 / 32 |
| `mlp` | `decode` | kernel_set | 2 | 1 | 2 | 4 | 24 / 24 |
| `mlp` | `decode` | sota | 1 | 1 | 4 | 3 | 8 / 15 |
| `mlp` | `prefill` | kernel_set | 2 | 2 | 2 | 4 | 48 / 48 |
| `mlp` | `prefill` | sota | 1 | 1 | 4 | 3 | 8 / 15 |
| `mlp` | `training` | kernel_set | 1 | 3 | 1 | 4 | 18 / 18 |
| `moe` | `prefill` | kernel_set | 4 | 4 | 1 | 4 | 48 / 48 |
| `moe` | `prefill` | sota | 3 | 1 | 3 | 2 | 3 / 6 |
| `norm` | `decode` | kernel_set | 2 | 1 | 2 | 4 | 24 / 24 |
| `norm` | `decode` | sota | 2 | 1 | 5 | 3 | 17 / 30 |
| `norm` | `prefill` | kernel_set | 2 | 3 | 2 | 4 | 72 / 72 |
| `norm` | `prefill` | sota | 2 | 2 | 5 | 3 | 34 / 60 |
| `norm` | `training` | kernel_set | 2 | 4 | 1 | 4 | 48 / 48 |
| `optimizer` | `training` | kernel_set | 3 | 2 | 1 | 4 | 36 / 36 |
| `position_encoding` | `decode` | kernel_set | 1 | 1 | 1 | 4 | 6 / 6 |
| `position_encoding` | `decode` | sota | 1 | 1 | 4 | 2 | 6 / 11 |
| `position_encoding` | `prefill` | kernel_set | 1 | 1 | 1 | 4 | 6 / 6 |
| `position_encoding` | `prefill` | sota | 1 | 1 | 4 | 2 | 6 / 11 |
| `position_encoding` | `training` | kernel_set | 1 | 2 | 1 | 4 | 12 / 12 |
| `quant` | `prefill` | kernel_set | 5 | 4 | 1 | 4 | 54 / 58 |
| `quant` | `weight` | kernel_set | 1 | 2 | 1 | 4 | 14 / 14 |
| `sampling` | `decode` | kernel_set | 3 | 4 | 2 | 4 | 72 / 72 |
| `ssm` | `prefill` | sota | 3 | 1 | 3 | 2 | 0 / 6 |

## Representative Large Kernels

These rows keep the README focused on the model-dominant kernels: attention,
MLA, GEMM/FP8 GEMM, and MoE routing/dispatch. They may be single-provider
measurements when no comparable third-party provider was present in that run.

| part | op | GPU / suite | shape | impl | latency | source |
|---|---|---|---|---|---:|---|
| `attention` | `attn_prefill` | NVIDIA H20 (sm90, bf16, sota) | `b=1,seq=2048,qh=32,kvh=8,hd=128` | `flashinfer` | 335.7 us | [2026-06-05t16-32-42-nvidia-h20-bf16-sota.json](runs/2026-06-05t16-32-42-nvidia-h20-bf16-sota.json) |
| `attention` | `attn_prefill` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120, bf16, sota) | `b=1,seq=2048,qh=32,kvh=8,hd=128` | `kernel-set` | 15245.7 us | [2026-06-05t07-03-26-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-sota.json](runs/2026-06-05t07-03-26-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-sota.json) |
| `attention` | `attn_prefill` | NVIDIA L4 (sm89, fp16, sota) | `b=1,seq=2048,qh=32,kvh=8,hd=128` | `flashinfer` | 599.0 us | [2026-06-04t16-21-49-nvidia-l4-fp16-sota.json](runs/2026-06-04t16-21-49-nvidia-l4-fp16-sota.json) |
| `attention` | `attention_prefill` | NVIDIA A100-SXM4-40GB (sm80, bf16, kernel_set) | `b=4,seq=1024,qh=32,kvh=8,hd=128` | `kernel-set` | 30793.7 us | [2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json](runs/2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json) |
| `attention` | `attn_decode` | NVIDIA H20 (sm90, bf16, sota) | `seqs=64,ctx=2048,qh=32,kvh=8,hd=128` | `flashinfer` | 340.8 us | [2026-06-05t16-32-42-nvidia-h20-bf16-sota.json](runs/2026-06-05t16-32-42-nvidia-h20-bf16-sota.json) |
| `attention` | `attn_decode` | NVIDIA L4 (sm89, fp16, sota) | `seqs=64,ctx=2048,qh=32,kvh=8,hd=128` | `flashinfer` | 2283.5 us | [2026-06-04t16-21-49-nvidia-l4-fp16-sota.json](runs/2026-06-04t16-21-49-nvidia-l4-fp16-sota.json) |
| `attention` | `attention_decode` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120, bf16, kernel_set) | `seqs=64,ctx=2048,qh=32,kvh=8,hd=128` | `kernel-set` | 1780.7 us | [2026-06-05t07-01-25-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-kernel_set.json](runs/2026-06-05t07-01-25-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-kernel_set.json) |
| `attention` | `attention_decode` | NVIDIA A100-SXM4-40GB (sm80, bf16, kernel_set) | `seqs=64,ctx=2048,qh=32,kvh=8,hd=128` | `kernel-set` | 4455.4 us | [2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json](runs/2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json) |
| `attention` | `mla_decode` | NVIDIA H20 (sm90, bf16, sota) | `seqs=64,ctx=2048,h=128,lora=512,rope=64` | `flash-mla` | 303.2 us | [2026-06-05t16-32-42-nvidia-h20-bf16-sota.json](runs/2026-06-05t16-32-42-nvidia-h20-bf16-sota.json) |
| `attention` | `mla_decode` | NVIDIA L4 (sm89, fp16, sota) | `seqs=64,ctx=2048,h=128,lora=512,rope=64` | `kernel-set` | 79271.9 us | [2026-06-05t06-23-21-nvidia-l4-fp16-sota.json](runs/2026-06-05t06-23-21-nvidia-l4-fp16-sota.json) |
| `linear` | `gemm` | NVIDIA H20 (sm90, bf16, sota) | `M=4096,N=4096,K=4096` | `torch-cublas` | 1039.8 us | [2026-06-05t16-32-42-nvidia-h20-bf16-sota.json](runs/2026-06-05t16-32-42-nvidia-h20-bf16-sota.json) |
| `linear` | `gemm` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120, bf16, sota) | `M=4096,N=4096,K=4096` | `torch-cublas` | 369.6 us | [2026-06-05t07-03-26-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-sota.json](runs/2026-06-05t07-03-26-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-sota.json) |

## Representative Memory-Bound Kernels

| part | op | GPU / suite | shape | impl | latency | source |
|---|---|---|---|---|---:|---|
| `norm` | `fused_add_rmsnorm` | NVIDIA H20 (sm90, bf16, sota) | `rows=1,hidden=4096` | `flashinfer-norm` | 15.6 us | [2026-06-05t16-32-42-nvidia-h20-bf16-sota.json](runs/2026-06-05t16-32-42-nvidia-h20-bf16-sota.json) |
| `norm` | `fused_add_rmsnorm` | NVIDIA L4 (sm89, fp16, sota) | `rows=1,hidden=4096` | `flashinfer-norm` | 15.4 us | [2026-06-04t16-21-49-nvidia-l4-fp16-sota.json](runs/2026-06-04t16-21-49-nvidia-l4-fp16-sota.json) |
| `norm` | `rmsnorm` | NVIDIA H20 (sm90, bf16, sota) | `rows=1,hidden=4096` | `flashinfer-norm` | 10.9 us | [2026-06-05t16-32-42-nvidia-h20-bf16-sota.json](runs/2026-06-05t16-32-42-nvidia-h20-bf16-sota.json) |
| `norm` | `rmsnorm` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120, bf16, sota) | `rows=1,hidden=4096` | `kernel-set` | 16.4 us | [2026-06-05t07-03-26-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-sota.json](runs/2026-06-05t07-03-26-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-sota.json) |
| `norm` | `rmsnorm` | NVIDIA L4 (sm89, fp16, sota) | `rows=1,hidden=4096` | `liger-norm` | 11.3 us | [2026-06-04t16-21-49-nvidia-l4-fp16-sota.json](runs/2026-06-04t16-21-49-nvidia-l4-fp16-sota.json) |
| `norm` | `rmsnorm` | NVIDIA A100-SXM4-40GB (sm80, bf16, kernel_set) | `rows=1,hidden=4096` | `kernel-set` | 21.5 us | [2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json](runs/2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json) |
| `mlp` | `swiglu` | NVIDIA H20 (sm90, bf16, sota) | `rows=1,inter=14336` | `kernel-set` | 12.1 us | [2026-06-05t16-32-42-nvidia-h20-bf16-sota.json](runs/2026-06-05t16-32-42-nvidia-h20-bf16-sota.json) |
| `mlp` | `swiglu` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120, bf16, sota) | `rows=1,inter=14336` | `kernel-set` | 13.7 us | [2026-06-05t07-03-26-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-sota.json](runs/2026-06-05t07-03-26-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-sota.json) |
| `mlp` | `swiglu` | NVIDIA L4 (sm89, fp16, sota) | `rows=1,inter=14336` | `kernel-set` | 12.3 us | [2026-06-04t16-21-49-nvidia-l4-fp16-sota.json](runs/2026-06-04t16-21-49-nvidia-l4-fp16-sota.json) |
| `mlp` | `swiglu` | NVIDIA A100-SXM4-40GB (sm80, bf16, kernel_set) | `rows=1,inter=14336` | `kernel-set` | 8.2 us | [2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json](runs/2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json) |
| `mlp` | `geglu` | NVIDIA H20 (sm90, bf16, kernel_set) | `rows=1,inter=14336` | `kernel-set` | 6.5 us | [2026-06-06t06-32-10-nvidia-h20-bf16-kernel_set.json](runs/2026-06-06t06-32-10-nvidia-h20-bf16-kernel_set.json) |
| `mlp` | `geglu` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120, bf16, kernel_set) | `rows=1,inter=14336` | `kernel-set` | 8.2 us | [20260616-pro6000-full-kernel-set-cross-entropy-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json](runs/20260616-pro6000-full-kernel-set-cross-entropy-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json) |

## Grouped Provider Winners

| model part | position | op | GPU / suite | shape | winner | runner-up | ratio | source |
|---|---|---|---|---|---|---|---:|---|
| `attention` | `decode` | `attn_decode` | NVIDIA H20 (sm90, bf16, sota) | `seqs=64,ctx=2048,qh=32,kvh=8,hd=128` | `flashinfer` 340.8 us | `kernel-set` 3521.9 us | 10.33x | [2026-06-05t16-32-42-nvidia-h20-bf16-sota.json](runs/2026-06-05t16-32-42-nvidia-h20-bf16-sota.json) |
| `attention` | `decode` | `attn_decode` | NVIDIA L4 (sm89, fp16, sota) | `seqs=64,ctx=2048,qh=32,kvh=8,hd=128` | `flashinfer` 2285.6 us | `kernel-set` 7866.4 us | 3.44x | [2026-06-05t06-23-21-nvidia-l4-fp16-sota.json](runs/2026-06-05t06-23-21-nvidia-l4-fp16-sota.json) |
| `attention` | `decode` | `mla_decode` | NVIDIA H20 (sm90, bf16, sota) | `seqs=64,ctx=2048,h=128,lora=512,rope=64` | `flash-mla` 303.2 us | `kernel-set` 39288.0 us | 129.58x | [2026-06-05t16-32-42-nvidia-h20-bf16-sota.json](runs/2026-06-05t16-32-42-nvidia-h20-bf16-sota.json) |
| `attention` | `prefill` | `attention_prefill` | NVIDIA A100-SXM4-40GB (sm80, bf16, kernel_set) | `b=1,seq=2048,qh=32,kvh=8,hd=128` | `sdpa(flash/efficient)` 344.1 us | `kernel-set` 31170.0 us | 90.58x | [2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json](runs/2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json) |
| `attention` | `prefill` | `attention_prefill` | NVIDIA H20 (sm90, bf16, kernel_set) | `b=1,seq=2048,qh=32,kvh=8,hd=128` | `sdpa(flash/efficient)` 615.7 us | `kernel-set` 26138.0 us | 42.45x | [2026-06-06t06-32-10-nvidia-h20-bf16-kernel_set.json](runs/2026-06-06t06-32-10-nvidia-h20-bf16-kernel_set.json) |
| `attention` | `prefill` | `attention_prefill` | NVIDIA L4 (sm89, fp16, kernel_set) | `b=1,seq=2048,qh=32,kvh=8,hd=128` | `sdpa(flash/efficient)` 764.9 us | `kernel-set` 52264.4 us | 68.33x | [20260616t133654z-colab-kernel-set-rmsnorm-plus-4-nvidia-l4-fp16.json](runs/20260616t133654z-colab-kernel-set-rmsnorm-plus-4-nvidia-l4-fp16.json) |
| `attention` | `prefill` | `attention_prefill` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120, bf16, kernel_set) | `b=1,seq=2048,qh=32,kvh=8,hd=128` | `sdpa(flash/efficient)` 243.6 us | `kernel-set` 16269.5 us | 66.79x | [20260616-pro6000-full-kernel-set-rmsnorm-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json](runs/20260616-pro6000-full-kernel-set-rmsnorm-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json) |
| `attention` | `prefill` | `attn_prefill` | NVIDIA H20 (sm90, bf16, sota) | `b=1,seq=2048,qh=32,kvh=8,hd=128` | `flashinfer` 335.7 us | `kernel-set` 26172.2 us | 77.96x | [2026-06-05t16-32-42-nvidia-h20-bf16-sota.json](runs/2026-06-05t16-32-42-nvidia-h20-bf16-sota.json) |
| `attention` | `prefill` | `attn_prefill` | NVIDIA L4 (sm89, fp16, sota) | `b=1,seq=2048,qh=32,kvh=8,hd=128` | `flashinfer` 599.0 us | `kernel-set` 51517.0 us | 86.01x | [2026-06-04t16-21-49-nvidia-l4-fp16-sota.json](runs/2026-06-04t16-21-49-nvidia-l4-fp16-sota.json) |
| `elementwise` | `bulk` | `ew_add` | NVIDIA A100-SXM4-40GB (sm80, bf16, kernel_set) | `n=16777216` | `kernel-set` 85.0 us | `eager` 92.2 us | 1.08x | [2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json](runs/2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json) |
| `elementwise` | `bulk` | `ew_add` | NVIDIA H20 (sm90, bf16, kernel_set) | `n=16777216` | `kernel-set` 34.9 us | `eager` 43.2 us | 1.24x | [2026-06-06t06-32-10-nvidia-h20-bf16-kernel_set.json](runs/2026-06-06t06-32-10-nvidia-h20-bf16-kernel_set.json) |
| `elementwise` | `bulk` | `ew_add` | NVIDIA L4 (sm89, fp16, kernel_set) | `n=16777216` | `kernel-set` 456.7 us | `eager` 460.8 us | 1.01x | [2026-06-05t03-27-10-nvidia-l4-fp16-kernel_set.json](runs/2026-06-05t03-27-10-nvidia-l4-fp16-kernel_set.json) |
| `elementwise` | `bulk` | `ew_add` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120, bf16, kernel_set) | `n=16777216` | `kernel-set` 75.8 us | `eager` 84.0 us | 1.11x | [2026-06-05t07-01-25-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-kernel_set.json](runs/2026-06-05t07-01-25-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-kernel_set.json) |
| `elementwise` | `bulk` | `ew_axpby` | NVIDIA A100-SXM4-40GB (sm80, bf16, kernel_set) | `n=67108864` | `kernel-set` 305.2 us | `eager` 706.6 us | 2.32x | [2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json](runs/2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json) |
| `elementwise` | `bulk` | `ew_axpby` | NVIDIA H20 (sm90, bf16, kernel_set) | `n=16777216` | `kernel-set` 34.9 us | `eager` 89.4 us | 2.56x | [2026-06-06t06-32-10-nvidia-h20-bf16-kernel_set.json](runs/2026-06-06t06-32-10-nvidia-h20-bf16-kernel_set.json) |
| `elementwise` | `bulk` | `ew_axpby` | NVIDIA L4 (sm89, fp16, kernel_set) | `n=67108864` | `kernel-set` 1752.1 us | `eager` 4065.8 us | 2.32x | [2026-06-05t03-27-10-nvidia-l4-fp16-kernel_set.json](runs/2026-06-05t03-27-10-nvidia-l4-fp16-kernel_set.json) |
| `elementwise` | `bulk` | `ew_axpby` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120, bf16, kernel_set) | `n=67108864` | `kernel-set` 277.5 us | `eager` 651.3 us | 2.35x | [2026-06-05t07-01-25-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-kernel_set.json](runs/2026-06-05t07-01-25-nvidia-rtx-pro-6000-blackwell-server-edition-bf16-kernel_set.json) |
| `elementwise` | `bulk` | `ew_cast` | NVIDIA A100-SXM4-40GB (sm80, bf16, kernel_set) | `n=16777216` | `kernel-set` 86.0 us | `eager(.to)` 101.4 us | 1.18x | [2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json](runs/2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json) |
| `elementwise` | `bulk` | `ew_cast` | NVIDIA H20 (sm90, bf16, kernel_set) | `n=67108864` | `kernel-set` 248.7 us | `eager(.to)` 316.2 us | 1.27x | [2026-06-06t06-32-10-nvidia-h20-bf16-kernel_set.json](runs/2026-06-06t06-32-10-nvidia-h20-bf16-kernel_set.json) |
| `elementwise` | `bulk` | `ew_cast` | NVIDIA L4 (sm89, fp16, kernel_set) | `n=16777216` | `kernel-set` 419.8 us | `eager(.to)` 441.3 us | 1.05x | [20260616t133654z-colab-kernel-set-swiglu-bwd-plus-4-nvidia-l4-fp16.json](runs/20260616t133654z-colab-kernel-set-swiglu-bwd-plus-4-nvidia-l4-fp16.json) |
| `elementwise` | `bulk` | `ew_cast` | NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120, bf16, kernel_set) | `n=67108864` | `kernel-set` 282.6 us | `eager(.to)` 306.3 us | 1.08x | [20260616-pro6000-full-kernel-set-swiglu-bwd-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json](runs/20260616-pro6000-full-kernel-set-swiglu-bwd-plus-4-nvidia-rtx-pro-6000-blackwell-server-edition-bf16.json) |
| `elementwise` | `bulk` | `ew_mul` | NVIDIA A100-SXM4-40GB (sm80, bf16, kernel_set) | `n=16777216` | `kernel-set` 85.0 us | `eager` 91.1 us | 1.07x | [2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json](runs/2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json) |
| `elementwise` | `bulk` | `ew_mul` | NVIDIA H20 (sm90, bf16, kernel_set) | `n=16777216` | `kernel-set` 34.9 us | `eager` 42.2 us | 1.21x | [2026-06-06t06-32-10-nvidia-h20-bf16-kernel_set.json](runs/2026-06-06t06-32-10-nvidia-h20-bf16-kernel_set.json) |
| `elementwise` | `bulk` | `ew_mul` | NVIDIA L4 (sm89, fp16, kernel_set) | `n=67108864` | `kernel-set` 1708.5 us | `eager` 1764.9 us | 1.03x | [20260616t133654z-colab-kernel-set-swiglu-bwd-plus-4-nvidia-l4-fp16.json](runs/20260616t133654z-colab-kernel-set-swiglu-bwd-plus-4-nvidia-l4-fp16.json) |

## Inference Engine Smoke

Single-prompt decode smoke runs are integration checks, not apples-to-apples
engine throughput benchmarks. They verify tokenizer/output parity for the
composed engine paths.
Rows with kernel coverage are integration rows, not serving-system benchmarks:
`kernel_set_engine` keeps dense linears on torch/cuBLAS and uses
shape-aware provider selection for the measured Qwen3 shapes.

| model / GPU | engine | scope | new tok/s | token match | notes | source |
|---|---|---|---:|---|---|---|
| Qwen/Qwen3-8B / NVIDIA L4 (sm89, bf16) | `transformers` | HuggingFace generate | 15.20 | yes | reference; historical baseline from 20260618-qwen3-8b-l4-long-greedy-vllm | [20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json](inference/20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json) |
| Qwen/Qwen3-8B / NVIDIA L4 (sm89, bf16) | `vllm` | vLLM LLM.generate | 16.24 | yes | historical baseline from 20260618-qwen3-8b-l4-long-greedy-vllm | [20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json](inference/20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json) |
| Qwen/Qwen3-8B / NVIDIA L4 (sm89, bf16) | `sglang` | SGLang Engine.generate | 16.40 | no (104/197) | Colab L4 SGLang Engine.generate; greedy output diverged from HF reference after prompt + 3 generated tokens | [20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json](inference/20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json) |
| Qwen/Qwen3-8B / NVIDIA L4 (sm89, bf16) | `kernel_set_engine` | single-request Python engine; torch/cuBLAS linears + ks RMSNorm/RoPE/KV write/short-decode/SwiGLU + shape-aware embedding/attention | 15.65 | no (136/197) | shape-aware kernel-set engine composition from Qwen3 kernel microbench; Python loop/allocation and unfused Q/K/V + gate/up remain; historical baseline from 20260618-qwen3-8b-l4-long-greedy-vllm | [20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json](inference/20260618-qwen3-8b-l4-long-greedy-vllm-sglang.json) |

Kernel coverage for the composed kernel-set engine:

| engine | covered kernel-set kernels | remaining torch/Python path | counted calls |
|---|---|---|---|
| `kernel_set_engine` | `ks_embedding_lookup(auto multi-token)`<br>`ks_rms_norm`<br>`ks_rope_gather`<br>`ks_paged_attn_decode(auto short-context)`<br>`ks_reshape_and_cache`<br>`ks_swiglu` | embedding single-token=torch<br>linear=torch/cuBLAS<br>attention prefill/long-context=torch SDPA<br>argmax=torch<br>residual add<br>tensor reshape/view/allocation<br>Python request/decode loop<br>paged block scheduler | `embedding_lookup`=1<br>`paged_attn_decode`=3420<br>`reshape_and_cache`=3456<br>`rmsnorm`=13920<br>`rope`=3456<br>`swiglu`=3456<br>`torch_argmax`=96<br>`torch_attention_prefill`=36<br>`torch_embedding`=95<br>`torch_linear`=24288 |

Quantized checkpoint engine smoke:

| model / GPU | quant mode | engine | Transformers tok/s | engine tok/s | engine / HF | token match | peak GB | source |
|---|---|---|---:|---:|---:|---|---:|---|
| Qwen/Qwen3-8B / NVIDIA L4 (sm89, bf16) | `bnb_nf4` | `kernel_set_engine` | 10.83 | 13.72 | 1.27x | yes | 5.84 | [20260618-qwen3-8b-l4-quantized-engine.json](inference/20260618-qwen3-8b-l4-quantized-engine.json) |

Rows are real checkpoint loads with greedy decode and exact token parity against the same quantized Transformers model. Non-exact diagnostic engine rows remain in the JSON but are not displayed as comparison data.

## Regenerate

```bash
python benchmarks/render_results_readme.py --root-readme README.md
python benchmarks/persist.py validate benchmarks/results/runs
```
