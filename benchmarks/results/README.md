# kernel-set benchmark results

This directory keeps human-readable benchmark reports plus canonical
JSON runs under `runs/`. Use the JSON files as the durable data
source; Markdown files are display artifacts.

- **Canonical runs:** 20
- **GPU coverage:** NVIDIA A100-SXM4-40GB (sm80), NVIDIA H20 (sm90), NVIDIA L4 (sm89), NVIDIA RTX PRO 6000 Blackwell Server Edition (sm120)
- **Suites:** kernel_set=16, sota=4
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
| `linear` | `prefill` | kernel_set | 4 | 7 | 3 | 4 | 84 / 84 |
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
| `quant` | `prefill` | kernel_set | 4 | 2 | 1 | 4 | 44 / 48 |
| `quant` | `weight` | kernel_set | 1 | 2 | 1 | 4 | 12 / 12 |
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
| `attention` | `attention_prefill` | NVIDIA A100-SXM4-40GB (sm80, bf16, kernel_set) | `b=1,seq=2048,qh=32,kvh=8,hd=128` | `sdpa(flash/efficient)` | 344.1 us | [2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json](runs/2026-06-05t03-39-51-nvidia-a100-sxm4-40gb-bf16-kernel_set.json) |
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
`kernel_set_best_practice` keeps dense linears on torch/cuBLAS and uses
shape-aware provider selection for the measured Qwen3 shapes;
`kernel_set_full_kernels` is the slower all-kernel coverage smoke that
also routes linears through kernel-set's auditable reference GEMM path.

| model / GPU | engine | scope | new tok/s | token match | notes | source |
|---|---|---|---:|---|---|---|
| Qwen/Qwen3-8B / NVIDIA L4 (sm89, bf16) | `transformers` | HuggingFace generate | 14.58 | yes | reference; historical baseline from 20260617-qwen3-8b-daily-l4-full-kernels-vllm | [20260617-qwen3-8b-daily-l4-shape-aware-kernel-microbench.json](inference/20260617-qwen3-8b-daily-l4-shape-aware-kernel-microbench.json) |
| Qwen/Qwen3-8B / NVIDIA L4 (sm89, bf16) | `vllm` | vLLM LLM.generate | 15.82 | yes | historical baseline from 20260617-qwen3-8b-daily-l4-full-kernels-vllm | [20260617-qwen3-8b-daily-l4-shape-aware-kernel-microbench.json](inference/20260617-qwen3-8b-daily-l4-shape-aware-kernel-microbench.json) |
| Qwen/Qwen3-8B / NVIDIA L4 (sm89, bf16) | `kernel_set_best_practice` | single-request Python engine; torch/cuBLAS linears + ks RMSNorm/RoPE/KV write/short-decode/SwiGLU + shape-aware embedding/attention | 15.65 | yes | shape-aware best-practice composition from Qwen3 kernel microbench; Python loop/allocation and unfused Q/K/V + gate/up remain | [20260617-qwen3-8b-daily-l4-shape-aware-kernel-microbench.json](inference/20260617-qwen3-8b-daily-l4-shape-aware-kernel-microbench.json) |
| Qwen/Qwen3-8B / NVIDIA L4 (sm89, bf16) | `kernel_set_full_kernels` | single-request Python engine; full ks kernel path | 2.64 | yes | covers embedding/GEMM/RMSNorm/RoPE/FlashAttn/KV write/paged decode/SwiGLU/argmax; Python loop remains; historical baseline from 20260617-qwen3-8b-daily-l4-full-kernels-vllm | [20260617-qwen3-8b-daily-l4-shape-aware-kernel-microbench.json](inference/20260617-qwen3-8b-daily-l4-shape-aware-kernel-microbench.json) |

Qwen3-shape kernel microbench:

| op | shape | kernel-set | reference | winner | ratio | err |
|---|---|---:|---:|---|---:|---|
| `embedding_lookup` | `tokens=1,hidden=4096` | 18.0 us | `torch_embedding` 16.0 us | `torch_embedding` | torch_embedding 1.13x | rel 0.0000 |
| `embedding_lookup` | `tokens=21,hidden=4096` | 17.4 us | `torch_embedding` 24.2 us | `kernel-set` | kernel-set 1.39x | rel 0.0000 |
| `rms_norm` | `decode_hidden,rows=1,hidden=4096` | 16.7 us | `torch_rms` 91.0 us | `kernel-set` | kernel-set 5.45x | rel 0.0056 |
| `rms_norm` | `prefill_hidden,rows=21,hidden=4096` | 16.7 us | `torch_rms` 91.7 us | `kernel-set` | kernel-set 5.49x | rel 0.0069 |
| `rms_norm` | `decode_q_norm,rows=32,hidden=128` | 16.5 us | `torch_rms` 91.1 us | `kernel-set` | kernel-set 5.51x | rel 0.0047 |
| `rms_norm` | `decode_k_norm,rows=8,hidden=128` | 16.5 us | `torch_rms` 92.4 us | `kernel-set` | kernel-set 5.59x | rel 0.0041 |
| `rms_norm` | `prefill_q_norm,rows=672,hidden=128` | 16.8 us | `torch_rms` 92.0 us | `kernel-set` | kernel-set 5.47x | rel 0.0052 |
| `rms_norm` | `prefill_k_norm,rows=168,hidden=128` | 16.4 us | `torch_rms` 91.1 us | `kernel-set` | kernel-set 5.55x | rel 0.0046 |
| `rope_gather` | `tokens=1,qh=32,kvh=8,hd=128` | 17.4 us | `torch_rope` 198.1 us | `kernel-set` | kernel-set 11.38x | rel 0.0000 |
| `rope_gather` | `tokens=21,qh=32,kvh=8,hd=128` | 17.2 us | `torch_rope` 214.9 us | `kernel-set` | kernel-set 12.48x | rel 0.0067 |
| `reshape_and_cache` | `tokens=1,kvh=8,hd=128,block=16` | 73.0 us | `torch_scatter` 110.5 us | `kernel-set` | kernel-set 1.51x | rel 0.0000 |
| `reshape_and_cache` | `tokens=21,kvh=8,hd=128,block=16` | 102.7 us | `torch_scatter` 135.2 us | `kernel-set` | kernel-set 1.32x | rel 0.0000 |
| `flash_attn_prefill` | `b=1,seq=21,qh=32,kvh=8,hd=128` | 57.5 us | `torch_sdpa` 74.7 us | `kernel-set` | kernel-set 1.30x | rel 0.0025 |
| `flash_attn_prefill` | `b=1,seq=128,qh=32,kvh=8,hd=128` | 279.0 us | `torch_sdpa` 75.7 us | `torch_sdpa` | torch_sdpa 3.68x | rel 0.0050 |
| `flash_attn_prefill` | `b=1,seq=512,qh=32,kvh=8,hd=128` | 3303.6 us | `torch_sdpa` 76.0 us | `torch_sdpa` | torch_sdpa 43.46x | rel 0.0047 |
| `paged_attn_decode` | `seqs=1,ctx=22,qh=32,kvh=8,hd=128,block=16` | 19.7 us | `torch_gather_sdpa` 125.8 us<br>`torch_dense_sdpa` 77.9 us | `kernel-set` | kernel-set 6.40x | rel 0.0031 |
| `paged_attn_decode` | `seqs=1,ctx=24,qh=32,kvh=8,hd=128,block=16` | 19.6 us | `torch_gather_sdpa` 125.3 us<br>`torch_dense_sdpa` 77.0 us | `kernel-set` | kernel-set 6.40x | rel 0.0034 |
| `paged_attn_decode` | `seqs=1,ctx=128,qh=32,kvh=8,hd=128,block=16` | 62.7 us | `torch_gather_sdpa` 124.0 us<br>`torch_dense_sdpa` 77.2 us | `kernel-set` | kernel-set 1.98x | rel 0.0050 |
| `paged_attn_decode` | `seqs=1,ctx=512,qh=32,kvh=8,hd=128,block=16` | 244.5 us | `torch_gather_sdpa` 133.1 us<br>`torch_dense_sdpa` 84.8 us | `torch_gather_sdpa` | torch_gather_sdpa 1.84x | rel 0.0040 |
| `paged_attn_decode` | `seqs=1,ctx=2048,qh=32,kvh=8,hd=128,block=16` | 974.6 us | `torch_gather_sdpa` 187.0 us<br>`torch_dense_sdpa` 141.3 us | `torch_gather_sdpa` | torch_gather_sdpa 5.21x | rel 0.0030 |
| `paged_attn_decode` | `seqs=64,ctx=2048,qh=32,kvh=8,hd=128,block=16` | 7632.1 us | `torch_gather_sdpa` 23993.8 us<br>`torch_dense_sdpa` 20088.9 us | `kernel-set` | kernel-set 3.14x | rel 0.0038 |
| `swiglu` | `rows=1,inter=12288` | 15.8 us | `torch_silu_mul` 17.0 us | `kernel-set` | kernel-set 1.08x | rel 0.0040 |
| `swiglu` | `rows=21,inter=12288` | 16.1 us | `torch_silu_mul` 17.2 us | `kernel-set` | kernel-set 1.07x | rel 0.0066 |
| `argmax` | `rows=1,vocab=151936` | 113.9 us | `torch_argmax` 29.1 us | `torch_argmax` | torch_argmax 3.92x | exact |

These are per-kernel CUDA-event timings at the Qwen3-8B shapes used by the engine smoke; they are provider-selection evidence, not serving throughput.

Kernel coverage for the composed kernel-set engine:

| engine | covered kernel-set kernels | remaining torch/Python path | counted calls |
|---|---|---|---|
| `kernel_set_full_kernels` | `ks_embedding_lookup`<br>`ks_gemm`<br>`ks_rms_norm`<br>`ks_rope_gather`<br>`ks_flash_attn`<br>`ks_reshape_and_cache`<br>`ks_paged_attn_decode`<br>`ks_swiglu`<br>`ks_argmax` | residual add<br>tensor reshape/view/allocation<br>Python request/decode loop<br>paged block scheduler | `argmax`=4<br>`embedding_lookup`=4<br>`flash_attn`=36<br>`gemm`=1012<br>`paged_attn_decode`=108<br>`reshape_and_cache`=144<br>`rmsnorm`=580<br>`rope`=144<br>`swiglu`=144 |
| `kernel_set_best_practice` | `ks_embedding_lookup(auto multi-token)`<br>`ks_rms_norm`<br>`ks_rope_gather`<br>`ks_paged_attn_decode(auto short-context)`<br>`ks_reshape_and_cache`<br>`ks_swiglu` | embedding single-token=torch<br>linear=torch/cuBLAS<br>attention prefill/long-context=torch SDPA<br>argmax=torch<br>residual add<br>tensor reshape/view/allocation<br>Python request/decode loop<br>paged block scheduler | `embedding_lookup`=1<br>`paged_attn_decode`=108<br>`reshape_and_cache`=144<br>`rmsnorm`=580<br>`rope`=144<br>`swiglu`=144<br>`torch_argmax`=4<br>`torch_attention_prefill`=36<br>`torch_embedding`=3<br>`torch_linear`=1012 |

Composition ablation from the best-practice path:

| variant | new tok/s | vs best-practice | token match | changed component | notes |
|---|---:|---:|---|---|---|
| `kernel_set_best_practice` | 15.65 | +0.0% | yes | baseline | baseline: torch/cuBLAS linears plus ks non-linear/cache kernels and shape-aware embedding/attention |
| `ks_embedding` | 15.61 | -0.3% | yes | embedding=ks | force kernel-set embedding lookup for every token shape |
| `torch_embedding` | 15.65 | +0.0% | yes | embedding=torch | force torch embedding lookup for every token shape |
| `torch_norm` | 15.09 | -3.6% | yes | norm=torch | replace ks hidden/QK RMSNorm with HF torch modules |
| `torch_rope` | 15.30 | -2.3% | yes | rope=torch | replace ks RoPE gather with torch rotate-half RoPE |
| `torch_cache_write` | 15.53 | -0.8% | yes | cache=torch | replace ks reshape_and_cache with torch cache scatter |
| `ks_attention` | 15.42 | -1.5% | yes | attention=ks | force kernel-set FlashAttn/paged decode for every attention shape |
| `torch_attention` | 15.51 | -0.9% | yes | attention=torch | force torch SDPA/manual decode for every attention shape |
| `torch_swiglu` | 15.62 | -0.2% | yes | swiglu=torch | replace ks SwiGLU with torch silu(gate)*up |
| `ks_argmax` | 15.60 | -0.3% | yes | argmax=ks | force kernel-set argmax instead of torch argmax |
| `manual_torch_ops` | 14.58 | -6.9% | yes | attention=torch<br>cache=torch<br>embedding=torch<br>norm=torch<br>rope=torch<br>swiglu=torch | manual Python engine with torch ops for every replaceable component |

Each row changes one component from the best-practice path unless the name says manual_torch_ops; same prompt, greedy 4-token decode.

## Regenerate

```bash
python benchmarks/render_results_readme.py --root-readme README.md
python benchmarks/persist.py validate benchmarks/results/runs
```
