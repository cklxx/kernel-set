# kernel-set vs SOTA — NVIDIA RTX PRO 6000 Blackwell Server Edition

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, CC 12.0, 188 SMs, 95.0 GB)
- **detected via**: kernel_set
- **clocks**: SM 2422 MHz, mem 12481 MHz (throttle: 0x0000000000000000)
- **driver**: 580.82.07 | CUDA 12.8 | cuDNN 91900 | power cap 600.00 W | ECC Enabled
- **dtype**: bf16 | TF32 matmul=True (fp32_precision=high)
- **timing**: L2-flush=on | method=cuda-events (flushed) | target-ms=40.0 | iters=auto | warmup=10 | L2-buffer=256 MB
- **kernel-set**: 0.1.0 (backend cuda)
- **torch**: 2.11.0+cu128
- **nvcc**: Cuda compilation tools, release 12.8, V12.8.93
- **harness commit**: 9f6417f
- **timestamp**: 2026-06-05T07:03:26
- **host**: Linux-6.6.122+-x86_64-with-glibc2.35

**Providers**: ok=20 · skip=0 · import-fail=28 · error=2 · incorrect=0. Correctness is gated against a shared fp32 reference at the dtype tolerance BEFORE speed is reported.

Latency cells show **median (min)** microseconds (CUDA events, L2-flushed). The perf column is GB/s (bandwidth-bound ops) or TFLOP/s (compute-bound), with **% of dense peak** for this SKU+dtype. Arch-gated providers SKIP on unsupported SMs (`needs smXX`) and are never imported there.

## gemm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| gemm | M=4096,N=4096,K=4096 | kernel-set | bf16 | 6062.0 (6042.6) | 22.7 TFLOP/s | 2.83e-03 | ok |
| gemm | M=4096,N=4096,K=4096 | torch-cublas | bf16 | 369.6 (366.7) | 371.9 TFLOP/s | 2.83e-03 | ok |
| gemm | M=4096,N=4096,K=4096 | torch-compile | bf16 | 470.3 (467.2) | 292.2 TFLOP/s | 2.83e-03 | ok |
| gemm | M=8192,N=8192,K=8192 | kernel-set | bf16 | 45888.9 (45861.9) | 24.0 TFLOP/s | 1.80e-03 | ok |
| gemm | M=8192,N=8192,K=8192 | torch-cublas | bf16 | 2727.9 (2720.7) | 403.1 TFLOP/s | 1.79e-03 | ok |
| gemm | M=8192,N=8192,K=8192 | torch-compile | bf16 | 3118.1 (3111.9) | 352.6 TFLOP/s | 1.79e-03 | ok |
| gemm | M=4096,N=14336,K=4096 | kernel-set | bf16 | 20104.2 (20101.2) | 23.9 TFLOP/s | 2.72e-03 | ok |
| gemm | M=4096,N=14336,K=4096 | torch-cublas | bf16 | 1188.7 (1183.7) | 404.7 TFLOP/s | 2.71e-03 | ok |
| gemm | M=4096,N=14336,K=4096 | torch-compile | bf16 | 1412.9 (1408.0) | 340.5 TFLOP/s | 2.71e-03 | ok |

**kernel-set vs best-SOTA**:

- `M=4096,N=4096,K=4096`: kernel-set 6062.0us vs best `torch-cublas` 369.6us => **0.06x** (slower)
- `M=8192,N=8192,K=8192`: kernel-set 45888.9us vs best `torch-cublas` 2727.9us => **0.06x** (slower)
- `M=4096,N=14336,K=4096`: kernel-set 20104.2us vs best `torch-cublas` 1188.7us => **0.06x** (slower)

## attn_prefill

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | kernel-set | bf16 | 15245.7 (15225.1) | 2.3 TFLOP/s | 2.13e-03 | ok |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | flash-attn | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flash_attn' |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | sdpa-flash | bf16 | - | - | 2.13e-03 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | flashinfer | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | sgl-attn | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'sgl_kernel' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | kernel-set | bf16 | 15594.3 (15578.6) | 2.2 TFLOP/s | 2.15e-03 | ok |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | flash-attn | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flash_attn' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | sdpa-flash | bf16 | - | - | 2.75e-03 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | flashinfer | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | sgl-attn | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'sgl_kernel' |

**kernel-set vs best-SOTA**:

- `b=1,seq=2048,qh=32,kvh=8,hd=128`: no SOTA provider ran (all skip/fail)
- `b=4,seq=1024,qh=32,kvh=8,hd=128`: no SOTA provider ran (all skip/fail)

## rmsnorm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| rmsnorm | rows=4096,hidden=4096 | kernel-set | bf16 | 62.8 (58.7) | 1067.8 GB/s | 2.46e-03 | ok |
| rmsnorm | rows=4096,hidden=4096 | flashinfer-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| rmsnorm | rows=4096,hidden=4096 | vllm-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=4096,hidden=4096 | liger-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'liger_kernel' |
| rmsnorm | rows=4096,hidden=4096 | sgl-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'sgl_kernel' |
| rmsnorm | rows=8192,hidden=8192 | kernel-set | bf16 | 203.5 (198.7) | 1319.2 GB/s | 1.77e-03 | ok |
| rmsnorm | rows=8192,hidden=8192 | flashinfer-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| rmsnorm | rows=8192,hidden=8192 | vllm-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=8192,hidden=8192 | liger-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'liger_kernel' |
| rmsnorm | rows=8192,hidden=8192 | sgl-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'sgl_kernel' |
| rmsnorm | rows=1,hidden=4096 | kernel-set | bf16 | 16.4 (12.2) | 1.0 GB/s | 1.94e-03 | ok |
| rmsnorm | rows=1,hidden=4096 | flashinfer-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| rmsnorm | rows=1,hidden=4096 | vllm-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=1,hidden=4096 | liger-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'liger_kernel' |
| rmsnorm | rows=1,hidden=4096 | sgl-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'sgl_kernel' |

**kernel-set vs best-SOTA**:

- `rows=4096,hidden=4096`: no SOTA provider ran (all skip/fail)
- `rows=8192,hidden=8192`: no SOTA provider ran (all skip/fail)
- `rows=1,hidden=4096`: no SOTA provider ran (all skip/fail)

## swiglu

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| swiglu | rows=4096,inter=14336 | kernel-set | bf16 | 251.3 (247.2) | 1402.0 GB/s | 3.06e-03 | ok |
| swiglu | rows=4096,inter=14336 | flashinfer-act | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| swiglu | rows=4096,inter=14336 | vllm-act | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| swiglu | rows=4096,inter=14336 | sgl-act | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'sgl_kernel' |
| swiglu | rows=1,inter=14336 | kernel-set | bf16 | 13.7 (9.6) | 6.3 GB/s | 2.00e-03 | ok |
| swiglu | rows=1,inter=14336 | flashinfer-act | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| swiglu | rows=1,inter=14336 | vllm-act | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| swiglu | rows=1,inter=14336 | sgl-act | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'sgl_kernel' |

**kernel-set vs best-SOTA**:

- `rows=4096,inter=14336`: no SOTA provider ran (all skip/fail)
- `rows=1,inter=14336`: no SOTA provider ran (all skip/fail)

## cross_entropy

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| cross_entropy | tokens=4096,vocab=32000 | kernel-set | bf16 | 465.2 (460.8) | 1127.0 GB/s | 1.34e-07 | ok |
| cross_entropy | tokens=4096,vocab=32000 | liger-ce | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'liger_kernel' |
| cross_entropy | tokens=4096,vocab=32000 | torch-ce | bf16 | 443.5 (397.3) | 1182.1 GB/s | 2.20e-03 | ok |
| cross_entropy | tokens=8192,vocab=128256 | kernel-set | bf16 | 4361.7 (4352.0) | 963.5 GB/s | 1.16e-07 | ok |
| cross_entropy | tokens=8192,vocab=128256 | liger-ce | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'liger_kernel' |
| cross_entropy | tokens=8192,vocab=128256 | torch-ce | bf16 | 3440.8 (3414.8) | 1221.4 GB/s | 1.91e-03 | ok |

**kernel-set vs best-SOTA**:

- `tokens=4096,vocab=32000`: kernel-set 465.2us vs best `torch-ce` 443.5us => **0.95x** (slower)
- `tokens=8192,vocab=128256`: kernel-set 4361.7us vs best `torch-ce` 3440.8us => **0.79x** (slower)

## fused_linear_ce

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fused_linear_ce | tokens=4096,vocab=32000 | cut-cross-entropy | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'cut_cross_entropy' |
| fused_linear_ce | tokens=8192,vocab=128256 | cut-cross-entropy | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'cut_cross_entropy' |

**kernel-set vs best-SOTA**:

- `tokens=4096,vocab=32000`: kernel-set did not run
- `tokens=8192,vocab=128256`: kernel-set did not run

_Legend: %pk = % of dense peak. `skip: needs smXX` = arch-gated (provider needs a newer GPU; not imported here). `import-fail` = library not installed. `incorrect` = disagrees with the fp32 reference._

