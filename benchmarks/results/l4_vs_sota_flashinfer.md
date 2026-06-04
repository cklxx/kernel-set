# kernel-set vs SOTA — NVIDIA L4

- **GPU**: NVIDIA L4 (sm_89, CC 8.9, 58 SMs, 22.0 GB) | dense peak BW ~300 GB/s | dense fp16/bf16 TC ~121 TFLOP/s | dense fp8/int8 TC ~242 TFLOP/s
- **detected via**: kernel_set
- **clocks**: SM 210 MHz, mem 405 MHz (throttle: 0x0000000000000001)
- **driver**: 580.82.07 | CUDA 12.4 | cuDNN 90100 | power cap 72.00 W | ECC Enabled
- **dtype**: fp16 | TF32 matmul=True (fp32_precision=high)
- **timing**: L2-flush=on | method=cuda-events (flushed) | target-ms=50.0 | iters=auto | warmup=10 | L2-buffer=96 MB
- **kernel-set**: 0.1.0 (backend cuda)
- **torch**: 2.5.1+cu124
- **nvcc**: Cuda compilation tools, release 12.8, V12.8.93
- **harness commit**: 14c0ad8
- **timestamp**: 2026-06-04T16:21:49
- **host**: Linux-6.6.122+-x86_64-with-glibc2.35

**Providers**: ok=45 · skip=1 · import-fail=12 · error=6 · incorrect=0. Correctness is gated against a shared fp32 reference at the dtype tolerance BEFORE speed is reported.

Latency cells show **median (min)** microseconds (CUDA events, L2-flushed). The perf column is GB/s (bandwidth-bound ops) or TFLOP/s (compute-bound), with **% of dense peak** for this SKU+dtype. Arch-gated providers SKIP on unsupported SMs (`needs smXX`) and are never imported there.

## attn_prefill

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 51517.0 (50986.0) | 0.7 TFLOP/s (1%) | 2.91e-04 | ok |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | flash-attn | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flash_attn' |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 3.18e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | flashinfer | fp16 | 599.0 (521.2) | 57.4 TFLOP/s (47%) | 3.07e-04 | ok |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 51519.0 (51011.6) | 0.7 TFLOP/s (1%) | 2.53e-04 | ok |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | flash-attn | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flash_attn' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 3.45e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | flashinfer | fp16 | - | - | - | skip: flashinfer single_prefill: b==1 only here |

**kernel-set vs best-SOTA**:

- `b=1,seq=2048,qh=32,kvh=8,hd=128`: kernel-set 51517.0us vs best `flashinfer` 599.0us => **0.01x** (slower)
- `b=4,seq=1024,qh=32,kvh=8,hd=128`: no SOTA provider ran (all skip/fail)

## attn_decode

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 7792.1 (7687.2) | 68.9 GB/s (23%) | 3.13e-04 | ok |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | flashinfer | fp16 | 2283.5 (2279.4) | 235.1 GB/s (78%) | 3.13e-04 | ok |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 3.65e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 14485.5 (14404.6) | 74.1 GB/s (25%) | 2.69e-04 | ok |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | flashinfer | fp16 | 4436.0 (4431.9) | 242.1 GB/s (81%) | 2.69e-04 | ok |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 2.69e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |

**kernel-set vs best-SOTA**:

- `seqs=64,ctx=2048,qh=32,kvh=8,hd=128`: kernel-set 7792.1us vs best `flashinfer` 2283.5us => **0.29x** (slower)
- `seqs=256,ctx=1024,qh=32,kvh=8,hd=128`: kernel-set 14485.5us vs best `flashinfer` 4436.0us => **0.31x** (slower)

## rmsnorm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| rmsnorm | rows=4096,hidden=4096 | kernel-set | fp16 | 312.3 (307.2) | 214.9 GB/s (72%) | 2.72e-04 | ok |
| rmsnorm | rows=4096,hidden=4096 | flashinfer-norm | fp16 | 301.1 (294.9) | 222.9 GB/s (74%) | 2.72e-04 | ok |
| rmsnorm | rows=4096,hidden=4096 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=4096,hidden=4096 | liger-norm | fp16 | 299.0 (290.8) | 224.4 GB/s (75%) | 5.23e-04 | ok |
| rmsnorm | rows=8192,hidden=8192 | kernel-set | fp16 | 1179.6 (1168.4) | 227.6 GB/s (76%) | 2.03e-04 | ok |
| rmsnorm | rows=8192,hidden=8192 | flashinfer-norm | fp16 | 1140.7 (1131.5) | 235.3 GB/s (78%) | 2.03e-04 | ok |
| rmsnorm | rows=8192,hidden=8192 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=8192,hidden=8192 | liger-norm | fp16 | 1145.9 (1138.7) | 234.3 GB/s (78%) | 4.07e-04 | ok |
| rmsnorm | rows=1,hidden=4096 | kernel-set | fp16 | 23.6 (21.5) | 0.7 GB/s (0%) | 3.11e-04 | ok |
| rmsnorm | rows=1,hidden=4096 | flashinfer-norm | fp16 | 12.3 (10.2) | 1.3 GB/s (0%) | 3.11e-04 | ok |
| rmsnorm | rows=1,hidden=4096 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=1,hidden=4096 | liger-norm | fp16 | 11.3 (9.2) | 1.5 GB/s (0%) | 6.15e-04 | ok |

**kernel-set vs best-SOTA**:

- `rows=4096,hidden=4096`: kernel-set 312.3us vs best `liger-norm` 299.0us => **0.96x** (slower)
- `rows=8192,hidden=8192`: kernel-set 1179.6us vs best `flashinfer-norm` 1140.7us => **0.97x** (slower)
- `rows=1,hidden=4096`: kernel-set 23.6us vs best `liger-norm` 11.3us => **0.48x** (slower)

## fused_add_rmsnorm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fused_add_rmsnorm | rows=4096,hidden=4096 | kernel-set | fp16 | 614.4 (605.2) | 218.5 GB/s (73%) | 4.87e-04 | ok |
| fused_add_rmsnorm | rows=4096,hidden=4096 | flashinfer-norm | fp16 | 1139.7 (1126.4) | 117.8 GB/s (39%) | 2.89e-04 | ok |
| fused_add_rmsnorm | rows=4096,hidden=4096 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fused_add_rmsnorm | rows=8192,hidden=8192 | kernel-set | fp16 | 2335.7 (2323.5) | 229.9 GB/s (77%) | 5.55e-04 | ok |
| fused_add_rmsnorm | rows=8192,hidden=8192 | flashinfer-norm | fp16 | 4610.0 (4583.4) | 116.5 GB/s (39%) | 3.63e-04 | ok |
| fused_add_rmsnorm | rows=8192,hidden=8192 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fused_add_rmsnorm | rows=1,hidden=4096 | kernel-set | fp16 | 20.5 (17.4) | 1.6 GB/s (1%) | 4.58e-04 | ok |
| fused_add_rmsnorm | rows=1,hidden=4096 | flashinfer-norm | fp16 | 15.4 (13.3) | 2.1 GB/s (1%) | 2.73e-04 | ok |
| fused_add_rmsnorm | rows=1,hidden=4096 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |

**kernel-set vs best-SOTA**:

- `rows=4096,hidden=4096`: kernel-set 614.4us vs best `flashinfer-norm` 1139.7us => **1.85x** (faster)
- `rows=8192,hidden=8192`: kernel-set 2335.7us vs best `flashinfer-norm` 4610.0us => **1.97x** (faster)
- `rows=1,hidden=4096`: kernel-set 20.5us vs best `flashinfer-norm` 15.4us => **0.75x** (slower)

## rope

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| rope | tokens=4096,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 601.1 (592.9) | 139.6 GB/s (47%) | 2.71e-04 | ok |
| rope | tokens=4096,qh=32,kvh=8,hd=128 | flashinfer-rope | fp16 | - | - | - | error: ValueError: cos_sin_cache should be float32 |
| rope | tokens=4096,qh=32,kvh=8,hd=128 | liger-rope | fp16 | 376.8 (368.6) | 222.6 GB/s (74%) | 3.79e-04 | ok |
| rope | tokens=1,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 14.3 (12.3) | 1.4 GB/s (0%) | 2.92e-04 | ok |
| rope | tokens=1,qh=32,kvh=8,hd=128 | flashinfer-rope | fp16 | - | - | - | error: ValueError: cos_sin_cache should be float32 |
| rope | tokens=1,qh=32,kvh=8,hd=128 | liger-rope | fp16 | 6.1 (5.1) | 3.3 GB/s (1%) | 2.92e-04 | ok |

**kernel-set vs best-SOTA**:

- `tokens=4096,qh=32,kvh=8,hd=128`: kernel-set 601.1us vs best `liger-rope` 376.8us => **0.63x** (slower)
- `tokens=1,qh=32,kvh=8,hd=128`: kernel-set 14.3us vs best `liger-rope` 6.1us => **0.43x** (slower)

## swiglu

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| swiglu | rows=4096,inter=14336 | kernel-set | fp16 | 1539.1 (1525.8) | 228.9 GB/s (76%) | 2.54e-04 | ok |
| swiglu | rows=4096,inter=14336 | flashinfer-act | fp16 | 1532.9 (1518.6) | 229.8 GB/s (77%) | 2.54e-04 | ok |
| swiglu | rows=4096,inter=14336 | vllm-act | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| swiglu | rows=1,inter=14336 | kernel-set | fp16 | 12.3 (10.2) | 7.0 GB/s (2%) | 2.59e-04 | ok |
| swiglu | rows=1,inter=14336 | flashinfer-act | fp16 | 13.3 (11.3) | 6.5 GB/s (2%) | 2.59e-04 | ok |
| swiglu | rows=1,inter=14336 | vllm-act | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |

**kernel-set vs best-SOTA**:

- `rows=4096,inter=14336`: kernel-set 1539.1us vs best `flashinfer-act` 1532.9us => **1.00x** (slower)
- `rows=1,inter=14336`: kernel-set 12.3us vs best `flashinfer-act` 13.3us => **1.08x** (faster)

## gemm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| gemm | M=4096,N=4096,K=4096 | kernel-set | fp16 | 28053.5 (27709.4) | 4.9 TFLOP/s (4%) | 3.60e-04 | ok |
| gemm | M=4096,N=4096,K=4096 | torch-cublas | fp16 | 2689.0 (2034.7) | 51.1 TFLOP/s (42%) | 3.51e-04 | ok |
| gemm | M=4096,N=4096,K=4096 | torch-compile | fp16 | 3128.3 (2704.4) | 43.9 TFLOP/s (36%) | 3.51e-04 | ok |
| gemm | M=8192,N=8192,K=8192 | kernel-set | fp16 | 280090.1 (277763.1) | 3.9 TFLOP/s (3%) | 3.70e-04 | ok |
| gemm | M=8192,N=8192,K=8192 | torch-cublas | fp16 | 21503.5 (20828.2) | 51.1 TFLOP/s (42%) | 5.56e-04 | ok |
| gemm | M=8192,N=8192,K=8192 | torch-compile | fp16 | 21190.7 (20393.0) | 51.9 TFLOP/s (43%) | 5.56e-04 | ok |
| gemm | M=4096,N=14336,K=4096 | kernel-set | fp16 | 123858.9 (122249.2) | 3.9 TFLOP/s (3%) | 3.42e-04 | ok |
| gemm | M=4096,N=14336,K=4096 | torch-cublas | fp16 | 8609.3 (8352.8) | 55.9 TFLOP/s (46%) | 3.36e-04 | ok |
| gemm | M=4096,N=14336,K=4096 | torch-compile | fp16 | 8953.9 (8840.2) | 53.7 TFLOP/s (44%) | 3.36e-04 | ok |

**kernel-set vs best-SOTA**:

- `M=4096,N=4096,K=4096`: kernel-set 28053.5us vs best `torch-cublas` 2689.0us => **0.10x** (slower)
- `M=8192,N=8192,K=8192`: kernel-set 280090.1us vs best `torch-compile` 21190.7us => **0.08x** (slower)
- `M=4096,N=14336,K=4096`: kernel-set 123858.9us vs best `torch-cublas` 8609.3us => **0.07x** (slower)

## cross_entropy

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| cross_entropy | tokens=4096,vocab=32000 | kernel-set | fp16 | 2475.0 (2439.2) | 211.8 GB/s (71%) | 1.32e-07 | ok |
| cross_entropy | tokens=4096,vocab=32000 | liger-ce | fp16 | 1386.5 (1359.9) | 378.1 GB/s (126%) | 2.69e-04 | ok |
| cross_entropy | tokens=4096,vocab=32000 | torch-ce | fp16 | 2274.3 (2262.0) | 230.5 GB/s (77%) | 2.69e-04 | ok |
| cross_entropy | tokens=8192,vocab=128256 | kernel-set | fp16 | 26973.2 (26866.7) | 155.8 GB/s (52%) | 1.20e-07 | ok |
| cross_entropy | tokens=8192,vocab=128256 | liger-ce | fp16 | 8391.7 (8380.4) | 500.8 GB/s (167%) | 2.46e-04 | ok |
| cross_entropy | tokens=8192,vocab=128256 | torch-ce | fp16 | 18350.1 (18302.0) | 229.0 GB/s (76%) | 2.46e-04 | ok |

**kernel-set vs best-SOTA**:

- `tokens=4096,vocab=32000`: kernel-set 2475.0us vs best `liger-ce` 1386.5us => **0.56x** (slower)
- `tokens=8192,vocab=128256`: kernel-set 26973.2us vs best `liger-ce` 8391.7us => **0.31x** (slower)

## fused_linear_ce

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fused_linear_ce | tokens=4096,vocab=32000 | cut-cross-entropy | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'cut_cross_entropy' |
| fused_linear_ce | tokens=8192,vocab=128256 | cut-cross-entropy | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'cut_cross_entropy' |

**kernel-set vs best-SOTA**:

- `tokens=4096,vocab=32000`: kernel-set did not run
- `tokens=8192,vocab=128256`: kernel-set did not run

_Legend: %pk = % of dense peak. `skip: needs smXX` = arch-gated (provider needs a newer GPU; not imported here). `import-fail` = library not installed. `incorrect` = disagrees with the fp32 reference._

