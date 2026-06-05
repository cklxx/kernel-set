# kernel-set vs SOTA — NVIDIA H20

- **GPU**: NVIDIA H20 (sm_90, CC 9.0, 78 SMs, 95.2 GB)
- **detected via**: kernel_set
- **clocks**: SM 345 MHz, mem 2619 MHz (throttle: 0x0000000000000001)
- **driver**: 535.161.08 | CUDA 12.9 | cuDNN 91002 | power cap 500.00 W | ECC Enabled
- **dtype**: bf16 | TF32 matmul=True (fp32_precision=high)
- **timing**: L2-flush=on | method=cuda-events (flushed) | target-ms=200.0 | iters=auto | warmup=10 | L2-buffer=120 MB
- **kernel-set**: 0.1.0 (backend cuda)
- **torch**: 2.9.1+cu129
- **nvcc**: Cuda compilation tools, release 12.9, V12.9.86
- **timestamp**: 2026-06-05T16:32:42
- **host**: Linux-5.4.250-9-velinux1u2-amd64-x86_64-with-glibc2.39

**Providers**: ok=60 · skip=2 · import-fail=31 · error=10 · incorrect=0. Correctness is gated against a shared fp32 reference at the dtype tolerance BEFORE speed is reported.

Latency cells show **median (min)** microseconds (CUDA events, L2-flushed). The perf column is GB/s (bandwidth-bound ops) or TFLOP/s (compute-bound), with **% of dense peak** for this SKU+dtype. Arch-gated providers SKIP on unsupported SMs (`needs smXX`) and are never imported there.

## attn_prefill

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | kernel-set | bf16 | 26172.2 (26124.6) | 1.3 TFLOP/s | 1.99e-03 | ok |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | flash-attn | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flash_attn' |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | sdpa-flash | bf16 | - | - | 2.01e-03 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | flashinfer | bf16 | 335.7 (334.2) | 102.3 TFLOP/s | 2.01e-03 | ok |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | sgl-attn | bf16 | - | - | - | import-fail: ImportError: cannot import name 'flash_attn_varlen_func' from 'sgl_kernel' (/usr/local/lib/python3.12/dist-packages/sgl_ |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | kernel-set | bf16 | 25914.8 (25848.5) | 1.3 TFLOP/s | 1.92e-03 | ok |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | flash-attn | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flash_attn' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | sdpa-flash | bf16 | - | - | 2.58e-03 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | flashinfer | bf16 | - | - | - | skip: flashinfer single_prefill: b==1 only here |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | sgl-attn | bf16 | - | - | - | import-fail: ImportError: cannot import name 'flash_attn_varlen_func' from 'sgl_kernel' (/usr/local/lib/python3.12/dist-packages/sgl_ |

**kernel-set vs best-SOTA**:

- `b=1,seq=2048,qh=32,kvh=8,hd=128`: kernel-set 26172.2us vs best `flashinfer` 335.7us => **0.01x** (slower)
- `b=4,seq=1024,qh=32,kvh=8,hd=128`: no SOTA provider ran (all skip/fail)

## attn_decode

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | kernel-set | bf16 | 3521.9 (3513.6) | 152.4 GB/s | 2.29e-03 | ok |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | flashinfer | bf16 | 340.8 (339.5) | 1575.5 GB/s | 2.29e-03 | ok |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | sdpa-flash | bf16 | - | - | 2.87e-03 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | sgl-attn | bf16 | - | - | - | import-fail: ImportError: cannot import name 'flash_attn_with_kvcache' from 'sgl_kernel' (/usr/local/lib/python3.12/dist-packages/sgl |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | kernel-set | bf16 | 6139.5 (6127.4) | 174.9 GB/s | 2.66e-03 | ok |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | flashinfer | bf16 | 611.2 (609.8) | 1756.9 GB/s | 2.66e-03 | ok |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | sdpa-flash | bf16 | - | - | 2.73e-03 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | sgl-attn | bf16 | - | - | - | import-fail: ImportError: cannot import name 'flash_attn_with_kvcache' from 'sgl_kernel' (/usr/local/lib/python3.12/dist-packages/sgl |

**kernel-set vs best-SOTA**:

- `seqs=64,ctx=2048,qh=32,kvh=8,hd=128`: kernel-set 3521.9us vs best `flashinfer` 340.8us => **0.10x** (slower)
- `seqs=256,ctx=1024,qh=32,kvh=8,hd=128`: kernel-set 6139.5us vs best `flashinfer` 611.2us => **0.10x** (slower)

## mla_decode

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| mla_decode | seqs=64,ctx=2048,h=128,lora=512,rope=64 | kernel-set | bf16 | 39288.0 (39270.4) | 3.8 GB/s | 2.94e-03 | ok |
| mla_decode | seqs=64,ctx=2048,h=128,lora=512,rope=64 | flash-mla | bf16 | 303.2 (302.0) | 498.0 GB/s | - | ok |
| mla_decode | seqs=64,ctx=2048,h=128,lora=512,rope=64 | sgl-mla | bf16 | - | - | - | import-fail: ImportError: cannot import name 'get_mla_metadata' from 'sgl_kernel' (/usr/local/lib/python3.12/dist-packages/sgl_kernel |

**kernel-set vs best-SOTA**:

- `seqs=64,ctx=2048,h=128,lora=512,rope=64`: kernel-set 39288.0us vs best `flash-mla` 303.2us => **0.01x** (slower)

## gemm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| gemm | M=4096,N=4096,K=4096 | kernel-set | bf16 | 19576.6 (19387.2) | 7.0 TFLOP/s | 2.81e-03 | ok |
| gemm | M=4096,N=4096,K=4096 | torch-cublas | bf16 | 1039.8 (1038.1) | 132.2 TFLOP/s | 2.81e-03 | ok |
| gemm | M=4096,N=4096,K=4096 | torch-compile | bf16 | 1063.0 (1061.0) | 129.3 TFLOP/s | 2.81e-03 | ok |
| gemm | M=8192,N=8192,K=8192 | kernel-set | bf16 | 170115.6 (168856.3) | 6.5 TFLOP/s | 1.88e-03 | ok |
| gemm | M=8192,N=8192,K=8192 | torch-cublas | bf16 | 7935.0 (7924.6) | 138.6 TFLOP/s | 1.87e-03 | ok |
| gemm | M=8192,N=8192,K=8192 | torch-compile | bf16 | 8218.1 (8207.8) | 133.8 TFLOP/s | 1.87e-03 | ok |
| gemm | M=4096,N=14336,K=4096 | kernel-set | bf16 | 80474.7 (80342.0) | 6.0 TFLOP/s | 2.80e-03 | ok |
| gemm | M=4096,N=14336,K=4096 | torch-cublas | bf16 | 3388.1 (3384.1) | 142.0 TFLOP/s | 2.80e-03 | ok |
| gemm | M=4096,N=14336,K=4096 | torch-compile | bf16 | 3543.8 (3539.3) | 135.7 TFLOP/s | 2.80e-03 | ok |

**kernel-set vs best-SOTA**:

- `M=4096,N=4096,K=4096`: kernel-set 19576.6us vs best `torch-cublas` 1039.8us => **0.05x** (slower)
- `M=8192,N=8192,K=8192`: kernel-set 170115.6us vs best `torch-cublas` 7935.0us => **0.05x** (slower)
- `M=4096,N=14336,K=4096`: kernel-set 80474.7us vs best `torch-cublas` 3388.1us => **0.04x** (slower)

## w4a16

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| w4a16 | M=4096,N=4096,K=4096 | kernel-set | bf16 | 19182.0 (19168.4) | 7.2 TFLOP/s | - | ok |
| w4a16 | M=4096,N=4096,K=4096 | vllm-marlin | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| w4a16 | M=8192,N=8192,K=8192 | kernel-set | bf16 | 155949.5 (155883.0) | 7.1 TFLOP/s | - | ok |
| w4a16 | M=8192,N=8192,K=8192 | vllm-marlin | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |

**kernel-set vs best-SOTA**:

- `M=4096,N=4096,K=4096`: no SOTA provider ran (all skip/fail)
- `M=8192,N=8192,K=8192`: no SOTA provider ran (all skip/fail)

## fp8_gemm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fp8_gemm | M=4096,N=4096,K=4096 | torch-scaled-mm | bf16 | 536.7 (530.0) | 256.1 TFLOP/s | 3.43e-02 | ok |
| fp8_gemm | M=4096,N=4096,K=4096 | vllm-cutlass-fp8 | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fp8_gemm | M=4096,N=4096,K=4096 | deepgemm | bf16 | 540.4 (537.7) | 254.3 TFLOP/s | 3.43e-02 | ok |
| fp8_gemm | M=4096,N=4096,K=4096 | sgl-fp8 | bf16 | - | - | - | error: RuntimeError: size of scales_a is not matched |
| fp8_gemm | M=4096,N=4096,K=4096 | sgl-int8 | bf16 | - | - | - | error: RuntimeError: mat_b must be a column major tensor |
| fp8_gemm | M=4096,N=4096,K=4096 | kernel-set | bf16 | 10009.4 (10003.1) | 13.7 TFLOP/s | - | ok |
| fp8_gemm | M=8192,N=8192,K=8192 | torch-scaled-mm | bf16 | 3974.3 (3971.4) | 276.7 TFLOP/s | 3.72e-02 | ok |
| fp8_gemm | M=8192,N=8192,K=8192 | vllm-cutlass-fp8 | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fp8_gemm | M=8192,N=8192,K=8192 | deepgemm | bf16 | 4009.0 (4007.0) | 274.3 TFLOP/s | 3.72e-02 | ok |
| fp8_gemm | M=8192,N=8192,K=8192 | sgl-fp8 | bf16 | - | - | - | error: RuntimeError: size of scales_a is not matched |
| fp8_gemm | M=8192,N=8192,K=8192 | sgl-int8 | bf16 | - | - | - | error: RuntimeError: mat_b must be a column major tensor |
| fp8_gemm | M=8192,N=8192,K=8192 | kernel-set | bf16 | 80750.3 (80728.4) | 13.6 TFLOP/s | - | ok |

**kernel-set vs best-SOTA**:

- `M=4096,N=4096,K=4096`: kernel-set 10009.4us vs best `torch-scaled-mm` 536.7us => **0.05x** (slower)
- `M=8192,N=8192,K=8192`: kernel-set 80750.3us vs best `torch-scaled-mm` 3974.3us => **0.05x** (slower)

## rmsnorm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| rmsnorm | rows=4096,hidden=4096 | kernel-set | bf16 | 89.6 (88.0) | 749.3 GB/s | 1.97e-03 | ok |
| rmsnorm | rows=4096,hidden=4096 | flashinfer-norm | bf16 | 47.4 (45.1) | 1415.1 GB/s | 1.97e-03 | ok |
| rmsnorm | rows=4096,hidden=4096 | vllm-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=4096,hidden=4096 | liger-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'liger_kernel' |
| rmsnorm | rows=4096,hidden=4096 | sgl-norm | bf16 | 46.7 (44.4) | 1438.4 GB/s | 1.97e-03 | ok |
| rmsnorm | rows=8192,hidden=8192 | kernel-set | bf16 | 296.5 (294.2) | 905.4 GB/s | 1.98e-03 | ok |
| rmsnorm | rows=8192,hidden=8192 | flashinfer-norm | bf16 | 158.3 (153.2) | 1696.0 GB/s | 1.98e-03 | ok |
| rmsnorm | rows=8192,hidden=8192 | vllm-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=8192,hidden=8192 | liger-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'liger_kernel' |
| rmsnorm | rows=8192,hidden=8192 | sgl-norm | bf16 | 157.5 (152.5) | 1704.0 GB/s | 1.98e-03 | ok |
| rmsnorm | rows=1,hidden=4096 | kernel-set | bf16 | 22.1 (21.3) | 0.7 GB/s | 2.11e-03 | ok |
| rmsnorm | rows=1,hidden=4096 | flashinfer-norm | bf16 | 10.9 (10.4) | 1.5 GB/s | 2.11e-03 | ok |
| rmsnorm | rows=1,hidden=4096 | vllm-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=1,hidden=4096 | liger-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'liger_kernel' |
| rmsnorm | rows=1,hidden=4096 | sgl-norm | bf16 | 12.5 (11.2) | 1.3 GB/s | 2.11e-03 | ok |

**kernel-set vs best-SOTA**:

- `rows=4096,hidden=4096`: kernel-set 89.6us vs best `sgl-norm` 46.7us => **0.52x** (slower)
- `rows=8192,hidden=8192`: kernel-set 296.5us vs best `sgl-norm` 157.5us => **0.53x** (slower)
- `rows=1,hidden=4096`: kernel-set 22.1us vs best `flashinfer-norm` 10.9us => **0.49x** (slower)

## fused_add_rmsnorm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fused_add_rmsnorm | rows=4096,hidden=4096 | kernel-set | bf16 | 100.7 (99.4) | 1332.8 GB/s | 4.80e-03 | ok |
| fused_add_rmsnorm | rows=4096,hidden=4096 | flashinfer-norm | bf16 | 101.4 (99.7) | 1324.0 GB/s | 1.99e-03 | ok |
| fused_add_rmsnorm | rows=4096,hidden=4096 | vllm-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fused_add_rmsnorm | rows=4096,hidden=4096 | sgl-norm | bf16 | 101.9 (100.4) | 1317.7 GB/s | 1.99e-03 | ok |
| fused_add_rmsnorm | rows=8192,hidden=8192 | kernel-set | bf16 | 368.6 (365.8) | 1456.6 GB/s | 4.12e-03 | ok |
| fused_add_rmsnorm | rows=8192,hidden=8192 | flashinfer-norm | bf16 | 363.2 (359.6) | 1478.0 GB/s | 3.00e-03 | ok |
| fused_add_rmsnorm | rows=8192,hidden=8192 | vllm-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fused_add_rmsnorm | rows=8192,hidden=8192 | sgl-norm | bf16 | 363.6 (360.7) | 1476.5 GB/s | 3.00e-03 | ok |
| fused_add_rmsnorm | rows=1,hidden=4096 | kernel-set | bf16 | 19.2 (18.5) | 1.7 GB/s | 3.85e-03 | ok |
| fused_add_rmsnorm | rows=1,hidden=4096 | flashinfer-norm | bf16 | 15.6 (14.3) | 2.1 GB/s | 2.09e-03 | ok |
| fused_add_rmsnorm | rows=1,hidden=4096 | vllm-norm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fused_add_rmsnorm | rows=1,hidden=4096 | sgl-norm | bf16 | 17.2 (15.0) | 1.9 GB/s | 2.09e-03 | ok |

**kernel-set vs best-SOTA**:

- `rows=4096,hidden=4096`: kernel-set 100.7us vs best `flashinfer-norm` 101.4us => **1.01x** (faster)
- `rows=8192,hidden=8192`: kernel-set 368.6us vs best `flashinfer-norm` 363.2us => **0.99x** (slower)
- `rows=1,hidden=4096`: kernel-set 19.2us vs best `flashinfer-norm` 15.6us => **0.81x** (slower)

## rope

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| rope | tokens=4096,qh=32,kvh=8,hd=128 | kernel-set | bf16 | 145.7 (144.1) | 575.8 GB/s | 1.91e-03 | ok |
| rope | tokens=4096,qh=32,kvh=8,hd=128 | flashinfer-rope | bf16 | - | - | - | error: ValueError: cos_sin_cache should be float32 |
| rope | tokens=4096,qh=32,kvh=8,hd=128 | liger-rope | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'liger_kernel' |
| rope | tokens=4096,qh=32,kvh=8,hd=128 | sgl-rope | bf16 | 114.0 (111.6) | 735.5 GB/s | 3.92e-03 | ok |
| rope | tokens=1,qh=32,kvh=8,hd=128 | kernel-set | bf16 | 22.8 (19.5) | 0.9 GB/s | 3.52e-03 | ok |
| rope | tokens=1,qh=32,kvh=8,hd=128 | flashinfer-rope | bf16 | - | - | - | error: ValueError: cos_sin_cache should be float32 |
| rope | tokens=1,qh=32,kvh=8,hd=128 | liger-rope | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'liger_kernel' |
| rope | tokens=1,qh=32,kvh=8,hd=128 | sgl-rope | bf16 | 20.7 (18.0) | 1.0 GB/s | 3.88e-03 | ok |

**kernel-set vs best-SOTA**:

- `tokens=4096,qh=32,kvh=8,hd=128`: kernel-set 145.7us vs best `sgl-rope` 114.0us => **0.78x** (slower)
- `tokens=1,qh=32,kvh=8,hd=128`: kernel-set 22.8us vs best `sgl-rope` 20.7us => **0.91x** (slower)

## swiglu

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| swiglu | rows=4096,inter=14336 | kernel-set | bf16 | 141.8 (140.8) | 2485.3 GB/s | 1.89e-03 | ok |
| swiglu | rows=4096,inter=14336 | flashinfer-act | bf16 | 135.3 (134.5) | 2604.1 GB/s | 1.89e-03 | ok |
| swiglu | rows=4096,inter=14336 | vllm-act | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| swiglu | rows=4096,inter=14336 | sgl-act | bf16 | 137.1 (135.7) | 2570.0 GB/s | 1.89e-03 | ok |
| swiglu | rows=1,inter=14336 | kernel-set | bf16 | 12.1 (10.6) | 7.1 GB/s | 2.28e-03 | ok |
| swiglu | rows=1,inter=14336 | flashinfer-act | bf16 | 12.5 (11.9) | 6.9 GB/s | 2.28e-03 | ok |
| swiglu | rows=1,inter=14336 | vllm-act | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| swiglu | rows=1,inter=14336 | sgl-act | bf16 | 12.6 (12.1) | 6.8 GB/s | 2.28e-03 | ok |

**kernel-set vs best-SOTA**:

- `rows=4096,inter=14336`: kernel-set 141.8us vs best `flashinfer-act` 135.3us => **0.95x** (slower)
- `rows=1,inter=14336`: kernel-set 12.1us vs best `flashinfer-act` 12.5us => **1.04x** (faster)

## moe_grouped_gemm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| moe_grouped_gemm | tokens=4096,h=4096,inter=14336,E=8,k=2 | kernel-set | bf16 | 247502.2 (247484.6) | 3.9 TFLOP/s | 5.46e-03 | ok |

**kernel-set vs best-SOTA**:

- `tokens=4096,h=4096,inter=14336,E=8,k=2`: no SOTA provider ran (all skip/fail)

## fused_moe

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fused_moe | tokens=4096,h=4096,inter=14336,E=8,k=2 | vllm-fused-moe | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |

**kernel-set vs best-SOTA**:

- `tokens=4096,h=4096,inter=14336,E=8,k=2`: kernel-set did not run

## moe_gate

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| moe_gate | tokens=4096,h=4096,inter=14336,E=8,k=2 | sgl-moe-gate | bf16 | 8.2 (7.3) | 24.0 GB/s | 1.31e-07 | ok |

**kernel-set vs best-SOTA**:

- `tokens=4096,h=4096,inter=14336,E=8,k=2`: kernel-set did not run

## ssm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| ssm | b=4,d=2048,L=2048,n=16 | kernel-set | bf16 | - | - | - | skip: kernel-set has no SSM op (N/A) |

**kernel-set vs best-SOTA**:

- `b=4,d=2048,L=2048,n=16`: kernel-set did not run

## ssm_selective_scan

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| ssm_selective_scan | b=4,d=2048,L=2048,n=16 | mamba-ssm | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'mamba_ssm' |

**kernel-set vs best-SOTA**:

- `b=4,d=2048,L=2048,n=16`: kernel-set did not run

## ssm_causal_conv1d

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| ssm_causal_conv1d | b=4,d=2048,L=2048,n=16 | causal-conv1d | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'causal_conv1d' |

**kernel-set vs best-SOTA**:

- `b=4,d=2048,L=2048,n=16`: kernel-set did not run

## cross_entropy

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| cross_entropy | tokens=4096,vocab=32000 | kernel-set | bf16 | 968.0 (964.6) | 541.6 GB/s | 1.35e-07 | ok |
| cross_entropy | tokens=4096,vocab=32000 | liger-ce | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'liger_kernel' |
| cross_entropy | tokens=4096,vocab=32000 | torch-ce | bf16 | 386.9 (385.7) | 1355.1 GB/s | 2.21e-03 | ok |
| cross_entropy | tokens=8192,vocab=128256 | kernel-set | bf16 | 8313.5 (8307.1) | 505.5 GB/s | 1.21e-07 | ok |
| cross_entropy | tokens=8192,vocab=128256 | liger-ce | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'liger_kernel' |
| cross_entropy | tokens=8192,vocab=128256 | torch-ce | bf16 | 2500.6 (2475.9) | 1680.7 GB/s | 1.99e-03 | ok |

**kernel-set vs best-SOTA**:

- `tokens=4096,vocab=32000`: kernel-set 968.0us vs best `torch-ce` 386.9us => **0.40x** (slower)
- `tokens=8192,vocab=128256`: kernel-set 8313.5us vs best `torch-ce` 2500.6us => **0.30x** (slower)

## fused_linear_ce

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fused_linear_ce | tokens=4096,vocab=32000 | cut-cross-entropy | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'cut_cross_entropy' |
| fused_linear_ce | tokens=8192,vocab=128256 | cut-cross-entropy | bf16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'cut_cross_entropy' |

**kernel-set vs best-SOTA**:

- `tokens=4096,vocab=32000`: kernel-set did not run
- `tokens=8192,vocab=128256`: kernel-set did not run

_Legend: %pk = % of dense peak. `skip: needs smXX` = arch-gated (provider needs a newer GPU; not imported here). `import-fail` = library not installed. `incorrect` = disagrees with the fp32 reference._

