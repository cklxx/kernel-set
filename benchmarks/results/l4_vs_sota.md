# kernel-set vs SOTA — NVIDIA L4

- **GPU**: NVIDIA L4 (sm_89, CC 8.9, 58 SMs, 22.0 GB) | dense peak BW ~300 GB/s | dense fp16/bf16 TC ~121 TFLOP/s | dense fp8/int8 TC ~242 TFLOP/s
- **detected via**: kernel_set
- **clocks**: SM 210 MHz, mem 405 MHz (throttle: 0x0000000000000001)
- **driver**: 580.82.07 | CUDA 12.4 | cuDNN 90100 | power cap 72.00 W | ECC Enabled
- **dtype**: fp16 | TF32 matmul=True (fp32_precision=high)
- **timing**: L2-flush=on | method=cuda-events (flushed) | target-ms=60.0 | iters=auto | warmup=10 | L2-buffer=96 MB
- **kernel-set**: 0.1.0 (backend cuda)
- **torch**: 2.5.1+cu124
- **nvcc**: Cuda compilation tools, release 12.8, V12.8.93
- **harness commit**: b78616c
- **timestamp**: 2026-06-05T06:23:21
- **host**: Linux-6.6.122+-x86_64-with-glibc2.35

**Providers**: ok=51 · skip=12 · import-fail=32 · error=6 · incorrect=2. Correctness is gated against a shared fp32 reference at the dtype tolerance BEFORE speed is reported.

Latency cells show **median (min)** microseconds (CUDA events, L2-flushed). The perf column is GB/s (bandwidth-bound ops) or TFLOP/s (compute-bound), with **% of dense peak** for this SKU+dtype. Arch-gated providers SKIP on unsupported SMs (`needs smXX`) and are never imported there.

## attn_prefill

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 51571.7 (51300.4) | 0.7 TFLOP/s (1%) | 2.87e-04 | ok |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | flash-attn | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flash_attn' |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 3.37e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | flashinfer | fp16 | 603.6 (521.3) | 56.9 TFLOP/s (47%) | 3.31e-04 | ok |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | sgl-attn | fp16 | - | - | - | skip: needs sm90 (device is sm89) |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 51627.0 (51245.1) | 0.7 TFLOP/s (1%) | 2.16e-04 | ok |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | flash-attn | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flash_attn' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 2.31e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | flashinfer | fp16 | - | - | - | skip: flashinfer single_prefill: b==1 only here |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | sgl-attn | fp16 | - | - | - | skip: needs sm90 (device is sm89) |

**kernel-set vs best-SOTA**:

- `b=1,seq=2048,qh=32,kvh=8,hd=128`: kernel-set 51571.7us vs best `flashinfer` 603.6us => **0.01x** (slower)
- `b=4,seq=1024,qh=32,kvh=8,hd=128`: no SOTA provider ran (all skip/fail)

## attn_decode

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 7866.4 (7820.3) | 68.2 GB/s (23%) | 3.47e-04 | ok |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | flashinfer | fp16 | 2285.6 (2279.4) | 234.9 GB/s (78%) | 3.47e-04 | ok |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 4.42e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | sgl-attn | fp16 | - | - | - | skip: needs sm90 (device is sm89) |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 14485.0 (14351.4) | 74.1 GB/s (25%) | 2.98e-04 | ok |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | flashinfer | fp16 | 4434.9 (4431.9) | 242.1 GB/s (81%) | 2.98e-04 | ok |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 3.25e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | sgl-attn | fp16 | - | - | - | skip: needs sm90 (device is sm89) |

**kernel-set vs best-SOTA**:

- `seqs=64,ctx=2048,qh=32,kvh=8,hd=128`: kernel-set 7866.4us vs best `flashinfer` 2285.6us => **0.29x** (slower)
- `seqs=256,ctx=1024,qh=32,kvh=8,hd=128`: kernel-set 14485.0us vs best `flashinfer` 4434.9us => **0.31x** (slower)

## mla_decode

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| mla_decode | seqs=64,ctx=2048,h=128,lora=512,rope=64 | kernel-set | fp16 | 79271.9 (77232.1) | 1.9 GB/s (1%) | 5.48e-04 | ok |
| mla_decode | seqs=64,ctx=2048,h=128,lora=512,rope=64 | flash-mla | fp16 | - | - | - | skip: needs sm90 (device is sm89) |
| mla_decode | seqs=64,ctx=2048,h=128,lora=512,rope=64 | sgl-mla | fp16 | - | - | - | skip: needs sm90 (device is sm89) |

**kernel-set vs best-SOTA**:

- `seqs=64,ctx=2048,h=128,lora=512,rope=64`: no SOTA provider ran (all skip/fail)

## gemm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| gemm | M=4096,N=4096,K=4096 | kernel-set | fp16 | 28212.7 (27968.5) | 4.9 TFLOP/s (4%) | 3.53e-04 | ok |
| gemm | M=4096,N=4096,K=4096 | torch-cublas | fp16 | 2647.0 (1858.6) | 51.9 TFLOP/s (43%) | 3.47e-04 | ok |
| gemm | M=4096,N=4096,K=4096 | torch-compile | fp16 | 2928.6 (2575.4) | 46.9 TFLOP/s (39%) | 3.47e-04 | ok |
| gemm | M=8192,N=8192,K=8192 | kernel-set | fp16 | 278436.4 (275313.7) | 3.9 TFLOP/s (3%) | 2.76e-04 | ok |
| gemm | M=8192,N=8192,K=8192 | torch-cublas | fp16 | 20302.8 (19893.2) | 54.2 TFLOP/s (45%) | 6.16e-04 | ok |
| gemm | M=8192,N=8192,K=8192 | torch-compile | fp16 | 20760.6 (19917.8) | 53.0 TFLOP/s (44%) | 6.16e-04 | ok |
| gemm | M=4096,N=14336,K=4096 | kernel-set | fp16 | 122719.2 (119906.3) | 3.9 TFLOP/s (3%) | 3.35e-04 | ok |
| gemm | M=4096,N=14336,K=4096 | torch-cublas | fp16 | 8234.5 (8233.0) | 58.4 TFLOP/s (48%) | 3.28e-04 | ok |
| gemm | M=4096,N=14336,K=4096 | torch-compile | fp16 | 8761.9 (8731.6) | 54.9 TFLOP/s (45%) | 3.28e-04 | ok |

**kernel-set vs best-SOTA**:

- `M=4096,N=4096,K=4096`: kernel-set 28212.7us vs best `torch-cublas` 2647.0us => **0.09x** (slower)
- `M=8192,N=8192,K=8192`: kernel-set 278436.4us vs best `torch-cublas` 20302.8us => **0.07x** (slower)
- `M=4096,N=14336,K=4096`: kernel-set 122719.2us vs best `torch-cublas` 8234.5us => **0.07x** (slower)

## w4a16

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| w4a16 | M=4096,N=4096,K=4096 | kernel-set | fp16 | 33610.8 (32958.5) | 4.1 TFLOP/s (3%) | - | ok |
| w4a16 | M=4096,N=4096,K=4096 | vllm-marlin | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| w4a16 | M=8192,N=8192,K=8192 | kernel-set | fp16 | 262690.8 (260451.3) | 4.2 TFLOP/s (3%) | - | ok |
| w4a16 | M=8192,N=8192,K=8192 | vllm-marlin | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |

**kernel-set vs best-SOTA**:

- `M=4096,N=4096,K=4096`: no SOTA provider ran (all skip/fail)
- `M=8192,N=8192,K=8192`: no SOTA provider ran (all skip/fail)

## fp8_gemm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fp8_gemm | M=4096,N=4096,K=4096 | torch-scaled-mm | fp16 | - | - | 3.89e-02 | incorrect (rel_err=3.89e-02>tol=2.0e-02) |
| fp8_gemm | M=4096,N=4096,K=4096 | vllm-cutlass-fp8 | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fp8_gemm | M=4096,N=4096,K=4096 | deepgemm | fp16 | - | - | - | skip: needs sm90 (device is sm89) |
| fp8_gemm | M=4096,N=4096,K=4096 | sgl-fp8 | fp16 | - | - | - | skip: needs sm90 (device is sm89) |
| fp8_gemm | M=4096,N=4096,K=4096 | sgl-int8 | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |
| fp8_gemm | M=4096,N=4096,K=4096 | kernel-set | fp16 | 24344.6 (24144.9) | 5.6 TFLOP/s (2%) | - | ok |
| fp8_gemm | M=8192,N=8192,K=8192 | torch-scaled-mm | fp16 | - | - | 3.81e-02 | incorrect (rel_err=3.81e-02>tol=2.0e-02) |
| fp8_gemm | M=8192,N=8192,K=8192 | vllm-cutlass-fp8 | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fp8_gemm | M=8192,N=8192,K=8192 | deepgemm | fp16 | - | - | - | skip: needs sm90 (device is sm89) |
| fp8_gemm | M=8192,N=8192,K=8192 | sgl-fp8 | fp16 | - | - | - | skip: needs sm90 (device is sm89) |
| fp8_gemm | M=8192,N=8192,K=8192 | sgl-int8 | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |
| fp8_gemm | M=8192,N=8192,K=8192 | kernel-set | fp16 | 241431.0 (238478.3) | 4.6 TFLOP/s (2%) | - | ok |

**kernel-set vs best-SOTA**:

- `M=4096,N=4096,K=4096`: no SOTA provider ran (all skip/fail)
- `M=8192,N=8192,K=8192`: no SOTA provider ran (all skip/fail)

## rmsnorm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| rmsnorm | rows=4096,hidden=4096 | kernel-set | fp16 | 313.3 (308.2) | 214.2 GB/s (71%) | 2.76e-04 | ok |
| rmsnorm | rows=4096,hidden=4096 | flashinfer-norm | fp16 | 300.0 (293.9) | 223.7 GB/s (75%) | 2.76e-04 | ok |
| rmsnorm | rows=4096,hidden=4096 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=4096,hidden=4096 | liger-norm | fp16 | 301.1 (292.9) | 222.9 GB/s (74%) | 6.25e-04 | ok |
| rmsnorm | rows=4096,hidden=4096 | sgl-norm | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |
| rmsnorm | rows=8192,hidden=8192 | kernel-set | fp16 | 1172.5 (1161.2) | 228.9 GB/s (76%) | 2.35e-04 | ok |
| rmsnorm | rows=8192,hidden=8192 | flashinfer-norm | fp16 | 1150.0 (1136.6) | 233.4 GB/s (78%) | 2.35e-04 | ok |
| rmsnorm | rows=8192,hidden=8192 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=8192,hidden=8192 | liger-norm | fp16 | 1150.0 (1137.7) | 233.4 GB/s (78%) | 5.13e-04 | ok |
| rmsnorm | rows=8192,hidden=8192 | sgl-norm | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |
| rmsnorm | rows=1,hidden=4096 | kernel-set | fp16 | 23.6 (21.5) | 0.7 GB/s (0%) | 2.90e-04 | ok |
| rmsnorm | rows=1,hidden=4096 | flashinfer-norm | fp16 | 12.3 (10.2) | 1.3 GB/s (0%) | 2.90e-04 | ok |
| rmsnorm | rows=1,hidden=4096 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=1,hidden=4096 | liger-norm | fp16 | 12.3 (10.2) | 1.3 GB/s (0%) | 5.36e-04 | ok |
| rmsnorm | rows=1,hidden=4096 | sgl-norm | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |

**kernel-set vs best-SOTA**:

- `rows=4096,hidden=4096`: kernel-set 313.3us vs best `flashinfer-norm` 300.0us => **0.96x** (slower)
- `rows=8192,hidden=8192`: kernel-set 1172.5us vs best `flashinfer-norm` 1150.0us => **0.98x** (slower)
- `rows=1,hidden=4096`: kernel-set 23.6us vs best `flashinfer-norm` 12.3us => **0.52x** (slower)

## fused_add_rmsnorm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fused_add_rmsnorm | rows=4096,hidden=4096 | kernel-set | fp16 | 615.4 (606.2) | 218.1 GB/s (73%) | 5.75e-04 | ok |
| fused_add_rmsnorm | rows=4096,hidden=4096 | flashinfer-norm | fp16 | 1137.7 (1129.5) | 118.0 GB/s (39%) | 2.68e-04 | ok |
| fused_add_rmsnorm | rows=4096,hidden=4096 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fused_add_rmsnorm | rows=4096,hidden=4096 | sgl-norm | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |
| fused_add_rmsnorm | rows=8192,hidden=8192 | kernel-set | fp16 | 2333.7 (2325.5) | 230.1 GB/s (77%) | 5.65e-04 | ok |
| fused_add_rmsnorm | rows=8192,hidden=8192 | flashinfer-norm | fp16 | 4622.8 (4602.9) | 116.1 GB/s (39%) | 4.74e-04 | ok |
| fused_add_rmsnorm | rows=8192,hidden=8192 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fused_add_rmsnorm | rows=8192,hidden=8192 | sgl-norm | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |
| fused_add_rmsnorm | rows=1,hidden=4096 | kernel-set | fp16 | 19.5 (15.4) | 1.7 GB/s (1%) | 4.66e-04 | ok |
| fused_add_rmsnorm | rows=1,hidden=4096 | flashinfer-norm | fp16 | 15.4 (13.3) | 2.1 GB/s (1%) | 3.00e-04 | ok |
| fused_add_rmsnorm | rows=1,hidden=4096 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fused_add_rmsnorm | rows=1,hidden=4096 | sgl-norm | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |

**kernel-set vs best-SOTA**:

- `rows=4096,hidden=4096`: kernel-set 615.4us vs best `flashinfer-norm` 1137.7us => **1.85x** (faster)
- `rows=8192,hidden=8192`: kernel-set 2333.7us vs best `flashinfer-norm` 4622.8us => **1.98x** (faster)
- `rows=1,hidden=4096`: kernel-set 19.5us vs best `flashinfer-norm` 15.4us => **0.79x** (slower)

## rope

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| rope | tokens=4096,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 597.0 (585.7) | 140.5 GB/s (47%) | 3.59e-04 | ok |
| rope | tokens=4096,qh=32,kvh=8,hd=128 | flashinfer-rope | fp16 | - | - | - | error: ValueError: cos_sin_cache should be float32 |
| rope | tokens=4096,qh=32,kvh=8,hd=128 | liger-rope | fp16 | 374.8 (367.6) | 223.8 GB/s (75%) | 4.34e-04 | ok |
| rope | tokens=4096,qh=32,kvh=8,hd=128 | sgl-rope | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |
| rope | tokens=1,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 14.3 (13.3) | 1.4 GB/s (0%) | 2.67e-04 | ok |
| rope | tokens=1,qh=32,kvh=8,hd=128 | flashinfer-rope | fp16 | - | - | - | error: ValueError: cos_sin_cache should be float32 |
| rope | tokens=1,qh=32,kvh=8,hd=128 | liger-rope | fp16 | 6.1 (5.1) | 3.3 GB/s (1%) | 4.52e-04 | ok |
| rope | tokens=1,qh=32,kvh=8,hd=128 | sgl-rope | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |

**kernel-set vs best-SOTA**:

- `tokens=4096,qh=32,kvh=8,hd=128`: kernel-set 597.0us vs best `liger-rope` 374.8us => **0.63x** (slower)
- `tokens=1,qh=32,kvh=8,hd=128`: kernel-set 14.3us vs best `liger-rope` 6.1us => **0.43x** (slower)

## swiglu

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| swiglu | rows=4096,inter=14336 | kernel-set | fp16 | 1540.1 (1526.8) | 228.8 GB/s (76%) | 2.61e-04 | ok |
| swiglu | rows=4096,inter=14336 | flashinfer-act | fp16 | 1535.0 (1518.6) | 229.5 GB/s (77%) | 2.61e-04 | ok |
| swiglu | rows=4096,inter=14336 | vllm-act | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| swiglu | rows=4096,inter=14336 | sgl-act | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |
| swiglu | rows=1,inter=14336 | kernel-set | fp16 | 12.3 (11.3) | 7.0 GB/s (2%) | 2.65e-04 | ok |
| swiglu | rows=1,inter=14336 | flashinfer-act | fp16 | 12.3 (11.3) | 7.0 GB/s (2%) | 2.65e-04 | ok |
| swiglu | rows=1,inter=14336 | vllm-act | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| swiglu | rows=1,inter=14336 | sgl-act | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |

**kernel-set vs best-SOTA**:

- `rows=4096,inter=14336`: kernel-set 1540.1us vs best `flashinfer-act` 1535.0us => **1.00x** (slower)
- `rows=1,inter=14336`: kernel-set 12.3us vs best `flashinfer-act` 12.3us => **1.00x** (faster)

## moe_grouped_gemm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| moe_grouped_gemm | tokens=4096,h=4096,inter=14336,E=8,k=2 | kernel-set | fp16 | 890830.3 (885569.5) | 1.1 TFLOP/s (1%) | 6.93e-04 | ok |

**kernel-set vs best-SOTA**:

- `tokens=4096,h=4096,inter=14336,E=8,k=2`: no SOTA provider ran (all skip/fail)

## fused_moe

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fused_moe | tokens=4096,h=4096,inter=14336,E=8,k=2 | vllm-fused-moe | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |

**kernel-set vs best-SOTA**:

- `tokens=4096,h=4096,inter=14336,E=8,k=2`: kernel-set did not run

## moe_gate

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| moe_gate | tokens=4096,h=4096,inter=14336,E=8,k=2 | sgl-moe-gate | fp16 | - | - | - | import-fail: ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library! Attempted locations: 1. Architecture-specific |

**kernel-set vs best-SOTA**:

- `tokens=4096,h=4096,inter=14336,E=8,k=2`: kernel-set did not run

## ssm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| ssm | b=4,d=2048,L=2048,n=16 | kernel-set | fp16 | - | - | - | skip: kernel-set has no SSM op (N/A) |

**kernel-set vs best-SOTA**:

- `b=4,d=2048,L=2048,n=16`: kernel-set did not run

## ssm_selective_scan

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| ssm_selective_scan | b=4,d=2048,L=2048,n=16 | mamba-ssm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'mamba_ssm' |

**kernel-set vs best-SOTA**:

- `b=4,d=2048,L=2048,n=16`: kernel-set did not run

## ssm_causal_conv1d

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| ssm_causal_conv1d | b=4,d=2048,L=2048,n=16 | causal-conv1d | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'causal_conv1d' |

**kernel-set vs best-SOTA**:

- `b=4,d=2048,L=2048,n=16`: kernel-set did not run

## cross_entropy

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| cross_entropy | tokens=4096,vocab=32000 | kernel-set | fp16 | 2426.9 (2335.7) | 216.0 GB/s (72%) | 1.35e-07 | ok |
| cross_entropy | tokens=4096,vocab=32000 | liger-ce | fp16 | 1377.3 (1366.0) | 380.7 GB/s (127%) | 2.77e-04 | ok |
| cross_entropy | tokens=4096,vocab=32000 | torch-ce | fp16 | 2274.3 (2254.8) | 230.5 GB/s (77%) | 2.77e-04 | ok |
| cross_entropy | tokens=8192,vocab=128256 | kernel-set | fp16 | 26984.4 (26871.8) | 155.7 GB/s (52%) | 1.15e-07 | ok |
| cross_entropy | tokens=8192,vocab=128256 | liger-ce | fp16 | 8394.2 (8371.2) | 500.7 GB/s (167%) | 2.36e-04 | ok |
| cross_entropy | tokens=8192,vocab=128256 | torch-ce | fp16 | 18282.5 (18249.7) | 229.9 GB/s (77%) | 2.36e-04 | ok |

**kernel-set vs best-SOTA**:

- `tokens=4096,vocab=32000`: kernel-set 2426.9us vs best `liger-ce` 1377.3us => **0.57x** (slower)
- `tokens=8192,vocab=128256`: kernel-set 26984.4us vs best `liger-ce` 8394.2us => **0.31x** (slower)

## fused_linear_ce

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fused_linear_ce | tokens=4096,vocab=32000 | cut-cross-entropy | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'cut_cross_entropy' |
| fused_linear_ce | tokens=8192,vocab=128256 | cut-cross-entropy | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'cut_cross_entropy' |

**kernel-set vs best-SOTA**:

- `tokens=4096,vocab=32000`: kernel-set did not run
- `tokens=8192,vocab=128256`: kernel-set did not run

_Legend: %pk = % of dense peak. `skip: needs smXX` = arch-gated (provider needs a newer GPU; not imported here). `import-fail` = library not installed. `incorrect` = disagrees with the fp32 reference._

