# kernel-set vs SOTA — NVIDIA L4

- **GPU**: NVIDIA L4 (sm_89, CC 8.9, 58 SMs, 22.0 GB) | dense peak BW ~300 GB/s | dense fp16/bf16 TC ~121 TFLOP/s | dense fp8/int8 TC ~242 TFLOP/s
- **detected via**: kernel_set
- **clocks**: SM 210 MHz, mem 405 MHz (throttle: 0x0000000000000001)
- **driver**: 580.82.07 | CUDA 12.8 | cuDNN 91900 | power cap 72.00 W | ECC Enabled
- **dtype**: fp16 | TF32 matmul=True (fp32_precision=high)
- **timing**: L2-flush=on | method=cuda-events (flushed) | target-ms=60.0 | iters=auto | warmup=10 | L2-buffer=96 MB
- **kernel-set**: 0.1.0 (backend cuda)
- **torch**: 2.11.0+cu128
- **nvcc**: Cuda compilation tools, release 12.8, V12.8.93
- **harness commit**: 012085a
- **timestamp**: 2026-06-04T15:44:49
- **host**: Linux-6.6.122+-x86_64-with-glibc2.35

**Providers**: ok=40 · skip=4 · import-fail=33 · error=4 · incorrect=2. Correctness is gated against a shared fp32 reference at the dtype tolerance BEFORE speed is reported.

Latency cells show **median (min)** microseconds (CUDA events, L2-flushed). The perf column is GB/s (bandwidth-bound ops) or TFLOP/s (compute-bound), with **% of dense peak** for this SKU+dtype. Arch-gated providers SKIP on unsupported SMs (`needs smXX`) and are never imported there.

## attn_prefill

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 203647.0 (203519.0) | 0.2 TFLOP/s (0%) | 2.74e-04 | ok |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | flash-attn | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flash_attn' |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 3.73e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | flashinfer | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 204036.6 (203947.0) | 0.2 TFLOP/s (0%) | 4.59e-04 | ok |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | flash-attn | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flash_attn' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 4.59e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | flashinfer | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |

**kernel-set vs best-SOTA**:

- `b=1,seq=2048,qh=32,kvh=8,hd=128`: no SOTA provider ran (all skip/fail)
- `b=4,seq=1024,qh=32,kvh=8,hd=128`: no SOTA provider ran (all skip/fail)

## attn_decode

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 7708.7 (7639.0) | 69.6 GB/s (23%) | 3.28e-04 | ok |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | flashinfer | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 4.04e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 14926.8 (14526.5) | 71.9 GB/s (24%) | 3.82e-04 | ok |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | flashinfer | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 3.82e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |

**kernel-set vs best-SOTA**:

- `seqs=64,ctx=2048,qh=32,kvh=8,hd=128`: no SOTA provider ran (all skip/fail)
- `seqs=256,ctx=1024,qh=32,kvh=8,hd=128`: no SOTA provider ran (all skip/fail)

## mla_decode

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| mla_decode | seqs=64,ctx=2048,h=128,lora=512,rope=64 | kernel-set | fp16 | 79367.7 (77388.8) | 1.9 GB/s (1%) | 4.19e-04 | ok |
| mla_decode | seqs=64,ctx=2048,h=128,lora=512,rope=64 | flash-mla | fp16 | - | - | - | skip: needs sm90 (device is sm89) |

**kernel-set vs best-SOTA**:

- `seqs=64,ctx=2048,h=128,lora=512,rope=64`: no SOTA provider ran (all skip/fail)

## gemm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| gemm | M=4096,N=4096,K=4096 | kernel-set | fp16 | 28362.8 (28031.0) | 4.8 TFLOP/s (4%) | 3.44e-04 | ok |
| gemm | M=4096,N=4096,K=4096 | torch-cublas | fp16 | 2605.6 (2128.9) | 52.7 TFLOP/s (44%) | 3.43e-04 | ok |
| gemm | M=4096,N=4096,K=4096 | torch-compile | fp16 | 3011.1 (2686.0) | 45.6 TFLOP/s (38%) | 3.43e-04 | ok |
| gemm | M=8192,N=8192,K=8192 | kernel-set | fp16 | 282169.8 (278807.6) | 3.9 TFLOP/s (3%) | 2.78e-04 | ok |
| gemm | M=8192,N=8192,K=8192 | torch-cublas | fp16 | 20401.7 (19896.3) | 53.9 TFLOP/s (45%) | 6.31e-04 | ok |
| gemm | M=8192,N=8192,K=8192 | torch-compile | fp16 | 20673.5 (20377.6) | 53.2 TFLOP/s (44%) | 6.31e-04 | ok |
| gemm | M=4096,N=14336,K=4096 | kernel-set | fp16 | 123806.7 (121589.8) | 3.9 TFLOP/s (3%) | 3.59e-04 | ok |
| gemm | M=4096,N=14336,K=4096 | torch-cublas | fp16 | 8235.0 (8116.2) | 58.4 TFLOP/s (48%) | 3.53e-04 | ok |
| gemm | M=4096,N=14336,K=4096 | torch-compile | fp16 | 9244.7 (9036.8) | 52.0 TFLOP/s (43%) | 3.53e-04 | ok |

**kernel-set vs best-SOTA**:

- `M=4096,N=4096,K=4096`: kernel-set 28362.8us vs best `torch-cublas` 2605.6us => **0.09x** (slower)
- `M=8192,N=8192,K=8192`: kernel-set 282169.8us vs best `torch-cublas` 20401.7us => **0.07x** (slower)
- `M=4096,N=14336,K=4096`: kernel-set 123806.7us vs best `torch-cublas` 8235.0us => **0.07x** (slower)

## w4a16

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| w4a16 | M=4096,N=4096,K=4096 | kernel-set | fp16 | 33623.0 (32958.5) | 4.1 TFLOP/s (3%) | - | ok |
| w4a16 | M=4096,N=4096,K=4096 | vllm-marlin | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| w4a16 | M=8192,N=8192,K=8192 | kernel-set | fp16 | 263781.4 (261606.4) | 4.2 TFLOP/s (3%) | - | ok |
| w4a16 | M=8192,N=8192,K=8192 | vllm-marlin | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |

**kernel-set vs best-SOTA**:

- `M=4096,N=4096,K=4096`: no SOTA provider ran (all skip/fail)
- `M=8192,N=8192,K=8192`: no SOTA provider ran (all skip/fail)

## fp8_gemm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fp8_gemm | M=4096,N=4096,K=4096 | torch-scaled-mm | fp16 | - | - | 3.88e-02 | incorrect (rel_err=3.88e-02>tol=2.0e-02) |
| fp8_gemm | M=4096,N=4096,K=4096 | vllm-cutlass-fp8 | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fp8_gemm | M=4096,N=4096,K=4096 | deepgemm | fp16 | - | - | - | skip: needs sm90 (device is sm89) |
| fp8_gemm | M=4096,N=4096,K=4096 | kernel-set | fp16 | 24549.4 (24546.3) | 5.6 TFLOP/s (2%) | - | ok |
| fp8_gemm | M=8192,N=8192,K=8192 | torch-scaled-mm | fp16 | - | - | 3.75e-02 | incorrect (rel_err=3.75e-02>tol=2.0e-02) |
| fp8_gemm | M=8192,N=8192,K=8192 | vllm-cutlass-fp8 | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fp8_gemm | M=8192,N=8192,K=8192 | deepgemm | fp16 | - | - | - | skip: needs sm90 (device is sm89) |
| fp8_gemm | M=8192,N=8192,K=8192 | kernel-set | fp16 | 244717.6 (242552.8) | 4.5 TFLOP/s (2%) | - | ok |

**kernel-set vs best-SOTA**:

- `M=4096,N=4096,K=4096`: no SOTA provider ran (all skip/fail)
- `M=8192,N=8192,K=8192`: no SOTA provider ran (all skip/fail)

## rmsnorm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| rmsnorm | rows=4096,hidden=4096 | kernel-set | fp16 | 314.4 (310.3) | 213.5 GB/s (71%) | 2.91e-04 | ok |
| rmsnorm | rows=4096,hidden=4096 | flashinfer-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| rmsnorm | rows=4096,hidden=4096 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=4096,hidden=4096 | liger-norm | fp16 | 295.9 (290.8) | 226.8 GB/s (76%) | 5.94e-04 | ok |
| rmsnorm | rows=8192,hidden=8192 | kernel-set | fp16 | 1175.0 (1163.3) | 228.4 GB/s (76%) | 2.40e-04 | ok |
| rmsnorm | rows=8192,hidden=8192 | flashinfer-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| rmsnorm | rows=8192,hidden=8192 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=8192,hidden=8192 | liger-norm | fp16 | 1156.1 (1140.7) | 232.2 GB/s (77%) | 4.97e-04 | ok |
| rmsnorm | rows=1,hidden=4096 | kernel-set | fp16 | 22.5 (20.5) | 0.7 GB/s (0%) | 2.74e-04 | ok |
| rmsnorm | rows=1,hidden=4096 | flashinfer-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| rmsnorm | rows=1,hidden=4096 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| rmsnorm | rows=1,hidden=4096 | liger-norm | fp16 | 11.3 (9.2) | 1.5 GB/s (0%) | 4.86e-04 | ok |

**kernel-set vs best-SOTA**:

- `rows=4096,hidden=4096`: kernel-set 314.4us vs best `liger-norm` 295.9us => **0.94x** (slower)
- `rows=8192,hidden=8192`: kernel-set 1175.0us vs best `liger-norm` 1156.1us => **0.98x** (slower)
- `rows=1,hidden=4096`: kernel-set 22.5us vs best `liger-norm` 11.3us => **0.50x** (slower)

## fused_add_rmsnorm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fused_add_rmsnorm | rows=4096,hidden=4096 | kernel-set | fp16 | 614.4 (607.2) | 218.5 GB/s (73%) | 5.80e-04 | ok |
| fused_add_rmsnorm | rows=4096,hidden=4096 | flashinfer-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| fused_add_rmsnorm | rows=4096,hidden=4096 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fused_add_rmsnorm | rows=8192,hidden=8192 | kernel-set | fp16 | 2327.6 (2313.2) | 230.7 GB/s (77%) | 4.77e-04 | ok |
| fused_add_rmsnorm | rows=8192,hidden=8192 | flashinfer-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| fused_add_rmsnorm | rows=8192,hidden=8192 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| fused_add_rmsnorm | rows=1,hidden=4096 | kernel-set | fp16 | 18.4 (16.4) | 1.8 GB/s (1%) | 4.05e-04 | ok |
| fused_add_rmsnorm | rows=1,hidden=4096 | flashinfer-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| fused_add_rmsnorm | rows=1,hidden=4096 | vllm-norm | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |

**kernel-set vs best-SOTA**:

- `rows=4096,hidden=4096`: no SOTA provider ran (all skip/fail)
- `rows=8192,hidden=8192`: no SOTA provider ran (all skip/fail)
- `rows=1,hidden=4096`: no SOTA provider ran (all skip/fail)

## rope

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| rope | tokens=4096,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 603.1 (591.9) | 139.1 GB/s (46%) | 2.64e-04 | ok |
| rope | tokens=4096,qh=32,kvh=8,hd=128 | flashinfer-rope | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| rope | tokens=4096,qh=32,kvh=8,hd=128 | liger-rope | fp16 | 386.0 (376.8) | 217.3 GB/s (72%) | 6.73e-04 | ok |
| rope | tokens=1,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 13.3 (12.3) | 1.5 GB/s (1%) | 2.23e-04 | ok |
| rope | tokens=1,qh=32,kvh=8,hd=128 | flashinfer-rope | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| rope | tokens=1,qh=32,kvh=8,hd=128 | liger-rope | fp16 | 6.1 (5.1) | 3.3 GB/s (1%) | 3.58e-04 | ok |

**kernel-set vs best-SOTA**:

- `tokens=4096,qh=32,kvh=8,hd=128`: kernel-set 603.1us vs best `liger-rope` 386.0us => **0.64x** (slower)
- `tokens=1,qh=32,kvh=8,hd=128`: kernel-set 13.3us vs best `liger-rope` 6.1us => **0.46x** (slower)

## swiglu

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| swiglu | rows=4096,inter=14336 | kernel-set | fp16 | 1541.1 (1520.6) | 228.6 GB/s (76%) | 2.67e-04 | ok |
| swiglu | rows=4096,inter=14336 | flashinfer-act | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| swiglu | rows=4096,inter=14336 | vllm-act | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |
| swiglu | rows=1,inter=14336 | kernel-set | fp16 | 12.3 (10.3) | 7.0 GB/s (2%) | 2.90e-04 | ok |
| swiglu | rows=1,inter=14336 | flashinfer-act | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| swiglu | rows=1,inter=14336 | vllm-act | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |

**kernel-set vs best-SOTA**:

- `rows=4096,inter=14336`: no SOTA provider ran (all skip/fail)
- `rows=1,inter=14336`: no SOTA provider ran (all skip/fail)

## moe_grouped_gemm

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| moe_grouped_gemm | tokens=4096,h=4096,inter=14336,E=8,k=2 | kernel-set | fp16 | 902056.9 (895180.8) | 1.1 TFLOP/s (1%) | 6.42e-04 | ok |

**kernel-set vs best-SOTA**:

- `tokens=4096,h=4096,inter=14336,E=8,k=2`: no SOTA provider ran (all skip/fail)

## fused_moe

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fused_moe | tokens=4096,h=4096,inter=14336,E=8,k=2 | vllm-fused-moe | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'vllm' |

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
| cross_entropy | tokens=4096,vocab=32000 | kernel-set | fp16 | 2363.4 (2329.6) | 221.8 GB/s (74%) | 1.33e-07 | ok |
| cross_entropy | tokens=4096,vocab=32000 | liger-ce | fp16 | 1384.4 (1372.2) | 378.7 GB/s (126%) | 2.73e-04 | ok |
| cross_entropy | tokens=4096,vocab=32000 | torch-ce | fp16 | 2272.8 (2262.0) | 230.7 GB/s (77%) | 2.73e-04 | ok |
| cross_entropy | tokens=8192,vocab=128256 | kernel-set | fp16 | 26903.6 (26814.5) | 156.2 GB/s (52%) | 1.21e-07 | ok |
| cross_entropy | tokens=8192,vocab=128256 | liger-ce | fp16 | 8450.6 (8420.4) | 497.3 GB/s (166%) | 2.47e-04 | ok |
| cross_entropy | tokens=8192,vocab=128256 | torch-ce | fp16 | 18285.1 (18247.7) | 229.8 GB/s (77%) | 2.47e-04 | ok |

**kernel-set vs best-SOTA**:

- `tokens=4096,vocab=32000`: kernel-set 2363.4us vs best `liger-ce` 1384.4us => **0.59x** (slower)
- `tokens=8192,vocab=128256`: kernel-set 26903.6us vs best `liger-ce` 8450.6us => **0.31x** (slower)

## fused_linear_ce

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| fused_linear_ce | tokens=4096,vocab=32000 | cut-cross-entropy | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'cut_cross_entropy' |
| fused_linear_ce | tokens=8192,vocab=128256 | cut-cross-entropy | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'cut_cross_entropy' |

**kernel-set vs best-SOTA**:

- `tokens=4096,vocab=32000`: kernel-set did not run
- `tokens=8192,vocab=128256`: kernel-set did not run

_Legend: %pk = % of dense peak. `skip: needs smXX` = arch-gated (provider needs a newer GPU; not imported here). `import-fail` = library not installed. `incorrect` = disagrees with the fp32 reference._

