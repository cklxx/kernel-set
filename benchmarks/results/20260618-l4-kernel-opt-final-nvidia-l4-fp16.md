# kernel-set benchmark — NVIDIA L4

- **GPU**: NVIDIA L4 (sm_89, CC 8.9, 58 SMs, 22.0 GB) | dense peak BW ~300 GB/s | dense fp16/bf16 TC ~121 TFLOP/s
- **detected via**: kernel_set
- **clocks**: SM 210 MHz, mem 405 MHz | clocks UNLOCKED (boost/throttle not controlled) | throttle: 0x0000000000000001
- **driver**: 580.82.07 | CUDA 12.8 | cuDNN 91900 | power cap 72.00 W | ECC Enabled
- **dtype**: fp16 | TF32 matmul=True cudnn=True (fp32_precision=high)
- **timing**: L2-flush=on | method=cuda-events (flushed) | target-ms=100.0 | iters=auto | warmup=10 | L2-flush-buffer=96 MB
- **launch overhead included**: yes (single launch / event pair)
- **kernel-set**: 0.2.1 (backend cuda)
- **torch**: 2.11.0+cu128
- **nvcc**: Cuda compilation tools, release 12.8, V12.8.93
- **timestamp**: 20260618-l4-kernel-opt-final
- **host**: Linux-6.6.122+-x86_64-with-glibc2.35

**fast_1 = -%** of comparable ops are BOTH correct AND >= baseline speed (fast_0=-%, fast_2=-%). correct=14 · incorrect=0 · error=0 · skip=0 · no-ref=0. mean speedup over correct-only = -x.

Latency cells show **median (min)** microseconds over the auto-calibrated timed iterations (CUDA events, L2-flushed unless noted). `GB/s`/`TFLOP/s` cells append the achieved value's **% of dense peak** for this SKU+dtype. `rel_err` is gated at the dtype tolerance BEFORE speed is reported; a kernel that fails is marked **INCORRECT**. `speedup` is best_baseline_us / ks_us.

| op | shape | dtype | ks us (min) | ref us (min) | GB/s (%pk) | TFLOP/s (%pk) | rel_err | spd | base | iters | m | notes |
|---|---|---|--:|--:|--:|--:|--:|--:|---|--:|:-:|---|
| quantize_fp8 | rows=4096,cols=4096 | fp16 | 360.4 (340.0) | - | 139.6 (47%) | - | 2.65e-02 | - | - | 377 | E | round-trip dequant(quant(x))~x; rel-L2 gate @0.10 |
| dequantize_fp8 | rows=4096,cols=4096 | fp16 | 252.9 (250.9) | - | 199.0 (66%) | - | 0.00e+00 | - | - | 579 | E | ref = float(fp8)*scale |
| quantize_fp8_group | rows=4096,cols=4096,g=128 | fp16 | 474.1 (416.8) | - | 106.2 (35%) | - | 2.57e-02 | - | - | 278 | E | 1x128 fp8 activation quant; scale=group_amax/448; rel-L2 gate @0.10 |
| quantize_int8 | rows=4096,cols=4096 | fp16 | 259.1 (254.0) | - | 194.3 (65%) | - | 8.69e-03 | - | - | 726 | E | ref scale=amax/127; round-trip rel-L2 gate @0.03 |
| dequantize_int8 | rows=4096,cols=4096 | fp16 | 271.4 (268.3) | - | 185.5 (62%) | - | 0.00e+00 | - | - | 476 | E | ref = int8*scale (per-token) |
| quantize_fp8 | rows=8192,cols=8192 | fp16 | 1397.8 (1391.6) | - | 144.0 (48%) | - | 2.65e-02 | - | - | 75 | E | round-trip dequant(quant(x))~x; rel-L2 gate @0.10 |
| dequantize_fp8 | rows=8192,cols=8192 | fp16 | 872.4 (869.4) | - | 230.8 (77%) | - | 0.00e+00 | - | - | 119 | E | ref = float(fp8)*scale |
| quantize_fp8_group | rows=8192,cols=8192,g=128 | fp16 | 2128.9 (1919.0) | - | 94.6 (32%) | - | 2.57e-02 | - | - | 60 | E | 1x128 fp8 activation quant; scale=group_amax/448; rel-L2 gate @0.10 |
| quantize_int8 | rows=8192,cols=8192 | fp16 | 909.3 (906.2) | - | 221.4 (74%) | - | 9.06e-03 | - | - | 117 | E | ref scale=amax/127; round-trip rel-L2 gate @0.03 |
| dequantize_int8 | rows=8192,cols=8192 | fp16 | 904.2 (900.1) | - | 222.7 (74%) | - | 0.00e+00 | - | - | 115 | E | ref = int8*scale (per-token) |
| dequantize_int4 | K=4096,N=4096,g=128 | fp16 | 250.9 (234.5) | - | 167.2 (56%) | - | 0.00e+00 | - | - | 478 | E | exact AWQ/GPTQ unpack ref |
| dequantize_int4 | K=4096,N=14336,g=128 | fp16 | 865.3 (704.5) | - | 169.7 (57%) | - | 0.00e+00 | - | - | 146 | E | exact AWQ/GPTQ unpack ref |
| reshape_and_cache | tokens=4096,kvh=8,hd=128,blk=16 | fp16 | 188.4 (180.2) | - | 178.1 (59%) | - | 0.00e+00 | - | - | 1000 | E | scatter-then-gather-by-slot ref |
| reshape_and_cache | tokens=1,kvh=8,hd=128,blk=16 | fp16 | 7.2 (4.1) | - | 1.1 (0%) | - | 0.00e+00 | - | - | 1000 | E | scatter-then-gather-by-slot ref |

_Legend: m = timing method (E=cuda-events flushed, C=cudagraph replay). %pk = % of dense peak. spd = speedup vs the fastest baseline named in `base`._

