# kernel-set benchmark — NVIDIA RTX PRO 6000 Blackwell Server Edition

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, CC 12.0, 188 SMs, 95.0 GB)
- **detected via**: kernel_set
- **clocks**: SM 180 MHz, mem 12481 MHz | clocks UNLOCKED (boost/throttle not controlled) | throttle: 0x0000000000000000
- **driver**: 580.82.07 | CUDA 12.8 | cuDNN 91900 | power cap 600.00 W | ECC Enabled
- **dtype**: bf16 | TF32 matmul=True cudnn=True (fp32_precision=high)
- **timing**: L2-flush=on | method=cuda-events (flushed) | target-ms=40.0 | iters=auto | warmup=10 | L2-flush-buffer=256 MB
- **launch overhead included**: yes (single launch / event pair)
- **kernel-set**: 0.1.0 (backend cuda)
- **torch**: 2.11.0+cu128
- **nvcc**: Cuda compilation tools, release 12.8, V12.8.93
- **harness commit**: 9f6417f
- **timestamp**: 2026-06-05T07:01:25
- **host**: Linux-6.6.122+-x86_64-with-glibc2.35

**fast_1 = 67%** of comparable ops are BOTH correct AND >= baseline speed (fast_0=100%, fast_2=20%). correct=100 · incorrect=0 · error=0 · skip=0 · no-ref=5. mean speedup over correct-only = 1.50x.

Latency cells show **median (min)** microseconds over the auto-calibrated timed iterations (CUDA events, L2-flushed unless noted). `GB/s`/`TFLOP/s` cells append the achieved value's **% of dense peak** for this SKU+dtype. `rel_err` is gated at the dtype tolerance BEFORE speed is reported; a kernel that fails is marked **INCORRECT**. `speedup` is best_baseline_us / ks_us.

| op | shape | dtype | ks us (min) | ref us (min) | GB/s (%pk) | TFLOP/s (%pk) | rel_err | spd | base | iters | m | notes |
|---|---|---|--:|--:|--:|--:|--:|--:|---|--:|:-:|---|
| rmsnorm | rows=4096,hidden=4096 | bf16 | 55.3 (53.1) | 241.8 (235.6) | 1212.9 | - | 4.17e-03 | 4.37x | eager | 968 | E |  |
| rmsnorm | rows=8192,hidden=8192 | bf16 | 194.5 (190.4) | 1702.0 (1696.8) | 1379.9 | - | 4.00e-03 | 8.75x | eager | 213 | E |  |
| rmsnorm | rows=2048,hidden=3584 | bf16 | 32.8 (28.6) | 121.6 (118.0) | 896.0 | - | 4.41e-03 | 3.71x | eager | 1000 | E |  |
| rmsnorm | rows=1,hidden=4096 | bf16 | 10.3 (7.3) | 28.7 (26.5) | 1.6 | - | 5.15e-03 | 2.79x | eager | 1000 | E |  |
| layernorm | rows=4096,hidden=4096 | bf16 | 57.3 (55.1) | 63.4 (58.1) | 1170.3 | - | 1.02e-03 | 1.11x | eager | 1000 | E |  |
| layernorm | rows=8192,hidden=8192 | bf16 | 202.7 (198.7) | 196.6 (193.3) | 1324.4 | - | 2.08e-03 | 0.97x | eager | 214 | E |  |
| layernorm | rows=2048,hidden=3584 | bf16 | 34.8 (29.9) | 36.8 (34.6) | 844.8 | - | 1.07e-03 | 1.06x | eager | 1000 | E |  |
| layernorm | rows=1,hidden=4096 | bf16 | 10.3 (8.2) | 18.3 (16.1) | 1.6 | - | 0.00e+00 | 1.78x | eager | 1000 | E |  |
| swiglu | rows=4096,inter=14336 | bf16 | 245.8 (242.7) | 408.2 (405.5) | 1433.6 | - | 4.96e-04 | 1.66x | eager | 167 | E |  |
| swiglu | rows=2048,inter=28672 | bf16 | 245.8 (243.6) | 408.3 (405.5) | 1433.6 | - | 4.56e-04 | 1.66x | eager | 167 | E |  |
| swiglu | rows=1,inter=14336 | bf16 | 6.1 (4.8) | 12.3 (10.1) | 14.0 | - | 0.00e+00 | 2.01x | eager | 1000 | E |  |
| rope | tokens=4096,qh=32,kvh=8,hd=128 | bf16 | 90.0 (87.2) | - | 931.9 | - | 0.00e+00 | - | - | 718 | E | neox/rotate_half |
| rope | tokens=1,qh=32,kvh=8,hd=128 | bf16 | 6.9 (4.1) | - | 3.0 | - | 0.00e+00 | - | - | 1000 | E | neox/rotate_half |
| attention_prefill | b=1,seq=2048,qh=32,kvh=8,hd=128 | bf16 | 15925.1 (15897.0) | 241.0 (237.6) | - | 2.2 | 5.76e-04 | 0.02x | sdpa(flash/efficient) | 20 | E |  |
| attention_prefill | b=4,seq=1024,qh=32,kvh=8,hd=128 | bf16 | 16210.3 (16201.3) | 272.4 (268.6) | - | 2.1 | 1.10e-03 | 0.02x | sdpa(flash/efficient) | 20 | E |  |
| attention_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | bf16 | 1780.7 (1776.6) | - | 301.5 | - | - | - | - | 22 | E | memory-bound (KV read) |
| attention_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | bf16 | 3350.2 (3342.7) | - | 320.5 | - | - | - | - | 20 | E | memory-bound (KV read) |
| gemm_bf16 | M=4096,N=4096,K=4096 | bf16 | 6060.0 (6046.7) | 387.0 (383.9) | - | 22.7 | 5.59e-03 | 0.06x | cublas(a@b) | 20 | E | cublas=387.0us compile=492.3us |
| gemm_bf16 | M=8192,N=8192,K=8192 | bf16 | 45931.4 (45899.8) | 2681.8 (2658.3) | - | 23.9 | 3.95e-03 | 0.06x | cublas(a@b) | 20 | E | cublas=2681.8us compile=3067.6us |
| gemm_bf16 | M=4096,N=14336,K=4096 | bf16 | 20112.7 (20107.3) | 1177.6 (1173.5) | - | 23.9 | 5.46e-03 | 0.06x | cublas(a@b) | 20 | E | cublas=1177.6us compile=1394.9us |
| gemm_bf16 | M=2048,N=4096,K=14336 | bf16 | 11795.8 (11784.3) | 647.2 (645.1) | - | 20.4 | 6.54e-03 | 0.05x | cublas(a@b) | 20 | E | cublas=647.2us compile=914.1us |
| w8a8 | M=4096,N=4096,K=4096 | bf16 | 5718.0 (5713.9) | - | - | 24.0 | 0.00e+00 | - | - | 20 | E | int8xint8->acc int32, per-token/per-channel dequant |
| w8a8 | M=8192,N=8192,K=8192 | bf16 | 44028.9 (44016.6) | - | - | 25.0 | 0.00e+00 | - | - | 20 | E | int8xint8->acc int32, per-token/per-channel dequant |
| w8a8 | M=4096,N=14336,K=4096 | bf16 | 19321.8 (19315.7) | - | - | 24.9 | 0.00e+00 | - | - | 20 | E | int8xint8->acc int32, per-token/per-channel dequant |
| w4a16 | M=4096,N=4096,K=4096,g=128 | bf16 | 7121.9 (7102.5) | - | - | 19.3 | - | - | - | 20 | E | no portable torch int4 ref; throughput only (uncorrectness-gated) |
| w4a16 | M=8192,N=8192,K=8192,g=128 | bf16 | 53848.1 (53794.2) | - | - | 20.4 | - | - | - | 20 | E | no portable torch int4 ref; throughput only (uncorrectness-gated) |
| w4a16 | M=4096,N=14336,K=4096,g=128 | bf16 | 23546.9 (23542.1) | - | - | 20.4 | - | - | - | 20 | E | no portable torch int4 ref; throughput only (uncorrectness-gated) |
| moe_gate | tokens=4096,E=8,k=2 | bf16 | 10.4 (9.5) | - | - | - | 1.24e-07 | - | - | 1000 | E | softmax top-k gating |
| moe_grouped_gemm | tokens=4096,h=4096,E=8,k=2 | bf16 | 150482.4 (150474.7) | - | - | 6.4 | 5.62e-03 | - | - | 20 | E | grouped GEMM over experts |
| moe_gate | tokens=2048,E=64,k=6 | bf16 | 16.4 (13.6) | - | - | - | 1.61e-07 | - | - | 1000 | E | softmax top-k gating |
| moe_grouped_gemm | tokens=2048,h=2048,E=64,k=6 | bf16 | 11183.7 (11176.8) | - | - | 6.3 | 3.82e-03 | - | - | 20 | E | grouped GEMM over experts |
| sampling | seqs=256,vocab=32000 | bf16 | 42.9 (39.9) | 36.9 (32.7) | 381.5 | - | 0.00e+00 | 0.86x | eager | 1000 | E | argmax (greedy); rel_err = mismatch fraction |
| sampling | seqs=64,vocab=128256 | bf16 | 79.1 (75.0) | 55.3 (51.4) | 207.5 | - | 0.00e+00 | 0.70x | eager | 1000 | E | argmax (greedy); rel_err = mismatch fraction |
| cross_entropy | tokens=4096,vocab=32000 | bf16 | 464.9 (459.8) | 401.5 (396.6) | 1127.8 | - | 6.54e-08 | 0.86x | eager(fwd-only) | 90 | E | fused fwd+bwd; rel_err on forward loss |
| cross_entropy | tokens=8192,vocab=128256 | bf16 | 4362.9 (4354.0) | 3422.6 (3406.6) | 963.3 | - | 1.17e-07 | 0.78x | eager(fwd-only) | 20 | E | fused fwd+bwd; rel_err on forward loss |
| adamw | n=16777216 | bf16 | 255.7 (253.2) | - | 1443.4 | - | 6.11e-10 | - | - | 158 | E | fused AdamW step; memory-bound |
| adamw | n=67108864 | bf16 | 1007.6 (1004.8) | - | 1465.2 | - | 6.37e-10 | - | - | 39 | E | fused AdamW step; memory-bound |
| rmsnorm_bwd | rows=4096,hidden=4096 | bf16 | 125.8 (122.8) | - | 799.9 | - | 2.10e-03 | - | - | 372 | E | autograd ref (grad_input + grad_weight_fp32) |
| rmsnorm_bwd | rows=8192,hidden=8192 | bf16 | 310.7 (307.6) | - | 1296.1 | - | 2.00e-03 | - | - | 132 | E | autograd ref (grad_input + grad_weight_fp32) |
| rmsnorm_bwd | rows=2048,hidden=3584 | bf16 | 81.9 (73.7) | - | 537.8 | - | 2.17e-03 | - | - | 581 | E | autograd ref (grad_input + grad_weight_fp32) |
| rmsnorm_bwd | rows=1,hidden=4096 | bf16 | 14.3 (11.5) | - | 1.7 | - | 2.23e-03 | - | - | 1000 | E | autograd ref (grad_input + grad_weight_fp32) |
| layernorm_bwd | rows=4096,hidden=4096 | bf16 | 110.7 (107.8) | - | 909.4 | - | 2.25e-03 | - | - | 457 | E | autograd ref (grad_input + grad_weight/bias_fp32) |
| layernorm_bwd | rows=8192,hidden=8192 | bf16 | 356.2 (350.2) | - | 1130.4 | - | 1.98e-03 | - | - | 113 | E | autograd ref (grad_input + grad_weight/bias_fp32) |
| layernorm_bwd | rows=2048,hidden=3584 | bf16 | 85.3 (79.9) | - | 516.4 | - | 2.24e-03 | - | - | 571 | E | autograd ref (grad_input + grad_weight/bias_fp32) |
| layernorm_bwd | rows=1,hidden=4096 | bf16 | 22.5 (18.4) | - | 1.1 | - | 2.13e-03 | - | - | 1000 | E | autograd ref (grad_input + grad_weight/bias_fp32) |
| geglu | rows=4096,inter=14336 | bf16 | 245.8 (242.7) | 410.5 (407.5) | 1433.6 | - | 0.00e+00 | 1.67x | eager | 168 | E |  |
| geglu | rows=2048,inter=28672 | bf16 | 245.7 (242.6) | 410.3 (407.4) | 1433.8 | - | 0.00e+00 | 1.67x | eager | 167 | E |  |
| geglu | rows=1,inter=14336 | bf16 | 8.2 (5.4) | 16.4 (12.3) | 10.5 | - | 0.00e+00 | 2.00x | eager | 1000 | E |  |
| swiglu_bwd | rows=4096,inter=14336 | bf16 | 403.5 (400.4) | - | 1455.4 | - | 3.27e-03 | - | - | 100 | E | autograd ref (grad_gate + grad_up) |
| swiglu_bwd | rows=2048,inter=28672 | bf16 | 403.4 (399.4) | - | 1455.5 | - | 3.63e-03 | - | - | 100 | E | autograd ref (grad_gate + grad_up) |
| swiglu_bwd | rows=1,inter=14336 | bf16 | 8.2 (5.3) | - | 17.5 | - | 2.37e-03 | - | - | 1000 | E | autograd ref (grad_gate + grad_up) |
| rope_bwd | tokens=4096,qh=32,kvh=8,hd=128 | bf16 | 114.8 (110.8) | - | 731.0 | - | 0.00e+00 | - | - | 548 | E | conjugate-rotation ref (neox) |
| rope_bwd | tokens=1,qh=32,kvh=8,hd=128 | bf16 | 12.3 (10.1) | - | 1.7 | - | 0.00e+00 | - | - | 1000 | E | conjugate-rotation ref (neox) |
| embedding | tokens=4096,vocab=32000,d=4096 | bf16 | 53.3 (49.8) | 67.6 (63.5) | 1258.8 | - | 0.00e+00 | 1.27x | eager(index_select) | 1000 | E | index_select ref |
| embedding | tokens=8192,vocab=128256,d=4096 | bf16 | 102.4 (100.3) | 114.6 (107.8) | 1310.3 | - | 0.00e+00 | 1.12x | eager(index_select) | 668 | E | index_select ref |
| embedding | tokens=1,vocab=128256,d=4096 | bf16 | 8.2 (5.4) | 16.4 (12.3) | 2.0 | - | 0.00e+00 | 2.00x | eager(index_select) | 1000 | E | index_select ref |
| embedding_bwd | tokens=4096,vocab=32000,d=4096 | bf16 | 469.0 (466.0) | - | 357.7 | - | 0.00e+00 | - | - | 89 | E | scatter-add (index_add) ref; grad_table fp32 |
| embedding_bwd | tokens=8192,vocab=128256,d=4096 | bf16 | 1625.1 (1620.9) | - | 206.5 | - | 0.00e+00 | - | - | 24 | E | scatter-add (index_add) ref; grad_table fp32 |
| embedding_bwd | tokens=1,vocab=128256,d=4096 | bf16 | 1382.7 (1380.6) | - | 0.0 | - | 0.00e+00 | - | - | 28 | E | scatter-add (index_add) ref; grad_table fp32 |
| ew_add | n=16777216 | bf16 | 75.8 (73.7) | 84.0 (79.8) | 1327.3 | - | 0.00e+00 | 1.11x | eager | 1000 | E |  |
| ew_add | n=67108864 | bf16 | 277.2 (273.7) | 283.4 (280.5) | 1452.5 | - | 0.00e+00 | 1.02x | eager | 146 | E |  |
| ew_add | n=4096 | bf16 | 6.2 (5.1) | 12.3 (9.1) | 4.0 | - | 0.00e+00 | 1.98x | eager | 1000 | E |  |
| ew_mul | n=16777216 | bf16 | 76.5 (72.9) | 84.0 (79.9) | 1315.7 | - | 0.00e+00 | 1.10x | eager | 1000 | E |  |
| ew_mul | n=67108864 | bf16 | 277.3 (273.7) | 283.4 (280.4) | 1452.1 | - | 0.00e+00 | 1.02x | eager | 146 | E |  |
| ew_mul | n=4096 | bf16 | 6.2 (5.0) | 12.3 (8.2) | 3.9 | - | 0.00e+00 | 1.97x | eager | 1000 | E |  |
| ew_add_residual | n=16777216 | bf16 | 90.2 (87.2) | - | 1115.9 | - | 0.00e+00 | - | - | 1000 | E | in-place residual += x |
| ew_add_residual | n=67108864 | bf16 | 458.9 (456.0) | - | 877.5 | - | 0.00e+00 | - | - | 87 | E | in-place residual += x |
| ew_add_residual | n=4096 | bf16 | 10.3 (7.5) | - | 2.4 | - | 0.00e+00 | - | - | 1000 | E | in-place residual += x |
| ew_scale | n=16777216 | bf16 | 49.2 (46.9) | 55.3 (53.1) | 1364.4 | - | 0.00e+00 | 1.12x | eager | 1000 | E |  |
| ew_scale | n=67108864 | bf16 | 190.4 (186.3) | 196.0 (191.8) | 1409.9 | - | 0.00e+00 | 1.03x | eager | 216 | E |  |
| ew_scale | n=4096 | bf16 | 6.2 (4.1) | 12.3 (8.2) | 2.7 | - | 0.00e+00 | 1.99x | eager | 1000 | E |  |
| ew_cast | n=16777216 | bf16 | 94.2 (91.3) | 96.4 (93.4) | 1068.9 | - | 0.00e+00 | 1.02x | eager(.to) | 820 | E | bf16->fp32 cast |
| ew_cast | n=67108864 | bf16 | 286.7 (284.5) | 307.3 (304.4) | 1404.5 | - | 0.00e+00 | 1.07x | eager(.to) | 143 | E | bf16->fp32 cast |
| ew_cast | n=4096 | bf16 | 6.2 (4.1) | 12.3 (7.3) | 4.0 | - | 0.00e+00 | 1.99x | eager(.to) | 1000 | E | bf16->fp32 cast |
| ew_axpby | n=16777216 | bf16 | 75.8 (73.7) | 123.3 (120.1) | 1327.3 | - | 6.25e-03 | 1.63x | eager | 1000 | E | out = a*alpha + b*beta |
| ew_axpby | n=67108864 | bf16 | 277.5 (273.7) | 651.3 (647.1) | 1451.0 | - | 6.17e-03 | 2.35x | eager | 146 | E | out = a*alpha + b*beta |
| ew_axpby | n=4096 | bf16 | 6.2 (4.1) | 17.8 (13.7) | 4.0 | - | 5.05e-03 | 2.87x | eager | 1000 | E | out = a*alpha + b*beta |
| quantize_fp8 | rows=4096,cols=4096 | bf16 | 1037.3 (1022.9) | - | 48.5 | - | 2.65e-02 | - | - | 41 | E | round-trip dequant(quant(x))~x; rel-L2 gate @0.10 |
| dequantize_fp8 | rows=4096,cols=4096 | bf16 | 1037.3 (986.1) | - | 48.5 | - | 0.00e+00 | - | - | 43 | E | ref = float(fp8)*scale |
| quantize_int8 | rows=4096,cols=4096 | bf16 | 49.2 (46.2) | - | 1023.3 | - | 8.68e-03 | - | - | 1000 | E | ref scale=amax/127; round-trip rel-L2 gate @0.03 |
| dequantize_int8 | rows=4096,cols=4096 | bf16 | 90.1 (87.0) | - | 558.5 | - | 0.00e+00 | - | - | 659 | E | ref = int8*scale (per-token) |
| quantize_fp8 | rows=8192,cols=8192 | bf16 | 1181.7 (988.1) | - | 170.4 | - | 2.65e-02 | - | - | 33 | E | round-trip dequant(quant(x))~x; rel-L2 gate @0.10 |
| dequantize_fp8 | rows=8192,cols=8192 | bf16 | 1123.3 (887.8) | - | 179.2 | - | 0.00e+00 | - | - | 41 | E | ref = float(fp8)*scale |
| quantize_int8 | rows=8192,cols=8192 | bf16 | 166.1 (163.1) | - | 1211.8 | - | 9.05e-03 | - | - | 272 | E | ref scale=amax/127; round-trip rel-L2 gate @0.03 |
| dequantize_int8 | rows=8192,cols=8192 | bf16 | 202.8 (199.8) | - | 992.8 | - | 0.00e+00 | - | - | 213 | E | ref = int8*scale (per-token) |
| dequantize_int4 | K=4096,N=4096,g=128 | bf16 | 92.1 (89.3) | - | 455.6 | - | 0.00e+00 | - | - | 516 | E | exact AWQ/GPTQ unpack ref |
| dequantize_int4 | K=4096,N=14336,g=128 | bf16 | 197.9 (195.7) | - | 741.7 | - | 0.00e+00 | - | - | 215 | E | exact AWQ/GPTQ unpack ref |
| moe_permute | tokens=4096,h=4096,E=8,k=2 | bf16 | 79.2 (75.0) | - | 1694.0 | - | 0.00e+00 | - | - | 1000 | E | gather-by-sorted_token_ids ref |
| moe_permute | tokens=2048,h=2048,E=64,k=6 | bf16 | 45.1 (42.1) | - | 2234.2 | - | 0.00e+00 | - | - | 1000 | E | gather-by-sorted_token_ids ref |
| moe_unpermute | tokens=4096,h=4096,E=8,k=2 | bf16 | 117.5 (113.8) | - | 856.7 | - | 1.76e-03 | - | - | 569 | E | weighted scatter-add ref |
| moe_unpermute | tokens=2048,h=2048,E=64,k=6 | bf16 | 81.8 (78.0) | - | 718.2 | - | 2.86e-03 | - | - | 817 | E | weighted scatter-add ref |
| reshape_and_cache | tokens=4096,kvh=8,hd=128,blk=16 | bf16 | 34.9 (31.9) | - | 961.1 | - | 0.00e+00 | - | - | 1000 | E | scatter-then-gather-by-slot ref |
| reshape_and_cache | tokens=1,kvh=8,hd=128,blk=16 | bf16 | 8.2 (4.1) | - | 1.0 | - | 0.00e+00 | - | - | 1000 | E | scatter-then-gather-by-slot ref |
| flash_attn_bwd | b=1,seq=2048,qh=32,kvh=8,hd=128 | bf16 | 62063.6 (61871.9) | - | - | 1.4 | 3.31e-03 | - | - | 20 | E | autograd-through-SDPA ref (grad_q/k/v) |
| flash_attn_bwd | b=4,seq=1024,qh=32,kvh=8,hd=128 | bf16 | 61952.6 (61833.3) | - | - | 1.4 | 3.06e-03 | - | - | 20 | E | autograd-through-SDPA ref (grad_q/k/v) |
| fused_linear_ce | tokens=4096,d=4096,vocab=32000 | bf16 | 555413.3 (555391.8) | - | - | 5.8 | 2.14e-03 | - | - | 20 | E | F.cross_entropy(h@W^T)+autograd; per-token loss, sum-reduction grads |
| fused_linear_ce | tokens=2048,d=4096,vocab=128256 | bf16 | 1234199.7 (1234112.8) | - | - | 5.2 | 2.23e-03 | - | - | 20 | E | F.cross_entropy(h@W^T)+autograd; per-token loss, sum-reduction grads |
| sgd_momentum | n=16777216 | bf16 | 149.6 (146.7) | - | 1570.1 | - | 1.02e-05 | - | - | 327 | E | manual SGD+momentum step ref |
| sgd_momentum | n=67108864 | bf16 | 1002.5 (998.7) | - | 937.2 | - | 1.04e-05 | - | - | 40 | E | manual SGD+momentum step ref |
| global_grad_norm | n=16777216 | bf16 | 53.3 (50.2) | - | 629.8 | - | 1.49e-06 | - | - | 1000 | E | sqrt(sum||g||^2) vs torch vector_norm |
| global_grad_norm | n=67108864 | bf16 | 142.2 (138.5) | - | 943.6 | - | 5.37e-07 | - | - | 491 | E | sqrt(sum||g||^2) vs torch vector_norm |
| argmax | seqs=256,vocab=32000 | bf16 | 42.3 (38.2) | 36.9 (32.7) | 387.6 | - | 0.00e+00 | 0.87x | eager | 1000 | E | torch.argmax ref; rel_err = mismatch fraction |
| argmax | seqs=64,vocab=128256 | bf16 | 79.1 (76.1) | 55.3 (52.4) | 207.5 | - | 0.00e+00 | 0.70x | eager | 1000 | E | torch.argmax ref; rel_err = mismatch fraction |
| log_softmax | rows=256,vocab=32000 | bf16 | 58.7 (54.5) | 51.2 (47.1) | 558.6 | - | 0.00e+00 | 0.87x | eager | 1000 | E | torch.log_softmax(fp32) ref |
| log_softmax | rows=64,vocab=128256 | bf16 | 136.2 (133.1) | 53.2 (49.1) | 241.1 | - | 0.00e+00 | 0.39x | eager | 422 | E | torch.log_softmax(fp32) ref |

_Legend: m = timing method (E=cuda-events flushed, C=cudagraph replay). %pk = % of dense peak. spd = speedup vs the fastest baseline named in `base`._

