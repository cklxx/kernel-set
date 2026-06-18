# kernel-set vs SOTA — NVIDIA L4

- **GPU**: NVIDIA L4 (sm_89, CC 8.9, 58 SMs, 22.0 GB) | dense peak BW ~300 GB/s | dense fp16/bf16 TC ~121 TFLOP/s | dense fp8/int8 TC ~242 TFLOP/s
- **detected via**: kernel_set
- **clocks**: SM 2040 MHz, mem 6251 MHz (throttle: 0x0000000000000001)
- **driver**: 580.82.07 | CUDA 12.8 | cuDNN 91900 | power cap 72.00 W | ECC Enabled
- **dtype**: fp16 | TF32 matmul=True (fp32_precision=high)
- **timing**: L2-flush=on | method=cuda-events (flushed) | target-ms=80.0 | iters=auto | warmup=10 | L2-buffer=96 MB
- **kernel-set**: 0.2.1 (backend cuda)
- **torch**: 2.11.0+cu128
- **nvcc**: Cuda compilation tools, release 12.8, V12.8.93
- **timestamp**: 20260618-l4-splitk-attn-sota-mla-decode
- **host**: Linux-6.6.122+-x86_64-with-glibc2.35

**Providers**: ok=2 · skip=2 · import-fail=0 · error=0 · incorrect=0. Correctness is gated against a shared fp32 reference at the dtype tolerance BEFORE speed is reported.

Latency cells show **median (min)** microseconds (CUDA events, L2-flushed). The perf column is GB/s (bandwidth-bound ops) or TFLOP/s (compute-bound), with **% of dense peak** for this SKU+dtype. Arch-gated providers SKIP on unsupported SMs (`needs smXX`) and are never imported there.

## mla_decode

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| mla_decode | seqs=64,ctx=2048,h=128,lora=512,rope=64 | kernel-set | fp16 | 78373.4 (76834.8) | 1.9 GB/s (1%) | 4.24e-04 | ok |
| mla_decode | seqs=64,ctx=2048,h=128,lora=512,rope=64 | kernel-set-splitk | fp16 | 103387.1 (103088.1) | 1.5 GB/s (0%) | 4.24e-04 | ok |
| mla_decode | seqs=64,ctx=2048,h=128,lora=512,rope=64 | flash-mla | fp16 | - | - | - | skip: needs sm90 (device is sm89) |
| mla_decode | seqs=64,ctx=2048,h=128,lora=512,rope=64 | sgl-mla | fp16 | - | - | - | skip: needs sm90 (device is sm89) |

**kernel-set vs best-SOTA**:

- `seqs=64,ctx=2048,h=128,lora=512,rope=64`: kernel-set 78373.4us vs best `kernel-set-splitk` 103387.1us => **1.32x** (faster)

_Legend: %pk = % of dense peak. `skip: needs smXX` = arch-gated (provider needs a newer GPU; not imported here). `import-fail` = library not installed. `incorrect` = disagrees with the fp32 reference._

