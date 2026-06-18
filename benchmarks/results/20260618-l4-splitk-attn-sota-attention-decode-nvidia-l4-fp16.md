# kernel-set vs SOTA — NVIDIA L4

- **GPU**: NVIDIA L4 (sm_89, CC 8.9, 58 SMs, 22.0 GB) | dense peak BW ~300 GB/s | dense fp16/bf16 TC ~121 TFLOP/s | dense fp8/int8 TC ~242 TFLOP/s
- **detected via**: kernel_set
- **clocks**: SM 210 MHz, mem 405 MHz (throttle: 0x0000000000000001)
- **driver**: 580.82.07 | CUDA 12.8 | cuDNN 91900 | power cap 72.00 W | ECC Enabled
- **dtype**: fp16 | TF32 matmul=True (fp32_precision=high)
- **timing**: L2-flush=on | method=cuda-events (flushed) | target-ms=80.0 | iters=auto | warmup=10 | L2-buffer=96 MB
- **kernel-set**: 0.2.1 (backend cuda)
- **torch**: 2.11.0+cu128
- **nvcc**: Cuda compilation tools, release 12.8, V12.8.93
- **timestamp**: 20260618-l4-splitk-attn-sota-attention-decode
- **host**: Linux-6.6.122+-x86_64-with-glibc2.35

**Providers**: ok=4 · skip=2 · import-fail=2 · error=2 · incorrect=0. Correctness is gated against a shared fp32 reference at the dtype tolerance BEFORE speed is reported.

Latency cells show **median (min)** microseconds (CUDA events, L2-flushed). The perf column is GB/s (bandwidth-bound ops) or TFLOP/s (compute-bound), with **% of dense peak** for this SKU+dtype. Arch-gated providers SKIP on unsupported SMs (`needs smXX`) and are never imported there.

## attn_decode

| op | shape | impl | dtype | lat us (min) | GB/s or TFLOP/s (%pk) | rel_err | status |
|---|---|---|---|--:|--:|--:|---|
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 7501.3 (7465.0) | 71.6 GB/s (24%) | 3.43e-04 | ok |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | kernel-set-splitk | fp16 | 7240.2 (7228.4) | 74.2 GB/s (25%) | 4.59e-04 | ok |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | flashinfer | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 4.01e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_decode | seqs=64,ctx=2048,qh=32,kvh=8,hd=128 | sgl-attn | fp16 | - | - | - | skip: needs sm90 (device is sm89) |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | kernel-set | fp16 | 14669.8 (14343.2) | 73.2 GB/s (24%) | 3.71e-04 | ok |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | kernel-set-splitk | fp16 | 14616.6 (14606.3) | 73.5 GB/s (24%) | 5.87e-04 | ok |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | flashinfer | fp16 | - | - | - | import-fail: ModuleNotFoundError: No module named 'flashinfer' |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | sdpa-flash | fp16 | - | - | 3.80e-04 | error: AttributeError: '_GeneratorContextManager' object has no attribute 'args' |
| attn_decode | seqs=256,ctx=1024,qh=32,kvh=8,hd=128 | sgl-attn | fp16 | - | - | - | skip: needs sm90 (device is sm89) |

**kernel-set vs best-SOTA**:

- `seqs=64,ctx=2048,qh=32,kvh=8,hd=128`: kernel-set 7501.3us vs best `kernel-set-splitk` 7240.2us => **0.97x** (slower)
- `seqs=256,ctx=1024,qh=32,kvh=8,hd=128`: kernel-set 14669.8us vs best `kernel-set-splitk` 14616.6us => **1.00x** (slower)

_Legend: %pk = % of dense peak. `skip: needs smXX` = arch-gated (provider needs a newer GPU; not imported here). `import-fail` = library not installed. `incorrect` = disagrees with the fp32 reference._

