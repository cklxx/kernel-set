# SSM + FP8-GEMM — GPU correctness verification (L4, sm89)

Self-developed kernels verified on a real L4 via ctypes-direct calls vs inline
torch references (bf16; sequential-recurrence ref for the scan, dequant ref for fp8).

| kernel | reference | rel_err | verdict |
|---|---|---|---|
| `ks_causal_conv1d` (depthwise causal + SiLU) | F.conv1d depthwise causal + bias + SiLU | 0.00265 | ✓ correct |
| `ks_selective_scan` (Mamba scan) | fp32 sequential recurrence (softplus, exp(dt·A), C·h, D skip, z gate) | 0.00240 | ✓ correct |
| `ks_gemm_fp8` (e4m3, per-token×per-channel scales) | (a_fp8·a_scale) @ (b_fp8·b_scale) | 0.00246 | ✓ correct |

All within bf16 tolerance. Build: `-DCMAKE_CUDA_ARCHITECTURES=89`, CUDA 12.x.
