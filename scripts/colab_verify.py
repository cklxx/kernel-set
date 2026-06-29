#!/usr/bin/env python3
"""kernel-set — Colab verification script (L4 / sm_89).

Builds kernel_set for sm_89, runs the nvfp4 quantize roundtrip test AND the
FlashInfer attention correctness test. Also runs on T4 (sm_75) and A100 (sm_80).

Gating:
  - sm_70 (V100): FlashInfer falls back to ks_flash_attn (no ldmatrix/mma)
  - sm_75+ (T4, A100, L4, H100): FlashInfer tensor-core kernels are dispatched

Usage on Colab:
    !git clone https://github.com/.../kernel-set.git
    !python kernel-set/scripts/colab_verify.py
"""

import os, subprocess, sys
import ctypes
import torch
import numpy as np


def run(cmd, check=True):
    print(">", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        print(f"  FAILED (rc={r.returncode})", flush=True)
        sys.exit(1)
    return r.returncode


# ---------------------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT, "build")
LIB = os.path.join(BUILD_DIR, "libkernel_set.so")

# Detect GPU arch
gpu_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
sm = major * 10 + minor
print(f"GPU: {gpu_name} (sm_{sm})", flush=True)

if not os.path.exists(LIB):
    os.chdir(ROOT)
    os.environ["KS_JOBS"] = "2"
    run(f"cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES={sm} >/tmp/cfg.log 2>&1")
    run("cmake --build build -j 2 2>/tmp/build.log")
print("BUILD OK", flush=True)

# ---------------------------------------------------------------------------
# 2. Load library
# ---------------------------------------------------------------------------
lib = ctypes.CDLL(LIB)
print("Library loaded OK", flush=True)

# ---------------------------------------------------------------------------
# 3. NVFP4 quantize roundtrip test
# ---------------------------------------------------------------------------
KS_DTYPE_F16 = 1
KS_DTYPE_F32 = 0

torch.manual_seed(0)
rows, cols = 64, 256
x = torch.randn(rows, cols, dtype=torch.float16, device="cuda") * 2.0

# Locate the quantize function
lib.ks_quantize_nvfp4.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_float, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p
]
lib.ks_quantize_nvfp4.restype = ctypes.c_int

out_fp4 = torch.empty((rows, cols // 2), dtype=torch.uint8, device="cuda")
out_scales = torch.empty((rows, cols // 16), dtype=torch.uint8, device="cuda")
ret = lib.ks_quantize_nvfp4(out_fp4.data_ptr(), out_scales.data_ptr(), x.data_ptr(),
                             1.0, rows, cols, KS_DTYPE_F16, None)
torch.cuda.synchronize()
print(f"quantize_nvfp4 returned {ret}, out_fp4 {tuple(out_fp4.shape)} scales {tuple(out_scales.shape)}", flush=True)

# Decode on host to verify round-trip error
def decode_e2m1(b):
    s = (b >> 3) & 1; e = (b >> 1) & 3; m = b & 1
    if e == 0: v = 0.5 if m else 0.0
    elif e == 1: v = 1.5 if m else 1.0
    elif e == 2: v = 3.0 if m else 2.0
    else: v = 6.0 if m else 4.0
    return -v if s else v

fp4 = out_fp4.cpu().numpy()
sc = out_scales.view(torch.float8_e4m3fn).float().cpu().numpy()
xref = x.float().cpu().numpy()

err = 0.0; denom = 0.0
for r in range(rows):
    for blk in range(cols // 16):
        scale = sc[r, blk]
        for j in range(16):
            c = blk * 16 + j
            byte = fp4[r, c // 2]
            nib = (byte & 0xF) if (c % 2 == 0) else (byte >> 4)
            deq = decode_e2m1(int(nib)) * scale
            err += abs(deq - xref[r, c])
            denom += abs(xref[r, c])
rel = err / denom
print(f"nvfp4 roundtrip mean rel_err = {rel:.4f}", flush=True)
assert rel < 0.15, "rel_err too high"
print("NVFP4 QUANTIZE TEST PASSED", flush=True)

# ---------------------------------------------------------------------------
# 4. FlashInfer attention correctness test
# ---------------------------------------------------------------------------
print("\n--- FlashInfer attention verification ---", flush=True)

ARGS_ATTN = [
    ctypes.c_void_p, ctypes.c_void_p,  # out, lse
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # q, k, v
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # batch, seqlen_q, seqlen_k
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # num_heads, num_kv_heads, head_dim
    ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,  # scale, causal, dtype, stream
]
lib.ks_flash_attn.argtypes = ARGS_ATTN
lib.ks_flash_attn.restype = ctypes.c_int
lib.ks_flashinfer_attn.argtypes = ARGS_ATTN
lib.ks_flashinfer_attn.restype = ctypes.c_int

B, S, H, D = 1, 256, 12, 64
scale = 1.0 / (D ** 0.5)
device = "cuda"
dtype = torch.float16

q = torch.randn(B, S, H, D, device=device, dtype=dtype)
k = torch.randn(B, S, H, D, device=device, dtype=dtype)
v = torch.randn(B, S, H, D, device=device, dtype=dtype)

# fp32 reference
q_r = q.float().squeeze(0).permute(1, 0, 2)
k_r = k.float().squeeze(0).permute(1, 0, 2)
v_r = v.float().squeeze(0).permute(1, 0, 2)
scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale
attn = torch.softmax(scores, dim=-1)
ref = torch.matmul(attn, v_r).permute(1, 0, 2).unsqueeze(0).to(dtype)

# Causal reference
causal_mask = torch.triu(torch.ones(S, S, device=device), diagonal=1) * -1e9
scores_c = scores + causal_mask
attn_c = torch.softmax(scores_c, dim=-1)
ref_c = torch.matmul(attn_c, v_r).permute(1, 0, 2).unsqueeze(0).to(dtype)

for causal, ref_t, label in [(0, ref, "non-causal"), (1, ref_c, "causal")]:
    for name, fn in [("ks_flash_attn", lib.ks_flash_attn), ("ks_flashinfer_attn", lib.ks_flashinfer_attn)]:
        out = torch.zeros(B, S, H, D, device=device, dtype=dtype)
        fn(out.data_ptr(), None, q.data_ptr(), k.data_ptr(), v.data_ptr(),
           B, S, S, H, H, D, scale, causal, KS_DTYPE_F16, None)
        torch.cuda.synchronize()
        err = (out - ref_t).abs().max().item()
        status = "PASS" if err < 0.05 else "FAIL"
        print(f"  {name} {label} (hd=64): max_err={err:.6f}  {status}", flush=True)
        assert err < 0.05, f"{name} {label} failed: err={err}"

# GQA test
Hq, Hkv, D128 = 12, 4, 128
scale128 = 1.0 / (D128 ** 0.5)
q128 = torch.randn(B, S, Hq, D128, device=device, dtype=dtype)
k128 = torch.randn(B, S, Hkv, D128, device=device, dtype=dtype)
v128 = torch.randn(B, S, Hkv, D128, device=device, dtype=dtype)

q_128r = q128.float().squeeze(0).permute(1, 0, 2)
k_128r = k128.float().squeeze(0).permute(1, 0, 2)
v_128r = v128.float().squeeze(0).permute(1, 0, 2)
gs = Hq // Hkv
ref_128 = torch.zeros(S, Hq, D128, dtype=torch.float32, device=device)
for h in range(Hq):
    kv_h = h // gs
    s = torch.matmul(q_128r[h], k_128r[kv_h].T) * scale128
    a = torch.softmax(s, dim=-1)
    ref_128[:, h, :] = torch.matmul(a, v_128r[kv_h])
ref_128 = ref_128.unsqueeze(0).to(dtype)

out_128 = torch.zeros(B, S, Hq, D128, device=device, dtype=dtype)
lib.ks_flashinfer_attn(out_128.data_ptr(), None, q128.data_ptr(), k128.data_ptr(), v128.data_ptr(),
                       B, S, S, Hq, Hkv, D128, scale128, 0, KS_DTYPE_F16, None)
torch.cuda.synchronize()
err_128 = (out_128 - ref_128).abs().max().item()
print(f"  ks_flashinfer_attn GQA (hd=128): max_err={err_128:.6f}  {'PASS' if err_128 < 0.05 else 'FAIL'}", flush=True)
assert err_128 < 0.05, f"GQA failed: err={err_128}"

print("\n========================================")
print("ALL TESTS PASSED (nvfp4 + attention)")
print("========================================", flush=True)