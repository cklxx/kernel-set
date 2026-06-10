import os, subprocess, sys

def run(cmd):
    print(">", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    print("  rc=", r.returncode, flush=True)
    return r.returncode

LIB = "/content/build/libkernel_set.so"

# 1. extract python bindings + sources (keep existing build dir for cache)
run("tar -xzf ks.tar.gz")

# 2. build single-arch (L4 = sm_89) only if lib missing; reuse cache otherwise
if not os.path.exists(LIB):
    os.environ["KS_JOBS"] = "2"
    run("cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 >/tmp/cfg.log 2>&1")
    rc = run("cmake --build build -j 2 2>/tmp/build.log")
    if rc != 0:
        run("tail -n 40 /tmp/build.log")
        sys.exit(1)
print("BUILD OK", flush=True)

# 3. nvfp4 quantize roundtrip test
sys.path.insert(0, "/content/bindings/python")
os.environ["KERNEL_SET_LIB"] = "/content/build/libkernel_set.so"
import torch
import kernel_set as ks
from kernel_set import quant

torch.manual_seed(0)
rows, cols = 64, 256
x = torch.randn(rows, cols, dtype=torch.float16, device="cuda") * 2.0

out_fp4 = torch.empty((rows, cols // 2), dtype=torch.uint8, device="cuda")
out_scales = torch.empty((rows, cols // 16), dtype=torch.uint8, device="cuda")
quant.quantize_nvfp4(out_fp4, out_scales, x, 1.0, rows, cols)
torch.cuda.synchronize()
print("quantize_nvfp4 OK, out_fp4", tuple(out_fp4.shape), "scales", tuple(out_scales.shape), flush=True)

# decode on host to verify round-trip error
def decode_e2m1(b):
    s = (b >> 3) & 1; e = (b >> 1) & 3; m = b & 1
    if e == 0: v = 0.5 if m else 0.0
    elif e == 1: v = 1.5 if m else 1.0
    elif e == 2: v = 3.0 if m else 2.0
    else: v = 6.0 if m else 4.0
    return -v if s else v

# e4m3 scale decode: reinterpret uint8 as fp8e4m3 via torch
fp4 = out_fp4.cpu().numpy()
sc_bytes = out_scales.cpu().view(torch.uint8).numpy()
# decode scales using torch fp8
sc = out_scales.view(torch.float8_e4m3fn).float().cpu().numpy()
xref = x.float().cpu().numpy()

import numpy as np
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
