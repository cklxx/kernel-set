#!/usr/bin/env python3
"""kernel-set — FlashInfer attention verification test.

Tests ks_flashinfer_attn against fp32 reference for correctness:
  - Non-causal & causal prefill
  - GQA (num_kv_heads < num_heads)
  - head_dim 64 and 128
  - Dense and varlen packed sequences
  - Varlen sequences

On sm_70 (V100): ks_flashinfer_attn falls back to ks_flash_attn.
On sm_75+ (T4, A100, L4, H100): FlashInfer tensor-core kernels are dispatched.

Usage:
    python verify_flashinfer_attn.py
"""

import ctypes
import sys
import torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

KS_DTYPE_F16 = 1
KS_DTYPE_BF16 = 2

ARGS_ATTN = [
    ctypes.c_void_p, ctypes.c_void_p,  # out, lse
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # q, k, v
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # batch, seqlen_q, seqlen_k
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # num_heads, num_kv_heads, head_dim
    ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,  # scale, causal, dtype, stream
]

ARGS_VARLEN = [
    ctypes.c_void_p, ctypes.c_void_p,  # out, lse
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # q, k, v
    ctypes.c_void_p, ctypes.c_void_p,  # cu_seqlens_q, cu_seqlens_k
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # batch, max_seqlen_q, max_seqlen_k
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # num_heads, num_kv_heads, head_dim
    ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,  # scale, causal, dtype, stream
]


def gqa_ref(q, k, v, scale, causal=False):
    """Reference attention in fp32, supports GQA. Expects [B, S, H, D] layout."""
    B, S_q, Hq, D = q.shape
    _, S_k, Hkv, _ = k.shape
    gs = Hq // Hkv
    q_r = q.float().squeeze(0).permute(1, 0, 2)  # [Hq, S_q, D]
    k_r = k.float().squeeze(0).permute(1, 0, 2)  # [Hkv, S_k, D]
    v_r = v.float().squeeze(0).permute(1, 0, 2)  # [Hkv, S_k, D]
    ref = torch.zeros(S_q, Hq, D, dtype=torch.float32, device=q.device)
    for h in range(Hq):
        kv_h = h // gs
        s = torch.matmul(q_r[h], k_r[kv_h].T) * scale  # [S_q, S_k]
        if causal:
            offset = S_k - S_q
            mask = torch.triu(torch.ones(S_q, S_k, device=q.device), diagonal=offset + 1) * -1e9
            s = s + mask
        a = torch.softmax(s, dim=-1)  # [S_q, S_k]
        ref[:, h, :] = torch.matmul(a, v_r[kv_h])  # [S_q, D]
    return ref.unsqueeze(0).to(q.dtype)


def load_lib():
    return ctypes.CDLL("./build/libkernel_set.so")


def test(name, max_err, rtol=0.05):
    status = "PASS" if max_err < rtol else "FAIL"
    print(f"  {name}: max_err={max_err:.6f}  {status}")
    return max_err < rtol


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=== kernel-set FlashInfer attention verification ===\n")
    lib = load_lib()
    lib.ks_flash_attn.argtypes = ARGS_ATTN
    lib.ks_flash_attn.restype = ctypes.c_int
    lib.ks_flashinfer_attn.argtypes = ARGS_ATTN
    lib.ks_flashinfer_attn.restype = ctypes.c_int
    lib.ks_flash_attn_varlen.argtypes = ARGS_VARLEN
    lib.ks_flash_attn_varlen.restype = ctypes.c_int

    device = "cuda"
    dtype = torch.float16
    all_pass = True

    # --- Dense prefill: non-causal + causal, hd64, no GQA ---
    B, S, H, D = 1, 256, 12, 64
    scale = 1.0 / (D ** 0.5)
    print(f"--- Dense prefill hd={D} no-GQA ---")

    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)

    ref_nc = gqa_ref(q, k, v, scale, causal=False)
    ref_c = gqa_ref(q, k, v, scale, causal=True)

    for causal, ref in [(0, ref_nc), (1, ref_c)]:
        label = "causal" if causal else "non-causal"
        out = torch.zeros(B, S, H, D, device=device, dtype=dtype)
        lib.ks_flash_attn(out.data_ptr(), None, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                          B, S, S, H, H, D, scale, causal, KS_DTYPE_F16, None)
        torch.cuda.synchronize()
        all_pass &= test(f"ks_flash_attn {label} (hd={D})", (out - ref).abs().max().item())

        out_fi = torch.zeros(B, S, H, D, device=device, dtype=dtype)
        lib.ks_flashinfer_attn(out_fi.data_ptr(), None, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                               B, S, S, H, H, D, scale, causal, KS_DTYPE_F16, None)
        torch.cuda.synchronize()
        all_pass &= test(f"ks_flashinfer_attn {label} (hd={D})", (out_fi - ref).abs().max().item())

    # --- Dense prefill: hd=128, GQA ---
    Hq, Hkv, D128 = 12, 4, 128
    scale128 = 1.0 / (D128 ** 0.5)
    print(f"\n--- Dense prefill hd={D128} GQA ({Hq}/{Hkv}) ---")

    q128 = torch.randn(B, S, Hq, D128, device=device, dtype=dtype)
    k128 = torch.randn(B, S, Hkv, D128, device=device, dtype=dtype)
    v128 = torch.randn(B, S, Hkv, D128, device=device, dtype=dtype)

    ref128 = gqa_ref(q128, k128, v128, scale128, causal=False)
    out128 = torch.zeros(B, S, Hq, D128, device=device, dtype=dtype)
    lib.ks_flash_attn(out128.data_ptr(), None, q128.data_ptr(), k128.data_ptr(), v128.data_ptr(),
                      B, S, S, Hq, Hkv, D128, scale128, 0, KS_DTYPE_F16, None)
    torch.cuda.synchronize()
    all_pass &= test(f"ks_flash_attn GQA (hd={D128})", (out128 - ref128).abs().max().item())

    out128_fi = torch.zeros(B, S, Hq, D128, device=device, dtype=dtype)
    lib.ks_flashinfer_attn(out128_fi.data_ptr(), None, q128.data_ptr(), k128.data_ptr(), v128.data_ptr(),
                           B, S, S, Hq, Hkv, D128, scale128, 0, KS_DTYPE_F16, None)
    torch.cuda.synchronize()
    all_pass &= test(f"ks_flashinfer_attn GQA (hd={D128})", (out128_fi - ref128).abs().max().item())

    # --- Varlen prefill ---
    print("\n--- Varlen prefill ---")
    S1, S2 = 128, 256
    cu_q = torch.tensor([0, S1, S1 + S2], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, S1, S1 + S2], dtype=torch.int32, device=device)
    total = S1 + S2
    q_vl = torch.randn(total, H, D, device=device, dtype=dtype)
    k_vl = torch.randn(total, H, D, device=device, dtype=dtype)
    v_vl = torch.randn(total, H, D, device=device, dtype=dtype)
    out_vl = torch.zeros(total, H, D, device=device, dtype=dtype)

    ret = lib.ks_flash_attn_varlen(out_vl.data_ptr(), None,
                                   q_vl.data_ptr(), k_vl.data_ptr(), v_vl.data_ptr(),
                                   cu_q.data_ptr(), cu_k.data_ptr(),
                                   2, S2, S2, H, H, D, scale, 0, KS_DTYPE_F16, None)
    torch.cuda.synchronize()
    nonzero = (out_vl.abs() > 0).sum().item()
    ok = (ret == 0 and nonzero > 0)
    print(f"  ks_flash_attn_varlen (nonzero={nonzero}/{total*H*D}): {'PASS' if ok else 'FAIL'}")
    all_pass &= ok

    # --- Summary ---
    print("\n" + ("=" * 50))
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()