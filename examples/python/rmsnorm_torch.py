#!/usr/bin/env python3
"""kernel-set Python example: RMSNorm on a torch CUDA tensor, end to end.

Calls ``ks_rms_norm`` via the ``kernel_set`` ctypes binding, then checks the
result against a pure-torch reference.

Run:
    pip install ./bindings/python "kernel_set[torch]"
    export KERNEL_SET_LIB="$PWD/build/libkernel_set.so"   # the built shared lib
    python examples/python/rmsnorm_torch.py
"""

import torch
import kernel_set as ks


def rms_norm_reference(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure-torch RMSNorm in fp32 (matches the kernel's fp32 accumulation)."""
    xf = x.float()
    inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv_rms).to(x.dtype) * w


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example needs a CUDA device.")

    print(f"kernel-set {ks.version()} ({ks.backend_name()} backend)")

    rows, cols, eps = 8, 4096, 1e-6
    dtype = torch.float16

    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    w = torch.randn(cols, device="cuda", dtype=dtype)
    out = torch.empty_like(x)

    # The kernel call. rows/cols/dtype/stream are inferred from the tensors;
    # the wrapper extracts data_ptr()s and orders the launch on torch's
    # current CUDA stream.
    ks.norm.rms_norm(out, x, w, eps=eps)
    torch.cuda.synchronize()

    ref = rms_norm_reference(x, w, eps)
    max_abs_err = (out.float() - ref.float()).abs().max().item()
    print(f"out shape       : {tuple(out.shape)}")
    print(f"max abs error   : {max_abs_err:.4e}")
    assert max_abs_err < 2e-2, "RMSNorm result diverged from the reference"
    print("OK")


if __name__ == "__main__":
    main()
