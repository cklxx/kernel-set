"""FFI verification driver for the kernel_set Python binding against the stub.

Exercises: lib load, symbol resolution, ctypes argtype marshalling, and
ks_status_t return handling — without touching any device memory (dummy/0
pointers only). Real GPU math is verified separately.
"""

from __future__ import annotations

import ctypes
import sys


def main() -> int:
    import kernel_set as ks
    from kernel_set._lib import _MISSING, lib

    print(f"[py] loaded lib: {lib._name}")

    # --- introspection ----------------------------------------------------
    v = ks.version()
    print(f"[py] ks.version() = {v!r}")
    assert v == "0.0.0-stub", f"unexpected version: {v!r}"

    bn = ks.backend_name()
    print(f"[py] ks.backend_name() = {bn!r}")
    assert bn == "stub", f"unexpected backend: {bn!r}"

    print(f"[py] ks.dtype_name(F16) = {ks.dtype_name(ks.DType.F16)!r}")
    print(f"[py] ks.dtype_size_bits(F16) = {ks.dtype_size_bits(ks.DType.F16)}")
    assert ks.dtype_size_bits(ks.DType.F16) == 16

    from kernel_set.runtime import status_string

    print(f"[py] status_string(SUCCESS) = {status_string(0)!r}")
    print(f"[py] last_error_string() = {ks.runtime.last_error_string()!r}")

    # --- device queries (out-params) -------------------------------------
    n = ks.device_count()
    print(f"[py] ks.device_count() = {n}")
    assert n == 1, f"stub should report 1 device, got {n}"
    print(f"[py] ks.get_device() = {ks.get_device()}")
    ks.set_device(0)
    props = ks.get_device_properties(0)
    print(f"[py] device props name = {props.name!r}, warp={props.warp_size}")
    assert props.name == "stub-device"
    assert props.warp_size == 32

    # --- streams + memory (out-params, no deref) -------------------------
    s = ks.runtime.stream_create()
    print(f"[py] stream_create() = {s}")
    ks.runtime.stream_synchronize(s)
    ks.runtime.stream_destroy(s)

    p = ks.runtime.malloc_device(1024)
    print(f"[py] malloc_device(1024) = {p}")
    ks.runtime.free_device(p)

    # --- op wrappers reaching the C call with dummy (0) device pointers ---
    # These confirm ctypes argtypes marshal correctly and a ks_status_t
    # comes back as KS_SUCCESS. We call the *raw* lib for ops to avoid the
    # higher-level wrappers' tensor-shape requirements, then a couple of the
    # pythonic wrappers that accept raw int pointers.
    from kernel_set._lib import check

    # raw elementwise add: out, a, b, n, dtype, stream
    check(lib.ks_add(0, 0, 0, 16, ks.DType.F16, None), "ks_add")
    print("[py] raw lib.ks_add(... dummy ptrs ...) -> KS_SUCCESS")

    # raw rms_norm: out, input, weight, rows, cols, eps, dtype, stream
    check(
        lib.ks_rms_norm(0, 0, 0, 2, 8, ctypes.c_float(1e-6), ks.DType.F16, None),
        "ks_rms_norm",
    )
    print("[py] raw lib.ks_rms_norm(... dummy ptrs ...) -> KS_SUCCESS")

    # raw gemm: exercises many int64 + 2 floats + enums (calling convention)
    check(
        lib.ks_gemm(
            0, 0, 0,            # c, a, b
            4, 4, 4,            # m, n, k
            0, 0,               # trans_a, trans_b
            4, 4, 4,            # lda, ldb, ldc
            ctypes.c_float(1.0), ctypes.c_float(0.0),
            ks.DType.F16, None,
        ),
        "ks_gemm",
    )
    print("[py] raw lib.ks_gemm(... dummy ptrs ...) -> KS_SUCCESS")

    # raw sample: exercises uint64 seed/offset marshalling
    check(
        lib.ks_sample(
            None, None, 0,      # out_tokens, out_probs, logits
            None, None, None,   # temperatures, top_ks, top_ps
            4, 32000,           # num_seqs, vocab_size
            ctypes.c_uint64(1234), ctypes.c_uint64(0),
            ks.DType.F32, None,
        ),
        "ks_sample",
    )
    print("[py] raw lib.ks_sample(... dummy ptrs, uint64 ...) -> KS_SUCCESS")

    # higher-level wrapper that accepts a raw int pointer for out-param:
    # softmax via raw lib (rows, cols, temperature float)
    check(
        lib.ks_softmax(0, 0, 4, 32000, ctypes.c_float(1.0), ks.DType.F32, None),
        "ks_softmax",
    )
    print("[py] raw lib.ks_softmax(... dummy ptrs ...) -> KS_SUCCESS")

    if _MISSING:
        print(f"[py] WARNING missing symbols in lib: {_MISSING}")
    else:
        print("[py] all declared symbols resolved (no _MISSING)")

    print("[py] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
