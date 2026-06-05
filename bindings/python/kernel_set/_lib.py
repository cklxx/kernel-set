"""Low-level ctypes FFI layer for the ``kernel_set`` shared library.

This module is responsible for three things:

1. Locating and ``dlopen``-ing the prebuilt ``kernel_set`` shared library
   (``libkernel_set.so`` / ``libkernel_set.dylib`` / ``kernel_set.dll``).
2. Declaring ``argtypes`` / ``restype`` for *every* function in the C ABI so
   that ctypes marshals arguments correctly (especially 64-bit integers and
   pointers on every platform / Python build).
3. Wrapping the ``ks_status_t`` returning calls so that a non-zero status is
   turned into a :class:`KernelSetError` carrying both the symbolic status
   string (``ks_status_string``) and the thread-local backend error message
   (``ks_last_error_string``).

The declarations here mirror ``include/kernel_set/*.h`` exactly. Device
pointers are modelled as ``ctypes.c_void_p`` and are passed as raw integer
addresses; streams are ``ctypes.c_void_p`` as well (``0`` / ``None`` = the
default stream).

Higher-level, pythonic wrappers live in the per-category op modules
(``norm.py``, ``attention.py``, ...). End users normally do not import this
module directly, but it is fully usable on its own for raw FFI access::

    from kernel_set._lib import lib, check, KS_DTYPE_F16
    check(lib.ks_rms_norm(out, x, w, rows, cols, eps, KS_DTYPE_F16, None))
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    POINTER,
    Structure,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_int32,
    c_int64,
    c_size_t,
    c_uint64,
    c_void_p,
)
from typing import Optional

__all__ = [
    "lib",
    "check",
    "KernelSetError",
    "KsDeviceProperties",
    # status codes
    "KS_SUCCESS",
    "KS_ERROR_INVALID_ARGUMENT",
    "KS_ERROR_UNSUPPORTED_DTYPE",
    "KS_ERROR_UNSUPPORTED_SHAPE",
    "KS_ERROR_CUDA",
    "KS_ERROR_NOT_IMPLEMENTED",
    "KS_ERROR_OUT_OF_MEMORY",
    "KS_ERROR_ARCH_UNSUPPORTED",
    "KS_ERROR_INTERNAL",
    # dtypes
    "KS_DTYPE_F32",
    "KS_DTYPE_F16",
    "KS_DTYPE_BF16",
    "KS_DTYPE_F8E4M3",
    "KS_DTYPE_F8E5M2",
    "KS_DTYPE_F64",
    "KS_DTYPE_I64",
    "KS_DTYPE_I32",
    "KS_DTYPE_I8",
    "KS_DTYPE_U8",
    "KS_DTYPE_I4",
    # activations
    "KS_ACT_NONE",
    "KS_ACT_RELU",
    "KS_ACT_GELU",
    "KS_ACT_GELU_TANH",
    "KS_ACT_SILU",
    # quant modes
    "KS_QUANT_PER_TENSOR",
    "KS_QUANT_PER_TOKEN",
    "KS_QUANT_PER_CHANNEL",
    "KS_QUANT_GROUPWISE",
    # memcpy kinds
    "KS_MEMCPY_HOST_TO_DEVICE",
    "KS_MEMCPY_DEVICE_TO_HOST",
    "KS_MEMCPY_DEVICE_TO_DEVICE",
]

# ---------------------------------------------------------------------------
# Enum constants (mirror include/kernel_set/types.h + runtime.h)
# ---------------------------------------------------------------------------

# ks_status_t
KS_SUCCESS = 0
KS_ERROR_INVALID_ARGUMENT = 1
KS_ERROR_UNSUPPORTED_DTYPE = 2
KS_ERROR_UNSUPPORTED_SHAPE = 3
KS_ERROR_CUDA = 4
KS_ERROR_NOT_IMPLEMENTED = 5
KS_ERROR_OUT_OF_MEMORY = 6
KS_ERROR_ARCH_UNSUPPORTED = 7
KS_ERROR_INTERNAL = 8

# ks_dtype_t
KS_DTYPE_F32 = 0
KS_DTYPE_F16 = 1
KS_DTYPE_BF16 = 2
KS_DTYPE_F8E4M3 = 3
KS_DTYPE_F8E5M2 = 4
KS_DTYPE_F64 = 5
KS_DTYPE_I64 = 6
KS_DTYPE_I32 = 7
KS_DTYPE_I8 = 8
KS_DTYPE_U8 = 9
KS_DTYPE_I4 = 10

# ks_activation_t
KS_ACT_NONE = 0
KS_ACT_RELU = 1
KS_ACT_GELU = 2
KS_ACT_GELU_TANH = 3
KS_ACT_SILU = 4

# ks_quant_mode_t
KS_QUANT_PER_TENSOR = 0
KS_QUANT_PER_TOKEN = 1
KS_QUANT_PER_CHANNEL = 2
KS_QUANT_GROUPWISE = 3

# ks_memcpy_kind_t
KS_MEMCPY_HOST_TO_DEVICE = 0
KS_MEMCPY_DEVICE_TO_HOST = 1
KS_MEMCPY_DEVICE_TO_DEVICE = 2


# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------


class KsDeviceProperties(Structure):
    """ctypes mirror of ``ks_device_properties_t`` (runtime.h)."""

    _fields_ = [
        ("name", c_char * 256),
        ("compute_major", c_int),
        ("compute_minor", c_int),
        ("multiprocessor_count", c_int),
        ("max_threads_per_block", c_int),
        ("max_shared_memory_per_block", c_int),
        ("warp_size", c_int),
        ("total_global_memory", c_size_t),
        ("supports_bf16", c_int),
        ("supports_fp8", c_int),
        ("supports_tf32", c_int),
    ]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class KernelSetError(RuntimeError):
    """Raised when a ``kernel_set`` C ABI call returns a non-zero status.

    Attributes
    ----------
    status:
        The numeric ``ks_status_t`` value.
    status_name:
        Symbolic status string from ``ks_status_string``.
    backend_message:
        Thread-local backend message from ``ks_last_error_string`` (may be
        empty if the backend did not record one).
    func:
        Name of the C function that failed (if known).
    """

    def __init__(
        self,
        status: int,
        status_name: str,
        backend_message: str = "",
        func: Optional[str] = None,
    ) -> None:
        self.status = status
        self.status_name = status_name
        self.backend_message = backend_message
        self.func = func
        parts = []
        if func:
            parts.append(f"{func} failed")
        else:
            parts.append("kernel_set call failed")
        parts.append(f"[{status_name} (={status})]")
        if backend_message:
            parts.append(f": {backend_message}")
        super().__init__(" ".join(parts))


# ---------------------------------------------------------------------------
# Library location & loading
# ---------------------------------------------------------------------------

_LIB_BASENAME = "kernel_set"


def _candidate_library_names() -> list[str]:
    """Platform-appropriate filenames for the shared library."""
    if sys.platform.startswith("win"):
        return ["kernel_set.dll", "libkernel_set.dll"]
    if sys.platform == "darwin":
        return ["libkernel_set.dylib", "kernel_set.dylib"]
    return ["libkernel_set.so", "kernel_set.so"]


def _candidate_dirs() -> list[str]:
    """Directories to search for the shared library, in priority order."""
    dirs: list[str] = []

    # 1. Explicit override of the *directory* containing the lib.
    env_dir = os.environ.get("KERNEL_SET_LIB_DIR")
    if env_dir:
        dirs.append(env_dir)

    # 2. Bundled next to this package (wheel that vendored the .so).
    here = os.path.dirname(os.path.abspath(__file__))
    dirs.append(here)
    dirs.append(os.path.join(here, "lib"))
    dirs.append(os.path.join(here, ".libs"))

    # 3. Repo-relative build trees (editable installs from a source checkout).
    #    bindings/python/kernel_set/_lib.py -> repo root is three levels up.
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    for sub in ("build", os.path.join("build", "lib"), "lib", "install/lib"):
        dirs.append(os.path.join(repo_root, sub))

    # 4. Common system locations.
    dirs.extend(
        [
            "/usr/local/lib",
            "/usr/lib",
            "/opt/kernel_set/lib",
        ]
    )
    return dirs


def _load_library() -> ctypes.CDLL:
    """Locate and ``dlopen`` the kernel_set shared library.

    Resolution order:

    1. ``KERNEL_SET_LIB`` — a full path *or* bare filename of the library.
    2. ``KERNEL_SET_LIB_DIR`` + each platform basename.
    3. Directories bundled with / near this package and repo build trees.
    4. The platform loader's default search path (``LD_LIBRARY_PATH`` etc.).
    """
    names = _candidate_library_names()
    tried: list[str] = []

    # 1. KERNEL_SET_LIB may be an absolute path or a bare name.
    explicit = os.environ.get("KERNEL_SET_LIB")
    if explicit:
        explicit = os.path.expanduser(explicit)
        candidates = [explicit]
        if not os.path.isabs(explicit) and os.path.dirname(explicit) == "":
            # bare filename: let the loader resolve via search path too.
            candidates.append(explicit)
        for cand in candidates:
            tried.append(cand)
            try:
                return ctypes.CDLL(cand)
            except OSError:
                continue

    # 2 & 3. Search candidate directories for each platform name.
    for d in _candidate_dirs():
        for name in names:
            full = os.path.join(d, name)
            if os.path.isfile(full):
                tried.append(full)
                try:
                    return ctypes.CDLL(full)
                except OSError:
                    continue

    # 4. Fall back to the loader's own search path.
    for name in names:
        tried.append(f"<loader search path>/{name}")
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue

    raise OSError(
        "Could not locate the kernel_set shared library "
        f"({', '.join(names)}).\n"
        "Set KERNEL_SET_LIB to the full path of the prebuilt library, or "
        "KERNEL_SET_LIB_DIR to the directory containing it.\n"
        "Searched:\n  " + "\n  ".join(tried)
    )


lib: ctypes.CDLL = _load_library()


# ---------------------------------------------------------------------------
# argtypes / restype declarations
# ---------------------------------------------------------------------------

# Convenience aliases for readability below.
_VP = c_void_p  # device pointer / stream / generic pointer
_STREAM = c_void_p
_DTYPE = c_int  # enums are plain ints in the ABI
_STATUS = c_int


_MISSING: list[str] = []


def _decl(name: str, argtypes, restype=_STATUS) -> None:
    """Attach argtypes/restype to a function if it exists in the library.

    A missing symbol does not abort import (a slightly older prebuilt library
    may lack a newer entry point); it is recorded in ``_MISSING`` and a clear
    :class:`AttributeError` is raised only if/when that function is actually
    called.
    """
    try:
        fn = getattr(lib, name)
    except AttributeError:
        _MISSING.append(name)
        return
    fn.argtypes = argtypes
    fn.restype = restype


# ---- runtime.h ------------------------------------------------------------

_decl("ks_version", [], c_char_p)
_decl("ks_status_string", [_STATUS], c_char_p)
_decl("ks_dtype_size_bits", [_DTYPE], c_int)
_decl("ks_dtype_name", [_DTYPE], c_char_p)
_decl("ks_backend_name", [], c_char_p)
_decl("ks_last_error_string", [], c_char_p)

_decl("ks_device_count", [POINTER(c_int)])
_decl("ks_set_device", [c_int])
_decl("ks_get_device", [POINTER(c_int)])
_decl("ks_get_device_properties", [c_int, POINTER(KsDeviceProperties)])

_decl("ks_stream_create", [POINTER(c_void_p)])
_decl("ks_stream_destroy", [_STREAM])
_decl("ks_stream_synchronize", [_STREAM])

_decl("ks_malloc_device", [POINTER(c_void_p), c_size_t])
_decl("ks_free_device", [_VP])
_decl("ks_memcpy", [_VP, _VP, c_size_t, c_int, _STREAM])
_decl("ks_memset_device", [_VP, c_int, c_size_t, _STREAM])

# ---- norm.h ---------------------------------------------------------------

_decl(
    "ks_rms_norm",
    [_VP, _VP, _VP, c_int64, c_int64, c_float, _DTYPE, _STREAM],
)
_decl(
    "ks_rms_norm_residual",
    [_VP, _VP, _VP, _VP, _VP, c_int64, c_int64, c_float, _DTYPE, _STREAM],
)
_decl(
    "ks_layer_norm",
    [_VP, _VP, _VP, _VP, c_int64, c_int64, c_float, _DTYPE, _STREAM],
)
_decl(
    "ks_rms_norm_backward",
    [_VP, _VP, _VP, _VP, _VP, c_int64, c_int64, c_float, _DTYPE, _STREAM],
)
_decl(
    "ks_layer_norm_backward",
    [_VP, _VP, _VP, _VP, _VP, _VP, c_int64, c_int64, c_float, _DTYPE, _STREAM],
)

# ---- activation.h ---------------------------------------------------------

_decl("ks_silu", [_VP, _VP, c_int64, _DTYPE, _STREAM])
_decl("ks_gelu", [_VP, _VP, c_int64, c_int, _DTYPE, _STREAM])
_decl("ks_relu", [_VP, _VP, c_int64, _DTYPE, _STREAM])
_decl("ks_swiglu", [_VP, _VP, _VP, c_int64, c_int64, _DTYPE, _STREAM])
_decl("ks_swiglu_packed", [_VP, _VP, c_int64, c_int64, _DTYPE, _STREAM])
_decl("ks_geglu", [_VP, _VP, _VP, c_int64, c_int64, c_int, _DTYPE, _STREAM])
_decl(
    "ks_swiglu_backward",
    [_VP, _VP, _VP, _VP, _VP, c_int64, c_int64, _DTYPE, _STREAM],
)

# ---- attention.h ----------------------------------------------------------

_decl(
    "ks_flash_attn_varlen",
    [
        _VP, _VP, _VP, _VP, _VP,            # out, lse, q, k, v
        POINTER(c_int32), POINTER(c_int32),  # cu_seqlens_q, cu_seqlens_k
        c_int, c_int, c_int,               # batch, max_seqlen_q, max_seqlen_k
        c_int, c_int, c_int,               # num_heads, num_kv_heads, head_dim
        c_float, c_int, _DTYPE, _STREAM,   # scale, causal, dtype, stream
    ],
)
_decl(
    "ks_flash_attn",
    [
        _VP, _VP, _VP, _VP, _VP,           # out, lse, q, k, v
        c_int, c_int, c_int,               # batch, seqlen_q, seqlen_k
        c_int, c_int, c_int,               # num_heads, num_kv_heads, head_dim
        c_float, c_int, _DTYPE, _STREAM,
    ],
)
_decl(
    "ks_paged_attn_decode",
    [
        _VP, _VP, _VP, _VP,                # out, q, k_cache, v_cache
        POINTER(c_int32), POINTER(c_int32),  # block_tables, seq_lens
        c_int, c_int, c_int, c_int,        # num_seqs, num_heads, num_kv_heads, head_dim
        c_int, c_int,                      # block_size, max_blocks_per_seq
        c_float, _DTYPE, _STREAM,
    ],
)
_decl(
    "ks_reshape_and_cache",
    [
        _VP, _VP, _VP, _VP,                # k_cache, v_cache, key, value
        POINTER(c_int32),                  # slot_mapping
        c_int, c_int, c_int, c_int,        # num_tokens, num_kv_heads, head_dim, block_size
        _DTYPE, _STREAM,
    ],
)
_decl(
    "ks_mla_decode",
    [
        _VP, _VP, _VP, _VP,                # out, q_nope, q_pe, kv_cache
        POINTER(c_int32), POINTER(c_int32),  # block_tables, seq_lens
        c_int, c_int, c_int, c_int,        # num_seqs, num_heads, kv_lora_rank, rope_dim
        c_int, c_int,                      # block_size, max_blocks_per_seq
        c_float, _DTYPE, _STREAM,
    ],
)
_decl(
    "ks_flash_attn_backward",
    [
        _VP, _VP, _VP,                     # grad_q, grad_k, grad_v
        _VP, _VP, _VP, _VP,                # grad_out, q, k, v
        _VP, _VP,                          # out, softmax_lse
        c_int, c_int, c_int,               # batch, seqlen_q, seqlen_k
        c_int, c_int, c_int,               # num_heads, num_kv_heads, head_dim
        c_float, c_int, _DTYPE, _STREAM,
    ],
)

# ---- gemm.h ---------------------------------------------------------------

_decl(
    "ks_gemm",
    [
        _VP, _VP, _VP,                     # c, a, b
        c_int64, c_int64, c_int64,         # m, n, k
        c_int, c_int,                      # trans_a, trans_b
        c_int64, c_int64, c_int64,         # lda, ldb, ldc
        c_float, c_float, _DTYPE, _STREAM,  # alpha, beta, dtype, stream
    ],
)
_decl(
    "ks_gemm_bias_act",
    [
        _VP, _VP, _VP, _VP,                # d, a, b, bias
        c_int64, c_int64, c_int64,         # m, n, k
        c_float, c_int, _DTYPE, _STREAM,   # alpha, act, dtype, stream
    ],
)
_decl(
    "ks_gemm_batched",
    [
        _VP, _VP, _VP,                     # c, a, b
        c_int64, c_int64, c_int64, c_int64,  # batch, m, n, k
        c_int, c_int,                      # trans_a, trans_b
        c_int64, c_int64, c_int64,         # stride_a, stride_b, stride_c
        c_float, c_float, _DTYPE, _STREAM,
    ],
)
_decl(
    "ks_gemm_w8a8",
    [
        _VP, _VP, _VP,                     # c, a_i8, b_i8
        POINTER(c_float), POINTER(c_float),  # a_scale, b_scale
        _VP,                               # bias
        c_int64, c_int64, c_int64,         # m, n, k
        c_int, c_int,                      # a_mode, b_mode
        _DTYPE, _STREAM,                   # out_dtype, stream
    ],
)
_decl(
    "ks_gemm_w4a16",
    [
        _VP, _VP, _VP,                     # c, a, b_packed
        _VP, _VP, _VP,                     # scales, zeros, bias
        c_int64, c_int64, c_int64,         # m, n, k
        c_int, _DTYPE, _STREAM,            # group_size, dtype, stream
    ],
)
_decl(
    "ks_gemm_fp8",
    [
        _VP, _VP, _VP,                     # out, a_fp8, b_fp8
        POINTER(c_float), POINTER(c_float),  # a_scale, b_scale
        c_int64, c_int64, c_int64,         # m, n, k
        c_int, c_int,                      # a_mode, b_mode
        _DTYPE, _DTYPE, _STREAM,           # fp8_dtype, out_dtype, stream
    ],
)
_decl(
    "ks_gemm_fp8_blockwise",
    [
        _VP, _VP, _VP,                     # out, a_fp8, b_fp8
        POINTER(c_float), POINTER(c_float),  # a_scale, b_scale
        c_int64, c_int64, c_int64,         # m, n, k
        c_int, c_int,                      # block_n, block_k
        _DTYPE, _DTYPE, _STREAM,           # fp8_dtype, out_dtype, stream
    ],
)

# ---- ssm.h ----------------------------------------------------------------

_decl(
    "ks_causal_conv1d",
    [
        _VP, _VP, _VP, _VP,                # out, x, weight, bias
        c_int, c_int, c_int, c_int,        # batch, dim, seqlen, width
        c_int, _DTYPE, _STREAM,            # silu, dtype, stream
    ],
)
_decl(
    "ks_selective_scan",
    [
        _VP, _VP, _VP,                     # out, x, dt
        _VP, _VP, _VP, _VP,                # A, B, C, D
        _VP, _VP,                          # z, dt_bias
        c_int,                             # delta_softplus
        c_int, c_int, c_int, c_int,        # batch, dim, seqlen, dstate
        _DTYPE, _STREAM,                   # dtype, stream
    ],
)
_decl(
    "ks_selective_scan_update",
    [
        _VP, _VP, _VP, _VP,                # state, out, x, dt
        _VP, _VP, _VP, _VP,                # A, B, C, D
        _VP, _VP,                          # z, dt_bias
        c_int,                             # delta_softplus
        c_int, c_int, c_int,               # batch, dim, dstate
        _DTYPE, _STREAM,                   # dtype, stream
    ],
)

# ---- moe.h ----------------------------------------------------------------

_decl(
    "ks_moe_gate_softmax_topk",
    [
        POINTER(c_float), POINTER(c_int32), _VP,  # out_weights, out_indices, logits
        c_int64, c_int, c_int, c_int,             # num_tokens, num_experts, top_k, renorm
        _DTYPE, _STREAM,
    ],
)
_decl(
    "ks_moe_gate_sigmoid_group_topk",
    [
        POINTER(c_float), POINTER(c_int32), _VP, _VP,  # out_w, out_i, logits, correction_bias
        c_int64, c_int,                                 # num_tokens, num_experts
        c_int, c_int, c_int, c_int,                     # n_group, topk_group, top_k, renorm
        c_float, _DTYPE, _STREAM,                       # routed_scaling_factor, dtype, stream
    ],
)
_decl(
    "ks_moe_compute_permutation",
    [
        POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),  # sorted_ids, offsets, topk_indices
        c_int64, c_int, c_int, _STREAM,
    ],
)
_decl(
    "ks_moe_permute",
    [
        _VP, _VP, POINTER(c_int32),        # permuted, input, sorted_token_ids
        c_int64, c_int, c_int64, _DTYPE, _STREAM,
    ],
)
_decl(
    "ks_moe_unpermute",
    [
        _VP, _VP, POINTER(c_int32), POINTER(c_float),  # out, permuted, sorted_ids, routing_w
        c_int64, c_int, c_int64, _DTYPE, _STREAM,
    ],
)
_decl(
    "ks_moe_grouped_gemm",
    [
        _VP, _VP, _VP, POINTER(c_int32),   # c, a, b, expert_offsets
        c_int, c_int64, c_int64, c_int64,  # num_experts, total_rows, n, k
        _DTYPE, _STREAM,
    ],
)

# ---- rope.h ---------------------------------------------------------------

_decl(
    "ks_rope_inplace",
    [
        _VP, _VP, _VP, _VP,                # q, k, cos, sin
        c_int64, c_int, c_int, c_int,      # num_tokens, num_q_heads, num_kv_heads, head_dim
        c_int, _DTYPE, _STREAM,            # interleaved, dtype, stream
    ],
)
_decl(
    "ks_rope",
    [
        _VP, _VP, _VP, _VP, _VP, _VP,      # q_out, k_out, q, k, cos, sin
        c_int64, c_int, c_int, c_int,      # num_tokens, num_q_heads, num_kv_heads, head_dim
        c_int, _DTYPE, _STREAM,
    ],
)
_decl(
    "ks_rope_gather",
    [
        _VP, _VP, _VP, _VP, POINTER(c_int32),  # q, k, cos_cache, sin_cache, positions
        c_int64, c_int, c_int, c_int,           # num_tokens, num_q_heads, num_kv_heads, head_dim
        c_int, _DTYPE, _STREAM,
    ],
)
_decl(
    "ks_rope_backward",
    [
        _VP, _VP, _VP, _VP,                # grad_q, grad_k, cos, sin
        c_int64, c_int, c_int, c_int,
        c_int, _DTYPE, _STREAM,
    ],
)

# ---- quant.h --------------------------------------------------------------

_decl(
    "ks_quantize_fp8",
    [
        _VP, POINTER(c_float), _VP,        # out, scale, input
        c_int64, c_int64,                  # rows, cols
        _DTYPE, _DTYPE, c_int, _STREAM,    # in_dtype, fp8_dtype, mode, stream
    ],
)
_decl(
    "ks_quantize_fp8_group",
    [
        _VP, POINTER(c_float), _VP,        # out, scale, input
        c_int64, c_int64, c_int,           # rows, cols, group_size
        _DTYPE, _DTYPE, _STREAM,           # in_dtype, fp8_dtype, stream
    ],
)
_decl(
    "ks_dequantize_fp8",
    [
        _VP, _VP, POINTER(c_float),        # out, input, scale
        c_int64, c_int64,
        _DTYPE, _DTYPE, c_int, _STREAM,    # out_dtype, fp8_dtype, mode, stream
    ],
)
_decl(
    "ks_quantize_int8",
    [
        _VP, POINTER(c_float), _VP,        # out, scale, input
        c_int64, c_int64, _DTYPE, c_int, _STREAM,  # rows, cols, in_dtype, mode, stream
    ],
)
_decl(
    "ks_dequantize_int8",
    [
        _VP, _VP, POINTER(c_float),        # out, input, scale
        c_int64, c_int64, _DTYPE, c_int, _STREAM,  # rows, cols, out_dtype, mode, stream
    ],
)
_decl(
    "ks_dequantize_int4",
    [
        _VP, _VP, _VP, _VP,                # out, qweight_packed, scales, zeros
        c_int64, c_int64, c_int,           # k, n, group_size
        _DTYPE, _STREAM,                   # out_dtype, stream
    ],
)

# ---- sampling.h -----------------------------------------------------------

_decl(
    "ks_softmax",
    [_VP, _VP, c_int64, c_int64, c_float, _DTYPE, _STREAM],
)
_decl(
    "ks_log_softmax",
    [_VP, _VP, c_int64, c_int64, _DTYPE, _STREAM],
)
_decl(
    "ks_argmax",
    [POINTER(c_int32), _VP, c_int64, c_int64, _DTYPE, _STREAM],
)
_decl(
    "ks_sample",
    [
        POINTER(c_int32), POINTER(c_float), _VP,  # out_tokens, out_probs, logits
        POINTER(c_float), POINTER(c_int32), POINTER(c_float),  # temps, top_ks, top_ps
        c_int64, c_int64,                          # num_seqs, vocab_size
        c_uint64, c_uint64,                        # seed, philox_offset
        _DTYPE, _STREAM,
    ],
)

# ---- embedding.h ----------------------------------------------------------

_decl(
    "ks_embedding_lookup",
    [
        _VP, _VP, _VP, c_int,              # out, table, indices, indices_i64
        c_int64, c_int64, _DTYPE, _STREAM,  # num_tokens, embed_dim, dtype, stream
    ],
)
_decl(
    "ks_embedding_backward",
    [
        _VP, _VP, _VP, c_int,              # grad_table_fp32, grad_out, indices, indices_i64
        c_int64, c_int64, _DTYPE, _STREAM,
    ],
)

# ---- elementwise.h --------------------------------------------------------

_decl("ks_add", [_VP, _VP, _VP, c_int64, _DTYPE, _STREAM])
_decl("ks_mul", [_VP, _VP, _VP, c_int64, _DTYPE, _STREAM])
_decl("ks_add_residual", [_VP, _VP, c_int64, _DTYPE, _STREAM])
_decl("ks_scale", [_VP, _VP, c_float, c_int64, _DTYPE, _STREAM])
_decl("ks_cast", [_VP, _DTYPE, _VP, _DTYPE, c_int64, _STREAM])
_decl("ks_axpby", [_VP, _VP, c_float, _VP, c_float, c_int64, _DTYPE, _STREAM])

# ---- loss.h ---------------------------------------------------------------

_decl(
    "ks_cross_entropy",
    [
        POINTER(c_float), _VP, _VP, _VP,   # losses, grad_logits, logits, targets
        c_int,                             # targets_i64
        c_int64, c_int64, c_int64,         # num_tokens, vocab, ignore_index
        c_float, _DTYPE, _STREAM,          # label_smoothing, dtype, stream
    ],
)
_decl(
    "ks_fused_linear_cross_entropy",
    [
        POINTER(c_float), _VP, _VP,        # losses, grad_hidden, grad_weight_fp32
        _VP, _VP, _VP, c_int,              # hidden, weight, targets, targets_i64
        c_int64, c_int64, c_int64, c_int64,  # num_tokens, hidden_dim, vocab, ignore_index
        c_float, c_int, _DTYPE, _STREAM,   # label_smoothing, chunk_size, dtype, stream
    ],
)

# ---- optimizer.h ----------------------------------------------------------

_decl(
    "ks_adamw",
    [
        _VP, _VP, _VP,                     # param, master_param, grad
        POINTER(c_float), POINTER(c_float),  # exp_avg, exp_avg_sq
        c_float, c_float, c_float, c_float,  # lr, beta1, beta2, eps
        c_float, c_int64, c_float,         # weight_decay, step, grad_scale
        c_int64, _DTYPE, _STREAM,          # n, dtype, stream
    ],
)
_decl(
    "ks_sgd_momentum",
    [
        _VP, _VP, _VP, POINTER(c_float),   # param, master_param, grad, momentum
        c_float, c_float, c_float,         # lr, momentum_factor, weight_decay
        c_int, c_float, c_int64,           # nesterov, grad_scale, n
        _DTYPE, _STREAM,
    ],
)
_decl(
    "ks_global_grad_norm",
    [
        POINTER(c_float), POINTER(c_void_p), POINTER(c_int64),  # out_norm, grads, sizes
        c_int, _DTYPE, _STREAM,            # num_tensors, dtype, stream
    ],
)


# ---------------------------------------------------------------------------
# Status checking helper
# ---------------------------------------------------------------------------


def status_string(status: int) -> str:
    """Return the symbolic name for a ``ks_status_t`` value."""
    raw = lib.ks_status_string(c_int(status))
    return raw.decode("utf-8", "replace") if raw else f"status {status}"


def last_error_string() -> str:
    """Return the thread-local backend error message (may be empty)."""
    raw = lib.ks_last_error_string()
    return raw.decode("utf-8", "replace") if raw else ""


def check(status: int, func: Optional[str] = None) -> None:
    """Raise :class:`KernelSetError` if ``status`` is not ``KS_SUCCESS``.

    Always reads ``ks_status_string`` (for the symbolic name) and, when the
    failure looks backend-related, ``ks_last_error_string`` (for the detailed
    thread-local message). Pass ``func`` to name the failing call.
    """
    if status == KS_SUCCESS:
        return
    name = status_string(status)
    backend = last_error_string()
    raise KernelSetError(status, name, backend, func)
