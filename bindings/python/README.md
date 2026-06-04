# kernel_set — Python bindings

Pure-`ctypes` Python bindings for **kernel-set**, a library of high-performance
CUDA/HIP kernels for LLM inference and training (norm, activation, attention,
gemm, moe, rope, quant, sampling, embedding, elementwise, loss, optimizer).

* **No compilation on install.** The package `dlopen`s a *prebuilt* shared
  library (`libkernel_set.so` / `libkernel_set.dylib` / `kernel_set.dll`).
* **torch-friendly, but torch-optional.** Every wrapper accepts a
  `torch.Tensor` *or* a raw integer device pointer. `torch` is only needed for
  the tensor-convenience paths.
* **Full ABI coverage** — all ~60 C functions, plus runtime/device/stream/memory
  helpers and a `torch <-> ks_dtype` mapping.

## Install

```bash
pip install ./bindings/python          # editable: pip install -e ./bindings/python
```

The binding has **no required dependencies**. Install torch for the ergonomic
tensor paths:

```bash
pip install "kernel_set[torch]"
```

## Locating the shared library

At import time the package searches, in order:

1. **`KERNEL_SET_LIB`** — full path *or* bare filename of the library
   (e.g. `export KERNEL_SET_LIB=/opt/kernel_set/lib/libkernel_set.so`).
2. **`KERNEL_SET_LIB_DIR`** — a directory to search for the platform basename.
3. Next to the installed package (`kernel_set/`, `kernel_set/lib/`) — for wheels
   that vendor the `.so`.
4. Repo build trees relative to the source checkout (`build/`, `build/lib/`,
   `lib/`, `install/lib/`).
5. Common system paths (`/usr/local/lib`, `/usr/lib`, `/opt/kernel_set/lib`) and
   the platform loader's own search path (`LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`,
   `PATH`).

If none is found, a clear `OSError` lists every path tried.

## Usage

```python
import torch
import kernel_set as ks

print(ks.version(), ks.backend_name())     # "0.1.0" "cuda"

# --- RMSNorm (dtype, shape, and stream all inferred from the tensors) -------
x   = torch.randn(8, 4096, device="cuda", dtype=torch.float16)
w   = torch.ones(4096,     device="cuda", dtype=torch.float16)
out = torch.empty_like(x)
ks.norm.rms_norm(out, x, w, eps=1e-6)

# --- SwiGLU MLP gate ------------------------------------------------------
gate = torch.randn(8, 11008, device="cuda", dtype=torch.bfloat16)
up   = torch.randn(8, 11008, device="cuda", dtype=torch.bfloat16)
y    = torch.empty_like(gate)
ks.activation.swiglu(y, gate, up)

# --- Greedy decode --------------------------------------------------------
logits = torch.randn(2, 32000, device="cuda", dtype=torch.float16)
tokens = torch.empty(2, device="cuda", dtype=torch.int32)
ks.sampling.argmax(tokens, logits, num_seqs=2, vocab_size=32000)

torch.cuda.synchronize()
print(tokens)
```

### Raw device pointers (no torch)

```python
import kernel_set as ks
from kernel_set import DType

rows, cols = 8, 4096
nbytes = rows * cols * 2          # float16

x   = ks.runtime.malloc_device(nbytes)
w   = ks.runtime.malloc_device(cols * 2)
out = ks.runtime.malloc_device(nbytes)

# ...fill x/w via ks.runtime.memcpy(..., ks.MemcpyKind.HOST_TO_DEVICE)...

ks.norm.rms_norm(out, x, w, rows=rows, cols=cols, eps=1e-6,
                 dtype=DType.F16, stream=0)
ks.runtime.stream_synchronize(0)

for p in (x, w, out):
    ks.runtime.free_device(p)
```

## Tensor-library interop (passing GPU pointers)

The wrappers extract device pointers and streams from the ecosystem's tensor
lib for you. The rules:

* **Pointers** come from `tensor.data_ptr()`. Tensors must be **CUDA** and
  **contiguous** (call `.contiguous()` first); both are validated and raise a
  clear `ValueError` otherwise. Any object exposing `data_ptr()` (e.g. CuPy
  arrays, Numba device arrays) works too.
* **Streams**: if you don't pass `stream=`, the wrappers default to
  `torch.cuda.current_stream(tensor.device).cuda_stream` for torch inputs so
  launches are ordered against your torch work. For raw-pointer inputs the
  default is the **default stream** (`0`). You can override with
  `stream=<int>`, `stream=torch.cuda.Stream(...)`, `stream="current"`, or a
  `kernel_set.runtime.Stream`.
* **dtype**: inferred from the torch tensor's `.dtype` via the
  `TORCH_TO_KS` map; pass `dtype=ks.DType.*` for raw pointers or to override.

```python
import torch, kernel_set as ks

a = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
b = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
c = torch.empty(256, 128, device="cuda", dtype=torch.bfloat16)

# C = A @ B  (M, N, K). Run on a custom stream:
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    ks.gemm.gemm(c, a, b, m=256, n=128, k=512, stream=s)
s.synchronize()
```

### dtype mapping (`torch <-> ks_dtype`)

| torch dtype       | `ks.DType`   |
|-------------------|--------------|
| `float32`         | `F32`        |
| `float16`         | `F16`        |
| `bfloat16`        | `BF16`       |
| `float64`         | `F64`        |
| `float8_e4m3fn`   | `F8E4M3`     |
| `float8_e5m2`     | `F8E5M2`     |
| `int64`           | `I64`        |
| `int32`           | `I32`        |
| `int8`            | `I8`         |
| `uint8`           | `U8`         |

(`ks.DType.I4` is packed int4 with no native torch dtype — pass raw pointers.)

```python
ks.dtype_to_ks(torch.bfloat16)      # -> ks.DType.BF16
ks.ks_to_torch_dtype(ks.DType.F16)  # -> torch.float16
```

## Error handling

Every C call returns a `ks_status_t`; non-zero statuses raise
`kernel_set.KernelSetError`, which carries the symbolic status name (from
`ks_status_string`) and the thread-local backend message (from
`ks_last_error_string`):

```python
try:
    ks.norm.rms_norm(out, x, w, eps=1e-6)
except ks.KernelSetError as e:
    print(e.status, e.status_name, e.backend_message)
```

## Low-level FFI

For full control, the raw ctypes layer is available — every C function with its
`argtypes`/`restype` set:

```python
from kernel_set._lib import lib, check, KS_DTYPE_F16
check(lib.ks_rms_norm(out_ptr, x_ptr, w_ptr, rows, cols, 1e-6, KS_DTYPE_F16, 0),
      "ks_rms_norm")
```

## API surface

| Module                    | Functions |
|---------------------------|-----------|
| `kernel_set.runtime`      | `version`, `backend_name`, `dtype_size_bits`, `dtype_name`, `device_count`, `get/set_device`, `get_device_properties`, `Stream`, `stream_create/destroy/synchronize`, `malloc_device`, `free_device`, `memcpy`, `memset_device` |
| `kernel_set.norm`         | `rms_norm`, `rms_norm_residual`, `layer_norm`, `rms_norm_backward`, `layer_norm_backward` |
| `kernel_set.activation`   | `silu`, `gelu`, `relu`, `swiglu`, `swiglu_packed`, `geglu`, `swiglu_backward` |
| `kernel_set.attention`    | `flash_attn`, `flash_attn_varlen`, `paged_attn_decode`, `reshape_and_cache`, `mla_decode`, `flash_attn_backward` |
| `kernel_set.gemm`         | `gemm`, `gemm_bias_act`, `gemm_batched`, `gemm_w8a8`, `gemm_w4a16` |
| `kernel_set.moe`          | `gate_softmax_topk`, `gate_sigmoid_group_topk`, `compute_permutation`, `permute`, `unpermute`, `grouped_gemm` |
| `kernel_set.rope`         | `rope_inplace`, `rope`, `rope_gather`, `rope_backward` |
| `kernel_set.quant`        | `quantize_fp8`, `dequantize_fp8`, `quantize_int8`, `dequantize_int8`, `dequantize_int4` |
| `kernel_set.sampling`     | `softmax`, `log_softmax`, `argmax`, `sample` |
| `kernel_set.embedding`    | `embedding_lookup`, `embedding_backward` |
| `kernel_set.elementwise`  | `add`, `mul`, `add_residual`, `scale`, `cast`, `axpby` |
| `kernel_set.loss`         | `cross_entropy`, `fused_linear_cross_entropy` |
| `kernel_set.optimizer`    | `adamw`, `sgd_momentum`, `global_grad_norm` |

Enums: `ks.DType`, `ks.Activation`, `ks.QuantMode`, `ks.MemcpyKind`, `ks.Status`.
