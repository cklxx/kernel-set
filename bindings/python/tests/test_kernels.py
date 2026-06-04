"""Correctness tests: kernel_set kernels vs. torch references (CUDA).

These tests exercise the Python binding's tensor-convenience API (see
``bindings/python/README.md``) — every call passes torch CUDA tensors, so
shape/dtype/stream are inferred — and compare each kernel's output against the
equivalent torch op within a dtype-appropriate tolerance.

The whole module SKIPS unless **both** are available at import time:

* a CUDA-enabled torch build with a visible device, and
* the prebuilt ``kernel_set`` shared library (``import kernel_set`` succeeds and
  reports the ``cuda`` backend).

Run from anywhere once the binding is importable and the .so is locatable
(see the binding README for ``KERNEL_SET_LIB`` / ``KERNEL_SET_LIB_DIR``)::

    pip install -e ./bindings/python
    pytest bindings/python/tests/test_kernels.py -v
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Import + capability gating. Any failure here skips the entire module rather
# than erroring, so the suite is a no-op on machines without CUDA/torch/the .so.
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="torch is required for these tests")

if not torch.cuda.is_available():
    pytest.skip("CUDA device not available", allow_module_level=True)

ks = pytest.importorskip("kernel_set", reason="kernel_set binding not importable")

try:
    _BACKEND = ks.backend_name()
except Exception as exc:  # pragma: no cover - library failed to load
    pytest.skip(f"kernel_set library not loadable: {exc}", allow_module_level=True)

if _BACKEND != "cuda":
    pytest.skip(
        f"kernel_set backend is {_BACKEND!r}, need a CUDA build",
        allow_module_level=True,
    )

DEVICE = "cuda"

# Floating dtypes the kernels support. bf16 needs sm_80+; skip it otherwise.
FLOAT_DTYPES = [torch.float32, torch.float16]
if torch.cuda.get_device_capability(0)[0] >= 8:
    FLOAT_DTYPES.append(torch.bfloat16)

# Per-dtype tolerances (atol, rtol). fp16/bf16 are loose to cover the storage
# rounding the kernels apply on read and write; both compute in fp32.
TOL = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.float16: dict(atol=3e-3, rtol=3e-3),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}


def _id(dt):
    return str(dt).replace("torch.", "")


def assert_close(got, ref, dtype):
    """Compare in fp32 with the dtype's tolerance."""
    torch.cuda.synchronize()
    t = TOL[dtype]
    torch.testing.assert_close(
        got.float(), ref.float(), atol=t["atol"], rtol=t["rtol"]
    )


def randn(*shape, dtype):
    return torch.randn(*shape, device=DEVICE, dtype=dtype)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_rms_norm(dtype):
    rows, cols, eps = 8, 512, 1e-6
    x = randn(rows, cols, dtype=dtype)
    w = randn(cols, dtype=dtype)
    out = torch.empty_like(x)

    ks.norm.rms_norm(out, x, w, eps=eps)

    xf = x.float()
    inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    ref = (xf * inv_rms * w.float()).to(dtype)
    assert_close(out, ref, dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "nobias"])
def test_layer_norm(dtype, with_bias):
    rows, cols, eps = 6, 384, 1e-5
    x = randn(rows, cols, dtype=dtype)
    w = randn(cols, dtype=dtype)
    b = randn(cols, dtype=dtype) if with_bias else None
    out = torch.empty_like(x)

    ks.norm.layer_norm(out, x, w, bias=b, eps=eps)

    # torch F.layer_norm uses biased variance, matching layer_norm.cu.
    ref = torch.nn.functional.layer_norm(
        x.float(), (cols,), weight=w.float(),
        bias=(b.float() if with_bias else None), eps=eps,
    ).to(dtype)
    assert_close(out, ref, dtype)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_silu(dtype):
    x = randn(4, 1024, dtype=dtype)
    out = torch.empty_like(x)
    ks.activation.silu(out, x)
    ref = torch.nn.functional.silu(x.float()).to(dtype)
    assert_close(out, ref, dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_gelu_erf(dtype):
    x = randn(4, 1024, dtype=dtype)
    out = torch.empty_like(x)
    ks.activation.gelu(out, x, tanh_approx=False)
    ref = torch.nn.functional.gelu(x.float(), approximate="none").to(dtype)
    assert_close(out, ref, dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_gelu_tanh(dtype):
    x = randn(4, 1024, dtype=dtype)
    out = torch.empty_like(x)
    ks.activation.gelu(out, x, tanh_approx=True)
    ref = torch.nn.functional.gelu(x.float(), approximate="tanh").to(dtype)
    assert_close(out, ref, dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_relu(dtype):
    x = randn(4, 1024, dtype=dtype)
    out = torch.empty_like(x)
    ks.activation.relu(out, x)
    ref = torch.relu(x.float()).to(dtype)
    assert_close(out, ref, dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_swiglu(dtype):
    rows, inter = 8, 1024
    gate = randn(rows, inter, dtype=dtype)
    up = randn(rows, inter, dtype=dtype)
    out = torch.empty_like(gate)
    ks.activation.swiglu(out, gate, up)
    ref = (torch.nn.functional.silu(gate.float()) * up.float()).to(dtype)
    assert_close(out, ref, dtype)


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_add(dtype):
    a = randn(64, 128, dtype=dtype)
    b = randn(64, 128, dtype=dtype)
    out = torch.empty_like(a)
    ks.elementwise.add(out, a, b)
    assert_close(out, (a.float() + b.float()).to(dtype), dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_mul(dtype):
    a = randn(64, 128, dtype=dtype)
    b = randn(64, 128, dtype=dtype)
    out = torch.empty_like(a)
    ks.elementwise.mul(out, a, b)
    assert_close(out, (a.float() * b.float()).to(dtype), dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_scale(dtype):
    x = randn(64, 128, dtype=dtype)
    out = torch.empty_like(x)
    ks.elementwise.scale(out, x, 1.75)
    assert_close(out, (x.float() * 1.75).to(dtype), dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_axpby(dtype):
    a = randn(64, 128, dtype=dtype)
    b = randn(64, 128, dtype=dtype)
    out = torch.empty_like(a)
    alpha, beta = 0.5, -1.25
    ks.elementwise.axpby(out, a, alpha, b, beta)
    assert_close(out, (a.float() * alpha + b.float() * beta).to(dtype), dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_add_residual(dtype):
    residual = randn(64, 128, dtype=dtype)
    x = randn(64, 128, dtype=dtype)
    ref = (residual.float() + x.float()).to(dtype)
    ks.elementwise.add_residual(residual, x)  # in-place into residual
    assert_close(residual, ref, dtype)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
def test_embedding_lookup(dtype, idx_dtype):
    vocab, embed, tokens = 128, 256, 40
    table = randn(vocab, embed, dtype=dtype)
    indices = torch.randint(0, vocab, (tokens,), device=DEVICE, dtype=idx_dtype)
    out = torch.empty(tokens, embed, device=DEVICE, dtype=dtype)

    ks.embedding.embedding_lookup(
        out, table, indices, num_tokens=tokens, embed_dim=embed
    )

    ref = table.index_select(0, indices.to(torch.int64))
    assert_close(out, ref, dtype)


# ---------------------------------------------------------------------------
# Softmax / sampling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
@pytest.mark.parametrize("temperature", [1.0, 0.7], ids=["T1.0", "T0.7"])
def test_softmax(dtype, temperature):
    rows, cols = 6, 1000
    x = randn(rows, cols, dtype=dtype)
    out = torch.empty_like(x)
    ks.sampling.softmax(out, x, rows=rows, cols=cols, temperature=temperature)
    ref = torch.softmax(x.float() / temperature, dim=-1).to(dtype)
    assert_close(out, ref, dtype)
    # Rows of the softmax sum to ~1.
    sums = out.float().sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_log_softmax(dtype):
    rows, cols = 4, 800
    x = randn(rows, cols, dtype=dtype)
    out = torch.empty_like(x)
    ks.sampling.log_softmax(out, x, rows=rows, cols=cols)
    ref = torch.log_softmax(x.float(), dim=-1).to(dtype)
    assert_close(out, ref, dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=_id)
def test_argmax(dtype):
    seqs, vocab = 5, 1024
    # Distinct rows so the argmax is unambiguous (tie-break differences moot).
    logits = randn(seqs, vocab, dtype=dtype)
    tokens = torch.empty(seqs, device=DEVICE, dtype=torch.int32)
    ks.sampling.argmax(tokens, logits, num_seqs=seqs, vocab_size=vocab)
    torch.cuda.synchronize()
    ref = logits.float().argmax(dim=-1).to(torch.int32)
    torch.testing.assert_close(tokens, ref)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
