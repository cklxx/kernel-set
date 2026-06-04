"""Unit tests for the best-available-backend dispatcher (``kernel_set.dispatch``).

These tests exercise the *selection logic* — they do NOT run any GPU kernel, so
they pass on a no-GPU, no-CUDA, no-shared-library host (this CI). Provider
availability is mocked by monkeypatching the import/arch probes, so we can prove:

* when flash-attn is "available", attention prefill picks it;
* when nothing is available, every op falls back to kernel-set;
* arch gating drops sm90-only providers (DeepGEMM) on an sm89 GPU;
* dtype gating drops providers that don't support the requested dtype;
* ``which`` / ``available`` / ``providers`` introspection is consistent.

Run::

    python3 -m pytest bindings/python/tests/test_dispatch.py -q
"""

from __future__ import annotations

import pytest

# dispatch is import-safe with no torch / CUDA / shared library.
from kernel_set import dispatch
from kernel_set.backends import KERNEL_SET, OP_ORDER, OPS, SGL_KERNEL


# --------------------------------------------------------------------------- #
# Helpers: install a mocked availability map for the probes the selector uses.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with an empty selection cache."""
    dispatch.reset_cache()
    yield
    dispatch.reset_cache()


def _mock_available(monkeypatch, available_libs, sm=None):
    """Make ``available_libs`` (a set of provider import-check snippets OR top
    module names) report as importable; everything else unavailable. ``sm`` sets
    the (mocked) device compute capability for arch gating (None = no GPU)."""
    import kernel_set.dispatch as d
    import kernel_set.backends._probe as probe

    def fake_can_import(check: str) -> bool:
        if check in available_libs:
            return True
        # also match by any extracted module name
        mods = probe._module_names(check)
        return any(m in available_libs for m in mods)

    # patch the symbol the dispatch module imported into its namespace
    monkeypatch.setattr(d, "can_import", fake_can_import)
    # arch gate: resolve_sm is called with the caller's gpu; force our sm when
    # no explicit gpu is passed by patching detect_sm.
    monkeypatch.setattr(probe, "detect_sm", lambda: sm)


# --------------------------------------------------------------------------- #
# Basic structure / introspection.
# --------------------------------------------------------------------------- #
def test_every_op_ends_in_kernel_set_fallback():
    for op in OP_ORDER:
        chain = dispatch.providers(op)
        assert chain[-1] == KERNEL_SET, f"{op} must end with kernel-set fallback"
        assert chain.count(KERNEL_SET) == 1


def test_ops_listing_matches_registry():
    assert dispatch.ops() == list(OP_ORDER)
    for op in OP_ORDER:
        assert op in OPS


def test_unknown_op_raises():
    with pytest.raises(KeyError):
        dispatch.which("not_a_real_op")
    with pytest.raises(KeyError):
        dispatch.providers("not_a_real_op")


# --------------------------------------------------------------------------- #
# Fallback: nothing installed -> kernel-set everywhere.
# --------------------------------------------------------------------------- #
def test_nothing_available_falls_back_to_kernel_set(monkeypatch):
    _mock_available(monkeypatch, available_libs=set(), sm=90)
    for op in OP_ORDER:
        assert dispatch.which(op) == KERNEL_SET
    avail = dispatch.available()
    for op in OP_ORDER:
        assert avail[op] == [KERNEL_SET]


# --------------------------------------------------------------------------- #
# flash-attn available -> attention prefill picks it (rank 1).
# --------------------------------------------------------------------------- #
def test_flash_attn_wins_attention_prefill(monkeypatch):
    _mock_available(monkeypatch, available_libs={"flash_attn"}, sm=90)
    assert dispatch.which("attention_prefill") == "flash-attn"
    # other ops with no provider installed still fall back to kernel-set
    assert dispatch.which("rmsnorm") == KERNEL_SET


def test_rank_order_respected_when_multiple_available(monkeypatch):
    # flashinfer (rank 1) and liger (rank 3) both available -> rank 1 wins.
    _mock_available(
        monkeypatch,
        available_libs={"flashinfer", "liger_kernel"},
        sm=90,
    )
    assert dispatch.which("rmsnorm") == "flashinfer"
    # If only liger is available, it is chosen over the ks fallback.
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"liger_kernel"}, sm=90)
    assert dispatch.which("rmsnorm") == "liger"


# --------------------------------------------------------------------------- #
# Arch gating: DeepGEMM (sm90) is dropped on sm89; lower-rank provider or the
# kernel-set fallback is chosen instead.
# --------------------------------------------------------------------------- #
def test_arch_gating_drops_sm90_provider_on_sm89(monkeypatch):
    # deep_gemm (rank1, sm90) AND torch-scaled-mm (rank2, sm89) both installed.
    _mock_available(monkeypatch, available_libs={"deep_gemm", "torch"}, sm=89)
    # sm89 cannot run DeepGEMM -> falls through to torch-scaled-mm (sm89 ok).
    assert dispatch.which("fp8_gemm", dtype="fp8") == "torch-scaled-mm"

    # On sm90 the same install picks DeepGEMM (rank 1).
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"deep_gemm", "torch"}, sm=90)
    assert dispatch.which("fp8_gemm", dtype="fp8") == "deep_gemm"


def test_arch_gating_explicit_gpu_overrides_detection(monkeypatch):
    # Detected device is sm90, but caller explicitly targets an L4 (sm89).
    _mock_available(monkeypatch, available_libs={"deep_gemm", "torch"}, sm=90)
    assert dispatch.which("fp8_gemm", gpu="l4", dtype="fp8") == "torch-scaled-mm"
    assert dispatch.which("fp8_gemm", gpu="h100", dtype="fp8") == "deep_gemm"


def test_liger_dropped_on_sm75(monkeypatch):
    # liger needs sm80; on a T4 (sm75) it is gated out -> kernel-set fallback.
    _mock_available(monkeypatch, available_libs={"liger_kernel"}, sm=75)
    assert dispatch.which("rmsnorm") == KERNEL_SET
    # flashinfer (sm75) would be fine though:
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"flashinfer"}, sm=75)
    assert dispatch.which("rmsnorm") == "flashinfer"


# --------------------------------------------------------------------------- #
# dtype gating: flash-attn supports only fp16/bf16, not fp8.
# --------------------------------------------------------------------------- #
def test_dtype_gating(monkeypatch):
    _mock_available(monkeypatch, available_libs={"flash_attn"}, sm=90)
    assert dispatch.which("attention_prefill", dtype="bf16") == "flash-attn"
    # flash-attn dtypes string is "fp16, bf16"; fp8 isn't covered -> falls back.
    assert dispatch.which("attention_prefill", dtype="fp8") == KERNEL_SET


# --------------------------------------------------------------------------- #
# available() reflects gating and rank order.
# --------------------------------------------------------------------------- #
def test_available_lists_selectable_in_rank_order(monkeypatch):
    _mock_available(
        monkeypatch,
        available_libs={"flashinfer", "vllm", "liger_kernel"},
        sm=90,
    )
    avail = dispatch.available()
    # rmsnorm chain: flashinfer(1), vllm(2), liger(3), kernel-set
    assert avail["rmsnorm"] == ["flashinfer", "vllm", "liger", KERNEL_SET]


def test_available_arch_filters(monkeypatch):
    # On sm75, liger (sm80) drops out of the rmsnorm chain.
    _mock_available(
        monkeypatch,
        available_libs={"flashinfer", "vllm", "liger_kernel"},
        sm=75,
    )
    avail = dispatch.available()
    assert "liger" not in avail["rmsnorm"]
    assert avail["rmsnorm"] == ["flashinfer", "vllm", KERNEL_SET]


# --------------------------------------------------------------------------- #
# Backend handle + chain() detail view.
# --------------------------------------------------------------------------- #
def test_backend_handle(monkeypatch):
    _mock_available(monkeypatch, available_libs={"flashinfer"}, sm=90)
    b = dispatch.Backend("rmsnorm")
    assert b.name == "flashinfer"
    assert b.op == "rmsnorm"
    assert b.provider.rank == 1


def test_chain_detail_view(monkeypatch):
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=90)
    rows = dispatch.chain("rmsnorm")
    names = [r["name"] for r in rows]
    assert names == ["flashinfer", "sgl-kernel", "vllm", "liger", KERNEL_SET]
    sel = {r["name"]: r["selectable"] for r in rows}
    assert sel["flashinfer"] is False   # not installed in this mock
    assert sel["sgl-kernel"] is False   # not installed in this mock
    assert sel["vllm"] is True
    assert sel[KERNEL_SET] is True      # always selectable
    # which() must agree with the first selectable in chain()
    first_selectable = next(r["name"] for r in rows if r["selectable"])
    assert dispatch.which("rmsnorm") == first_selectable


def test_which_alias():
    assert dispatch.which_provider is dispatch.which


# --------------------------------------------------------------------------- #
# sgl-kernel: the hard-op alignment target. Ranked #1 for the MoE gate ops and
# grouped-MoE; competitive elsewhere. Verify selection + arch gating.
# --------------------------------------------------------------------------- #
def test_sgl_kernel_wins_moe_ops_when_available(monkeypatch):
    # On an sm80 Ampere device, sgl-kernel (rank 1) wins the MoE specialty ops.
    _mock_available(monkeypatch, available_libs={"sgl_kernel"}, sm=80)
    assert dispatch.which("moe") == SGL_KERNEL
    assert dispatch.which("moe_gate") == SGL_KERNEL
    assert dispatch.which("moe_group_gate") == SGL_KERNEL


def test_sgl_kernel_beats_vllm_for_moe(monkeypatch):
    # Both sgl-kernel (rank 1) and vllm (rank 2) installed -> sgl-kernel wins.
    _mock_available(monkeypatch, available_libs={"sgl_kernel", "vllm"}, sm=90)
    assert dispatch.which("moe") == SGL_KERNEL
    assert dispatch.which("moe_gate") == SGL_KERNEL


def test_sgl_kernel_competitive_rank_on_norms(monkeypatch):
    # flashinfer (rank1) present -> it wins; sgl-kernel is rank 2.
    _mock_available(
        monkeypatch, available_libs={"flashinfer", "sgl_kernel"}, sm=90)
    assert dispatch.which("rmsnorm") == "flashinfer"
    # Only sgl-kernel present -> it is chosen over the ks fallback.
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"sgl_kernel"}, sm=90)
    assert dispatch.which("rmsnorm") == SGL_KERNEL
    assert dispatch.which("fused_add_rmsnorm") == SGL_KERNEL
    assert dispatch.which("gemma_rmsnorm") == SGL_KERNEL
    assert dispatch.which("rope") == SGL_KERNEL
    assert dispatch.which("swiglu") == SGL_KERNEL


def test_sgl_kernel_mla_and_attention_need_sm90(monkeypatch):
    # FlashMLA / FA3 paths are sm90; gated out on an sm89 (L4) device.
    _mock_available(monkeypatch, available_libs={"sgl_kernel"}, sm=89)
    assert dispatch.which("mla_decode") == KERNEL_SET
    assert dispatch.which("attention_decode") == KERNEL_SET
    # On sm90 the same install selects sgl-kernel for MLA (rank 1).
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"sgl_kernel"}, sm=90)
    assert dispatch.which("mla_decode") == SGL_KERNEL


def test_sgl_kernel_int8_gemm_rank1(monkeypatch):
    _mock_available(monkeypatch, available_libs={"sgl_kernel"}, sm=80)
    assert dispatch.which("int8_gemm", dtype="int8") == SGL_KERNEL


def test_sgl_kernel_fp8_gemm_needs_sm90(monkeypatch):
    # sgl-kernel fp8_scaled_mm is sm90; below it the chain falls through.
    _mock_available(monkeypatch, available_libs={"sgl_kernel"}, sm=89)
    assert dispatch.which("fp8_gemm", dtype="fp8") == KERNEL_SET
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"sgl_kernel"}, sm=90)
    assert dispatch.which("fp8_gemm", dtype="fp8") == SGL_KERNEL


def test_sgl_kernel_appears_in_chains():
    # Static chain membership (regardless of availability) for the wired ops.
    for op in ("moe", "moe_gate", "moe_group_gate", "mla_decode", "int8_gemm",
               "rmsnorm", "fused_add_rmsnorm", "gemma_rmsnorm", "rope",
               "swiglu", "fp8_gemm", "attention_prefill", "attention_decode",
               "sampling"):
        assert SGL_KERNEL in dispatch.providers(op), op


def test_sgl_kernel_import_check_is_sgl_kernel():
    # Every sgl-kernel provider probes `import sgl_kernel`.
    for op in OP_ORDER:
        for p in OPS[op].providers:
            if p.name == SGL_KERNEL:
                assert p.import_check == "import sgl_kernel", op


# --------------------------------------------------------------------------- #
# Import-safety: dispatch must import and introspect with no torch needed for
# selection. (The module imported at top already proves import works.)
# --------------------------------------------------------------------------- #
def test_introspection_runs_without_gpu():
    # No mocking: whatever is actually installed on the host. Must not raise and
    # every op must still resolve to *some* provider (at worst kernel-set).
    avail = dispatch.available()
    assert set(avail) == set(OP_ORDER)
    for op in OP_ORDER:
        name = dispatch.which(op)
        assert name in dispatch.providers(op)
