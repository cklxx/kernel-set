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

import sys
import types

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


def test_tf32_arch_gating():
    from kernel_set.backends import dtype_arch_ok, normalize_dtype

    assert normalize_dtype("tf32") == "tf32"
    assert dtype_arch_ok("tf32", 75) is False
    assert dtype_arch_ok("tf32", 80) is True
    assert dtype_arch_ok("fp32", 75) is True


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
def test_sgl_kernel_wins_moe_gate_ops_when_available(monkeypatch):
    # sgl-kernel (rank 1) wins the MoE *gate* specialty ops on any sm80+ arch
    # (its fused softmax/sigmoid group-topk gate is best-in-class everywhere).
    _mock_available(monkeypatch, available_libs={"sgl_kernel"}, sm=80)
    assert dispatch.which("moe_gate") == SGL_KERNEL
    assert dispatch.which("moe_group_gate") == SGL_KERNEL
    # The sgl-kernel grouped-MoE GEMM (CUTLASS blockwise-fp8) is sm90, so on an
    # Ampere sm80 host with nothing else it correctly drops to kernel-set.
    assert dispatch.which("moe") == KERNEL_SET


def test_deepgemm_wins_moe_grouped_gemm_on_hopper(monkeypatch):
    # On sm90 DeepGEMM grouped FP8 is the rank-1 MoE GEMM (DeepSeek-V3 path).
    _mock_available(monkeypatch, available_libs={"deep_gemm"}, sm=90)
    assert dispatch.which("moe", dtype="fp8") == "deep_gemm"
    # sgl-kernel grouped GEMM is the rank-2 alignment target on sm90.
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"sgl_kernel"}, sm=90)
    assert dispatch.which("moe", dtype="fp8") == SGL_KERNEL
    # DeepGEMM + sgl + vllm all present on sm90 -> DeepGEMM (rank 1) wins.
    dispatch.reset_cache()
    _mock_available(
        monkeypatch, available_libs={"deep_gemm", "sgl_kernel", "vllm"}, sm=90)
    assert dispatch.which("moe", dtype="fp8") == "deep_gemm"


def test_vllm_fused_moe_on_ampere(monkeypatch):
    # Ampere/Ada have no FP8 hw: DeepGEMM/sgl grouped GEMM gate out (sm90); the
    # optimal grouped path is vLLM Triton fused_experts (sm80+).
    _mock_available(
        monkeypatch, available_libs={"deep_gemm", "sgl_kernel", "vllm"}, sm=80)
    assert dispatch.which("moe") == "vllm"
    # Nothing external -> kernel-set portable fallback.
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs=set(), sm=80)
    assert dispatch.which("moe") == KERNEL_SET


def test_moe_gate_vllm_rank2_wired(monkeypatch):
    # The vLLM rank-2 gate adapters are now wired (were call=None): with only
    # vLLM present they are selected over the kernel-set fallback.
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=80)
    assert dispatch.which("moe_gate") == "vllm"
    assert dispatch.which("moe_group_gate") == "vllm"


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


def test_sgl_kernel_int8_gemm_selected_when_only_sgl(monkeypatch):
    # sgl-kernel int8_scaled_mm is the rank-2 alignment target; when it is the
    # only INT8 provider installed it still wins over the kernel-set fallback.
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


def test_sampling_dispatch_contract_returns_int_token_ids(monkeypatch):
    torch = pytest.importorskip("torch")
    from kernel_set.backends import _registry

    probs = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
                         dtype=torch.float32)
    sampling_providers = {p.name: p for p in OPS["sampling"].providers}

    def assert_ids(ids):
        assert tuple(ids.shape) == (2,)
        assert ids.dtype in (torch.int32, torch.int64)

    # FlashInfer selected: provider returns sampled ids.
    monkeypatch.setattr(
        sampling_providers["flashinfer"], "call",
        lambda *_a, **_k: torch.tensor([1, 2], dtype=torch.int32))
    _mock_available(monkeypatch, available_libs={"flashinfer.sampling"}, sm=90)
    assert dispatch.which("sampling", dtype="fp32") == "flashinfer"
    assert_ids(dispatch.sampling(probs, top_k=1, _dtype="fp32"))

    # SGL selected: real adapter renorms then samples token ids.
    dispatch.reset_cache()

    class FakeSGL:
        def top_k_renorm_prob(self, _probs, _top_k):
            return torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                dtype=torch.float32)

        def top_p_renorm_prob(self, probs, _top_p):
            return probs

    monkeypatch.setattr(
        _registry, "_imp",
        lambda mod: FakeSGL() if mod == "sgl_kernel" else None)
    _mock_available(monkeypatch, available_libs={"sgl_kernel"}, sm=90)
    assert dispatch.which("sampling", dtype="fp32") == SGL_KERNEL
    ids = dispatch.sampling(probs, top_k=1, _dtype="fp32")
    assert_ids(ids)
    assert ids.tolist() == [1, 2]

    # kernel-set fallback selected: provider returns sampled ids.
    dispatch.reset_cache()
    monkeypatch.setattr(
        sampling_providers[KERNEL_SET], "call",
        lambda *_a, **_k: torch.tensor([2, 0], dtype=torch.int32))
    _mock_available(monkeypatch, available_libs=set(), sm=90)
    assert dispatch.which("sampling", dtype="fp32") == KERNEL_SET
    assert_ids(dispatch.sampling(probs, top_k=1, _dtype="fp32"))


def test_import_probe_checks_from_import_attributes(monkeypatch):
    from kernel_set.backends import _probe

    _probe._IMPORT_CACHE.clear()
    mod = types.ModuleType("fake_kernel_set_probe_mod")
    monkeypatch.setitem(sys.modules, "fake_kernel_set_probe_mod", mod)
    assert _probe.can_import(
        "from fake_kernel_set_probe_mod import missing_attr") is False

    _probe._IMPORT_CACHE.clear()
    mod.present_attr = object()
    assert _probe.can_import(
        "from fake_kernel_set_probe_mod import present_attr") is True


def test_new_flashinfer_import_checks_are_specific():
    assert next(p for p in OPS["nvfp4_gemm"].providers
                if p.name == "flashinfer").import_check == \
        "from flashinfer.gemm import mm_fp4"
    assert next(p for p in OPS["mxfp4_gemm"].providers
                if p.name == "flashinfer").import_check == \
        "from flashinfer.gemm import mm_fp4"
    assert next(p for p in OPS["sampling"].providers
                if p.name == "flashinfer").import_check == \
        "import flashinfer.sampling"


def test_mxfp4_providers_are_blackwell_only():
    providers = {p.name: p for p in OPS["mxfp4_gemm"].providers}
    assert providers["flashinfer"].min_sm == 100
    assert providers["vllm"].min_sm == 100
    assert providers["torchao"].min_sm == 100


# =========================================================================== #
# COMPUTE-BOUND OPTIMAL SELECTION (docs/OPTIMAL_SELECTION.md).
#
# Strategy: for every compute-bound op the dispatcher must (a) pick an *external*
# best-in-class provider whenever one is installed & arch/dtype-compatible, and
# (b) fall back to kernel-set ONLY when no external provider is available for the
# arch. kernel-set must NEVER be preferred over an available external provider on
# these ops, and must always be the LAST entry in the chain.
# =========================================================================== #

# The compute-bound ops kernel-set adopts external kernels for. (Memory-bound
# ops — norm/rope/act/sampling — are validated separately above; ks is SOTA-class
# there and is a legitimate competitive provider, not merely a fallback.)
COMPUTE_BOUND_OPS = [
    "gemm", "fp8_gemm", "int8_gemm", "w4a16",
    "attention_prefill", "attention_decode", "mla_decode",
    "moe", "selective_scan", "causal_conv1d",
]


def test_compute_bound_kernel_set_is_always_last():
    # For every compute-bound op, kernel-set must be the final (fallback) entry.
    for op in COMPUTE_BOUND_OPS:
        chain = dispatch.providers(op)
        assert chain[-1] == KERNEL_SET, op
        assert chain.count(KERNEL_SET) == 1, op
        # ...and it must NOT be the rank-1 entry (an external provider leads).
        assert chain[0] != KERNEL_SET, op


def test_compute_bound_prefers_external_when_available(monkeypatch):
    # With the rank-1 external provider installed (on a suitable arch), the
    # dispatcher must pick it — never kernel-set.
    cases = [
        # (op, libs, sm, dtype, expected_external)
        ("gemm", {"torch"}, 80, "bf16", "torch"),
        ("fp8_gemm", {"deep_gemm"}, 90, "fp8", "deep_gemm"),
        ("fp8_gemm", {"torch"}, 89, "fp8", "torch-scaled-mm"),
        ("int8_gemm", {"vllm"}, 80, "int8", "vllm"),
        ("int8_gemm", {"sgl_kernel"}, 80, "int8", SGL_KERNEL),
        ("w4a16", {"vllm"}, 80, "int4", "vllm-marlin"),
        ("w4a16", {"vllm"}, 90, "int4", "vllm-machete"),
        ("attention_prefill", {"flash_attn"}, 80, "bf16", "flash-attn"),
        ("attention_decode", {"flashinfer"}, 80, "fp16", "flashinfer"),
        ("mla_decode", {"sgl_kernel"}, 90, "bf16", SGL_KERNEL),
        ("mla_decode", {"flashinfer"}, 80, "bf16", "flashinfer"),
        ("moe", {"deep_gemm"}, 90, None, "deep_gemm"),
        ("moe", {"vllm"}, 80, None, "vllm"),
        ("selective_scan", {"mamba_ssm"}, 80, "bf16", "mamba-ssm"),
        ("causal_conv1d", {"causal_conv1d"}, 80, "bf16", "causal-conv1d"),
    ]
    for op, libs, sm, dtype, expected in cases:
        dispatch.reset_cache()
        _mock_available(monkeypatch, available_libs=libs, sm=sm)
        got = dispatch.which(op, dtype=dtype)
        assert got == expected, f"{op} sm{sm} {dtype}: {got} != {expected}"
        assert got != KERNEL_SET, op


def test_compute_bound_falls_back_to_ks_when_nothing_external(monkeypatch):
    # On every arch, with NO external provider installed, each compute-bound op
    # falls back to kernel-set (the portable C-ABI path) — and only then.
    for sm in (80, 89, 90, 100):
        for op in COMPUTE_BOUND_OPS:
            dispatch.reset_cache()
            _mock_available(monkeypatch, available_libs=set(), sm=sm)
            assert dispatch.which(op) == KERNEL_SET, f"{op} sm{sm}"


def test_gemm_optimal_is_cublas_torch_every_arch(monkeypatch):
    # Dense GEMM optimal = cuBLAS/cuBLASLt via torch on A100/L4/H100/B200.
    for sm in (80, 89, 90, 100):
        dispatch.reset_cache()
        _mock_available(monkeypatch, available_libs={"torch"}, sm=sm)
        assert dispatch.which("gemm", dtype="bf16") == "torch", f"sm{sm}"


def test_fp8_gemm_arch_gating_full_ladder(monkeypatch):
    # sm90/sm100: DeepGEMM rank-1. sm89: DeepGEMM gated out -> torch._scaled_mm.
    # sm80 (no FP8 tensor cores): DeepGEMM + sgl gated; torch-scaled-mm is sm89
    # so it too gates out -> kernel-set bf16-cast fallback (no FP8 hw on Ampere).
    _mock_available(
        monkeypatch, available_libs={"deep_gemm", "torch", "sgl_kernel"}, sm=90)
    assert dispatch.which("fp8_gemm", dtype="fp8") == "deep_gemm"
    dispatch.reset_cache()
    _mock_available(
        monkeypatch, available_libs={"deep_gemm", "torch", "sgl_kernel"}, sm=100)
    assert dispatch.which("fp8_gemm", dtype="fp8") == "deep_gemm"
    dispatch.reset_cache()
    _mock_available(
        monkeypatch, available_libs={"deep_gemm", "torch", "sgl_kernel"}, sm=89)
    assert dispatch.which("fp8_gemm", dtype="fp8") == "torch-scaled-mm"
    dispatch.reset_cache()
    _mock_available(
        monkeypatch, available_libs={"deep_gemm", "torch", "sgl_kernel"}, sm=80)
    assert dispatch.which("fp8_gemm", dtype="fp8") == KERNEL_SET


def test_int8_gemm_rank_order_vllm_then_sgl(monkeypatch):
    # Registry true rank-1 is vLLM CUTLASS int8; sgl-kernel is rank-2.
    _mock_available(monkeypatch, available_libs={"vllm", "sgl_kernel"}, sm=80)
    assert dispatch.which("int8_gemm", dtype="int8") == "vllm"
    # On Blackwell sm100 CUTLASS int8 is unsupported -> Marlin-int8 leads.
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"vllm", "sgl_kernel"}, sm=100)
    assert dispatch.which("int8_gemm", dtype="int8") == "vllm-marlin-int8"


def test_w4a16_marlin_wired_and_machete_on_hopper(monkeypatch):
    # The vllm-marlin adapter is now wired (was call=None) -> selectable.
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=80)
    assert dispatch.which("w4a16", dtype="int4") == "vllm-marlin"
    assert dispatch.which("w4a16", dtype="int4") != KERNEL_SET
    # On Hopper sm90 Machete outranks Marlin.
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=90)
    assert dispatch.which("w4a16", dtype="int4") == "vllm-machete"
    # Machete is sm90-gated: on sm80 it drops, Marlin leads.
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=80)
    assert dispatch.which("w4a16", dtype="int4") == "vllm-marlin"


def test_fa4_blackwell_only(monkeypatch):
    # FlashAttention-4 (flash_attn.cute) is the rank-0 sm100 prefill path; on
    # Hopper/Ampere it gates out and FA2/FA3 (flash-attn) leads.
    _mock_available(monkeypatch, available_libs={"flash_attn"}, sm=90)
    assert dispatch.which("attention_prefill", dtype="bf16") == "flash-attn"
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"flash_attn"}, sm=80)
    assert dispatch.which("attention_prefill", dtype="bf16") == "flash-attn"


def test_mla_decode_portable_flashinfer_pre_hopper(monkeypatch):
    # Pre-Hopper (A100 sm80 / L4 sm89): FlashMLA is sm90-gated; FlashInfer MLA
    # is now wired (sm80+) so MLA decode has a real SOTA path, not the ks 1%-BW
    # fallback.
    for sm in (80, 89):
        dispatch.reset_cache()
        _mock_available(
            monkeypatch, available_libs={"sgl_kernel", "flashinfer"}, sm=sm)
        assert dispatch.which("mla_decode") == "flashinfer", f"sm{sm}"
    # On Hopper FlashMLA (rank 1) beats FlashInfer (rank 2).
    dispatch.reset_cache()
    _mock_available(
        monkeypatch, available_libs={"sgl_kernel", "flashinfer"}, sm=90)
    assert dispatch.which("mla_decode") == SGL_KERNEL


def test_ssm_and_conv_external_providers_are_sm80_gated(monkeypatch):
    _mock_available(
        monkeypatch, available_libs={"mamba_ssm", "causal_conv1d"}, sm=75)
    assert dispatch.which("selective_scan", dtype="fp16") == KERNEL_SET
    assert dispatch.which("causal_conv1d", dtype="fp16") == KERNEL_SET

    dispatch.reset_cache()
    _mock_available(
        monkeypatch, available_libs={"mamba_ssm", "causal_conv1d"}, sm=80)
    assert dispatch.which("selective_scan", dtype="fp16") == "mamba-ssm"
    assert dispatch.which("selective_scan", dtype="bf16") == "mamba-ssm"
    assert dispatch.which("causal_conv1d", dtype="fp16") == "causal-conv1d"
    assert dispatch.which("causal_conv1d", dtype="bf16") == "causal-conv1d"

    dispatch.reset_cache()
    _mock_available(
        monkeypatch, available_libs={"mamba_ssm", "causal_conv1d"}, sm=80)
    assert dispatch.which("selective_scan", dtype="fp32") == KERNEL_SET
    assert dispatch.which("causal_conv1d", dtype="fp32") == KERNEL_SET


def test_no_compute_bound_op_has_unwired_rank1(monkeypatch):
    # Regression for the old call=None gaps (vllm-marlin / vllm gate). For every
    # compute-bound op, the rank-1 (lowest-rank) external provider must have a
    # wired call adapter — otherwise it can never actually dispatch.
    for op in COMPUTE_BOUND_OPS:
        externals = [p for p in OPS[op].providers if p.name != KERNEL_SET]
        assert externals, op
        lead = min(externals, key=lambda p: p.rank)
        assert lead.call is not None, f"{op} rank-{lead.rank} {lead.name} unwired"


def test_arch_gates_preserved_for_sm90_providers():
    # DeepGEMM / FlashMLA / Machete must stay sm90-gated; Marlin sm80; FA4 sm100.
    def gate(op, name):
        return next(p.min_sm for p in OPS[op].providers if p.name == name)
    assert gate("fp8_gemm", "deep_gemm") == 90
    assert gate("moe", "deep_gemm") == 90
    assert gate("mla_decode", SGL_KERNEL) == 90
    assert gate("w4a16", "vllm-machete") == 90
    assert gate("w4a16", "vllm-marlin") == 80
    assert gate("int8_gemm", "vllm") == 80
    assert gate("int8_gemm", "vllm-marlin-int8") == 100
    assert gate("attention_prefill", "flash-attn-cute") == 100
    assert gate("attention_prefill", "flash-attn") == 80
    assert gate("mla_decode", "flashinfer") == 80
    assert gate("selective_scan", "mamba-ssm") == 80
    assert gate("causal_conv1d", "causal-conv1d") == 80


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
