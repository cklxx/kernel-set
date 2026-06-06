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


class _FakeTensor:
    def __init__(self, shape, *, device="cuda:0", dtype="bf16"):
        self.shape = shape
        self.ndim = len(shape)
        self.device = device
        self.dtype = dtype


def _install_fake_fla(monkeypatch, calls):
    fla = types.ModuleType("fla")
    ops = types.ModuleType("fla.ops")

    def record(name):
        def fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return f"{name}_out", "final_state"
        return fn

    for name in (
        "chunk_gated_delta_rule", "chunk_kda", "chunk_gla",
        "chunk_simple_gla", "chunk_lightning_attn", "chunk_rwkv7",
        "fused_recurrent_gated_delta_rule", "fused_recurrent_gla",
        "fused_recurrent_simple_gla", "fused_recurrent_rwkv7",
        "parallel_nsa", "native_sparse_attention",
    ):
        setattr(ops, name, record(name))
    fla.ops = ops
    monkeypatch.setitem(sys.modules, "fla", fla)
    monkeypatch.setitem(sys.modules, "fla.ops", ops)


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


def test_attention_plain_path_does_not_pass_new_optional_kwargs(monkeypatch):
    torch = pytest.importorskip("torch")

    flash_attn = types.ModuleType("flash_attn")
    calls = []

    def flash_attn_func(q, k, v, *, causal=True, softmax_scale=None):
        calls.append((q, k, v, {"causal": causal,
                                "softmax_scale": softmax_scale}))
        return "plain_attn"

    flash_attn.flash_attn_func = flash_attn_func
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    _mock_available(monkeypatch, available_libs={"flash_attn"}, sm=90)

    q = torch.empty(1, 2, 1, 4)
    k = torch.empty(1, 2, 1, 4)
    v = torch.empty(1, 2, 1, 4)
    assert dispatch.attention_prefill(
        q, k, v, _dtype="bf16", causal=True) == "plain_attn"
    assert calls[-1][3] == {"causal": True, "softmax_scale": None}


def test_attention_flash_attn_kwargs_thread_through(monkeypatch):
    torch = pytest.importorskip("torch")
    from kernel_set.backends import _registry

    flash_attn = types.ModuleType("flash_attn")
    flash_attn_interface = types.ModuleType("flash_attn_interface")
    calls = []

    def flash_attn_func(q, k, v, *, causal=True, softmax_scale=None,
                        window_size=None, softcap=0.0, sinks=None):
        calls.append({
            "causal": causal,
            "softmax_scale": softmax_scale,
            "window_size": window_size,
            "softcap": softcap,
            "sinks": sinks,
        })
        return "feature_attn"

    flash_attn.flash_attn_func = flash_attn_func
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setitem(sys.modules, "flash_attn_interface",
                        flash_attn_interface)
    _mock_available(monkeypatch, available_libs={"flash_attn"}, sm=90)

    q = torch.empty(1, 2, 1, 4)
    k = torch.empty(1, 2, 1, 4)
    v = torch.empty(1, 2, 1, 4)
    sinks = torch.empty(1)
    assert dispatch.attention_prefill(
        q, k, v, _dtype="bf16", window_size=(128, 0),
        softcap=50.0, sinks=sinks) == "feature_attn"
    assert calls[-1]["window_size"] == (128, 0)
    assert calls[-1]["softcap"] == 50.0
    assert calls[-1]["sinks"] is sinks

    def flash_attn_with_kvcache(q, k, v, *, page_table=None,
                                cache_seqlens=None, softmax_scale=None,
                                causal=False, window_size=None, softcap=0.0,
                                sinks=None):
        calls.append({
            "decode": True,
            "window_size": window_size,
            "softcap": softcap,
            "sinks": sinks,
            "page_table": page_table,
            "cache_seqlens": cache_seqlens,
        })
        return q

    flash_attn_interface.flash_attn_with_kvcache = flash_attn_with_kvcache
    qd = torch.empty(2, 1, 4)
    kc = torch.empty(4, 1, 8, 4)
    vc = torch.empty(4, 1, 8, 4)
    block_tables = torch.zeros(2, 1, dtype=torch.int32)
    seq_lens = torch.full((2,), 8, dtype=torch.int32)
    assert _registry._attn_decode_fa3(
        qd, kc, vc, block_tables, seq_lens, block_size=8,
        max_blocks_per_seq=1, window_size=(64, 0),
        softcap=25.0, sinks=sinks).shape == qd.shape
    assert calls[-1]["decode"] is True
    assert calls[-1]["window_size"] == (64, 0)
    assert calls[-1]["softcap"] == 25.0
    assert calls[-1]["sinks"] is sinks


def test_attention_flashinfer_kwargs_thread_through(monkeypatch):
    torch = pytest.importorskip("torch")
    from kernel_set.backends import _registry

    flashinfer = types.ModuleType("flashinfer")
    prefill = types.ModuleType("flashinfer.prefill")
    decode = types.ModuleType("flashinfer.decode")
    calls = []

    def single_prefill_with_kv_cache(
            q, k, v, *, causal=True, kv_layout="NHD", sm_scale=None,
            window_left=None, logits_soft_cap=0.0, custom_mask=None,
            packed_custom_mask=None):
        calls.append(("prefill", {
            "causal": causal,
            "kv_layout": kv_layout,
            "sm_scale": sm_scale,
            "window_left": window_left,
            "logits_soft_cap": logits_soft_cap,
            "custom_mask": custom_mask,
            "packed_custom_mask": packed_custom_mask,
        }))
        return q

    class BatchDecodeWithPagedKVCacheWrapper:
        def __init__(self, workspace, kv_layout="NHD"):
            calls.append(("decode_init", {"kv_layout": kv_layout,
                                          "device": workspace.device}))

        def plan(self, kv_indptr, kv_indices, last, qh, kvh, hd, block_size,
                 *, pos_encoding_mode="NONE", data_type=None,
                 q_data_type=None, window_left=None, logits_soft_cap=0.0):
            calls.append(("decode_plan", {
                "window_left": window_left,
                "logits_soft_cap": logits_soft_cap,
                "pos_encoding_mode": pos_encoding_mode,
                "data_type": data_type,
                "q_data_type": q_data_type,
            }))

        def run(self, q, kv):
            calls.append(("decode_run", kv))
            return q

    prefill.single_prefill_with_kv_cache = single_prefill_with_kv_cache
    decode.BatchDecodeWithPagedKVCacheWrapper = BatchDecodeWithPagedKVCacheWrapper
    flashinfer.prefill = prefill
    flashinfer.decode = decode
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", prefill)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", decode)

    q = torch.empty(1, 2, 1, 4)
    k = torch.empty(1, 2, 1, 4)
    v = torch.empty(1, 2, 1, 4)
    mask = torch.ones(2, 2, dtype=torch.bool)

    out = _registry._attn_prefill_flashinfer(
        q, k, v, window_size=(128, 0), softcap=50.0, custom_mask=mask)
    assert tuple(out.shape) == tuple(q.shape)
    assert calls[-1][0] == "prefill"
    assert calls[-1][1]["window_left"] == 128
    assert calls[-1][1]["logits_soft_cap"] == 50.0
    assert calls[-1][1]["custom_mask"] is mask

    qd = torch.empty(2, 1, 4)
    kc = torch.empty(4, 1, 8, 4)
    vc = torch.empty(4, 1, 8, 4)
    block_tables = torch.zeros(2, 1, dtype=torch.int32)
    seq_lens = torch.full((2,), 8, dtype=torch.int32)
    assert _registry._attn_decode_flashinfer(
        qd, kc, vc, block_tables, seq_lens, block_size=8,
        max_blocks_per_seq=1, window_size=(64, 0), softcap=25.0) is qd
    plan = next(c for c in reversed(calls) if c[0] == "decode_plan")
    assert plan[1]["window_left"] == 64
    assert plan[1]["logits_soft_cap"] == 25.0


def test_attention_call_specific_fallback_and_ks_error(monkeypatch):
    torch = pytest.importorskip("torch")

    flash_attn = types.ModuleType("flash_attn")

    def flash_attn_func(q, k, v, *, causal=True, softmax_scale=None):
        raise AssertionError("custom mask must skip flash-attn")

    flash_attn.flash_attn_func = flash_attn_func
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

    flashinfer = types.ModuleType("flashinfer")
    prefill = types.ModuleType("flashinfer.prefill")

    def single_prefill_with_kv_cache(
            q, k, v, *, causal=True, kv_layout="NHD", sm_scale=None,
            custom_mask=None):
        assert custom_mask is not None
        return q

    prefill.single_prefill_with_kv_cache = single_prefill_with_kv_cache
    flashinfer.prefill = prefill
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", prefill)

    q = torch.empty(1, 2, 1, 4)
    k = torch.empty(1, 2, 1, 4)
    v = torch.empty(1, 2, 1, 4)
    mask = torch.ones(2, 2, dtype=torch.bool)

    _mock_available(monkeypatch, available_libs={"flash_attn", "flashinfer"},
                    sm=90)
    out = dispatch.attention_prefill(q, k, v, _dtype="bf16", custom_mask=mask)
    assert tuple(out.shape) == tuple(q.shape)

    # SDPA is a valid plain-attention fallback, but extras must skip it and
    # reach the terminal ks error when no feature-capable provider is installed.
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"torch"}, sm=80)
    with pytest.raises(NotImplementedError, match="kernel-set attention fallback"):
        dispatch.attention_prefill(
            q, k, v, _dtype="bf16", window_size=(128, 0))


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
    assert names == [
        "flashinfer", "sgl-kernel", "vllm", "liger", "quack", KERNEL_SET]
    sel = {r["name"]: r["selectable"] for r in rows}
    assert sel["flashinfer"] is False   # not installed in this mock
    assert sel["sgl-kernel"] is False   # not installed in this mock
    assert sel["vllm"] is True
    assert sel["quack"] is False
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

    _probe._IMPORT_CACHE.clear()
    pkg = types.ModuleType("fake_kernel_set_probe_pkg")
    sub = types.ModuleType("fake_kernel_set_probe_pkg.ops")
    sub.kernel = object()
    pkg.ops = sub
    monkeypatch.setitem(sys.modules, "fake_kernel_set_probe_pkg", pkg)
    monkeypatch.setitem(sys.modules, "fake_kernel_set_probe_pkg.ops", sub)
    assert _probe.can_import(
        "from fake_kernel_set_probe_pkg import ops as ops; ops.kernel") is True


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
    assert next(p for p in OPS["fp8_kv_cache"].providers
                if p.name == "flashinfer").import_check == \
        "from flashinfer.page import append_paged_kv_cache"


def test_new_deferred_op_import_checks_are_specific():
    assert next(p for p in OPS["w8a16_fp8"].providers
                if p.name == "vllm-fp8-marlin").import_check == \
        "from vllm import _custom_ops as ops; ops.marlin_gemm; " \
        "ops.gptq_marlin_repack; ops.awq_marlin_repack"
    assert next(p for p in OPS["fused_linear_ce"].providers
                if p.name == "liger").import_check == \
        "from liger_kernel.ops.fused_linear_cross_entropy import " \
        "LigerFusedLinearCrossEntropyFunction"
    assert next(p for p in OPS["sparse_2_4_gemm"].providers
                if p.name == "vllm-cutlass-sparse").import_check == \
        "from vllm import _custom_ops as ops; " \
        "ops.cutlass_scaled_sparse_mm; ops.cutlass_sparse_compress"
    assert next(p for p in OPS["bitnet_gemm"].providers
                if p.name == "bitblas").import_check == \
        "from bitblas import Matmul"
    assert next(p for p in OPS["mrope"].providers
                if p.name == "vllm").import_check == \
        "from vllm.model_executor.layers.rotary_embedding.mrope " \
        "import triton_mrope"
    assert next(p for p in OPS["fused_rmsnorm_gated"].providers
                if p.name == "flash-linear-attention").import_check == \
        "from fla.modules import FusedRMSNormGated"
    assert next(p for p in OPS["min_p_sampling"].providers
                if p.name == "flashinfer").import_check == \
        "import flashinfer.sampling; " \
        "flashinfer.sampling.min_p_sampling_from_probs"
    assert next(p for p in OPS["chain_speculative_sampling"].providers
                if p.name == "flashinfer").import_check == \
        "import flashinfer.sampling; " \
        "flashinfer.sampling.chain_speculative_sampling"
    assert next(p for p in OPS["attention_state_merge"].providers
                if p.name == "flashinfer").import_check == \
        "import flashinfer.cascade; flashinfer.cascade.merge_state; " \
        "flashinfer.cascade.merge_states"
    assert next(p for p in OPS["fp4_quantize"].providers
                if p.name == "vllm").import_check == \
        "from vllm import _custom_ops as ops; ops.scaled_fp4_quant"
    assert next(p for p in OPS["mxfp8_quantize"].providers
                if p.name == "vllm").import_check == \
        "from vllm import _custom_ops as ops; ops.mxfp8_experts_quant"
    assert next(p for p in OPS["apply_token_bitmask"].providers
                if p.name == "xgrammar").import_check == \
        "import xgrammar; xgrammar.apply_token_bitmask_inplace; " \
        "xgrammar.allocate_token_bitmask"
    assert next(p for p in OPS["sparse_mla_attention"].providers
                if p.name == "flash-mla").import_check == \
        "from flash_mla import flash_mla_sparse_fwd, " \
        "flash_mla_with_kvcache, get_mla_metadata"
    assert next(p for p in OPS["dsa_indexer_logits"].providers
                if p.name == "deep_gemm").import_check == \
        "import deep_gemm; deep_gemm.fp8_mqa_logits; " \
        "deep_gemm.fp8_paged_mqa_logits"
    assert next(p for p in OPS["dsa_topk_select"].providers
                if p.name == "flashinfer").import_check == \
        "import flashinfer; flashinfer.top_k"
    assert next(p for p in OPS["nsa_selection_attention"].providers
                if p.name == "flash-linear-attention").import_check == \
        "import fla.ops"


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
    "gemm", "fp8_gemm", "int8_gemm", "w4a16", "w4a8", "w8a16_fp8",
    "sparse_2_4_gemm", "bitnet_gemm",
    "attention_prefill", "attention_decode", "mla_decode",
    "sparse_mla_attention", "dsa_indexer_logits", "dsa_topk_select",
    "nsa_selection_attention",
    "moe", "selective_scan", "causal_conv1d",
    "gated_delta_rule", "gated_linear_attn", "rwkv_wkv7",
    "fused_linear_ce",
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
        ("w4a8", {"vllm"}, 80, "int4", "vllm-marlin"),
        ("w4a8", {"vllm"}, 90, "int4", "vllm-machete"),
        ("w8a16_fp8", {"vllm"}, 80, "bf16", "vllm-fp8-marlin"),
        ("sparse_2_4_gemm", {"vllm"}, 90, "fp8", "vllm-cutlass-sparse"),
        ("bitnet_gemm", {"bitblas"}, 80, "fp16", "bitblas"),
        ("attention_prefill", {"flash_attn"}, 80, "bf16", "flash-attn"),
        ("attention_decode", {"flashinfer"}, 80, "fp16", "flashinfer"),
        ("mla_decode", {"sgl_kernel"}, 90, "bf16", SGL_KERNEL),
        ("mla_decode", {"flashinfer"}, 80, "bf16", "flashinfer"),
        ("sparse_mla_attention", {"flash_mla"}, 90, "bf16", "flash-mla"),
        ("dsa_indexer_logits", {"deep_gemm"}, 90, "fp8", "deep_gemm"),
        ("dsa_topk_select", {"flashinfer"}, 80, "bf16", "flashinfer"),
        ("nsa_selection_attention", {"fla"}, 80, "bf16",
         "flash-linear-attention"),
        ("moe", {"deep_gemm"}, 90, None, "deep_gemm"),
        ("moe", {"vllm"}, 80, None, "vllm"),
        ("selective_scan", {"mamba_ssm"}, 80, "bf16", "mamba-ssm"),
        ("causal_conv1d", {"causal_conv1d"}, 80, "bf16", "causal-conv1d"),
        ("gated_delta_rule", {"fla"}, 80, "bf16", "flash-linear-attention"),
        ("gated_linear_attn", {"fla"}, 80, "fp16", "flash-linear-attention"),
        ("rwkv_wkv7", {"fla"}, 80, "bf16", "flash-linear-attention"),
        ("fused_linear_ce", {"liger_kernel"}, 80, "bf16", "liger"),
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


def test_w4a8_new_op_marlin_ampere_machete_hopper(monkeypatch):
    assert "w4a8" in dispatch.ops()
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=80)
    assert dispatch.which("w4a8", dtype="int4") == "vllm-marlin"
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=90)
    assert dispatch.which("w4a8", dtype="int4") == "vllm-machete"
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs=set(), sm=90)
    assert dispatch.which("w4a8", dtype="int4") == KERNEL_SET


def test_w8a16_fp8_new_op_marlin_weight_only(monkeypatch):
    assert "w8a16_fp8" in dispatch.ops()
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=80)
    assert dispatch.which("w8a16_fp8", dtype="bf16") == "vllm-fp8-marlin"
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=86)
    assert dispatch.which("w8a16_fp8", dtype="fp16") == "vllm-fp8-marlin"
    # The dtype gate is for activation compute. fp8 activations remain
    # hardware-infeasible on sm80, even though this op stores weights in fp8.
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=80)
    assert dispatch.which("w8a16_fp8", dtype="fp8") == KERNEL_SET
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs=set(), sm=90)
    assert dispatch.which("w8a16_fp8", dtype="bf16") == KERNEL_SET


def test_fused_linear_ce_new_op_liger_and_ks(monkeypatch):
    assert "fused_linear_ce" in dispatch.ops()
    _mock_available(monkeypatch, available_libs={"liger_kernel"}, sm=80)
    assert dispatch.which("fused_linear_ce", dtype="bf16") == "liger"
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"liger_kernel"}, sm=75)
    assert dispatch.which("fused_linear_ce", dtype="fp16") == KERNEL_SET
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs=set(), sm=90)
    assert dispatch.which("fused_linear_ce", dtype="bf16") == KERNEL_SET


def test_sparse_2_4_new_op_cutlass_sparse_hopper(monkeypatch):
    assert "sparse_2_4_gemm" in dispatch.ops()
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=90)
    assert dispatch.which("sparse_2_4_gemm", dtype="fp8") == \
        "vllm-cutlass-sparse"
    assert dispatch.which("sparse_2_4_gemm", dtype="int8") == \
        "vllm-cutlass-sparse"
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=89)
    assert dispatch.which("sparse_2_4_gemm", dtype="fp8") == KERNEL_SET
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs=set(), sm=90)
    assert dispatch.which("sparse_2_4_gemm", dtype="fp8") == KERNEL_SET


def test_bitnet_new_op_bitblas_ampere(monkeypatch):
    assert "bitnet_gemm" in dispatch.ops()
    _mock_available(monkeypatch, available_libs={"bitblas"}, sm=80)
    assert dispatch.which("bitnet_gemm", dtype="fp16") == "bitblas"
    assert dispatch.which("bitnet_gemm", dtype="int8") == "bitblas"
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"bitblas"}, sm=75)
    assert dispatch.which("bitnet_gemm", dtype="fp16") == KERNEL_SET
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"bitblas"}, sm=80)
    assert dispatch.which("bitnet_gemm", dtype="bf16") == KERNEL_SET


def test_wave2b_dsa_provider_only_ops_select_and_gate(monkeypatch):
    assert "sparse_mla_attention" in dispatch.ops()
    assert "dsa_indexer_logits" in dispatch.ops()
    assert "dsa_topk_select" in dispatch.ops()
    assert "nsa_selection_attention" in dispatch.ops()

    cases = [
        ("sparse_mla_attention", {"flash_mla"}, 90, "bf16", "flash-mla"),
        ("sparse_mla_attention", {"flash_mla"}, 90, "fp8", "flash-mla"),
        ("dsa_indexer_logits", {"deep_gemm"}, 90, "fp8", "deep_gemm"),
        ("dsa_topk_select", {"flashinfer"}, 80, "fp16", "flashinfer"),
        ("dsa_topk_select", {"flashinfer"}, 80, "bf16", "flashinfer"),
        ("nsa_selection_attention", {"fla"}, 80, "fp16",
         "flash-linear-attention"),
    ]
    for op, libs, sm, dtype, expected in cases:
        dispatch.reset_cache()
        _mock_available(monkeypatch, available_libs=libs, sm=sm)
        assert dispatch.which(op, dtype=dtype) == expected

    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"flash_mla"}, sm=89)
    assert dispatch.which("sparse_mla_attention", dtype="bf16") == KERNEL_SET
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"deep_gemm"}, sm=90)
    assert dispatch.which("dsa_indexer_logits", dtype="bf16") == KERNEL_SET
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"fla"}, sm=75)
    assert dispatch.which("nsa_selection_attention", dtype="fp16") == KERNEL_SET


def test_wave1_provider_only_ops_select_and_gate(monkeypatch):
    cases = [
        ("mrope", {"vllm"}, 80, "bf16", "vllm"),
        ("fused_rmsnorm_gated", {"fla"}, 80, "fp16",
         "flash-linear-attention"),
        ("min_p_sampling", {"flashinfer"}, 75, "fp32", "flashinfer"),
        ("chain_speculative_sampling", {"flashinfer"}, 75, "fp16",
         "flashinfer"),
        ("attention_state_merge", {"flashinfer"}, 80, "bf16",
         "flashinfer"),
        ("attention_state_merge", {"flashinfer"}, 89, "fp8",
         "flashinfer"),
        ("fp4_quantize", {"vllm"}, 100, "fp4", "vllm"),
        ("mxfp8_quantize", {"vllm"}, 100, "fp8", "vllm"),
        ("apply_token_bitmask", {"xgrammar"}, 75, "fp32", "xgrammar"),
    ]
    for op, libs, sm, dtype, expected in cases:
        dispatch.reset_cache()
        _mock_available(monkeypatch, available_libs=libs, sm=sm)
        assert dispatch.which(op, dtype=dtype) == expected

    # Arch/dtype gates remain real despite provider-only terminal fallback.
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=90)
    assert dispatch.which("fp4_quantize", dtype="fp4") == KERNEL_SET
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"vllm"}, sm=90)
    assert dispatch.which("mxfp8_quantize", dtype="fp8") == KERNEL_SET
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"xgrammar"}, sm=75)
    assert dispatch.which("apply_token_bitmask", dtype="bf16") == KERNEL_SET


def test_wave1_provider_only_adapters_call_expected_external_apis(monkeypatch):
    torch = pytest.importorskip("torch")
    from kernel_set.backends import _registry

    calls = []

    def install_module(name, module):
        monkeypatch.setitem(sys.modules, name, module)

    # FlashInfer sampling + cascade.
    flashinfer = types.ModuleType("flashinfer")
    sampling = types.ModuleType("flashinfer.sampling")
    cascade = types.ModuleType("flashinfer.cascade")

    def min_p(*args, **kwargs):
        calls.append(("min_p", args, kwargs))
        return "min_p_ids"

    def chain(*args, **kwargs):
        calls.append(("chain", args, kwargs))
        return "chain_out"

    def merge_state(*args, **kwargs):
        calls.append(("merge_state", args, kwargs))
        return "merged_pair"

    def merge_states(*args, **kwargs):
        calls.append(("merge_states", args, kwargs))
        return "merged_many"

    sampling.min_p_sampling_from_probs = min_p
    sampling.chain_speculative_sampling = chain
    cascade.merge_state = merge_state
    cascade.merge_states = merge_states
    flashinfer.sampling = sampling
    flashinfer.cascade = cascade
    install_module("flashinfer", flashinfer)
    install_module("flashinfer.sampling", sampling)
    install_module("flashinfer.cascade", cascade)

    probs = torch.ones(2, 4)
    draft = torch.ones(2, 3, 4)
    ids = torch.zeros(2, 3, dtype=torch.int32)
    assert _registry._min_p_sampling_flashinfer(probs, 0.05) == "min_p_ids"
    assert calls[-1][0] == "min_p"
    assert calls[-1][1][:2] == (probs, 0.05)
    assert _registry._chain_speculative_sampling_flashinfer(
        draft, ids, draft) == "chain_out"
    assert calls[-1][0] == "chain"
    assert calls[-1][1][:3] == (draft, ids, draft)
    assert _registry._attention_state_merge_flashinfer(probs, probs) == \
        "merged_many"
    assert calls[-1][0] == "merge_states"
    assert _registry._attention_state_merge_flashinfer(
        probs, probs, probs, probs) == "merged_pair"
    assert calls[-1][0] == "merge_state"

    # xgrammar in-place bitmask apply.
    xgrammar = types.ModuleType("xgrammar")

    def apply_mask(*args, **kwargs):
        calls.append(("bitmask", args, kwargs))

    xgrammar.apply_token_bitmask_inplace = apply_mask
    xgrammar.allocate_token_bitmask = object()
    install_module("xgrammar", xgrammar)
    bitmask = torch.zeros(2, 1, dtype=torch.int32)
    logits = torch.zeros(2, 4)
    assert _registry._apply_token_bitmask_xgrammar(logits, bitmask) is logits
    assert calls[-1][0] == "bitmask"
    assert calls[-1][1][:2] == (logits, bitmask)
    assert calls[-1][2]["vocab_size"] == 4

    # vLLM quantize + mRoPE modules.
    vllm = types.ModuleType("vllm")
    ops = types.ModuleType("vllm._custom_ops")

    def scaled_fp4_quant(*args, **kwargs):
        calls.append(("fp4_quant", args, kwargs))
        return "fp4_out", "fp4_scale"

    def mxfp8_experts_quant(*args, **kwargs):
        calls.append(("mxfp8_quant", args, kwargs))

    ops.scaled_fp4_quant = scaled_fp4_quant
    ops.mxfp8_experts_quant = mxfp8_experts_quant
    vllm._custom_ops = ops
    install_module("vllm", vllm)
    install_module("vllm._custom_ops", ops)

    assert _registry._nvfp4_quantize_vllm(probs, None) == \
        ("fp4_out", "fp4_scale")
    assert calls[-1][0] == "fp4_quant"
    assert calls[-1][2]["is_sf_swizzled_layout"] is True

    problem_sizes = torch.ones(1, dtype=torch.int32)
    expert_offsets = torch.zeros(1, dtype=torch.int32)
    blockscale_offsets = torch.zeros(1, dtype=torch.int32)
    quant_output = torch.empty_like(probs)
    scale_factor = torch.empty(1, dtype=torch.uint8)
    got = _registry._mxfp8_quantize_vllm(
        probs, problem_sizes, expert_offsets, blockscale_offsets,
        quant_output=quant_output, scale_factor=scale_factor)
    assert got == (quant_output, scale_factor)
    assert calls[-1][0] == "mxfp8_quant"
    for got_arg, expected_arg in zip(
            calls[-1][1],
            (probs, problem_sizes, expert_offsets, blockscale_offsets,
             quant_output, scale_factor)):
        assert got_arg is expected_arg

    model_executor = types.ModuleType("vllm.model_executor")
    layers = types.ModuleType("vllm.model_executor.layers")
    rotary = types.ModuleType("vllm.model_executor.layers.rotary_embedding")
    mrope = types.ModuleType(
        "vllm.model_executor.layers.rotary_embedding.mrope")

    def triton_mrope(*args, **kwargs):
        calls.append(("mrope", args, kwargs))
        return "mrope_out"

    mrope.triton_mrope = triton_mrope
    rotary.mrope = mrope
    layers.rotary_embedding = rotary
    model_executor.layers = layers
    vllm.model_executor = model_executor
    install_module("vllm.model_executor", model_executor)
    install_module("vllm.model_executor.layers", layers)
    install_module("vllm.model_executor.layers.rotary_embedding", rotary)
    install_module("vllm.model_executor.layers.rotary_embedding.mrope", mrope)
    assert _registry._mrope_vllm(
        probs, probs, probs, probs, (16, 24, 24), rotary_dim=32) == "mrope_out"
    assert calls[-1][0] == "mrope"
    assert calls[-1][2]["rotary_dim"] == 32

    # FLA FusedRMSNormGated module.
    fla = types.ModuleType("fla")
    fla_modules = types.ModuleType("fla.modules")

    class FusedRMSNormGated:
        def __init__(self, hidden_size, **kwargs):
            calls.append(("fla_init", (hidden_size,), kwargs))
            self.weight = torch.empty(hidden_size)

        def to(self, *, device=None, dtype=None):
            calls.append(("fla_to", (device, dtype), {}))
            return self

        def __call__(self, x, gate):
            calls.append(("fla_call", (x, gate), {}))
            return "gated_norm_out"

    fla_modules.FusedRMSNormGated = FusedRMSNormGated
    fla.modules = fla_modules
    install_module("fla", fla)
    install_module("fla.modules", fla_modules)
    weight = torch.ones(4)
    gate = torch.zeros_like(probs)
    assert _registry._fused_rmsnorm_gated_fla(
        probs, weight, gate, activation="sigmoid") == "gated_norm_out"
    assert calls[-3][0] == "fla_init"
    assert calls[-1][0] == "fla_call"


def test_sparse_and_bitnet_adapters_call_expected_external_apis(monkeypatch):
    torch = pytest.importorskip("torch")
    from kernel_set.backends import _registry

    vllm = types.ModuleType("vllm")
    ops = types.ModuleType("vllm._custom_ops")
    sparse_calls = []

    def sparse_mm(*args):
        sparse_calls.append(args)
        return "sparse_out"

    ops.cutlass_scaled_sparse_mm = sparse_mm
    ops.cutlass_sparse_compress = object()
    vllm._custom_ops = ops
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", ops)

    a = torch.empty(2, 4)
    meta = torch.empty(1)
    bq = torch.empty(4, 8)
    sa = torch.empty(1)
    sb = torch.empty(1)
    assert _registry._sparse_2_4_gemm_vllm(a, meta, bq, sa, sb) == \
        "sparse_out"
    assert sparse_calls[-1] == (a, meta, bq, sa, sb, a.dtype, None)

    bitblas = types.ModuleType("bitblas")
    bitblas_calls = []

    class Matmul:
        def __init__(self, config):
            self.config = config

        def __call__(self, *args, **kwargs):
            bitblas_calls.append((self.config, args, kwargs))
            return "bitnet_out"

    bitblas.Matmul = Matmul
    monkeypatch.setitem(sys.modules, "bitblas", bitblas)

    ternary = torch.empty(8, 4)
    assert _registry._bitnet_gemm_bitblas(a, ternary, scale=sa) == \
        "bitnet_out"
    config, args, kwargs = bitblas_calls[-1]
    assert config["W_dtype"] == "int2"
    assert config["bitnet"] is True
    assert args[:2] == (a, ternary)
    assert kwargs["scale"] is sa


def test_wave2b_dsa_adapters_call_expected_external_apis(monkeypatch):
    torch = pytest.importorskip("torch")
    from kernel_set.backends import _registry

    calls = []

    flash_mla = types.ModuleType("flash_mla")

    def flash_mla_sparse_fwd(*args, **kwargs):
        calls.append(("flash_mla_sparse_fwd", args, kwargs))
        return "sparse_prefill_out", "lse"

    def get_mla_metadata(*args, **kwargs):
        calls.append(("get_mla_metadata", args, kwargs))
        return "tile_md", "num_splits"

    def flash_mla_with_kvcache(*args, **kwargs):
        calls.append(("flash_mla_with_kvcache", args, kwargs))
        return "sparse_decode_out", "lse"

    flash_mla.flash_mla_sparse_fwd = flash_mla_sparse_fwd
    flash_mla.get_mla_metadata = get_mla_metadata
    flash_mla.flash_mla_with_kvcache = flash_mla_with_kvcache
    monkeypatch.setitem(sys.modules, "flash_mla", flash_mla)

    q_nope = torch.zeros(2, 4, 8)
    q_pe = torch.zeros(2, 4, 2)
    kv = torch.zeros(3, 1, 10)
    block_tables = torch.zeros(2, 3, dtype=torch.int32)
    seq_lens = torch.tensor([8, 8], dtype=torch.int32)
    indices = torch.zeros(2, 4, 3, dtype=torch.int32)

    assert _registry._sparse_mla_attention_flash_mla(
        q_nope, q_pe, kv, indices=indices, topk=3, prefill=True) == \
        "sparse_prefill_out"
    assert calls[-1][0] == "flash_mla_sparse_fwd"
    assert calls[-1][1][2] is indices
    assert calls[-1][2]["topk"] == 3

    assert _registry._sparse_mla_attention_flash_mla(
        q_nope, q_pe, kv, block_tables, seq_lens, indices,
        heads=4, lora=8, rope_dim=2, topk=3, is_fp8=True) == \
        "sparse_decode_out"
    assert calls[-2][0] == "get_mla_metadata"
    assert calls[-2][2]["is_fp8"] is True
    assert calls[-2][2]["topk"] == 3
    assert calls[-1][0] == "flash_mla_with_kvcache"
    assert calls[-1][2]["indices"] is indices
    assert calls[-1][2]["topk"] == 3

    deep_gemm = types.ModuleType("deep_gemm")

    def fp8_mqa_logits(*args, **kwargs):
        calls.append(("fp8_mqa_logits", args, kwargs))
        return "prefill_logits"

    def fp8_paged_mqa_logits(*args, **kwargs):
        calls.append(("fp8_paged_mqa_logits", args, kwargs))
        return "paged_logits"

    deep_gemm.fp8_mqa_logits = fp8_mqa_logits
    deep_gemm.fp8_paged_mqa_logits = fp8_paged_mqa_logits
    monkeypatch.setitem(sys.modules, "deep_gemm", deep_gemm)
    assert _registry._dsa_indexer_logits_deepgemm(q_nope, kv) == \
        "prefill_logits"
    assert calls[-1][0] == "fp8_mqa_logits"
    assert _registry._dsa_indexer_logits_deepgemm(
        q_nope, kv, paged=True, block_tables=block_tables,
        seq_lens=seq_lens) == "paged_logits"
    assert calls[-1][0] == "fp8_paged_mqa_logits"
    assert calls[-1][1][:4] == (q_nope, kv, block_tables, seq_lens)

    flashinfer = types.ModuleType("flashinfer")
    values = torch.ones(2, 3)
    out_indices = torch.ones(2, 3, dtype=torch.int32)

    def top_k(*args, **kwargs):
        calls.append(("top_k", args, kwargs))
        return values, out_indices

    flashinfer.top_k = top_k
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    assert _registry._dsa_topk_select_flashinfer(q_nope, 3) is out_indices
    assert calls[-1][0] == "top_k"
    assert calls[-1][1][:2] == (q_nope, 3)

    calls.clear()
    _install_fake_fla(monkeypatch, calls)
    assert _registry._nsa_selection_attention_fla(
        q_nope, q_nope, q_nope, block_tables=block_tables) == \
        "parallel_nsa_out"
    assert calls[-1][0] == "parallel_nsa"
    assert calls[-1][2]["block_tables"] is block_tables


def test_vllm_marlin_adapters_use_unified_marlin_gemm(monkeypatch):
    torch = pytest.importorskip("torch")
    from kernel_set.backends import _registry

    calls = []
    vllm = types.ModuleType("vllm")
    ops = types.ModuleType("vllm._custom_ops")

    def marlin_gemm(*args):
        calls.append(args)
        return "marlin_out"

    ops.marlin_gemm = marlin_gemm
    vllm._custom_ops = ops
    scalar_mod = types.ModuleType("vllm.scalar_type")
    scalar_mod.scalar_types = types.SimpleNamespace(
        uint4b8="uint4b8", uint4="uint4", int8="int8", int4="int4",
        float8_e4m3fn="float8_e4m3fn")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", ops)
    monkeypatch.setitem(sys.modules, "vllm.scalar_type", scalar_mod)

    a = torch.zeros(2, 4)
    b = torch.zeros(1, 1, dtype=torch.int32)
    scales = torch.ones(1, 8)
    assert _registry._w4a16_marlin(a, b, scales, None) == "marlin_out"
    assert calls[-1][11] == "uint4b8"
    assert calls[-1][12:15] == (2, 8, 4)

    zeros = torch.zeros_like(scales)
    assert _registry._w4a16_marlin(a, b, scales, zeros) == "marlin_out"
    assert calls[-1][7] is zeros
    assert calls[-1][11] == "uint4"

    a8 = torch.zeros(2, 4, dtype=torch.int8)
    b8 = torch.zeros(4, 8, dtype=torch.int8)
    a_scale = torch.ones(2, 1)
    b_scale = torch.ones(1, 8)
    assert _registry._int8_gemm_marlin(a8, b8, a_scale, b_scale) == \
        "marlin_out"
    assert calls[-1][5] is a_scale
    assert calls[-1][11] == "int8"

    assert _registry._w8a16_fp8_marlin(a, b, scales) == "marlin_out"
    assert calls[-1][11] == "float8_e4m3fn"
    assert calls[-1][12:15] == (2, 8, 4)


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
    assert dispatch.which("selective_scan", dtype="fp32") == "mamba-ssm"
    assert dispatch.which("causal_conv1d", dtype="fp32") == "causal-conv1d"


def test_linear_attn_external_provider_is_sm80_gated(monkeypatch):
    _mock_available(monkeypatch, available_libs={"fla"}, sm=75)
    for op in ("gated_delta_rule", "gated_linear_attn", "rwkv_wkv7"):
        assert dispatch.which(op, dtype="fp16") == KERNEL_SET

    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"fla"}, sm=80)
    for op in ("gated_delta_rule", "gated_linear_attn", "rwkv_wkv7"):
        assert dispatch.which(op, dtype="fp16") == "flash-linear-attention"
        assert dispatch.which(op, dtype="bf16") == "flash-linear-attention"

    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"fla"}, sm=80)
    for op in ("gated_delta_rule", "gated_linear_attn", "rwkv_wkv7"):
        assert dispatch.which(op, dtype="fp32") == KERNEL_SET


def test_linear_attn_public_dispatch_calls_fla_adapters(monkeypatch):
    calls = []
    _install_fake_fla(monkeypatch, calls)
    _mock_available(monkeypatch, available_libs={"fla"}, sm=80)

    q = _FakeTensor((2, 3, 4, 8))
    k = _FakeTensor((2, 3, 4, 8))
    v = _FakeTensor((2, 3, 4, 16))
    g_scalar = _FakeTensor((2, 3, 4))
    g_vector = _FakeTensor((2, 3, 4, 8))
    beta = _FakeTensor((2, 3, 4))

    assert dispatch.gated_delta_rule(
        q, k, v, g_scalar, beta, use_qk_l2norm=1, scale=0.5) == \
        "chunk_gated_delta_rule_out"
    assert calls[-1][0] == "chunk_gated_delta_rule"
    assert calls[-1][2]["use_qk_l2norm_in_kernel"] is True
    assert calls[-1][2]["scale"] == 0.5

    assert dispatch.gated_delta_rule(q, k, v, g_vector, beta) == \
        "chunk_kda_out"
    assert calls[-1][0] == "chunk_kda"

    assert dispatch.gated_linear_attn(q, k, v, g_vector) == "chunk_gla_out"
    assert calls[-1][0] == "chunk_gla"
    assert dispatch.gated_linear_attn(q, k, v, g_scalar) == \
        "chunk_simple_gla_out"
    assert calls[-1][0] == "chunk_simple_gla"
    assert dispatch.gated_linear_attn(
        q, k, v, layer_idx=1, num_layers=4) == "chunk_lightning_attn_out"
    assert calls[-1][0] == "chunk_lightning_attn"

    assert dispatch.rwkv_wkv7(q, g_vector, k, v, q, k, scale=0.25) == \
        "chunk_rwkv7_out"
    assert calls[-1][0] == "chunk_rwkv7"
    assert calls[-1][2]["scale"] == 0.25


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
    assert gate("w4a8", "vllm-machete") == 90
    assert gate("w4a8", "vllm-marlin") == 80
    assert gate("int8_gemm", "vllm") == 80
    assert gate("int8_gemm", "vllm-marlin-int8") == 100
    assert gate("attention_prefill", "flash-attn-cute") == 100
    assert gate("attention_prefill", "flash-attn") == 80
    assert gate("mla_decode", "flashinfer") == 80
    assert gate("selective_scan", "mamba-ssm") == 80
    assert gate("causal_conv1d", "causal-conv1d") == 80
    assert gate("gated_delta_rule", "flash-linear-attention") == 80
    assert gate("gated_linear_attn", "flash-linear-attention") == 80
    assert gate("rwkv_wkv7", "flash-linear-attention") == 80
    assert gate("sparse_mla_attention", "flash-mla") == 90
    assert gate("dsa_indexer_logits", "deep_gemm") == 90
    assert gate("dsa_topk_select", "flashinfer") == 80
    assert gate("nsa_selection_attention", "flash-linear-attention") == 80


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
