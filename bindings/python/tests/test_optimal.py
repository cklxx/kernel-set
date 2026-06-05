"""Unit tests for the Cartesian optimal-selection table + pure selector.

Covers:

* the generated ``providers/optimal.json`` is well-formed (every cell has a
  provider/source/fallback_chain terminating in kernel-set; measured cells carry
  a metric + gpu);
* ``select_optimal`` returns the **measured winner** where one exists
  (attention.prefill sm89 fp16 -> flashinfer; norm.rmsnorm sm80 bf16 ->
  kernel-set; rmsnorm sm89 fp16 -> liger), via O(1) exact-cell lookup;
* graceful fallback: nearest feasible dtype, then the kernel-set terminal;
* arch/dtype gating: fp8 omitted on sm75 -> kernel-set fallback;
* the runtime dispatcher consults the table (measured override flows through to
  ``which`` when the winning provider is installed).

These run on a no-GPU / no-torch host (selection logic only).

Run::

    python3 -m pytest bindings/python/tests/test_optimal.py -q
"""

from __future__ import annotations

import json
import os

import pytest

from kernel_set import dispatch
from kernel_set.backends import KERNEL_SET
from kernel_set.backends import optimal as O

_OPTIMAL_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "providers", "optimal.json")


# --------------------------------------------------------------------------- #
# The table file itself.
# --------------------------------------------------------------------------- #
def test_optimal_json_exists_and_loads():
    assert O.has_table(), "providers/optimal.json must be present + parseable"
    with open(_OPTIMAL_JSON) as f:
        doc = json.load(f)
    assert doc["cells"], "table must have cells"
    stats = doc["stats"]
    assert stats["total"] == len(doc["cells"])
    assert stats["measured"] + stats["heuristic"] == stats["total"]
    assert stats["measured"] > 0 and stats["heuristic"] > 0


def test_every_cell_is_well_formed():
    with open(_OPTIMAL_JSON) as f:
        doc = json.load(f)
    for c in doc["cells"]:
        assert c["source"] in ("measured", "heuristic"), c
        assert c["fallback_chain"], c
        # chain terminates in kernel-set, exactly once.
        assert c["fallback_chain"][-1] == KERNEL_SET, c
        assert c["fallback_chain"].count(KERNEL_SET) == 1, c
        # INVARIANT 1: the named provider ALWAYS leads its chain (a measured
        # kernel-set winner emits chain ["kernel-set"], so this holds for ks too).
        assert c["fallback_chain"][0] == c["provider"], c
        # measured cells carry provenance.
        if c["source"] == "measured":
            assert c.get("metric"), c
            assert c.get("gpu"), c


def test_no_fp8_or_infeasible_dtype_on_sm75():
    # fp8 has no Tensor-Core support pre-Ada; the table omits it on sm75.
    with open(_OPTIMAL_JSON) as f:
        doc = json.load(f)
    for c in doc["cells"]:
        if c["sm"] == 75:
            assert c["dtype"] != "fp8", f"fp8 must be omitted on sm75: {c}"


def test_no_infeasible_sm_dtype_cells():
    # INVARIANT 3: no cell may carry a dtype its SM cannot run (bf16<80,
    # tf32<80, fp8<89, fp4<100), per the gpu_caps thresholds. sm75/bf16 rows
    # in particular must be absent.
    min_sm = {"bf16": 80, "tf32": 80, "fp8": 89, "fp4": 100}
    with open(_OPTIMAL_JSON) as f:
        doc = json.load(f)
    for c in doc["cells"]:
        floor = min_sm.get(c["dtype"])
        if floor is not None:
            assert c["sm"] >= floor, f"infeasible dtype cell: {c}"
    # No sm75 bf16 anywhere.
    assert not [c for c in doc["cells"]
                if c["sm"] == 75 and c["dtype"] == "bf16"]


def test_every_provider_exists_in_registry_for_its_op():
    # INVARIANT 2: every provider + every fallback_chain entry must be a real
    # runtime provider for that logical_op in the dispatch registry.
    from kernel_set.backends import _registry
    op_providers = {name: {p.name for p in op.providers}
                    for name, op in _registry.OPS.items()}
    with open(_OPTIMAL_JSON) as f:
        doc = json.load(f)
    for c in doc["cells"]:
        known = op_providers.get(c["logical_op"])
        assert known is not None, f"op not in registry: {c}"
        for p in [c["provider"]] + c["fallback_chain"]:
            assert p in known, (
                f"provider {p!r} not registered for {c['logical_op']!r} "
                f"(known: {sorted(known)})")


def test_every_non_kernel_set_chain_entry_is_runtime_selectable():
    from kernel_set.backends import _probe, _registry
    providers = {name: {p.name: p for p in op.providers}
                 for name, op in _registry.OPS.items()}
    with open(_OPTIMAL_JSON) as f:
        doc = json.load(f)
    for c in doc["cells"]:
        known = providers[c["logical_op"]]
        for name in [c["provider"]] + c["fallback_chain"]:
            p = known[name]
            if name == KERNEL_SET:
                continue
            assert p.call is not None, c
            assert int(c["sm"]) >= p.min_sm, c
            assert _probe.dtype_arch_ok(c["dtype"], c["sm"]), c
            assert _probe.dtype_ok(c["dtype"], p.dtypes), c


def test_selector_probe_generator_thresholds_agree():
    import importlib.util
    from kernel_set.backends import _probe

    gen_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "scripts", "gen_optimal.py")
    spec = importlib.util.spec_from_file_location("gen_optimal", gen_path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    for key in ("bf16", "tf32", "fp8", "fp4"):
        assert O._SM_THRESHOLDS[key] == _probe._SM_THRESHOLDS[key]
        assert gen._SM_THRESHOLDS[key] == _probe._SM_THRESHOLDS[key]


def test_sm120_folds_to_sm100_for_selector_and_dispatch():
    from kernel_set.backends import _probe

    assert _probe.gpu_to_sm("rtx5090") == 120
    assert dispatch.resolve_sm("rtx5090") == 100
    assert O.select_optimal("gemm", 120, "bf16") == \
        O.select_optimal("gemm", 100, "bf16")


def test_table_logical_ops_match_dispatch_op_order():
    # INVARIANT 6: the table is Cartesian over the dispatch OP_ORDER — every
    # dispatch op (incl. gemma_rmsnorm + sampling) has >=1 cell, and the table
    # names no op outside OP_ORDER.
    from kernel_set.backends import _registry
    with open(_OPTIMAL_JSON) as f:
        doc = json.load(f)
    table_ops = {c["logical_op"] for c in doc["cells"]}
    order = set(_registry.OP_ORDER)
    assert order - table_ops == set(), (
        f"dispatch ops missing from optimal.json: {order - table_ops}")
    assert table_ops - order == set(), (
        f"optimal.json ops not in OP_ORDER: {table_ops - order}")
    # The two ops the review flagged as missing are now present.
    assert "gemma_rmsnorm" in table_ops
    assert "sampling" in table_ops


# --------------------------------------------------------------------------- #
# select_optimal: measured winners (exact-cell override).
# --------------------------------------------------------------------------- #
def test_measured_winner_attention_prefill_sm89_flashinfer():
    # Dotted taxonomy alias resolves to the canonical op.
    r = O.select_optimal("attention.prefill", 89, "fp16")
    assert r["provider"] == "flashinfer"
    assert r["source"] == "measured"
    assert r["fallback_chain"][-1] == KERNEL_SET
    assert r["fallback_chain"][0] == "flashinfer"
    assert "us" in r["metric"]
    # underscore alias gives the same result.
    assert O.select_optimal("attention_prefill", 89, "fp16")["provider"] == \
        "flashinfer"


def test_measured_winner_rmsnorm_sm80_is_kernel_set():
    # On A100 bf16 the measured rmsnorm winner is kernel-set itself.
    r = O.select_optimal("norm.rmsnorm", 80, "bf16")
    assert r["provider"] == KERNEL_SET
    assert r["source"] == "measured"
    # A measured kernel-set winner emits chain ["kernel-set"] so the invariant
    # provider == chain[0] holds (it cannot sit non-terminally in the chain).
    assert r["fallback_chain"] == [KERNEL_SET]


def test_measured_winner_rmsnorm_sm89_is_liger():
    r = O.select_optimal("rmsnorm", 89, "fp16")
    assert r["provider"] == "liger"
    assert r["source"] == "measured"
    assert r["fallback_chain"][0] == "liger"
    assert r["fallback_chain"][-1] == KERNEL_SET


def test_measured_winner_gemm_sm89_is_torch():
    r = O.select_optimal("gemm.dense", 89, "fp16")
    assert r["provider"] == "torch"
    assert r["source"] == "measured"


def test_measured_cross_entropy_cells():
    # cross_entropy only exists in the table via measured cells (no heuristic
    # fill row): sm89 fp16 -> liger, sm80 bf16 -> kernel-set.
    assert O.select_optimal("loss.cross_entropy", 89, "fp16")["provider"] == \
        "liger"
    r = O.select_optimal("cross_entropy", 80, "bf16")
    assert r["provider"] == KERNEL_SET and r["source"] == "measured"


# --------------------------------------------------------------------------- #
# select_optimal: heuristic fill where no measurement exists.
# --------------------------------------------------------------------------- #
def test_heuristic_fill_rmsnorm_sm90():
    r = O.select_optimal("rmsnorm", 90, "bf16")
    assert r["provider"] == "flashinfer"
    assert r["source"] == "heuristic"   # sm90 unbenched -> heuristic baseline


def test_heuristic_fill_moe_fp8_hopper():
    r = O.select_optimal("moe", 90, "fp8")
    assert r["provider"] == "deep_gemm"
    assert r["source"] == "heuristic"
    assert r["fallback_chain"][-1] == KERNEL_SET


# --------------------------------------------------------------------------- #
# Graceful fallback ladder.
# --------------------------------------------------------------------------- #
def test_nearest_dtype_fallback():
    # rmsnorm sm80 has measured bf16 but the fp16 request has its own heuristic
    # cell; conversely an fp64-ish unknown request degrades to a feasible dtype.
    r = O.select_optimal("rmsnorm", 80, "fp16")
    assert r["provider"] == "flashinfer" and r["source"] == "heuristic"
    # A dtype with no exact cell for this (op, sm) but a near neighbour: gemm
    # sm89 has fp16/bf16/fp32 — request "half" (alias of fp16) resolves exact.
    assert O.select_optimal("gemm", 89, "half")["provider"] == "torch"


def test_fp8_on_sm75_falls_back_to_kernel_set():
    # fp8 is infeasible on sm75 (omitted) and no nearest dtype exists for the
    # fp8-only op fp8_gemm at sm75 -> kernel-set terminal fallback.
    r = O.select_optimal("fp8_gemm", 75, "fp8")
    assert r["provider"] == KERNEL_SET
    assert r["source"] == "fallback"
    assert r["fallback_chain"] == [KERNEL_SET]


def test_infeasible_dtype_never_leaks_into_other_dtype_cell():
    # INVARIANT 4: an arch-infeasible dtype request must resolve to kernel-set
    # IMMEDIATELY, never nearest-dtype-fall into a different-dtype cell (which
    # would let dispatch gate an installed provider against the infeasible dtype).
    # attention_prefill sm75 HAS an fp16 cell; an fp8 request must NOT borrow it.
    r = O.select_optimal("attention_prefill", 75, "fp8")
    assert r["provider"] == KERNEL_SET and r["source"] == "fallback"
    # bf16 on sm75 (rmsnorm has an fp16 cell) must also go straight to ks.
    r = O.select_optimal("rmsnorm", 75, "bf16")
    assert r["provider"] == KERNEL_SET and r["source"] == "fallback"


def test_fp4_does_not_borrow_the_int4_cell():
    # fp4 (NVFP4/MXFP4) must NEVER nearest-fall into an int4 cell: a fp4 request
    # on a dense/int4 op resolves to the kernel-set fallback, not w4a16's int4
    # cell. (fp4 is intentionally absent from _DTYPE_NEAREST.)
    r = O.select_optimal("w4a16", 100, "fp4")
    assert r["provider"] == KERNEL_SET and r["source"] == "fallback"
    # ...and on a dense-gemm op too.
    assert O.select_optimal("gemm", 100, "fp4")["provider"] == KERNEL_SET


def test_fp4_cells_only_for_dedicated_fp4_ops_on_blackwell():
    # NVFP4/MXFP4 are now WIRED: fp4 cells exist, but ONLY for the dedicated fp4
    # ops (nvfp4_gemm / mxfp4_gemm) and ONLY where fp4 is arch-feasible (sm>=100,
    # the gpu_caps threshold). No other op may carry an fp4 cell, and none below
    # sm100.
    fp4_cells = [c for c in O._CELLS.values() if c["dtype"] == "fp4"]
    assert fp4_cells, "NVFP4/MXFP4 are wired now -> fp4 cells must exist"
    assert {c["logical_op"] for c in fp4_cells} == {"nvfp4_gemm", "mxfp4_gemm"}
    assert all(c["sm"] >= 100 for c in fp4_cells), "fp4 is sm100+ only"
    # The dedicated fp4 op selects its real external winner on Blackwell.
    r = O.select_optimal("nvfp4_gemm", 100, "fp4")
    assert r["provider"] == "flashinfer"
    assert r["fallback_chain"][-1] == KERNEL_SET
    # On a non-Blackwell arch fp4 is infeasible -> kernel-set fallback.
    assert O.select_optimal("nvfp4_gemm", 90, "fp4")["provider"] == KERNEL_SET


def test_unknown_op_falls_back_to_kernel_set():
    r = O.select_optimal("not_a_real_op", 90, "bf16")
    assert r["provider"] == KERNEL_SET
    assert r["source"] == "fallback"


def test_unknown_arch_falls_back_to_kernel_set():
    # An SM with no table row (e.g. sm70 Volta) -> kernel-set.
    r = O.select_optimal("rmsnorm", 70, "fp16")
    assert r["provider"] == KERNEL_SET
    assert r["source"] == "fallback"


def test_none_sm_falls_back_to_kernel_set():
    # No arch known (CPU host) -> the selector can't pick an external provider.
    r = O.select_optimal("rmsnorm", None, "fp16")
    assert r["provider"] == KERNEL_SET
    assert r["source"] == "fallback"


def test_select_optimal_is_pure_returns_fresh_dict():
    a = O.select_optimal("rmsnorm", 89, "fp16")
    a["provider"] = "MUTATED"
    b = O.select_optimal("rmsnorm", 89, "fp16")
    assert b["provider"] == "liger", "lookup must not be mutated by the caller"


def test_optimal_chain_helper():
    chain = O.optimal_chain("attention.prefill", 89, "fp16")
    assert chain[0] == "flashinfer"
    assert chain[-1] == KERNEL_SET


# --------------------------------------------------------------------------- #
# The runtime dispatcher consults the table (single source of truth).
# --------------------------------------------------------------------------- #
def _mock_available(monkeypatch, available_libs, sm=None):
    import kernel_set.dispatch as d
    import kernel_set.backends._probe as probe

    def fake_can_import(check: str) -> bool:
        if check in available_libs:
            return True
        mods = probe._module_names(check)
        return any(m in available_libs for m in mods)

    monkeypatch.setattr(d, "can_import", fake_can_import)
    monkeypatch.setattr(probe, "detect_sm", lambda: sm)


@pytest.fixture(autouse=True)
def _clear_cache():
    dispatch.reset_cache()
    yield
    dispatch.reset_cache()


def test_dispatch_applies_measured_override_rmsnorm_sm89(monkeypatch):
    # The measured L4 (sm89 fp16) winner is liger. With BOTH flashinfer (static
    # rank-1) and liger installed, the table's measured override makes dispatch
    # pick liger — proving it consults the optimal table, not the static rank.
    _mock_available(
        monkeypatch, available_libs={"flashinfer", "liger_kernel"}, sm=89)
    assert dispatch.which("rmsnorm", dtype="fp16") == "liger"
    # On sm90 (heuristic, unbenched) the static rank-1 flashinfer wins again.
    dispatch.reset_cache()
    _mock_available(
        monkeypatch, available_libs={"flashinfer", "liger_kernel"}, sm=90)
    assert dispatch.which("rmsnorm", dtype="fp16") == "flashinfer"


def test_dispatch_measured_override_attention_prefill_sm89(monkeypatch):
    # Measured sm89 fp16 winner is flashinfer (FA2 not installed in the bench);
    # with flash-attn (static rank-1) AND flashinfer present, the table promotes
    # flashinfer on sm89 fp16.
    _mock_available(
        monkeypatch, available_libs={"flash_attn", "flashinfer"}, sm=89)
    assert dispatch.which("attention_prefill", dtype="fp16") == "flashinfer"
    # bf16 on sm89 is heuristic -> flash-attn (static rank-1) leads.
    dispatch.reset_cache()
    _mock_available(
        monkeypatch, available_libs={"flash_attn", "flashinfer"}, sm=89)
    assert dispatch.which("attention_prefill", dtype="bf16") == "flash-attn"


def test_dispatch_falls_back_to_ks_when_measured_winner_absent(monkeypatch):
    # Measured winner (liger) not installed, nothing else either -> kernel-set.
    _mock_available(monkeypatch, available_libs=set(), sm=89)
    assert dispatch.which("rmsnorm", dtype="fp16") == KERNEL_SET


def test_dispatch_dtype_arch_gate_blocks_fp8_on_sm75(monkeypatch):
    # INVARIANT 4 (runtime): FlashInfer is min_sm=75 with dtypes "fp16, bf16,
    # fp8", but sm75 hardware has no fp8 Tensor Cores. Even with flashinfer
    # installed, an fp8 request on sm75 must NOT select flashinfer -> kernel-set.
    _mock_available(monkeypatch, available_libs={"flashinfer"}, sm=75)
    assert dispatch.which("attention_decode", dtype="fp8") == KERNEL_SET
    # fp16 on sm75 is feasible -> flashinfer is selected normally.
    dispatch.reset_cache()
    _mock_available(monkeypatch, available_libs={"flashinfer"}, sm=75)
    assert dispatch.which("attention_decode", dtype="fp16") == "flashinfer"


def test_dispatch_dtype_arch_gate_blocks_bf16_on_sm75(monkeypatch):
    # bf16 needs sm80; on sm75 even an installed provider is gated out -> ks.
    _mock_available(monkeypatch, available_libs={"flashinfer", "vllm"}, sm=75)
    assert dispatch.which("rmsnorm", dtype="bf16") == KERNEL_SET


def test_dispatch_matches_every_table_cell_with_extra_providers_available(
        monkeypatch):
    import kernel_set.dispatch as d

    monkeypatch.setattr(d, "can_import", lambda _check: True)
    with open(_OPTIMAL_JSON) as f:
        doc = json.load(f)
    for c in doc["cells"]:
        dispatch.reset_cache()
        got = dispatch.which(
            c["logical_op"], gpu=f"sm{c['sm']}", dtype=c["dtype"])
        assert got == c["provider"], c


# --------------------------------------------------------------------------- #
# Planner (models/select.py) and runtime dispatch agree on the same optimal
# provider for a sampled (model/op, gpu, dtype).
# --------------------------------------------------------------------------- #
def test_planner_and_dispatch_agree_on_sampled_cells(monkeypatch):
    import importlib.util

    sel_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "models", "select.py")
    spec = importlib.util.spec_from_file_location("ks_select", sel_path)
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)

    # (plan_op, gpu, sm, dtype) sampled across archs/dtypes. The installed-lib
    # set is the cell's own chain (minus the kernel-set terminal) — the
    # apples-to-apples host the table's chain describes — so the planner's
    # named optimal_provider must equal dispatch's runtime choice.
    _LIB_OF = {
        "flashinfer": "flashinfer", "liger": "liger_kernel",
        "flash-attn": "flash_attn", "sgl-kernel": "sgl_kernel",
        "vllm": "vllm", "torch": "torch", "deep_gemm": "deep_gemm",
    }
    samples = [
        # L4 (sm89) fp16 measured winners.
        ("attn_norm", 89, "fp16"),    # measured -> liger
        ("rope", 89, "fp16"),         # measured -> liger
        ("attn_prefill", 89, "fp16"),  # measured -> flashinfer
        # A100 (sm80) bf16: measured kernel-set winner (chain == [kernel-set]).
        ("attn_norm", 80, "bf16"),
        # Heuristic fills (no measurement).
        ("rope", 90, "bf16"),         # heuristic -> flashinfer
        ("moe_grouped_gemm", 90, "fp8"),  # heuristic -> deep_gemm
    ]
    for plan_op, sm, scheme in samples:
        cell = sel.optimal_cell(plan_op, sm, scheme)
        assert cell is not None, (plan_op, sm, scheme)
        planner_provider = cell["provider"]

        # Install exactly the chain providers the cell names (the host the table
        # describes), so planner and dispatch are compared apples-to-apples.
        libs = {_LIB_OF[p] for p in cell["fallback_chain"]
                if p in _LIB_OF}
        dispatch.reset_cache()
        _mock_available(monkeypatch, available_libs=libs, sm=sm)
        runtime_op = sel.optimal_lookup_op(plan_op, scheme)
        runtime_provider = dispatch.which(runtime_op, dtype=scheme)

        assert runtime_provider == planner_provider, (
            plan_op, sm, scheme, planner_provider, runtime_provider)


def test_planner_quantized_dense_gemm_annotations_use_quantized_optimal_ops():
    import importlib.util

    sel_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "models", "select.py")
    spec = importlib.util.spec_from_file_location("ks_select", sel_path)
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)

    cases = [
        ("fp8", "h100", "fp8_gemm"),
        ("int8", "a100", "int8_gemm"),
        ("w4a16", "a100", "w4a16"),
    ]
    for dtype, gpu, optimal_op in cases:
        plan = sel.select("llama-3-8b", gpu, dtype)
        sm = sel._optimal_table_sm(plan["sm"])
        scheme = plan["resolved_scheme"]
        entry = plan["ops"]["qkv_proj"]
        expected = O.select_optimal(
            optimal_op, sm, sel._SCHEME_TO_DTYPE.get(scheme, scheme))
        dense = O.select_optimal("gemm", sm, "bf16")
        assert sel.optimal_lookup_op("qkv_proj", scheme) == optimal_op
        assert entry["optimal_provider"] == expected["provider"]
        if expected["provider"] != dense["provider"]:
            assert entry["optimal_provider"] != dense["provider"]
