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
        # the named provider leads the chain UNLESS it is kernel-set itself
        # (a measured ks-winner keeps the heuristic chain order, ks terminal).
        if c["provider"] != KERNEL_SET:
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
    # ks-winner keeps the heuristic chain order, terminating in kernel-set.
    assert r["fallback_chain"][-1] == KERNEL_SET
    assert r["fallback_chain"][0] == "flashinfer"


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
