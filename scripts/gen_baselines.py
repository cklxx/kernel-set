#!/usr/bin/env python3
"""Regenerate benchmarks/baselines.yaml from providers/registry.json.

``benchmarks/baselines.yaml`` is the list of real, current best-in-class
(rank-1) upstream kernels the benchmark harness installs and compares
kernel-set against. It is DERIVED from ``providers/registry.json`` (the merged
industry-operator provider catalog) — this script is its regeneration path so
the two never silently drift.

Usage
-----
    python3 scripts/gen_baselines.py                 # write benchmarks/baselines.yaml
    python3 scripts/gen_baselines.py --check         # diff vs the checked-in file, exit 1 on drift
    python3 scripts/gen_baselines.py --stdout        # print to stdout, do not write

Pure stdlib (json + a small hand-rolled YAML emitter). No third-party imports.

How the registry maps to baselines
-----------------------------------
* One baseline per operator = its **rank-1** provider.
* Operators are bucketed into four top-level YAML sections by ``domain``:
    attention            <- attention, ssm-linear-attn
    gemm_quant           <- gemm-dense, gemm-quant, norm-act-rope, sampling-logitproc
    moe_comm_training    <- moe-comm, loss-optim-misc
    sgl_kernel_aligned   <- the sgl-kernel provider (any rank) for each op whose
                            domain is attention / gemm-quant / norm-act-rope /
                            sampling-logitproc (the "hard-op alignment" targets;
                            MoE/EP and the speculative-tree control op are excluded).
* Per-entry fields are derived from the chosen provider + operator:
    name              "{lib} — {op}"
    lib/import_check/python_call            verbatim from provider
    pip_install                             provider.install (null => build-only)
    notes                                   provider.perf_note
    confidence                              provider.confidence
    ks_op                                   operator.op
    maps_to                                 [] (kernel-set ABI is named via ks_op)
    bench_category                          BENCH_CATEGORY[op] / domain default
    gpu_arch_required                       SM tokens >= the min sm in provider.gpu_arch
    blackwell_only / hopper_only            from that min sm (100 / 90)
    benchable                               False for ep_comm (multi-GPU), else True
    dtypes                                  parsed + canonically ordered from provider.dtypes
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REGISTRY = os.path.join(ROOT, "providers", "registry.json")
OUTPUT = os.path.join(ROOT, "benchmarks", "baselines.yaml")

# --------------------------------------------------------------------------- #
# domain -> top-level YAML section
# --------------------------------------------------------------------------- #
DOMAIN_GROUP = {
    "attention": "attention",
    "ssm-linear-attn": "attention",
    "gemm-dense": "gemm_quant",
    "gemm-quant": "gemm_quant",
    "norm-act-rope": "gemm_quant",
    "sampling-logitproc": "gemm_quant",
    "moe-comm": "moe_comm_training",
    "loss-optim-misc": "moe_comm_training",
}
GROUP_ORDER = ["attention", "gemm_quant", "moe_comm_training", "sgl_kernel_aligned"]

# Domains whose sgl-kernel provider feeds the sgl_kernel_aligned section.
SGL_DOMAINS = {"attention", "gemm-quant", "norm-act-rope", "sampling-logitproc"}
# Ops excluded from sgl_kernel_aligned even though sgl-kernel ships them:
# speculative_verify_tree is a control-flow/verification op, not a single
# benchable kernel.
SGL_EXCLUDE_OPS = {"speculative_verify_tree"}

# --------------------------------------------------------------------------- #
# bench_category: domain default + per-op overrides (matches bench.py buckets).
# --------------------------------------------------------------------------- #
DOMAIN_BENCH_CATEGORY = {
    "attention": "attention",
    "gemm-dense": "gemm",
    "norm-act-rope": "norm_act_rope",
    "moe-comm": "moe",
    "sampling-logitproc": "sampling",
}
BENCH_CATEGORY = {
    # ssm-linear-attn
    "causal_conv1d": "ssm", "mamba1_selective_scan": "ssm",
    "mamba2_ssd_chunk_scan": "ssm",
    "delta_rule": "linear_attention", "gated_delta_rule": "linear_attention",
    "gated_linear_attention": "linear_attention",
    "gated_slot_attention_gsa": "linear_attention", "hgrn2": "linear_attention",
    "lightning_attention": "linear_attention",
    "linear_attention_basic": "linear_attention",
    "retention_retnet": "linear_attention", "rwkv6_wkv": "linear_attention",
    "rwkv7_wkv": "linear_attention",
    "attention_state_merge": "attention",
    # gemm-quant
    "gemm_fp8": "gemm", "fp8_gemm_blockwise": "gemm", "fp8_gemm_scaled_mm": "gemm",
    "awq_gemm": "w4a16", "dequantize_int4": "w4a16", "w4a16_gemm": "w4a16",
    "int4_weight_only_gemm_tinygemm": "w4a16",
    "nf4_fp4_blockwise_quant_linear": "w4a16",
    "dequantize_int8": "w8a8", "int8_gemm_w8a8": "w8a8",
    "int8_llm_int8_linear": "w8a8", "quantize_int8_dynamic": "w8a8",
    "w4a8_gemm": "w4a8", "sparse_2_4_gemm": "sparse_quant",
    "bitnet_gemm": "bitnet",
    "fp4_quantize": "fp4_gemm", "mxfp4_gemm": "fp4_gemm", "nvfp4_gemm": "fp4_gemm",
    "mxfp8_quantize": "quant",
    "dequantize_fp8": "quant", "quantize_fp8_dynamic": "quant",
    "mrope": "rope", "fused_rmsnorm_gated": "rmsnorm",
    "min_p_sampling": "sampling",
    "chain_speculative_sampling": "sampling",
    "apply_token_bitmask": "sampling",
    # loss-optim-misc
    "adafactor_optimizer": "optimizer", "adamw_8bit": "optimizer",
    "adamw_fused": "optimizer", "global_grad_norm_clip": "optimizer",
    "lion_optimizer": "optimizer", "muon_optimizer": "optimizer",
    "sgd_momentum_fused": "optimizer",
    "cross_entropy_fused": "cross_entropy", "dpo_loss": "cross_entropy",
    "fused_linear_cross_entropy": "cross_entropy", "jsd_distillation": "cross_entropy",
    "kl_divergence": "cross_entropy", "orpo_loss": "cross_entropy",
    "preference_losses_simpo_cpo_kto_grpo": "cross_entropy",
    "tvd_loss": "cross_entropy", "z_loss": "cross_entropy",
    "embedding_backward_scatter": "embedding", "embedding_lookup": "embedding",
    "axpby_fused": "misc", "dtype_cast": "misc", "fp8_convert_quantize": "misc",
    "kv_cache_copy_swap_blocks": "misc", "kv_cache_reshape_and_cache": "misc",
    # moe-comm
    "ep_combine_alltoall": "ep_comm", "ep_dispatch_alltoall": "ep_comm",
    "ep_low_latency_dispatch_combine": "ep_comm", "moe_tp_allreduce_fused": "ep_comm",
}

# Categories that cannot be benched on a single GPU (multi-GPU comm).
NON_BENCHABLE_CATEGORIES = {"ep_comm"}
# Prefix the curated catalog adds to non-benchable comm ops' notes.
NON_BENCHABLE_NOTE = "MULTI-GPU comm op: cataloged but NOT single-GPU benchable. "

# The sgl_kernel_aligned section uses the finer bench_sota provider-group buckets
# (matching the live cross-impl harness) instead of the coarse primary buckets.
SGL_BENCH_CATEGORY = {
    "fp8_gemm_blockwise": "w8a8", "fp8_gemm_scaled_mm": "w8a8",
    "rmsnorm": "rmsnorm", "fused_add_rmsnorm": "rmsnorm",
    "gemma_rmsnorm": "rmsnorm", "silu_and_mul": "swiglu", "rope": "rope",
}

# SM tiers the benchmark targets, lowest -> highest.
SM_TIERS = [70, 75, 80, 89, 90, 100, 120]
MIN_BENCH_SM = 80  # ops listed as sm60/sm70 build but the bench targets sm80 baseline


def sm_token(sm: int) -> str:
    return f"sm{sm}"


def bench_category(op: dict) -> str:
    name = op["op"]
    if name in BENCH_CATEGORY:
        return BENCH_CATEGORY[name]
    return DOMAIN_BENCH_CATEGORY.get(op["domain"], op["domain"])


def gpu_arch_required(gpu_arch: str) -> list:
    """SM tokens a baseline needs, from a free-form gpu_arch string.

    Rules (match the curated catalog):
      * ``smNN+``  -> every bench tier >= NN (an open-ended floor).
      * discrete ``smXX/smYY/smZZ`` (no ``+``) -> exactly those tokens.
      * ``any`` / nothing -> all tiers (sm70+).
    """
    import re

    low = gpu_arch.lower()
    plus = re.findall(r"sm(\d+)\+", low)
    discrete = re.findall(r"sm(\d+)", low)
    if plus:
        lo = min(int(x) for x in plus)
        return [sm_token(s) for s in SM_TIERS if s >= lo]
    if discrete:
        ds = sorted(set(int(x) for x in discrete))
        toks = [sm_token(s) for s in ds if s in SM_TIERS]
        return toks or [sm_token(s) for s in ds]
    if "any" in low:
        return [sm_token(s) for s in SM_TIERS]
    return [sm_token(s) for s in SM_TIERS if s >= MIN_BENCH_SM]


def min_sm_of(tokens: list) -> int:
    """Lowest numeric SM from a list of ``smNN`` tokens (for the *_only flags)."""
    nums = [int(t[2:]) for t in tokens if t.startswith("sm") and t[2:].isdigit()]
    return min(nums) if nums else MIN_BENCH_SM


# --------------------------------------------------------------------------- #
# dtype parsing: free-form provider.dtypes -> canonical ordered token list.
# --------------------------------------------------------------------------- #
DTYPE_ORDER = ["fp32", "tf32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "fp8",
               "nvfp4", "mxfp8", "fp4", "nf4", "int8", "int4"]


def parse_dtypes(s: str) -> list:
    """Extract dtype tokens from a free-form description, canonically ordered.

    Resolution rules mirror the curated file: a bare 'fp8' with no e4m3/e5m2
    qualifier and not part of a KV-cache phrase becomes 'fp8_e4m3'; an explicit
    e4m3/e5m2 pair is kept split; tf32 and fp4 family kept distinct.
    """
    low = s.lower()
    found = set()

    if "fp32" in low or "float32" in low:
        found.add("fp32")
    if "tf32" in low:
        found.add("tf32")
    if "bf16" in low or "bfloat16" in low:
        found.add("bf16")
    if "fp16" in low or "float16" in low or "half" in low:
        found.add("fp16")

    has_e4m3 = "e4m3" in low
    has_e5m2 = "e5m2" in low
    if has_e4m3:
        found.add("fp8_e4m3")
    if has_e5m2:
        found.add("fp8_e5m2")
    if "fp8" in low and not (has_e4m3 or has_e5m2):
        # bare fp8 defaults to the e4m3 variant (the catalog's convention).
        found.add("fp8_e4m3")

    if "nvfp4" in low:
        found.add("nvfp4")
    if "mxfp8" in low:
        found.add("mxfp8")
    if "fp4" in low and "nvfp4" not in low and "mxfp4" not in low:
        found.add("fp4")
    if "mxfp4" in low:
        found.add("fp4")
    if "nf4" in low:
        found.add("nf4")
    if "int8" in low or "w8a8" in low:
        found.add("int8")
    if "int4" in low or "w4a16" in low or "uint4" in low or "awq" in low \
            or "gptq" in low:
        found.add("int4")

    ordered = [d for d in DTYPE_ORDER if d in found]
    # any unmatched leftover dtype literals — none expected, but keep stable.
    return ordered or ["fp16"]


# --------------------------------------------------------------------------- #
# entry construction
# --------------------------------------------------------------------------- #
def split_install(install: str):
    """Return (pip_install, build_from_source). A non-pip install command (git
    clone / build-from-source) goes into build_from_source with pip_install
    null; a normal ``pip install ...`` stays in pip_install."""
    if not install:
        return None, None
    if install.strip().lower().startswith("pip install"):
        return install, None
    return None, install


def make_entry(op: dict, provider: dict, sgl: bool = False) -> dict:
    arch = gpu_arch_required(provider.get("gpu_arch", ""))
    min_sm = min_sm_of(arch)
    abi = op.get("kernel_set_abi")
    if sgl:
        cat = SGL_BENCH_CATEGORY.get(op["op"], bench_category(op))
    else:
        cat = bench_category(op)
    benchable = cat not in NON_BENCHABLE_CATEGORIES
    notes = provider.get("perf_note", "")
    if not benchable:
        notes = NON_BENCHABLE_NOTE + notes
    pip_install, build_from_source = split_install(provider.get("install", ""))
    return {
        "name": f"{provider['lib']} — {op['op']}",
        "lib": provider["lib"],
        "pip_install": pip_install,
        "build_from_source": build_from_source,
        "import_check": provider.get("import_check", ""),
        "python_call": provider.get("python_call", ""),
        "maps_to": [abi] if abi else [],
        "ks_op": op["op"],
        "bench_category": cat,
        "gpu_arch_required": arch,
        "blackwell_only": min_sm >= 100,
        "hopper_only": min_sm == 90,
        "dtypes": parse_dtypes(provider.get("dtypes", "")),
        "benchable": benchable,
        "confidence": provider.get("confidence", "medium"),
        "notes": notes,
    }


# Field emission order (matches the checked-in schema).
FIELD_ORDER = [
    "name", "lib", "pip_install", "build_from_source", "import_check",
    "python_call", "maps_to", "ks_op", "bench_category", "gpu_arch_required",
    "blackwell_only", "hopper_only", "dtypes", "benchable", "confidence",
    "notes",
]


def build(registry: dict):
    ops = registry["operators"]
    sections = {g: [] for g in GROUP_ORDER}

    for op in ops:
        rank1 = next((p for p in op["providers"] if p.get("rank") == 1), None)
        if rank1 is None and op["providers"]:
            rank1 = sorted(op["providers"], key=lambda p: p.get("rank", 99))[0]
        if rank1 is None:
            continue
        group = DOMAIN_GROUP.get(op["domain"])
        if group:
            sections[group].append(make_entry(op, rank1))

        # sgl_kernel_aligned: the sgl-kernel provider for "hard-op" domains.
        if op["domain"] in SGL_DOMAINS and op["op"] not in SGL_EXCLUDE_OPS:
            sgl = next((p for p in op["providers"] if p["lib"] == "sgl-kernel"),
                       None)
            if sgl is not None:
                sections["sgl_kernel_aligned"].append(
                    make_entry(op, sgl, sgl=True))

    return sections


# --------------------------------------------------------------------------- #
# tiny YAML emitter (block style, stdlib only). Produces standard, loadable
# YAML; does not attempt the cosmetic line-wrapping of the hand-edited file.
# --------------------------------------------------------------------------- #
def _needs_quote(s: str) -> bool:
    if s == "":
        return True
    if s in ("null", "true", "false", "yes", "no", "~"):
        return True
    if s[0] in "!&*?|>%@`\"'#,[]{}":
        return True
    if s[0] in " -" or s[-1] == " ":
        return True
    if ": " in s or s.endswith(":") or " #" in s:
        return True
    if s[0].isdigit():
        try:
            float(s)
            return True
        except ValueError:
            pass
    return False


def _scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if _needs_quote(s):
        return "'" + s.replace("'", "''") + "'"
    return s


def emit_entry(entry: dict, out: list):
    first = True
    for key in FIELD_ORDER:
        val = entry.get(key)
        prefix = "- " if first else "  "
        first = False
        if isinstance(val, list):
            if not val:
                out.append(f"{prefix}{key}: []")
            else:
                out.append(f"{prefix}{key}:")
                for item in val:
                    out.append(f"  - {_scalar(item)}")
        else:
            out.append(f"{prefix}{key}: {_scalar(val)}")


HEADER = """\
# kernel-set external baselines — real upstream SOTA kernels to compare against.
#
# !! AUTO-GENERATED by scripts/gen_baselines.py from providers/registry.json. !!
# !! Do not edit by hand; re-run `python3 scripts/gen_baselines.py` instead.  !!
#
# PURPOSE
#   This is the config that makes the bench "comprehensive": instead of only
#   comparing kernel-set vs a naive PyTorch reference, the harness installs the
#   real, current best-in-class (rank-1) provider for each op and compares
#   against it. One rank-1 baseline per operator, derived from
#   providers/registry.json (the merged industry-operator provider catalog).
#
# SCOPE
#   2026-H1 best-in-class providers (see docs/OPERATOR_CATALOG.md for the full
#   ranked catalog with every provider, exact python_call, install, gpu_arch,
#   dtypes, perf_note and source link).
#
# SCHEMA (per baseline entry)
#   name              display name (lib — op)
#   lib               provider library
#   pip_install       shell command(s) to install (null => build-from-source only)
#   build_from_source git/build note if pip is not the canonical path (else null)
#   import_check      python expr that must succeed to confirm the lib is usable
#   python_call       the exact rank-1 call the bench should drive
#   maps_to           kernel-set op ABI symbol(s) this baseline competes with (ks_*)
#   ks_op             the registry operator name this baseline implements
#   bench_category    bench.py op category bucket (attention/gemm/w8a8/w4a16/moe/...)
#   gpu_arch_required min/required SM archs; bench should SKIP (not error) otherwise
#   blackwell_only    true => needs sm100+/sm120 (Blackwell); skip on T4/L4/A100/Hopper
#   hopper_only       true => needs sm90 (WGMMA/TMA); skip on T4/L4/A100 and consumer
#   dtypes            dtypes the baseline supports for this op
#   benchable         false => multi-GPU / comm-only, cannot be benched single-GPU here
#   confidence        high | medium | low  (from the research catalog)
#   notes             perf note / anything the harness/operator should know
#
# ARCH TOKENS: sm70(V100) sm75(T4) sm80(A100) sm89(L4/4090) sm90(Hopper H100/H200/H800)
#              sm100(Blackwell DC: B200/GB200/GB300) sm120(consumer Blackwell/RTX50xx)
#
# COLAB TARGETS are T4(sm75)/L4(sm89)/A100(sm80): every blackwell_only or
# hopper_only entry below will SKIP there — that is expected, not a failure.
"""


def render(sections: dict, registry: dict) -> str:
    out = [HEADER.rstrip("\n"), ""]
    out.append("schema_version: 2")
    out.append(f"generated: '{registry.get('generated', '2026-06-05')}'")
    out.append(f"generated_for: {registry.get('generated_for', '2026-H1')}")
    out.append("source_registry: providers/registry.json")
    out.append("catalog_doc: docs/OPERATOR_CATALOG.md")
    for group in GROUP_ORDER:
        entries = sections.get(group, [])
        out.append(f"{group}:")
        for e in entries:
            emit_entry(e, out)
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=REGISTRY, help="providers/registry.json")
    ap.add_argument("--output", default=OUTPUT, help="baselines.yaml to write")
    ap.add_argument("--stdout", action="store_true", help="print, do not write")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the checked-in file differs from regen")
    args = ap.parse_args(argv)

    with open(args.registry) as f:
        registry = json.load(f)

    sections = build(registry)
    text = render(sections, registry)
    n = sum(len(v) for v in sections.values())

    if args.stdout:
        sys.stdout.write(text)
        return 0

    if args.check:
        try:
            with open(args.output) as f:
                current = f.read()
        except FileNotFoundError:
            current = ""
        if current == text:
            print(f"baselines.yaml is up to date ({n} entries).")
            return 0
        print("baselines.yaml differs from regeneration "
              "(run scripts/gen_baselines.py).", file=sys.stderr)
        return 1

    with open(args.output, "w") as f:
        f.write(text)
    print(f"wrote {args.output} ({n} entries across "
          f"{len([g for g in GROUP_ORDER if sections.get(g)])} sections).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
