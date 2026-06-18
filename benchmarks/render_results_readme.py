#!/usr/bin/env python3
"""Render benchmark result indexes and README summaries from canonical runs."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from persist import derive_row_metadata, validate_run  # noqa: E402


START = "<!-- BENCHMARK_SUMMARY:START -->"
END = "<!-- BENCHMARK_SUMMARY:END -->"

DEFAULT_INFERENCE = "benchmarks/results/inference/*.json"


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _load_runs(paths: Sequence[str]) -> List[Dict[str, Any]]:
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "*.json"))))
        else:
            files.extend(sorted(glob.glob(path)))
    runs: List[Dict[str, Any]] = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        errors = validate_run(run, path)
        if errors:
            raise ValueError("\n".join(errors))
        run["_path"] = path
        runs.append(run)
    runs.sort(key=lambda r: (str(r.get("timestamp") or ""), str(r.get("run_id") or "")))
    return runs


def _load_inference(paths: Sequence[str]) -> List[Dict[str, Any]]:
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "*.json"))))
        else:
            files.extend(sorted(glob.glob(path)))
    out: List[Dict[str, Any]] = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            item = json.load(f)
        item["_path"] = path
        out.append(item)
    out.sort(key=lambda r: (str(r.get("timestamp") or ""), str(r.get("run_id") or "")))
    return out


def _fmt_num(x: Any, digits: int = 1) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        if math.isnan(float(x)):
            return "-"
        return f"{float(x):.{digits}f}"
    return str(x)


def _fmt_latency(x: Any) -> str:
    return f"{_fmt_num(x, 1)} us" if x is not None else "-"


def _status_counts(rows: Iterable[Dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts[str(row.get("status") or "unknown")] += 1
    return counts


def _row_group_fields(row: Dict[str, Any]) -> Tuple[str, str]:
    meta = derive_row_metadata(row)
    model_part = str(row.get("model_part") or meta["model_part"])
    position_kind = str(row.get("position_kind_row") or meta["position_kind_row"])
    return model_part, position_kind


def _source_link(path: Optional[str], base_dir: Optional[str] = None) -> str:
    if not path:
        return "-"
    if base_dir:
        base_abs = os.path.abspath(base_dir)
        path_abs = path if os.path.isabs(path) else os.path.abspath(path)
        rel = os.path.relpath(path_abs, base_abs)
    else:
        rel = path
        if os.path.isabs(path):
            try:
                rel = os.path.relpath(path, os.getcwd())
            except ValueError:
                rel = path
    return f"[{os.path.basename(rel)}]({rel})"


_LARGE_KERNEL_PRIORITY = {
    "attention_prefill": 0,
    "attn_prefill": 0,
    "attention_decode": 1,
    "attn_decode": 1,
    "mla_decode": 2,
    "gemm": 3,
    "gemm_bf16": 3,
    "gemm_fp16": 3,
    "fp8_gemm": 4,
    "fp8_gemm_blockwise": 4,
    "w8a8": 5,
    "w4a16": 6,
    "w4a8": 7,
    "moe_grouped_gemm": 8,
    "fused_moe": 8,
    "moe_gate": 9,
    "moe_permute": 10,
    "moe_unpermute": 11,
}

_MEMORY_KERNEL_PRIORITY = {
    "fused_add_rmsnorm": 0,
    "rmsnorm": 1,
    "swiglu": 2,
    "geglu": 3,
    "rope": 4,
    "argmax": 5,
    "log_softmax": 6,
}

_LARGE_KERNEL_CANONICAL = {
    "attn_prefill": "attention_prefill",
    "attention_prefill": "attention_prefill",
    "attention_decode": "attn_decode",
    "attn_decode": "attn_decode",
    "mla_decode": "mla_decode",
    "gemm": "gemm",
    "gemm_bf16": "gemm",
    "gemm_fp16": "gemm",
    "fp8_gemm": "fp8_gemm",
    "fp8_gemm_blockwise": "fp8_gemm_blockwise",
    "w8a8": "w8a8",
    "w4a16": "w4a16",
    "w4a8": "w4a8",
    "moe_grouped_gemm": "moe_grouped_gemm",
    "fused_moe": "moe_grouped_gemm",
    "moe_gate": "moe_gate",
    "moe_permute": "moe_permute",
    "moe_unpermute": "moe_unpermute",
}


def _is_large_kernel(op: Any) -> bool:
    return str(op) in _LARGE_KERNEL_PRIORITY


def _is_memory_kernel(op: Any) -> bool:
    return str(op) in _MEMORY_KERNEL_PRIORITY


def _gpu_rank(name: Any) -> int:
    text = str(name or "").lower()
    if "h100" in text or "h20" in text or "h200" in text:
        return 0
    if "pro 6000" in text or "blackwell" in text:
        return 1
    if "a100" in text:
        return 2
    if "l4" in text:
        return 3
    return 9


def _canonical_large_op(op: Any) -> str:
    return _LARGE_KERNEL_CANONICAL.get(str(op), str(op))


def _run_summary(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        counts = _status_counts(run.get("rows") or [])
        rows.append({
            "run_id": run["run_id"],
            "timestamp": run.get("timestamp"),
            "gpu": run.get("gpu_name"),
            "sm": run.get("gpu_sm"),
            "suite": run.get("suite"),
            "dtype": run.get("dtype"),
            "timing": run.get("timing_profile"),
            "rows": len(run.get("rows") or []),
            "ok": counts.get("ok", 0),
            "skip": counts.get("skip", 0),
            "import_fail": counts.get("import-fail", 0),
            "error": counts.get("error", 0),
            "incorrect": counts.get("incorrect", 0),
            "source_report": run.get("source_report"),
            "run_json": run.get("_path"),
            "imported_from": run.get("imported_from"),
        })
    return rows


def _best_comparisons(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    run_meta: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        run_meta[run["run_id"]] = run
        for row in run.get("rows") or []:
            if row.get("status") == "ok" and row.get("latency_us") is not None:
                item = dict(row)
                item["_run_id"] = run["run_id"]
                grouped[f"{run['run_id']}|{row.get('comparison_key')}"].append(item)

    out: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        impl_best: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            impl = str(row.get("impl"))
            prev = impl_best.get(impl)
            if prev is None or float(row["latency_us"]) < float(prev["latency_us"]):
                impl_best[impl] = row
        if len(impl_best) < 2:
            continue
        ordered = sorted(impl_best.values(), key=lambda r: float(r["latency_us"]))
        winner = ordered[0]
        second = ordered[1]
        run = run_meta[winner["_run_id"]]
        speedup = float(second["latency_us"]) / float(winner["latency_us"])
        model_part, position_kind = _row_group_fields(winner)
        out.append({
            "comparison_key": key,
            "model_part": model_part,
            "position_kind_row": position_kind,
            "gpu": run.get("gpu_name"),
            "sm": run.get("gpu_sm"),
            "suite": run.get("suite"),
            "dtype": winner.get("dtype"),
            "timing": run.get("timing_profile"),
            "context": run.get("context"),
            "op": winner.get("op"),
            "shape": winner.get("shape"),
            "winner": winner.get("impl"),
            "winner_latency_us": winner.get("latency_us"),
            "runner_up": second.get("impl"),
            "runner_up_latency_us": second.get("latency_us"),
            "winner_vs_next": speedup,
            "impl_count": len(impl_best),
            "run_id": winner["_run_id"],
            "source_report": run.get("source_report"),
            "run_json": run.get("_path"),
        })
    out.sort(key=lambda r: (
        str(r.get("gpu")),
        str(r.get("suite")),
        str(r.get("op")),
        -float(r.get("winner_vs_next") or 0.0),
    ))
    return out


def _representative_rows(
    runs: List[Dict[str, Any]],
    predicate,
    limit: int = 12,
    unique_by: str = "op_gpu",
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for run in runs:
        for row in run.get("rows") or []:
            if row.get("status") != "ok" or row.get("latency_us") is None:
                continue
            if row.get("source_role") == "baseline":
                continue
            if not predicate(row.get("op")):
                continue
            model_part, position_kind = _row_group_fields(row)
            item = dict(row)
            item["model_part"] = model_part
            item["position_kind_row"] = position_kind
            item["_run_id"] = run.get("run_id")
            item["_timestamp"] = run.get("timestamp")
            item["_gpu"] = run.get("gpu_name")
            item["_sm"] = run.get("gpu_sm")
            item["_suite"] = run.get("suite")
            item["_dtype"] = run.get("dtype")
            item["_run_json"] = run.get("_path")
            candidates.append(item)

    def priority(row: Dict[str, Any]) -> Tuple[Any, ...]:
        op = str(row.get("op"))
        pri = _LARGE_KERNEL_PRIORITY if _is_large_kernel(op) else _MEMORY_KERNEL_PRIORITY
        canon = _canonical_large_op(op) if _is_large_kernel(op) else op
        # Prefer SOTA rows for provider comparisons, otherwise latest clean
        # kernel_set rows. This keeps the root table useful without hiding
        # standalone MoE rows that have no second provider yet.
        suite_rank = 0 if row.get("_suite") == "sota" else 1
        return (
            pri.get(canon, pri.get(op, 100)),
            suite_rank,
            _gpu_rank(row.get("_gpu")),
            float(row.get("latency_us") or math.inf),
            str(row.get("shape") or ""),
        )

    candidates.sort(key=priority)
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in candidates:
        op = str(row.get("op"))
        canon = _canonical_large_op(op) if _is_large_kernel(op) else op
        if unique_by == "op":
            key = canon
        elif unique_by == "op_gpu":
            key = (canon, row.get("_gpu"))
        else:
            key = (canon, row.get("_gpu"), row.get("shape"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _render_representative_row_table(
    rows: List[Dict[str, Any]],
    limit: int = 12,
    base_dir: Optional[str] = None,
) -> List[str]:
    lines = [
        "| part | op | GPU / suite | shape | impl | latency | source |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| `{row.get('model_part')}` | `{row.get('op')}` | "
            f"{row.get('_gpu')} (sm{row.get('_sm')}, {row.get('_dtype')}, {row.get('_suite')}) "
            f"| `{row.get('shape')}` | `{row.get('impl')}` | "
            f"{_fmt_latency(row.get('latency_us'))} | "
            f"{_source_link(row.get('_run_json'), base_dir)} |"
        )
    return lines


def build_index(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "runs": _run_summary(runs),
        "comparisons": _best_comparisons(runs),
    }


def _render_run_table(
    runs: List[Dict[str, Any]],
    limit: int = 12,
    base_dir: Optional[str] = None,
) -> List[str]:
    lines = [
        "| run | GPU | suite | dtype | timing | rows | ok / skip / import-fail / error | data |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for run in reversed(runs[-limit:]):
        counts = _status_counts(run.get("rows") or [])
        lines.append(
            f"| `{run['run_id']}` | {run.get('gpu_name')} (sm{run.get('gpu_sm')}) "
            f"| {run.get('suite')} | {run.get('dtype')} | {run.get('timing_profile')} "
            f"| {len(run.get('rows') or [])} | "
            f"{counts.get('ok', 0)} / {counts.get('skip', 0)} / "
            f"{counts.get('import-fail', 0)} / {counts.get('error', 0)} "
            f"| {_source_link(run.get('_path'), base_dir)} |"
        )
    return lines


def _coverage_summary(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for run in runs:
        gpu = f"{run.get('gpu_name')} sm{run.get('gpu_sm')}"
        for row in run.get("rows") or []:
            model_part, position_kind = _row_group_fields(row)
            suite = str(row.get("suite") or run.get("suite") or "unknown")
            key = (model_part, position_kind, suite)
            item = grouped.setdefault(key, {
                "model_part": model_part,
                "position_kind": position_kind,
                "suite": suite,
                "rows": 0,
                "ok": 0,
                "ops": set(),
                "shapes": set(),
                "impls": set(),
                "gpus": set(),
            })
            item["rows"] += 1
            if row.get("status") == "ok":
                item["ok"] += 1
            item["ops"].add(str(row.get("op")))
            item["shapes"].add(str(row.get("shape")))
            item["impls"].add(str(row.get("impl")))
            item["gpus"].add(gpu)

    out = list(grouped.values())
    out.sort(key=lambda r: (
        str(r["model_part"]),
        str(r["position_kind"]),
        str(r["suite"]),
    ))
    return out


def _render_coverage_table(runs: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| model part | position | suite | ops | shapes | impls | GPUs | ok / total |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in _coverage_summary(runs):
        lines.append(
            f"| `{row['model_part']}` | `{row['position_kind']}` | {row['suite']} "
            f"| {len(row['ops'])} | {len(row['shapes'])} | {len(row['impls'])} "
            f"| {len(row['gpus'])} | {row['ok']} / {row['rows']} |"
        )
    return lines


def _grouped_winners(comparisons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str, str, str], Dict[str, Any]] = {}
    for row in comparisons:
        model_part, position_kind = _row_group_fields(row)
        key = (
            model_part,
            position_kind,
            str(row.get("op")),
            str(row.get("gpu")),
            str(row.get("suite")),
            str(row.get("dtype")),
            str(row.get("timing")),
        )
        item = dict(row)
        item["model_part"] = model_part
        item["position_kind_row"] = position_kind
        prev = grouped.get(key)
        if prev is None:
            grouped[key] = item
            continue
        ratio = float(item.get("winner_vs_next") or 0.0)
        prev_ratio = float(prev.get("winner_vs_next") or 0.0)
        if ratio > prev_ratio:
            grouped[key] = item
        elif ratio == prev_ratio:
            lat = float(item.get("winner_latency_us") or math.inf)
            prev_lat = float(prev.get("winner_latency_us") or math.inf)
            if lat < prev_lat:
                grouped[key] = item

    out = list(grouped.values())
    out.sort(key=lambda r: (
        str(r.get("model_part")),
        str(r.get("position_kind_row")),
        str(r.get("op")),
        0 if r.get("suite") == "sota" else 1,
        str(r.get("gpu")),
        -float(r.get("winner_vs_next") or 0.0),
    ))
    return out


def _render_grouped_winner_table(
    comparisons: List[Dict[str, Any]],
    limit: int = 24,
    base_dir: Optional[str] = None,
) -> List[str]:
    lines = [
        "| model part | position | op | GPU / suite | shape | winner | runner-up | ratio | source |",
        "|---|---|---|---|---|---|---|---:|---|",
    ]
    for row in _grouped_winners(comparisons)[:limit]:
        lines.append(
            f"| `{row.get('model_part')}` | `{row.get('position_kind_row')}` "
            f"| `{row.get('op')}` | {row.get('gpu')} (sm{row.get('sm')}, "
            f"{row.get('dtype')}, {row.get('suite')}) | `{row.get('shape')}` | "
            f"`{row.get('winner')}` {_fmt_latency(row.get('winner_latency_us'))} | "
            f"`{row.get('runner_up')}` {_fmt_latency(row.get('runner_up_latency_us'))} | "
            f"{float(row.get('winner_vs_next') or 0.0):.2f}x | "
            f"{_source_link(row.get('run_json'), base_dir)} |"
        )
    return lines


def _latest_inference_run(runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    standard = [
        run
        for run in runs
        if run.get("kind") != "quantized_engine_compare" and isinstance(run.get("engines"), dict)
    ]
    if not standard:
        return None
    return standard[-1]


def _latest_quantized_engine_run(runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    quant = [
        run
        for run in runs
        if run.get("kind") == "quantized_engine_compare"
        and isinstance(run.get("variants"), dict)
    ]
    if not quant:
        return None
    return quant[-1]


def _engine_exact(engine: Dict[str, Any]) -> str:
    exact = engine.get("exact_same_as_reference")
    if exact is True:
        return "yes"
    if exact is False:
        prefix = engine.get("token_match_prefix")
        overlap = engine.get("token_overlap")
        if prefix is not None and overlap is not None:
            return f"no ({prefix}/{overlap})"
        return "no"
    return "-"


def _render_inference_table(
    inference_runs: List[Dict[str, Any]],
    base_dir: Optional[str] = None,
) -> List[str]:
    run = _latest_inference_run(inference_runs)
    if not run:
        return ["No inference-engine smoke runs checked in yet."]
    lines = [
        "| model / GPU | engine | scope | new tok/s | token match | notes | source |",
        "|---|---|---|---:|---|---|---|",
    ]
    engines = run.get("engines") or {}
    order = [
        "transformers",
        "vllm",
        "sglang",
        "kernel_set_best_practice",
        "kernel_set_full_kernels",
        "kernel_set_ops",
        "kernel_set_full_smoke",
    ]
    for name in order + sorted(k for k in engines if k not in order):
        if name not in engines:
            continue
        engine = engines[name] or {}
        if engine.get("status") == "not_run" or "error" in engine:
            continue
        tps = _fmt_num(engine.get("tokens_per_s_new"), 2)
        match = _engine_exact(engine)
        note = str(engine.get("note") or "")
        scope = str(engine.get("scope") or "single prompt greedy")
        lines.append(
            f"| {run.get('model')} / {run.get('gpu_name')} "
            f"(sm{run.get('gpu_sm')}, {run.get('dtype')}) | `{name}` | "
            f"{scope} | {tps} | {match} | {note} | "
            f"{_source_link(run.get('_path'), base_dir)} |"
        )
    coverage = _render_inference_kernel_coverage(run)
    microbench = _render_inference_kernel_microbench(run)
    if microbench:
        lines.append("")
        lines.extend(microbench)
    if coverage:
        lines.append("")
        lines.extend(coverage)
    ablation = _render_inference_ablation(run)
    if ablation:
        lines.append("")
        lines.extend(ablation)
    return lines


def _render_quantized_engine_table(
    inference_runs: List[Dict[str, Any]],
    base_dir: Optional[str] = None,
) -> List[str]:
    run = _latest_quantized_engine_run(inference_runs)
    if not run:
        return []
    lines = [
        "Quantized checkpoint engine smoke:",
        "",
        "| model / GPU | quant mode | engine | Transformers tok/s | engine tok/s | engine / HF | token match | peak GB | source |",
        "|---|---|---|---:|---:|---:|---|---:|---|",
    ]
    for mode, variant in (run.get("variants") or {}).items():
        if not isinstance(variant, dict) or variant.get("status") != "ok":
            continue
        engines = variant.get("engines") or {}
        hf = engines.get("transformers") or {}
        hf_tps = hf.get("tokens_per_s_new")
        if hf_tps is None:
            continue
        display_engines = sorted(
            k
            for k in engines
            if k.startswith("kernel_set_") or k == "manual_torch_ops"
        )
        for engine_name in display_engines:
            engine = engines.get(engine_name) or {}
            if engine.get("exact_same_as_reference") is not True:
                continue
            engine_tps = engine.get("tokens_per_s_new")
            if engine_tps is None:
                continue
            ratio = float(engine_tps) / float(hf_tps) if float(hf_tps) else None
            peak = engine.get("peak_memory_gb") or hf.get("peak_memory_gb")
            lines.append(
                f"| {run.get('model')} / {run.get('gpu_name')} "
                f"(sm{run.get('gpu_sm')}, {run.get('dtype')}) | `{mode}` | "
                f"`{engine_name}` | {_fmt_num(hf_tps, 2)} | "
                f"{_fmt_num(engine_tps, 2)} | {_fmt_num(ratio, 2)}x | "
                f"{_engine_exact(engine)} | {_fmt_num(peak, 2)} | "
                f"{_source_link(run.get('_path'), base_dir)} |"
            )
    if len(lines) == 3:
        return []
    lines.extend(
        [
            "",
            "Rows are real checkpoint loads with greedy decode and exact token parity against the same quantized Transformers model. Non-exact diagnostic engine rows remain in the JSON but are not displayed as comparison data.",
        ]
    )
    return lines


def _fmt_pct(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        if math.isnan(float(x)):
            return "-"
        return f"{float(x):+.1f}%"
    return str(x)


def _mode_changes(modes: Dict[str, Any]) -> str:
    changes = []
    for key in sorted(BEST_PRACTICE_BASELINE):
        value = str(modes.get(key) or "")
        if value and value != BEST_PRACTICE_BASELINE[key]:
            changes.append(f"{key}={value}")
    return "<br>".join(changes) if changes else "baseline"


BEST_PRACTICE_BASELINE = {
    "argmax": "torch",
    "attention": "auto",
    "cache": "ks",
    "embedding": "auto",
    "linear": "torch",
    "norm": "ks",
    "rope": "ks",
    "swiglu": "ks",
}


def _fmt_microbench_ratio(row: Dict[str, Any]) -> str:
    ratio = row.get("speedup_ref_over_ks")
    if ratio is None:
        return "-"
    ratio = float(ratio)
    if ratio >= 1.0:
        return f"kernel-set {ratio:.2f}x"
    ref_impl = str(row.get("ref_impl") or "reference")
    return f"{ref_impl} {1.0 / ratio:.2f}x"


def _fmt_microbench_err(row: Dict[str, Any]) -> str:
    if row.get("exact") is True:
        return "exact"
    rel = row.get("rel_err")
    if rel is None:
        return "-"
    return f"rel {_fmt_num(rel, 4)}"


def _render_inference_kernel_microbench(run: Dict[str, Any]) -> List[str]:
    bench = run.get("kernel_microbench") or {}
    rows = [
        row for row in bench.get("rows") or []
        if isinstance(row, dict) and row.get("status") == "ok"
    ]
    if not rows:
        return []
    out = [
        "Qwen3-shape kernel microbench:",
        "",
        "| op | shape | kernel-set | reference | winner | ratio | err |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for row in rows:
        ref_impl = str(row.get("ref_impl") or "reference")
        ref = f"`{ref_impl}` {_fmt_latency(row.get('ref_us'))}"
        dense = row.get("torch_dense_sdpa_us")
        if dense is not None:
            ref += f"<br>`torch_dense_sdpa` {_fmt_latency(dense)}"
        out.append(
            f"| `{row.get('op')}` | `{row.get('shape')}` | "
            f"{_fmt_latency(row.get('ks_us'))} | {ref} | "
            f"`{row.get('winner')}` | {_fmt_microbench_ratio(row)} | "
            f"{_fmt_microbench_err(row)} |"
        )
    out.extend([
        "",
        "These are per-kernel CUDA-event timings at the Qwen3-8B shapes used by the engine smoke; they are provider-selection evidence, not serving throughput.",
    ])
    return out


def _render_inference_ablation(run: Dict[str, Any]) -> List[str]:
    ablation = run.get("optimization_ablation") or {}
    variants = ablation.get("variants") or []
    if not variants:
        return []
    rows = [
        "Composition ablation from the best-practice path:",
        "",
        "| variant | new tok/s | vs best-practice | token match | changed component | notes |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in variants:
        if not isinstance(row, dict):
            continue
        rows.append(
            f"| `{row.get('name')}` | {_fmt_num(row.get('tokens_per_s_new'), 2)} "
            f"| {_fmt_pct(row.get('vs_best_practice_pct'))} | {_engine_exact(row)} "
            f"| {_mode_changes(row.get('op_modes') or {})} | {row.get('note') or ''} |"
        )
    note = ablation.get("note")
    if note:
        rows.extend(["", str(note)])
    return rows


def _render_inference_kernel_coverage(run: Dict[str, Any]) -> List[str]:
    engines = run.get("engines") or {}
    rows: List[str] = []
    for name, engine in engines.items():
        if not isinstance(engine, dict) or "error" in engine:
            continue
        coverage = engine.get("kernel_coverage") or {}
        stats = engine.get("stats") or {}
        covered = coverage.get("covered") or []
        if not covered and not stats:
            continue
        fallbacks = coverage.get("torch_fallback") or []
        stat_items = [
            (str(k).removeprefix("ks_").removesuffix("_calls"), v)
            for k, v in sorted(stats.items())
            if v not in (None, 0)
        ]
        rows.append(
            "| "
            + f"`{name}` | "
            + "<br>".join(f"`{item}`" for item in covered)
            + " | "
            + ("<br>".join(fallbacks) if fallbacks else "-")
            + " | "
            + ("<br>".join(f"`{k}`={v}" for k, v in stat_items) if stat_items else "-")
            + " |"
        )
    if not rows:
        return []
    return [
        "Kernel coverage for the composed kernel-set engine:",
        "",
        "| engine | covered kernel-set kernels | remaining torch/Python path | counted calls |",
        "|---|---|---|---|",
        *rows,
    ]


def _render_comparison_table(
    comparisons: List[Dict[str, Any]],
    limit: int = 24,
    base_dir: Optional[str] = None,
) -> List[str]:
    lines = [
        "| GPU | op | shape | winner | runner-up | ratio | source |",
        "|---|---|---|---|---|---:|---|",
    ]
    selected = sorted(
        comparisons,
        key=lambda r: (
            0 if r.get("suite") == "sota" else 1,
            str(r.get("gpu")),
            str(r.get("op")),
            -float(r.get("winner_vs_next") or 0.0),
        ),
    )[:limit]
    for row in selected:
        lines.append(
            f"| {row.get('gpu')} (sm{row.get('sm')}, {row.get('dtype')}) "
            f"| `{row.get('op')}` | `{row.get('shape')}` | "
            f"`{row.get('winner')}` {_fmt_latency(row.get('winner_latency_us'))} | "
            f"`{row.get('runner_up')}` {_fmt_latency(row.get('runner_up_latency_us'))} | "
            f"{float(row.get('winner_vs_next') or 0.0):.2f}x | "
            f"{_source_link(row.get('run_json'), base_dir)} |"
        )
    return lines


def render_results_readme(
    runs: List[Dict[str, Any]],
    index_path: str,
    output_path: str,
    inference_runs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    comparisons = _best_comparisons(runs)
    base_dir = os.path.dirname(os.path.abspath(output_path))
    gpus = sorted({f"{r.get('gpu_name')} (sm{r.get('gpu_sm')})" for r in runs})
    suites = Counter(str(r.get("suite")) for r in runs)
    large_rows = _representative_rows(runs, _is_large_kernel, limit=24)
    memory_rows = _representative_rows(runs, _is_memory_kernel, limit=12)
    lines: List[str] = []
    lines.append("# kernel-set benchmark results")
    lines.append("")
    lines.append("This directory keeps human-readable benchmark reports plus canonical")
    lines.append("JSON runs under `runs/`. Use the JSON files as the durable data")
    lines.append("source; Markdown files are display artifacts.")
    lines.append("")
    lines.append(f"- **Canonical runs:** {len(runs)}")
    lines.append(f"- **GPU coverage:** {', '.join(gpus) if gpus else '-'}")
    lines.append(f"- **Suites:** " + ", ".join(f"{k}={v}" for k, v in sorted(suites.items())))
    lines.append(f"- **Index:** [`{os.path.basename(index_path)}`]({os.path.basename(index_path)})")
    lines.append("")
    lines.append("## Latest Runs")
    lines.append("")
    lines.extend(_render_run_table(runs, base_dir=base_dir))
    lines.append("")
    lines.append("## Model-Part Coverage")
    lines.append("")
    lines.extend(_render_coverage_table(runs))
    lines.append("")
    lines.append("## Representative Large Kernels")
    lines.append("")
    lines.append("These rows keep the README focused on the model-dominant kernels: attention,")
    lines.append("MLA, GEMM/FP8 GEMM, and MoE routing/dispatch. They may be single-provider")
    lines.append("measurements when no comparable third-party provider was present in that run.")
    lines.append("")
    if large_rows:
        lines.extend(_render_representative_row_table(large_rows, base_dir=base_dir))
    else:
        lines.append("No large-kernel rows found yet.")
    lines.append("")
    lines.append("## Representative Memory-Bound Kernels")
    lines.append("")
    if memory_rows:
        lines.extend(_render_representative_row_table(memory_rows, base_dir=base_dir))
    else:
        lines.append("No memory-bound rows found yet.")
    lines.append("")
    lines.append("## Grouped Provider Winners")
    lines.append("")
    if comparisons:
        lines.extend(_render_grouped_winner_table(comparisons, base_dir=base_dir))
    else:
        lines.append("No grouped provider comparisons found yet.")
    lines.append("")
    lines.append("## Inference Engine Smoke")
    lines.append("")
    lines.append("Single-prompt decode smoke runs are integration checks, not apples-to-apples")
    lines.append("engine throughput benchmarks. They verify tokenizer/output parity for the")
    lines.append("composed engine paths.")
    lines.append("Rows with kernel coverage are integration rows, not serving-system benchmarks:")
    lines.append("`kernel_set_best_practice` keeps dense linears on torch/cuBLAS and uses")
    lines.append("shape-aware provider selection for the measured Qwen3 shapes;")
    lines.append("`kernel_set_full_kernels` is the slower all-kernel coverage smoke that")
    lines.append("also routes linears through kernel-set's auditable reference GEMM path.")
    lines.append("")
    lines.extend(_render_inference_table(inference_runs or [], base_dir=base_dir))
    quant_table = _render_quantized_engine_table(inference_runs or [], base_dir=base_dir)
    if quant_table:
        lines.append("")
        lines.extend(quant_table)
    lines.append("")
    lines.append("## Regenerate")
    lines.append("")
    lines.append("```bash")
    lines.append("python benchmarks/render_results_readme.py --root-readme README.md")
    lines.append("python benchmarks/persist.py validate benchmarks/results/runs")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def render_root_summary(
    runs: List[Dict[str, Any]],
    readme_path: str = "benchmarks/results/README.md",
    inference_runs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    comparisons = _best_comparisons(runs)
    large_rows = _representative_rows(runs, _is_large_kernel, limit=8, unique_by="op")
    memory_comparisons = [r for r in comparisons if _is_memory_kernel(r.get("op"))]
    gpus = sorted({f"{r.get('gpu_name')} sm{r.get('gpu_sm')}" for r in runs})
    ok = sum(_status_counts(r.get("rows") or []).get("ok", 0) for r in runs)
    total = sum(len(r.get("rows") or []) for r in runs)
    lines: List[str] = []
    lines.append(START)
    lines.append("## Benchmarks")
    lines.append("")
    lines.append(
        f"Canonical benchmark data is checked in under "
        f"[`benchmarks/results/runs/`]({os.path.dirname(readme_path)}/runs/) and summarized in "
        f"[`benchmarks/results/README.md`]({readme_path}). Current coverage: "
        f"**{len(runs)} runs**, **{ok}/{total} ok rows**, GPUs: "
        f"{', '.join(gpus) if gpus else '-'}."
    )
    lines.append("")
    lines.append(
        "Rows are scoped by their suite: `sota` rows compare installed "
        "production providers; `kernel_set` rows are diagnostic "
        "kernel-set/reference runs and are not promoted to default routing by "
        "themselves."
    )
    if large_rows:
        lines.append("")
        lines.append("Representative large-kernel rows:")
        lines.append("")
        lines.append("| GPU | op | shape | measured impl | latency |")
        lines.append("|---|---|---|---|---:|")
        for row in large_rows:
            lines.append(
                f"| {row.get('_gpu')} sm{row.get('_sm')} | `{row.get('op')}` | "
                f"`{row.get('shape')}` | `{row.get('impl')}` | "
                f"{_fmt_latency(row.get('latency_us'))} |"
            )
    if memory_comparisons:
        lines.append("")
        lines.append("Memory-bound provider highlights:")
        lines.append("")
        lines.append("| GPU | op | fastest measured impl | runner-up | ratio |")
        lines.append("|---|---|---|---|---:|")
        for row in _render_root_top(memory_comparisons):
            lines.append(
                f"| {row.get('gpu')} sm{row.get('sm')} | `{row.get('op')}` | "
                f"`{row.get('winner')}` {_fmt_latency(row.get('winner_latency_us'))} | "
                f"`{row.get('runner_up')}` {_fmt_latency(row.get('runner_up_latency_us'))} | "
                f"{float(row.get('winner_vs_next') or 0.0):.2f}x |"
            )
    if inference_runs:
        lines.append("")
        lines.append("Engine smoke:")
        lines.append("")
        lines.append(
            "Kernel-coverage rows prove call-path coverage and token parity; "
            "checked-in kernel benchmark tables provide provider-selection evidence."
        )
        lines.append("")
        lines.extend(_render_inference_table(inference_runs))
        quant_table = _render_quantized_engine_table(inference_runs)
        if quant_table:
            lines.append("")
            lines.extend(quant_table)
    lines.append("")
    lines.append(END)
    return "\n".join(lines)


def _render_root_top(comparisons: List[Dict[str, Any]], limit: int = 6) -> List[Dict[str, Any]]:
    priority = {
        "fused_add_rmsnorm": 0,
        "swiglu": 1,
        "rmsnorm": 2,
        "rope": 3,
        "attn_decode": 4,
        "attention_decode": 4,
        "mla_decode": 5,
        "fp8_gemm": 6,
        "gemm": 7,
    }
    rows = sorted(
        comparisons,
        key=lambda r: (
            priority.get(str(r.get("op")), 100),
            0 if r.get("suite") == "sota" else 1,
            str(r.get("gpu")),
            -float(r.get("winner_vs_next") or 0.0),
        ),
    )
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = (row.get("gpu"), row.get("op"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def update_root_readme(path: str, block: str) -> str:
    text = _read(path)
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        return before.rstrip() + "\n\n" + block + "\n\n" + after.lstrip()
    anchor = "\n## Try it on a real model\n"
    if anchor not in text:
        return text.rstrip() + "\n\n" + block + "\n"
    before, after = text.split(anchor, 1)
    return before.rstrip() + "\n\n" + block + "\n" + anchor + after


def cmd_render(args: argparse.Namespace) -> int:
    runs = _load_runs(args.runs)
    if not runs:
        raise ValueError("no canonical benchmark runs found")
    inference_runs = _load_inference(args.inference)
    index = build_index(runs)
    index_text = json.dumps(index, indent=2, sort_keys=False) + "\n"
    results_text = render_results_readme(
        runs, args.index, args.output, inference_runs=inference_runs)
    root_text: Optional[str] = None
    if args.root_readme:
        root_text = update_root_readme(
            args.root_readme,
            render_root_summary(runs, inference_runs=inference_runs),
        )

    checks: List[Tuple[str, str]] = [
        (args.index, index_text),
        (args.output, results_text),
    ]
    if args.root_readme and root_text is not None:
        checks.append((args.root_readme, root_text))

    if args.check:
        ok = True
        for path, expected in checks:
            if not os.path.exists(path) or _read(path) != expected:
                print(f"{path} is out of date", file=sys.stderr)
                ok = False
        return 0 if ok else 1

    _write(args.index, index_text)
    _write(args.output, results_text)
    if args.root_readme and root_text is not None:
        _write(args.root_readme, root_text)
    print(f"wrote {args.index}")
    print(f"wrote {args.output}")
    if args.root_readme:
        print(f"updated {args.root_readme}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=["benchmarks/results/runs"],
                   help="canonical run JSON file, glob, or directory")
    p.add_argument("--index", default="benchmarks/results/index.json")
    p.add_argument("--output", default="benchmarks/results/README.md")
    p.add_argument("--root-readme", default=None,
                   help="optional root README to update between benchmark markers")
    p.add_argument("--inference", nargs="+", default=[DEFAULT_INFERENCE],
                   help="inference engine JSON file, glob, or directory")
    p.add_argument("--check", action="store_true",
                   help="exit nonzero if generated files are out of date")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_render(args)


if __name__ == "__main__":
    raise SystemExit(main())
