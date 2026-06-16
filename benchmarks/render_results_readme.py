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


def render_results_readme(runs: List[Dict[str, Any]], index_path: str, output_path: str) -> str:
    comparisons = _best_comparisons(runs)
    base_dir = os.path.dirname(os.path.abspath(output_path))
    gpus = sorted({f"{r.get('gpu_name')} (sm{r.get('gpu_sm')})" for r in runs})
    suites = Counter(str(r.get("suite")) for r in runs)
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
    lines.append("## Grouped Provider Winners")
    lines.append("")
    if comparisons:
        lines.extend(_render_grouped_winner_table(comparisons, base_dir=base_dir))
    else:
        lines.append("No grouped provider comparisons found yet.")
    lines.append("")
    lines.append("## Regenerate")
    lines.append("")
    lines.append("```bash")
    lines.append("python benchmarks/render_results_readme.py --root-readme README.md")
    lines.append("python benchmarks/persist.py validate benchmarks/results/runs")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def render_root_summary(runs: List[Dict[str, Any]], readme_path: str = "benchmarks/results/README.md") -> str:
    comparisons = _best_comparisons(runs)
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
    if comparisons:
        lines.append("")
        lines.append("| GPU | op | fastest measured impl | runner-up | ratio |")
        lines.append("|---|---|---|---|---:|")
        for row in _render_root_top(comparisons):
            lines.append(
                f"| {row.get('gpu')} sm{row.get('sm')} | `{row.get('op')}` | "
                f"`{row.get('winner')}` {_fmt_latency(row.get('winner_latency_us'))} | "
                f"`{row.get('runner_up')}` {_fmt_latency(row.get('runner_up_latency_us'))} | "
                f"{float(row.get('winner_vs_next') or 0.0):.2f}x |"
            )
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
    index = build_index(runs)
    index_text = json.dumps(index, indent=2, sort_keys=False) + "\n"
    results_text = render_results_readme(runs, args.index, args.output)
    root_text: Optional[str] = None
    if args.root_readme:
        root_text = update_root_readme(args.root_readme, render_root_summary(runs))

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
    p.add_argument("--check", action="store_true",
                   help="exit nonzero if generated files are out of date")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_render(args)


if __name__ == "__main__":
    raise SystemExit(main())
