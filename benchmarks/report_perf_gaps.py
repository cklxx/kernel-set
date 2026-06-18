#!/usr/bin/env python3
"""Report persisted benchmark rows where kernel-set trails another backend.

The input is the canonical JSON under benchmarks/results/runs/*.json.  For each
op+shape+device+dtype group, this reports the fastest observed kernel-set row
when it is slower than the fastest observed ok non-kernel-set backend.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_RUNS_GLOB = "benchmarks/results/runs/*.json"
RATIO_EXCLUDED_STATUSES = {"error", "skip", "import-fail"}


@dataclass(frozen=True)
class BenchRow:
    op: str
    shape: str
    device: str
    dtype: str
    impl: str
    status: str
    latency_us: float
    timing: str
    run_id: str
    source_role: str
    path: str


@dataclass(frozen=True)
class Gap:
    key: Tuple[str, str, str, str]
    kernel_set: BenchRow
    best_backend: BenchRow
    ok_impl_count: int
    ratio: float


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    text = str(value).strip()
    if not text or text in {"-", "nan", "NaN"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    out = float(match.group(0))
    return out if math.isfinite(out) else None


def _gpu_sm(run: Dict[str, Any]) -> Optional[int]:
    if run.get("gpu_sm") is not None:
        return int(run["gpu_sm"])
    gpu = run.get("gpu")
    if not isinstance(gpu, dict):
        return None
    if gpu.get("sm_arch") is not None:
        return int(gpu["sm_arch"])
    major = gpu.get("compute_major")
    minor = gpu.get("compute_minor")
    if major is not None and minor is not None:
        return int(major) * 10 + int(minor)
    return None


def _device(run: Dict[str, Any]) -> str:
    gpu = run.get("gpu") if isinstance(run.get("gpu"), dict) else {}
    name = str(run.get("gpu_name") or gpu.get("name") or "unknown")
    sm = _gpu_sm(run)
    return f"{name} (sm{sm})" if sm is not None else name


def _status(value: Any) -> str:
    return str(value or "unknown").strip().lower()


def _is_kernel_set_impl(impl: Any) -> bool:
    text = re.sub(r"[\s_]+", "-", str(impl or "").strip().lower())
    return text == "kernel-set" or text.startswith("kernel-set-")


def _fmt_us(value: float) -> str:
    if value >= 1000:
        return f"{value:,.1f} us"
    if value >= 100:
        return f"{value:.1f} us"
    if value >= 10:
        return f"{value:.2f} us"
    return f"{value:.3f} us"


def _md_code(value: Any) -> str:
    text = str(value)
    text = text.replace("`", "\\`")
    return f"`{text}`"


def _md_link(path: str) -> str:
    rel = os.path.relpath(path, os.getcwd()) if os.path.isabs(path) else path
    return f"[{os.path.basename(rel)}]({rel})"


def _expand_paths(patterns: Sequence[str]) -> List[str]:
    files: List[str] = []
    seen = set()
    for pattern in patterns:
        matches: List[str]
        if os.path.isdir(pattern):
            matches = sorted(glob.glob(os.path.join(pattern, "*.json")))
        else:
            matches = sorted(glob.glob(pattern))
            if not matches and os.path.isfile(pattern):
                matches = [pattern]
        for path in matches:
            real = os.path.abspath(path)
            if real not in seen:
                seen.add(real)
                files.append(path)
    return sorted(files)


def _iter_rows(paths: Sequence[str]) -> Tuple[List[BenchRow], Counter]:
    rows: List[BenchRow] = []
    counts: Counter = Counter()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        if not isinstance(run, dict):
            raise ValueError(f"{path}: expected a JSON object")

        device = _device(run)
        run_dtype = str(run.get("dtype") or "unknown")
        timing = str(run.get("timing_profile") or "unknown")
        run_id = str(run.get("run_id") or os.path.basename(path))
        for raw in run.get("rows") or []:
            if not isinstance(raw, dict):
                continue
            status = _status(raw.get("status"))
            counts[status] += 1
            latency = _num(raw.get("latency_us"))
            if status != "ok" or latency is None or latency <= 0:
                continue
            rows.append(
                BenchRow(
                    op=str(raw.get("op") or "unknown"),
                    shape=str(raw.get("shape") or "unknown"),
                    device=device,
                    dtype=str(raw.get("dtype") or run_dtype),
                    impl=str(raw.get("impl") or "unknown"),
                    status=status,
                    latency_us=latency,
                    timing=timing,
                    run_id=run_id,
                    source_role=str(raw.get("source_role") or "unknown"),
                    path=path,
                )
            )
    return rows, counts


def _group_key(row: BenchRow) -> Tuple[str, str, str, str]:
    return (row.op, row.shape, row.device, row.dtype)


def _best_by_impl(rows: Iterable[BenchRow]) -> Dict[str, BenchRow]:
    best: Dict[str, BenchRow] = {}
    for row in rows:
        impl_key = re.sub(r"[\s_]+", "-", row.impl.strip().lower())
        prev = best.get(impl_key)
        if prev is None or row.latency_us < prev.latency_us:
            best[impl_key] = row
    return best


def find_gaps(rows: Iterable[BenchRow], min_ratio: float) -> List[Gap]:
    grouped: DefaultDict[Tuple[str, str, str, str], List[BenchRow]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    gaps: List[Gap] = []
    for key, group_rows in grouped.items():
        best_by_impl = _best_by_impl(group_rows)
        kernel_rows = [row for row in best_by_impl.values() if _is_kernel_set_impl(row.impl)]
        backend_rows = [row for row in best_by_impl.values() if not _is_kernel_set_impl(row.impl)]
        if not kernel_rows or not backend_rows:
            continue
        kernel_set = min(kernel_rows, key=lambda row: row.latency_us)
        best_backend = min(backend_rows, key=lambda row: row.latency_us)
        ratio = kernel_set.latency_us / best_backend.latency_us
        if ratio > min_ratio:
            gaps.append(
                Gap(
                    key=key,
                    kernel_set=kernel_set,
                    best_backend=best_backend,
                    ok_impl_count=len(best_by_impl),
                    ratio=ratio,
                )
            )
    gaps.sort(key=lambda gap: (-gap.ratio, gap.kernel_set.op, gap.kernel_set.device, gap.kernel_set.shape))
    return gaps


def _render_summary(
    paths: Sequence[str],
    rows: Sequence[BenchRow],
    counts: Counter,
    gaps: Sequence[Gap],
    min_ratio: float,
) -> List[str]:
    excluded = {name: counts.get(name, 0) for name in sorted(RATIO_EXCLUDED_STATUSES)}
    other_non_ok = {
        status: count
        for status, count in sorted(counts.items())
        if status != "ok" and status not in RATIO_EXCLUDED_STATUSES
    }
    ok_groups = len({_group_key(row) for row in rows})
    lines = [
        "# kernel-set performance gaps",
        "",
        f"- Input files: {len(paths)}",
        "- Ratio rule: fastest ok `kernel-set` / fastest ok non-kernel-set backend "
        "for the same `op+shape+device+dtype`.",
        f"- Minimum ratio: `{min_ratio:.3g}x`",
        f"- Ok rows used for ratios: {len(rows)} across {ok_groups} groups.",
        f"- Excluded from ratios by status: "
        + ", ".join(f"{status}={count}" for status, count in excluded.items()),
    ]
    if other_non_ok:
        lines.append(
            "- Other non-ok rows not used: "
            + ", ".join(f"{status}={count}" for status, count in other_non_ok.items())
        )
    lines.append(f"- Gaps found: {len(gaps)}")
    return lines


def _render_top_gaps(gaps: Sequence[Gap], limit: int) -> List[str]:
    lines = [
        "",
        f"## Top {min(limit, len(gaps))} gaps",
        "",
    ]
    if not gaps:
        lines.append("No slower kernel-set rows found.")
        return lines

    lines.extend(
        [
            "| gap | op | shape | device | dtype | kernel-set | best backend | ok impls | timing | source |",
            "|---:|---|---|---|---|---:|---|---:|---|---|",
        ]
    )
    for gap in gaps[:limit]:
        ks = gap.kernel_set
        best = gap.best_backend
        timing = ks.timing if ks.timing == best.timing else f"{ks.timing} vs {best.timing}"
        source = f"{_md_link(ks.path)} / {_md_link(best.path)}"
        lines.append(
            f"| {gap.ratio:.2f}x | {_md_code(ks.op)} | {_md_code(ks.shape)} | "
            f"{ks.device} | {ks.dtype} | {_fmt_us(ks.latency_us)} | "
            f"{_md_code(best.impl)} {_fmt_us(best.latency_us)} | {gap.ok_impl_count} | "
            f"{timing} | {source} |"
        )
    return lines


def _render_op_groups(gaps: Sequence[Gap], limit: int) -> List[str]:
    lines = [
        "",
        "## By op",
        "",
    ]
    if not gaps:
        lines.append("No gap groups to summarize.")
        return lines

    grouped: DefaultDict[Tuple[str, str, str], List[Gap]] = defaultdict(list)
    for gap in gaps:
        op, _shape, device, dtype = gap.key
        grouped[(op, device, dtype)].append(gap)

    items = sorted(
        grouped.items(),
        key=lambda item: (-max(g.ratio for g in item[1]), item[0][0], item[0][1], item[0][2]),
    )
    lines.extend(
        [
            "| worst gap | gap rows | op | device | dtype | best competing backend | worst shape |",
            "|---:|---:|---|---|---|---|---|",
        ]
    )
    for (op, device, dtype), group in items[:limit]:
        worst = max(group, key=lambda gap: gap.ratio)
        backend = worst.best_backend.impl
        lines.append(
            f"| {worst.ratio:.2f}x | {len(group)} | {_md_code(op)} | {device} | {dtype} | "
            f"{_md_code(backend)} | {_md_code(worst.kernel_set.shape)} |"
        )
    return lines


def render_markdown(paths: Sequence[str], rows: Sequence[BenchRow], counts: Counter, gaps: Sequence[Gap], args: argparse.Namespace) -> str:
    lines: List[str] = []
    lines.extend(_render_summary(paths, rows, counts, gaps, args.min_ratio))
    lines.extend(_render_top_gaps(gaps, args.limit))
    lines.extend(_render_op_groups(gaps, args.group_limit))
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print markdown for persisted benchmark groups where kernel-set is slower than the best ok backend.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[DEFAULT_RUNS_GLOB],
        help=f"Run JSON files, directories, or globs. Default: {DEFAULT_RUNS_GLOB}",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows in the top-gaps table.")
    parser.add_argument("--group-limit", type=int, default=20, help="Maximum rows in the by-op table.")
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=1.0,
        help="Only report gaps strictly greater than this kernel-set/backend ratio.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = _expand_paths(args.paths)
    if not paths:
        print(f"error: no run JSON files matched: {', '.join(args.paths)}", file=sys.stderr)
        return 2
    rows, counts = _iter_rows(paths)
    gaps = find_gaps(rows, args.min_ratio)
    sys.stdout.write(render_markdown(paths, rows, counts, gaps, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
