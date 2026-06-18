#!/usr/bin/env python3
"""Persist kernel-set benchmark reports as canonical comparison rows.

The benchmark harnesses keep their human-readable Markdown output, but durable
data should live in one stable JSON shape:

    benchmark raw JSON -> canonical run JSON -> README/index generation

`from-legacy-md` exists only to bootstrap the already checked-in Markdown reports.
New benchmark runs should use `bench.py --json-output` or
`bench_sota.py --json-output`, then `persist.py from-report`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence


SCHEMA_VERSION = 1
DEFAULT_CONTEXT = {
    "model_id": "synthetic",
    "layer_idx": None,
    "position_kind": None,
    "position": None,
}
ROW_METADATA_KEYS = ("model_part", "position_kind_row")

_MODEL_PART_BY_OP = {
    "adamw": "optimizer",
    "argmax": "sampling",
    "attention_decode": "attention",
    "attention_prefill": "attention",
    "attn_decode": "attention",
    "attn_prefill": "attention",
    "cross_entropy": "loss",
    "dequantize_fp8": "quant",
    "dequantize_int4": "quant",
    "dequantize_int8": "quant",
    "embedding": "embedding",
    "embedding_bwd": "embedding",
    "flash_attn_bwd": "attention",
    "fp8_gemm": "linear",
    "fp8_gemm_blockwise": "linear",
    "fused_add_rmsnorm": "norm",
    "fused_linear_ce": "loss",
    "fused_moe": "moe",
    "geglu": "mlp",
    "gemm": "linear",
    "gemm_bf16": "linear",
    "gemm_fp16": "linear",
    "global_grad_norm": "optimizer",
    "layernorm": "norm",
    "layernorm_bwd": "norm",
    "log_softmax": "sampling",
    "mla_decode": "attention",
    "moe_gate": "moe",
    "moe_grouped_gemm": "moe",
    "moe_permute": "moe",
    "moe_unpermute": "moe",
    "quantize_fp8": "quant",
    "quantize_fp8_group": "quant",
    "quantize_int8": "quant",
    "reshape_and_cache": "attention",
    "rmsnorm": "norm",
    "rmsnorm_bwd": "norm",
    "rope": "position_encoding",
    "rope_bwd": "position_encoding",
    "sampling": "sampling",
    "sgd_momentum": "optimizer",
    "ssm": "ssm",
    "ssm_causal_conv1d": "ssm",
    "ssm_selective_scan": "ssm",
    "swiglu": "mlp",
    "swiglu_bwd": "mlp",
    "w4a16": "linear",
    "w4a8": "linear",
    "w8a8": "linear",
}

_TRAINING_POSITION_OPS = {
    "adamw",
    "cross_entropy",
    "embedding_bwd",
    "flash_attn_bwd",
    "fused_linear_ce",
    "global_grad_norm",
    "layernorm_bwd",
    "rmsnorm_bwd",
    "rope_bwd",
    "sgd_momentum",
    "swiglu_bwd",
}

_DECODE_POSITION_OPS = {"argmax", "log_softmax", "sampling"}
_WEIGHT_POSITION_OPS = {"dequantize_int4"}


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")
    return s or "unknown"


def _gpu_sm(gpu: Dict[str, Any]) -> int:
    if "sm_arch" in gpu:
        return int(gpu.get("sm_arch") or 0)
    return int(gpu.get("compute_major") or 0) * 10 + int(gpu.get("compute_minor") or 0)


def _gpu_name(gpu: Dict[str, Any]) -> str:
    return str(gpu.get("name") or "unknown")


def _context(config: Dict[str, Any]) -> Dict[str, Any]:
    ctx = dict(DEFAULT_CONTEXT)
    raw = config.get("context")
    if isinstance(raw, dict):
        ctx.update({k: raw.get(k) for k in ctx})
    return ctx


def _timing_profile(config: Dict[str, Any]) -> str:
    if config.get("cudagraph"):
        return "cudagraph-warm-l2"
    if config.get("l2_flush"):
        return "events-l2-flush"
    return "events-warm-l2"


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        return None if math.isnan(x) else x
    text = str(value).strip()
    if not text or text in {"-", "nan", "NaN"}:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    return float(m.group(0)) if m else None


def _shape_int(value: str) -> Optional[int]:
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\*\d+)+", text):
        out = 1
        for part in text.split("*"):
            out *= int(part)
        return out
    if re.fullmatch(r"\d+", text):
        return int(text)
    return None


def _shape_dims(shape: Any) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^,]+)", str(shape or "")):
        parsed = _shape_int(value)
        if parsed is not None:
            dims[key.lower()] = parsed
    return dims


def _derive_model_part(op: Any) -> str:
    op_name = str(op or "").strip().lower()
    if op_name in _MODEL_PART_BY_OP:
        return _MODEL_PART_BY_OP[op_name]
    if op_name.startswith("ew_"):
        return "elementwise"
    if "moe" in op_name:
        return "moe"
    if "attn" in op_name or "attention" in op_name or op_name.startswith("mla"):
        return "attention"
    if "norm" in op_name:
        return "norm"
    if "gemm" in op_name or op_name in {"w4a16", "w4a8", "w8a8"}:
        return "linear"
    if "rope" in op_name:
        return "position_encoding"
    if "quant" in op_name:
        return "quant"
    return op_name or "unknown"


def _derive_position_kind(op: Any, shape: Any, status: Any) -> str:
    del status  # Status stays available for future compatibility.
    op_name = str(op or "").strip().lower()
    dims = _shape_dims(shape)

    if op_name in _TRAINING_POSITION_OPS or op_name.endswith("_bwd"):
        return "training"
    if "decode" in op_name or op_name in _DECODE_POSITION_OPS:
        return "decode"
    if "prefill" in op_name:
        return "prefill"
    if op_name in _WEIGHT_POSITION_OPS:
        return "weight"

    for key in ("tokens", "rows", "m"):
        value = dims.get(key)
        if value == 1:
            return "decode"
        if value is not None and value > 1:
            return "prefill"
    if dims.get("seqs") is not None and (
        dims.get("ctx") is not None or dims.get("vocab") is not None
    ):
        return "decode"
    for key in ("seq", "l"):
        value = dims.get(key)
        if value == 1:
            return "decode"
        if value is not None and value > 1:
            return "prefill"

    n_value = dims.get("n")
    if n_value is not None:
        return "decode" if n_value <= 8192 else "bulk"
    if dims:
        return "batch"
    return "unknown"


def derive_row_metadata(row: Dict[str, Any]) -> Dict[str, str]:
    """Return row-level grouping metadata derived from op/shape/status."""
    return {
        "model_part": _derive_model_part(row.get("op")),
        "position_kind_row": _derive_position_kind(
            row.get("op"), row.get("shape"), row.get("status")
        ),
    }


def _latency_us(value: Any) -> Optional[float]:
    return _num(value)


def _util_pct(value: Any) -> Optional[float]:
    if value is None:
        return None
    m = re.search(r"\(([-+]?\d+(?:\.\d+)?)%\)", str(value))
    return float(m.group(1)) if m else None


def _comparison_key(run: Dict[str, Any], row: Dict[str, Any]) -> str:
    ctx = run["context"]
    pieces = [
        f"sm{run['gpu_sm']}",
        str(row.get("dtype") or run["dtype"]),
        run["timing_profile"],
        str(ctx.get("model_id")),
        str(ctx.get("layer_idx")),
        str(ctx.get("position_kind")),
        str(ctx.get("position")),
        str(row.get("op")),
        str(row.get("shape")),
    ]
    return "|".join(pieces)


def _base_run(
    suite: str,
    gpu: Dict[str, Any],
    config: Dict[str, Any],
    source_report: Optional[str],
    imported_from: str,
) -> Dict[str, Any]:
    timestamp = str(config.get("timestamp") or _utc_now())
    dtype = str(config.get("dtype") or "unknown")
    gpu_name = _gpu_name(gpu)
    run_id = f"{_slug(timestamp)}-{_slug(gpu_name)}-{dtype}-{suite}"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "suite": suite,
        "source_report": source_report,
        "imported_from": imported_from,
        "generated_at": _utc_now(),
        "timestamp": timestamp,
        "gpu": gpu,
        "gpu_name": gpu_name,
        "gpu_sm": _gpu_sm(gpu),
        "dtype": dtype,
        "timing_profile": _timing_profile(config),
        "config": config,
        "context": _context(config),
        "summary": {},
        "rows": [],
    }


def _canonical_row(run: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    row.update(derive_row_metadata(row))
    row["comparison_key"] = _comparison_key(run, row)
    return row


def normalize_report(
    report: Dict[str, Any],
    source_report: Optional[str] = None,
    suite: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize a raw `bench.py` or `bench_sota.py` JSON report."""
    if suite is None:
        if "results" in report:
            suite = "kernel_set"
        elif "rows" in report:
            suite = "sota"
        else:
            raise ValueError("cannot infer report suite; expected results or rows")

    gpu = dict(report.get("gpu") or {})
    config = dict(report.get("config") or {})
    run = _base_run(suite, gpu, config, source_report, "raw_json")
    if suite == "kernel_set":
        run["summary"] = dict(report.get("aggregate") or {})
        for item in report.get("results") or []:
            status = str(item.get("status") or "ok")
            base = {
                "suite": suite,
                "source_role": "candidate",
                "op": item.get("op"),
                "shape": item.get("shape"),
                "dtype": item.get("dtype") or run["dtype"],
                "impl": "kernel-set",
                "status": status.lower(),
                "latency_us": _latency_us(item.get("ks_us")),
                "min_latency_us": _latency_us(item.get("ks_min_us")),
                "p20_latency_us": _latency_us(item.get("ks_p20_us")),
                "p80_latency_us": _latency_us(item.get("ks_p80_us")),
                "gbps": _num(item.get("gbps")),
                "tflops": _num(item.get("tflops")),
                "util_pct": _num(item.get("bw_util")) or _num(item.get("compute_util")),
                "metric": "GB/s" if _num(item.get("gbps")) is not None else (
                    "TFLOP/s" if _num(item.get("tflops")) is not None else None),
                "rel_err": _num(item.get("rel_err")),
                "tol": _num(item.get("tol")),
                "is_correct": item.get("is_correct"),
                "speedup": _num(item.get("speedup")),
                "baseline": item.get("baseline") or None,
                "iters": item.get("n_iters"),
                "method": item.get("method") or None,
                "note": item.get("note") or "",
            }
            run["rows"].append(_canonical_row(run, base))
            if _latency_us(item.get("ref_us")) is not None and item.get("baseline"):
                ref = dict(base)
                ref.update({
                    "source_role": "baseline",
                    "impl": str(item.get("baseline")),
                    "status": "ok" if status == "ok" else status.lower(),
                    "latency_us": _latency_us(item.get("ref_us")),
                    "min_latency_us": _latency_us(item.get("ref_min_us")),
                    "p20_latency_us": None,
                    "p80_latency_us": None,
                    "gbps": None,
                    "tflops": None,
                    "util_pct": None,
                    "metric": None,
                    "speedup": None,
                    "baseline": None,
                    "note": "bench.py baseline",
                })
                run["rows"].append(_canonical_row(run, ref))
    elif suite == "sota":
        run["summary"] = dict(report.get("summary") or {})
        for item in report.get("rows") or []:
            row = {
                "suite": suite,
                "source_role": "provider",
                "op": item.get("op"),
                "shape": item.get("shape"),
                "dtype": item.get("dtype") or run["dtype"],
                "impl": item.get("impl"),
                "status": str(item.get("status") or "ok").lower(),
                "latency_us": _latency_us(item.get("lat_us")),
                "min_latency_us": _latency_us(item.get("min_us")),
                "p20_latency_us": None,
                "p80_latency_us": None,
                "gbps": _num(item.get("gbps")),
                "tflops": _num(item.get("tflops")),
                "util_pct": _num(item.get("util")),
                "metric": item.get("metric") or None,
                "rel_err": _num(item.get("rel_err")),
                "tol": _num(item.get("tol")),
                "is_correct": None if str(item.get("status") or "") != "ok" else True,
                "speedup": None,
                "baseline": None,
                "iters": item.get("iters"),
                "method": item.get("method") or None,
                "note": item.get("note") or "",
            }
            run["rows"].append(_canonical_row(run, row))
    else:
        raise ValueError(f"unknown suite: {suite}")
    return run


def _parse_gpu_line(text: str) -> Dict[str, Any]:
    gpu: Dict[str, Any] = {"name": "unknown", "compute_major": 0, "compute_minor": 0}
    m = re.search(r"\*\*GPU\*\*:\s*(.*?)\s*\(sm_(\d+),\s*CC\s*(\d+)\.(\d+)", text)
    if m:
        gpu["name"] = m.group(1).strip()
        sm = int(m.group(2))
        gpu["compute_major"] = int(m.group(3))
        gpu["compute_minor"] = int(m.group(4))
        gpu["sm_arch"] = sm
    mem = re.search(r",\s*([0-9.]+)\s*GB\)", text)
    if mem:
        gpu["total_mem_gb"] = float(mem.group(1))
    sms = re.search(r",\s*(\d+)\s*SMs,", text)
    if sms:
        gpu["sm_count"] = int(sms.group(1))
    return gpu


def _split_md_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _status_from_cell(cell: str) -> str:
    cell = (cell or "").strip()
    if not cell:
        return "ok"
    if cell.startswith("skip"):
        return "skip"
    if cell.startswith("import-fail"):
        return "import-fail"
    if cell.startswith("error"):
        return "error"
    if cell.startswith("incorrect"):
        return "incorrect"
    return cell.lower()


def normalize_legacy_markdown(path: str) -> Dict[str, Any]:
    """Best-effort importer for checked-in legacy Markdown reports."""
    lines = open(path, encoding="utf-8").read().splitlines()
    title = next((l for l in lines if l.startswith("# ")), "")
    if title.startswith("# kernel-set vs SOTA"):
        suite = "sota"
    elif title.startswith("# kernel-set benchmark"):
        suite = "kernel_set"
    else:
        raise ValueError(f"{path}: not a supported benchmark markdown report")

    gpu: Dict[str, Any] = {"name": "unknown", "compute_major": 0, "compute_minor": 0}
    config: Dict[str, Any] = {
        "dtype": "unknown",
        "timestamp": None,
        "l2_flush": None,
        "cudagraph": False,
        "context": dict(DEFAULT_CONTEXT),
    }
    summary: Dict[str, Any] = {}
    for line in lines:
        if line.startswith("- **GPU**:"):
            gpu = _parse_gpu_line(line)
        elif line.startswith("- **dtype**:"):
            m = re.search(r"\*\*dtype\*\*:\s*([^|]+)", line)
            if m:
                config["dtype"] = m.group(1).strip()
        elif line.startswith("- **timing**:"):
            config["l2_flush"] = "L2-flush=on" in line
            config["cudagraph"] = "cudagraph" in line
            m = re.search(r"target-ms=([0-9.]+)", line)
            if m:
                config["target_ms"] = float(m.group(1))
        elif line.startswith("- **timestamp**:"):
            config["timestamp"] = line.split(":", 1)[1].strip()
        elif line.startswith("- **harness commit**:"):
            config["git_commit"] = line.split(":", 1)[1].strip()
        elif line.startswith("**fast_1"):
            summary["text"] = line
        elif line.startswith("**Providers**:"):
            summary["text"] = line

    run = _base_run(suite, gpu, config, path, "legacy_markdown")
    run["summary"] = summary

    for line in lines:
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        cells = _split_md_row(line)
        if not cells or cells[0] == "op":
            continue
        if suite == "sota" and len(cells) >= 8:
            op, shape, impl, dtype, lat, perf, rel, status = cells[:8]
            metric = None
            gbps = None
            tflops = None
            if "GB/s" in perf:
                metric = "GB/s"
                gbps = _num(perf)
            elif "TFLOP/s" in perf:
                metric = "TFLOP/s"
                tflops = _num(perf)
            row = {
                "suite": suite,
                "source_role": "provider",
                "op": op,
                "shape": shape,
                "dtype": dtype,
                "impl": impl,
                "status": _status_from_cell(status),
                "latency_us": _latency_us(lat),
                "min_latency_us": None,
                "p20_latency_us": None,
                "p80_latency_us": None,
                "gbps": gbps,
                "tflops": tflops,
                "util_pct": _util_pct(perf),
                "metric": metric,
                "rel_err": _num(rel),
                "tol": None,
                "is_correct": True if _status_from_cell(status) == "ok" else None,
                "speedup": None,
                "baseline": None,
                "iters": None,
                "method": None,
                "note": status if _status_from_cell(status) != "ok" else "",
            }
            run["rows"].append(_canonical_row(run, row))
        elif suite == "kernel_set" and len(cells) >= 13:
            (op, shape, dtype, ks_lat, ref_lat, gb, tf, rel, spd, base,
             iters, method, notes) = cells[:13]
            status = "ok"
            if ks_lat == "skip":
                status = "skip"
            elif ks_lat == "err":
                status = "error"
            elif ks_lat == "INCORRECT":
                status = "incorrect"
            metric = None
            gbps = _num(gb)
            tflops = _num(tf)
            if gbps is not None:
                metric = "GB/s"
            elif tflops is not None:
                metric = "TFLOP/s"
            row = {
                "suite": suite,
                "source_role": "candidate",
                "op": op,
                "shape": shape,
                "dtype": dtype,
                "impl": "kernel-set",
                "status": status,
                "latency_us": _latency_us(ks_lat),
                "min_latency_us": None,
                "p20_latency_us": None,
                "p80_latency_us": None,
                "gbps": gbps,
                "tflops": tflops,
                "util_pct": _util_pct(gb) or _util_pct(tf),
                "metric": metric,
                "rel_err": _num(rel),
                "tol": None,
                "is_correct": True if status == "ok" and rel != "-" else None,
                "speedup": _num(spd),
                "baseline": None if base == "-" else base,
                "iters": int(_num(iters) or 0) or None,
                "method": method if method != "-" else None,
                "note": notes,
            }
            run["rows"].append(_canonical_row(run, row))
            if _latency_us(ref_lat) is not None and base != "-":
                ref = dict(row)
                ref.update({
                    "source_role": "baseline",
                    "impl": base,
                    "latency_us": _latency_us(ref_lat),
                    "gbps": None,
                    "tflops": None,
                    "util_pct": None,
                    "metric": None,
                    "speedup": None,
                    "baseline": None,
                    "note": "legacy bench.py baseline",
                })
                run["rows"].append(_canonical_row(run, ref))

    if not run["rows"]:
        raise ValueError(f"{path}: no benchmark rows found")
    return run


def validate_run(run: Dict[str, Any], path: str = "<memory>") -> List[str]:
    errors: List[str] = []
    if run.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{path}: schema_version must be {SCHEMA_VERSION}")
    for key in ("run_id", "suite", "gpu_sm", "dtype", "timing_profile", "rows"):
        if key not in run:
            errors.append(f"{path}: missing {key}")
    rows = run.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append(f"{path}: rows must be a non-empty list")
        return errors
    for i, row in enumerate(rows):
        for key in ("op", "shape", "dtype", "impl", "status", "comparison_key"):
            if row.get(key) in (None, ""):
                errors.append(f"{path}: row {i} missing {key}")
        for key in ROW_METADATA_KEYS:
            if key in row and row.get(key) in (None, ""):
                errors.append(f"{path}: row {i} has empty {key}")
        lat = row.get("latency_us")
        if row.get("status") == "ok" and lat is not None and float(lat) < 0:
            errors.append(f"{path}: row {i} has negative latency")
    return errors


def _write_json(doc: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")


def _default_output(run: Dict[str, Any], output_dir: str) -> str:
    return os.path.join(output_dir, f"{run['run_id']}.json")


def _iter_json_files(paths: Sequence[str]) -> Iterable[str]:
    for path in paths:
        if os.path.isdir(path):
            yield from sorted(glob.glob(os.path.join(path, "*.json")))
        else:
            yield path


def cmd_from_report(args: argparse.Namespace) -> int:
    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)
    run = normalize_report(report, source_report=args.source_report or args.report,
                           suite=args.suite)
    errors = validate_run(run, args.report)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    out = args.output or _default_output(run, args.output_dir)
    _write_json(run, out)
    print(out)
    return 0


def cmd_from_legacy_md(args: argparse.Namespace) -> int:
    run = normalize_legacy_markdown(args.report)
    errors = validate_run(run, args.report)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    out = args.output or _default_output(run, args.output_dir)
    _write_json(run, out)
    print(out)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    ok = True
    files = list(_iter_json_files(args.paths))
    if not files:
        print("no JSON files to validate", file=sys.stderr)
        return 1
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                run = json.load(f)
            errors = validate_run(run, path)
        except Exception as exc:
            errors = [f"{path}: {type(exc).__name__}: {exc}"]
        if errors:
            ok = False
            print("\n".join(errors), file=sys.stderr)
        elif args.verbose:
            print(f"ok {path}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("from-report", help="normalize raw benchmark JSON")
    r.add_argument("report", help="raw JSON from bench.py or bench_sota.py")
    r.add_argument("--suite", choices=["kernel_set", "sota"], default=None)
    r.add_argument("--source-report", default=None,
                   help="human report path recorded in the canonical JSON")
    r.add_argument("--output", default=None)
    r.add_argument("--output-dir", default="benchmarks/results/runs")
    r.set_defaults(func=cmd_from_report)

    m = sub.add_parser("from-legacy-md",
                       help="bootstrap canonical JSON from checked-in Markdown")
    m.add_argument("report", help="legacy benchmark Markdown report")
    m.add_argument("--output", default=None)
    m.add_argument("--output-dir", default="benchmarks/results/runs")
    m.set_defaults(func=cmd_from_legacy_md)

    v = sub.add_parser("validate", help="validate canonical run JSON files")
    v.add_argument("paths", nargs="+", help="JSON file or directory")
    v.add_argument("--verbose", action="store_true")
    v.set_defaults(func=cmd_validate)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
