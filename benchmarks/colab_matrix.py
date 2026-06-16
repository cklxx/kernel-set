#!/usr/bin/env python3
"""Colab benchmark matrix runner.

Run from an unpacked kernel-set checkout on a Colab GPU VM. The script builds
kernel-set once, then runs selected benchmark op shards through the existing
benchmark harnesses. Each shard writes:

  * a Markdown report,
  * the raw harness JSON,
  * canonical JSON via ``persist.py from-report``.

At the end it refreshes ``results/index.json`` and ``results/README.md`` with
``render_results_readme.py``.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


SUITE_TO_SCRIPT = {
    "kernel_set": "bench.py",
    "sota": "bench_sota.py",
}


@dataclass
class ShardResult:
    suite: str
    ops: List[str]
    markdown: Path
    raw_json: Path
    canonical_json: Path
    bench_rc: int
    persist_rc: Optional[int]

    @property
    def ok(self) -> bool:
        return self.bench_rc == 0 and self.persist_rc == 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")
    return out or "unknown"


def _run(cmd: Sequence[str], *, cwd: Path, env: Optional[Dict[str, str]] = None,
         check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(list(cmd), cwd=str(cwd), env=env, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def _capture(cmd: Sequence[str], *, cwd: Path,
             env: Optional[Dict[str, str]] = None) -> Optional[str]:
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _all_ops(script: Path) -> List[str]:
    """Read ALL_OPS from a benchmark harness without importing GPU deps."""
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "ALL_OPS"
                   for t in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(isinstance(x, str)
                                                  for x in value):
            raise ValueError(f"{script}: ALL_OPS must be a list[str]")
        return list(value)
    raise ValueError(f"{script}: could not find ALL_OPS")


def _selected_suites(name: str) -> List[str]:
    if name == "both":
        return ["kernel_set", "sota"]
    return [name]


def _selected_ops(requested: str, available: Sequence[str]) -> List[str]:
    if requested == "all":
        return list(available)
    wanted = [x.strip() for x in requested.split(",") if x.strip()]
    return [x for x in wanted if x in available]


def _chunks(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i:i + size])


def _detect_arch(args: argparse.Namespace, repo: Path) -> str:
    if args.cuda_arch:
        return str(args.cuda_arch)
    out = _capture([
        "nvidia-smi",
        "--query-gpu=compute_cap",
        "--format=csv,noheader",
    ], cwd=repo)
    if out:
        return out.splitlines()[0].strip().replace(".", "")
    print("WARNING: nvidia-smi unavailable; defaulting to sm_89. "
          "Override with --cuda-arch.", file=sys.stderr)
    return "89"


def _build_once(args: argparse.Namespace, repo: Path) -> Path:
    if shutil.which("cmake") is None:
        raise SystemExit("ERROR: cmake not found")

    arch = _detect_arch(args, repo)
    build_dir = (repo / args.build_dir).resolve()
    jobs = str(args.jobs or os.cpu_count() or 4)

    print(f"==> Configuring kernel-set for sm_{arch}", file=sys.stderr)
    _run([
        "cmake", "-S", str(repo), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
    ], cwd=repo)

    print(f"==> Building kernel-set once (jobs={jobs})", file=sys.stderr)
    _run(["cmake", "--build", str(build_dir), "-j", jobs], cwd=repo)

    candidates = [
        build_dir / "libkernel_set.so",
        build_dir / "lib" / "libkernel_set.so",
        build_dir / "libkernel_set.dylib",
        build_dir / "lib" / "libkernel_set.dylib",
    ]
    for path in candidates:
        if path.exists():
            return path
    found = sorted(build_dir.rglob("libkernel_set.so"))
    found += sorted(build_dir.rglob("libkernel_set.dylib"))
    if found:
        return found[0]
    raise SystemExit(f"ERROR: could not find libkernel_set under {build_dir}")


def _bench_env(repo: Path, lib: Path) -> Dict[str, str]:
    env = os.environ.copy()
    pybind = str(repo / "bindings" / "python")
    old_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = pybind if not old_path else pybind + os.pathsep + old_path
    env["KERNEL_SET_LIB"] = str(lib)
    env["KERNEL_SET_LIB_DIR"] = str(lib.parent)
    return env


def _gpu_slug(repo: Path, env: Dict[str, str]) -> str:
    out = _capture([sys.executable, "benchmarks/bench.py", "--gpu-only"],
                   cwd=repo, env=env)
    if out:
        try:
            gpu = json.loads(out)
            name = str(gpu.get("name") or "")
            if name:
                return _slug(name)
            sm = int(gpu.get("compute_major") or 0) * 10
            sm += int(gpu.get("compute_minor") or 0)
            if sm:
                return f"sm{sm}"
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    out = _capture([
        "nvidia-smi",
        "--query-gpu=name",
        "--format=csv,noheader",
    ], cwd=repo, env=env)
    return _slug(out.splitlines()[0]) if out else "gpu"


def _common_bench_args(args: argparse.Namespace) -> List[str]:
    out = [
        "--dtype", args.dtype,
        "--target-ms", str(args.target_ms),
        "--model-id", args.model_id,
    ]
    if args.iters is not None:
        out += ["--iters", str(args.iters)]
    if args.position_kind is not None:
        out += ["--position-kind", args.position_kind]
    return out


def _run_shard(
    args: argparse.Namespace,
    repo: Path,
    env: Dict[str, str],
    suite: str,
    ops: List[str],
    shard_index: int,
    total_shards: int,
    gpu: str,
) -> ShardResult:
    bench_dir = repo / "benchmarks"
    script = bench_dir / SUITE_TO_SCRIPT[suite]
    out_dir = (repo / args.output_dir).resolve()
    raw_dir = out_dir / "raw"
    runs_dir = out_dir / "runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    ops_arg = ",".join(ops)
    ops_slug = _slug(ops[0] if len(ops) == 1 else f"{ops[0]}-plus-{len(ops) - 1}")
    label = f"{args.run_label}-{suite}-{ops_slug}"
    stem = _slug(f"{label}-{gpu}-{args.dtype}")
    markdown = out_dir / f"{stem}.md"
    raw_json = raw_dir / f"{stem}.raw.json"
    canonical_json = runs_dir / f"{stem}.json"

    print(f"==> Shard {shard_index}/{total_shards}: "
          f"{suite} ops={ops_arg}", file=sys.stderr)
    cmd = [
        sys.executable, str(script),
        "--ops", ops_arg,
        "--timestamp", label,
        "--format", "md",
        "--output", str(markdown),
        "--json-output", str(raw_json),
    ] + _common_bench_args(args)
    bench_rc = _run(cmd, cwd=repo, env=env, check=False).returncode

    persist_rc: Optional[int] = None
    if raw_json.exists():
        try:
            source_report = str(markdown.relative_to(repo))
        except ValueError:
            source_report = str(markdown)
        persist_cmd = [
            sys.executable, str(bench_dir / "persist.py"), "from-report",
            str(raw_json),
            "--suite", suite,
            "--source-report", source_report,
            "--output", str(canonical_json),
        ]
        persist_rc = _run(persist_cmd, cwd=repo, env=env,
                          check=False).returncode
    else:
        print(f"WARNING: raw JSON missing for {suite}:{ops_arg}; "
              "skipping persist", file=sys.stderr)

    return ShardResult(
        suite=suite,
        ops=ops,
        markdown=markdown,
        raw_json=raw_json,
        canonical_json=canonical_json,
        bench_rc=bench_rc,
        persist_rc=persist_rc,
    )


def _render_index(args: argparse.Namespace, repo: Path,
                  env: Dict[str, str]) -> int:
    out_dir = (repo / args.output_dir).resolve()
    runs_dir = out_dir / "runs"
    if not any(runs_dir.glob("*.json")):
        print(f"WARNING: no canonical JSON found in {runs_dir}; "
              "skipping README/index refresh", file=sys.stderr)
        return 1
    return _run([
        sys.executable, str(repo / "benchmarks" / "render_results_readme.py"),
        "--runs", str(runs_dir),
        "--index", str(out_dir / "index.json"),
        "--output", str(out_dir / "README.md"),
    ], cwd=repo, env=env, check=False).returncode


def _default_label() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build once and run Colab benchmark shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--suite", choices=["kernel_set", "sota", "both"],
                   default="kernel_set",
                   help="benchmark suite to run")
    p.add_argument("--ops", default="all",
                   help="comma-separated op groups, or 'all' per selected suite")
    p.add_argument("--dtype", default="fp16",
                   help="dtype forwarded to bench.py / bench_sota.py")
    p.add_argument("--target-ms", type=float, default=200.0,
                   help="per-shard timing budget")
    p.add_argument("--iters", type=int, default=None,
                   help="fixed timed launches; overrides --target-ms in harness")
    p.add_argument("--run-label", default=_default_label(),
                   help="base label used in report timestamps and filenames")
    p.add_argument("--model-id", default="synthetic",
                   help="model/context label recorded in JSON rows")
    p.add_argument("--position-kind", default=None,
                   help="optional position label recorded in JSON rows")
    p.add_argument("--output-dir", default="benchmarks/results",
                   help="directory for Markdown, raw JSON, runs, README, index")
    p.add_argument("--shard-size", type=int, default=1,
                   help="number of op groups to run per subprocess")
    p.add_argument("--build-dir", default="build",
                   help="CMake build directory")
    p.add_argument("--cuda-arch", default=None,
                   help="force CMAKE_CUDA_ARCHITECTURES, e.g. 89")
    p.add_argument("--jobs", type=int, default=None,
                   help="parallel build jobs")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.shard_size < 1:
        print("ERROR: --shard-size must be >= 1", file=sys.stderr)
        return 2

    repo = _repo_root()
    bench_dir = repo / "benchmarks"
    ops_by_suite = {
        suite: _all_ops(bench_dir / script)
        for suite, script in SUITE_TO_SCRIPT.items()
    }

    shard_plan: List[tuple[str, List[str]]] = []
    for suite in _selected_suites(args.suite):
        selected = _selected_ops(args.ops, ops_by_suite[suite])
        unknown = []
        if args.ops != "all":
            requested = [x.strip() for x in args.ops.split(",") if x.strip()]
            unknown = [x for x in requested if x not in ops_by_suite[suite]]
        if unknown:
            print(f"WARNING: {suite} does not have op(s): "
                  f"{','.join(unknown)}", file=sys.stderr)
        if not selected:
            print(f"WARNING: no selected ops for suite {suite}; skipping",
                  file=sys.stderr)
            continue
        for chunk in _chunks(selected, args.shard_size):
            shard_plan.append((suite, chunk))

    if not shard_plan:
        print("ERROR: no benchmark shards selected", file=sys.stderr)
        return 2

    lib = _build_once(args, repo)
    env = _bench_env(repo, lib)
    gpu = _gpu_slug(repo, env)
    print(f"==> Running {len(shard_plan)} shard(s) on {gpu}", file=sys.stderr)

    results: List[ShardResult] = []
    for i, (suite, ops) in enumerate(shard_plan, start=1):
        results.append(_run_shard(args, repo, env, suite, ops, i,
                                  len(shard_plan), gpu))

    render_rc = _render_index(args, repo, env)

    print("\n==> Summary", file=sys.stderr)
    for r in results:
        status = "ok" if r.ok else "failed"
        print(f"{status:6} {r.suite}:{','.join(r.ops)} "
              f"bench_rc={r.bench_rc} persist_rc={r.persist_rc} "
              f"md={r.markdown}", file=sys.stderr)

    failed = [r for r in results if not r.ok]
    if render_rc != 0:
        print(f"failed render_results_readme.py rc={render_rc}",
              file=sys.stderr)
    return 1 if failed or render_rc != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
