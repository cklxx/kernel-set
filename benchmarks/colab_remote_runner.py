#!/usr/bin/env python3
"""Remote entrypoint for Colab CLI benchmark runs.

Upload a source tarball to the runtime, then execute this small script with
``colab exec --file``. It unpacks the checkout, runs ``colab_matrix.py``, and
packages the benchmark results plus a log for download.

Optional config path: ``/content/kernel-set-colab-config.json``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
from typing import Any, Dict, List


DEFAULT_CONFIG: Dict[str, Any] = {
    "archive": "/content/kernel-set.tgz",
    "workdir": "/content/kernel-set",
    "log": "/content/kernel-set-matrix.log",
    "result_tgz": "/content/kernel-set-results.tgz",
    "suite": "kernel_set",
    "ops": "all",
    "dtype": "fp16",
    "target_ms": 20,
    "run_label": None,
    "model_id": "synthetic",
    "position_kind": "all",
    "shard_size": 5,
    "jobs": 4,
    "cuda_arch": None,
}


def _load_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    path = pathlib.Path("/content/kernel-set-colab-config.json")
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    if not cfg.get("run_label"):
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cfg["run_label"] = f"{stamp}-colab"
    return cfg


def _run(cmd: List[str], log, cwd: pathlib.Path | None = None) -> int:
    print("$ " + " ".join(cmd), file=log, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(f"RC={proc.returncode}", file=log, flush=True)
    return int(proc.returncode)


def _extract(archive: pathlib.Path, workdir: pathlib.Path) -> None:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(workdir)


def _matrix_cmd(cfg: Dict[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        "benchmarks/colab_matrix.py",
        "--suite", str(cfg["suite"]),
        "--ops", str(cfg["ops"]),
        "--dtype", str(cfg["dtype"]),
        "--target-ms", str(cfg["target_ms"]),
        "--run-label", str(cfg["run_label"]),
        "--model-id", str(cfg["model_id"]),
        "--position-kind", str(cfg["position_kind"]),
        "--shard-size", str(cfg["shard_size"]),
        "--jobs", str(cfg["jobs"]),
    ]
    if cfg.get("cuda_arch"):
        cmd += ["--cuda-arch", str(cfg["cuda_arch"])]
    return cmd


def main() -> int:
    cfg = _load_config()
    archive = pathlib.Path(str(cfg["archive"]))
    workdir = pathlib.Path(str(cfg["workdir"]))
    log_path = pathlib.Path(str(cfg["log"]))
    result_tgz = pathlib.Path(str(cfg["result_tgz"]))

    rc = 0
    with log_path.open("w", encoding="utf-8") as log:
        print("CONFIG " + json.dumps(cfg, sort_keys=True), file=log, flush=True)
        if not archive.exists():
            print(f"missing archive: {archive}", file=log, flush=True)
            rc = 2
        else:
            _extract(archive, workdir)
            commands = [
                ["nvidia-smi"],
                [sys.executable, "-m", "pip", "install", "-q", "cmake>=3.24"],
                _matrix_cmd(cfg),
            ]
            for cmd in commands:
                rc = _run(cmd, log, cwd=workdir)
                if rc != 0:
                    break

        if workdir.exists():
            _run(
                [
                    "tar", "-czf", str(result_tgz),
                    "benchmarks/results",
                    "-C", "/content",
                    log_path.name,
                ],
                log,
                cwd=workdir,
            )

    print("MATRIX_RC", rc)
    print("RESULT_TGZ", result_tgz, result_tgz.stat().st_size if result_tgz.exists() else "missing")
    print("LOG_TAIL_BEGIN")
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    print("\n".join(text.splitlines()[-120:]))
    print("LOG_TAIL_END")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
