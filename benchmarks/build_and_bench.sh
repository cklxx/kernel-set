#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build_and_bench.sh — build kernel-set for the detected GPU and benchmark it.
#
# Steps:
#   1. Detect the GPU (nvidia-smi compute capability) and map it to a CUDA arch
#      (L4 -> 89, A100 -> 80, otherwise the device's native sm_XY).
#   2. cmake configure + build libkernel_set.so for that single arch (fast).
#   3. Export KERNEL_SET_LIB so the python binding dlopens the fresh build.
#   4. Run bench.py and write the report to benchmarks/results/<gpu>.md.
#
# Usage:
#   benchmarks/build_and_bench.sh [extra bench.py args...]
#
# Environment overrides:
#   KS_ARCH      force a CUDA arch (e.g. 89, 80, 90) instead of auto-detect.
#   KS_DTYPE     dtype passed to bench.py (default: fp16).
#   KS_OPS       op selector passed to bench.py (default: all).
#   KS_ITERS     timed iterations (default: 50).
#   KS_BUILD_DIR build directory (default: <repo>/build).
#   KS_JOBS      parallel build jobs (default: nproc).
#   KS_RUN_LABEL run label recorded in reports (default: UTC timestamp).
# ---------------------------------------------------------------------------
set -euo pipefail

# --- locate the repo root (this script lives in <repo>/benchmarks) ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_DIR="${SCRIPT_DIR}"
RESULTS_DIR="${BENCH_DIR}/results"
RAW_RESULTS_DIR="${RESULTS_DIR}/raw"
RUNS_DIR="${RESULTS_DIR}/runs"
BUILD_DIR="${KS_BUILD_DIR:-${REPO_ROOT}/build}"
PYBIND_DIR="${REPO_ROOT}/bindings/python"

DTYPE="${KS_DTYPE:-fp16}"
OPS="${KS_OPS:-all}"
ITERS="${KS_ITERS:-50}"
JOBS="${KS_JOBS:-$( (nproc 2>/dev/null) || (sysctl -n hw.ncpu 2>/dev/null) || echo 4)}"
RUN_LABEL="${KS_RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "${RESULTS_DIR}" "${RAW_RESULTS_DIR}" "${RUNS_DIR}"

# --- pick a python --------------------------------------------------------
PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  PYTHON=python
fi

# --- detect GPU compute capability ----------------------------------------
echo "==> Detecting GPU"
CC=""
GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || true)"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
fi

# --- map GPU -> CUDA arch (sm_XY as the 2-digit number CMake wants) --------
ARCH=""
if [[ -n "${KS_ARCH:-}" ]]; then
  ARCH="${KS_ARCH}"
  echo "    using forced arch from KS_ARCH=${ARCH}"
elif [[ -n "${CC}" ]]; then
  # CC like "8.9" -> "89". This is the device's native arch; we then override
  # the well-known Colab GPUs explicitly so the intent is documented.
  ARCH="${CC//./}"
  lname="$(echo "${GPU_NAME}" | tr '[:upper:]' '[:lower:]')"
  case "${lname}" in
    *l4*)   ARCH=89 ;;   # Colab L4  -> sm_89
    *a100*) ARCH=80 ;;   # Colab A100 -> sm_80
    *h100*) ARCH=90 ;;
    *t4*)   ARCH=75 ;;
  esac
  echo "    GPU='${GPU_NAME}' compute_cap=${CC} -> CUDA arch ${ARCH}"
else
  ARCH=89
  echo "    nvidia-smi unavailable; defaulting CUDA arch to ${ARCH} (L4). "\
       "Override with KS_ARCH=<arch>."
fi

# --- configure + build -----------------------------------------------------
echo "==> Configuring (cmake) for sm_${ARCH}"
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"

echo "==> Building libkernel_set (jobs=${JOBS})"
cmake --build "${BUILD_DIR}" -j "${JOBS}"

# --- locate the freshly built shared library ------------------------------
LIB=""
for cand in \
  "${BUILD_DIR}/libkernel_set.so" \
  "${BUILD_DIR}/lib/libkernel_set.so" \
  "${BUILD_DIR}/libkernel_set.dylib"; do
  if [[ -f "${cand}" ]]; then LIB="${cand}"; break; fi
done
if [[ -z "${LIB}" ]]; then
  # last resort: find it anywhere under the build dir
  LIB="$(find "${BUILD_DIR}" -name 'libkernel_set.so' -o -name 'libkernel_set.dylib' 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${LIB}" ]]; then
  echo "ERROR: could not find libkernel_set in ${BUILD_DIR}" >&2
  exit 1
fi
export KERNEL_SET_LIB="${LIB}"
echo "==> KERNEL_SET_LIB=${KERNEL_SET_LIB}"

# --- make the binding importable ------------------------------------------
# Prefer an installed binding; otherwise add the source dir to PYTHONPATH.
if ! "${PYTHON}" -c "import kernel_set" >/dev/null 2>&1; then
  export PYTHONPATH="${PYBIND_DIR}:${PYTHONPATH:-}"
  echo "    kernel_set not installed; using PYTHONPATH=${PYBIND_DIR}"
fi

# --- name the results file after the detected GPU --------------------------
GPU_SLUG="$("${PYTHON}" "${BENCH_DIR}/bench.py" --gpu-only 2>/dev/null \
  | "${PYTHON}" -c 'import sys,json,re; d=json.load(sys.stdin); n=d["name"].lower().replace(" ","");
keys=["a100","h100","h200","l40s","l40","l4","a10g","a10","t4","v100","4090","3090","a6000"];
m=next((k for k in keys if k in n), None);
print(m or (re.sub(r"[^a-z0-9]+","-",n).strip("-") or ("sm"+str(d["compute_major"]*10+d["compute_minor"]))))' \
  2>/dev/null || echo "sm${ARCH}")"
[[ -z "${GPU_SLUG}" ]] && GPU_SLUG="sm${ARCH}"
OUT="${RESULTS_DIR}/${GPU_SLUG}.md"
RAW_JSON="${RAW_RESULTS_DIR}/${RUN_LABEL}-${GPU_SLUG}-${DTYPE}-bench.raw.json"
RUN_JSON="${RUNS_DIR}/${RUN_LABEL}-${GPU_SLUG}-${DTYPE}-bench.json"

# --- run the benchmark -----------------------------------------------------
echo "==> Running benchmarks (dtype=${DTYPE}, ops=${OPS}, iters=${ITERS})"
echo "    report -> ${OUT}"
"${PYTHON}" "${BENCH_DIR}/bench.py" \
  --ops "${OPS}" \
  --dtype "${DTYPE}" \
  --iters "${ITERS}" \
  --timestamp "${RUN_LABEL}" \
  --format md \
  --output "${OUT}" \
  --json-output "${RAW_JSON}" \
  "$@"

echo "==> Persisting canonical benchmark JSON"
"${PYTHON}" "${BENCH_DIR}/persist.py" from-report "${RAW_JSON}" \
  --source-report "${OUT}" \
  --output "${RUN_JSON}"

echo "==> Updating benchmark result index"
"${PYTHON}" "${BENCH_DIR}/render_results_readme.py" \
  --runs "${RUNS_DIR}" \
  --index "${RESULTS_DIR}/index.json" \
  --output "${RESULTS_DIR}/README.md"

echo "==> Done. Results written to ${OUT}"
echo "==> Raw JSON written to ${RAW_JSON}"
echo "==> Canonical run written to ${RUN_JSON}"
