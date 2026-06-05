#!/usr/bin/env bash
# install_baselines.sh — install the industry SOTA baseline kernels that
# bench_sota.py compares kernel-set against.
#
# DESIGN GOALS
#   * Fault-tolerant: `set -u` (catch unset vars) but NOT `set -e` — a failed
#     install of one library must NOT abort the rest. Every install is guarded.
#   * Prefer prebuilt wheels (source builds for flash-attn / mamba-ssm are slow).
#   * Arch-aware: detect the GPU SM via nvidia-smi and SKIP sm90+-only libs on
#     sm<90 (L4=sm89, A100=sm80, T4=sm75). DeepGEMM / Machete / NVFP4 are
#     sm90+/sm100 only and are explicitly skipped with a clear message.
#   * Idempotent + Colab-friendly: re-running is safe; uses `pip install -U` and
#     per-lib import checks; never assumes root.
#
# USAGE
#   bash benchmarks/install_baselines.sh            # auto-detect arch
#   FORCE_SM=90 bash benchmarks/install_baselines.sh  # override detected SM
#   SKIP_VLLM=1 bash benchmarks/install_baselines.sh  # skip the heavy vLLM install
#   INSTALL_FLASH_ATTN=0 bash benchmarks/install_baselines.sh  # skip flash-attn
#
# After it finishes it prints a summary of which libs import OK vs failed.

set -u

# ----------------------------------------------------------------------------
# Knobs (env-overridable)
# ----------------------------------------------------------------------------
PIP="${PIP:-pip}"
PYTHON="${PYTHON:-python3}"
SKIP_VLLM="${SKIP_VLLM:-0}"            # vLLM is heavy; set 1 to skip (no Marlin)
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
INSTALL_SAGE="${INSTALL_SAGE:-0}"      # sageattention optional (source build)
INSTALL_MAMBA="${INSTALL_MAMBA:-1}"    # mamba-ssm + causal-conv1d (may build)
INSTALL_SGL="${INSTALL_SGL:-1}"        # sgl-kernel (SGLang; needs torch + CUDA)

LOG_OK=()       # "name" succeeded import check
LOG_FAIL=()     # "name: reason"
LOG_SKIP=()     # "name: reason"

note()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ok]\033[0m   %s\n' "$*"; }
err()   { printf '\033[1;31m[fail]\033[0m %s\n' "$*"; }

# Run a command, logging but never aborting the script on failure.
run() {
  echo "    \$ $*"
  if "$@"; then
    return 0
  else
    warn "command failed (continuing): $*"
    return 1
  fi
}

# import_check NAME "python import expr"
# Records into LOG_OK / LOG_FAIL.
import_check() {
  local name="$1"; shift
  local expr="$1"; shift
  if "$PYTHON" -c "$expr" >/dev/null 2>&1; then
    ok "$name importable"
    LOG_OK+=("$name")
    return 0
  else
    local reason
    reason="$("$PYTHON" -c "$expr" 2>&1 | tail -1)"
    err "$name NOT importable: $reason"
    LOG_FAIL+=("$name: $reason")
    return 1
  fi
}

skip() {
  local name="$1"; shift
  local reason="$1"; shift
  warn "SKIP $name: $reason"
  LOG_SKIP+=("$name: $reason")
}

# ----------------------------------------------------------------------------
# GPU arch detection (nvidia-smi -> compute_cap like "8.9" -> SM int 89)
# ----------------------------------------------------------------------------
detect_sm() {
  if [ -n "${FORCE_SM:-}" ]; then
    echo "$FORCE_SM"; return 0
  fi
  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [ -z "$cap" ]; then
    echo "0"; return 0
  fi
  # "8.9" -> 89
  echo "$cap" | awk -F. '{ printf "%d", ($1*10)+$2 }'
}

SM="$(detect_sm)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
note "Detected GPU: ${GPU_NAME:-<none>} (sm${SM})"
if [ "$SM" = "0" ]; then
  warn "No NVIDIA GPU detected (nvidia-smi unavailable). Installing the CPU-/"
  warn "arch-agnostic wheels anyway; arch-gated libs will be skipped."
fi

# Helper: is the detected SM >= a required SM?
sm_ge() { [ "$SM" -ge "$1" ] 2>/dev/null; }

# ----------------------------------------------------------------------------
# 0) Core: ensure pip is fresh (best-effort)
# ----------------------------------------------------------------------------
note "Upgrading pip (best-effort)"
run "$PIP" install -U pip setuptools wheel || true

# ----------------------------------------------------------------------------
# 1) triton — Triton (used by liger, mamba-ssm triton path, torch.compile)
# ----------------------------------------------------------------------------
note "triton"
run "$PIP" install -U triton
import_check "triton" "import triton"

# ----------------------------------------------------------------------------
# 2) flashinfer — its own wheel index (decode/prefill/norm/rope/activation)
# ----------------------------------------------------------------------------
note "flashinfer-python (own wheel index)"
# flashinfer publishes per-CUDA wheels at its index; fall back to PyPI.
if ! run "$PIP" install -U flashinfer-python \
      -i https://flashinfer.ai/whl/cu124/torch2.4 ; then
  warn "flashinfer index install failed; trying plain PyPI"
  run "$PIP" install -U flashinfer-python || true
fi
import_check "flashinfer" "import flashinfer"

# ----------------------------------------------------------------------------
# 3) liger-kernel — Triton rmsnorm/rope/swiglu/cross_entropy (sm80+, but wheel
#    installs anywhere; kernels JIT at runtime)
# ----------------------------------------------------------------------------
note "liger-kernel"
run "$PIP" install -U liger-kernel
import_check "liger-kernel" "from liger_kernel.ops.rms_norm import LigerRMSNormFunction"

# ----------------------------------------------------------------------------
# 4) bitsandbytes — 8-bit optim / quant (availability probe)
# ----------------------------------------------------------------------------
note "bitsandbytes"
run "$PIP" install -U bitsandbytes
import_check "bitsandbytes" "import bitsandbytes"

# ----------------------------------------------------------------------------
# 5) flash-attn — prefill/decode. Prefer a prebuilt wheel; a source build needs
#    --no-build-isolation and is SLOW (many minutes). Guarded so failure is fine.
# ----------------------------------------------------------------------------
if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
  if sm_ge 80; then
    note "flash-attn (Ampere+; prefer prebuilt wheel, source build is slow)"
    # Try a plain wheel first (pip will pick a matching prebuilt wheel if the
    # torch/cuda/python combo matches an upstream release asset).
    if ! run "$PIP" install -U flash-attn ; then
      warn "flash-attn wheel install failed; attempting --no-build-isolation "
      warn "source build (this can take many minutes)…"
      run "$PIP" install flash-attn --no-build-isolation || true
    fi
    import_check "flash-attn" "from flash_attn import flash_attn_func"
  else
    skip "flash-attn" "needs sm80+ (Ampere); device is sm${SM}"
  fi
else
  skip "flash-attn" "INSTALL_FLASH_ATTN=0"
fi

# ----------------------------------------------------------------------------
# 6) mamba-ssm + causal-conv1d — SSM probes. May build from source (slow).
# ----------------------------------------------------------------------------
if [ "$INSTALL_MAMBA" = "1" ]; then
  note "causal-conv1d (may build from source)"
  if ! run "$PIP" install -U "causal-conv1d>=1.4.0" ; then
    run "$PIP" install "causal-conv1d>=1.4.0" --no-build-isolation || true
  fi
  import_check "causal-conv1d" "from causal_conv1d import causal_conv1d_fn"

  note "mamba-ssm (may build from source)"
  if ! run "$PIP" install -U mamba-ssm ; then
    run "$PIP" install mamba-ssm --no-build-isolation || true
  fi
  import_check "mamba-ssm" "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn"
else
  skip "mamba-ssm" "INSTALL_MAMBA=0"
  skip "causal-conv1d" "INSTALL_MAMBA=0"
fi

# ----------------------------------------------------------------------------
# 7) sageattention — optional quantized attention (source build, optional)
# ----------------------------------------------------------------------------
if [ "$INSTALL_SAGE" = "1" ]; then
  if sm_ge 80; then
    note "sageattention (optional; source build)"
    run "$PIP" install sageattention==2.2.0 --no-build-isolation || true
    import_check "sageattention" "from sageattention import sageattn"
  else
    skip "sageattention" "needs sm80+; device is sm${SM}"
  fi
else
  skip "sageattention" "INSTALL_SAGE=0 (optional)"
fi

# ----------------------------------------------------------------------------
# 8) vLLM (for Marlin W4A16 + cutlass_scaled_mm fp8 + fused_experts MoE).
#    Heavy install. If SKIP_VLLM=1, Marlin/fp8-cutlass/fused-moe are skipped.
# ----------------------------------------------------------------------------
if [ "$SKIP_VLLM" = "1" ]; then
  skip "vllm (Marlin/cutlass-fp8/fused-moe)" "SKIP_VLLM=1 (heavy install)"
else
  if sm_ge 80; then
    note "vllm (heavy: pulls CUDA kernels for Marlin W4A16, cutlass fp8, fused MoE)"
    run "$PIP" install -U vllm || warn "vllm install failed (heavy dep); Marlin/fp8-cutlass will be unavailable"
    import_check "vllm-ops" "from vllm import _custom_ops as ops; ops.gptq_marlin_gemm"
  else
    skip "vllm (Marlin)" "Marlin needs sm80+; device is sm${SM}"
  fi
fi

# ----------------------------------------------------------------------------
# 8b) sgl-kernel — SGLang's fused CUDA kernels (the hard-op alignment target):
#     MoE gate (topk_softmax / moe_fused_gate), grouped-MoE, rmsnorm /
#     fused_add_rmsnorm / gemma_rmsnorm, rotary_embedding, silu_and_mul,
#     fp8/int8 scaled-mm, FA3 attention + FlashMLA, sampling renorm. Needs torch
#     and a recent CUDA toolkit; prefer the prebuilt wheel. Guarded; continue on
#     failure. fp8/FlashMLA paths need sm90 but the wheel installs on sm80+.
# ----------------------------------------------------------------------------
if [ "$INSTALL_SGL" = "1" ]; then
  if sm_ge 80; then
    note "sgl-kernel (SGLang fused kernels; prefer prebuilt wheel; needs torch + recent CUDA)"
    if ! run "$PIP" install -U sgl-kernel ; then
      warn "sgl-kernel wheel install failed (needs matching torch/CUDA);"
      warn "see https://github.com/sgl-project/sglang for per-CUDA wheels. Continuing."
    fi
    import_check "sgl-kernel" "import sgl_kernel"
  else
    skip "sgl-kernel" "fp8/FlashMLA paths need sm90; device is sm${SM} (wheel may still install on sm80+)"
  fi
else
  skip "sgl-kernel" "INSTALL_SGL=0"
fi

# ----------------------------------------------------------------------------
# 9) cut-cross-entropy — fused-linear cross-entropy (git install)
# ----------------------------------------------------------------------------
note "cut-cross-entropy (fused-linear CE)"
run "$PIP" install -U "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git" \
  || warn "cut-cross-entropy install failed (git); skipping"
import_check "cut-cross-entropy" "from cut_cross_entropy import linear_cross_entropy"

# ----------------------------------------------------------------------------
# 10) Arch-gated sm90+/sm100 providers — explicitly skipped on sm<90.
#     These are source/JIT builds gated to Hopper/Blackwell.
# ----------------------------------------------------------------------------
if sm_ge 90; then
  note "DeepGEMM (sm90+) — JIT FP8/grouped GEMM"
  warn "DeepGEMM is a source/JIT install: "
  warn "  git clone --recursive https://github.com/deepseek-ai/DeepGEMM && cd DeepGEMM && ./install.sh"
  warn "Not auto-installed here (build env varies); run the above on your Hopper/Blackwell box."
  LOG_SKIP+=("DeepGEMM: manual source/JIT install (sm${SM} supports it)")

  note "FlashMLA (sm90+) — MLA decode"
  warn "FlashMLA is a source install:"
  warn "  git clone --recursive https://github.com/deepseek-ai/FlashMLA && cd FlashMLA && pip install -v ."
  warn "Not auto-installed here; run the above on your Hopper/Blackwell box."
  LOG_SKIP+=("FlashMLA: manual source install (sm${SM} supports it)")
else
  skip "DeepGEMM (fp8/grouped GEMM)" "needs sm90+ (Hopper/Blackwell); device is sm${SM}"
  skip "FlashMLA (mla_decode)"       "needs sm90+ (Hopper); device is sm${SM}"
  skip "Machete W4A16 (vllm)"        "needs sm90a (Hopper); Marlin used instead on sm${SM}"
  skip "NVFP4/MXFP4 GEMM"            "needs sm100 (Blackwell); device is sm${SM}"
fi

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
note "INSTALL SUMMARY (GPU: ${GPU_NAME:-<none>}, sm${SM})"

echo ""
echo "Importable (${#LOG_OK[@]}):"
if [ "${#LOG_OK[@]}" -eq 0 ]; then echo "  (none)"; fi
for x in "${LOG_OK[@]:-}"; do [ -n "$x" ] && echo "  ok   - $x"; done

echo ""
echo "Failed (${#LOG_FAIL[@]}):"
if [ "${#LOG_FAIL[@]}" -eq 0 ]; then echo "  (none)"; fi
for x in "${LOG_FAIL[@]:-}"; do [ -n "$x" ] && echo "  fail - $x"; done

echo ""
echo "Skipped (${#LOG_SKIP[@]}):"
if [ "${#LOG_SKIP[@]}" -eq 0 ]; then echo "  (none)"; fi
for x in "${LOG_SKIP[@]:-}"; do [ -n "$x" ] && echo "  skip - $x"; done

# ----------------------------------------------------------------------------
# INSTALL_ATOMIC — the cross-library atomic-op benchmark (bench_atomic.py)
#
# bench_atomic.py groups providers/atomic_ops.json (476 atomic library entry
# points across flashinfer / sgl / vllm) by their logical_op and runs a
# per-logical-op cross-LIBRARY comparison. It REUSES the exact provider call
# adapters from bench_sota.py, so it needs NO new dependencies — the libraries
# installed above (flashinfer, sgl-kernel, vllm) are the same ones it drives.
# Nothing extra to install here; this note just documents the mapping:
#
#   logical_op               benched via the libs installed above
#   -----------------------  --------------------------------------------------
#   norm.rmsnorm             flashinfer.norm.rmsnorm / vllm.rms_norm /
#                            sgl.rmsnorm (+ kernel-set ks_rmsnorm)
#   norm.fused_add_rmsnorm   flashinfer / vllm.fused_add_rms_norm / sgl
#                            (+ kernel-set ks_fused_add_rmsnorm)
#   act.silu_mul             flashinfer / vllm.silu_and_mul / sgl
#                            (+ kernel-set ks_silu_and_mul)
#   act.gelu_mul             flashinfer / vllm.gelu_and_mul / sgl
#   act.gelu_tanh_mul        flashinfer / vllm.gelu_tanh_and_mul / sgl
#   rope.apply               flashinfer apply_rope_with_cos_sin_cache /
#                            vllm.rotary_embedding / sgl.rotary_embedding
#                            (+ kernel-set ks_rope)
#   moe.gate_softmax         sgl.topk_softmax / vllm.topk_softmax
#   gemm.w8a8                sgl.int8_scaled_mm / vllm.cutlass_scaled_mm
#                            (+ kernel-set ks_gemm_w8a8)
#
# Atomic ops WITHOUT a verified call adapter (the other multi-lib logical_ops)
# are reported `cataloged, no-adapter` rather than guessed at. To see the full
# catalog + adapter coverage WITHOUT a GPU: python3 benchmarks/bench_atomic.py
# --list  (works on a CPU box; it is just JSON introspection).
# ----------------------------------------------------------------------------
note "INSTALL_ATOMIC: bench_atomic.py reuses the libs installed above (flashinfer/sgl-kernel/vllm) — nothing extra to install."

echo ""
note "Next: python3 benchmarks/bench_sota.py --dtype fp16 --output results/sota.md"
note "  or: python3 benchmarks/bench_atomic.py --dtype fp16 --output results/atomic.md  (cross-library atomic-op comparison)"
echo "(providers that failed/skipped above will show as import-fail / skip / cataloged-no-adapter rows — that is expected, not a crash.)"

# Always exit 0: a partial install is a valid, expected outcome.
exit 0
