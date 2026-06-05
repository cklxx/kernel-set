#!/usr/bin/env python3
"""Evaluate kernel-set on a *real* HuggingFace model (Qwen / Gemma / Llama …).

What this does, end-to-end, on whatever GPU you run it on:

1. **AOT plan** — resolves the strongest *installed* kernel for every op the
   model uses (``ks.dispatch.which``), on this exact GPU + dtype, and freezes it
   to ``plan.json``. This is the ahead-of-time "best kernel per op" artifact.
2. **Hot-swap** — monkey-patches the model's ``*RMSNorm`` and gated-MLP
   (SwiGLU) modules to route through ``ks.dispatch`` (best-backend; kernel-set's
   own kernel as the portable fallback). GEMM/attention stay on torch/cuBLAS —
   exactly what the dispatcher would pick anyway (see docs/OPTIMAL_SELECTION.md).
3. **Correctness** — runs one forward on the real prompt with the *real weights*
   and compares patched-vs-baseline logits (max-abs-err, rel-err, top-1 token
   agreement). Real weights, real activations — not synthetic tensors.
4. **Speed** — times prefill + decode generation both ways (tokens/s, speedup)
   and micro-benchmarks each swapped op on the model's true shapes.
5. **Generate** — prints greedily-decoded text from both paths so you can see
   the output is unchanged.

Run (Colab T4/L4/A100, small models load fine):

    pip install -U "transformers>=4.44" accelerate torch
    cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j
    export KERNEL_SET_LIB=$PWD/build/libkernel_set.so
    export PYTHONPATH=bindings/python
    python examples/eval_model.py --model Qwen/Qwen2.5-0.5B-Instruct
    python examples/eval_model.py --model google/gemma-2-2b-it --dtype bf16

Everything degrades gracefully: no GPU, no transformers, or no built library
each prints a clear, actionable message instead of a traceback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager

# --------------------------------------------------------------------------- #
# Imports with friendly diagnostics (this example has real heavyweight deps).
# --------------------------------------------------------------------------- #
try:
    import torch
except ImportError:
    sys.exit("This example needs PyTorch:  pip install torch")

# Make `import kernel_set` work from a source checkout without installing.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYBIND = os.path.join(_ROOT, "bindings", "python")
if _PYBIND not in sys.path:
    sys.path.insert(0, _PYBIND)

import kernel_set as ks  # noqa: E402  (after sys.path tweak)


# --------------------------------------------------------------------------- #
# Small helpers.
# --------------------------------------------------------------------------- #
def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def dtype_of(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16,
            "fp32": torch.float32}[name]


@contextmanager
def cuda_timer():
    """Yields a 1-element list that gets the elapsed-ms after the block."""
    out = [0.0]
    if torch.cuda.is_available():
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        s.record()
        yield out
        e.record()
        torch.cuda.synchronize()
        out[0] = s.elapsed_time(e)
    else:
        t0 = time.perf_counter()
        yield out
        out[0] = (time.perf_counter() - t0) * 1e3


def rel_err(a, b) -> float:
    a, b = a.float(), b.float()
    denom = b.abs().max().clamp_min(1e-12)
    return ((a - b).abs().max() / denom).item()


# --------------------------------------------------------------------------- #
# Op-level hot-swap: patch RMSNorm + gated-MLP forwards to ks.dispatch.
# We patch by *behaviour*, detected from module attributes / class name, so the
# same code works across Qwen2/Qwen3, Gemma/Gemma2/Gemma3, Llama, Mistral, …
# --------------------------------------------------------------------------- #
def _is_rmsnorm(m) -> bool:
    return type(m).__name__.endswith("RMSNorm") and hasattr(m, "weight")


def _is_gemma_norm(m) -> bool:
    # Gemma RMSNorm applies (1 + weight); everyone else applies weight.
    return "Gemma" in type(m).__name__


def _is_gated_mlp(m) -> bool:
    return (hasattr(m, "gate_proj") and hasattr(m, "up_proj")
            and hasattr(m, "down_proj") and hasattr(m, "act_fn"))


def _gate_activation(m):
    """Return ``(kind, tanh_approx)`` where kind is 'silu' or 'gelu'. SwiGLU
    models (Llama/Qwen/Mistral) are SiLU; GeGLU models (Gemma) are GELU — Gemma
    specifically uses the tanh approximation (``gelu_pytorch_tanh``)."""
    name = getattr(m.act_fn, "__name__", type(m.act_fn).__name__).lower()
    if "silu" in name or "swish" in name:
        return "silu", False
    if "gelu" in name:
        # gelu_pytorch_tanh / gelu_new / fast-gelu are tanh approximations.
        return "gelu", ("tanh" in name or "new" in name or "fast" in name)
    return "silu", False  # default: treat unknown gated act as SiLU


def patch_model(model, *, counters: dict):
    """Replace RMSNorm + SwiGLU-MLP forwards in-place with ks.dispatch calls.
    ``counters`` accumulates per-op invocation counts so we can prove the
    kernels actually ran. Returns the chosen provider name per op."""
    chosen: dict[str, str] = {}

    for m in model.modules():
        if _is_rmsnorm(m):
            gemma = _is_gemma_norm(m)
            eps = float(getattr(m, "variance_epsilon",
                                getattr(m, "eps", 1e-6)))
            op = "gemma_rmsnorm" if gemma else "rmsnorm"
            chosen[op] = ks.dispatch.which(op)

            def _norm_fwd(x, _w=m.weight, _eps=eps, _gemma=gemma):
                shp = x.shape
                x2 = x.reshape(-1, shp[-1]).contiguous()
                if _gemma:
                    out = ks.dispatch.gemma_rmsnorm(x2, _w, eps=_eps)
                else:
                    out = ks.dispatch.rms_norm(x2, _w, eps=_eps)
                counters[op] = counters.get(op, 0) + 1
                return out.reshape(shp)

            m.forward = _norm_fwd

        elif _is_gated_mlp(m):
            # CRITICAL: pick the gate activation the model actually uses. Llama/
            # Qwen/Mistral are SiLU (SwiGLU); Gemma is GELU-tanh (GeGLU). Using
            # the wrong one is silently wrong (coherent-but-different output).
            kind, tanh = _gate_activation(m)
            op = "swiglu" if kind == "silu" else "geglu"
            chosen[op] = (ks.dispatch.which("swiglu") if kind == "silu"
                          else "kernel-set")

            def _mlp_fwd(x, _m=m, _kind=kind, _tanh=tanh, _op=op):
                gate = _m.gate_proj(x)
                up = _m.up_proj(x)
                shp = gate.shape
                g2 = gate.reshape(-1, shp[-1]).contiguous()
                u2 = up.reshape(-1, shp[-1]).contiguous()
                if _kind == "silu":
                    act = ks.dispatch.swiglu(g2, u2)
                else:
                    from kernel_set import activation as _ksact
                    act = torch.empty_like(g2)
                    _ksact.geglu(act, g2, u2, tanh_approx=_tanh)
                counters[_op] = counters.get(_op, 0) + 1
                return _m.down_proj(act.reshape(shp))

            m.forward = _mlp_fwd

    return chosen


# --------------------------------------------------------------------------- #
# The AOT plan: which kernel wins for every op this model touches.
# --------------------------------------------------------------------------- #
def build_plan(dtype_name: str) -> dict:
    ops = ["rmsnorm", "gemma_rmsnorm", "swiglu", "rope",
           "attention_prefill", "attention_decode", "gemm", "cross_entropy"]
    plan = {}
    for op in ops:
        try:
            plan[op] = ks.dispatch.which(op, dtype=dtype_name)
        except Exception as e:  # unknown op on this build
            plan[op] = f"<n/a: {e}>"
    return plan


# --------------------------------------------------------------------------- #
# Per-op micro-benchmark on the model's *real* shapes.
# --------------------------------------------------------------------------- #
def microbench(cfg, dtype, device, *, seq=2048, iters=100) -> list:
    hidden = cfg.hidden_size
    inter = getattr(cfg, "intermediate_size", 4 * hidden)
    n_heads = getattr(cfg, "num_attention_heads", max(1, hidden // 128))
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", hidden // n_heads)
    rows = seq
    rows_t = torch.randn(rows, hidden, device=device, dtype=dtype)
    w = torch.ones(hidden, device=device, dtype=dtype)
    gate = torch.randn(rows, inter, device=device, dtype=dtype)
    up = torch.randn(rows, inter, device=device, dtype=dtype)
    q = torch.randn(rows, n_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(rows, n_kv, head_dim, device=device, dtype=dtype)
    cos = torch.randn(rows, head_dim // 2, device=device, dtype=dtype)
    sin = torch.randn(rows, head_dim // 2, device=device, dtype=dtype)

    def bench(fn):
        for _ in range(5):
            fn()
        with cuda_timer() as t:
            for _ in range(iters):
                fn()
        return t[0] / iters

    def torch_rms():
        x = rows_t.float()
        v = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(v + 1e-6)).to(dtype) * w

    def torch_swiglu():
        return torch.nn.functional.silu(gate) * up

    def torch_rope():
        # NeoX rotate_half on (tokens, heads, head_dim) with half-width cos/sin.
        c = torch.cat([cos, cos], -1)[:, None, :]
        s = torch.cat([sin, sin], -1)[:, None, :]
        def rot(t):
            h = t.shape[-1] // 2
            return torch.cat([-t[..., h:], t[..., :h]], -1)
        return q * c + rot(q) * s, k * c + rot(k) * s

    results = []
    for name, ks_fn, ref_fn in [
        ("rmsnorm", lambda: ks.dispatch.rms_norm(rows_t, w), torch_rms),
        ("swiglu", lambda: ks.dispatch.swiglu(gate, up), torch_swiglu),
        ("rope", lambda: ks.dispatch.rope(q, k, cos, sin), torch_rope),
    ]:
        prov = ks.dispatch.which(name)
        try:
            ks_ms = bench(ks_fn)
            ref_ms = bench(ref_fn)
            a, b = ks_fn(), ref_fn()
            if isinstance(a, tuple):
                err = max(rel_err(x, y) for x, y in zip(a, b))
            else:
                err = rel_err(a, b)
            results.append((name, prov, ks_ms, ref_ms, ref_ms / ks_ms, err))
        except Exception as e:
            results.append((name, prov, None, None, None, str(e)))
    return results


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HF model id (small ones run on Colab T4/L4)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--prompt", default="Explain what a GPU kernel is, briefly.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--plan-out", default="plan.json")
    ap.add_argument("--plan-only", action="store_true",
                    help="just emit the AOT plan (no model download / GPU)")
    args = ap.parse_args()

    dtype = dtype_of(args.dtype)
    sm = ks.dispatch.resolve_sm(None)
    print(f"== kernel-set real-model eval ==")
    print(f"GPU       : {gpu_name()}  (sm{sm or '?'})")
    print(f"model     : {args.model}")
    print(f"dtype     : {args.dtype}\n")

    # 1) AOT plan — always works, no GPU / model needed.
    plan = build_plan(args.dtype)
    print("AOT plan (strongest installed kernel per op):")
    for op, prov in plan.items():
        print(f"  {op:18s} -> {prov}")
    with open(args.plan_out, "w") as f:
        json.dump({"gpu": gpu_name(), "sm": sm,
                   "dtype": args.dtype, "model": args.model, "plan": plan},
                  f, indent=2)
    print(f"\nfroze plan -> {args.plan_out}")
    if args.plan_only:
        return 0

    if not torch.cuda.is_available():
        print("\n[!] no CUDA GPU — plan emitted above; run on Colab for perf.")
        return 0
    if not getattr(ks, "_LIB_AVAILABLE", False):
        print("\n[!] libkernel_set not built/loaded — set KERNEL_SET_LIB after "
              "`cmake --build build`. Plan still emitted above.")
        return 0

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("\n[!] needs transformers:  pip install -U transformers accelerate")
        return 0

    device = "cuda"
    print(f"\nloading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(device).eval()
    cfg = model.config

    def _as_ids(x):
        # transformers ≥5 may return a BatchEncoding/dict; normalize to a tensor.
        if hasattr(x, "input_ids"):
            x = x.input_ids
        elif isinstance(x, dict):
            x = x["input_ids"]
        return x.to(device)

    msgs = [{"role": "user", "content": args.prompt}]
    try:
        ids = _as_ids(tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"))
    except Exception:
        ids = _as_ids(tok(args.prompt, return_tensors="pt"))

    # 2) Baseline forward (real weights) -> reference logits.
    with torch.no_grad():
        base_logits = model(ids).logits[:, -1, :].float().clone()

    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                  pad_token_id=tok.eos_token_id)
    with torch.no_grad():
        for _ in range(2):  # warmup
            model.generate(ids, max_new_tokens=4, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        with cuda_timer() as t_base:
            base_out = model.generate(ids, **gen_kw)
    n_new = base_out.shape[1] - ids.shape[1]
    base_text = tok.decode(base_out[0, ids.shape[1]:], skip_special_tokens=True)

    # 3) Hot-swap to kernel-set and re-run.
    counters: dict[str, int] = {}
    chosen = patch_model(model, counters=counters)
    print("\nhot-swapped ops -> providers:")
    for op, prov in chosen.items():
        print(f"  {op:18s} -> {prov}")

    with torch.no_grad():
        ks_logits = model(ids).logits[:, -1, :].float().clone()
        for _ in range(2):
            model.generate(ids, max_new_tokens=4, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        with cuda_timer() as t_ks:
            ks_out = model.generate(ids, **gen_kw)
    ks_text = tok.decode(ks_out[0, ids.shape[1]:], skip_special_tokens=True)

    # 4) Correctness on the real last-token logit vector + generated tokens.
    err = rel_err(ks_logits, base_logits)
    max_abs = (ks_logits - base_logits).abs().max().item()
    top1 = (ks_logits.argmax(-1) == base_logits.argmax(-1)).all().item()
    # First position where the two greedy decodes diverge (bf16 noise compounds
    # over layers×steps; an exact match for many tokens is the realistic bar).
    a, b = base_out[0, ids.shape[1]:], ks_out[0, ids.shape[1]:]
    n = min(a.numel(), b.numel())
    match = int((a[:n] == b[:n]).sum().item())

    print("\n--- correctness (real weights, real activations) ---")
    print(f"  logits rel-err     : {err:.3e}")
    print(f"  logits max-abs-err : {max_abs:.3e}")
    print(f"  next-token top-1   : {'match' if top1 else 'DIFFER'}")
    print(f"  greedy tokens match: {match}/{n} (bf16 noise compounds after)")
    print(f"  op invocations     : {counters}")

    # 5) End-to-end speed.
    base_tps = n_new / (t_base[0] / 1e3)
    ks_tps = n_new / (t_ks[0] / 1e3)
    print("\n--- end-to-end generation (prefill + decode) ---")
    print(f"  baseline (torch)   : {t_base[0]:8.1f} ms  ({base_tps:6.1f} tok/s)")
    print(f"  kernel-set swapped : {t_ks[0]:8.1f} ms  ({ks_tps:6.1f} tok/s)")
    print(f"  speedup            : {t_base[0]/t_ks[0]:.3f}x  ({n_new} new tokens)")

    # 6) Per-op microbench on the model's real shapes.
    bench_seq = 2048
    print(f"\n--- per-op microbench (model shapes, seq={bench_seq}) ---")
    print("  (torch ref is eager silu*mul / fp32-rms / rotate_half, not a "
          "fused lib)")
    print(f"  {'op':10s} {'provider':12s} {'ks(ms)':>9s} {'torch(ms)':>10s} "
          f"{'speedup':>8s} {'rel-err':>10s}")
    for name, prov, ks_ms, ref_ms, sp, err in microbench(cfg, dtype, device,
                                                         seq=bench_seq):
        if ks_ms is None:
            print(f"  {name:10s} {prov:12s}  <skipped: {err}>")
        else:
            print(f"  {name:10s} {prov:12s} {ks_ms:9.4f} {ref_ms:10.4f} "
                  f"{sp:7.2f}x {err:10.2e}")

    print("\n--- generated text ---")
    print(f"[baseline ]  {base_text!r}")
    print(f"[kernelset]  {ks_text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
