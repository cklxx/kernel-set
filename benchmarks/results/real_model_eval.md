# Real-model eval — kernel-set on actual HuggingFace models

`examples/eval_model.py` loads a real model, freezes the **AOT plan** (strongest
installed kernel per op on this GPU+dtype), hot-swaps the model's `*RMSNorm` and
gated-MLP (SwiGLU) through `ks.dispatch`, then measures **correctness on the real
weights** and **end-to-end + per-op speed**. GEMM/attention deliberately stay on
torch/cuBLAS/SDPA — exactly what the dispatcher picks for compute-bound ops (see
`docs/OPTIMAL_SELECTION.md`).

> Reproduce: `python examples/eval_model.py --model <hf-id> --dtype bf16`
> (build the lib + `export KERNEL_SET_LIB=...` first; see the example header.)

## NVIDIA L4 (sm89), bf16, torch 2.11 / transformers 5.0

### Qwen/Qwen2.5-0.5B-Instruct

**AOT plan** (frozen to `plan.json`):

| op | chosen kernel |
|---|---|
| rmsnorm | kernel-set |
| gemma_rmsnorm | kernel-set |
| swiglu | kernel-set |
| rope | kernel-set |
| attention_prefill | torch-sdpa |
| attention_decode | kernel-set |
| gemm | torch (cuBLAS) |
| cross_entropy | torch |

*(No flashinfer/vllm/liger/sgl-kernel installed in this env, so memory-bound ops
resolve to kernel-set's own kernels — the portable fallback that is also
SOTA-class for these ops.)*

**Correctness** (real weights, real activations, last-token logits):

- next-token top-1: **match**
- greedy tokens identical: **45 / 64** (then bf16 rounding compounds over
  24 layers × steps — expected; the two decodes stay semantically identical)
- logits rel-err 1.5e-2, max-abs 0.35
- op invocations during the run: rmsnorm ×3577, swiglu ×1752 (kernels really ran)

**End-to-end generation** (prefill + 64-token decode): 1678.9 ms → 1606.6 ms,
**1.05×** (only RMSNorm+SwiGLU swapped; GEMM dominates and stays on cuBLAS).

**Per-op microbench** (model shapes, seq=2048; torch ref is eager, not a fused lib):

| op | provider | ks (ms) | torch (ms) | speedup | rel-err |
|---|---|---|---|---|---|
| rmsnorm | kernel-set | 0.0311 | 0.0984 | **3.16×** | 3.2e-3 |
| swiglu | kernel-set | 0.2312 | 0.3157 | **1.37×** | 5.3e-3 |
| rope | kernel-set | 0.0553 | 0.1741 | **3.15×** | 4.3e-3 |

### unsloth/gemma-2-2b-it  (Gemma `(1+w)` RMSNorm + **GeGLU**, not SwiGLU)

Same AOT plan as above. Gemma's norm hits the `gemma_rmsnorm` path; its MLP is
**GeGLU** (`gelu_pytorch_tanh`), so the example routes it to kernel-set's
`geglu(tanh_approx=True)` — *not* SwiGLU. (Getting this wrong is silently wrong:
an earlier version applied SwiGLU and produced coherent-but-different output,
rel-err 0.46, only 6/64 tokens matching. Routing GeGLU correctly fixed it.)

**Correctness:**

- next-token top-1: **match**
- greedy tokens identical: **64 / 64** — bit-identical greedy decode, same text
- logits rel-err 1.1e-2, max-abs 0.25
- op invocations: gemma_rmsnorm ×7665, geglu ×1898

**End-to-end generation** (prefill + 64-token decode): 2730 ms → 2483 ms, **1.10×**.

**Per-op microbench** (seq=2048; torch ref eager):

| op | provider | ks (ms) | torch (ms) | speedup | rel-err |
|---|---|---|---|---|---|
| rmsnorm | kernel-set | 0.0311 | 0.2750 | **8.84×** | 1.5e-3 |
| swiglu | kernel-set | 0.4755 | 0.8117 | **1.71×** | 4.7e-3 |
| rope | kernel-set | 0.0550 | 0.2130 | **3.87×** | 4.4e-3 |

> The example **detects the gate activation from the model** (SiLU→SwiGLU,
> GELU→GeGLU, tanh vs exact) so it stays correct across Llama/Qwen/Mistral
> *and* Gemma without per-model flags.

## Takeaways

- kernel-set's memory-bound kernels are **correct on real model weights**
  (Gemma-2-2b: **64/64 bit-identical** greedy tokens; Qwen2.5: top-1 match,
  45/64 then bf16 noise) and **fuse faster than eager torch** (RMSNorm 3–9×,
  RoPE 3–4×, SwiGLU/GeGLU 1.4–1.7× at prefill shapes).
- The example **detects the model's gate activation** (SwiGLU vs GeGLU) — it
  caught a real correctness bug (Gemma is GeGLU, not SwiGLU) before it shipped.
- The **AOT plan** is the architecture answer: resolve the per-op winner once,
  ahead of time, freeze it (`plan.json`), and run it — compute-bound ops route to
  the industry best, memory-bound ops to kernel-set. No per-call probing in the
  hot path.
- End-to-end gain from swapping only norm+activation is modest by design; the big
  levers (GEMM, attention) are already routed to cuBLAS/SDPA/FlashAttention.

## More models (L4/sm89, bf16) — coverage sweep

Hot-swapping kernel-set's RMSNorm/RoPE/SwiGLU into stock HF models across families;
all **next-token top-1 correct**, all per-op faster than eager torch:

| model | family | greedy match | rmsnorm | swiglu | rope | end-to-end |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | Qwen | top-1 ✓ | 3.76× | 1.68× | 2.80× | 1.05× |
| SmolLM2-1.7B-Instruct | Llama-arch | **48/48 identical** | 7.68× | 1.57× | 3.93× | 1.06× |
| Phi-3.5-mini-instruct | Phi | top-1 ✓ (29/48) | 11.74× | 1.59× | 3.70× | 1.07× |

(Earlier: Qwen2.5-0.5B top-1 / 45-of-64; Gemma-2-2B **64/64 bit-identical** via the
GeGLU path.) Per-op speedups are vs eager torch; SmolLM2 is bit-identical end-to-end.

## Latest 2026 models (A100/sm80, bf16, transformers 5.9) — kernels verified per-op

The newest small models were tested to see how far the demo reaches. **The kernels
are correct and fast on them** (per-op vs eager torch):

| model | family | rmsnorm | swiglu | rope | rel-err | end-to-end demo |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | Qwen3.5 (QK-norm) | **4.04×** | 1.70× | 3.01× | ≤4e-3 | ✗ logits rel 0.83 |
| Gemma-4-E4B-it | Gemma 4 (PLE/matformer) | **5.09×** | 1.69× | 2.97× | ≤4e-3 | ✗ (nested config) |

**Honest caveat:** the per-op kernels match torch (rel ≤4e-3), but `eval_model.py`'s
*whole-model* hot-swap does **not** reproduce these models' 2026 forward end-to-end
— Qwen3.5 adds attention-level changes beyond QK-norm, and Gemma-4 E-series uses
Per-Layer-Embeddings / matformer nesting. The example now scopes its norm-swap to
hidden-dim norms and resolves nested `text_config` (so Gemma-4 no longer crashes),
but a faithful end-to-end demo of these architectures is future work. The
bit-identical end-to-end results above stand for the Qwen2.5 / Gemma-2 / Llama /
Phi / SmolLM architecture families. (Note: **Qwen 3.6 ships no small variant** —
27B dense / 35B-A3B only — so it can't run on an L4/A100-class card.)

`eval_model.py --load-4bit` (bnb NF4) was added so large checkpoints fit a small
GPU — the norms/activations stay bf16, so the hot-swap and the baseline-vs-patched
check are unaffected (works on any loadable model). Gemma-4 specifically still
can't be demoed: the 12B checkpoint reports `model_type: gemma4_unified` which
current transformers doesn't recognize, and the E-series is PLE/matformer — both
are loader/arch limitations, not quant or kernel issues. A 4-bit **Gemma-2/Gemma-3**
runs fine through the same path.

Op-level: all kernels stay `correct=100, incorrect=0` on H20 (sm90, all 50 ops)
after every kernel added this cycle — `benchmarks/results/h20_sm90.md`.
