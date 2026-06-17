# Quantization operators — what kernel-set ships, what dispatch routes, and the gaps

This is the quantization companion to the routing docs. It answers three
questions, in order:

1. **What kernel-set implements itself** — the real `.cu` quant/dequant/GEMM
   kernels behind the C ABI (`include/kernel_set/quant.h`,
   `include/kernel_set/gemm.h`), with each op's GPU-verification status.
2. **What is dispatchable** — the logical quant ops the Tier-2 runtime dispatcher
   routes to the industry-best provider, with kernel-set as the terminal
   fallback; plus the broader gemm-quant catalog enumerated in
   `providers/registry.json`.
3. **The prioritized gap analysis** — split into *true holes* (a provider exists
   in the catalog but is not yet wired into dispatch, so it dead-ends at runtime)
   and *dispatch-covered gaps* (kernel-set has no own kernel but a routed
   provider exists, so self-dev is lower ROI).

See [`OPTIMAL_SELECTION.md`](OPTIMAL_SELECTION.md) for the `op × sm × dtype`
decision matrix and [`OPERATOR_CATALOG.md`](OPERATOR_CATALOG.md) for the full
ranked provider catalog across all 127 operators.

---

## kernel-set's quant strategy

The standing policy from [`OPTIMAL_SELECTION.md`](OPTIMAL_SELECTION.md) applies
in full to quantization: **route to industry-best, self-develop only where it
adds value.** Quantized GEMM is compute-bound, so on a fully-provisioned host the
rank-1 is always the specialist kernel (DeepGEMM, vLLM Marlin/Machete/CUTLASS,
FlashInfer FP4) and kernel-set is the rank-99 *correctness* fallback. Two things
make kernel-set's own quant kernels worth keeping anyway:

* **Portable sm80+ fallbacks.** Every dispatch chain ends in `kernel-set`, so a
  call never dead-ends on a bare host (offline / Colab / a Rust or Go embedding
  with no PyTorch). kernel-set's quant GEMMs are SIMT software-dequant paths that
  *compile and run on sm80+* — including archs where the specialist has **no
  provider at all** (e.g. there is no FP8 GEMM provider for A100/A800, the core
  DeepSeek serving GPU). The portable fallback is the whole point there.
* **Round-trip correctness primitives.** The standalone quant/dequant kernels
  (`ks_quantize_fp8`, `ks_dequantize_int4`, …) are GPU-verified and feed the
  reference/test harness and the C-ABI bindings, independent of any Python
  provider being installed.

The flip side: kernel-set's quant GEMMs run at ~2 % of peak today. Preferring
them over an installed specialist would be a 50–100× regression, which is exactly
why they sit terminal in every chain.

---

## 1. What kernel-set implements (the `kernel_set_abi` surface)

Eight quant ops are real CUDA kernels in `kernels/src/quant/` and
`kernels/src/gemm/`. Status legend: **GPU-verified** = correctness-gated against
a torch reference on real silicon; **correctness-only** = numerically verified
but not perf-competitive; **throughput-only** = runs and benched but *not*
correctness-gated; **pending** = just-added, header-declared, awaiting GPU verify.

| ABI symbol | File | What it does | Layout / modes | Status |
| --- | --- | --- | --- | --- |
| `ks_quantize_fp8` / `ks_dequantize_fp8` | `quant/quant_fp8.cu` | fp8 e4m3/e5m2 ↔ f16/bf16/f32 | per-tensor / per-token; fp32 scale | **GPU-verified** sm89+ (round-trip rel-L2 ~2.65e-2; dequant 0.0 on L4 + Blackwell). `ARCH_UNSUPPORTED` on sm80 (no FP8 HW) |
| `ks_quantize_int8` / `ks_dequantize_int8` | `quant/quant_int8.cu` | int8 symmetric ↔ f16/bf16/f32 | per-tensor / per-token; `scale=amax/127` | **GPU-verified** (round-trip rel-L2 ~8.6e-3; dequant 0.0) |
| `ks_dequantize_int4` | `quant/dequant_int4.cu` | int4 packed AWQ/GPTQ affine → out dtype | group-wise along K; scales/zeros `[K/group, N]`. **Packing: int32 words, 8 nibbles packed along K** (`[K/8, N]`, `q = (word >> 4j) & 0xF`) | **GPU-verified** (rel_err 0 vs exact unpack ref) |
| `ks_gemm_w8a8` | `gemm/gemm_w8a8.cu` | int8 × int8 → out dtype | dp4a int32 accum; A per-token, B per-channel | **GPU-verified** (rel_err 0) but **~2 % peak** |
| `ks_gemm_fp8` | `gemm/gemm_fp8.cu` | fp8 × fp8 → out dtype, SIMT software-dequant, fp32 accum, runs **sm80+** | per-tensor / per-token / per-channel; **no blockwise mode** | **Correctness-only** (rel_err 0.00246 on L4, 5.6 TFLOP/s ≈ 2 % peak). HW fp8-MMA path inert behind `KS_ENABLE_FP8_MMA` |
| `ks_gemm_w4a16` | `gemm/gemm_w4a16.cu` | int4 weights × f16/bf16, group-wise | **Packing: two int4/byte** (even-K low nibble, odd-K high) — **DIFFERENT** from `ks_dequantize_int4` and from on-disk AWQ/GPTQ | **Throughput-only** — *not correctness-verified on any GPU* |
| `ks_gemm_fp8_blockwise` | `gemm/gemm_fp8_blockwise.cu` | DeepSeek-V3 blockwise fp8: 128×128 weight block / 1×128 act tile, fp32 two-level accumulation, sm80+ software dequant | `a_scale [M, ⌈K/block_k⌉]`, `b_scale [⌈K/block_k⌉, ⌈N/block_n⌉]` | **NEW / GPU-verified L4** (max rel_err 3.7e-2, incl. ragged shapes) |
| `ks_quantize_fp8_group` | `quant/quant_fp8_group.cu` | 1×group dynamic fp8 quant (per-token-group activation format the blockwise GEMM consumes) | `scale [rows, ⌈cols/group_size⌉]`, group_size typ. 128 | **NEW / GPU-verified L4** (round-trip rel_err 3.1e-2) |

### Sharp edges in kernel-set's own quant kernels

* **Two incompatible int4 packings.** `ks_dequantize_int4` reads int32 words
  (8 nibbles along K, `[K/8, N]`); `ks_gemm_w4a16` reads two int4 per byte
  (even-K low / odd-K high). Neither matches on-disk AWQ/GPTQ, and **kernel-set
  ships no weight-repack op** — so a real int4 checkpoint cannot be loaded into
  `ks_gemm_w4a16` without an external repack. (See the repack gap below.)
* **No KV-cache quant.** `ks_reshape_and_cache` is **dtype-preserving** — it does
  not quantize the cache.
* **No quantized MoE.** kernel-set's MoE grouped-GEMM (`ks_moe_grouped_gemm`) is
  **bf16 only**; there is no fp8/int8/int4 MoE path.
* **`ks_gemm_fp8` has no blockwise mode** — that is what the just-added
  `ks_gemm_fp8_blockwise` adds.

---

## 2. What is dispatchable

### Wired logical ops (Tier-2 dispatch)

Exactly three quant logical ops are wired into the runtime dispatcher today
(`backends/_registry.py`, `gemm-quant` domain). Each routes to the industry-best
provider for the `(sm, dtype)` cell and falls back to kernel-set:

| Logical op | sm80 (Ampere) | sm89 (Ada) | sm90 (Hopper) | sm100 (Blackwell) | Terminal fallback |
| --- | --- | --- | --- | --- | --- |
| `fp8_gemm` | — *(no FP8 HW on Ampere)* | vLLM-CUTLASS / torch `_scaled_mm` / FBGEMM | deep_gemm (blockwise) | deep_gemm (blockwise) | `ks_gemm_w8a8`* |
| `int8_gemm` | vLLM CUTLASS / GemLite | vLLM CUTLASS / GemLite | vLLM CUTLASS | vLLM Marlin-int8 | `ks_gemm_w8a8` |
| `w4a16` | unified Marlin / GemLite / torchao-int4 | unified Marlin / GemLite | vLLM Machete | vLLM Machete | `ks_gemm_w4a16` |
| `w4a8` *(new)* | unified Marlin-QQQ | unified Marlin-QQQ | vLLM Machete | vLLM Machete | `ks` raise |

Chain detail (from `_registry.py`, 2026 re-survey wiring):

* **`fp8_gemm`**: deep_gemm (sm90/100 blockwise + masked grouped) → vLLM-CUTLASS
  `cutlass_scaled_mm` → torch-scaled-mm → FBGEMM rowwise → sgl-kernel →
  kernel-set. No sm80/sm86 cell (no FP8 tensor cores on Ampere). \*ks fallback is
  `ks_gemm_w8a8`; the portable `ks_gemm_fp8_blockwise` covers sm80.
* **`int8_gemm`**: vLLM Marlin-int8 (sm100) → vLLM CUTLASS (sm80) → GemLite →
  kernel-set.
* **`w4a16`**: vLLM Machete (sm90a) → **unified Marlin** GPTQ/AWQ (sm80) → GemLite
  → torchao-int4 → kernel-set.
* **`w4a8`** *(new op)*: vLLM Machete (int4 weight + fp8/int8 act, sm90) → unified
  Marlin-QQQ (sm80) → kernel-set terminal.

> **Marlin is unified.** The vendored 2026 vLLM collapsed `gptq_marlin_gemm`,
> `fp8_marlin_gemm`, `marlin_qqq_gemm`, `gptq_marlin_24_gemm` into a single
> `ops.marlin_gemm(..., b_q_type, ...)` selected by scalar type (GPTQ `kU4B8`,
> AWQ `kU4`, int8 `kS8`, QQQ `kS4`, fp8 `kFE4M3fn`). kernel-set's adapters were
> updated to call it (the old per-flavor symbols no longer exist).

### Broader gemm-quant catalog (`providers/registry.json`)

The provider catalog enumerates more quant ops than dispatch currently wires.
These have ranked providers documented but are **not** all reachable via Tier-2
dispatch (see the gap analysis — the unwired ones are the *true holes*):

| Catalog op | Rank-1 provider | Wired into dispatch? |
| --- | --- | --- |
| `fp8_gemm_blockwise` | deep_gemm | via `fp8_gemm` (sm90/100) |
| `fp8_gemm_scaled_mm` | vllm | via `fp8_gemm` (sm89) |
| `quantize_fp8_dynamic` | vllm | partial — ks has per-token only |
| `dequantize_fp8` | torchao (`ks_dequantize_fp8`) | ks ABI only |
| `int8_gemm_w8a8` | vllm | via `int8_gemm` |
| `quantize_int8_dynamic` | vllm | ks ABI only |
| `dequantize_int8` | torchao (`ks_dequantize_int8`) | ks ABI only |
| `w4a16_gemm` | vllm (Marlin / GPTQ-Marlin) | via `w4a16` |
| `awq_gemm` | vllm (awq_marlin) | via `w4a16` (after repack) |
| `dequantize_int4` | autoawq (`ks_dequantize_int4`) | ks ABI only |
| `int4_weight_only_gemm_tinygemm` | torchao | **yes** — torchao-int4 in `w4a16` |
| `w4a8_gemm` | vllm (Machete W4A8) | **yes** — own `w4a8` op (Machete + Marlin-QQQ) |
| `nvfp4_gemm` | flashinfer | yes — `nvfp4_gemm` op (sm100) |
| `mxfp4_gemm` | torchao / gemlite | yes — `mxfp4_gemm` op |
| `fp4_quantize` | vllm | **no — true hole** |
| `nf4_fp4_blockwise_quant_linear` | bitsandbytes | **no** |
| `int8_llm_int8_linear` | bitsandbytes | **no** |
| `quantized_low_precision_attention` | sageattention (SageAttention2++) / FA3-fp8 | **no — true hole** |

---

## 3. Gap analysis

Two distinct classes of gap, with different remediation priorities.

### Class 1 — TRUE HOLES (provider exists in `registry.json`, NOT wired into dispatch)

These dead-end at runtime: `optimal.json` emits no cell, so `select_optimal`
returns the kernel-set fallback even though a real specialist provider is cataloged.
These are being wired into dispatch (`_OPS_RAW` / `optimal.json`) now.

| Gap | Priority | Needed by | SOTA reference | Why it's a hole |
| --- | --- | --- | --- | --- |
| **NVFP4 GEMM** | P0 / P1 | DeepSeek / Llama / Qwen ModelOpt NVFP4 on B200 / RTX 5090 | flashinfer `mm_fp4`, vllm CUTLASS FP4 | Wired in `optimal.json` for sm100 fp4 via FlashInfer → vLLM → kernel-set; needs promoted Blackwell measurement. |
| **MXFP4 GEMM** | P1 | OpenAI gpt-oss MXFP4 MoE | vllm Marlin-MXFP4, torchao | Wired in `optimal.json` for sm100 fp4 via FlashInfer → vLLM → torchao → kernel-set; Ampere/Hopper emulation remains a future policy decision. |
| **FP8 KV-cache quant** | P1 | Default in every serving engine (2× KV reduction) | vllm `reshape_and_cache` (fp8) | `ks_reshape_and_cache` is dtype-preserving; registry `kv_cache_reshape_and_cache` is documentation-only |
| **Weight repack GPTQ/AWQ → Marlin/Machete** | P1 | Prerequisite to LOAD any int4 checkpoint into fast kernels | vllm `gptq_marlin_repack` / `awq_marlin_repack` / `machete_prepack_B` | Repack atomics exist but there is no logical op — and ks's two int4 packings don't match on-disk format |
| **Per-token-group (1×128) dynamic quant** | P1 | Prerequisite for all blockwise fp8 | vllm / DeepGEMM act quant | ks had per-token only (addressed by the new `ks_quantize_fp8_group`) |
| **FP8 attention compute** | P1 | DeepSeek MLA fp8 prefill | FA3 fp8, flashinfer fp8 attention | `quantized_low_precision_attention` providers exist (SageAttention / FA3-fp8) but are not wired |

### Class 2 — kernel-set lacks own kernel, but dispatch covers it (lower self-dev ROI)

A routed provider exists for the common archs, so a missing kernel-set kernel is
less urgent — except where the *portable fallback* itself is the value.

| Gap | Priority | Needed by | SOTA reference | Covered by dispatch? |
| --- | --- | --- | --- | --- |
| **FP8 blockwise GEMM** | P0 | DeepSeek-V3 recipe | deep_gemm (sm90/100) | Partly — deep_gemm covers sm90/100, but **sm80/A100 has no fp8 provider at all**. A100/A800 is a core DeepSeek serving GPU → this is why ks is self-developing `ks_gemm_fp8_blockwise` (portable sm80+) |
| **FP8 grouped MoE GEMM** | P0 | DeepSeek / Mixtral fp8 MoE | deep_gemm / sgl-kernel (sm90/100) | Yes on sm90/100; ks MoE is bf16 only |
| **W4A8** | P2 | QServe / Machete W4A8 serving | vllm Machete | Yes (catalog) |
| **bitsandbytes NF4** | P2 | QLoRA fine-tuning | bitsandbytes | Yes (catalog) |
| **MXFP8** | P2 | Blackwell MXFP8 training/inference | transformer-engine | Partial |
| **2:4 structured sparsity** | P2 | Sparse-tensor-core inference | CUTLASS / cuSPARSELt | No |
| **int8 / int4 KV-cache** | P2 | Long-context KV reduction | vllm / kvquant | No |
| **GGUF** | P2 | llama.cpp checkpoints | llama.cpp / vllm GGUF | No |
| **FP8 training (delayed scaling)** | P2 | FP8 pretraining | transformer-engine | No |

---

## What was added

The landed and newly-wired pieces, in one place:

* **`ks_gemm_fp8_blockwise`** (new own kernel, **GPU-verified L4**) — DeepSeek-V3
  blockwise recipe: 128×128 weight block / 1×128 activation tile with fp32
  two-level accumulation, portable SIMT software-dequant on sm80+. This is the
  portable fallback for archs DeepGEMM doesn't cover, **notably A100/A800 sm80**,
  which has no FP8 GEMM provider at all. (max rel_err 3.7e-2 vs bf16 ref, incl.
  ragged shapes.)
* **`ks_quantize_fp8_group`** (new own kernel, **GPU-verified L4**) — 1×group
  dynamic fp8 quantization producing the per-token-group activation format
  (`scale [rows, ⌈cols/group⌉]`) that `ks_gemm_fp8_blockwise` and DeepGEMM
  consume. Closes the Class-1 "per-token-group dynamic quant" prerequisite.
  (round-trip rel_err 3.1e-2.)
* **Six newly-wired dispatch ops** — `fp8_gemm_blockwise`,
  `per_token_group_quant`, `nvfp4_gemm`, `mxfp4_gemm`, `fp8_attention`, and
  `fp8_kv_cache` are now first-class entries in `_OPS_RAW` / `dispatch` /
  `optimal.json` (real provider adapters in `backends/_quant_ext.py`), so they no
  longer dead-end at runtime. NVFP4/MXFP4 resolve to FlashInfer/vLLM on Blackwell
  (sm100+); the rest carry kernel-set terminals (the new blockwise/group kernels,
  or a clear "needs FlashInfer/vLLM" error for fp4/fp8-attn/fp8-KV). The strong
  `gen_optimal.py --check` gate is green (533 cells); promote any measured
  Blackwell FP4 winners into the promoted `MEASURED` block per
  [`OPTIMAL_SELECTION.md`](OPTIMAL_SELECTION.md).

---

## 4. 2026 re-survey (集百家之长) — expanded provider coverage

A fresh survey of the SOTA quant/kernel landscape (web-verified) drove a provider
expansion so dispatch always selects best-of-all-libraries per `op × sm × dtype`.
Landed (commit `b764f57`, 29 dispatch ops, 287 optimal cells):

* **Marlin unified** → `ops.marlin_gemm(b_q_type=…)` (P0 correctness fix; the old
  `gptq_marlin_gemm`/`fp8_marlin_gemm`/`qqq`/`marlin_24` symbols were removed).
* **New providers per op** (lazy adapters, `min_sm`/dtype-gated, ks terminal):
  * fp8: vLLM-CUTLASS `cutlass_scaled_mm`, FBGEMM `f8f8bf16_rowwise`, DeepGEMM
    masked grouped (`m_grouped_fp8_gemm_nt_masked`).
  * int4/int8: GemLite (Triton low-bit), torchao-int4 (tinygemm).
  * MoE: FlashInfer `cutlass_fused_moe`, DeepGEMM grouped.
  * attention: flash-attn-3 native, FlashMLA native, FlashInfer-trtllm; SageAttn.
  * norm / cross-entropy: Quack (Hopper sm90).
  * linear-attn / SSM: FLA `fused_recurrent_*` (decode-optimal), mamba-ssm
    `selective_state_update`, `causal_conv1d_update`.
* **New `w4a8` op** — int4 weight + fp8/int8 activation: Machete (sm90) → unified
  Marlin-QQQ (sm80) → ks terminal. Best W4A8 path Ampere→Hopper.
* **Linear-attn fallbacks cross-checked vs FLA** (fla 0.5.0, GPU): `gated_delta_rule`
  1.0e-2, `gated_linear_attn` 2.6e-3, `rwkv_wkv7` 7.4e-3 — the ks fallbacks and the
  FLA provider are now numerically interchangeable (an rwkv-7 sign bug vs FLA's
  convention was found and fixed here, `da56dad`).

### Still deferred (documented gaps, need vendoring or a model target)

| Gap | Why deferred | SOTA path if needed |
| --- | --- | --- |
| **2:4 structured sparse** | needs vendoring IST-DASLab Sparse-Marlin (legacy `gptq_marlin_24_gemm` removed); dense w4a16/w4a8 higher ROI | vLLM `cutlass_scaled_sparse_mm` (sm90) / Sparse-Marlin (sm80/89) |
| **BitNet W1.58A8 ternary** | no general PTQ→ternary fused GPU GEMM in mainstream serving; only native-BitNet | BitBLAS (pip) / microsoft/BitNet CUDA (source) |
| **w8a16-fp8** | niche (sm80/86/89 fp8 ckpts without fp8 MMA); on sm89+ native `fp8_gemm` preferred | unified Marlin `b_q_type=kFE4M3fn` |
| **fused-linear-CE** | distinct signature (hidden+lm_head, not logits-in); training-only | Liger `LigerFusedLinearCrossEntropy` |
