# Optimal kernel selection — the compute-bound decision matrix

This is the **strategy + decision matrix** behind kernel-set's Tier-2 runtime
dispatcher (`kernel_set.dispatch`, table in
[`backends/_registry.py`](../bindings/python/kernel_set/backends/_registry.py)).
It records, **per compute-bound op × GPU arch × dtype**, the optimal provider,
what dispatch now selects, and the **adopt-external vs self-develop** call.

See [`ROUTING.md`](ROUTING.md) for how the three routing tiers fit together.

---

## The overall strategy (read this first)

kernel-set is *not* trying to out-engineer NVIDIA, vLLM, DeepSeek, or the
FlashAttention authors on the primitives they already own. The policy is:

1. **ADOPT external best-in-class for compute-bound ops.**
   GEMM, attention, MoE grouped-GEMM, and the quantized GEMMs (FP8 / INT8 /
   W4A16) are the most heavily hand-tuned kernels in the ecosystem. For these,
   the dispatcher's **rank-1 is always the industry-best installed provider for
   the (arch, dtype)**; kernel-set is appended **last** (rank-99) as the
   *portable correctness fallback only* — it is never preferred when any external
   provider is available. Measured kernel-set throughput on these ops is
   0.01–0.10× SOTA, so preferring it would be a 10–100× regression.

2. **KEEP + improve kernel-set's own SOTA-class memory-bound kernels.**
   norm / elementwise / rope / activation / sampling / optimizer / loss are
   bandwidth-bound and kernel-set's own C-ABI kernels measure **84–87 % of peak
   HBM bandwidth on A100** — genuinely competitive with FlashInfer/SGLang/Liger.
   For those ops kernel-set is a legitimate ranked provider (not just a
   fallback), and it is the **portable C-ABI path** that makes the Rust / Go / TS
   bindings work with zero Python-ecosystem dependencies. (Those ops are *not*
   in the compute-bound matrix below; their ranking is unchanged.)

3. **SELF-DEVELOP only where there is no good OSS equivalent** *or* where a
   portable, dependency-free fallback must exist for the non-Python bindings.
   See the [SELF-DEVELOP shortlist](#self-develop-shortlist) at the end.

The net effect: on a fully-provisioned GPU host, every compute-bound op routes to
the SOTA kernel; on a bare host (offline / Colab / a Rust or Go embedding with no
PyTorch), the same call still works via the kernel-set C ABI — just slower.

---

## How to read the dispatch decision

For an op + (resolved GPU `sm`, requested dtype) the dispatcher walks the
rank-ordered provider chain and picks the **first selectable** entry:

* its library **imports** (cached probe), AND
* the device **`min_sm`** gate is met (e.g. DeepGEMM/FlashMLA/Machete sm90,
  FA4/NVFP4/Marlin-int8 sm100, Marlin/CUTLASS-int8 sm80, FlashInfer-MLA sm80),
  AND
* its **dtype** support covers the request.

Arch-/import-/dtype-gated providers are skipped silently; the chain always ends
at the always-selectable kernel-set fallback, so dispatch never dead-ends.

Inspect it live:

```bash
python3 models/ksctl backends --gpu h100 --dtype fp8     # per-op chain + choice
python3 -c "import kernel_set; print(kernel_set.dispatch.which('moe', gpu='h100'))"
```

---

## The matrix — optimal provider per compute-bound op × arch × dtype

Legend: **bold** = dispatch rank-1 for that arch (what gets selected when
installed). `→ ks` means "if nothing external is installed for that arch, fall
back to the kernel-set C ABI".

### `gemm` — dense fp16/bf16/tf32 GEMM

| Arch | dtype | Optimal | Dispatch selects | Call |
|---|---|---|---|---|
| A100 / sm80 | bf16/fp16/tf32 | cuBLASLt (torch `@`) | **torch** → ks | adopt-external |
| L4 / sm89 | fp16/bf16 | cuBLASLt (torch) | **torch** → ks | adopt-external |
| H100 / sm90 | bf16/fp16 | cuBLASLt (auto WGMMA) | **torch** → ks | adopt-external |
| B200 / sm100 | bf16 | cuBLASLt (auto tcgen05) | **torch** → ks | adopt-external |

cuBLAS/cuBLASLt is arch-optimal on **every** GPU (it heuristically selects the
WGMMA/tcgen05 path per shape/arch at ~42–48 % of dense peak on L4 and dominates
on A100/H100). CUTLASS only wins when you need a **fused epilogue** — not exposed
as a separate op here. **No rank-1 gap.** kernel-set's own dense GEMM is
0.03–0.10× cuBLAS, so it is the rank-99 correctness fallback only.
**Call: adopt-external.**

### `fp8_gemm` — FP8 (blockwise / scaled) GEMM

| Arch | dtype | Optimal | Dispatch selects | Call |
|---|---|---|---|---|
| A100 / sm80 | — (no FP8 TC) | bf16 GEMM (cuBLAS) | DeepGEMM/torch/sgl all gated → **ks** (bf16-cast) | n/a on Ampere |
| L4 / sm89 | fp8 e4m3 | torch._scaled_mm / TE | **torch-scaled-mm** → ks | hybrid |
| H100 / sm90 | fp8 e4m3 | DeepGEMM (block-scaled) | **deep_gemm** → torch-scaled-mm → sgl → ks | adopt-external |
| B200 / sm100 | fp8 | DeepGEMM / cuBLASLt FP8 | **deep_gemm** → … → ks | adopt-external |

Rank order (DeepGEMM sm90 > torch-scaled-mm sm89 > sgl sm90 > ks) is optimal and
the arch gates are correct. **Gap: arch-gating, not rank-order.** On sm89 the
measured `torch._scaled_mm` failed the accuracy gate with per-tensor scaling
(rel_err ~3.8e-2); the adapter should use per-row/block scales before that path
is trusted in production. **Call: hybrid** — adopt-external on Hopper/Blackwell
(DeepGEMM), fix the sm89 scaled-mm accuracy; a native block-scaled FP8 ks ABI is
low-priority (the archs with FP8 hardware are all covered externally).

### `int8_gemm` (W8A8) — INT8 SmoothQuant GEMM

| Arch | dtype | Optimal | Dispatch selects | Call |
|---|---|---|---|---|
| A100 / sm80 | int8 w8a8 | CUTLASS INT8 (vLLM) | **vllm** → sgl-kernel → ks | adopt-external |
| L4 / sm89 | int8 w8a8 | CUTLASS INT8 (vLLM) | **vllm** → sgl-kernel → ks | adopt-external |
| H100 / sm90 | int8 w8a8 | CUTLASS INT8 (vLLM) | **vllm** → sgl-kernel → ks | adopt-external |
| B200 / sm100 | int8 w8a8 | Marlin-int8 (CUTLASS int8 unsupported) | **vllm-marlin-int8** → vllm → sgl → ks | adopt-external |

**Fixed gap:** the dispatch chain previously wired only sgl-kernel + ks, skipping
the registry's true rank-1 (vLLM CUTLASS int8). Now vLLM CUTLASS int8 is rank-1
(sm80–sm90), sgl-kernel `int8_scaled_mm` is rank-2 (alignment target), and a
**Marlin-int8** entry leads on Blackwell (sm100) where CUTLASS int8 is
unsupported. kernel-set has a native `ks_gemm_w8a8` ABI so the fallback is
correct (just unbenched vs CUTLASS). **Call: adopt-external.**

### `w4a16` — mixed-input INT4×FP16 GEMM

| Arch | dtype | Optimal | Dispatch selects | Call |
|---|---|---|---|---|
| A100 / sm80 | int4 / fp16 acts | GPTQ/AWQ-Marlin (vLLM) | **vllm-marlin** → ks | adopt-external |
| L4 / sm89 | int4 / fp16 acts | Marlin (vLLM) | **vllm-marlin** → ks | adopt-external |
| H100 / sm90a | int4 / fp16 acts | **Machete** (CUTLASS TMA+WGMMA prepack) | **vllm-machete** → vllm-marlin → ks | adopt-external |
| B200 / sm100 | int4 / nvfp4 | NVFP4-Marlin / Marlin | **vllm-marlin** → ks | adopt-external |

**Fixed two gaps:** (1) the rank-1 `vllm-marlin` entry had `call=None` — it could
be *named* but never actually dispatched, so every host silently dropped to the
~3 %-peak ks INT4 kernel. The adapter is now wired (`ops.gptq_marlin_gemm`). (2)
Hopper conflated Marlin and Machete; a distinct **`vllm-machete`** provider
(sm90-gated) now leads on Hopper, with Marlin behind it for Ampere/Ada/Blackwell.
**Call: adopt-external.**

### `attention_prefill` — dense exact attention

| Arch | dtype | Optimal | Dispatch selects | Call |
|---|---|---|---|---|
| A100 / sm80 | fp16/bf16 | FlashAttention-2 | **flash-attn** → sgl → sdpa → flashinfer → ks | adopt-external |
| L4 / sm89 | fp16/bf16 | FlashAttention-2 | **flash-attn** → … → ks | adopt-external |
| H100 / sm90 | fp16/bf16/fp8 | FlashAttention-3 (via flash-attn) | **flash-attn** → sgl-FA3 → … → ks | adopt-external |
| B200 / sm100 | bf16/fp16 | **FlashAttention-4** (CuTe-DSL) | **flash-attn-cute** → flash-attn → … → ks | adopt-external |

flash-attn is rank-1 for sm80/sm89 (FA2) and sm90 (FA3 auto path). **Fixed gap:**
added a dedicated **`flash-attn-cute` (FA4)** provider gated sm100 so Blackwell
routes to FA4 instead of under-routing to FA2/FA3. FlashInfer is the portable
serving-grade alternative (rank-4, b==1). The ks prefill is ~0.01× SOTA (a naive
non-flash kernel) — fallback only. **Call: adopt-external.**

### `attention_decode` — paged-KV decode

| Arch | dtype | Optimal | Dispatch selects | Call |
|---|---|---|---|---|
| A100 / sm80 | fp16/bf16 | FlashInfer paged decode | **flashinfer** → ks | adopt-external |
| L4 / sm89 | fp16/bf16, fp8 KV | FlashInfer (+FP8 KV) | **flashinfer** → ks | adopt-external |
| H100 / sm90 | fp16/bf16, fp8 KV | FlashInfer / sgl FA3 decode | **flashinfer** → sgl → ks | adopt-external |
| B200 / sm100 | fp16/bf16 | FlashInfer trtllm-gen | **flashinfer** → ks | adopt-external |

FlashInfer (split-K load balancing, plan/run for CUDA graphs, FP8 KV) is
best-in-class and most portable on every arch — already rank-1. **No rank-1
gap.** Decode is memory-bound so the ks fallback is "only" ~3× off (~23 % BW vs
FlashInfer ~78 %); acceptable as a safety net. **Call: adopt-external.**

### `mla_decode` — absorbed-MLA paged decode (DeepSeek)

| Arch | dtype | Optimal | Dispatch selects | Call |
|---|---|---|---|---|
| A100 / sm80 | bf16 | **FlashInfer MLA** (only viable SOTA) | **flashinfer** → ks | hybrid |
| L4 / sm89 | bf16 | FlashInfer MLA | **flashinfer** → ks | hybrid |
| H100 / sm90 | bf16, fp8 KV | **FlashMLA** (DeepSeek official) | **sgl-kernel** (FlashMLA) → flashinfer → ks | adopt-external |
| B200 / sm100 | bf16/fp8 | FlashMLA / FlashInfer trtllm-gen MLA | **sgl-kernel** → flashinfer → ks | adopt-external |

**Fixed two gaps:** (1) on pre-Hopper the only non-ks provider was FlashMLA
(sm90-gated) — A100/L4 had nothing but the ks MLA kernel, the single worst
measured op at ~1 % bandwidth. A **FlashInfer MLA** provider (sm80+) is now wired
(rank-2), giving Ampere/Ada a real SOTA path. (2) On Hopper/Blackwell FlashMLA
(via sgl-kernel) stays rank-1. **Call: hybrid** — adopt-external on every arch
(FlashMLA sm90+, FlashInfer MLA sm80+); self-dev only to lift the ks MLA fallback
off its ~1 % floor for hosts with neither library.

### `moe` (grouped GEMM) — Mixture-of-Experts expert FFN

| Arch | dtype | Optimal | Dispatch selects | Call |
|---|---|---|---|---|
| A100 / sm80 | bf16 / int4 | CUTLASS grouped / vLLM fused_moe | **vllm** (fused_experts) → ks | adopt-external |
| L4 / sm89 | bf16 | vLLM Triton fused_moe | **vllm** → ks | adopt-external |
| H100 / sm90 | fp8 e4m3 | **DeepGEMM grouped** (DeepSeek-V3) | **deep_gemm** → sgl (CUTLASS grouped) → vllm → ks | adopt-external |
| B200 / sm100 | fp8 / fp8×fp4 | DeepGEMM Mega-MoE / vLLM MXFP4 | **deep_gemm** → sgl → vllm → ks | adopt-external |

**Fixed three gaps:** (1) DeepGEMM grouped FP8 — the reference MoE GEMM
(DeepSeek-V3 production, Mega-MoE) — was **not wired into the `moe` op at all**;
it is now rank-1, sm90-gated. (2) the sgl-kernel `moe` adapter was a **stub** that
probe-imported `sgl_kernel` then called the *ks* grouped GEMM; it now calls the
real `fp8_blockwise_scaled_grouped_mm` (rank-2, sm90). (3) Ampere/Ada (no FP8 hw)
route to vLLM Triton `fused_experts` (rank-3, sm80). ks grouped GEMM (~1 % peak,
second-worst op) is the portable fallback. **Call: adopt-external.**

### `moe_gate` / `moe_group_gate` — MoE routing gate

| Arch | dtype | Optimal | Dispatch selects | Call |
|---|---|---|---|---|
| all (sm80+) | fp32 routing | **sgl-kernel** fused gate | **sgl-kernel** → vllm → ks | adopt-external |

MoE gating is sgl-kernel's explicit specialty (fused softmax/sigmoid group-topk,
DeepSeek-V3 biased `moe_fused_gate`, up to 256 experts) and is already correctly
rank-1 with a genuine adapter. **Smallest gap:** the vLLM rank-2 entries had
`call=None` — now wired (`ops.topk_softmax` / `grouped_topk`) for coverage. The
gate is tiny/cheap and the ks gate is near-parity (it is one of the few ops where
ks is competitive), so it is a fine fallback. **Call: adopt-external; this op is
essentially done.**

---

## Summary — optimal per op (one line each)

| Op | Rank-1 (Ampere/Ada) | Rank-1 (Hopper) | Rank-1 (Blackwell) | Fallback |
|---|---|---|---|---|
| `gemm` | torch/cuBLASLt | torch/cuBLASLt | torch/cuBLASLt | kernel-set |
| `fp8_gemm` | torch-scaled-mm (sm89) | deep_gemm | deep_gemm | kernel-set |
| `int8_gemm` | vllm CUTLASS int8 | vllm CUTLASS int8 | vllm-marlin-int8 | kernel-set |
| `w4a16` | vllm-marlin | vllm-machete | vllm-marlin | kernel-set |
| `attention_prefill` | flash-attn (FA2) | flash-attn (FA3) | flash-attn-cute (FA4) | kernel-set |
| `attention_decode` | flashinfer | flashinfer | flashinfer | kernel-set |
| `mla_decode` | flashinfer (MLA) | sgl-kernel (FlashMLA) | sgl-kernel (FlashMLA) | kernel-set |
| `moe` | vllm fused_experts | deep_gemm grouped | deep_gemm grouped | kernel-set |
| `moe_gate` / `moe_group_gate` | sgl-kernel | sgl-kernel | sgl-kernel | kernel-set |

For **every** compute-bound op, kernel-set is the **last** chain entry and is
never preferred over an installed, arch/dtype-compatible external provider —
verified by the dispatch tests in
[`tests/test_dispatch.py`](../bindings/python/tests/test_dispatch.py).

---

## Rank changes applied (vs the previous registry)

| Op | Change |
|---|---|
| `int8_gemm` | Added **vllm CUTLASS int8 as rank-1** (was missing from the chain) and **vllm-marlin-int8 rank-0 (sm100)**; sgl-kernel demoted to rank-2. |
| `w4a16` | **Wired the `vllm-marlin` adapter** (was `call=None`, never dispatchable); added a distinct **`vllm-machete` rank-0 (sm90)** Hopper-optimal provider. |
| `moe` | Added **deep_gemm grouped rank-1 (sm90)**; **fixed the sgl-kernel stub** to call the real `fp8_blockwise_scaled_grouped_mm` and re-gated it to sm90 (rank-2); vLLM fused_experts demoted to rank-3 (sm80). |
| `mla_decode` | Added **FlashInfer MLA rank-2 (sm80+)** so Ampere/Ada have a real SOTA path (FlashMLA stays rank-1, sm90). |
| `attention_prefill` | Added **`flash-attn-cute` (FA4) rank-0 (sm100)** so Blackwell routes to FA4. |
| `moe_gate` / `moe_group_gate` | **Wired the vLLM rank-2 adapters** (were `call=None`). |
| `gemm`, `fp8_gemm`, `attention_decode` | No rank change (already optimal); clarified notes only. |

---

## SELF-DEVELOP shortlist

Self-developing a kernel is worth it **only** when one of these holds:

1. **Portable fallback that must exist for non-Python bindings.** The kernel-set
   C ABI is what makes the Rust / Go / TS bindings dependency-free. Every op
   above needs a *correct* (not necessarily fast) ks fallback so those bindings
   work offline. This is the primary self-dev justification and is already in
   place (correctness verified, rel_err small) — keep it maintained.

2. **`mla_decode` ks fallback (highest ROI fix).** It is the single worst op at
   ~1 % bandwidth, and even after wiring FlashInfer MLA (sm80+) a host with
   *neither* FlashMLA nor FlashInfer has no good option. A real tiled
   absorbed-MLA decode would lift the floor materially. Do the cheap adoption
   first (FlashInfer MLA, done), then improve this fallback.

3. **`moe` bf16 grouped-GEMM ks fallback** (~1 % peak, second-worst). A
   non-trivially-better bf16 grouped GEMM is justified for Ampere/Ada hosts
   without vLLM; lower priority than the dispatch wiring (now done).

4. **Native block-scaled FP8 ks ABI** (currently absent — the "fp8" ks path
   upcasts to bf16). Low priority: the archs with FP8 hardware are all covered by
   DeepGEMM (sm90+) / torch._scaled_mm (sm89).

**Not worth self-developing** (adopt external, full stop): dense `gemm`
(cuBLAS), `attention_prefill`/`attention_decode` (FlashAttention / FlashInfer),
`w4a16` (Marlin/Machete), `int8_gemm` (CUTLASS), `moe` hot-path FP8 grouped GEMM
(DeepGEMM), and `moe_gate` (sgl-kernel) — these are multi-quarter, NVIDIA-grade
efforts to merely reach parity, with zero product ROI.

**Keep + improve (kernel-set is already SOTA-class, ~84–87 % peak BW on A100):**
the memory-bound ops — `rmsnorm` / `fused_add_rmsnorm` / `gemma_rmsnorm` / `rope`
/ `swiglu` (activation) / `cross_entropy` (loss) / sampling / optimizer /
elementwise. These are *not* in the compute-bound matrix; kernel-set is a
genuine ranked provider there and the portable C-ABI path for all bindings.
