# kernel-set evolution plan

Date: 2026-06-16

This plan focuses the next evolution of `kernel-set` around measured evidence,
release truthfulness, and high-leverage provider routing. The goal is not to add
more operator names by default; it is to make every public claim traceable to a
checked-in source of truth and to turn the most important heuristic routing cells
into measured cells.

## Current State

- The core generation and binding gates are healthy. The validation chain
  `gen_optimal.py --check`, `gen_baselines.py --check`,
  `models/_gen_registry.py --check`, `check_abi.py`,
  `xcheck_signatures.py`, and `pytest bindings/python/tests/ -q` is the right
  baseline for dispatch/planner work.
- `providers/optimal.json` currently has 533 cells: 4 promoted measured,
  529 heuristic, and 9 observed-not-promoted diagnostic rows. That ratio is the
  central product risk: the selector is broad, but most high-end GPU cells still
  rely on curated expectations rather than promoted measured winners.
- Benchmark reports now have a durable data layer:
  `benchmarks/results/runs/*.json` stores canonical rows keyed by
  `(gpu_sm, dtype, timing_profile, model_id, layer_idx, position_kind, position,
  op, shape)`, with `impl` as the compared dimension. The generated
  `benchmarks/results/index.json`, `benchmarks/results/README.md`, and root
  README block are derived from those runs.
- The checked-in runs now cover L4 sm89, A100 sm80, H20 sm90, and RTX PRO 6000
  Blackwell sm120. The 2026-06-16 Colab refresh adds full `kernel_set` op
  coverage for L4 and RTX PRO 6000 across 12 total shards; the most important
  sm90/sm100/sm120 provider paths still need cleaner measured promotion into
  `optimal.json`; rows with missing providers or shape-sensitive winners should
  stay diagnostic until a shape gate or full provider suite exists.

## External Drift Checked

This was a short freshness pass against official/project sources, not a full
new landscape survey.

- FlashInfer has active 0.6.13 release-candidate/nightly activity after the
  existing 2026-H1 landscape snapshot:
  <https://github.com/flashinfer-ai/flashinfer/releases>
- FlashAttention has a 2026-06-10 v2.8.3.post1 release in the 2.x line while
  FA4 remains the Blackwell direction:
  <https://github.com/Dao-AILab/flash-attention/releases>
- DeepGEMM keeps moving around Mega-MoE, FP4 indexer, and SM90/SM120 gaps:
  <https://github.com/deepseek-ai/DeepGEMM/commits>
  <https://github.com/deepseek-ai/DeepGEMM/issues/317>
- Qwen FlashQLA is a concrete new GDN/linear-attention benchmark target, built
  on TileLang and reporting 2-3x forward speedups over FLA Triton on Hopper:
  <https://github.com/QwenLM/FlashQLA>
  <https://qwen.ai/blog?id=flashqla>
- SageAttention3 remains the Blackwell FP4 attention candidate to track:
  <https://github.com/thu-ml/SageAttention>
- vLLM's DeepSeek V4 path explicitly calls out further work on DeepGEMM MegaMoE
  and paged prefill:
  <https://vllm.ai/blog/2026-04-24-deepseek-v4>

Implication: static docs will drift quickly. The project needs refreshable data
products and small checkers more than another large hand-written survey.

## P0: Make Claims Release-Safe

1. **Keep benchmark data as source of truth.**
   - Default GPU runs should leave `*.md`, raw JSON, canonical run JSON, and a
     refreshed results index.
   - README benchmark tables must be generated, not hand edited.
   - CI should validate canonical runs and `render_results_readme.py --check`.

2. **Add a docs-facts checker before the next release.**
   - Check that docs claiming `optimal.json` stats match the real
     `providers/optimal.json` stats.
   - Check operator/provider/model/ABI counts against their real registries.
   - Treat drift as a release blocker, not a cleanup task.

3. **Close Blackwell compile coverage.**
   - Regular CI currently builds sm80/sm89/sm90. Release builds claim sm75-sm120.
   - Add compile+dlopen smoke for sm100/sm120 once the GitHub CUDA image/toolkit
     is pinned to a version that supports those architectures.
   - Minimum target: headers, ABI, Python dispatch, and NVFP4/FP4 routing paths
     compile and load.

4. **Promote high-impact sm90/sm100/sm120 cells from heuristic to measured.**
   - First wave: attention prefill/decode, MLA decode, fp8/blockwise GEMM, MoE
     grouped GEMM/fused MoE, fused norm, RoPE, SwiGLU, cross entropy.
   - Promotion rule: same GPU, same dtype, same timing profile, representative
     shape range, correctness-gated winner, full production provider coverage,
     with the source run preserved in `benchmarks/results/runs/`. If winner
     flips by shape, add a shape policy instead of a whole-cell override.

## P1: Make Routing Quality Explicit

1. **Add measured-winner ingestion for `gen_optimal.py`.**
   - Today measured winners are embedded in Python constants. Replace or
     supplement that block with a generated `benchmarks/results/measured_winners.json`
     derived from canonical runs.
   - Keep the current invariant: `provider == fallback_chain[0]`, and every chain
     terminates at `kernel-set`.

2. **Classify fallback truthfully.**
   - Add `fallback_status: implemented | stub | external_required` to provider
     and dispatch metadata.
   - Surface it in `ksctl`, docs, and dispatch availability. This prevents
     "portable fallback" from silently covering external-only op families such
     as sparse/DSA attention, BitNet, FP4, and some experimental FP8 paths.

3. **Make model registry production quality visible.**
   - Add `confidence`, `status`, `placeholder_fields`, and `load_bearing_ops`.
   - `ksctl plan` should distinguish production-ready from experimental model
     entries instead of presenting all registry rows equally.

4. **Prioritize from atomic catalogs, not raw operator count.**
   - Next operator/provider work should be ranked by runtime leverage:
     KV-cache IO/quant, sparse/DSA attention, speculative decoding/logit
     processors, EP/MoE communication, and linear-attention/SSM serving kernels.
   - Route/adopt external providers first; self-implement only portable fallback
     gaps or areas with no usable upstream kernel.

## P2: Platform and Documentation Cleanup

1. **ROCm/HIP: milestone or reserved.**
   - If AMD support is a near-term goal, add hipcc compile-only plus a tiny
     dtype/operator smoke. Otherwise, docs should call HIP reserved rather than
     implying equal support.

2. **Move old simplification plans to historical status.**
   - `SIMPLIFICATION_PLAN.md` contains several completed items. Mark it as
     historical or trim it to remaining risks so future roadmap work does not
     reopen already-finished cleanup.

3. **Colab/remote benchmark ergonomics.**
   - Keep `build_and_bench.sh` single-command.
   - Add a SOTA provider install preset for the target GPU tier before running
     `bench_sota.py`; otherwise import-fail counts hide real provider quality.
   - Preserve `KS_RUN_LABEL` so parallel Colab agents produce non-overwriting
     canonical runs.

## Stop Conditions

- Do not promote a measured winner from a Markdown table by hand; use canonical
  JSON rows.
- Do not promote rows from a suite with missing production providers or a known
  shape reversal; keep them as observed-not-promoted diagnostics.
- Do not publish a new best-provider claim unless the source run is checked in
  and `render_results_readme.py --check` passes.
- Do not broaden the operator surface until the existing high-impact heuristic
  cells have a measured path or an explicit "pending measurement" label.
