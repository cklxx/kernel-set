# kernel-set Simplification Plan

Synthesized from five per-subsystem reviews (CUDA kernel layer, benchmark harness,
selection/dispatch, language bindings, and routing/docs). De-duplicated, ranked by
value/risk, with an honest scope verdict.

## Verdict

**The codebase is in good shape and does not warrant a large refactor.** Every
subsystem reviewer independently noted that the existing code is clean, principled,
and works correctly (the Rust bindings, the bench_sota timing reuse, the
probe/registry split are all explicitly called out as well-architected). The
opportunities are almost entirely **de-duplication and consolidation**, not
re-architecture.

The real signal across all five reports is **one recurring theme**: the same facts
are encoded in multiple places with no single source of truth — GPU capability /
SM-threshold tables (3 copies), dtype alias maps (2+ copies), bench shape tables and
reference math (2 copies), CUDA grid-sizing helpers (~8 copies), and FFI
declarations (4 language mirrors). That duplication is a genuine maintenance and
consistency risk (adding a new GPU arch today means editing 3+ tables), and it is
worth fixing incrementally.

What is **not** worth doing now (over-engineering):

- A unified `unified_registry.json` merging models + providers + baselines
  (high risk, 500+ LOC churn, touches generators that work).
- A code-generation pipeline for all 72 FFI signatures × 4 languages (650–900 LOC
  of generator machinery to replace working, compile-verified hand-written code;
  the savings are real but the generator becomes a new thing to maintain and the
  Rust reviewer explicitly recommends leaving it hand-written).
- A grand "OpRegistry/selection_core" unification of the two selection engines.
  They have genuinely different semantics (model→ABI-symbol vs op→provider-chain);
  forcing them together adds an abstraction without removing real duplication.

Recommended posture: **do the safe, Python-side, independently-verifiable
consolidations now** (shared bench modules, shared GPU/dtype tables, the missing
baselines generator, docs merge). **Defer** anything touching CUDA kernels or
build/codegen until it can be GPU-verified, and treat the big registry/codegen ideas
as explicitly out of scope unless a concrete pain point forces them.

---

## Do Now (safe, high-value, Python/docs-side, independently verifiable)

### 1. Extract shared bench module: shapes + reference math + correctness gate
`bench.py` and `bench_sota.py` independently define identical shape tables
(`_NORM_SHAPES`, `_SWIGLU_SHAPES`, `_ROPE_SHAPES`, `_ATTN_*_SHAPES`, `_GEMM_SHAPES`,
`_MOE_SHAPES`, `_CE_SHAPES`), an identical `_ref_rope_neox`, near-identical attention
references, and two copies of the correctness gate (`gate_correctness` vs `_gate`).
`bench_sota` already imports timing utilities from `bench`, so a shared module is the
established pattern.

**Action:** create `benchmarks/_bench_common.py` (or `shapes.py` + `references.py`)
exporting the shape tables, reference implementations, and a single
`gate_correctness`. Import from both harnesses. **Verify:** `python -c "import
bench, bench_sota"` and a `--list`/dry-run of each harness on CPU (no GPU needed for
import + shape enumeration).
Risk: low. Saves ~80 LOC and removes the correctness-gate divergence risk.

### 2. Add the missing `scripts/gen_baselines.py` generator
`benchmarks/baselines.yaml` (131 KB) is explicitly marked "AUTO-GENERATED from
providers/registry.json" but **no generation script exists in the repo** (`scripts/`
only contains `check_abi.py`). The file is a hand-maintainable copy with no
regeneration path — exactly the consistency trap the header warns against.

**Action:** write `scripts/gen_baselines.py` that reads `providers/registry.json`,
filters to rank-1 providers per op, and emits the YAML. Document regeneration in
`benchmarks/README.md`. **Verify:** run it and diff against the checked-in
`baselines.yaml`; reconcile any drift (the diff itself is a useful audit).
Keep the file checked in (it is consumed by install scripts) but make it
regenerable. Risk: low (read-only over JSON; output diffable before adopting).

### 3. Single canonical GPU capability + SM-threshold table
GPU facts live in 3 places: `models/select.py` (`GPUS`, `GPU_ALIASES`),
`bindings/python/kernel_set/backends/_probe.py` (`GPU_SM`), and `baselines.yaml`
comments — plus 7+ hardcoded `sm >= 90` / `sm >= 80` / `sm >= 89` conditionals
scattered across `select.py`, `_probe.py`, `_registry.py`. Adding Blackwell required
edits in several files.

**Action:** create one `models/gpu_registry.py` (or `gpu_capabilities.json` + a tiny
loader) holding `{id, sm, aliases, caps}` and a `SM_THRESHOLDS` constant block
(FP8=89, BF16=80, TF32=80, etc.). Import it from `select.py` and `_probe.py`;
replace the magic-number conditionals with named thresholds. **Verify:** unit-check
that `select.py` and `_probe.py` resolve the same `sm`/caps for a sample of GPU names
(pure Python, no GPU). Risk: low — pure data consolidation. This is the single
highest-leverage consistency fix.

### 4. Single canonical dtype alias + support module
`select.py` (`DTYPE_NORMALIZE`, `KS_DTYPE`) and `_probe.py` (`_DTYPE_ALIASES`,
`normalize_dtype`, `dtype_ok`) maintain divergent alias sets and divergent matching
(explicit caps checks vs substring matching on provider text). Drift here silently
breaks dispatch (e.g. provider text "fp8"→"FP8" defeats the substring match).

**Action:** create `models/dtype_support.py` with the merged authoritative alias map,
`normalize()`, `check_device_support(dtype, sm)` (using `SM_THRESHOLDS` from #3), and
`check_provider_support(dtype, provider_str)`. Also move the `KS_DTYPE` enum-string
map next to the existing `bindings/python/kernel_set/enums.py` and extract the ad-hoc
activation-dtype rule in `select.py:251` into `activation_dtype(scheme, has_bf16)`.
Import from both engines. **Verify:** parametrized assertions that both old and new
paths agree on a table of `(dtype, sm)` inputs. Risk: low.

### 5. Merge the two overlapping kernel-landscape docs
`docs/KERNEL_LANDSCAPE.md` (evergreen best-in-class survey) and
`docs/KERNEL_LANDSCAPE_2026.md` (2026-H1 release notes) conceptually overlap;
readers cannot tell which is authoritative. `OPERATOR_CATALOG.md` remains the
generated exhaustive reference.

**Action:** merge the two landscape docs into one living doc with "Current
best-in-class (as of <date>)" and "Recent updates" sections; link out to
`OPERATOR_CATALOG.md` for the full ranked catalog. Pure docs edit. Risk: low.

### 6. Document the three-tier routing story + label module roles
Selection logic is spread across `models/select.py` (model→ABI symbol via
`models/registry.json`), `bindings/python/kernel_set/dispatch.py` (op→provider via
`providers/registry.json`), and the `backends/_probe.py`+`_registry.py` split, with
docs describing only one of them. No re-architecture needed — just a map.

**Action:** add a short `docs/ROUTING.md` (1–2 pages, flow diagram) and an
ARCHITECTURE.md section naming the tiers (Tier 1 model selection, Tier 2 backend
dispatch, Tier 3 ksctl CLI). Add module-level docstrings to `backends/__init__.py`,
`_probe.py`, `_registry.py` stating each role and that `_registry.py` is "derived
from providers/registry.json". Docs/comments only. Risk: low.

---

## Defer / GPU-verify (touches kernels or build — do not do blind)

### 7. Consolidate CUDA grid-sizing helpers into `common/`
`grid_for`, `grid_for_int4`, `grid_for_fp8`, `cast_grid`, plus inline
`(work+block-1)/block` clamped to 65535 are reimplemented ~8 times across
`elementwise/cast.cu`, `quant/*.cu`, `embedding/embedding.cu`,
`activation/activation_ops.cuh`, `optimizer/*.cu`.

**Action:** add a single `KS_GRID_FOR(work, block)` inline/macro to
`common/platform.cuh`; replace the reimplementations. High value (single source of
truth for a correctness-relevant clamp), low conceptual risk, but **requires a GPU
build + test pass** to confirm no behavioral change. `needs_gpu_verify`.

### 8. Add reusable launch / validation / grid-limit macros
Closely related kernel-side patterns the reviewers flagged together: collapse the
repeated `dim3 + KS_DISPATCH + KS_CHECK_LAUNCH` boilerplate (28 files) into a
`KS_LAUNCH_KERNEL(...)` macro; add `KS_CHECK_PTRS_N` / `KS_CHECK_SHAPE_2D` validation
macros; add a `KS_CHECK_GRID_LIMIT` helper (rope.cu / attention inline guards). Also
unify the vectorized load/store + grid-stride tail pattern
(`elementwise_common.cuh`, `activation_ops.cuh`, `gated_mlp.cu`) into a
`common/vec.cuh` template, and unify the ~13 block-size constants
(`kBlock=256`, `kGateBlock=128`) into `platform.cuh`.

**Action:** do these as one coordinated kernel-cleanup pass (they overlap and all
touch the same files). High aggregate LOC savings (~400) but **medium risk and
mandatory GPU verification** — macro/template changes to launch paths can subtly
change occupancy or alignment behavior. Defer until a GPU is available; land
incrementally with benchmarks before/after. `needs_gpu_verify`.

---

## Out of Scope (do not do now — over-engineering / high risk for working code)

### 9. Bench `OpBench`/`ProviderBench` measurement-pattern abstraction & provider registry
The per-op setup→ref→run→gate→time→metrics pattern repeats ~18× in `bench.py` and
per-provider in `bench_sota.py`, and `bench_sota`'s ~20 provider adapters are inline.
A class-based `OpBench` + `ProviderAdapter` registry would save ~300 LOC.

**Why deferred:** medium risk, and the value is real but the harnesses work and are
actively used. The shared-module extraction (#1) captures most of the consistency
benefit at a fraction of the risk. Revisit only if adding providers/ops becomes a
frequent, painful operation. Also encompasses the `timing_common.py` extraction and
the report/formatting consolidation — nice-to-haves, not needed.

### 10. Unified selection core merging both engines
`models/select.py` (model→single ABI symbol + rationale) and `dispatch.py`
(op→ranked provider chain) implement *different* selection semantics. A
`selection_core.Selection.select()` could unify them, but they don't actually share
a chain abstraction today.

**Why deferred:** medium risk, only ~100 LOC saved, and it risks coupling two systems
that legitimately differ. The capability/dtype unification (#3, #4) removes the
*real* duplication (the GPU/dtype facts) without forcing the algorithms together.

### 11. Unified `unified_registry.json` + FFI / enum / struct codegen across bindings
Merging models/providers/baselines into one cross-referenced registry (~500 LOC
churn, high risk, touches generators) and code-generating all FFI signatures / enums
/ device-properties struct across Python/Rust/Go/TS (650–900 LOC of *new generator*
machinery).

**Why out of scope:** highest risk in the whole set, and it replaces working,
compile-verified, well-documented hand-written bindings (the Rust reviewer
explicitly says leave them as-is) with a generator that itself must be maintained.
The library-discovery and error-handling consolidations are likewise per-language
idiomatic and already "decently factored" — documenting the shared contract in a
`bindings/*.md` spec is the most that is warranted, and even that is optional.

---

## Summary table

| # | Change | Risk | Value | Do now | GPU verify |
|---|--------|------|-------|--------|------------|
| 1 | Shared bench module (shapes/refs/gate) | low | high | yes | no |
| 2 | `scripts/gen_baselines.py` generator | low | high | yes | no |
| 3 | Canonical GPU capability + SM-threshold table | low | high | yes | no |
| 4 | Canonical dtype alias + support module | low | high | yes | no |
| 5 | Merge two kernel-landscape docs | low | high | yes | no |
| 6 | ROUTING.md + module-role docstrings | low | medium | yes | no |
| 7 | Consolidate CUDA grid-sizing helpers | low | high | no | yes |
| 8 | Kernel launch/validation/vec macros | medium | high | no | yes |
| 9 | Bench OpBench/provider-registry abstraction | medium | medium | no | no |
| 10 | Unified selection core | medium | medium | no | no |
| 11 | Unified registry + FFI/enum codegen | high | high* | no | partial |

\*High raw LOC savings but low net value once generator maintenance is accounted for.
