# GPU / LLM Kernel Benchmark Methodology

This document is the source of truth for how kernels in this repository should be
benchmarked, and a graded audit of the current harness at
`benchmarks/bench.py` (with `benchmarks/build_and_bench.sh` and
`benchmarks/README.md`).

It has four parts:

1. [The canonical benchmarking best-practice checklist](#1-canonical-best-practice-checklist) — the rules every GPU/LLM kernel benchmark should follow.
2. [The graded audit of `bench.py`](#2-graded-audit-of-benchmarksbenchpy) — what the current harness does and does not do.
3. [Overall verdict](#3-overall-verdict) — does `bench.py` meet industry best practice?
4. [Prioritized fix list](#4-prioritized-fix-list-criticalhigh-first) — concrete code changes, critical/high first.

Status legend: ✅ pass · ⚠️ partial · ❌ fail · ➖ not applicable

---

## 1. Canonical best-practice checklist

The checklist distilled from Triton (`triton.testing.do_bench`), PyTorch
(`torch.utils.benchmark`), NVIDIA (nvbench, Nsight Compute, CUDA best-practices),
CUTLASS, FlashAttention, KernelBench, TritonBench, and vLLM.

| id | name | category | severity | why (one line) | how-to (short) |
|----|------|----------|----------|----------------|----------------|
| BP-01 | Time on the GPU with CUDA events, never host wall-clock around an async launch | timing | critical | Launches are async; a host timer measures queue latency, not device time, often reporting impossibly-fast kernels. | Per-rep `torch.cuda.Event(enable_timing=True)` pairs around `fn()`; or `triton.testing.do_bench`. Wall-clock only for end-to-end serving latency. |
| BP-02 | Synchronize ONCE after the measured loop, then read `elapsed_time` per pair | timing | critical | Syncing inside the loop serializes host/GPU and perturbs clocks; no sync at all measures only launch overhead. | Enqueue all reps with event pairs, then one `torch.cuda.synchronize()`, then read `elapsed_time`. |
| BP-03 | Run a real warmup to absorb one-time costs (JIT/autotune, lazy load, cuBLAS algo pick, allocator/context init, clock ramp) | baseline | critical | First runs pay one-time costs (2775us vs 22us; 17.4ms cold vs 2.0ms warm); short warmup underestimated perf 30% (Triton #2306). | `cudaFree(0)`; ≥10–50 warmup iters of the exact op + sync, discarded; budget form `n=max(1,int(25ms/est))`. `CUDA_MODULE_LOADING=EAGER`. |
| BP-04 | Flush L2 between every measured iteration (or rotate input buffers) | caching | critical | Back-to-back same-input runs hit L2, reporting cache bandwidth (multiples of HBM), not DRAM. | `get_empty_cache_for_benchmark()`+`clear_cache()`, or `cache=torch.empty(256e6//4,int32)`; `zero_()` before each `fn()` (and in estimate/warmup). |
| BP-05 | Lock graphics AND memory clocks, persistence mode, sustainable (not max-boost) freq | clocks | critical | Dynamic boost/throttle is the largest variance source (10–30%), defeating careful statistics. | `nvidia-smi -pm 1`; `--lock-gpu-clocks` AND `--lock-memory-clocks` to a mid value from `-q -d SUPPORTED_CLOCKS`; verify no throttle; reset after. |
| BP-06 | Verify correctness vs the BEST reference (multiple inputs, dtype tolerance) BEFORE trusting timing | correctness | critical | A fast wrong kernel is worthless; a buggy short-circuit also undercounts bytes/FLOPs, double-inflating metrics. | Run kernel + trusted ref on ≥5 random inputs; `torch.testing.assert_close` at dtype tol (fp16/bf16 ~1e-2); only benchmark passers; report tol. |
| BP-07 | Compare against the strongest baseline(s): torch.compile, best library backends | baseline | critical | A speedup over a strawman is marketing; Inductor/flash/cuBLAS often close most of the gap. | Columns for eager, `torch.compile(max-autotune)`, each SDPA backend, FA-2/3, xformers, cuBLAS; headline vs the fastest and name it. |
| BP-08 | dtype / TF32 / accumulation-precision fairness across all compared impls | correctness | critical | TF32 is ~8x faster at 10-bit mantissa and PyTorch defaults vary; mismatched math is an unfair comparison. | Pin `matmul.allow_tf32`, `cudnn.allow_tf32`, `set_float32_matmul_precision` identically; one row per dtype; match accumulation; state settings. |
| BP-09 | Use CUDA graphs (or many back-to-back launches / divide) for tiny/many kernels | timing | critical | ~4–6us launch overhead dominates sub-100us kernels; single-launch event granularity is poor. | `torch.cuda.graph` capture N invocations, replay, divide by N (`do_bench_cudagraph`); or loop-and-divide; state if launch overhead is included. |
| BP-10 | Prevent dead-code elimination: outputs must be consumed/materialized | correctness | critical | Unobserved outputs get pruned (nvlink 11.5+); you benchmark a no-op (>100% peak / flat-vs-size is the smell). | Return/store/accumulate into a volatile sink; check SASS; assert duration scales with size; sanity-guard >100% of peak. |
| BP-11 | Reset accumulating grads to None each rep (`grad_to_none`), outside the timed window | correctness | high | `.grad` accumulates by default, so each backward does growing work; reset cost must be excluded. | Before recording start: `for x in grad_to_none: x.grad=None`. Placed before `start.record()`. |
| BP-12 | Auto-calibrate iteration/sample count from a budget or noise threshold | timing | high | A fixed count over/under-measures: a 5us and a 5ms kernel need wildly different reps for a stable median. | `n=max(1,int(rep_ms/est))` (rep≈100ms); or `blocked_autorange`/`adaptive_autorange`; or nvbench stdrel; print the count used. |
| BP-13 | Report robust distribution stats (median + spread/percentiles), not just mean | statistics | high | GPU timings are right-skewed; mean is outlier-pulled; min≈best, max/percentiles expose tails. | Collect full per-iter array (≥30–100 reps); headline median + min + p10/p90 or IQR; `do_bench(quantiles=[0.5,0.2,0.8])`; flag throttled samples. |
| BP-14 | Count FLOPs/bytes with correct op-specific formulas | metrics | high | Derived throughput is only as good as the counts: GEMM `2MNK`, attn `4·B·H·S²·d`, ~0.5 causal, bwd 2.5x, all read+write bytes. | GEMM `2MNK`; attn fwd `4BHS²d`×0.5 causal, bwd 2.5x; bytes = sum(elem·numel) over every read+write; decode KV via `n_kv_heads`. |
| BP-15 | Report TFLOP/s and GB/s as % of DENSE peak for the exact SKU+dtype | metrics | high | Absolute numbers are uninterpretable; datasheet sparsity (2x) and wrong SKU (SXM/PCIe/NVL ±30%) distort utilization. | `util=achieved/peak_dense`; hardcode a verified dense-peak table; assert SKU+dtype; print numerator, denominator, %. |
| BP-16 | Classify each op via roofline (AI vs device ridge) and report the metric that matters | metrics | high | Compute-bound iff AI = FLOPs/bytes > ridge = peak_FLOPs/peak_BW; otherwise % peak FLOPs is unreachable. | Compute AI and ridge; AI>ridge → % peak TFLOP/s else % peak GB/s; matmul AI≈batch B vs ridge; validate with ncu roofline. |
| BP-17 | Profile with Nsight Compute for true hardware metrics | metrics | high | Wall-clock says "how fast" not "why"; hand-counts miss spills/recompute/uncoalesced traffic. | `ncu --set full` or `--section SpeedOfLight,MemoryWorkloadAnalysis,Occupancy`; reconcile `dram__bytes.sum` with formula; surface occupancy/SoL. |
| BP-18 | Fully report environment and methodology for reproducibility | reporting | high | The same kernel differs across SKU, CUDA/cuDNN, clocks, power cap, cache policy, byte convention — each a large factor. | Emit metadata: GPU+SMs+mem, driver, CUDA/cuDNN, library versions, TF32, locked+achieved clocks, power cap, ECC, flush policy, method, commit hash. |
| BP-19 | Sweep realistic shapes (and fwd/bwd separately for trainable kernels) | reporting | high | Perf is shape-dependent (tile/wave quantization); one cherry-picked size misrepresents; bwd differs from fwd. | Shape matrix incl. non-power-of-2/odd seqlens + target shapes; perf_report table; report fwd-only, bwd-only, fwd+bwd with right multipliers. |
| BP-20 | Reach thermal steady-state, control power cap, randomize/cooldown between cases | clocks | high | Back-to-back heavy kernels heat the die so later cases throttle; fixed ordering biases by position; power cap drops clocks. | Cooldown to baseline temp between cases; interleave variants; discard first N reps; set `-pl <watts>`; drop throttled samples; run idle GPU. |
| BP-21 | For GQA/MQA decode, count KV bytes by `n_kv_heads`; decode is KV-read bound | metrics | high | Decode reads the whole KV cache with O(1) AI; GQA shares K/V so traffic scales with `n_kv_heads` (4–8x fewer). | `kv_bytes = 2·elem·B·n_kv_heads·S·head_dim`; FLOPs still use `n_query_heads`; report % peak BW. |
| BP-22 | Know that L2 is NOT flushed during CUDA-graph replay | pitfall | high | `do_bench_cudagraph` does not `clear_cache` between replays, so graph timings are warm-L2 and not comparable to flushed-eager. | Label each number with its method; use `do_bench` for cold-cache, `do_bench_cudagraph` for launch-overhead removal; never mix as equivalent. |
| BP-23 | Account for wave quantization / tail effect when choosing grid sizes | pitfall | high | CTAs run in waves of (#SMs × CTAs/SM); a partial last wave ~doubles time (109 vs 108 CTAs on 108 SMs). | Wave = `#SMs × CTAs/SM`; sweep around boundaries + non-power-of-2; prefer 100s–1000s of waves; `__launch_bounds__`/persistent kernel. |
| BP-24 | Isolated idle GPU; one process per device; matched input layout/contiguity | pitfall | medium | Co-located jobs/MPS contend for SMs/BW; a contiguous kernel vs strided baseline is an unfair layout comparison. | `CUDA_VISIBLE_DEVICES` one device; confirm no other compute via `nvidia-smi`; feed native layout, include/exclude packing for both; document layout. |
| BP-25 | `CUDA_LAUNCH_BLOCKING=1` and profiler instrumentation only for debugging, never timing | pitfall | medium | Blocking launches destroy async overlap; profiler overhead corrupts timing. | Set blocking only to diagnose errors; unset for perf; time without ncu attached. |
| BP-26 | Saturate the GPU for tiny kernels so the CPU doesn't outrun it (and the fast/thorough flush knob) | pitfall | medium | For light kernels the CPU outruns the GPU and you measure launch latency; the L2 flush itself costs time per rep. | `torch.cuda._sleep(cycles)` to keep GPU busy; flush buffer `int32` (fast) vs `int8` (thorough), `zero_()` per rep, bytes ≥ L2. |
| BP-27 | Prefer `torch.utils.benchmark.Timer` over raw `timeit`; use `Compare` for tables | reporting | medium | Raw `timeit` gives wrong CUDA numbers (no sync, no warmup, total not per-run); `Compare` gives sig-figs/colorize/warnings. | `Timer(stmt=..., globals=...).blocked_autorange(min_run_time=0.2)`; `Compare(results).trim_significant_figures().colorize()`; declare `num_threads`. |
| BP-28 | Use a joint correctness-AND-speedup aggregate (`fast_p`) when summarizing many kernels | reporting | medium | Mean speedup is gamed by correct-but-slow or fast-but-wrong entries. | `fast_p(p) = mean[ is_correct AND (t_base/t_gen > p) ]` for p∈{0,1,2}; plus mean speedup over correct-only. |
| BP-29 | Attention conventions: exclude softmax FLOPs from headline, but count its bytes for non-fused IO | pitfall | medium | softmax/exp runs on CUDA cores not Tensor Cores (inflates TC util) but the S×S scores dominate HBM for non-fused attention. | TFLOP/s numerator = matmul only (`4BHS²d`×0.5 causal); for non-fused baselines count O(N²) score/prob HBM traffic; note softmax FLOPs separately. |
| BP-30 | Use Callgrind instruction counts for deterministic A/B micro-comparisons | metrics | low | Wall-clock noise hides small real changes; instruction counts are deterministic (e.g. 13,693-instruction delta invisible to timing). | `Timer(...).collect_callgrind().counts(denoise=True)`; `as_standardized().delta(...)`. CPU-side dispatch only, not GPU kernel time. |

---

## 2. Graded audit of `benchmarks/bench.py`

Verified against `benchmarks/bench.py`, `benchmarks/build_and_bench.sh`, and
`benchmarks/README.md` (line numbers are from `bench.py` unless noted).

| id | item | status | severity | evidence | fix |
|----|------|:------:|----------|----------|-----|
| BP-01 | GPU CUDA-event timing | ✅ | critical | `time_cuda` (242–271) uses per-iter `torch.cuda.Event(enable_timing=True)` (253–254), records around `fn()` (256–258), reads `elapsed_time` after one sync (259–260). The `perf_counter` path (263–271) is dead — `main()` hard-exits at 1085–1089 if torch+CUDA is missing. | None. Optionally delete the dead `perf_counter` fallback (263–271) and the misleading docstring claim (13–15). |
| BP-02 | Sync once after the loop | ✅ | critical | Loop (255–258) enqueues `starts[i].record(); fn(); ends[i].record()` with no inner sync, one `torch.cuda.synchronize()` after (259), then reads `times_ms` (260). Warmup sync (252) is correctly before. Textbook-correct. | None. |
| BP-03 | Real warmup | ⚠️ | high | A genuine warmup exists: `max(1,warmup)` iters of `fn` then sync (250–252), default `warmup=10` (1039); each bench also calls the kernel once for its correctness check (e.g. 373, 398), incidentally warming it. Weakness: fixed count, not budget-derived, so sub-100us decode kernels may stay under clock-ramp; no explicit JIT/autotune trigger; no `CUDA_MODULE_LOADING=EAGER`. | Budget-derive: `n_warmup=max(warmup,int(25e-3/est_ms))`; trigger autotune/JIT once; set `CUDA_MODULE_LOADING=EAGER` in `build_and_bench.sh`. |
| BP-04 | L2 flush between iterations | ❌ | critical | Zero matches for `clear_cache`/`empty_cache`/`flush`/`l2` in either file. `time_cuda` re-runs `fn()` on the same inputs (255–258) with no flush/rotation. Memory-bound ops (rmsnorm, layernorm, swiglu, rope, paged decode, sampling, CE, adamw) stay L2-resident → reported GB/s is L2, not HBM. rmsnorm rows=1,hidden=4096 (348) is ~8–32KB, fully L2-resident. | Allocate a ≥L2 scratch (256MB int32), `zero_()` inside the timed loop (and warmup/estimate) before each `fn()`; or rotate N>L2-sized input buffers; or adopt `triton.testing.do_bench`. |
| BP-05 | Lock clocks + persistence | ❌ | critical | `build_and_bench.sh` never calls `nvidia-smi -pm/-lgc/-lmc/-pl` (grep finds nothing). `supports_tf32`/clocks are detected only for device naming. The `time_cuda` docstring (246) takes a median to be "robust to clock/boost jitter" — acknowledging unlocked clocks rather than fixing them. Largest variance source. | `nvidia-smi -pm 1`; pick sustainable clocks from `-q -d SUPPORTED_CLOCKS`; `--lock-gpu-clocks` + `--lock-memory-clocks`; reset after; record locked+achieved clocks and throttle reasons in cfg. |
| BP-06 | Correctness gates timing | ⚠️ | critical | Most ops compute `rel_err` vs a torch ref (rmsnorm 375, layernorm 400, swiglu 441, rope 496, attention 545, gemm 629, w8a8 665, moe_gate 741, sampling 802–803, CE 843, adamw 897). Problems: (a) one random input set, not multiple; (b) NO tolerance gate — timing runs before `rel_err` is computed (ks_us at 362/392/… BEFORE 375/400/…), so a wrong kernel still gets a "fast" headline; (c) w4a16 (701) and moe_grouped_gemm (769) have NO reference at all. | Run correctness on ≥5 inputs with `torch.testing.assert_close` at dtype tol BEFORE timing; skip/`status='error'` on failure; add refs for w4a16 (dequant-then-matmul) and grouped-gemm (per-expert torch GEMM over offsets). |
| BP-07 | Strongest baseline | ❌ | critical | Every reference is eager PyTorch: rmsnorm/layernorm/swiglu use plain `F.*`/python (368–370, 396–397, 438); gemm baseline is `a@b` (630); attention uses `F.scaled_dot_product_attention` (562) but does NOT pin/compare flash vs mem-efficient vs cudnn AND runs SDPA in fp32 (555–557), forcing the math backend. No `torch.compile(max-autotune)`, FA-2/3, xformers, or cuBLAS-vs-CUTLASS column. | Add baseline columns: `torch.compile(mode='max-autotune')`; pin each SDPA backend in the kernel's dtype; cuBLAS `a@b` at matching dtype; headline vs the FASTEST and label it. |
| BP-08 | dtype/TF32 fairness | ❌ | critical | No `matmul.allow_tf32`/`cudnn.allow_tf32`/`set_float32_matmul_precision` anywhere (only the `GpuInfo.supports_tf32` field). GEMM timed ref is dtype-fair (`a@b` at test dtype, 630; the fp32 cast at 626 is correctness-only) but TF32 is left at library default (undocumented). Real unfairness: `_ref_sdpa` casts q/k/v to fp32 (555–557) and times SDPA in fp32 (547) against an fp16 kernel — precision-mismatched, speedup-inflating. | Pin TF32 flags identically at startup and record them; time the SDPA ref in the kernel's dtype (remove `.float()` from the timed path; keep fp32 only for the correctness baseline); state dtype+TF32 per row. |
| BP-09 | CUDA graphs / launch amortization | ❌ | critical | No `cudagraph`/`cuda.graph` anywhere. `time_cuda` records one event pair around a single `fn()` per iter (256–258); no capture/replay, no divide-by-N. The suite explicitly benchmarks launch-bound shapes: rmsnorm rows=1 (348), swiglu rows=1 (420), rope tokens=1 (457), paged decode, sampling — reported us is launch latency, not kernel time. | Add a `do_bench_cudagraph`-style path (capture N, replay, divide by N) for tiny/decode shapes, or loop-many-and-divide; state per number whether launch overhead is included. |
| BP-10 | Prevent DCE | ⚠️ | critical | Kernel outputs go to persistent tensors read by the correctness check, so work is materialized for most ops. But the timed eager REFS discard their result each iter: `ref_rms(x)` (376), `silu(gate)*up` (443), `a@b` (630), `logits.argmax(-1)` (804) — eager won't DCE the launch, so impact is limited but unguarded. w4a16 (690–702) and moe_grouped_gemm (759–770) have no correctness read; only the in-loop write keeps them alive. No >100%-of-peak guard. | Add a >100%-of-peak DCE/caching assertion; have each timed ref consume its output (`.sum()`/sink); read w4a16/grouped-gemm output once; confirm duration scales with size. |
| BP-11 | grad_to_none for backward | ⚠️ | high | No `grad_to_none` anywhere. cross_entropy writes grad into a preallocated buffer (827, run 830), not into `.grad`, and adamw overwrites in place — so no `.grad` accumulation bug exists today, but there is no safeguard if a real `.backward()` op were added. | If any op ever uses autograd `.backward()`, set `.grad=None` outside the timed window each rep; document that CE/adamw use explicit buffers. |
| BP-12 | Auto-calibrated iterations | ❌ | high | `iters` fixed at 50 (1040), `warmup` fixed at 10 (1039), passed straight to `time_cuda`; no estimate loop, no `rep_ms` budget, no `blocked_autorange`, no stdev stopping. A ~5us decode kernel and a multi-ms 8192³ GEMM both get exactly 50 iters → fast kernels are jitter-dominated. | Estimate one-iter cost, then `n_repeat=max(1,int(rep_ms/est_ms))` (rep≈100); or `blocked_autorange(min_run_time=0.2)`; print the count used per case. |
| BP-13 | Distribution statistics | ⚠️ | high | `time_cuda` returns ONLY `statistics.median(times_ms)` (261) — no min/p10/p90/IQR/max. `Result` has single scalar `ks_us`/`ref_us` (282–283). README (17) and md header (991) advertise median only. Median is robust, but with unlocked clocks (BP-05) and no spread, the reader cannot see throttle blips. | Collect the full per-iter array; report median PLUS min and p10/p90 (or IQR); add columns to `Result` and the table; `do_bench(quantiles=[0.5,0.2,0.8])`; flag throttled samples. |
| BP-14 | FLOP/byte formulas | ⚠️ | high | Correct: gemm/w8a8/grouped-gemm `2MNK` (623, 656, 767); attn prefill `4·b·qh·seq²·hd·0.5` (537, correct causal 0.5, matmul-only); decode kv_bytes uses kvh not qh (589); swiglu `3·rows·inter` (435, read gate+up, write out — correct). Minor errors: rope bytes (488) omit reading cos/sin (`tokens·hd`); CE bytes (835) and adamw use a single `dtype_bytes(dt)` but CE losses are fp32 (826) and adamw states are fp32. | Add cos/sin reads to rope bytes; count each tensor's actual elem size (fp32 for CE losses / adamw states) instead of one `dtype_bytes(dt)`; document each formula. |
| BP-15 | % of dense peak | ⚠️ | high | `_GPU_PEAKS` (110–120) hardcodes dense bf16/fp16 TC TFLOP/s + BW and the md header prints them as marketing peaks (978–981), but the harness NEVER computes achieved/peak %. a100 bw=1935 is ambiguous (80GB ~2039 vs 40GB ~1555); h100 uses SXM but the slug can't distinguish SXM/PCIe/NVL. Reader must divide by hand. | Compute and emit `compute_util=tflops/peak_tf` and `bw_util=gbps/peak_bw` as % columns per row (numerator, denominator, SKU/dtype); disambiguate SXM/PCIe/NVL and 40/80GB. |
| BP-16 | Roofline classification | ⚠️ | high | Classification is hardcoded by convention, not computed: bandwidth-bound ops report gbps, compute-bound report tflops (matches README 19–20). Reasonable static split, but no AI=FLOPs/bytes and no ridge comparison, so a tiny/decode-like GEMM that is actually memory/launch-bound still gets a TFLOP/s headline. | Compute `AI=FLOPs/bytes` and `ridge=peak_FLOPs/peak_GBps`; pick the reported metric per case from AI vs ridge; print AI + classification. |
| BP-17 | Nsight Compute cross-check | ❌ | high | No `ncu`/`nsight`/`dram__`/`occupancy`/`SpeedOfLight` in either file. All byte/FLOP numbers are hand-derived with no hardware cross-check, so spills/recompute/uncoalesced traffic are invisible (compounded by the L2-flush gap, BP-04). | Add an optional ncu pass (`--section SpeedOfLight,MemoryWorkloadAnalysis,Occupancy --kernel-name <regex>`) and reconcile `dram__bytes.sum` with the analytic counts; at minimum document it as a recommended manual step. |
| BP-18 | Reproducibility metadata | ⚠️ | high | cfg (1106–1114) and md header (973–993) record GPU name/SMs/mem, detection source, dtype, warmup, iters, ks_version, backend, torch_version, host. MISSING: driver, CUDA toolkit/cuDNN/cuBLAS versions, TF32 setting, locked+achieved clocks, power cap, ECC, cache-flush policy, git commit hash, per-row % of peak, whether launch overhead/transfers are included. | Extend cfg: driver version, `nvcc --version`, `torch.version.cuda/cudnn`, TF32 flags, locked/achieved clocks + power cap + ECC, cache-flush policy, "timing includes launch overhead: yes", git commit hash, per-row % of peak. Emit in md + json. |
| BP-19 | Shape sweep + fwd/bwd | ⚠️ | high | Good coverage: norm/swiglu/rope/attention/gemm/moe/sampling/CE/adamw each have multiple LLM-derived shapes incl. decode (rows=1/tokens=1) and prefill with GQA head counts. But ALL shapes are powers-of-two / aligned (4096/8192/2048/14336/28672) — no odd seqlens, no tile remainders, so wave/tile quantization is untested. For trainable ops only CE times a fused fwd+bwd (849); no separate fwd/bwd; attention and gemm have no backward. | Add non-power-of-2 / odd seqlen / near-wave-boundary shapes; add fwd-only and bwd-only (and end-to-end) rows for attention and gemm with correct multipliers (bwd ~2.5x attn, ~2x gemm); split CE fwd vs bwd. |
| BP-20 | Thermal steady-state / power cap | ❌ | high | `run_benchmarks` (920–938) runs all cases sequentially in fixed registry order: no cooldown, no temperature/throttle polling, no kernel-vs-ref interleaving (kernel always timed first), no power cap, no discard of throttled samples. With unlocked clocks (BP-05), heavier later cases (8192³ gemm) run hotter and may throttle, biasing the table by position. | Set and record a power cap (`nvidia-smi -pl`); poll `temperature.gpu`/`throttle_reasons.active`/`clocks.sm` between cases until baseline; interleave kernel vs ref; discard first N reps; drop throttled samples. |
| BP-21 | GQA decode KV bytes | ✅ | high | `_bench_attn_decode` `kv_bytes = 2·num_seqs·kvh·ctx_len·hd·dtype_bytes` (589) correctly uses kvh (n_kv_heads=8), not qh (32). Shapes use qh=32, kvh=8 GQA (516–517). FLOPs correctly not reported for decode. | None. Optionally also report AI≈O(1) and % peak BW to confirm the memory-bound classification. |
| BP-22 | CUDA-graph L2 caveat | ➖ | high | No CUDA-graph path exists (BP-09 fail), so the replay-vs-flushed mixing pitfall cannot occur. But the eager path ALSO never flushes L2 (BP-04 fail), so all numbers are warm-cache eager, and the report does not state this caching policy. | When BP-09's graph path is added, label each number's method and never mix as equivalent. Independently, state the current warm-L2 eager caching policy in the report. |
| BP-23 | Wave quantization | ❌ | high | All shapes are powers-of-two / round multiples (e.g. 344–348, 605–609). `gpu.sm_count` is detected (155–161) but never used to compute wave size or choose grids near/away from boundaries. A single favorable aligned size per op can hide partial-wave penalties. | Compute wave size = `sm_count × CTAs/SM`; sweep around wave boundaries + non-power-of-2 (per BP-19); annotate results on a partial wave; prefer many-wave sizes. |
| BP-24 | GPU isolation + layout fairness | ⚠️ | medium | `build_and_bench.sh` never sets `CUDA_VISIBLE_DEVICES` and never checks for other compute processes; always uses device 0 (`Ctx.device='cuda'`, 309). For attention the kernel takes `[b,s,h,hd]` natively while `_ref_sdpa` transposes to `[b,h,s,hd]` (555–557) INSIDE the timed ref (547), so the ref pays a layout-conversion cost the kernel does not — asymmetric, ref-penalizing (partly offsetting the fp32-ref penalty in BP-08; neither is controlled). | Set `CUDA_VISIBLE_DEVICES` to one device and assert no other compute process; feed each impl its native layout with conversion excluded for both (or included for both); document per-impl layout. |
| BP-25 | No CUDA_LAUNCH_BLOCKING in timing | ✅ | medium | No `CUDA_LAUNCH_BLOCKING` anywhere and no profiler attached during timing. Async overlap is not destroyed. | None. Optionally assert `os.environ.get('CUDA_LAUNCH_BLOCKING') != '1'` at startup and warn. |
| BP-26 | Saturate tiny kernels / flush knob | ❌ | medium | No `_sleep` and no L2-flush buffer. The decode/single-token shapes (rows=1/tokens=1) are exactly the tiny-kernel regime where the CPU outruns the GPU and single-launch event granularity is poor, yet they are timed with single-launch event pairs and no saturation. Tied to BP-09/BP-04. | Inject `torch.cuda._sleep(cycles)` for tiny kernels and/or use the CUDA-graph divide-by-N path; add the L2-eviction scratch buffer (BP-04). |
| BP-27 | Timer/Compare over timeit | ⚠️ | medium | No `timeit` — the pitfall is correctly avoided; `time_cuda` hand-rolls correct CUDA-event timing (242–261). But it does not use `torch.utils.benchmark.Timer`/`blocked_autorange` (no auto-ranging, no sig-fig trim) nor `Compare`; `render_markdown` hand-builds the table (995–1012) with no trim/colorize/highlight; `num_threads` never declared. | Either adopt `Timer.blocked_autorange` + `Compare`, or keep the custom harness but add significant-figure trimming and an explicit `num_threads`. The custom timer is acceptable; missing auto-ranging/sig-figs is the gap. |
| BP-28 | fast_p aggregate | ❌ | medium | No `fast_p`/`percentile`. `render_markdown`/`render_json` emit per-row ks_us/rel_err/speedup but compute NO aggregate — no fast_0/1/2, no mean-speedup-over-correct-only. With no correctness gate (BP-06), a wrong-but-fast kernel contributes a misleading speedup. | After the run compute `fast_p(p)=mean[is_correct AND ref_us/ks_us>p]` for p∈{0,1,2} using a rel_err threshold, plus mean speedup over correct-only; print a summary block. |
| BP-29 | Attention FLOP/byte convention | ✅ | medium | Prefill FLOPs use matmul-only `4·b·qh·seq²·hd·0.5` (537) with a comment noting softmax is excluded; the kernel is fused flash attention so S×S scores are not materialized and no N² byte term is needed. Decode reports KV bytes only (589). Consistent with the FlashAttention convention. | None for the fused kernels. If a non-fused/standard-attention baseline is added, count its O(N²) score/prob HBM traffic. |
| BP-30 | Callgrind A/B | ➖ | low | No `callgrind`. Low-severity optional technique for CPU-side launch/dispatch A/B, not applicable to this GPU-kernel-time harness; absence is acceptable. | Optional: `Timer(...).collect_callgrind()` with `denoise=True` for CPU-side dispatch overhead. Not needed for GPU kernel timing. |

Tally: ✅ 5 · ⚠️ 11 · ❌ 12 · ➖ 2.

---

## 3. Overall verdict

**Grade: C-.** The timing **core** is correct and trustworthy — per-iteration
CUDA events, enqueue-all-then-synchronize-once, median reported (BP-01, BP-02
pass); a genuine warmup exists; the classic `timeit` and `CUDA_LAUNCH_BLOCKING`
pitfalls are avoided (BP-25, BP-27); GQA decode byte counting (BP-21) and the
FlashAttention FLOP convention (BP-29) are correct. As a measurement of *device
time for one kernel on warm inputs at whatever clock the GPU happened to be
running*, `bench.py` is sound.

**It does not yet meet industry best practice, and the speedup numbers it
produces are not trustworthy as written.** The most consequential gaps:

- **No L2 flush (BP-04, critical).** Every memory-bound op (rmsnorm, layernorm,
  swiglu, rope, decode, sampling, CE, adamw) is benchmarked on L2-resident
  inputs, so the reported GB/s is L2 bandwidth, not HBM — often several times
  too fast. rmsnorm rows=1 fits entirely in L2.
- **No clock locking or persistence (BP-05, critical).** Dynamic boost/throttle
  is the single largest variance source; the harness leans on the median to
  "be robust to" jitter instead of removing it, and reports no spread to expose
  it.
- **Correctness never gates timing (BP-06, critical).** `rel_err` is computed
  *after* the kernel is already timed, there is no tolerance gate, only one
  random input set is used, and two ops (w4a16, moe_grouped_gemm) have no
  reference at all — so a numerically broken kernel still earns a "fast"
  headline.
- **Strawman baselines (BP-07) and unfair precision (BP-08), critical.** Every
  reference is eager PyTorch; there is no `torch.compile`, FlashAttention, or
  cuBLAS-best column, and the attention reference is timed in fp32 against an
  fp16 kernel — a precision-mismatched comparison that inflates the speedup.
- **No launch-overhead amortization (BP-09, critical)** despite explicit
  rows=1/tokens=1 decode shapes whose reported microseconds are dominated by
  launch latency, not kernel time.

Until at least the L2 flush, clock locking, correctness gating, dtype/baseline
fairness, and a CUDA-graph decode path are in place, the table should be read as
"warm-L2, unlocked-clock, eager-strawman, ungated" device times — useful for
catching gross regressions in a single kernel over time, **not** for the
headline speedups it currently presents.

---

## 4. Prioritized fix list (critical/high first)

### P0 — Critical (do these before publishing any speedup)

1. **Add a Triton-style L2 flush between iterations (BP-04, BP-26).**
   Allocate one scratch buffer larger than any L2 and zero it before each
   `fn()`, inside `time_cuda`'s warmup, estimate, and timed loops:
   ```python
   # once, at module/Ctx init
   _l2_flush = torch.empty(int(256e6) // 4, dtype=torch.int32, device="cuda")  # 256MB, fast_flush

   # inside time_cuda, before each fn():
   for i in range(n):
       _l2_flush.zero_()        # evict inputs from L2
       starts[i].record()
       fn()
       ends[i].record()
   ```
   Or simply replace the hand-rolled loop with `triton.testing.do_bench(fn,
   quantiles=[0.5, 0.2, 0.8])`, which handles L2 eviction and budget-derived
   counts for free.

2. **Gate timing on multi-input correctness, and add the two missing references
   (BP-06, BP-10, BP-28).**
   Run correctness *first*, on ≥5 random input sets, with
   `torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)` for fp16/bf16;
   only time kernels that pass; set `status="error"` and skip timing otherwise.
   Add references: dequantize-then-`@` for w4a16; a per-expert torch GEMM loop
   over the offsets for moe_grouped_gemm. Also add a `>100%`-of-peak
   DCE/caching assertion.

3. **Lock and record clocks; enable persistence (BP-05, BP-20).**
   In `build_and_bench.sh`:
   ```sh
   sudo nvidia-smi -pm 1
   sudo nvidia-smi -pl <sustainable_watts>
   sudo nvidia-smi --lock-gpu-clocks=<v> --lock-memory-clocks=<v>   # mid value from -q -d SUPPORTED_CLOCKS
   python bench.py ...
   sudo nvidia-smi --reset-gpu-clocks --reset-memory-clocks
   ```
   Record locked + achieved SM/mem clocks, power cap, and
   `clocks.throttle_reasons.active` into cfg; discard or flag throttled samples.

4. **Make every reference fair in dtype and TF32 (BP-07, BP-08).**
   Pin once at startup and record:
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True   # explicit, recorded
   torch.backends.cudnn.allow_tf32 = True
   torch.set_float32_matmul_precision("high")
   ```
   Remove the `.float()` casts from the *timed* SDPA path (keep a separate fp32
   path only for the correctness baseline). Add baseline columns for
   `torch.compile(mode="max-autotune")`, each pinned SDPA backend
   (`torch.nn.attention.sdpa_kernel`), and cuBLAS `a@b` — all in the kernel's
   dtype — and headline the speedup vs the **fastest**, labeling which it was.

5. **Add a CUDA-graph timing path for tiny/decode shapes (BP-09, BP-22, BP-26).**
   ```python
   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       for _ in range(n_repeat):
           fn()
   torch.cuda.synchronize()
   # measure:
   start.record(); g.replay(); end.record(); torch.cuda.synchronize()
   per_iter_ms = start.elapsed_time(end) / n_repeat
   ```
   Use it for rows=1/tokens=1/decode/sampling cases; for tiny kernels also
   `torch.cuda._sleep(cycles)` so the CPU doesn't outrun the GPU. **Label each
   number** with its method (flushed-eager vs graph-replay) and never compare
   the two as equivalent (BP-22).

### P1 — High

6. **Report min + p20 + p80 (not just median) (BP-13).**
   Have `time_cuda` return the full per-iteration array; add `min`/`p20`/`p80`
   (or IQR) fields to `Result` and columns to the markdown/json. With
   `do_bench`, pass `quantiles=[0.5, 0.2, 0.8]`. Flag samples taken while
   `throttle_reasons.active`.

7. **Budget-derive warmup and iteration counts (BP-03, BP-12).**
   Estimate one-iter cost from a few cache-clearing iters, then
   `n_warmup = max(warmup, int(25e-3 / est_s))` and
   `n_iters = max(1, int(100e-3 / est_s))`; print the counts used per case.
   Set `CUDA_MODULE_LOADING=EAGER` and trigger any autotune/JIT once before
   warmup.

8. **Add %-of-peak columns and roofline classification (BP-15, BP-16).**
   Per row, emit `compute_util = tflops / peak_tf` and
   `bw_util = gbps / peak_bw` (print numerator, denominator, SKU+dtype).
   Compute `AI = FLOPs / bytes` and `ridge = peak_FLOPs / peak_GBps`; choose
   the reported metric from AI vs ridge and print AI + the classification.
   Disambiguate the peak table for SXM/PCIe/NVL and 40/80GB using detected
   memory size.

9. **Fix the byte/FLOP formula gaps (BP-14).**
   Add cos/sin reads to rope bytes (`+ tokens·hd·dtype_bytes`); count each
   tensor's actual element size (fp32 for the CE losses buffer and the adamw
   m/v states) instead of one `dtype_bytes(dt)`; document each formula inline.

10. **Sweep non-power-of-2 / wave-boundary shapes and split fwd/bwd
    (BP-19, BP-23).**
    Add odd seqlens and grids near `sm_count × CTAs/SM` boundaries; annotate
    partial-wave cases. Add forward-only, backward-only, and end-to-end rows
    for attention and gemm with the correct multipliers (attn bwd ~2.5x, gemm
    bwd ~2x); split CE forward vs backward.

11. **Complete the reproducibility metadata (BP-18).**
    Extend cfg/header with driver version, `nvcc --version`,
    `torch.version.cuda`/cuDNN, TF32 flags, locked + achieved clocks, power cap,
    ECC state, cache-flush policy, "timing includes launch overhead: yes/no",
    the git commit hash of the harness, and per-row % of peak — in both
    markdown and json.

12. **Document/automate the ncu cross-check (BP-17).**
    Add an optional `ncu --section SpeedOfLight,MemoryWorkloadAnalysis,Occupancy
    --kernel-name <regex> --launch-count N` pass and reconcile
    `dram__bytes.sum` with the analytic byte counts; surface achieved occupancy
    and SoL %. At minimum document it as a recommended manual step.

### P2 — Medium

13. **Enforce GPU isolation and symmetric layout (BP-24).** Set
    `CUDA_VISIBLE_DEVICES` to one device and assert no other compute process via
    `nvidia-smi` before running; for attention, exclude the layout transpose from
    the timed region for both impls (or include packing for both) and document
    each impl's layout.
14. **Compute a `fast_p` aggregate (BP-28).** After the run, print
    `fast_0/fast_1/fast_2` plus mean speedup over correct-only samples, using a
    `rel_err` threshold for `is_correct`.
15. **Add significant-figure trimming and an explicit `num_threads`** to the
    table (or adopt `torch.utils.benchmark.Compare`) (BP-27); assert
    `CUDA_LAUNCH_BLOCKING != '1'` at startup (BP-25).

### Cosmetic / cleanup

16. Delete the dead `perf_counter` fallback in `time_cuda` (263–271) and the
    docstring claim that it "falls back to a CPU wall-clock timer" (13–15)
    (BP-01) — `main()` already requires torch+CUDA.

---

*Relevant files: `/Users/bytedance/code/kernel-set/benchmarks/bench.py`,
`/Users/bytedance/code/kernel-set/benchmarks/build_and_bench.sh`,
`/Users/bytedance/code/kernel-set/benchmarks/README.md`.*
