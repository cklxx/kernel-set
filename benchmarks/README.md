# kernel-set benchmarks

A GPU benchmark harness for **kernel-set**. It detects the GPU, builds the
library for the right arch, and benchmarks every kernel category over
representative LLM shapes — reporting latency, achieved bandwidth/throughput,
and correctness + speedup vs a PyTorch reference.

Targets the GPUs you get on Colab: **L4** (`sm_89`) and **A100** (`sm_80`),
and works on any CUDA GPU (`sm_75`/`80`/`86`/`89`/`90`).

## What it measures

For each op category and shape:

| column | meaning |
|---|---|
| `ks us (min)` | kernel-set latency — **median (min)** microseconds over the auto-calibrated timed iterations, timed with CUDA events |
| `ref us (min)` | the fastest baseline's latency (same timing method), where a baseline exists |
| `GB/s (%pk)` | achieved memory bandwidth and its **% of dense peak** for this SKU+dtype (bandwidth-bound ops: norm, activation, rope, decode, sampling, CE, adamw) |
| `TFLOP/s (%pk)` | achieved compute throughput and its **% of dense peak** (compute-bound ops: gemm, w8a8, w4a16, prefill attention, grouped GEMM) |
| `rel_err` | max relative error of the kernel-set output vs the PyTorch reference (computed in fp32) |
| `spd` | speedup = `best_baseline_us / ks_us` |
| `base` | which baseline the speedup is against (e.g. `cublas(a@b)`, `torch.compile`, `sdpa(flash/efficient)`, `eager`) |
| `iters` | the auto-calibrated number of timed iterations actually used |
| `m` | timing method: `E` = cuda-events (L2-flushed), `C` = cudagraph replay (warm-L2, launch-overhead removed) |

**Correctness gates speed.** `rel_err` is checked against an op-appropriate,
dtype-aware tolerance (fp16 1e-2, bf16 3e-2, fp32 2e-3) **before** any timing
runs. A kernel that fails tolerance is reported as **INCORRECT** with its
`rel_err` — its speed is *not* reported, so a fast-but-wrong kernel can never
earn a clean speedup headline.

**Robust stats.** Each iteration is timed individually; the report shows the
median with the min in parentheses. The JSON additionally carries p20/p80 and
the min for both the kernel and the baseline.

**`fast_p` aggregate.** The report header prints a single joint
correctness-and-speed score:

> `fast_1` = fraction of comparable ops that are **both** correct **and** at
> least baseline speed (the headline); `fast_0` = correct at any speed;
> `fast_2` = correct and ≥ 2× baseline. Plus counts of
> correct / incorrect / error / skip / no-ref and the mean speedup over
> correct-only ops.

Op categories: `rmsnorm`, `layernorm`, `swiglu`, `rope`, `attention`
(prefill + decode), `gemm` (fp16/bf16), `w8a8`, `w4a16`, `moe` (gate +
grouped GEMM), `sampling`, `cross_entropy`, `adamw`.

Shapes are drawn from real LLMs (Llama-3-8B/70B, Mistral-7B, Mixtral,
DeepSeek-style MoE) and include both prefill (large token counts) and decode
(single-token) regimes.

### L2 flush vs CUDA graphs — pick one timing regime

Two mutually-exclusive timing regimes are available; **they are not comparable
to each other** and the method is labelled per row (`m` column) and in the
header:

* **L2 flush (default, `--l2-flush`).** A scratch buffer larger than the GPU's
  L2 (queried at ~2× L2, falling back to 256 MB) is zeroed immediately before
  *every* timed launch, so the op reads its inputs **cold from HBM**. This is
  the Triton `do_bench` regime and is the correct way to measure memory-bound
  ops — without it you measure L2 bandwidth (often several× HBM), not real DRAM
  throughput. Launch overhead **is** included (one event pair per launch).
* **CUDA graphs (`--cudagraph`).** The op is captured into a CUDA graph and
  replayed; timing divides by the replay count to **amortize the ~4–6 µs launch
  overhead** that dominates tiny / launch-bound decode kernels (rows=1,
  tokens=1, single-token decode, sampling). But graph replay does **not** flush
  L2, so these numbers are **warm-cache**. `--cudagraph` therefore disables the
  per-iteration L2 flush automatically and falls back to event timing if capture
  fails.

Rule of thumb: use the default (L2-flushed events) for HBM-bound ops and honest
throughput; add `--cudagraph` (implies `--no-l2-flush`) only to isolate kernel
time for launch-bound shapes, and never present the two side by side as equal.

### Clocks, TF32, and reproducibility

* **Clocks.** Current SM/mem clocks and active throttle reasons are **always**
  queried and printed in the header. `--lock-clocks` additionally attempts
  `nvidia-smi` persistence mode + lock-gpu-clocks to a sustainable mid value;
  if that is not permitted (e.g. Colab) the run still proceeds and the header
  states the lock was requested but not applied. Locked clocks are reset on exit.
* **TF32 fairness.** `torch.backends.cuda.matmul.allow_tf32` and
  `cudnn.allow_tf32` are pinned explicitly (default ON; `--no-tf32` to disable)
  and the setting is printed, so fp32-GEMM precision is never an undocumented
  variable.
* **Baselines.** GEMM is compared against the fastest of cuBLAS eager `a@b`
  **and** `torch.compile(max-autotune)` (when available); attention is compared
  against SDPA pinned to the **flash/efficient** backend in the *kernel's dtype*
  (the fp32 SDPA path is used only for the correctness check). The `base` column
  names the winner.
* **Repro header.** GPU + SMs + memory, driver, CUDA/cuDNN, nvcc, TF32 setting,
  clocks (+ locked?), L2-flush on/off and buffer size, timing method, whether
  launch overhead is included, target-ms / iters, harness git commit, and a
  timestamp (`--timestamp` to override).

## Files

| file | purpose |
|---|---|
| `bench.py` | the harness: GPU detection, timing, the benchmark registry, reporting (markdown/JSON) |
| `build_and_bench.sh` | build the lib for the detected arch, set `KERNEL_SET_LIB`, run `bench.py`, write `results/<gpu>.md` |
| `colab_bench.ipynb` | a copy-paste-runnable Colab notebook (L4/A100) |
| `results/` | generated reports, one per GPU (`results/l4.md`, `results/a100.md`, …) |

## Run locally

Prerequisites: a CUDA GPU, CUDA 12.x toolkit (`nvcc`), CMake ≥ 3.24, and
PyTorch built for CUDA (used to allocate tensors, time with CUDA events, and
provide the reference implementations).

One command — detects the GPU, builds, benchmarks, writes the report:

```bash
benchmarks/build_and_bench.sh
```

This writes `benchmarks/results/<gpu>.md` (e.g. `results/l4.md`). It maps the
detected GPU to a single CUDA arch (L4→`89`, A100→`80`, H100→`90`, T4→`75`,
otherwise the device's native `sm_XY`).

Useful environment overrides:

```bash
KS_DTYPE=bf16 KS_OPS=gemm,attention KS_ITERS=100 benchmarks/build_and_bench.sh
KS_ARCH=89 benchmarks/build_and_bench.sh      # force the CUDA arch
```

Any extra args are forwarded to `bench.py`, e.g.:

```bash
benchmarks/build_and_bench.sh --shape decode
```

### Running `bench.py` directly

If you already have a build, point the binding at it and run the harness:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
export KERNEL_SET_LIB=$PWD/build/libkernel_set.so

# make the binding importable (or `pip install ./bindings/python`)
export PYTHONPATH=$PWD/bindings/python:$PYTHONPATH

python benchmarks/bench.py --dtype fp16 --iters 50 \
    --output benchmarks/results/l4.md --format md
```

The `kernel_set` binding discovers the library through `KERNEL_SET_LIB` (full
path or bare filename) or `KERNEL_SET_LIB_DIR` (a directory) — see
[`bindings/python/README.md`](../bindings/python/README.md).

### `bench.py` CLI

```
--ops OPS         comma-separated op categories, or 'all' (default: all)
                  rmsnorm,layernorm,swiglu,rope,attention,gemm,w8a8,
                  w4a16,moe,sampling,cross_entropy,adamw
--dtype DTYPE     fp16 | bf16 | fp32                       (default: fp16)
--shape SUBSTR    only run shapes whose label contains SUBSTR (e.g. 'decode')
--warmup N        warmup launches (floor; budget may raise) (default: 10)
--target-ms MS    measurement budget; iteration count is auto-calibrated to
                  fill it, capped by --max-iters                (default: 200)
--iters N         fixed timed launches; OVERRIDES --target-ms auto-calibration
                  (default: auto)
--max-iters N     upper bound on the auto-calibrated count    (default: 1000)
--l2-flush        zero a >L2 buffer before each timed launch so each iteration
                  reads cold from HBM (Triton do_bench style)  (default: ON)
--no-l2-flush     disable the per-iteration L2 flush (warm-cache timing)
--cudagraph       time via CUDA-graph replay to amortize launch overhead for
                  tiny/launch-bound ops; implies --no-l2-flush (warm-L2),
                  falls back to events if capture fails        (default: off)
--lock-clocks     attempt nvidia-smi persistence + lock-gpu-clocks (no-op if
                  not permitted, e.g. Colab); clocks are always reported
--tf32 / --no-tf32  enable/disable TF32 for fp32 matmul+cudnn  (default: ON)
--timestamp STR   optional run label recorded in the header
--output FILE     write the report here (default: stdout)
--format {md,json}                                         (default: md)
--list-ops        print the op/shape catalog and exit
--gpu-only        print detected GPU info as JSON and exit
```

`--iters` (the legacy flag) still works and pins a fixed iteration count;
omitting it auto-calibrates from `--target-ms`.

Examples:

```bash
python benchmarks/bench.py --list-ops
python benchmarks/bench.py --ops rmsnorm,swiglu,gemm --dtype bf16 --target-ms 300

# launch-bound decode ops via cuda graphs (warm-L2, launch overhead removed)
python benchmarks/bench.py --ops attention --shape decode --cudagraph

# low-variance run: lock clocks if permitted, then full bf16 sweep
python benchmarks/bench.py --lock-clocks --dtype bf16

# disable the L2 flush to see warm-cache (L2-bandwidth) numbers
python benchmarks/bench.py --ops rmsnorm --no-l2-flush

python benchmarks/bench.py --format json --output results/l4.json
```

## Run on Colab (L4 / A100)

1. Open `benchmarks/colab_bench.ipynb` in Colab.
2. `Runtime ▸ Change runtime type ▸ GPU` → pick **L4** or **A100**.
3. Edit the *Get the source* cell so `KS_REPO_URL` points at your fork
   (or upload the repo to `KS_REPO_DIR`).
4. `Runtime ▸ Run all`.

The notebook shows the GPU (`nvidia-smi`), installs a recent CMake, builds
`libkernel_set.so` for the detected arch, runs `bench.py`, and renders the
results table inline. It is written to be runnable on a fresh Colab GPU runtime
with no extra setup beyond pointing it at the source.

## Notes & caveats

* **Reference selection.** Most ops compare against a clear PyTorch reference
  (`F.layer_norm`, `F.scaled_dot_product_attention` with GQA expansion, `A @ B`
  / `torch.compile`, `F.cross_entropy`). MoE grouped-GEMM is now checked against
  a per-expert torch GEMM loop over the CSR offsets. W4A16 still has no portable
  built-in torch int4 equivalent, so it reports **throughput only** (no
  `rel_err`, and is excluded from the correctness-gated `fast_p` population).
* **Correctness gates speed (see above).** Every op with a reference is verified
  to within its dtype tolerance *before* it is timed; failures are reported as
  **INCORRECT** instead of as a speedup.
* **Decode is memory-bound (and launch-bound).** Paged-attention decode and
  AdamW report `GB/s` (KV-cache / state traffic), not FLOP/s. Single-token decode
  shapes are dominated by launch overhead in the default event regime — use
  `--cudagraph` to isolate kernel time (warm-L2; see the L2-flush-vs-cudagraph
  note above).
* **dtype guards.** `bf16` needs `sm_80+`; W4A16 activations must be fp16/bf16.
  Unsupported cases are reported as `skip` rather than failing the run.
* **% of peak.** When the GPU is recognized, each `GB/s` / `TFLOP/s` cell is
  shown as a **% of the SKU+dtype dense peak** (sparsity stripped). The per-SKU
  table disambiguates A100 40 vs 80GB and H100 PCIe/SXM/NVL by detected memory
  size. An achieved value over ~105% of peak is the classic warm-cache /
  dead-code-elimination smell — the L2 flush (default) is what keeps the
  bandwidth numbers honest.
* If a single kernel raises, that row is marked `error: …` and the rest of the
  run continues; the process exits non-zero only if *every* benchmark failed.
