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
| `ks (us)` | kernel-set latency — **median** of `--iters` launches, timed with CUDA events |
| `ref (us)` | the PyTorch reference latency (same timing method), where a reference exists |
| `GB/s` | achieved memory bandwidth (bandwidth-bound ops: norm, activation, rope, decode, sampling, CE, adamw) |
| `TFLOP/s` | achieved compute throughput (compute-bound ops: gemm, w8a8, w4a16, prefill attention, grouped GEMM) |
| `rel_err` | max relative error of the kernel-set output vs the PyTorch reference (computed in fp32) |
| `speedup` | `ref_us / ks_us` |

Op categories: `rmsnorm`, `layernorm`, `swiglu`, `rope`, `attention`
(prefill + decode), `gemm` (fp16/bf16), `w8a8`, `w4a16`, `moe` (gate +
grouped GEMM), `sampling`, `cross_entropy`, `adamw`.

Shapes are drawn from real LLMs (Llama-3-8B/70B, Mistral-7B, Mixtral,
DeepSeek-style MoE) and include both prefill (large token counts) and decode
(single-token) regimes.

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
--warmup N        warmup launches                          (default: 10)
--iters N         timed launches                           (default: 50)
--output FILE     write the report here (default: stdout)
--format {md,json}                                         (default: md)
--list-ops        print the op/shape catalog and exit
--gpu-only        print detected GPU info as JSON and exit
```

Examples:

```bash
python benchmarks/bench.py --list-ops
python benchmarks/bench.py --ops rmsnorm,swiglu,gemm --dtype bf16 --iters 100
python benchmarks/bench.py --ops attention --shape decode
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
  (e.g. `F.layer_norm`, `F.scaled_dot_product_attention` with GQA expansion,
  `A @ B`, `F.cross_entropy`). W4A16 and MoE grouped-GEMM have no portable
  built-in torch equivalent, so they report **throughput only** (no `rel_err`).
* **Decode is memory-bound.** Paged-attention decode and AdamW report `GB/s`
  (KV-cache / state traffic), not FLOP/s, because they are bandwidth-limited.
* **dtype guards.** `bf16` needs `sm_80+`; W4A16 activations must be fp16/bf16.
  Unsupported cases are reported as `skip` rather than failing the run.
* **Roofline context.** When the GPU is recognized, the report header prints the
  device's marketing-peak bandwidth and fp16 tensor-core throughput so you can
  read achieved numbers as a fraction of peak.
* If a single kernel raises, that row is marked `error: …` and the rest of the
  run continues; the process exits non-zero only if *every* benchmark failed.
