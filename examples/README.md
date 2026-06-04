# kernel-set — examples

One minimal, end-to-end example per language, each calling **`ks_rms_norm`**
(RMSNorm) through that language's binding. They all do the same thing: allocate
device buffers, upload an input and a weight, run the kernel, copy the result
back, and print it.

All four need the built shared library on hand. Build it once from the repo root
and point the bindings at it:

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89   # 89=L4/4090, 80=A100, 90=H100
cmake --build build -j
export KERNEL_SET_LIB="$PWD/build/libkernel_set.so"
```

| Language | Path | Run |
|----------|------|-----|
| Python (torch) | [`python/rmsnorm_torch.py`](python/rmsnorm_torch.py) | `pip install ./bindings/python "kernel_set[torch]"` then `python examples/python/rmsnorm_torch.py` |
| Rust | [`rust/`](rust/) | `KERNEL_SET_LIB="$PWD/build" cargo run --manifest-path examples/rust/Cargo.toml` |
| Go | [`go/`](go/) | set `CGO_CFLAGS`/`CGO_LDFLAGS`/`LD_LIBRARY_PATH` (see below), then `go run ./examples/go` |
| TypeScript | [`ts/rmsnorm.ts`](ts/rmsnorm.ts) | `cd bindings/ts && npm install && npm run build`, then `npx ts-node examples/ts/rmsnorm.ts` |

The Python example uses torch tensors (dtype/shape/stream inferred). The Rust,
Go, and TS examples use the binding's built-in device allocator and explicit
host↔device copies, so they need no GPU tensor framework.

These examples are written against real CUDA hardware. On a machine without a
GPU they will compile/typecheck but fail at run time when the kernel launches —
that is expected. See [`../docs/USAGE.md`](../docs/USAGE.md) for the full
per-language quickstart and [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)
for the design.

### Go environment

cgo needs the headers at compile time and the library at link/run time:

```sh
export CGO_CFLAGS="-I$PWD/include"
export CGO_LDFLAGS="-L$(dirname "$KERNEL_SET_LIB")"
export LD_LIBRARY_PATH="$(dirname "$KERNEL_SET_LIB")"    # DYLD_LIBRARY_PATH on macOS
go run ./examples/go
```
