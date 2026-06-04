# Third-Party Notices

kernel-set bundles ("vendors") a number of upstream GPU/accelerator kernel libraries
under `third_party/`. **Each library is vendored verbatim under its own upstream
license. kernel-set does not modify these kernels — it only provides thin wrapper /
adapter code that exposes them through the kernel-set operator ABI (`ks_*`).**

Redistribution of kernel-set therefore redistributes these third-party works. The
notices below satisfy the attribution and license-retention requirements of each
upstream license. The full, unmodified license text for every library is preserved in
its own directory (`third_party/<category>/<lib>/LICENSE`, plus any upstream `NOTICE`
or `THIRDPARTYNOTICES.txt`). A per-library catalog with sizes, file counts, and the
kernel-set ops each one backs is in [`third_party/README.md`](third_party/README.md);
each library also carries a `SOURCE.md` documenting the exact upstream commit and
sparse-checkout scope.

The SPDX identifier in the table below is the canonical machine-readable id where one
exists. `cut-cross-entropy` ships a custom Apple permissive license that has no standard
SPDX id (see its dedicated note).

| Library | Upstream URL | Commit | SPDX License | License file(s) |
|---------|--------------|--------|--------------|-----------------|
| flash-attention | https://github.com/Dao-AILab/flash-attention | `d80a771` | `BSD-3-Clause` | `third_party/attention/flash-attention/LICENSE` |
| flashinfer | https://github.com/flashinfer-ai/flashinfer | `e00c306` | `Apache-2.0` | `third_party/attention/flashinfer/LICENSE`, `.../NOTICE` |
| FlashMLA | https://github.com/deepseek-ai/FlashMLA | `9241ae3` | `MIT` | `third_party/attention/FlashMLA/LICENSE` |
| SageAttention | https://github.com/thu-ml/SageAttention | `d1a57a5` | `Apache-2.0` | `third_party/attention/SageAttention/LICENSE` |
| DeepGEMM | https://github.com/deepseek-ai/DeepGEMM | `88965b0` | `MIT` | `third_party/gemm/DeepGEMM/LICENSE` |
| cutlass | https://github.com/NVIDIA/cutlass | `2599f29` | `BSD-3-Clause` | `third_party/gemm/cutlass/LICENSE.txt` |
| marlin | https://github.com/IST-DASLab/marlin | `1f25790` | `Apache-2.0` | `third_party/quant/marlin/LICENSE` |
| llm-awq | https://github.com/mit-han-lab/llm-awq | `d6e797a` | `MIT` | `third_party/quant/llm-awq/LICENSE` |
| exllamav2 | https://github.com/turboderp-org/exllamav2 | `7dc12af` | `MIT` | `third_party/quant/exllamav2/LICENSE` |
| bitsandbytes | https://github.com/bitsandbytes-foundation/bitsandbytes | `3343bac` | `MIT` | `third_party/quant/bitsandbytes/LICENSE`, `.../NOTICE.md` |
| vllm | https://github.com/vllm-project/vllm | `06ee2d8` | `Apache-2.0` | `third_party/vllm/LICENSE` |
| mamba | https://github.com/state-spaces/mamba | `6ff8ad1` | `Apache-2.0` | `third_party/ssm/mamba/LICENSE` |
| causal-conv1d | https://github.com/Dao-AILab/causal-conv1d | `4f6ae4e` | `BSD-3-Clause` | `third_party/ssm/causal-conv1d/LICENSE` |
| flash-linear-attention | https://github.com/fla-org/flash-linear-attention | `7378dfe` | `MIT` | `third_party/linear_attn/flash-linear-attention/LICENSE` |
| liger-kernel | https://github.com/linkedin/Liger-Kernel | `94236ea` | `BSD-2-Clause` | `third_party/triton/liger-kernel/LICENSE`, `.../NOTICE` |
| tilelang | https://github.com/tile-ai/tilelang | `550e25d` | `MIT` | `third_party/tilelang/tilelang/LICENSE`, `.../THIRDPARTYNOTICES.txt` |
| cut-cross-entropy | https://github.com/apple/ml-cross-entropy | `b7a0279` | `LicenseRef-Apple-Sample-Code` (custom; see note) | `third_party/training/cut-cross-entropy/LICENSE` |
| thunderkittens | https://github.com/HazyResearch/ThunderKittens | `34b15f7` | `MIT` | `third_party/megakernel/thunderkittens/LICENSE` |
| mirage | https://github.com/mirage-project/mirage | `b293bb6` | `Apache-2.0` | `third_party/megakernel/mirage/LICENSE` |
| hazy-megakernels | https://github.com/HazyResearch/Megakernels | `7309cec` | `MIT` | `third_party/megakernel/hazy-megakernels/LICENSE` |

## License-specific notes

- **Apache-2.0** libraries (flashinfer, SageAttention, mamba, vllm, marlin, mirage):
  redistribution must retain the `LICENSE` and, where present, the upstream `NOTICE`
  file. Both are preserved in-tree (flashinfer and bitsandbytes ship NOTICE files;
  Apache libs without an upstream NOTICE simply have none to retain).
- **BSD-3-Clause** (flash-attention, cutlass, causal-conv1d) and **BSD-2-Clause**
  (liger-kernel): redistribution must retain the copyright notice and license text
  (preserved in each `LICENSE`); the names of the projects/contributors must not be
  used for endorsement.
- **MIT** (FlashMLA, DeepGEMM, llm-awq, exllamav2, bitsandbytes, flash-linear-attention,
  tilelang, thunderkittens, hazy-megakernels): redistribution must retain the copyright
  notice and permission notice (preserved in each `LICENSE`).
- **cut-cross-entropy (Apple Sample Code License)**: this is a **custom Apple permissive
  license**, not a standard SPDX-listed license (referenced above as
  `LicenseRef-Apple-Sample-Code`). It explicitly permits redistribution of source with
  retention of the Apple copyright notice and license text — both are preserved verbatim
  in `third_party/training/cut-cross-entropy/LICENSE`. Copyright (C) 2024 Apple Inc.

## Submodules not vendored

Some upstream repos reference further dependencies as git submodules that were **empty
in the sparse checkouts and intentionally not vendored** (they are available standalone
in this tree where needed):

- flash-attention: `csrc/cutlass`, `csrc/composable_kernel` (submodules) — dropped.
- FlashMLA: `csrc/cutlass` (submodule) — dropped.
- hazy-megakernels: nested upstream `ThunderKittens` submodule — excluded
  (ThunderKittens is vendored separately at `third_party/megakernel/thunderkittens/`).

NVIDIA CUTLASS is independently vendored at `third_party/gemm/cutlass/` for the
libraries that depend on it.
