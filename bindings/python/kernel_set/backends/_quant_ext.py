"""External-provider adapters for quantization ops being wired into dispatch.

These are *lazy* thin adapters for quantization-domain ops that already have
catalogued providers in ``providers/registry.json`` but were not yet wired into
the curated dispatch table in ``_registry.py`` (NVFP4 / MXFP4 GEMM + quant,
fp8 KV-cache reshape-and-cache, GPTQ/AWQ/Machete weight repack, per-token-group
1x128 dynamic fp8 quant, and quantized low-precision attention).

Same conventions as the adapters in ``_registry.py``:

* Each adapter imports its backend library lazily *inside the function body* so
  heavy/optional deps stay out of import time.
* The exact API call mirrors the ``python_call`` string in
  ``providers/registry.json`` (the source of truth) for the corresponding op.
* Signatures are keyword-friendly and accept ``**_`` to swallow the gpu/dtype/
  arch kwargs the dispatcher threads through.
* NO try/except: import-availability and arch gating are handled upstream by the
  dispatch probe (``backends._probe``), which never calls an unavailable or
  arch-gated adapter. The adapter just imports and calls.

Each adapter returns the produced output tensor(s) with the same ergonomics as
the existing ``_registry.py`` adapters: GEMMs return the output matrix; quant
helpers return ``(packed_data, scale)`` tuples; the KV-write / repack ops return
the (possibly in-place-mutated) cache / repacked tensor.

Layouts / formats are documented per-function. ``registry.json`` op refs:
``nvfp4_gemm``, ``mxfp4_gemm``, ``fp4_quantize``, ``quantize_fp8_dynamic``,
``kv_cache_reshape_and_cache``, ``quantized_low_precision_attention``; atomic
ops (``providers/atomic_ops.json``): ``gptq_marlin_repack``,
``awq_marlin_repack``, ``machete_prepack_B``, ``per_token_group_fp8_quant``,
``sgl_per_token_group_quant_8bit``, ``reshape_and_cache_flash``,
``scaled_fp4_quant``, ``cutlass_scaled_fp4_mm``, ``mm_fp4``.
"""

from __future__ import annotations


def _imp(modpath: str):
    """Lazy import helper (mirrors ``_registry._imp``)."""
    import importlib

    return importlib.import_module(modpath)


# =========================================================================== #
# NVFP4 GEMM. registry.json op `nvfp4_gemm`.
# NVFP4 = 4-bit e2m1 data (packed 2/byte) with a 1x16 micro-block fp8-e4m3
# block scale + a per-tensor fp32 global scale (alpha). D = alpha * (A_fp4 @ B_fp4).
# Blackwell (sm100/sm120) tensor cores; 2-4x over fp8 on B200.
# =========================================================================== #
def _nvfp4_gemm_flashinfer(a_fp4, b_fp4, a_scale, b_scale, *, alpha=None,
                           out_dtype=None, **_):
    """FlashInfer NVFP4 GEMM (registry rank 1).

    ``flashinfer.gemm.mm_fp4(a, b, a_descale, b_descale, alpha, out_dtype,
    block_size=16, use_nvfp4=True, backend='auto')``. Multi-backend
    (trtllm/cutlass/cudnn/cute-dsl) NVFP4 GEMM with auto-selection — the
    production Blackwell FP4 GEMM used by SGLang/vLLM.

    a_fp4 (M, K/2) packed e2m1; b_fp4 (N, K/2) packed e2m1; a_scale/b_scale are
    the fp8-e4m3 1x16 block descales; ``alpha`` the fp32 per-tensor global
    scale. Quantize inputs with ``flashinfer.fp4_quantization.nvfp4_quantize``.
    """
    import torch
    fi = _imp("flashinfer")
    return fi.gemm.mm_fp4(a_fp4, b_fp4, a_scale, b_scale, alpha=alpha,
                          out_dtype=out_dtype or torch.bfloat16,
                          block_size=16, use_nvfp4=True, backend="auto")


def _nvfp4_gemm_vllm(a_fp4, b_fp4, a_scale, b_scale, *, alpha=None,
                     out_dtype=None, **_):
    """vLLM CUTLASS NVFP4 scaled-mm (registry rank 2).

    ``ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, block_scale_a, block_scale_b,
    alpha, out_dtype=torch.bfloat16)``. a_fp4 (M, K/2) packed e2m1; b_fp4
    (N, K/2) packed e2m1; block_scale_* are the swizzled fp8-e4m3 1x16 block
    scales; ``alpha`` the fp32 global scale. Quantize with
    ``ops.scaled_fp4_quant(input, input_global_scale)``. sm100, CUDA 12.8+.
    """
    import torch
    from vllm import _custom_ops as ops
    return ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_scale, b_scale, alpha,
                                     out_dtype=out_dtype or torch.bfloat16)


def _nvfp4_quantize_vllm(x, global_scale=None, *, is_sf_swizzled_layout=True,
                         **_):
    """vLLM fused NVFP4 quant (registry.json `fp4_quantize` rank 1).

    ``q_fp4, block_scale = ops.scaled_fp4_quant(input, input_global_scale,
    is_sf_swizzled_layout=True)``. Returns ``(packed fp4 e2m1, swizzled fp8-e4m3
    1x16 block scale)`` feeding ``cutlass_scaled_fp4_mm``. ``global_scale`` is
    the per-tensor fp32 input global scale.
    """
    from vllm import _custom_ops as ops
    return ops.scaled_fp4_quant(
        x, global_scale, is_sf_swizzled_layout=is_sf_swizzled_layout)


def _nvfp4_quantize_flashinfer(x, global_scale=None, **_):
    """FlashInfer NVFP4 quant (registry.json `fp4_quantize` rank 2).

    ``flashinfer.fp4_quantization.nvfp4_quantize(x, global_scale,
    sfLayout=SfLayout.layout_128x4, do_shuffle=False)`` -> ``(fp4_packed,
    scale_factors)``. Layout-aware (128x4 swizzle) fp4 quant matching the
    ``mm_fp4`` backends. ``global_scale`` is the fp32 per-tensor global scale.
    """
    fq = _imp("flashinfer.fp4_quantization")
    return fq.nvfp4_quantize(x, global_scale)


# =========================================================================== #
# MXFP4 GEMM. registry.json op `mxfp4_gemm`.
# MXFP4 = OCP microscaling: 4-bit e2m1 data with a shared ue8m0 (power-of-two)
# scale per 32-element block. Used by GPT-OSS / OCP MX inference on Blackwell.
# =========================================================================== #
def _mxfp4_gemm_flashinfer(a_fp4, b_fp4, a_scale, b_scale, *, alpha=None,
                           out_dtype=None, **_):
    """FlashInfer MXFP4 GEMM (registry rank 2).

    Same ``flashinfer.gemm.mm_fp4`` entrypoint as NVFP4 but toggled to MXFP4:
    ``mm_fp4(a, b, a_descale, b_descale, alpha=None, out_dtype=torch.bfloat16,
    block_size=32, use_nvfp4=False, backend='auto')``. ``use_nvfp4=False``
    selects MXFP4 (ue8m0 block scale, block 32). a_fp4/b_fp4 packed e2m1;
    a_scale/b_scale are the ue8m0 (float8_e8m0fnu) block scales. Serves GPT-OSS
    MXFP4 weights on Blackwell.
    """
    import torch
    fi = _imp("flashinfer")
    return fi.gemm.mm_fp4(a_fp4, b_fp4, a_scale, b_scale, alpha=alpha,
                          out_dtype=out_dtype or torch.bfloat16,
                          block_size=32, use_nvfp4=False, backend="auto")


def _mxfp4_gemm_torchao(x, weight, *, block_size=32, **_):
    """torchao MXFP4 inference linear (registry rank 1).

    torchao's MXFP4 path is a module-swap config, not a raw GEMM symbol:
    ``quantize_(model, MXFPInferenceConfig(block_size=32))``. MXFP4 uses
    ``torch.float4_e2m1fn_x2`` data + ``torch.float8_e8m0fnu`` (ue8m0) scale.
    This adapter applies the MXFP4 inference config to a single ``nn.Linear``
    (wrapping ``x``'s producer) and runs it: it builds a Linear from ``weight``,
    quantizes it in place, and returns ``linear(x)``. (NVFP4 via
    ``NVFP4InferenceConfig()``.) Blackwell sm100/sm120, CUDA 12.8+; prototype API.
    """
    import torch
    from torchao.quantization import quantize_
    from torchao.prototype.mx_formats import MXFPInferenceConfig
    out_features, in_features = weight.shape
    linear = torch.nn.Linear(in_features, out_features, bias=False,
                             device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(weight)
    quantize_(linear, MXFPInferenceConfig(block_size=block_size))
    return linear(x)


def _mxfp4_gemm_vllm(a, qweight, scales, *, num_bits=4, size_m=None,
                     size_n=None, size_k=None, workspace=None, alpha=None, **_):
    """vLLM Marlin-MXFP4 GEMM (registry rank 3).

    GPT-OSS MXFP4 is served through the Blackwell-gated Marlin/CUTLASS path:
    ``ops.marlin_gemm`` with an mxfp4 ScalarType. ``a`` is the fp16/bf16
    activation (M, K); ``qweight`` the Marlin-prepacked MXFP4 weight; ``scales``
    the ue8m0 block scales.
    """
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    m, k = a.shape
    size_m = size_m or m
    size_k = size_k or k
    size_n = size_n or scales.shape[-1]
    if workspace is None:
        workspace = torch.zeros(size_n // 64 * 16, device=a.device,
                                dtype=torch.int32)
    return ops.marlin_gemm(
        a, None, qweight, None, scales, None, alpha, None, None, None,
        workspace, scalar_types.float4_e2m1f, size_m, size_n, size_k,
        True, False, False, False)


# =========================================================================== #
# FP8 KV-CACHE reshape-and-cache (quantize-on-write).
# registry.json op `kv_cache_reshape_and_cache`; atomic `reshape_and_cache_flash`.
# Scatter K/V vectors into a paged (FlashAttention-layout) KV cache by slot
# mapping, quantizing to fp8 (e4m3/e5m2) with per-tensor/per-head fp32 scales.
# =========================================================================== #
def _reshape_and_cache_fp8_vllm(key, value, key_cache, value_cache,
                                slot_mapping, k_scale, v_scale, *,
                                kv_cache_dtype="fp8", **_):
    """vLLM fp8 paged-KV write (registry rank 1).

    ``ops.reshape_and_cache_flash(key, value, key_cache, value_cache,
    slot_mapping, kv_cache_dtype, k_scale, v_scale)``. Writes the new K/V
    vectors into the FlashAttention-layout paged caches in place, quantizing to
    fp8 with the (per-tensor or per-head) fp32 ``k_scale`` / ``v_scale``.

    key/value: ``(num_tokens, num_kv_heads, head_dim)``.
    key_cache/value_cache: FA paged layout ``(num_blocks, block_size,
    num_kv_heads, head_dim)`` (fp8 storage). slot_mapping: ``(num_tokens,)``
    int64 flat slot indices. Mutates the caches in place and returns
    ``(key_cache, value_cache)``. sm70+; fp8 needs sm89+.
    """
    from vllm import _custom_ops as ops
    ops.reshape_and_cache_flash(key, value, key_cache, value_cache,
                                slot_mapping, kv_cache_dtype, k_scale, v_scale)
    return key_cache, value_cache


# =========================================================================== #
# WEIGHT REPACK. atomic ops `gptq_marlin_repack`, `awq_marlin_repack`,
# `machete_prepack_B`. One-shot weight-layout transforms run at load time so the
# corresponding W4A16 GEMM (Marlin / Machete) can consume the weights.
# =========================================================================== #
def _repack_gptq_marlin_vllm(qweight, perm, size_k, size_n, *, num_bits=4,
                             is_a_8bit=False, **_):
    """vLLM GPTQ->Marlin weight repack.

    ``ops.gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits,
    is_a_8bit)`` -> Marlin-layout packed weight Tensor. ``qweight`` is the
    GPTQ-packed int4/int8 weight; ``perm`` the GPTQ column permutation;
    ``size_k``/``size_n`` the logical K/N dims. sm80+.
    """
    from vllm import _custom_ops as ops
    return ops.gptq_marlin_repack(qweight, perm, size_k, size_n, num_bits,
                                  is_a_8bit)


def _repack_awq_marlin_vllm(qweight, size_k, size_n, *, num_bits=4,
                            is_a_8bit=False, **_):
    """vLLM AWQ->Marlin weight repack.

    ``ops.awq_marlin_repack(b_q_weight, size_k, size_n, num_bits, is_a_8bit)``
    -> Marlin-layout packed weight Tensor (then served via marlin_gemm).
    ``qweight`` is the AWQ-packed int4/int8 weight; no GPTQ-style ``perm`` (AWQ
    has no g_idx permutation). sm80+.
    """
    from vllm import _custom_ops as ops
    return ops.awq_marlin_repack(qweight, size_k, size_n, num_bits, is_a_8bit)


def _prepack_machete_vllm(b, *, a_type=None, b_type=None,
                          group_scales_type=None, **_):
    """vLLM Machete weight prepack (sm90a).

    ``ops.machete_prepack_B(B, a_type, b_type, group_scales_type)`` -> Tensor
    reordered into the Machete (CUTLASS TMA+WGMMA) kernel layout. ``b`` is the
    raw int4/int8 weight; ``a_type`` the activation ScalarType (e.g.
    ``torch.bfloat16``); ``b_type`` the quant ScalarType (e.g.
    ``scalar_types.uint4b8``); ``group_scales_type`` the group-scale dtype.
    Pairs with ``ops.machete_mm``. sm90.
    """
    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    if a_type is None:
        a_type = torch.bfloat16
    if b_type is None:
        b_type = scalar_types.uint4b8
    return ops.machete_prepack_B(b, a_type, b_type, group_scales_type)


# =========================================================================== #
# PER-TOKEN-GROUP (1x128) dynamic fp8 quant. registry.json op
# `quantize_fp8_dynamic`; atomic ops `per_token_group_fp8_quant`,
# `sgl_per_token_group_quant_8bit`, deepgemm `per_token_cast_to_fp8`.
# Produces (fp8_e4m3 tensor, fp32 1x128 block scales) for blockwise fp8 GEMM.
# =========================================================================== #
def _per_token_group_quant_fp8_vllm(x, *, group_size=128, **_):
    """vLLM per-token-group (1x128) dynamic fp8 quant (registry rank 1).

    Via the fused dynamic-quant kernel with a blockwise group shape:
    ``q, scale = ops.scaled_fp8_quant(input, scale=None,
    use_per_token_if_dynamic=True, group_shape=(1, group_size))``. Returns
    ``(fp8_e4m3 tensor, fp32 group scales)``. (The raw
    ``ops.per_token_group_fp8_quant(input, output_q, output_s, group_size, eps,
    fp8_min, fp8_max, scale_ue8m0, ...)`` writes preallocated buffers in place;
    this wrapper uses the allocating ``scaled_fp8_quant`` entrypoint.) sm89+.
    """
    from vllm import _custom_ops as ops
    return ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True,
                                group_shape=(1, group_size))


def _per_token_group_quant_fp8_sgl(x, *, group_size=128, eps=1e-10,
                                   scale_ue8m0=False, **_):
    """sgl-kernel per-token-group fp8 quant.

    SGLang's ``sgl_per_token_group_quant_8bit`` writes preallocated buffers:
    ``sgl_per_token_group_quant_8bit(input, output_q!, output_s!, group_size,
    eps, fp8_min, fp8_max, scale_ue8m0)``. This adapter allocates the
    ``(fp8_e4m3 data, fp32 1x128 group scales)`` outputs, invokes the kernel,
    and returns ``(output_q, output_s)``. ``x`` is fp16/bf16 (rows, hidden)
    with ``hidden % group_size == 0``. sm80+.
    """
    import torch
    sk = _imp("sgl_kernel")
    rows, hidden = x.shape
    num_groups = hidden // group_size
    output_q = torch.empty(rows, hidden, device=x.device,
                           dtype=torch.float8_e4m3fn)
    output_s = torch.empty(rows, num_groups, device=x.device,
                           dtype=torch.float32)
    finfo = torch.finfo(torch.float8_e4m3fn)
    sk.sgl_per_token_group_quant_8bit(x, output_q, output_s, group_size, eps,
                                      finfo.min, finfo.max, scale_ue8m0)
    return output_q, output_s


def _per_token_group_quant_fp8_deepgemm(x, **_):
    """DeepGEMM blockwise fp8 cast helper.

    ``deep_gemm.testing.per_token_cast_to_fp8(x)`` -> ``(fp8_tensor, scales)``
    with DeepGEMM's 1x128 per-token block-scaling layout (use
    ``per_block_cast_to_fp8`` for the 128x128 weight layout). Align LHS scales
    for the TMA path with ``deep_gemm.get_col_major_tma_aligned_tensor``.
    Reference helpers matching DeepGEMM's scaling layout. sm90/sm100.
    """
    from deep_gemm.testing import per_token_cast_to_fp8
    return per_token_cast_to_fp8(x)


# =========================================================================== #
# FP8 ATTENTION (prefill). registry.json op `quantized_low_precision_attention`.
# Quantized (INT8/FP8 QK and PV) attention for 2-5x over FA2 with near-lossless
# quality; training-free, drop-in for prefill. Thin wrappers — the heavy lifting
# (quantize QK to int8 / PV to fp8) is internal to the backend kernels.
# =========================================================================== #
def _fp8_attention_sage(q, k, v, *, causal=False, softmax_scale=None,
                        tensor_layout="HND", **_):
    """SageAttention quantized attention (registry rank 1).

    ``from sageattention import sageattn; sageattn(q, k, v,
    tensor_layout='HND', is_causal=False, sm_scale=None)`` — auto-selects the
    best INT8-QK + FP8-PV kernel for the host arch (sm80..sm120). q/k/v are
    fp16/bf16 in HND (heads, seqlen, head_dim) layout by default; the kernel
    quantizes QK to int8 and PV to fp8 internally. 2-3x over FlashAttention-2
    with near-lossless quality.
    """
    sa = _imp("sageattention")
    return sa.sageattn(q, k, v, tensor_layout=tensor_layout, is_causal=causal,
                       sm_scale=softmax_scale)


def _fp8_attention_flashinfer(q_fp8, k_fp8, v_fp8, *, causal=True,
                              softmax_scale=None, **_):
    """FlashAttention-3 FP8 prefill (registry rank 3, Hopper sm90).

    ``import flash_attn_interface; flash_attn_interface.flash_attn_func(q_fp8,
    k_fp8, v_fp8, causal=True)`` — FA3 FP8 (e4m3) forward on Hopper with descale
    handling. q/k/v are pre-quantized fp8-e4m3 in dense ``(B, S, H, D)`` layout.
    Exact-ish low-precision attention for inference prefill; up to ~1.2 PFLOPS
    on H100. (Named "_flashinfer" per the dispatch wiring request; the backend
    is the FA3 hopper ``flash_attn_interface`` package.)
    """
    fa3 = _imp("flash_attn_interface")
    return fa3.flash_attn_func(q_fp8, k_fp8, v_fp8, causal=causal,
                               softmax_scale=softmax_scale)
