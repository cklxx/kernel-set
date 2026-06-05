# Operator Catalog — Industry Kernel Providers (2026-H1)

Exhaustive, ranked catalog of best-in-class open-source kernel providers for every
operator in the kernel-set surface. Auto-generated from `providers/registry.json`
(merged from the `providers/_frag_*.json` research fragments). For each operator the
providers are listed in rank order (rank 1 = recommended default).

## Summary

- **Total operators:** 127
- **Total providers:** 274
- **Domains:** 8
- **Schema version:** 1  •  **Generated for:** 2026-H1

### Operators & providers by domain

| Domain | Operators | Providers |
| --- | ---: | ---: |
| Attention (`attention`) | 20 | 46 |
| SSM / Linear Attention (`ssm-linear-attn`) | 13 | 17 |
| Dense GEMM (`gemm-dense`) | 6 | 16 |
| Quantized GEMM (`gemm-quant`) | 17 | 44 |
| Norm / Activation / RoPE (`norm-act-rope`) | 20 | 49 |
| MoE / Communication (`moe-comm`) | 12 | 28 |
| Sampling / Logit Processing (`sampling-logitproc`) | 16 | 36 |
| Loss / Optimizer / Misc (`loss-optim-misc`) | 23 | 38 |
| **Total** | **127** | **274** |

### Best default stack (rank-1 provider per operator)

The recommended single-library default for each operator. Build a model with these
and you get the current best-in-class kernel for every op.

| Operator | Domain | Rank-1 library |
| --- | --- | --- |
| `prefill_varlen_fwd` | Attention | flash-attn |
| `prefill_dense_fwd` | Attention | flash-attn |
| `attention_backward` | Attention | flash-attn |
| `decode_paged_attention` | Attention | flashinfer-python |
| `prefill_paged_attention` | Attention | flashinfer-python |
| `single_decode_attention` | Attention | flashinfer-python |
| `mla_decode_paged` | Attention | flash-mla (FlashMLA) |
| `mla_prefill` | Attention | flashinfer-python |
| `gqa_mqa_attention` | Attention | flash-attn |
| `sliding_window_local_attention` | Attention | flash-attn |
| `alibi_attention` | Attention | flash-attn |
| `logit_softcap_attention` | Attention | flash-attn |
| `attention_sink` | Attention | flash-attn |
| `cross_attention` | Attention | flash-attn |
| `append_kv_cache` | Attention | flashinfer-python |
| `ring_context_parallel_attention` | Attention | ring-flash-attn |
| `cascade_attention` | Attention | flashinfer-python |
| `tree_speculative_attention` | Attention | flashinfer-python |
| `quantized_low_precision_attention` | Attention | sageattention (SageAttention2++) |
| `tile_primitive_attention` | Attention | ThunderKittens |
| `mamba2_ssd_chunk_scan` | SSM / Linear Attention | mamba-ssm |
| `mamba1_selective_scan` | SSM / Linear Attention | mamba-ssm |
| `causal_conv1d` | SSM / Linear Attention | causal-conv1d |
| `gated_linear_attention` | SSM / Linear Attention | flash-linear-attention (fla-core) |
| `delta_rule` | SSM / Linear Attention | flash-linear-attention (fla-core) |
| `gated_delta_rule` | SSM / Linear Attention | flash-linear-attention (fla-core) |
| `retention_retnet` | SSM / Linear Attention | flash-linear-attention (fla-core) |
| `rwkv6_wkv` | SSM / Linear Attention | flash-linear-attention (fla-core) |
| `rwkv7_wkv` | SSM / Linear Attention | flash-linear-attention (fla-core) |
| `lightning_attention` | SSM / Linear Attention | flash-linear-attention (fla-core) |
| `linear_attention_basic` | SSM / Linear Attention | flash-linear-attention (fla-core) |
| `gated_slot_attention_gsa` | SSM / Linear Attention | flash-linear-attention (fla-core) |
| `hgrn2` | SSM / Linear Attention | flash-linear-attention (fla-core) |
| `gemm` | Dense GEMM | torch |
| `gemm_fp8` | Dense GEMM | DeepGEMM |
| `gemm_bias_act` | Dense GEMM | torch |
| `gemm_batched` | Dense GEMM | torch |
| `grouped_gemm` | Dense GEMM | DeepGEMM |
| `grouped_gemm_fp8` | Dense GEMM | DeepGEMM |
| `fp8_gemm_blockwise` | Quantized GEMM | deep_gemm |
| `fp8_gemm_scaled_mm` | Quantized GEMM | vllm |
| `quantize_fp8_dynamic` | Quantized GEMM | vllm |
| `dequantize_fp8` | Quantized GEMM | torchao |
| `int8_gemm_w8a8` | Quantized GEMM | vllm |
| `quantize_int8_dynamic` | Quantized GEMM | vllm |
| `dequantize_int8` | Quantized GEMM | torchao |
| `w4a16_gemm` | Quantized GEMM | vllm (Marlin / GPTQ-Marlin) |
| `awq_gemm` | Quantized GEMM | vllm (awq_marlin) |
| `dequantize_int4` | Quantized GEMM | autoawq |
| `int4_weight_only_gemm_tinygemm` | Quantized GEMM | torchao |
| `w4a8_gemm` | Quantized GEMM | vllm (Machete W4A8) |
| `nvfp4_gemm` | Quantized GEMM | flashinfer |
| `mxfp4_gemm` | Quantized GEMM | torchao |
| `fp4_quantize` | Quantized GEMM | vllm |
| `nf4_fp4_blockwise_quant_linear` | Quantized GEMM | bitsandbytes |
| `int8_llm_int8_linear` | Quantized GEMM | bitsandbytes |
| `rmsnorm` | Norm / Activation / RoPE | flashinfer-python |
| `fused_add_rmsnorm` | Norm / Activation / RoPE | flashinfer-python |
| `gemma_rmsnorm` | Norm / Activation / RoPE | flashinfer-python |
| `layernorm` | Norm / Activation / RoPE | apex |
| `qk_norm` | Norm / Activation / RoPE | flashinfer-python |
| `groupnorm` | Norm / Activation / RoPE | apex |
| `rmsnorm_quant` | Norm / Activation / RoPE | flashinfer-python |
| `silu_and_mul` | Norm / Activation / RoPE | flashinfer-python |
| `swiglu_oai_clamped` | Norm / Activation / RoPE | vllm |
| `gelu_and_mul` | Norm / Activation / RoPE | flashinfer-python |
| `silu` | Norm / Activation / RoPE | torch |
| `gelu` | Norm / Activation / RoPE | vllm |
| `relu` | Norm / Activation / RoPE | torch |
| `rope` | Norm / Activation / RoPE | flashinfer-python |
| `rope_train_backward` | Norm / Activation / RoPE | liger-kernel |
| `rope_llama31_scaling` | Norm / Activation / RoPE | flashinfer-python |
| `rope_yarn_ntk_scaling` | Norm / Activation / RoPE | vllm |
| `add_residual` | Norm / Activation / RoPE | torch |
| `cast` | Norm / Activation / RoPE | transformer-engine |
| `dropout` | Norm / Activation / RoPE | apex |
| `moe_gate_softmax_topk` | MoE / Communication | sgl-kernel |
| `moe_gate_sigmoid_group_topk` | MoE / Communication | sgl-kernel |
| `moe_align_block_size` | MoE / Communication | sgl-kernel |
| `moe_permute` | MoE / Communication | vllm |
| `moe_grouped_gemm_contiguous` | MoE / Communication | DeepGEMM |
| `moe_grouped_gemm_masked` | MoE / Communication | DeepGEMM |
| `moe_unpermute_combine` | MoE / Communication | vllm |
| `fused_moe_full` | MoE / Communication | vllm |
| `ep_dispatch_alltoall` | MoE / Communication | DeepEP |
| `ep_combine_alltoall` | MoE / Communication | DeepEP |
| `ep_low_latency_dispatch_combine` | MoE / Communication | DeepEP |
| `moe_tp_allreduce_fused` | MoE / Communication | flashinfer |
| `softmax` | Sampling / Logit Processing | flashinfer-python |
| `log_softmax` | Sampling / Logit Processing | torch |
| `argmax_greedy` | Sampling / Logit Processing | torch |
| `temperature_scaling` | Sampling / Logit Processing | flashinfer-python |
| `top_k_filter_mask` | Sampling / Logit Processing | flashinfer-python |
| `top_p_nucleus_filter` | Sampling / Logit Processing | flashinfer-python |
| `min_p_filter` | Sampling / Logit Processing | flashinfer-python |
| `typical_sampling` | Sampling / Logit Processing | transformers |
| `repetition_presence_frequency_penalty` | Sampling / Logit Processing | vllm |
| `logit_bias_badwords_mask` | Sampling / Logit Processing | vllm |
| `guided_grammar_mask` | Sampling / Logit Processing | xgrammar |
| `categorical_sample_from_probs` | Sampling / Logit Processing | flashinfer-python |
| `fused_temp_topk_topp_sample` | Sampling / Logit Processing | flashinfer-python |
| `speculative_verify_rejection_chain` | Sampling / Logit Processing | flashinfer-python |
| `speculative_verify_tree` | Sampling / Logit Processing | sgl-kernel |
| `topk_topp_renorm_probs` | Sampling / Logit Processing | flashinfer-python |
| `cross_entropy_fused` | Loss / Optimizer / Misc | liger-kernel |
| `fused_linear_cross_entropy` | Loss / Optimizer / Misc | cut-cross-entropy |
| `z_loss` | Loss / Optimizer / Misc | liger-kernel |
| `kl_divergence` | Loss / Optimizer / Misc | liger-kernel |
| `jsd_distillation` | Loss / Optimizer / Misc | liger-kernel |
| `tvd_loss` | Loss / Optimizer / Misc | liger-kernel |
| `dpo_loss` | Loss / Optimizer / Misc | liger-kernel |
| `orpo_loss` | Loss / Optimizer / Misc | liger-kernel |
| `preference_losses_simpo_cpo_kto_grpo` | Loss / Optimizer / Misc | liger-kernel |
| `adamw_fused` | Loss / Optimizer / Misc | apex |
| `adamw_8bit` | Loss / Optimizer / Misc | bitsandbytes |
| `lion_optimizer` | Loss / Optimizer / Misc | bitsandbytes |
| `adafactor_optimizer` | Loss / Optimizer / Misc | transformers |
| `muon_optimizer` | Loss / Optimizer / Misc | Moonlight (MoonshotAI scalable Muon) |
| `sgd_momentum_fused` | Loss / Optimizer / Misc | apex |
| `global_grad_norm_clip` | Loss / Optimizer / Misc | apex |
| `embedding_lookup` | Loss / Optimizer / Misc | torch |
| `embedding_backward_scatter` | Loss / Optimizer / Misc | torch (autograd) |
| `dtype_cast` | Loss / Optimizer / Misc | torch |
| `axpby_fused` | Loss / Optimizer / Misc | torch |
| `kv_cache_reshape_and_cache` | Loss / Optimizer / Misc | vllm |
| `kv_cache_copy_swap_blocks` | Loss / Optimizer / Misc | vllm |
| `fp8_convert_quantize` | Loss / Optimizer / Misc | vllm |

#### Rank-1 library frequency in the default stack

| Library | # rank-1 ops |
| --- | ---: |
| flashinfer-python | 25 |
| vllm | 16 |
| torch | 11 |
| flash-linear-attention (fla-core) | 10 |
| flash-attn | 9 |
| liger-kernel | 9 |
| apex | 6 |
| DeepGEMM | 5 |
| bitsandbytes | 4 |
| sgl-kernel | 4 |
| torchao | 4 |
| DeepEP | 3 |
| flashinfer | 2 |
| mamba-ssm | 2 |
| transformers | 2 |
| Moonlight (MoonshotAI scalable Muon) | 1 |
| ThunderKittens | 1 |
| autoawq | 1 |
| causal-conv1d | 1 |
| cut-cross-entropy | 1 |
| deep_gemm | 1 |
| flash-mla (FlashMLA) | 1 |
| ring-flash-attn | 1 |
| sageattention (SageAttention2++) | 1 |
| torch (autograd) | 1 |
| transformer-engine | 1 |
| vllm (Machete W4A8) | 1 |
| vllm (Marlin / GPTQ-Marlin) | 1 |
| vllm (awq_marlin) | 1 |
| xgrammar | 1 |

---

## Attention — `attention`

20 operators, 42 providers.

### `prefill_varlen_fwd`

Variable-length (ragged/packed) prefill attention forward over concatenated sequences using cu_seqlens; supports causal, GQA/MQA, sliding-window, ALiBi, softcap.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-attn | flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=True, window_size=(-1,-1), softcap=0.0, alibi_slopes=None, deterministic=False, block_table=None) ; q/k/v shape (total_tokens, nheads, headdim) | pip install flash-attn --no-build-isolation | sm80+ (Ampere/Ada/Hopper/Blackwell); FP8 path needs Hopper | fp16, bf16 | Industry-standard exact attention; near-roofline on A100/H100; varlen avoids padding waste. v2.8.3 (Jan 2026). _(confidence: high)_ | [link](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/flash_attn/flash_attn_interface.py) |
| 2 | flashinfer-python | flashinfer.prefill.single_prefill_with_kv_cache(q, k, v, causal=True, kv_layout='NHD', pos_encoding_mode='NONE', logits_soft_cap=0.0, window_left=-1, sm_scale=None) ; for batched ragged use flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(workspace).plan(qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim_qk, ...) then .run(q,k,v) | pip install flashinfer-python | sm75+, sm80/sm90/sm100 tuned | fp16, bf16, fp8 (e4m3/e5m2) | NVIDIA-backed serving kernel lib; JIT-tuned, plan/run split amortizes scheduling; FA3-class on Hopper. v0.6.x (2026). _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.prefill.single_prefill_with_kv_cache.html) |
| 3 | flash-attn-4 (flash_attn.cute) | from flash_attn.cute import flash_attn_varlen_func; flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True) | pip install flash-attn-4 | hopper (sm90) and blackwell (sm100); CuTeDSL-based | fp16, bf16 (fp8 fwd on Hopper) | FA4 CuTeDSL rewrite (v4.0.0.beta, 2026); targets H100/B200 with async/low-precision pipelines. Newer, less battle-tested. _(confidence: medium)_ | [link](https://github.com/Dao-AILab/flash-attention) |
| 4 | sgl-kernel | from sgl_kernel import flash_attn_varlen_func; flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale=None, causal=True, window_size=(-1,-1), softcap=0.0, sinks=None) -> out. | pip install sgl-kernel | sm90+ (Hopper FA3) | fp16/bf16 (fp8 descale) | SGLang vendored FA3 variable-length prefill over cu_seqlens; the hard-op attention alignment target. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/flash_attn.py) |

### `prefill_dense_fwd`

Dense batched prefill/self-attention forward with shape (batch, seqlen, nheads, headdim); causal/GQA/MQA, sliding window, ALiBi, softcap.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-attn | flash_attn.flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True, window_size=(-1,-1), softcap=0.0, alibi_slopes=None, deterministic=False) ; q (B,Sq,Hq,D), k/v (B,Sk,Hkv,D) | pip install flash-attn --no-build-isolation | sm80+ (Ampere/Ada/Hopper/Blackwell) | fp16, bf16 | De-facto standard exact dense attention; GQA via broadcasting Hkv<Hq with no extra memory. _(confidence: high)_ | [link](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/flash_attn/flash_attn_interface.py) |
| 2 | sgl-kernel | from sgl_kernel import flash_attn_varlen_func; flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale=None, causal=False, window_size=(-1,-1), softcap=0.0) -> out. q/k/v (total_tokens, nheads, headdim). | pip install sgl-kernel | sm90+ (Hopper FA3); sm80/86/89 supported for the FA3 build per sgl-kernel | fp16/bf16 (fp8 descale path) | SGLang's vendored FlashAttention-3 (FA3) prefill; production attention in the SGLang serving stack. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/flash_attn.py) |
| 3 | torch (SDPA) | torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) ; select backend via `with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION): ...` | pip install torch (>=2.5; cuDNN backend GA in 2.7) | FLASH_ATTENTION sm80+; EFFICIENT_ATTENTION sm70+; CUDNN_ATTENTION sm90+ (Hopper) | fp16, bf16, fp32 (math); cuDNN backend adds fp8 on Hopper | Zero-dep, fuses FA2/xFormers/cuDNN backends; enable_gqa avoids KV expansion; default for HF/torch models. Context-parallel context manager in 2.7. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) |
| 4 | xformers | xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None) ; causal via attn_bias=xformers.ops.LowerTriangularMask(); q/k/v shape (B, M, H, K) | pip install xformers | sm70+ (Volta+); FlashAttention backend sm80+; ROCm CK on MI200/MI300 | fp16, bf16, fp32 | Rich attn_bias system (no mask materialization); dispatches to FA/CK/cutlass kernels; backbone of diffusion/SD stacks. v0.0.35. _(confidence: high)_ | [link](https://facebookresearch.github.io/xformers/components/ops.html) |

### `attention_backward`

Backward pass (dQ, dK, dV) of flash/exact attention for training; supports causal, varlen, GQA, dropout, ALiBi, softcap, deterministic mode.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-attn | Autograd: out = flash_attn.flash_attn_func(q, k, v, causal=True, deterministic=False); out.backward(grad). Varlen training: flash_attn.flash_attn_varlen_func(..., deterministic=True). Backward is invoked automatically by autograd. | pip install flash-attn --no-build-isolation | sm80+ (bwd); FA3 bwd fp16/bf16 on Hopper | fp16, bf16 | Memory-efficient O(N) backward; deterministic=True for reproducible grads; the standard training-time attention backward. _(confidence: high)_ | [link](https://github.com/Dao-AILab/flash-attention) |
| 2 | flash-attn-3 (flash_attn_interface) | import flash_attn_interface; out = flash_attn_interface.flash_attn_func(q, k, v, causal=True); out.backward(grad) # Hopper-optimized fwd+bwd | cd flash-attention/hopper && python setup.py install | hopper-only (sm90, H100/H800) | fp16, bf16 (fwd+bwd); fp8 fwd only | FA3 Hopper async/warp-specialized; up to ~1.5-2x FA2 fwd, faster bwd; note: lacks attention-sink backward (use FA2 for gpt-oss training). _(confidence: high)_ | [link](https://pytorch.org/blog/flashattention-3/) |
| 3 | torch (SDPA) | out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True); out.backward(grad) # autograd dispatches FLASH/EFFICIENT/CUDNN backward | pip install torch | sm80+ flash bwd; sm70+ efficient bwd | fp16, bf16, fp32 | Fully autograd-integrated training path; backend selectable; works with torch.compile and context parallel. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) |

### `decode_paged_attention`

Single-token (decode) attention against a paged KV cache (block_table) for batched LLM serving; GQA/MQA, sliding window, softcap, fp8 KV.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | w = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout='NHD'); w.plan(kv_indptr, kv_indices, kv_last_page_len, num_qo_heads, num_kv_heads, head_dim, page_size, pos_encoding_mode='NONE', logits_soft_cap=0.0, window_left=-1, data_type=...); o = w.run(q, paged_kv_cache) | pip install flashinfer-python | sm75+, tuned sm80/sm90/sm100 | fp16, bf16, fp8 KV cache | Purpose-built decode kernel with CUDAGraph-friendly plan/run; powers vLLM/SGLang serving; load-balanced split-K decode. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/decode.html) |
| 2 | flash-attn | flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None, cache_seqlens=None, cache_batch_idx=None, block_table=None, softmax_scale=None, causal=True, window_size=(-1,-1), softcap=0.0) ; q (B,Sq,Hq,D), paged cache via block_table + (num_blocks,page_size,Hkv,D) | pip install flash-attn --no-build-isolation | sm80+ | fp16, bf16 | Single fused call does in-place KV append + RoPE + paged decode; widely used in custom inference loops. _(confidence: high)_ | [link](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/flash_attn/flash_attn_interface.py) |
| 3 | flashinfer-python (TRT-LLM backend) | flashinfer.decode.trtllm_batch_decode_with_kv_cache(query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len, bmm1_scale, bmm2_scale, ...) ; TensorRT-LLM fused decode kernel | pip install flashinfer-python | sm90/sm100 (Hopper/Blackwell) | fp16, bf16, fp8 | Wraps NVIDIA TRT-LLM XQA decode kernels; best raw decode throughput on Hopper/Blackwell with fp8 KV. _(confidence: medium)_ | [link](https://docs.flashinfer.ai/api/decode.html) |
| 4 | sgl-kernel | from sgl_kernel import flash_attn_with_kvcache; flash_attn_with_kvcache(q, k_cache, v_cache, page_table=..., cache_seqlens=..., softmax_scale=None, causal=False) -> out. Paged KV-cache decode (FA3). k/v cache (num_blocks, page_size, nheads_k, headdim). | pip install sgl-kernel | sm90+ (Hopper FA3) | fp16/bf16 (fp8 KV descale) | SGLang vendored FA3 paged decode with KV cache; production decode path in SGLang. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/flash_attn.py) |

### `prefill_paged_attention`

Batched prefill/append attention where queries are ragged and K/V live in a paged KV cache (chunked-prefill, prefix caching); causal, GQA, softcap, sliding window.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | w = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout='NHD'); w.plan(qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_qo_heads, num_kv_heads, head_dim_qk, page_size, causal=True, logits_soft_cap=0.0, window_left=-1); o = w.run(q, paged_kv_cache) | pip install flashinfer-python | sm75+, tuned sm80/sm90/sm100 | fp16, bf16, fp8 KV | Core chunked-prefill + prefix-reuse kernel for vLLM/SGLang; cascade-capable; handles mixed batch query lengths. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/prefill.html) |
| 2 | vllm (flash-attn backend) | from vllm.vllm_flash_attn import flash_attn_varlen_func; flash_attn_varlen_func(q, k, v, cu_seqlens_q, ..., block_table=block_table, causal=True, softcap=...) ; vendored vllm-flash-attention with paged block_table | pip install vllm | sm80+ (CUDA); ROCm AITER on MI300 | fp16, bf16, fp8 KV | vLLM-tuned FA2/FA3 fork integrated with PagedAttention + reshape_and_cache; production serving default. _(confidence: medium)_ | [link](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/flash_attn/) |

### `single_decode_attention`

Single-request single-token decode against contiguous (non-paged) KV cache; GQA/MQA, sliding window, softcap.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.decode.single_decode_with_kv_cache(q, k, v, kv_layout='NHD', pos_encoding_mode='NONE', logits_soft_cap=0.0, window_left=-1, sm_scale=None) ; q (num_qo_heads, head_dim), k/v (kv_len, num_kv_heads, head_dim) | pip install flashinfer-python | sm75+, tuned sm80/sm90/sm100 | fp16, bf16, fp8 | Lowest-overhead single-sequence decode; split-K for long contexts; ideal for latency-critical single-stream. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/decode.html) |

### `mla_decode_paged`

DeepSeek Multi-head Latent Attention decode against paged compressed-latent KV cache (kv_lora_rank + rope), matrix-absorbed; head_dim_ckv=512, head_dim_kpe=64.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-mla (FlashMLA) | from flash_mla import get_mla_metadata, flash_mla_with_kvcache; tile_md, num_splits = get_mla_metadata(cache_seqlens, s_q*h_q//h_kv, h_kv); o, lse = flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v, tile_md, num_splits, softmax_scale=None, causal=True, is_fp8_kvcache=False) | git clone --recursive https://github.com/deepseek-ai/FlashMLA && cd FlashMLA && pip install -v . | hopper (sm90); sparse path sm100 | bf16, fp8 (e4m3) KV cache | DeepSeek's official MLA decode; up to 3000 GB/s mem-bound / 580 TFLOPS compute-bound on H800; the reference best-in-class MLA kernel. _(confidence: high)_ | [link](https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py) |
| 2 | sgl-kernel | from sgl_kernel import get_mla_metadata, flash_mla_with_kvcache; tile_md, num_splits = get_mla_metadata(cache_seqlens, num_heads_q//num_heads_k, num_heads_k); flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v, tile_md, num_splits, softmax_scale=None, causal=False) -> (out, lse). | pip install sgl-kernel | sm90 (Hopper); FlashMLA | bf16 (fp8 KV cache path) | SGLang's vendored FlashMLA absorbed-MLA paged decode (DeepSeek-V3); the hard-op MLA alignment target. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/flash_mla.py) |
| 3 | flashinfer-python | w = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace_buffer, backend='auto'); w.plan(qo_indptr, kv_indptr, kv_indices, kv_len_arr, num_heads, head_dim_ckv=512, head_dim_kpe=64, page_size, causal, sm_scale, q_data_type, kv_data_type); o = w.run(q_nope, q_pe, ckv_cache, kpe_cache, return_lse=False) | pip install flashinfer-python | sm90 (Hopper); sm100 paths | bf16, fp8 | Production MLA wrapper used by SGLang/vLLM; plan/run + CUDAGraph; separate ckv (512) and kpe (64) caches. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/mla.html) |
| 4 | flashinfer-python (TRT-LLM MLA) | flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(query, kv_cache, workspace_buffer, qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim, block_tables, seq_lens, max_seq_len, bmm1_scale, bmm2_scale) | pip install flashinfer-python | sm90/sm100 (Hopper/Blackwell) | bf16, fp8 | Wraps NVIDIA TRT-LLM MLA decode; top throughput for DeepSeek serving on Blackwell with fp8. _(confidence: medium)_ | [link](https://docs.flashinfer.ai/api/mla.html) |

### `mla_prefill`

DeepSeek MLA prefill/context attention (absorb or non-absorb form) for the prompt phase, computing over latent KV; head_dim 192 (128 nope + 64 rope).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | Use flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper with head_dim_qk=192, head_dim_vo=128 for MLA context phase; w.plan(qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim_qk=192, head_dim_vo=128, causal=True); w.run(q, k, v) | pip install flashinfer-python | sm80/sm90/sm100 | fp16, bf16, fp8 | Asymmetric head_dim (qk=192, vo=128) ragged prefill is the standard MLA context-phase path in SGLang/vLLM. _(confidence: medium)_ | [link](https://docs.flashinfer.ai/api/prefill.html) |
| 2 | flash-attn (MLA-style varlen) | flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True) with q head_dim 192 padded; used for MLA prefill when latent decompressed to full heads | pip install flash-attn --no-build-isolation | sm80+ | fp16, bf16 | Decompress-then-FA varlen prefill; simpler but more KV memory than absorbed MLA. Fallback path. _(confidence: medium)_ | [link](https://github.com/Dao-AILab/flash-attention) |

### `gqa_mqa_attention`

Grouped-query / multi-query attention where num_kv_heads < num_query_heads, broadcasting KV across query-head groups without materializing expanded KV.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-attn | flash_attn.flash_attn_func(q, k, v, causal=True) with q (B,S,Hq,D) and k/v (B,S,Hkv,D) where Hq % Hkv == 0 ; GQA/MQA handled natively by head broadcasting | pip install flash-attn --no-build-isolation | sm80+ | fp16, bf16 | Native GQA: no KV expansion, kernel iterates query-head groups; the standard for Llama/Qwen/Mistral. _(confidence: high)_ | [link](https://github.com/Dao-AILab/flash-attention) |
| 2 | torch (SDPA) | torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True) ; enable_gqa broadcasts KV heads to query heads inside the kernel | pip install torch (>=2.5) | sm80+ flash backend | fp16, bf16, fp32 | enable_gqa=True avoids manual repeat_interleave of KV; clean native GQA in pure PyTorch. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) |

### `sliding_window_local_attention`

Sliding-window / local attention restricting each query to a left/right token window (Mistral, gpt-oss SWA layers); combined with causal masking.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-attn | flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(left, right)) ; window_size=(W-1, 0) for causal sliding window of size W. Also in flash_attn_varlen_func and flash_attn_with_kvcache. | pip install flash-attn --no-build-isolation | sm80+ | fp16, bf16 | Window masking fused into kernel (skips out-of-window blocks); the reference SWA implementation. _(confidence: high)_ | [link](https://github.com/Dao-AILab/flash-attention) |
| 2 | flashinfer-python | flashinfer.prefill.single_prefill_with_kv_cache(q, k, v, causal=True, window_left=W) ; window_left sets local window; same arg on decode/prefill wrappers (.plan(window_left=...)) | pip install flashinfer-python | sm75+, tuned sm80/sm90/sm100 | fp16, bf16, fp8 | window_left integrated in serving wrappers for per-layer SWA (gpt-oss/Mistral) with paged KV. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/prefill.html) |

### `alibi_attention`

Attention with ALiBi linear positional bias (per-head slopes) added to scores before softmax; no positional embeddings needed.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-attn | flash_attn.flash_attn_func(q, k, v, causal=True, alibi_slopes=slopes) ; slopes shape (nheads,) or (batch, nheads). Also flash_attn_varlen_func(..., alibi_slopes=...) | pip install flash-attn --no-build-isolation | sm80+ | fp16, bf16 | ALiBi bias applied in-kernel without materializing bias matrix; standard for BLOOM/MPT/Baichuan. _(confidence: high)_ | [link](https://github.com/Dao-AILab/flash-attention) |
| 2 | xformers | xformers.ops.memory_efficient_attention(q, k, v, attn_bias=xformers.ops.fmha.attn_bias.LowerTriangularMaskWithTensorBias(alibi_bias)) ; or build ALiBi bias tensor as attn_bias | pip install xformers | sm70+/sm80+ | fp16, bf16 | Flexible tensor-bias path supports ALiBi via additive bias; useful when bias is non-standard. _(confidence: medium)_ | [link](https://facebookresearch.github.io/xformers/components/ops.html) |

### `logit_softcap_attention`

Attention with logit soft-capping: scores = softcap * tanh(scores/softcap) before softmax (Gemma2, Grok), bounding attention logits.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-attn | flash_attn.flash_attn_func(q, k, v, causal=True, softcap=50.0) ; softcap>0 enables tanh logit capping. Also in flash_attn_varlen_func / flash_attn_with_kvcache. | pip install flash-attn --no-build-isolation | sm80+ | fp16, bf16 | tanh softcap fused in-kernel (added in FA 2.6); the standard Gemma2/Grok attention path. _(confidence: high)_ | [link](https://github.com/Dao-AILab/flash-attention) |
| 2 | flashinfer-python | flashinfer.prefill.single_prefill_with_kv_cache(q, k, v, causal=True, logits_soft_cap=50.0) ; logits_soft_cap arg on all prefill/decode wrappers (.plan(logits_soft_cap=...)) | pip install flashinfer-python | sm75+, tuned sm80/sm90/sm100 | fp16, bf16, fp8 | logits_soft_cap available across serving wrappers for Gemma2 paged inference. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/prefill.html) |

### `attention_sink`

Attention with learnable per-head sink logits (StreamingLLM / gpt-oss): extra always-attended sink token(s) stabilize softmax under sliding window / streaming.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-attn | flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(W,0), sinks=sink_logits) # 'sinks'/attn-sink arg added for gpt-oss; per-head sink logits tensor of shape (nheads,). (vLLM passes equivalent 's_aux'.) | pip install flash-attn --no-build-isolation (>=2.8 for gpt-oss sinks) | sm80+ (FA2); FA3 lacks sink backward | fp16, bf16 | FA2 supports attention-sink fwd+bwd (needed for gpt-oss training); FA3 fwd only. Exact arg name evolving across 2.8.x. _(confidence: low)_ | [link](https://huggingface.co/openai/gpt-oss-120b/discussions/41) |
| 2 | flash-mla (FlashMLA) | flash_mla.flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v, tile_md, num_splits, causal=True, attn_sink=sink_tensor) # attn_sink arg for sink logits in MLA decode | git clone --recursive https://github.com/deepseek-ai/FlashMLA && pip install -v . | hopper (sm90) | bf16, fp8 | Native attn_sink parameter in MLA decode interface for sink-token models. _(confidence: medium)_ | [link](https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py) |

### `cross_attention`

Cross-attention where queries attend to a separate key/value sequence (encoder-decoder, multimodal); differing q and kv sequence lengths, no causal mask.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-attn | flash_attn.flash_attn_func(q, k, v, causal=False) with q (B,Sq,H,D) and k/v (B,Skv,H,D), Sq != Skv ; varlen cross via flash_attn_varlen_func with distinct cu_seqlens_q/cu_seqlens_k | pip install flash-attn --no-build-isolation | sm80+ | fp16, bf16 | Handles asymmetric q/kv lengths natively; standard for Whisper/T5/multimodal cross-attn. _(confidence: high)_ | [link](https://github.com/Dao-AILab/flash-attention) |
| 2 | xformers | xformers.ops.memory_efficient_attention(q, k, v, attn_bias=BlockDiagonalMask.from_seqlens(q_seqlens, kv_seqlens)) ; from_seqlens with separate q/kv lengths enables packed cross-attention | pip install xformers | sm70+/sm80+ | fp16, bf16 | BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen) gives efficient packed cross-attention without padding. _(confidence: high)_ | [link](https://facebookresearch.github.io/xformers/components/ops.html) |

### `append_kv_cache`

Reshape-and-cache / append: scatter newly computed K,V into a paged KV cache at the right block/slot positions (KV cache write path).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.page.append_paged_kv_cache(append_key, append_value, batch_indices, positions, paged_kv_cache, kv_indices, kv_indptr, kv_last_page_len, kv_layout='NHD') ; MLA variant: flashinfer.page.append_paged_mla_kv_cache(...) | pip install flashinfer-python | sm75+ | fp16, bf16, fp8 | Vectorized scatter into paged cache; pairs with FlashInfer prefill/decode wrappers; MLA-aware variant. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/page.html) |
| 2 | vllm | from vllm import _custom_ops as ops; ops.reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale) ; flash-layout KV cache write | pip install vllm | sm70+ (CUDA); ROCm AITER | fp16, bf16, fp8 KV | reshape_and_cache_flash writes KV in flash-attn block layout with fp8 scaling; vLLM's production cache-write op. _(confidence: medium)_ | [link](https://github.com/vllm-project/vllm) |

### `ring_context_parallel_attention`

Ring / context-parallel attention: shard the sequence across GPUs and pass KV blocks around a ring with online-softmax merge, enabling million-token contexts.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | ring-flash-attn | from ring_flash_attn import zigzag_ring_flash_attn_func; out = zigzag_ring_flash_attn_func(q, k, v, dropout_p=0.0, causal=True, group=process_group) ; also ring_flash_attn_func, llama3_flash_attn_varlen_func, zigzag_ring_flash_attn_varlen_func | pip install ring-flash-attn | sm80+ (uses flash-attn under the hood) | fp16, bf16 | Zigzag variant load-balances causal work across ranks; most-used open ring-attention impl; fwd+bwd for training. _(confidence: high)_ | [link](https://github.com/zhuzilin/ring-flash-attention) |
| 2 | torch (Context Parallel) | from torch.distributed.tensor.experimental import context_parallel; with context_parallel(device_mesh, buffers=[q,k,v], buffer_seq_dims=[2,2,2]): out = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True) | pip install torch (>=2.7) | sm80+ (flash/efficient/cuDNN backends) | fp16, bf16 | Native PyTorch ring/all-gather CP wrapping SDPA; works across flash/efficient/cuDNN backends with torch.compile. _(confidence: medium)_ | [link](https://pytorch.org/blog/pytorch-2-7/) |

### `cascade_attention`

Cascade / multi-level shared-prefix attention: compute attention against a shared prefix once and per-request suffixes separately, then merge states (online softmax).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.cascade.MultiLevelCascadeAttentionWrapper(num_levels, workspace_buffer, kv_layout='NHD'); .plan(qo_indptr_arr, paged_kv_indptr_arr, paged_kv_indices_arr, paged_kv_last_page_len, num_qo_heads, num_kv_heads, head_dim, page_size, causal=True); .run(q, paged_kv_cache). Low-level merge: flashinfer.cascade.merge_state(v_a, s_a, v_b, s_b) | pip install flashinfer-python | sm80/sm90/sm100 | fp16, bf16, fp8 | Shared-prefix cascade massively cuts KV reads for many-request common prompts (system prompts, few-shot); merge_state via online softmax. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/cascade.html) |

### `tree_speculative_attention`

Tree / cascade attention for speculative decoding: verify a token tree (custom attention mask over draft branches) in one batched attention call.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | w = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(workspace); w.plan(qo_indptr, ..., custom_mask=tree_mask) ; pass packed custom_mask (token-tree attention mask) for speculative verification, then w.run(q, paged_kv_cache) | pip install flashinfer-python | sm80/sm90/sm100 | fp16, bf16, fp8 | custom_mask supports arbitrary tree masks for EAGLE/Medusa speculative verification in a single kernel; used by SGLang/vLLM spec decode. _(confidence: medium)_ | [link](https://docs.flashinfer.ai/api/prefill.html) |

### `quantized_low_precision_attention`

Quantized attention (INT8/FP8/FP4 QK and PV) for 2-5x speedup over FA2 with near-lossless quality; training-free, drop-in for prefill of LLMs/diffusion/video.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | sageattention (SageAttention2++) | from sageattention import sageattn; o = sageattn(q, k, v, tensor_layout='HND', is_causal=False, sm_scale=None) # auto-selects best kernel. Explicit: sageattn_qk_int8_pv_fp8_cuda(q,k,v,tensor_layout='HND',is_causal=True), sageattn_qk_int8_pv_fp8_cuda_sm90(...) for Hopper, sageattn_qk_int8_pv_fp16_cuda(...) | pip install sageattention==2.2.0 --no-build-isolation | sm80/sm86/sm89/sm90/sm120 (Ampere..Blackwell RTX5090) | int8 (QK^T) + fp8 e4m3 (PV); fp16/bf16 inputs | 2-3x over FlashAttention2 with near-lossless end-to-end quality; INT8 QK + FP8 PV; dominant quantized-attention lib for diffusion/video/LLM prefill. _(confidence: high)_ | [link](https://github.com/thu-ml/SageAttention) |
| 2 | sageattention (SageAttention3) | from sageattention import sageattn_blackwell # FP4 microscaling attention (sageattention3_blackwell module); sageattn_blackwell(q, k, v, tensor_layout='HND', is_causal=False) | pip install sageattention (build sageattention3_blackwell); requires python>=3.13, torch>=2.8, CUDA>=12.8 | blackwell (sm100/sm120), CUDA 12.8+ | fp4 microscaling (mxfp4/nvfp4) | SageAttention3 microscaling-FP4 attention for ~5x speedups on Blackwell; recommend SA2 for precision-sensitive use. _(confidence: medium)_ | [link](https://huggingface.co/jt-zhang/SageAttention3) |
| 3 | flash-attn-3 (FP8) | import flash_attn_interface; o = flash_attn_interface.flash_attn_func(q_fp8, k_fp8, v_fp8, causal=True) # FP8 forward on Hopper (e4m3) with descale handling | cd flash-attention/hopper && python setup.py install | hopper-only (sm90) | fp8 e4m3 (fwd) | FA3 FP8 forward up to ~1.2 PFLOPS on H100; exact-ish low-precision attention for inference prefill. _(confidence: medium)_ | [link](https://pytorch.org/blog/flashattention-3/) |

### `tile_primitive_attention`

Hand-written tile-primitive attention kernels (forward/prefill) for research-grade speed and custom variants on Hopper/Blackwell.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | ThunderKittens | Compile kernels/attention then call from PyTorch via the generated extension, e.g. import tk_attn; o = tk_attn.attention_forward(q, k, v, causal) # bind per-kernel; no top-level pip package (compile individually) | git clone https://github.com/HazyResearch/ThunderKittens && build kernels/ with PyTorch 2.8+ and pybind11 | sm90 (Hopper) and sm100 (Blackwell); MXFP8/NVFP4 in TK 2.0 | fp16, bf16, fp8, mxfp8, nvfp4 | TK 2.0 (2026) tile DSL hits FA3-class/faster on H100 and adds Blackwell + low-precision; for custom/research kernels, not a drop-in lib. _(confidence: medium)_ | [link](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2) |

---

## SSM / Linear Attention — `ssm-linear-attn`

13 operators, 17 providers.

### `mamba2_ssd_chunk_scan`

Mamba-2 State Space Duality (SSD) chunked selective scan. Computes y = SSM(A,B,C)(x) with scalar-per-head decay A, using the chunked/block-decomposition algorithm (intra-chunk diagonal + inter-chunk state recurrence). This is the core training/prefill kernel of Mamba-2 and the canonical efficient SSM op. Forward + backward (autograd).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | mamba-ssm | from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; y, final_state = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=256, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0,float('inf')), return_final_states=True). Shapes: x [B,L,nheads,headdim], dt [B,L,nheads], A [nheads], B/C [B,L,ngroups,dstate], D [nheads] or [nheads,headdim]. | pip install mamba-ssm --no-build-isolation (or: pip install mamba-ssm[causal-conv1d]) | sm70+ (V100/A100/H100); Triton path. CUDA selective-scan extension targets sm70+; best on sm80+/sm90. | fp32/fp16/bf16 (inputs fp16/bf16, accumulation fp32 internally) | Reference Mamba-2 SSD kernel from the original authors (Tri Dao / Albert Gu). Triton chunk-scan is the de-facto fastest open SSM training kernel; pairs with causal-conv1d. v2.3.2.post1 (May 2026). _(confidence: high)_ | [link](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_combined.py) |
| 2 | flash-linear-attention (fla-core) | from fla.ops.simple_gla import chunk_simple_gla; o, final_state = chunk_simple_gla(q, k, v, g=None, g_gamma=None, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None). simple_gla == Mamba-2/Gated-RetNet SSD with head-wise scalar decay g [B,T,H] (or g_gamma [H] log-decay). Layout [B,T,H,K]/[B,T,H,V]. head_first removed. | pip install flash-linear-attention (or: pip install fla-core) | sm80+ recommended (Triton); verified on NVIDIA/AMD/Intel. Needs Triton>=3.3 (mainline asks 3.5). | fp16/bf16 (fp32 accumulation) | Triton-only, no CUDA build. simple_gla is the SSD-equivalent kernel; often matches/beats mamba-ssm Triton path and supports varlen (cu_seqlens). v0.5.0 (Apr 2026). _(confidence: high)_ | [link](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/simple_gla/chunk.py) |

### `mamba1_selective_scan`

Mamba-1 selective scan: y = selective_scan(u, delta, A, B, C, D, z) — input-dependent (selective) SSM with full per-channel state-space matrices and optional gated output z (SiLU). Classic hand-written CUDA kernel, forward+backward.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | mamba-ssm | from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; out = selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False). Also mamba_inner_fn(xz, conv1d_weight, ...) fuses conv+proj+scan. Shapes: u [B,D,L], delta [B,D,L], A [D,N], B/C [B,N,L] or [B,G,N,L]. | pip install mamba-ssm --no-build-isolation | sm70+ (custom CUDA selective_scan extension); best sm80+ | fp32/fp16/bf16 (u/delta cast to fp32 internally; complex B/C variant supported) | Original Mamba-1 CUDA kernel by Gu & Dao. Memory-efficient recompute in backward; the canonical selective-scan reference. mamba_inner_fn is the fully-fused training path. _(confidence: high)_ | [link](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py) |

### `causal_conv1d`

Short causal depthwise (per-channel) 1D convolution with optional SiLU/swish activation, used as the token-mixing conv inside Mamba/Mamba-2 blocks. Forward (full sequence) + single-step update against a conv_state ring buffer for autoregressive decode. Forward+backward.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | causal-conv1d | from causal_conv1d import causal_conv1d_fn, causal_conv1d_update; y = causal_conv1d_fn(x, weight, bias=None, seq_idx=None, initial_states=None, return_final_states=False, final_states_out=None, activation='silu'); y = causal_conv1d_update(x, conv_state, weight, bias=None, activation='silu', cache_seqlens=None, conv_state_indices=None). Shapes: x [B,dim,L], weight [dim,width], conv_state [B,dim,state_len]. | pip install causal-conv1d>=1.4.0 --no-build-isolation | sm70+ (custom CUDA); width typically 2-4 | fp32/fp16/bf16 | Dao-AILab reference depthwise causal conv used by every Mamba impl. Fused activation + state update; far faster than nn.Conv1d for width<=4. v1.6.2.post1 (May 2026). _(confidence: high)_ | [link](https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py) |

### `gated_linear_attention`

Gated Linear Attention (GLA): linear attention with data-dependent per-dimension forget gates g (log-decay applied to keys/state). Chunk-parallel training kernel + fused-recurrent decode kernel. Forward+backward.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-linear-attention (fla-core) | from fla.ops.gla import chunk_gla, fused_recurrent_gla; o, final_state = chunk_gla(q, k, v, g, scale=None, initial_state=None, output_final_state=False, state_v_first=False, cu_seqlens=None). Decode: fused_recurrent_gla(q,k,v,g,...). Layout [B,T,H,K]/[B,T,H,V]; g is log-decay [B,T,H,K]. | pip install flash-linear-attention | sm80+ recommended (Triton) | fp16/bf16 (fp32 accumulation) | Reference GLA impl from the GLA paper authors (Yang et al). Secondary-chunking; the canonical hardware-efficient GLA. Supports varlen via cu_seqlens. v0.5.0. _(confidence: high)_ | [link](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py) |

### `delta_rule`

DeltaNet linear attention with the delta update rule: S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t k_t v_t^T (delta-rule memory write). Chunk-parallel (WY representation) training kernel + fused-recurrent decode. Forward+backward.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-linear-attention (fla-core) | from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule; o, final_state = chunk_delta_rule(q, k, v, beta, scale=None, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None). Layout [B,T,H,K]/[B,T,H,V]; beta [B,T,H] write strength. | pip install flash-linear-attention | sm80+ recommended (Triton) | fp16/bf16 (fp32 accumulation) | Canonical parallel DeltaNet kernel (Yang et al, 'Parallelizing Linear Transformers with the Delta Rule'). WY-representation chunking enables parallel training of the delta rule. Built-in optional QK L2-norm. _(confidence: high)_ | [link](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/delta_rule/chunk.py) |

### `gated_delta_rule`

Gated DeltaNet (a.k.a. GatedDeltaNet / GDN, the Qwen3-Next / Mamba-2+delta hybrid): delta-rule memory write combined with a scalar/head-wise decay gate g (and optional A_log/dt_bias Mamba-style gating). Chunk-parallel training + fused-recurrent decode. Forward+backward.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-linear-attention (fla-core) | from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule; o, final_state = chunk_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None). Decode (Mamba-style gating supported): fused_recurrent_gated_delta_rule(q,k,v,g=..,beta=..,A_log=..,dt_bias=..,use_gate_in_kernel=True,...). Layout q/k [B,T,H,K], v [B,T,HV,V] (GVA when HV>H). | pip install flash-linear-attention | sm80+ recommended (Triton) | fp16/bf16 (fp32 accumulation) | Reference Gated DeltaNet kernel (Yang et al, NeurIPS'24). Powers Qwen3-Next / Kimi-linear style hybrids; alias chunk_gdn/fused_recurrent_gdn also exported. Best-in-class for gated delta hybrids. _(confidence: high)_ | [link](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py) |

### `retention_retnet`

RetNet multi-scale retention: linear attention with fixed (data-independent) per-head exponential decay. Parallel, chunk-recurrent, and fused-recurrent forms. Forward+backward.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-linear-attention (fla-core) | from fla.ops.retention import chunk_retention, fused_recurrent_retention, parallel_retention; o, final_state = chunk_retention(q, k, v, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None). Per-head decay derived internally from head index. Layout [B,T,H,K]/[B,T,H,V]. | pip install flash-linear-attention | sm80+ recommended (Triton) | fp16/bf16 (fp32 accumulation) | Canonical RetNet retention kernel; chunk form for training, fused_recurrent for O(1) decode, parallel for short seqs. Maintained by FLA team. _(confidence: high)_ | [link](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/retention/chunk.py) |

### `rwkv6_wkv`

RWKV-6 (Finch) WKV linear-attention operator: receptance-weighted key-value with per-channel data-dependent decay w and bonus u. Chunk-parallel training + fused-recurrent decode. Forward+backward.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-linear-attention (fla-core) | from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6; o, final_state = chunk_rwkv6(r, k, v, w, u, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None). r/k/w [B,T,H,K], v [B,T,H,V], u [H,K] (bonus), w = log-decay. | pip install flash-linear-attention | sm80+ recommended (Triton) | fp16/bf16 (fp32 accumulation) | Triton chunk-parallel RWKV-6; trains far faster than the recurrent-only path and is portable (NVIDIA/AMD/Intel). Used by the RWKV community for training. _(confidence: high)_ | [link](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv6/chunk.py) |
| 2 | RWKV-CUDA / RWKV-LM (official) | torch.utils.cpp_extension.load(name='wkv6', sources=['cuda/wkv6_op.cpp','cuda/wkv6_cuda.cu'], ...); then a custom autograd WKV_6.apply(B,T,C,H, r,k,v,w,u). See RWKV-LM/RWKV-v6/src/model.py for the wrapper. Args: B,T,C,H ints + bf16 tensors r,k,v,w,u. | git clone https://github.com/BlinkDL/RWKV-LM (JIT-compiled via torch cpp_extension at runtime; needs nvcc + CUDA toolkit) | sm70+ (hand-written CUDA; bf16 tensor ops favor sm80+) | bf16 (fp32 state accumulation) | Author (BlinkDL) reference CUDA kernel — ground-truth numerics for RWKV-6 training. No pip package; JIT-compiled. Use FLA for portability/Triton. _(confidence: medium)_ | [link](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v6) |

### `rwkv7_wkv`

RWKV-7 (Goose) generalized-delta-rule WKV: state update with in-context learning-rate (vector-valued decay w and removal/addition keys a,b plus receptance r), i.e. a DPLR (diagonal-plus-low-rank) state transition. Chunk-parallel training + fused-recurrent decode. Forward+backward.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-linear-attention (fla-core) | from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7; o, final_state = chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=None, output_final_state=False, cu_seqlens=None). All [B,T,H,K] except v [B,T,H,V]; w = log-decay, a/b = remove/add vectors. (Equivalent to fla.ops.generalized_delta_rule chunk_dplr_delta_rule.) | pip install flash-linear-attention | sm80+ recommended (Triton) | fp16/bf16 (fp32 accumulation) | Reference RWKV-7 'Goose' kernel; DPLR delta-rule formulation, chunk-parallel for training. Co-developed with the RWKV-7 release; portable Triton. _(confidence: high)_ | [link](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv7/chunk.py) |
| 2 | RWKV-CUDA / RWKV-LM (official) | torch.utils.cpp_extension.load(name='wind_backstepping', sources=['cuda/wkv7_op.cpp','cuda/wkv7.cu'], ...); custom autograd WKV_7.apply(r,w,k,v,a,b). Inference forward: cuda_forward(int B,int T,int C,int H, bf16* r,w,k,v,a,b, bf16* y). See RWKV-LM/RWKV-v7/ and RWKV-CUDA/rwkv7_fast_fused. | git clone https://github.com/BlinkDL/RWKV-LM (JIT cpp_extension; needs nvcc/CUDA toolkit) | sm70+ (hand-written CUDA; bf16) | bf16 (fp32 state accumulation) | Author reference CUDA ('wind_backstepping' chunked training kernel) — ground-truth RWKV-7 numerics. rwkv7_fast_fused offers vanilla/state-tuning/infctx variants. No pip; JIT-compiled. _(confidence: medium)_ | [link](https://github.com/BlinkDL/RWKV-CUDA/tree/main/rwkv7_fast_fused) |

### `lightning_attention`

Lightning Attention (MiniMax-01/M1): linear attention with TransNormer-style fixed per-head exponential decay (slope), computed with a left-product/right-product block decomposition to avoid the cumsum bottleneck. Chunk-parallel training + fused-recurrent decode. Forward+backward.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-linear-attention (fla-core) | from fla.ops.lightning_attn import chunk_lightning_attn, fused_recurrent_lightning_attn; o, final_state = chunk_lightning_attn(q, k, v, layer_idx, num_layers, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None). Per-head decay g_gamma is derived internally from layer_idx/num_layers and head index (TransNormer slopes). Layout [B,T,H,K]/[B,T,H,V]. | pip install flash-linear-attention | sm80+ recommended (Triton) | fp16/bf16 (fp32 accumulation) | Portable Triton Lightning-Attention with built-in layer-dependent slope schedule; integrated, tested, and varlen-capable inside FLA. Easiest correct path for MiniMax-style models. _(confidence: high)_ | [link](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/lightning_attn/chunk.py) |
| 2 | lightning-attention (MiniMax/OpenNLPLab official) | from lightning_attn.ops import lightning_attn_func; from lightning_attn.utils import _build_slope_tensor; s = _build_slope_tensor(num_heads).to(q); o = lightning_attn_func(q, k, v, s). q/k/v [B,H,T,D]; s = per-head slope tensor [H,1,1]. | git clone https://github.com/OpenNLPLab/lightning-attention; pip install -e . (Triton-based) | sm80+ (Triton); developed/tested on A100 | fp16/bf16 | Original Lightning-Attention-2 reference from the MiniMax-01/TransNormer authors (Qin et al). Linear time/constant memory in seqlen; ground-truth for the algorithm. Lower-level API (explicit slope tensor). _(confidence: medium)_ | [link](https://github.com/OpenNLPLab/lightning-attention) |

### `linear_attention_basic`

Plain (ungated) linear attention / linear transformer: o = (phi(q) (phi(k)^T v)) with causal cumulative state, optional feature map and normalization. Chunk-parallel training + fused-recurrent decode. The baseline kernel of the family. Forward+backward.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-linear-attention (fla-core) | from fla.ops.linear_attn import chunk_linear_attn, fused_recurrent_linear_attn; o, final_state = chunk_linear_attn(q, k, v, scale=None, initial_state=None, output_final_state=False, normalize=True, cu_seqlens=None). Layout [B,T,H,K]/[B,T,H,V]. | pip install flash-linear-attention | sm80+ recommended (Triton) | fp16/bf16 (fp32 accumulation) | Reference chunked linear-attention; the shared building block other FLA ops specialize. Apply feature map (e.g. elu+1) to q/k before calling. _(confidence: high)_ | [link](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/linear_attn/chunk.py) |

### `gated_slot_attention_gsa`

Gated Slot Attention (GSA): bounded-memory linear attention via a fixed set of slots with ABC-style gating; two-pass softmax over slots. Chunk-parallel training + fused-recurrent decode. Forward+backward.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-linear-attention (fla-core) | from fla.ops.gsa import chunk_gsa, fused_recurrent_gsa; o, final_state = chunk_gsa(q, k, v, s, g, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None). q/k [B,T,H,K], v [B,T,H,V], s/g slot states+gates of slot-dim M. | pip install flash-linear-attention | sm80+ recommended (Triton) | fp16/bf16 (fp32 accumulation) | Reference GSA kernel (Zhang et al). Bounded slot memory gives strong recall at constant decode cost; maintained by FLA. _(confidence: medium)_ | [link](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gsa/chunk.py) |

### `hgrn2`

HGRN2: hierarchically-gated linear RNN with state expansion (outer-product state, data-dependent forget gate as the only decay). Fused-recurrent decode + chunk path via simple_gla. Forward+backward.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flash-linear-attention (fla-core) | Decode: from fla.ops.hgrn import fused_recurrent_hgrn; o, final_state = fused_recurrent_hgrn(x, g, initial_state=None, output_final_state=False, cu_seqlens=None). Chunked HGRN2 training routes through fla.ops.simple_gla.chunk_simple_gla (see fla/layers/hgrn2.py). g = log forget-gate. | pip install flash-linear-attention | sm80+ recommended (Triton) | fp16/bf16 (fp32 accumulation) | Reference HGRN/HGRN2 kernels (Qin et al). HGRN2 layer uses chunk_simple_gla for hardware-efficient training; fused_recurrent_hgrn for O(1) decode. _(confidence: medium)_ | [link](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/hgrn2.py) |

---

## Dense GEMM — `gemm-dense`

6 operators, 16 providers.

### `gemm`

Plain dense GEMM: C = alpha * op(A) @ op(B) + beta * C, fp16/bf16/fp32/tf32, fp32 accumulation. Tensor-core mma path; tf32 used for the fp32 path on Ampere+.

**kernel_set_abi:** `ks_gemm`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torch | torch.matmul(a, b) # or a @ b ; for C=alpha*A@B+beta*C use torch.addmm(c, a, b, beta=beta, alpha=alpha). Dispatches to cuBLAS/cuBLASLt; transpose via a.t()/b.t() (views, no copy). For tf32 set torch.backends.cuda.matmul.allow_tf32=True (or torch.set_float32_matmul_precision('high')). | pip install torch (CUDA build, e.g. --index-url https://download.pytorch.org/whl/cu128) | sm70+ (tensor cores); tf32 sm80+; fp32/any on all CUDA | fp16/bf16/fp32/tf32 (fp64 supported, slow) | Default best-in-class for arbitrary fp16/bf16/fp32 GEMM. cuBLAS/cuBLASLt heuristically picks heavily NVIDIA-tuned kernels per shape/arch; zero-friction, always-correct baseline that other libs are benchmarked against. Wins for general non-fused dense matmul of any shape. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html) |
| 2 | nvidia-cutlass-dsl (CuTe DSL) | from CuTeDSL examples: examples/python/CuTeDSL/hopper/dense_gemm.py (Hopper WGMMA+TMA) / blackwell/dense_gemm.py. Author kernel with `import cutlass; import cutlass.cute as cute; @cute.jit / @cute.kernel`, JIT-compile and launch on device tensors. Provides C++-equivalent perf with Python authoring. | pip install nvidia-cutlass-dsl | sm80+ (Ampere/Hopper/Blackwell); arch-specific kernels (sm90 WGMMA, sm100 tcgen05) | fp16/bf16/tf32/fp8/fp4 + blockscaled | v4.5.2 (2026-05-25). Matches CUTLASS C++ perf on dense GEMM with orders-of-magnitude faster compile than C++ templates. Wins when you need a custom tiling/epilogue or a specialized shape that cuBLAS heuristics miss, with full control over tensor-core atoms. _(confidence: medium)_ | [link](https://pypi.org/project/nvidia-cutlass-dsl/) |
| 3 | nvidia-cutlass (cutlass_cppgen, legacy Python emitter) | import cutlass, numpy as np; plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor); plan.run(A, B, C, D) # emit & run a CUTLASS C++ GEMM from Python. NOTE: package renamed to cutlass_cppgen in 4.x and DEPRECATED in favor of the CuTe DSL. | pip install nvidia-cutlass | sm70+ through sm90/sm100 depending on tile/dtype | fp16/bf16/tf32/fp32/fp8/int8/int4 | v4.2.0.0 (2025-09-19). High-level emitter to instantiate & autotune CUTLASS C++ GEMMs and export them as PyTorch CUDA extensions. Deprecated for new work (use CuTe DSL) but still the simplest path to ship a compiled CUTLASS kernel into PyTorch. _(confidence: medium)_ | [link](https://pypi.org/project/nvidia-cutlass/) |

### `gemm_fp8`

FP8 dense GEMM with scaling: C = (scale_a (x) scale_b) * (A_fp8 @ B_fp8), output bf16/fp16 (optionally fp8). Per-tensor or fine-grained (1x128 / 128x128 block) scaling.

**kernel_set_abi:** `ks_gemm`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | DeepGEMM | import deep_gemm; deep_gemm.fp8_gemm_nt((A_fp8, A_scale), (B_fp8, B_scale), out) # NT layout (A non-transposed, B transposed). Variants fp8_gemm_{nt,nn,tn,tt}. LHS scale must be TMA-aligned/transposed; SM90 scales fp32, SM100 packed UE8M0. NOTE: 2026 API renamed from gemm_fp8_fp8_bf16_nt to fp8_gemm_nt. | git clone --recursive https://github.com/deepseek-ai/DeepGEMM && cd DeepGEMM && ./install.sh (JIT, no kernels precompiled) | Hopper sm90 + Blackwell sm100 only | fp8 (e4m3) in, bf16 out; fp8xfp4 supported on Blackwell | DeepSeek-V3's production FP8 GEMM. Fine-grained (1x128/128x128) scaling, fully JIT, matches or beats expert-tuned cuBLASLt across LLM shapes on Hopper/Blackwell. Best for FP8 LLM training/inference dense projections. _(confidence: high)_ | [link](https://github.com/deepseek-ai/DeepGEMM) |
| 2 | torch | torch._scaled_mm(a_fp8, b_fp8, scale_a=sa, scale_b=sb, bias=None, out_dtype=torch.bfloat16, use_fast_accum=True) # a row-major fp8, b column-major fp8; scales are fp32 tensors (per-tensor [] or rowwise). Returns the matmul result. | pip install torch (CUDA 12.x build) | sm89 (Ada) + sm90 (Hopper) + sm100 (Blackwell) for fp8 tensor cores | fp8 e4m3/e5m2 in; bf16/fp16/fp32 out (or fp8 out with scale_result) | Native PyTorch FP8 GEMM over cuBLASLt; the standard fp8 primitive used by torchao/float8. Best when you want fp8 without an extra dependency; DeepGEMM edges it on fine-grained-scaled LLM shapes. Still a private (underscore) API, subject to change. _(confidence: high)_ | [link](https://gist.github.com/drisspg/783616821043ab4594b9784f556c6714) |
| 3 | TransformerEngine | import transformer_engine.pytorch as te; lin = te.Linear(in_features, out_features, bias=True); with te.fp8_autocast(enabled=True, fp8_recipe=recipe): y = lin(x) # manages fp8 scaling/amax, weight casting, and the cuBLASLt/TE fp8 GEMM internally. | pip install --no-build-isolation transformer_engine[pytorch] | Hopper/Ada (CC 8.9+ for fp8) + Blackwell; MXFP8/NVFP4 Blackwell-only | fp8 e4m3/e5m2, MXFP8, NVFP4 (Blackwell), bf16 | v2.15 (2026-05-13). Production fp8 layer with delayed/current scaling recipes, amax history, and fused cast; integrates the full fp8 training loop, not just the raw GEMM. Best when you want a drop-in fp8 nn.Linear replacement with recipe management. _(confidence: high)_ | [link](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html) |

### `gemm_bias_act`

Fused linear epilogue: D = act(alpha * A @ B^T + bias), bias broadcast [N], act in {none,relu,gelu}. The QKV/MLP projection epilogue fusion.

**kernel_set_abi:** `ks_gemm_bias_act`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torch | torch.nn.functional.linear(x, weight, bias) # = x @ weight.t() + bias; on CUDA fp16/bf16 routes to cuBLASLt which fuses the bias add (and ReLU/GELU) into the GEMM epilogue. Apply activation via F.relu/F.gelu; let torch.compile (Inductor) fuse act into the matmul epilogue. | pip install torch (CUDA 12.x build) | sm70+; epilogue fusion strongest on sm80+ cuBLASLt | fp16/bf16/fp32/tf32 | cuBLASLt epilogue fuses bias (+ optional GELU/ReLU) into the GEMM, avoiding an extra pass over D. Standard, dependency-free, correct. For full act fusion across shapes, torch.compile's Inductor selects/codegens fused mm+bias+act epilogues. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html) |
| 2 | torch.compile (Inductor) | f = torch.compile(lambda x: torch.nn.functional.gelu(torch.nn.functional.linear(x, w, b)), mode='max-autotune'); f(x) # Inductor autotunes ATen/cuBLASLt vs Triton/CUTLASS mm templates and fuses bias+activation into the epilogue. Enable CUTLASS backend via torch._inductor.config.max_autotune_gemm_backends='ATEN,TRITON,CUTLASS'. | pip install torch (CUDA 12.x build) | sm80+ (Triton/CUTLASS templates); sm90/sm100 for newer templates | fp16/bf16/fp32/tf32 (fp8 via inductor scaled mm) | max-autotune benchmarks multiple GEMM backends per shape and codegens a fused mm+bias+act epilogue, frequently beating eager F.linear on transformer MLP shapes. Best when shapes are static and you can pay a one-time autotune cost. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/torch.compiler.html) |
| 3 | TransformerEngine | import transformer_engine.pytorch as te; ln_lin = te.LayerNormLinear(hidden, 3*hidden, bias=True); with te.fp8_autocast(): qkv = ln_lin(x) # fuses LayerNorm+Linear(+bias); te.Linear fuses bias and supports fp8 epilogue. GELU MLP via te.LayerNormMLP. | pip install --no-build-isolation transformer_engine[pytorch] | Hopper/Ada/Blackwell (fp8 CC 8.9+); bf16 on Ampere+ | fp8 e4m3/e5m2, MXFP8, NVFP4, bf16 | v2.15. Fuses norm + projection + bias (and fp8 cast) into one module, reducing memory traffic for QKV/MLP. Best for fp8 transformer blocks where norm+linear fusion matters. _(confidence: high)_ | [link](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html) |

### `gemm_batched`

Batched / strided-batched GEMM: for each batch b, C[b] = alpha * op(A[b]) @ op(B[b]) + beta * C[b]. Uniform per-batch shapes.

**kernel_set_abi:** `ks_gemm_batched`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torch | torch.bmm(a, b) # a:[B,M,K], b:[B,K,N] -> [B,M,N]. For C=beta*C+alpha*(A@B) use torch.baddbmm(c, a, b, beta=beta, alpha=alpha). torch.matmul broadcasts leading batch dims automatically. Maps to cuBLAS strided-batched GEMM. | pip install torch (CUDA 12.x build) | sm70+ (tensor cores fp16/bf16); tf32 sm80+ | fp16/bf16/fp32/tf32 | Direct mapping to cuBLAS strided-batched GEMM; the canonical batched matmul (attention scores/context, batched projections). Always-correct, well-tuned baseline. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.baddbmm.html) |
| 2 | nvidia-cutlass-dsl (CuTe DSL) | examples/python/CuTeDSL/hopper/dense_gemm.py implements batched dense GEMM (C=A*B) over the L (batch) mode via TMA + WGMMA; author with cutlass.cute and JIT-launch. Batch is the third tensor mode in the CuTe layout. | pip install nvidia-cutlass-dsl | sm80+/sm90/sm100 | fp16/bf16/tf32/fp8 | C++-class performance with custom batched tiling; useful when many small/medium batched GEMMs need a fused or specialized schedule that cuBLAS strided-batched does not cover well. _(confidence: medium)_ | [link](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py) |

### `grouped_gemm`

Grouped GEMM: a list of independent GEMMs with DIFFERENT per-group M (and possibly N/K), packed contiguously with offsets (jagged) and run in one launch. Core MoE expert FFN op.

**kernel_set_abi:** `ks_moe_grouped_gemm`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | DeepGEMM | import deep_gemm; deep_gemm.m_grouped_bf16_gemm_nt_contiguous(A, B, out, m_indices) # contiguous (training/prefill, experts share shape). FP8: deep_gemm.m_grouped_fp8_gemm_nt_contiguous((A_fp8,A_s),(B_fp8,B_s),out,m_indices). Masked (decode + CUDA graph): m_grouped_fp8_gemm_nt_masked / m_grouped_bf16_gemm_nt_masked with masked_m. K-grouped (wgrad): k_grouped_*_contiguous. | git clone --recursive https://github.com/deepseek-ai/DeepGEMM && cd DeepGEMM && ./install.sh | Hopper sm90 + Blackwell sm100 | bf16 and fp8 (e4m3); fp8xfp4 mega-MoE on Blackwell | Best-in-class MoE grouped GEMM (DeepSeek-V3). Contiguous + masked layouts cover prefill and CUDA-graph decode; Mega-MoE overlaps compute/comm. JIT, fine-grained scaling. Top pick for fp8/bf16 expert FFNs on Hopper/Blackwell. _(confidence: high)_ | [link](https://github.com/deepseek-ai/DeepGEMM) |
| 2 | torch | torch.nn.functional.grouped_mm(mat_a, mat_b, offs=offs, bias=None, out_dtype=None) # offs: 1D int32 monotonically increasing group boundaries on the jagged dim; pass out_dtype=torch.float32 to accumulate bf16 in fp32. Now a public, non-differentiable API (was torch._grouped_mm). | pip install torch (recent 2.x CUDA build) | sm80+ for bf16 grouped_mm (sm90+ for some fp8 grouped paths) | bf16 (fp8 via torch._scaled_grouped_mm) | Native PyTorch grouped GEMM for MoE without an extra dependency; backed by CUTLASS grouped kernels and autotunable via Inductor. Best portable option (sm80+) when not on Hopper/Blackwell or when avoiding a DeepGEMM build. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grouped_mm.html) |
| 3 | grouped_gemm (CUTLASS bindings) | from grouped_gemm import ops; ops.gmm(a, b, batch_sizes, trans_b=False) # a packed [sum(m_i), K], b [num_groups, K, N], batch_sizes:int64 per-group rows. The classic MegaBlocks/Megatron MoE grouped GEMM. | pip install grouped_gemm (builds CUTLASS extension; or GROUPED_GEMM_CUTLASS=1) | sm80+ (Ampere/Hopper) | fp16/bf16 | Long-standing, widely deployed CUTLASS grouped GEMM used by Megatron-LM/MegaBlocks MoE. Stable and well-tested; superseded on Hopper/Blackwell by DeepGEMM and by native torch.grouped_mm, but a solid sm80 option. _(confidence: medium)_ | [link](https://github.com/tgale96/grouped_gemm) |

### `grouped_gemm_fp8`

FP8 grouped GEMM with per-group scaling for MoE expert FFNs (and weight-grad K-grouped). Different per-group M, fp8 inputs, bf16 output.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | DeepGEMM | import deep_gemm; deep_gemm.m_grouped_fp8_gemm_nt_contiguous((A_fp8,A_scale),(B_fp8,B_scale),out,m_indices) # contiguous MoE FFN; decode: m_grouped_fp8_gemm_nt_masked(..., masked_m, expected_m); wgrad: k_grouped_fp8_gemm_tn_contiguous. Fine-grained 1x128/128x128 scaling, NT layout. | git clone --recursive https://github.com/deepseek-ai/DeepGEMM && cd DeepGEMM && ./install.sh | Hopper sm90 + Blackwell sm100 | fp8 e4m3 in, bf16 out; fp8xfp4 on Blackwell | The reference fp8 MoE grouped GEMM from DeepSeek-V3; masked layout enables CUDA-graph decode. Best fp8 grouped GEMM on Hopper/Blackwell. _(confidence: high)_ | [link](https://github.com/deepseek-ai/DeepGEMM) |
| 2 | torch | torch._scaled_grouped_mm(a_fp8, b_fp8, scale_a, scale_b, offs=offs, bias=None, out_dtype=torch.bfloat16, use_fast_accum=True) # fp8 grouped GEMM with per-group offsets; private API, Inductor has a tune_scaled_grouped_mm autotuner path. | pip install torch (recent 2.x CUDA build) | sm90+ (Hopper/Blackwell) for fp8 grouped | fp8 e4m3/e5m2 in, bf16/fp16 out | Native PyTorch fp8 grouped GEMM (backed by CUTLASS), used by torchao MoE fp8 training. Best when staying in-framework; private/underscore API still stabilizing. _(confidence: medium)_ | [link](https://github.com/pytorch/pytorch/issues/166651) |

---

## Quantized GEMM — `gemm-quant`

17 operators, 41 providers.

### `fp8_gemm_blockwise`

FP8 (e4m3) GEMM with fine-grained block/group scaling (1x128 activations, 128x128 weights). D = A_fp8 @ B_fp8 with per-block fp32 dequant and two-level accumulation to fix FP8 tensor-core accumulation error. The DeepSeek-V3/R1 training+inference workhorse.

**kernel_set_abi:** `ks_gemm_w8a8 (closest; ks ABI has no native blockwise-fp8 path, this is fp8 not int8)`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | deep_gemm | deep_gemm.fp8_gemm_nt((a_fp8, a_scales), (b_fp8, b_scales), d) where each arg is a tuple (fp8_e4m3_tensor, fp32_scales); LHS row-major + scales transposed/TMA-aligned, RHS col-major; d is bf16 out. Layout variants: fp8_gemm_{nt,nn,tn,tt}. | pip install deep_gemm  (JIT-compiled at runtime, no nvcc needed at install; or ./install.sh from source) | sm90 (Hopper H100/H200/H800) and sm100 (Blackwell B200/B300); CUDA 12.3+ for SM90, 12.9+ for SM100 | fp8 e4m3 inputs, fp32 block scales (SM90) / packed UE8M0 (SM100), bf16 output | Best-in-class fine-grained FP8; near-cuBLAS/CUTLASS peak on Hopper with clean single-file kernels; two-level accumulation preserves accuracy. The reference impl for DeepSeek blockwise FP8. _(confidence: high)_ | [link](https://github.com/deepseek-ai/DeepGEMM) |
| 2 | flashinfer | flashinfer.gemm.group_deepgemm_fp8_nt_groupwise(...) for grouped blockwise; flashinfer.gemm.mm_fp8(a, b, a_scale, b_scale, out_dtype) for dense fp8. Wraps DeepGEMM/CUTLASS backends. | pip install flashinfer-python | sm90+ / sm100 (Blackwell); some paths sm75+ | fp8 e4m3/e5m2, fp32 scales, bf16/fp16 out | Production serving wrapper (used by SGLang); multi-backend (cutlass, trtllm, cudnn, deepgemm) with auto-selection. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/gemm.html) |
| 3 | transformer_engine | import transformer_engine.pytorch as te; layer = te.Linear(in, out); with te.fp8_autocast(enabled=True, fp8_recipe=recipe): y = layer(x). recipe from transformer_engine.common.recipe.DelayedScaling() or Float8CurrentScaling() or MXFP8BlockScaling(). | pip install transformer-engine[pytorch] | Hopper sm90, Ada sm89, Blackwell sm100/sm120 | fp8 e4m3 (fwd) / e5m2 (bwd), MXFP8 block, fp4 (Blackwell); bf16/fp16 master | NVIDIA's official FP8/FP4 training+inference module; integrates delayed/current scaling and MXFP8 block scaling recipes; fuses GEMM+epilogue. _(confidence: high)_ | [link](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) |
| 4 | sgl-kernel | from sgl_kernel import fp8_blockwise_scaled_mm; fp8_blockwise_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype) -> Tensor. mat_a (M,K) fp8, mat_b (K,N) fp8 col-major, blockwise fp32 scales. | pip install sgl-kernel | sm90+ (Hopper); CUTLASS blockwise fp8 | fp8 e4m3 with 1x128/128x128 block scales | SGLang CUTLASS blockwise-fp8 GEMM (DeepSeek-V3 fine-grained scaling); Hopper-class throughput; alignment target for kernel-set's blockwise-fp8 path. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/gemm.py) |

### `fp8_gemm_scaled_mm`

Per-tensor / per-token FP8 (e4m3) scaled GEMM: out = (scale_a * scale_b) * (A_fp8 @ B_fp8) + bias. The standard W8A8-FP8 linear used for FP8 inference checkpoints.

**kernel_set_abi:** `ks_gemm_w8a8 (analogous scaled-mm shape; ks ABI is int8 but same scale_a[M]/scale_b[N] structure)`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm import _custom_ops as ops; ops.cutlass_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None). a_fp8 [M,K], b_fp8 [K,N] col-major, scale_a per-token[M]/per-tensor[1], scale_b per-channel[N]/[1]. | pip install vllm | sm89 (Ada) / sm90 (Hopper) / sm100 (Blackwell) via CUTLASS; FP8-only on Blackwell (no int8 CUTLASS there) | fp8 e4m3 in, fp32 scales, bf16/fp16 out | Hand-tuned CUTLASS scaled-mm; the production FP8 GEMM in vLLM serving. Falls back to torch._scaled_mm where CUTLASS unsupported. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |
| 2 | sgl-kernel | from sgl_kernel import fp8_scaled_mm; fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None) -> Tensor. mat_a (M,K) fp8, mat_b (K,N) fp8 col-major, per-tensor/row scales. | pip install sgl-kernel | sm90+ (Hopper); CUTLASS fp8 | fp8 e4m3; scales fp32; out bf16/fp16 | SGLang CUTLASS fp8 scaled-mm with the same scale_a[M]/scale_b[N] structure as kernel-set's ks_gemm_w8a8; alignment target. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/gemm.py) |
| 3 | torch | torch._scaled_mm(a_fp8, b_fp8, scale_a=sa, scale_b=sb, bias=bias, out_dtype=torch.bfloat16). a row-major fp8_e4m3fn, b column-major fp8_e4m3fn; scalar or rowwise scales. | pip install torch  (CUDA build) | sm89/sm90/sm100; e4m3fn only on most paths | fp8 e4m3fn (e5m2 limited), fp32 scales, bf16/fp16 out | PyTorch-native cuBLASLt FP8 GEMM; the portable baseline that vLLM/torchao build on. Per-row scaling supported on Hopper+. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch._scaled_mm.html) |
| 4 | torchao | from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig; quantize_(model, Float8DynamicActivationFloat8WeightConfig()). Rewrites nn.Linear weights to fp8 + dynamic per-token act quant; runtime calls torch._scaled_mm/CUTLASS. | pip install torchao | Hopper sm90+ (and Blackwell sm100) | fp8 e4m3 weight+act dynamic, bf16 compute | PyTorch-native, torch.compile-friendly; clean module-swap API for FP8 W8A8 training and inference. _(confidence: high)_ | [link](https://docs.pytorch.org/ao/stable/generated/torchao.quantization.Float8DynamicActivationFloat8WeightConfig.html) |

### `quantize_fp8_dynamic`

Dynamic FP8 (e4m3) quantization of activations: computes scale (per-tensor or per-token) and writes fp8 tensor + fp32 scale. Pairs with fp8 scaled GEMM.

**kernel_set_abi:** `ks_quantize_fp8`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm import _custom_ops as ops; q, scale = ops.scaled_fp8_quant(input, scale=None, use_per_token_if_dynamic=True, scale_ub=None, group_shape=None). Returns (fp8_e4m3 tensor, fp32 scale). | pip install vllm | sm89+/sm90/sm100 | in fp16/bf16 -> fp8 e4m3, fp32 scale (per-tensor / per-token / group) | Fused dynamic quant kernel used directly before cutlass_scaled_mm in vLLM FP8 path; supports per-token, scale upper-bound clamp, and group-shape (blockwise). _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |
| 2 | flashinfer | q, s = flashinfer.fp8_quantization helpers / flashinfer.gemm pre-quant; commonly torch native used. (vLLM scaled_fp8_quant preferred.) | pip install flashinfer-python | sm90+/sm100 | fp16/bf16 -> fp8 e4m3, fp32 scale | Serving-oriented quant kernels bundled with FlashInfer GEMM; exact symbol varies by version. _(confidence: low)_ | [link](https://docs.flashinfer.ai/) |
| 3 | deep_gemm | deep_gemm.testing.per_token_cast_to_fp8(x) and per_block_cast_to_fp8(x) produce (fp8_tensor, scales) for blockwise GEMM input prep; get_col_major_tma_aligned_tensor aligns LHS scales. | pip install deep_gemm | sm90 / sm100 | fp16/bf16 -> fp8 e4m3, fp32 1x128 / 128x128 block scales | Reference blockwise (1x128 / 128x128) casting helpers matching DeepGEMM scaling layout; helper location may move between releases. _(confidence: medium)_ | [link](https://github.com/deepseek-ai/DeepGEMM) |

### `dequantize_fp8`

Dequantize fp8 (e4m3/e5m2) tensor back to fp16/bf16 using per-tensor/per-token/block scale.

**kernel_set_abi:** `ks_dequantize_fp8`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torchao | torchao Float8Tensor.dequantize() / underlying torch ops; or x_fp8.to(torch.bfloat16) * scale for simple per-tensor. torchao.float8 utilities handle scaled dequant. | pip install torchao | sm89+ | fp8 e4m3/e5m2 -> fp16/bf16 | Tensor-subclass dequant integrated with torch.compile; correct OCP fp8 scaling semantics. _(confidence: medium)_ | [link](https://github.com/pytorch/ao/tree/main/torchao/float8) |
| 2 | torch | out = input_fp8.to(torch.bfloat16) * scale (per-tensor); for per-token broadcast scale[:, None]. Native cast. | pip install torch | sm89+ | fp8 e4m3fn/e5m2 -> fp16/bf16 | Hardware fp8->fp16 cast is exact and fast; multiply by fp32 scale. Simplest portable dequant. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/tensors.html) |

### `int8_gemm_w8a8`

INT8 W8A8 GEMM: out = (a_scale (x) b_scale) * (A_i8 @ B_i8) + bias, with per-token activation scale [M] and per-channel weight scale [N] (or per-tensor). SmoothQuant-style inference.

**kernel_set_abi:** `ks_gemm_w8a8`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm import _custom_ops as ops; ops.cutlass_scaled_mm(a_i8, b_i8, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None). Same scaled-mm entrypoint as fp8 but with int8 a/b; per-token scale_a[M], per-channel scale_b[N]. | pip install vllm | sm75+ / sm80 / sm89 / sm90 (CUTLASS int8 mma); NOT on Blackwell sm100 CUTLASS (int8 path unsupported there) | int8 in, fp32 scales, bf16/fp16 out; optional asymmetric (azp) variant cutlass_scaled_mm_azp | Production CUTLASS int8 W8A8 GEMM (SmoothQuant/compressed-tensors). Symmetric + asymmetric (zero-point) kernels. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |
| 2 | sgl-kernel | from sgl_kernel import int8_scaled_mm; int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None) -> Tensor. mat_a (M,K) int8, mat_b (K,N) int8, scales_a (M,1)/scales_b (1,N) fp32. | pip install sgl-kernel | sm80+ (Ampere+); CUTLASS int8 tensor cores | int8 W8A8; scales fp32; out bf16/fp16 | SGLang CUTLASS int8 W8A8 scaled-mm; direct shape match for ks_gemm_w8a8; alignment target for kernel-set's int8 GEMM. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/gemm.py) |
| 3 | torchao | from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig; quantize_(model, Int8DynamicActivationInt8WeightConfig()). Per-token int8 act + per-channel int8 weight. | pip install torchao | sm75+/sm80+ | int8 weight+act dynamic, bf16 compute | PyTorch-native W8A8 (SmoothQuant-like) with torch.compile codegen; clean module-swap. _(confidence: high)_ | [link](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md) |
| 4 | vllm (marlin) | from vllm import _custom_ops as ops; ops.marlin_gemm(...) / QQQ-marlin for int8 W8A8 on Blackwell where CUTLASS int8 is absent; unified Marlin entrypoint handles int8 channelwise scales. | pip install vllm | sm80+ incl. sm100 (Marlin covers Blackwell where CUTLASS int8 does not) | int8/int4 weight, fp16/bf16 act, fp32 scales | Marlin fallback gives int8 W8A8 on Blackwell; very high throughput mixed-precision GEMM. _(confidence: medium)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |

### `quantize_int8_dynamic`

Dynamic symmetric (and optionally asymmetric) INT8 quantization of activations: compute scale (per-token/per-tensor), write int8 + fp32 scale (+ azp zero-point).

**kernel_set_abi:** `ks_quantize_int8`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm import _custom_ops as ops; q, scale, azp = ops.scaled_int8_quant(input, scale=None, azp=None, symmetric=True). Returns (int8, fp32 scale, optional int32 azp). | pip install vllm | sm75+/sm80+ | fp16/bf16 -> int8, fp32 scale, int32 azp | Fused dynamic int8 quant feeding cutlass_scaled_mm; supports symmetric and asymmetric (zero-point) quant. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |
| 2 | torchao | Handled inside Int8DynamicActivationInt8WeightConfig via quantize_; low-level torchao.quantization.quant_primitives.choose_qparams_affine + quantize_affine for explicit int8 quant. | pip install torchao | sm75+ | fp16/bf16 -> int8, fp32 scale | Composable affine quant primitives; torch.compile-fused. _(confidence: medium)_ | [link](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md) |

### `dequantize_int8`

Dequantize int8 tensor to fp16/bf16 using per-token/per-channel/per-tensor scale.

**kernel_set_abi:** `ks_dequantize_int8`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torchao | torchao.quantization.quant_primitives.dequantize_affine(int_data, block_size, scale, zero_point, ..., output_dtype=torch.bfloat16). | pip install torchao | sm70+ | int8 -> fp16/bf16, fp32 scale | General affine dequant primitive; torch.compile-fusible. _(confidence: medium)_ | [link](https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py) |
| 2 | torch | out = (input_i8.to(torch.float32) * scale).to(torch.bfloat16) (symmetric); subtract zero_point for asymmetric. Native. | pip install torch | any CUDA | int8 -> fp16/bf16 | Trivial portable dequant; fuse with downstream via torch.compile. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/tensors.html) |

### `w4a16_gemm`

W4A16 mixed-input GEMM: fp16/bf16 activations [M,K] x int4 group-quantized weights [K,N] with group-wise scales (+zeros). Memory-bound decode workhorse for GPTQ/AWQ checkpoints. Marlin/Machete are the best impls.

**kernel_set_abi:** `ks_gemm_w4a16`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm (Marlin / GPTQ-Marlin) | from vllm import _custom_ops as ops; ops.marlin_gemm(a, c_or_none, b_q_weight, b_bias_or_none, b_scales, a_scales_or_none, global_scale_or_none, b_zeros_or_none, g_idx_or_none, perm_or_none, workspace, b_q_type, size_m, size_n, size_k, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False). Weights must be pre-repacked via ops.gptq_marlin_repack / ops.awq_marlin_repack. b_q_type is a ScalarType (e.g. scalar_types.uint4b8 for GPTQ or scalar_types.uint4 for AWQ). | pip install vllm | sm80+ (Ampere A100, Ada, Hopper). Best on Ampere/Ada; Marlin also covers Blackwell sm100. | int4 (uint4b8) weights, fp16/bf16 act, fp16/bf16 group scales (group_size 32/64/128 or channelwise), optional fp32 zeros | Marlin: near-ideal 4x speedup at low batch, sustains gains to batch ~32-64 via clever pipelining; the de-facto W4A16 kernel in vLLM. Handles GPTQ (sym/asym, desc_act via g_idx/perm) and AWQ-repacked weights. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils.py) |
| 2 | vllm (Machete) | from vllm import _custom_ops as ops; ops.machete_mm(a, b_q, b_type, out_type=None, b_group_scales=..., b_group_zeros=..., b_group_size=128, b_channel_scales=None, a_token_scales=None, schedule=None). Pre-prepack weights with ops.machete_prepack_B. b_type is ScalarType. | pip install vllm | sm90a Hopper-only (CUTLASS 3.5.1, uses TMA + wgmma). Falls back to Marlin elsewhere. | int4/int8 weights, fp16/bf16 (and fp8 act for W4A8) act, group scales+zeros | Neural Magic's Hopper-optimized successor to Marlin; uses TMA, weight pre-shuffle, beats Marlin on H100. Also provides W4A8 (fp8 act) path. _(confidence: high)_ | [link](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel) |
| 3 | exllamav3 | EXL3 format (QTIP-derived trellis quant). Use via exllamav3.Linear / model.forward; kernel exposed as exllamav3.ext quant gemm. High-quality low-bitrate (2-4 bit) consumer-GPU inference. | pip install exllamav3 (or build from source) | sm80+ (Ampere/Ada/Hopper consumer) | int2-int8 trellis weights, fp16 act | Best-in-class accuracy at <=4 bit via QTIP trellis; the strongest consumer-GPU W4A16(-and-lower) inference. Also integrated as an EXL3 backend in GPTQModel. _(confidence: medium)_ | [link](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md) |

### `awq_gemm`

AWQ W4A16 GEMM: activation-aware 4-bit weights with group-wise scales+zeros, packed int4. Original AWQ CUDA GEMM (and GEMV for batch=1).

**kernel_set_abi:** `ks_gemm_w4a16`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm (awq_marlin) | AWQ weights are repacked to Marlin layout (ops.awq_marlin_repack) then served via ops.marlin_gemm with b_q_type=scalar_types.uint4 and b_zeros. Raw kernel: from vllm import _custom_ops as ops; ops.awq_gemm(input, qweight, scales, qzeros, split_k_iters). | pip install vllm | sm75+/sm80+ for raw awq_gemm; awq_marlin path sm80+ | int4 weights, fp16/bf16 act, fp16 group scales+zeros (group_size 128 typical) | awq_marlin (AWQ weights on Marlin kernel) is the fastest AWQ serving path; raw awq_gemm kept for compatibility. ~2x over fp16 at small batch. _(confidence: high)_ | [link](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/awq_marlin/) |
| 2 | autoawq | awq_ext.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, 8) for GEMM (split_k=8); awq_ext.dequantize_weights_cuda(qweight, scales, qzeros, 0,0,0, False) for large M; awq_ext.gemmv2_forward_cuda for batch=1 GEMV. Via WQLinear_GEMM / WQLinear_GEMV modules. | pip install autoawq autoawq-kernels | sm75+/sm80+ (Turing/Ampere/Ada); not Blackwell-tuned | int4 weights, fp16 act, fp16 scales+zeros | Original reference AWQ kernels; GEMV ~20% faster than GEMM at batch=1. NOTE: AutoAWQ is officially deprecated/unmaintained (last tested torch 2.6) — prefer vLLM awq_marlin or GPTQModel for production. _(confidence: high)_ | [link](https://github.com/casper-hansen/AutoAWQ/blob/main/awq/modules/linear/gemm.py) |
| 3 | gptqmodel | from gptqmodel import GPTQModel; model = GPTQModel.load(path, backend=BACKEND.MARLIN); model.generate(...). Supports AWQ/GPTQ/EXL3 checkpoints, auto-selects Marlin/Machete/ExLlamaV3 kernels. | pip install gptqmodel | sm80+ (NVIDIA), also AMD/Intel/CPU backends | int4 weights, fp16/bf16 act, group scales+zeros | Actively-maintained AWQ/GPTQ successor (AutoAWQ + AutoGPTQ are deprecated); routes to best kernel (Marlin/Machete/EXL3) automatically; integrates with vLLM/SGLang. _(confidence: high)_ | [link](https://github.com/ModelCloud/GPTQModel) |

### `dequantize_int4`

Dequantize group-wise int4 weights (AWQ/GPTQ layout): packed int4 -> fp16/bf16 using scales/zeros [K/group_size, N]. Used by W4A16 GEMM-via-dequant path at large M.

**kernel_set_abi:** `ks_dequantize_int4`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | autoawq | awq_ext.dequantize_weights_cuda(qweight, scales, qzeros, 0, 0, 0, False) -> fp16 weight [K,N]. | pip install autoawq-kernels | sm75+/sm80+ | packed int4 -> fp16, fp16 scales+zeros, group_size 128 | Fast int4 weight unpack for the dequant+cuBLAS path used at large batch where mixed-input GEMM loses. (Library deprecated; symbol still standard.) _(confidence: high)_ | [link](https://github.com/casper-hansen/AutoAWQ/blob/main/awq/modules/linear/gemm.py) |
| 2 | vllm | from vllm import _custom_ops as ops; ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0) / ops.gptq_marlin_repack for relayout. Marlin path usually fuses dequant into GEMM rather than materializing. | pip install vllm | sm75+/sm80+ | packed int4 -> fp16/bf16 | Standalone dequant used for repack/debug; production prefers fused Marlin GEMM. _(confidence: medium)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |
| 3 | torchao | Int4WeightOnlyConfig handles pack/dequant internally; low-level torchao.quantization.quant_primitives.dequantize_affine with 4-bit qmin/qmax + group block_size. | pip install torchao | sm80+ | packed int4 (tile_packed_to_4d) -> bf16 | Composable affine 4-bit dequant; tinygemm/tile-packed layouts; HQQ qparams option. _(confidence: medium)_ | [link](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md) |

### `int4_weight_only_gemm_tinygemm`

W4A16 weight-only GEMM via PyTorch tinygemm (_weight_int4pack_mm): bf16 act x int4 tile-packed weights with group scales/zeros. The torch-native int4 path.

**kernel_set_abi:** `ks_gemm_w4a16`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torchao | from torchao.quantization import quantize_, Int4WeightOnlyConfig; quantize_(model, Int4WeightOnlyConfig(group_size=128, int4_packing_format='tile_packed_to_4d', int4_choose_qparams_algorithm='hqq')). Calls torch.ops.aten._weight_int4pack_mm under the hood. | pip install torchao | sm80+ (Ampere/Ada/Hopper); also CPU/MPS variants | int4 tile-packed weights, bf16 act, bf16 group scales+zeros | PyTorch-native int4 weight-only; torch.compile codegen; HQQ qparams for higher accuracy. Cleanest non-CUDA-build option. _(confidence: high)_ | [link](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md) |
| 2 | torch | torch.ops.aten._weight_int4pack_mm(x_bf16, packed_w_int4, group_size, scales_and_zeros). Pack weights with torch.ops.aten._convert_weight_to_int4pack. | pip install torch (CUDA) | sm80+ | int4 packed weights, bf16 act, bf16 scales/zeros, group_size 32/64/128/256 | The tinygemm kernel torchao wraps; available directly in core PyTorch. _(confidence: medium)_ | [link](https://docs.pytorch.org/docs/stable/index.html) |

### `w4a8_gemm`

W4A8 GEMM: int4 group-quantized weights with fp8 (or int8) activations — processes 4-bit weights while leveraging 8-bit activation tensor cores for compute throughput. QServe QoQ / Machete-W4A8 / compressed-tensors.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm (Machete W4A8) | from vllm import _custom_ops as ops; ops.machete_mm(a_fp8, b_q4, b_type, out_type=torch.bfloat16, b_group_scales=..., b_group_size=128, b_channel_scales=..., a_token_scales=...). fp8 activations + int4 weights. | pip install vllm | sm90a Hopper-only (wgmma/TMA). No SM100 W4A8 kernel yet as of 2026. | int4 weight, fp8 e4m3 act, fp16/bf16 out, group scales + per-token act scales | Machete adds fp8-activation W4A8: 4-bit memory savings + fp8 compute. Best W4A8 on Hopper; compressed-tensors W4A8 routes here. _(confidence: medium)_ | [link](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel) |
| 2 | qserve (lmquant) | QServe QoQ W4A8KV4 kernels via omniserve/lmquant runtime (qserve_w4a8 GEMM with progressive int4->int8 dequant). Served through QServe engine, not a thin pip op. | build from source (omniserve / lmquant) | sm80+/sm86 (A100, L40S); int8 tensor cores | int4 weight (progressive to int8), int8 act, int4 KV cache | QoQ progressive quantization minimizes dequant overhead in W4A8 GEMM; large measured throughput gains over TensorRT-LLM at the time of publication. _(confidence: low)_ | [link](https://hanlab.mit.edu/projects/qserve) |

### `nvfp4_gemm`

NVFP4 (4-bit e2m1 with 1x16 micro-block fp8-e4m3 scale + per-tensor global scale) GEMM on Blackwell tensor cores: D = alpha * (A_fp4 @ B_fp4). 2-4x over fp8 on B200.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer | flashinfer.gemm.mm_fp4(a, b, a_descale, b_descale, alpha=alpha, out_dtype=torch.bfloat16, block_size=16, use_nvfp4=True, backend='auto'). Quantize inputs with flashinfer.fp4_quantization.nvfp4_quantize. | pip install flashinfer-python | sm100 (Blackwell B200/B300), sm120 (RTX 50xx/GB10); some backends sm90 | fp4 e2m1 (packed 2/byte) data, fp8-e4m3 block descale (1x16), fp32 global alpha, bf16/fp16 out | Multi-backend (trtllm/cutlass/cudnn/cute-dsl) NVFP4 GEMM with auto-selection; the production Blackwell FP4 GEMM used by SGLang/vLLM. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html) |
| 2 | vllm (cutlass fp4) | from vllm import _custom_ops as ops; ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, block_scale_a, block_scale_b, alpha, out_dtype=torch.bfloat16). Quantize with ops.scaled_fp4_quant(input, input_global_scale). | pip install vllm | sm100 (Blackwell); CUDA 12.8+ | fp4 e2m1 data, fp8-e4m3 1x16 block scale, fp32 global alpha, bf16/fp16 out | vLLM's native CUTLASS NVFP4 scaled-mm + fused fp4 quant; ~up to 61% e2e gain on Qwen3 reported. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |
| 3 | cutlass | C++/CuTe DSL: example 72_blackwell_narrow_precision_gemm (72b nvfp4_nvfp4). Python via nvidia-cutlass-dsl (cutlass.cute) block-scaled GEMM builder; not a single pip fn. | pip install nvidia-cutlass-dsl (or build CUTLASS C++) | sm100a (Blackwell); tcgen05.mma block-scaled instructions | nvfp4/mxfp4 e2m1, fp8/ue8m0 scale factors, fp16/bf16/fp32 out | The reference NVFP4/MXFP4 block-scaled GEMM building blocks (tcgen05.mma, 2-4x WGMMA); flashinfer/vLLM kernels derive from these. _(confidence: high)_ | [link](https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu) |

### `mxfp4_gemm`

MXFP4 (OCP microscaling: 4-bit e2m1 data with shared ue8m0 power-of-two scale per 32-element block) GEMM. Used by GPT-OSS and OCP MX inference on Blackwell.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torchao | from torchao.quantization import quantize_; from torchao.prototype.mx_formats import MXFPInferenceConfig; quantize_(model, MXFPInferenceConfig(block_size=32)). MXFP4 uses torch.float4_e2m1fn_x2 data + torch.float8_e8m0fnu scale. NVFP4 via NVFP4InferenceConfig(). | pip install torchao | sm100 (Blackwell B200) / sm120 (RTX 5090); CUDA 12.8+ | fp4 e2m1 (float4_e2m1fn_x2) data, ue8m0 (float8_e8m0fnu) block scale (block 32), bf16 out | PyTorch-native MXFP4/MXFP8/NVFP4 inference; ~1.68x diffusion speedup with NVFP4 on B200; torch.compile-integrated. Prototype API. _(confidence: high)_ | [link](https://docs.pytorch.org/ao/0.16/_modules/torchao/prototype/mx_formats/inference_workflow.html) |
| 2 | flashinfer | flashinfer.gemm.mm_fp4(a, b, a_descale, b_descale, alpha=None, out_dtype=torch.bfloat16, block_size=32, use_nvfp4=False, backend='auto'). use_nvfp4=False selects MXFP4 (ue8m0, block 32). Also flashinfer.gemm.mm_mxfp8 for MXFP8. | pip install flashinfer-python | sm100 (Blackwell), sm120 | fp4 e2m1 data, ue8m0 block scale (block 32), bf16/fp16 out | Same mm_fp4 entrypoint toggled to MXFP4; serves GPT-OSS MXFP4 weights on Blackwell. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html) |
| 3 | vllm (marlin mxfp4) | GPT-OSS MXFP4 served via Marlin MoE / FlashInfer on vLLM; MXFP4 linear uses ops.marlin_gemm with b_q_type=scalar_types.float4_e2m1f, with FlashInfer/CUTLASS fp4 preferred on Blackwell. | pip install vllm | sm100 (native fp4) | fp4 e2m1 data, ue8m0 block scale, fp16/bf16 act | Enables GPT-OSS native MXFP4 weights on Blackwell via native FP4 tensor cores. _(confidence: medium)_ | [link](https://github.com/vllm-project/vllm/issues/38022) |

### `fp4_quantize`

Quantize fp16/bf16 to NVFP4/MXFP4: produce packed e2m1 data + block scale (fp8-e4m3 for NVFP4 1x16, ue8m0 for MXFP4 1x32) + (NVFP4) per-tensor global scale.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm import _custom_ops as ops; q_fp4, block_scale = ops.scaled_fp4_quant(input, input_global_scale, is_sf_swizzled_layout=True). Returns packed fp4 + swizzled fp8 block-scale. | pip install vllm | sm100 (Blackwell); CUDA 12.8+ | fp16/bf16 -> fp4 e2m1 (packed) + fp8-e4m3 1x16 block scale | Fused NVFP4 quant feeding cutlass_scaled_fp4_mm; swizzled scale-factor layout for tensor-core consumption. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |
| 2 | flashinfer | flashinfer.fp4_quantization.nvfp4_quantize(x, global_scale, sfLayout=SfLayout.layout_128x4, do_shuffle=False) -> (fp4_packed, scale_factors). mxfp4 variant for ue8m0/block-32. | pip install flashinfer-python | sm100/sm120 | fp16/bf16 -> fp4 e2m1 + fp8/ue8m0 block scale | Layout-aware fp4 quant (128x4 / 8x4 swizzle) matching mm_fp4 backends. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/fp4_quantization.html) |

### `nf4_fp4_blockwise_quant_linear`

4-bit weight-only quantization (NF4 NormalFloat / FP4) with blockwise absmax scaling + double quantization; QLoRA. Weights stored 4-bit, GEMV dequant-on-the-fly, compute in bf16/fp16.

**kernel_set_abi:** `ks_gemm_w4a16 (analogous weight-only 4-bit linear; ks ABI assumes int4 group+zeros, bnb uses NF4/FP4 blockwise)`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | bitsandbytes | import bitsandbytes as bnb; layer = bnb.nn.Linear4bit(in, out, compute_dtype=torch.bfloat16, quant_type='nf4', use_double_quant=True). Functional: bnb.functional.quantize_4bit / dequantize_4bit(q, quant_state, quant_type='nf4'); bnb.functional.gemv_4bit(A, B, out, state=quant_state) for batch=1. | pip install bitsandbytes | sm75+/sm80+ (CUDA); also experimental ROCm/Intel/CPU multi-backend | NF4 / FP4 4-bit storage, fp16/bf16 compute, fp32 blockwise absmax scales | The standard QLoRA 4-bit path; NF4 is information-theoretically optimal for normally-distributed weights; double-quant compresses scales. gemv_4bit fast for decode. _(confidence: high)_ | [link](https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit) |

### `int8_llm_int8_linear`

LLM.int8(): vector-wise (per-row/col) int8 GEMM with separate fp16 mixed-precision handling of outlier feature dimensions. Zero-degradation 8-bit weight+activation linear.

**kernel_set_abi:** `ks_gemm_w8a8 (LLM.int8 adds an fp16 outlier path on top of int8 W8A8)`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | bitsandbytes | import bitsandbytes as bnb; layer = bnb.nn.Linear8bitLt(in, out, has_fp16_weights=False, threshold=6.0). Functional: bnb.functional.int8_linear (vectorwise quant + int8 matmul + outlier fp16 path). | pip install bitsandbytes | sm75+/sm80+; H100 supported since 0.45 | int8 weight+act vectorwise, fp16 outlier columns, fp16 compute | Outlier-aware int8 with no accuracy loss; the original LLM.int8() implementation. Best for accuracy-critical 8-bit inference/QLoRA base. _(confidence: high)_ | [link](https://huggingface.co/docs/transformers/quantization/bitsandbytes) |

---

## Norm / Activation / RoPE — `norm-act-rope`

20 operators, 44 providers.

### `rmsnorm`

Root-mean-square layer normalization: out = (x / sqrt(mean(x^2) + eps)) * weight. Inference-grade fused kernel.

**kernel_set_abi:** `ks_rmsnorm`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.norm.rmsnorm(input, weight, eps=1e-6, out=None, enable_pdl=None) -> Tensor; computes (x/RMS(x))*weight. input 2D (n,d) or 3D (n,h,d), weight (d,). | pip install flashinfer-python (optionally flashinfer-cubin flashinfer-jit-cache for cu128/cu129/cu130) | sm75+ (Turing) through Blackwell sm100/sm103/sm120; PDL path needs sm90+ | fp16/bf16 (fp8 via rmsnorm_quant variant) | NVIDIA-maintained serving kernel lib; single-pass fused kernel with optional programmatic dependent launch (PDL); standard in vLLM/SGLang serving stacks. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.norm.rmsnorm.html) |
| 2 | sgl-kernel | from sgl_kernel import rmsnorm; rmsnorm(input, weight, eps=1e-6, out=None, enable_pdl=None) -> Tensor; out=(x/RMS(x))*weight. input (n,d) or (n,h,d), weight (d,). | pip install sgl-kernel | sm80+ (Ampere+); PDL path auto-enabled on sm90 (Hopper); also HIP/ROCm | fp16/bf16 (fp32 via internal path) | SGLang's production RMSNorm; FlashInfer-derived single-pass fused kernel with optional programmatic dependent launch (PDL) on Hopper; the hard-op alignment target for kernel-set's ks_rmsnorm. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/elementwise.py) |
| 3 | vllm | vllm._custom_ops.rms_norm(out, input, weight, epsilon) -> None (in-place into out). Higher-level: vllm.model_executor.layers.layernorm.RMSNorm(hidden_size, eps).forward(x). | pip install vllm | sm70+ (CUDA); also ROCm/CPU backends | fp16/bf16/fp32 | Battle-tested CUDA kernel powering vLLM serving; torch.compile-fusion aware; out-param in-place form avoids alloc. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |
| 4 | liger-kernel | from liger_kernel.transformers import LigerRMSNorm; LigerRMSNorm(hidden_size, eps=1e-6, offset=0.0, casting_mode='llama').forward(x). Functional: liger_kernel.ops.rms_norm.LigerRMSNormFunction.apply(x, weight, eps, offset, casting_mode). | pip install liger-kernel | sm80+ (Triton; AMD ROCm supported) | fp16/bf16/fp32 | Triton kernel with full training backward; casting_mode handles llama/gemma numerics; ~20% throughput / 60% memory wins for training; offset enables gemma (+1). _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel) |

### `fused_add_rmsnorm`

Fused residual add then RMSNorm: residual += x; x = rmsnorm(residual) * weight. Updates residual in place; the canonical pre-norm transformer block fusion.

**kernel_set_abi:** `ks_fused_add_rmsnorm`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.norm.fused_add_rmsnorm(input, residual, weight, eps=1e-6, enable_pdl=None) -> None. In-place: residual = input+residual; input = rmsnorm(residual)*weight. | pip install flashinfer-python | sm75+ through Blackwell; PDL sm90+ | fp16/bf16 (fp8 out via fused_add_rmsnorm_quant) | Single-kernel residual+norm; eliminates an extra HBM round-trip vs separate add+norm; quant fused variant available. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.norm.fused_add_rmsnorm.html) |
| 2 | sgl-kernel | from sgl_kernel import fused_add_rmsnorm; fused_add_rmsnorm(input, residual, weight, eps=1e-6, enable_pdl=None) -> None. In-place: residual += input; input = rmsnorm(residual)*weight. | pip install sgl-kernel | sm80+ (Ampere+); PDL on sm90; HIP/ROCm | fp16/bf16 | SGLang fused residual-add + RMSNorm; single kernel saves an HBM round-trip; the canonical pre-norm transformer block fusion and kernel-set's alignment target. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/elementwise.py) |
| 3 | vllm | vllm._custom_ops.fused_add_rms_norm(input, residual, weight, epsilon) -> None. In-place updates both input and residual. | pip install vllm | sm70+ | fp16/bf16/fp32 | Used in every vLLM transformer block; fusion-friendly with quant via rms_norm_dynamic_per_token_quant(residual=...). _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |
| 4 | liger-kernel | from liger_kernel.transformers import LigerFusedAddRMSNorm; LigerFusedAddRMSNorm(hidden_size, eps=1e-6).forward(x, residual) -> (out, new_residual). | pip install liger-kernel | sm80+ (Triton; ROCm) | fp16/bf16/fp32 | Triton fused add+rmsnorm with training backward for both x and residual grads. _(confidence: medium)_ | [link](https://github.com/linkedin/Liger-Kernel) |

### `gemma_rmsnorm`

Gemma-style RMSNorm with (1+weight) scale: out = (x / RMS(x)) * (weight + 1). Computation in fp32 then cast.

**kernel_set_abi:** `ks_gemma_rmsnorm`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.norm.gemma_rmsnorm(input, weight, eps=1e-6, out=None, enable_pdl=None) -> Tensor; out=(x/RMS(x))*(weight+1). Fused-add: flashinfer.norm.gemma_fused_add_rmsnorm(input, residual, weight, eps=1e-6). | pip install flashinfer-python | sm75+ through Blackwell | fp16/bf16 | Dedicated gemma kernel handling the +1 weight offset and fp32 accumulation matching Gemma2/3 reference numerics; plus gemma_fused_add_rmsnorm. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.norm.gemma_rmsnorm.html) |
| 2 | sgl-kernel | from sgl_kernel import gemma_rmsnorm; gemma_rmsnorm(input, weight, eps=1e-6, out=None, enable_pdl=None) -> Tensor; out=(x/RMS(x))*(weight+1). Also gemma_fused_add_rmsnorm(input, residual, weight, eps). | pip install sgl-kernel | sm80+ (Ampere+); PDL on sm90; HIP/ROCm | fp16/bf16 | SGLang Gemma-style RMSNorm with (weight+1) scale computed in fp32; alignment target for ks_gemma_rmsnorm. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/elementwise.py) |
| 3 | liger-kernel | from liger_kernel.transformers import LigerRMSNorm; LigerRMSNorm(hidden_size, eps=1e-6, offset=1.0, casting_mode='gemma', in_place=False) -> gemma semantics via offset=1 and casting_mode='gemma'. | pip install liger-kernel | sm80+ (Triton) | fp16/bf16/fp32 | Gemma numerics via offset=1.0 + casting_mode='gemma'; full training backward; used by HF Liger Gemma patch. _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel) |

### `layernorm`

Standard layer normalization: out = (x - mean) / sqrt(var + eps) * weight + bias. Optional zero-centered gamma / gemma style.

**kernel_set_abi:** `ks_layernorm`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | apex | from apex.normalization import FusedLayerNorm; FusedLayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, memory_efficient=False)(x). Functional: apex.normalization.fused_layer_norm.fused_layer_norm_affine(input, weight, bias, normalized_shape, eps=1e-6). | pip install -v --no-build-isolation --config-settings '--build-option=--cpp_ext' --config-settings '--build-option=--cuda_ext' git+https://github.com/NVIDIA/apex.git | sm70+ (compiled CUDA extension fused_layer_norm_cuda) | fp16/bf16/fp32 (+ mixed via MixedFusedLayerNorm) | Long-standing reference fused LN with hand-tuned CUDA welford kernel and full training backward; memory_efficient flag recomputes in backward. _(confidence: high)_ | [link](https://github.com/NVIDIA/apex/blob/master/apex/normalization/fused_layer_norm.py) |
| 2 | transformer-engine | import transformer_engine.pytorch as te; te.LayerNorm(hidden_size, eps=1e-5, zero_centered_gamma=False)(x). Also te.pytorch.ops.LayerNorm for the operation-fuser graph. | pip install transformer-engine[pytorch] | sm80+; fp8 paths Hopper sm90 / Blackwell sm100 | fp16/bf16/fp32, fp8 (with autocast recipe) | NVIDIA TE fused LN supporting zero-centered gamma and fp8 fusion into following GEMM (LayerNormLinear); training-grade. _(confidence: high)_ | [link](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html) |
| 3 | liger-kernel | from liger_kernel.transformers import LigerLayerNorm; LigerLayerNorm(hidden_size, eps=1e-6, bias=True).forward(x). Functional: liger_kernel.ops.layer_norm.LigerLayerNormFunction.apply(x, weight, bias, eps). | pip install liger-kernel | sm80+ (Triton; ROCm) | fp16/bf16/fp32 | Triton LN with training backward; portable across NVIDIA/AMD. _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/layer_norm.py) |

### `qk_norm`

Per-head RMSNorm (or LayerNorm) applied to query and key vectors before attention (head_dim-wise normalization). Used by Qwen3, Gemma2/3, OLMo2, GPT-OSS.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | Apply flashinfer.norm.rmsnorm to reshaped (n, num_heads, head_dim) tensor (3D input supported) with per-head weight; for gemma qk-norm use flashinfer.norm.gemma_rmsnorm. No single 'qk_norm' symbol — reuse rmsnorm over head_dim. | pip install flashinfer-python | sm75+ through Blackwell | fp16/bf16 | rmsnorm accepts 3D (n,h,d) so QK-norm = rmsnorm over head_dim with shared per-head weight; fastest inference path. _(confidence: medium)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.norm.rmsnorm.html) |
| 2 | vllm | vllm.model_executor.layers.layernorm.RMSNorm(head_dim, eps).forward(q_or_k_reshaped); vLLM models (Qwen3, GPT-OSS) call RMSNorm on reshaped (..., head_dim) q/k. Underlying ops.rms_norm. | pip install vllm | sm70+ | fp16/bf16/fp32 | Reference QK-norm path in production vLLM model definitions (Qwen3Attention applies RMSNorm to q/k per head_dim). _(confidence: medium)_ | [link](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/layernorm/) |
| 3 | transformer-engine | te.pytorch DotProductAttention / MultiheadAttention with qk_norm_before_rope flag; RMSNorm applied to q/k. Or compose te.RMSNorm(head_dim) over reshaped heads. | pip install transformer-engine[pytorch] | sm80+ | fp16/bf16/fp32/fp8 | TE attention supports built-in QK-norm with configurable before/after RoPE ordering; training-grade. _(confidence: low)_ | [link](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py) |

### `groupnorm`

Group normalization: normalize over (channels/groups, spatial) groups; out = (x - mean_g)/sqrt(var_g+eps)*weight + bias. Common in diffusion/vision backbones.

**kernel_set_abi:** `ks_groupnorm`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | apex | from apex.contrib.group_norm import GroupNorm; GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)(x) (NHWC fused CUDA kernel; SiLU fusion option in some builds). | pip install -v --no-build-isolation --config-settings '--build-option=--cpp_ext' --config-settings '--build-option=--cuda_ext' --config-settings '--build-option=--group_norm' git+https://github.com/NVIDIA/apex.git | sm70+ (CUDA extension) | fp16/bf16/fp32 | Hand-tuned NHWC channels-last GroupNorm (+optional SiLU fusion) widely used in SD/diffusion training; faster than torch.nn.GroupNorm in NHWC. _(confidence: medium)_ | [link](https://github.com/NVIDIA/apex/tree/master/apex/contrib/group_norm) |
| 2 | torch | torch.nn.functional.group_norm(input, num_groups, weight=None, bias=None, eps=1e-5); module torch.nn.GroupNorm(num_groups, num_channels, eps, affine). | pip install torch | any CUDA (sm60+) / CPU / MPS | fp16/bf16/fp32 | Native fused ATen kernel; channels-last aware; ubiquitous baseline with full autograd. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.group_norm.html) |

### `rmsnorm_quant`

Fused RMSNorm + dynamic/static quantization to fp8 (or per-token int8) in a single kernel; optionally with residual add.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.norm.rmsnorm_quant(out, input, weight, scale, ...); residual+norm+quant: flashinfer.norm.fused_add_rmsnorm_quant(out, input, residual, weight, ...). Outputs fp8 + scale. | pip install flashinfer-python | fp8 needs sm89/sm90+ (Hopper/Ada/Blackwell) | in fp16/bf16; out fp8 (e4m3) + scale | Fuses norm and quantization to feed fp8 GEMM directly, removing a separate quant pass; key for fp8 serving. _(confidence: medium)_ | [link](https://docs.flashinfer.ai/api/norm.html) |
| 2 | vllm | vllm._custom_ops.rms_norm_dynamic_per_token_quant(input, weight, epsilon, quant_dtype, scale_ub=None, residual=None) -> (quantized_out, scales). | pip install vllm | fp8 sm89/sm90+; int8 sm75+ | in fp16/bf16; out fp8/int8 per-token + scales | Dynamic per-token quant fused into RMSNorm (with optional residual add); production fp8/int8 path in vLLM. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |

### `silu_and_mul`

SwiGLU gate activation: out = silu(x[..., :d]) * x[..., d:], where input last dim = 2d (gate concatenated with up projection).

**kernel_set_abi:** `ks_silu_and_mul`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.activation.silu_and_mul(input, out=None, enable_pdl=None) -> Tensor; input (..., 2*d) -> (..., d) = silu(input[...,:d])*input[...,d:]. | pip install flashinfer-python | sm75+ through Blackwell; PDL sm90+ | fp16/bf16 | Vectorized fused SiLU-gate-mul; NVIDIA-maintained; PDL for kernel overlap. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.activation.silu_and_mul.html) |
| 2 | sgl-kernel | from sgl_kernel import silu_and_mul; silu_and_mul(input, out=None) -> Tensor; computes silu(input[...,:d//2]) * input[...,d//2:]. Also gelu_and_mul, gelu_tanh_and_mul. | pip install sgl-kernel | sm80+ (Ampere+); HIP/ROCm | fp16/bf16 (last-dim bytes must be multiple of 16) | SGLang fused SwiGLU activation (128-bit vectorized loads); alignment target for ks_silu_and_mul. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/elementwise.py) |
| 3 | vllm | torch.ops._C.silu_and_mul(out, x) via vllm.model_executor.layers.activation.SiluAndMul()(x); x (..., 2d) -> (..., d). | pip install vllm | sm70+ (CUDA/ROCm) | fp16/bf16/fp32 | Default SwiGLU activation kernel in vLLM MLP; also MulAndSilu (silu on second half) variant. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py) |
| 4 | liger-kernel | from liger_kernel.transformers import LigerSwiGLUMLP; or functional liger_kernel.ops.swiglu.LigerSiLUMulFunction.apply(gate, up) -> silu(gate)*up (gate/up as separate tensors). | pip install liger-kernel | sm80+ (Triton; ROCm) | fp16/bf16/fp32 | Triton SiLU-mul with training backward; LigerSwiGLUMLP fuses the whole gate/up/down MLP; memory-efficient for training. _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel) |

### `swiglu_oai_clamped`

GPT-OSS clamped SwiGLU: gate clamped to [-inf, limit], up clamped to [-limit, limit]; out = up * sigmoid(alpha*gate) * (gate) form with alpha=1.702, limit=7.0 to bound activations for MXFP4.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | torch.ops._C.swigluoai_and_mul(out, x, alpha, limit) via vllm.model_executor.layers.activation.SwigluOAIAndMul(alpha=1.702, limit=7.0)(x). | pip install vllm | sm70+ | fp16/bf16 | Reference clamped-SwiGLU kernel for GPT-OSS experts; clamp at +/-7 prevents activation explosion in MXFP4; alpha/limit passed to fused kernel. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py) |
| 2 | flashinfer-python | flashinfer activation/MoE path for GPT-OSS (silu_and_mul_scaled_nvfp4_experts_quantize and gpt-oss mxfp4 MoE runner); clamped swiglu handled inside the fused MXFP4 MoE kernel rather than a standalone symbol. | pip install flashinfer-python flashinfer-cubin | sm90 (Hopper) / sm100 (Blackwell) for MXFP4 MoE | mxfp4 weights, bf16/fp16 act | GPT-OSS clamped SwiGLU fused inside FlashInfer's CUTLASS MXFP4 MoE backend (W4A16) for SM90/SM100. _(confidence: low)_ | [link](https://docs.flashinfer.ai/api/activation.html) |

### `gelu_and_mul`

GeGLU gate activation: out = gelu(x[..., :d]) * x[..., d:]. exact (erf) and tanh-approx variants.

**kernel_set_abi:** `ks_gelu_and_mul`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.activation.gelu_and_mul(input, out=None, enable_pdl=None) (exact/erf) and flashinfer.activation.gelu_tanh_and_mul(input, out=None, enable_pdl=None) (tanh approx). input (..., 2d) -> (..., d). | pip install flashinfer-python | sm75+ through Blackwell | fp16/bf16 | Both exact and tanh GeGLU fused gate kernels with PDL; matches Gemma (gelu_tanh) numerics. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.activation.gelu_and_mul.html) |
| 2 | vllm | torch.ops._C.gelu_and_mul(out, x) / gelu_tanh_and_mul(out, x) via vllm.model_executor.layers.activation.GeluAndMul(approximate='none'\|'tanh')(x). | pip install vllm | sm70+ | fp16/bf16/fp32 | approximate flag selects erf vs tanh kernel; production GeGLU for Gemma/Phi MLPs. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py) |
| 3 | liger-kernel | from liger_kernel.transformers import LigerGEGLUMLP; functional liger_kernel.ops.geglu.LigerGELUMulFunction.apply(gate, up) -> gelu_tanh(gate)*up. | pip install liger-kernel | sm80+ (Triton) | fp16/bf16/fp32 | Triton GeGLU (tanh approx) with training backward; LigerGEGLUMLP fuses full Gemma MLP. _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel) |

### `silu`

Elementwise SiLU/Swish activation: out = x * sigmoid(x).

**kernel_set_abi:** `ks_silu`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torch | torch.nn.functional.silu(input, inplace=False); module torch.nn.SiLU(). | pip install torch | any CUDA/CPU/MPS | fp16/bf16/fp32 | Native fused elementwise kernel, autograd; for gated form prefer silu_and_mul. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.silu.html) |

### `gelu`

Elementwise GeLU activation (exact erf or tanh approx), plus fast/quick/new variants used by GPT-2/BERT-style models.

**kernel_set_abi:** `ks_gelu`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | torch.ops._C.gelu_new(out, x) (NewGELU), gelu_fast(out, x) (FastGELU), gelu_quick(out, x) (QuickGELU = x*sigmoid(1.702x)) via vllm.model_executor.layers.activation.{NewGELU,FastGELU,QuickGELU}(). | pip install vllm | sm70+ | fp16/bf16/fp32 | Fused CUDA kernels for the GPT/CLIP gelu variants (new/fast/quick); QuickGELU used by CLIP/ViT. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py) |
| 2 | torch | torch.nn.functional.gelu(input, approximate='none'\|'tanh'); module torch.nn.GELU(approximate=...). | pip install torch | any CUDA/CPU/MPS | fp16/bf16/fp32 | Native fused gelu (erf and tanh); reference baseline with autograd. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html) |

### `relu`

Elementwise ReLU / ReLU^2 (squared relu used by some MLPs).

**kernel_set_abi:** `ks_relu`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torch | torch.nn.functional.relu(input, inplace=False); ReLU^2 = relu(x)**2. vLLM also exposes ReLU2 fused gate. | pip install torch | any CUDA/CPU/MPS | fp16/bf16/fp32/int | Native fused; trivially fast; for gated relu use vLLM ReLU2 / fused MLP. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html) |

### `rope`

Rotary positional embedding applied to q/k. Supports NeoX (rotate-halves, interleave=False) and GPT-J/interleaved (interleave=True) layouts, partial rotary_dim, and precomputed cos/sin cache.

**kernel_set_abi:** `ks_rope`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.rope.apply_rope_pos_ids(q, k, pos_ids, rotary_dim=None, interleave=False, rope_scale=1, rope_theta=1e4) -> (q,k); or with cache: flashinfer.rope.apply_rope_with_cos_sin_cache(positions, query, key, head_size, cos_sin_cache, is_neox=True) -> (q,k). interleave=False=>NeoX, True=>GPT-J; rotary_dim<head_dim => partial. | pip install flashinfer-python | sm75+ through Blackwell | fp16/bf16 | Computes cos/sin on the fly or from cache; supports partial rotary, pos_ids (paged), neox/gptj; the de-facto serving RoPE. Inplace variants avoid alloc. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/rope.html) |
| 2 | sgl-kernel | from sgl_kernel import rotary_embedding; rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox=True) -> None (in-place rotates query/key). positions int64 (tokens,); query/key flattened (tokens, heads*head_dim); cos_sin_cache (max_pos, head_dim). | pip install sgl-kernel | sm80+ (Ampere+); HIP/ROCm | fp16/bf16 | SGLang fused rotary embedding (NeoX + interleaved); in-place over packed q/k; alignment target for ks_rope. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/elementwise.py) |
| 3 | vllm | vllm._custom_ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox, rope_dim_offset=0, inverse=False) -> None (in-place). is_neox toggles NeoX vs interleaved. Higher-level: vllm.model_executor.layers.rotary_embedding.get_rope(...). | pip install vllm | sm70+ (CUDA/ROCm) | fp16/bf16/fp32 | Production paged RoPE with precomputed cos_sin_cache; batched_rotary_embedding for multi-LoRA/mrope; rope_dim_offset for partial rotary. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |
| 4 | transformer-engine | from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb; apply_rotary_pos_emb(t, freqs, tensor_format='sbhd', start_positions=None, interleaved=False, fused=False, cu_seqlens=None, cp_size=1, cp_rank=0) -> Tensor. fused=True uses CUDA kernel; trains (autograd). | pip install transformer-engine[pytorch] | sm80+ | fp16/bf16/fp32 | Training-grade fused RoPE (nvte_fused_rope_forward/backward); context-parallel aware (cp_size/cp_rank), THD/SBHD/BSHD formats, interleaved flag. _(confidence: high)_ | [link](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention/rope.py) |

### `rope_train_backward`

RoPE with autograd backward for training (q/k grads), cos/sin form. NeoX rotate-halves.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | liger-kernel | from liger_kernel.ops.rope import LigerRopeFunction; LigerRopeFunction.apply(q, k, cos, sin, position_ids=None, unsqueeze_dim=1) -> (q_rot, k_rot). q (b, n_qh, s, d), k (b, n_kvh, s, d), cos/sin (1\|b, s, d). | pip install liger-kernel | sm80+ (Triton; ROCm) | fp16/bf16/fp32 | Triton RoPE with full forward+backward; drop-in for HF liger_rotary_pos_emb; memory-efficient training. _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rope.py) |
| 2 | transformer-engine | transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb(t, freqs, fused=True, tensor_format='thd'\|'sbhd', cu_seqlens=...) (autograd backward via fused kernel). | pip install transformer-engine[pytorch] | sm80+ | fp16/bf16/fp32 | Fused RoPE fwd/bwd with context-parallel support; used in Megatron-LM training. _(confidence: high)_ | [link](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention/rope.py) |

### `rope_llama31_scaling`

Llama 3.1 RoPE frequency scaling (low/high freq factor smooth interpolation over old_context_len) applied during RoPE.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.rope.apply_llama31_rope_pos_ids(q, k, pos_ids, rotary_dim=None, interleave=False, rope_scale=8, rope_theta=5e5, low_freq_factor=1, high_freq_factor=4, old_context_len=8192) -> (q,k). Also apply_llama31_rope(...indptr,offsets...) ragged form and inplace variants. | pip install flashinfer-python | sm75+ through Blackwell | fp16/bf16 | Built-in Llama-3.1 smooth freq scaling inside the kernel (no precomputed cache needed); matches HF llama3 rope_scaling. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.rope.apply_llama31_rope.html) |
| 2 | vllm | vllm.model_executor.layers.rotary_embedding.get_rope(head_size, rotary_dim, max_position, base, is_neox_style=True, rope_scaling={'rope_type':'llama3','factor':8,'low_freq_factor':1,'high_freq_factor':4,'original_max_position_embeddings':8192}) -> module; builds cos_sin_cache then ops.rotary_embedding. | pip install vllm | sm70+ | fp16/bf16 | Llama3 scaling baked into precomputed cos_sin_cache via get_rope factory; production path for Llama 3.1/3.3. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/rotary_embedding) |

### `rope_yarn_ntk_scaling`

YaRN / NTK-aware / dynamic-NTK / linear RoPE context-extension scaling, producing the cos/sin cache (with attention temperature mscale for YaRN).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | vllm.model_executor.layers.rotary_embedding.get_rope(..., rope_scaling={'rope_type':'yarn'\|'dynamic'\|'linear','factor':...,'original_max_position_embeddings':...}) -> YaRNScalingRotaryEmbedding/etc; then ops.rotary_embedding on built cache. | pip install vllm | sm70+ | fp16/bf16 | Most complete RoPE-scaling factory (yarn/ntk/dynamic/linear/llama3/longrope/mrope/deepseek) with correct mscale; cache built on host, fused kernel applies it. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/rotary_embedding) |
| 2 | flashinfer-python | Precompute YaRN/NTK cos_sin_cache (e.g. via transformers/vllm), then flashinfer.rope.apply_rope_with_cos_sin_cache(positions, query, key, head_size, cos_sin_cache, is_neox=True) -> (q,k). | pip install flashinfer-python | sm75+ through Blackwell | fp16/bf16 | FlashInfer applies any externally-built (YaRN/NTK) cos/sin cache; fastest apply path, scaling computed upstream. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.rope.apply_rope_with_cos_sin_cache.html) |

### `add_residual`

Elementwise residual add (out = x + residual), often fused into norm; standalone fast path.

**kernel_set_abi:** `ks_add`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torch | torch.add(x, residual) / x + residual; prefer fusing via flashinfer.norm.fused_add_rmsnorm or vllm fused_add_rms_norm to avoid extra HBM pass. | pip install torch | any | fp16/bf16/fp32 | Standalone add is memory-bound; best practice is to fuse with the following norm (see fused_add_rmsnorm). _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.add.html) |

### `cast`

Dtype cast / fp8 quantize-cast for activations between kernels.

**kernel_set_abi:** `ks_cast`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | transformer-engine | transformer_engine.pytorch with fp8_autocast(); cast handled by quantized tensor classes / te.pytorch.ops cast ops. For plain cast: torch.Tensor.to(dtype). | pip install transformer-engine[pytorch] | fp8 sm89/sm90+; bf16 any | fp16/bf16/fp32 <-> fp8(e4m3/e5m2) | TE fuses cast+scale into adjacent ops (norm/GEMM) for fp8; plain dtype cast via torch.to is the trivial fallback. _(confidence: medium)_ | [link](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html) |
| 2 | torch | tensor.to(dtype) / torch.ops aten._to_copy; for fp8 use torch.float8_e4m3fn with scaling. | pip install torch | any (fp8 storage sm89+) | all | Native cast; memory-bound, prefer fusing into producing kernel. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html) |

### `dropout`

Bernoulli dropout (with scaling), optionally fused with bias-add/residual (training).

**kernel_set_abi:** `ks_dropout`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | apex | apex fused bias-dropout-residual via apex.contrib (e.g. fused_bias_dropout); else torch native. Many stacks use TE's fused bias+dropout+add. | pip install -v --no-build-isolation git+https://github.com/NVIDIA/apex.git | sm70+ | fp16/bf16/fp32 | Fused bias+dropout+residual reduces passes in training MLP/attn output; standalone dropout rarely the bottleneck. _(confidence: low)_ | [link](https://github.com/NVIDIA/apex) |
| 2 | torch | torch.nn.functional.dropout(input, p=0.5, training=True, inplace=False). | pip install torch | any | fp16/bf16/fp32 | Native fused dropout with Philox RNG; baseline, autograd; inference typically disables dropout entirely. _(confidence: high)_ | [link](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html) |

---

## MoE / Communication — `moe-comm`

12 operators, 28 providers.

### `moe_gate_softmax_topk`

Router gating: softmax over expert logits then top-k expert selection, producing per-token top-k routing weights (optionally renormalized) and expert indices. Standard Mixtral/Switch-style routing.

**kernel_set_abi:** `ks_moe_gate_softmax_topk`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | sgl-kernel | from sgl_kernel import topk_softmax; topk_softmax(topk_weights, topk_ids, gating_output, renormalize=False, moe_softcapping=0.0, correction_bias=None) # writes topk_weights[T,k] (fp32) and topk_ids[T,k] (int32) in-place from gating_output[T,E] | pip install sgl-kernel | sm80+ (Ampere+); also HIP/ROCm | logits fp16/bf16/fp32; weights fp32; ids int32 | Fused softmax+topk CUDA kernel, in-place output buffers, no Python overhead; production routing path for SGLang MoE (DeepSeek/Mixtral/Qwen). _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/moe.py) |
| 2 | vllm | from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk; topk_weights, topk_ids, token_expert_indices = fused_topk(hidden_states, gating_output, topk, renormalize) # CUDA topk_softmax kernel under the hood | pip install vllm | sm80+ (Ampere+); ROCm | logits fp16/bf16/fp32; weights fp32; ids int32 | Backed by the vllm._custom_ops.topk_softmax CUDA kernel; integrates directly into vLLM fused_experts pipeline. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) |
| 3 | megatron-core | from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function; probs, routing_map = topk_routing_with_score_function(logits, topk, use_pre_softmax=False, score_function='softmax', fused=True) | pip install megatron-core | sm80+; fused path requires Transformer Engine | logits fp16/bf16/fp32; probs fp32 | Training-grade router with optional fused kernel (fused=True via TE), supports load-balancing aux loss and router replay. _(confidence: high)_ | [link](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.moe_utils.html) |

### `moe_gate_sigmoid_group_topk`

DeepSeek-V3 style gating: sigmoid scoring + bias correction, group-limited (node-limited) routing that first selects top-k groups then top-k experts within them, then renormalize and apply routed_scaling_factor. Also covers Kimi-K2 fused gate.

**kernel_set_abi:** `ks_moe_gate_sigmoid_group_topk`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | sgl-kernel | from sgl_kernel import moe_fused_gate; topk_weights, topk_ids = moe_fused_gate(input_tensor, bias, num_expert_group, topk_group, topk, num_fused_shared_experts=0, routed_scaling_factor=0.0, apply_routed_scaling_factor_on_output=False) # single fused kernel: sigmoid+bias+group-mask+within-group-topk+softmax-renorm+scale. Also: kimi_k2_moe_fused_gate(input_tensor, bias, topk, renormalize=True, routed_scaling_factor=1.0) | pip install sgl-kernel | sm80+ (Ampere+); HIP/ROCm | input fp16/bf16/fp32; bias fp32; weights fp32; ids int32 | Fully fused DeepSeek-V3 biased group-topk gate in one CUDA kernel (csrc/moe/moe_fused_gate.cu); avoids the multi-kernel grouped_topk path. Best-in-class for DeepSeek/Kimi routing. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/moe/moe_fused_gate.cu) |
| 2 | vllm | from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk; topk_weights, topk_ids = grouped_topk(hidden_states, gating_output, topk, renormalize, num_expert_group=0, topk_group=0, scoring_func='sigmoid', e_score_correction_bias=None) # newer fused_grouped_topk adds routed_scaling_factor; scoring_func in {'softmax','sigmoid'} | pip install vllm | sm80+ (Ampere+); ROCm (AITER biased_grouped_topk) | logits fp16/bf16/fp32; weights fp32; ids int32 | Reference DeepSeek-V3 group-limited routing in vLLM; fused_grouped_topk variant integrates a TRT-LLM-style fused kernel for lower overhead. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) |
| 3 | megatron-core | from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function, group_limited_topk; probs, routing_map = topk_routing_with_score_function(logits, topk, score_function='sigmoid', num_groups=n_group, group_topk=topk_group, scaling_factor=routed_scaling_factor, expert_bias=correction_bias, fused=True) | pip install megatron-core | sm80+; fused path via Transformer Engine | logits fp16/bf16/fp32; probs fp32 | Training-side DeepSeek-V3 device/node-limited routing with expert_bias and sequence-level aux-loss-free load balancing. _(confidence: high)_ | [link](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.moe_utils.html) |

### `moe_align_block_size`

Compute the per-expert sorted token layout aligned/padded to the GEMM block size: produces sorted_token_ids, expert_ids per block, and num_tokens_post_pad so each expert's token segment starts on a block boundary for the grouped GEMM.

**kernel_set_abi:** `ks_moe_compute_permutation`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | sgl-kernel | from sgl_kernel import moe_align_block_size; moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids=False) # all outputs pre-allocated, written in-place | pip install sgl-kernel | sm80+ (Ampere+); HIP/ROCm | topk_ids int32; outputs int32 | Optimized align+sort design (single/multi-block cumsum) detailed in SGLang's MoE align-sort blog; lowest-overhead implementation of the alignment step. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/moe.py) |
| 2 | vllm | from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size; sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(topk_ids, block_size, num_experts, expert_map=None, pad_sorted_ids=False) | pip install vllm | sm80+ (Ampere+); ROCm | topk_ids int32; outputs int32 | Returns freshly-allocated tensors; CUDA kernel backing vLLM's Triton fused_moe block scheduling and expert_map for EP. _(confidence: high)_ | [link](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/moe_align_block_size/) |

### `moe_permute`

Permute/expand and gather tokens so all tokens routed to the same expert are contiguous (one row per (token,expert) pair). Produces permuted hidden states plus the inverse map for unpermute; may also permute fp8 activation scales.

**kernel_set_abi:** `ks_moe_permute`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_permute; permuted_hidden, a1q_scale, first_token_off, inv_perm, ... = moe_permute(hidden_states, a1q_scale, topk_ids, n_expert, n_local_expert=-1, expert_map=None, align_block_size=None, fill_invalid_expert=-1) | pip install vllm | sm80+ (Ampere+); ROCm | hidden fp16/bf16/fp8; scales fp32; indices int32 | CUDA permute kernel that also carries per-token fp8 scales and supports EP expert_map + block alignment in one pass. _(confidence: high)_ | [link](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/moe_permute_unpermute/) |
| 2 | megatron-core | from megatron.core.transformer.moe.moe_utils import permute; permuted_tokens, sorted_indices, ... = permute(tokens, routing_map, probs=None, num_out_tokens=None, fused=True, drop_and_pad=False, align_size=-1) | pip install megatron-core | sm80+; fused=True requires Transformer Engine permute fusion | tokens fp16/bf16/fp8; probs fp32; routing_map bool | Training-grade fused permute (--moe-permute-fusion), can fold probs into the activation to save an unpermute multiply; drop_and_pad capacity mode. _(confidence: high)_ | [link](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.moe_utils.html) |
| 3 | grouped_gemm (megablocks ecosystem) | from grouped_gemm import ops; permuted, row_id_map = ops.permute(tokens, indices) # MegaBlocks/grouped_gemm permute helper for grouped expert layout | GROUPED_GEMM_CUTLASS=1 pip install grouped_gemm | sm80+ (CUTLASS grouped GEMM kernels) | fp16/bf16 | Lightweight permute paired with CUTLASS grouped GEMM; used by MegaBlocks dMoE path. _(confidence: medium)_ | [link](https://github.com/tgale96/grouped_gemm) |

### `moe_grouped_gemm_contiguous`

Grouped/batched expert GEMM over a contiguous (M-grouped) token layout: for each expert segment C[off:off+m_e] = A[off:off+m_e] @ W_e, with N,K fixed across experts. Used for prefill/training where token counts per expert are known on host.

**kernel_set_abi:** `ks_moe_grouped_gemm`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | DeepGEMM | import deep_gemm; deep_gemm.m_grouped_fp8_gemm_nt_contiguous((lhs_fp8, lhs_scales), (rhs_fp8, rhs_scales), out, m_indices) # m_indices[total_m] maps each row to its expert; each segment aligned to block M. BF16 variant: deep_gemm.m_grouped_bf16_gemm_nt_contiguous(lhs, rhs, out, m_indices) | git clone --recursive https://github.com/deepseek-ai/DeepGEMM && cd DeepGEMM && ./install.sh | Hopper sm90 (CUDA 12.3+) and Blackwell sm100 (CUDA 12.9+) | fp8 e4m3 (1D2D fine-grained block scaling), bf16, fp4 (fp8_fp4 variants) | State-of-the-art FP8 grouped GEMM with fine-grained scaling and low-CPU-overhead JIT; powers DeepSeek-V3 MoE. M-axis-only grouping tailored to equal-shape experts. _(confidence: high)_ | [link](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/__init__.py) |
| 2 | sgl-kernel | from sgl_kernel import fp8_blockwise_scaled_grouped_mm, prepare_moe_input; prepare_moe_input(topk_ids, expert_offsets, problem_sizes1, problem_sizes2, input_permutation, output_permutation, num_experts, n, k); fp8_blockwise_scaled_grouped_mm(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace) | pip install sgl-kernel | Hopper sm90+ (CUTLASS grouped GEMM); sm100 paths | fp8 e4m3 blockwise-scaled, bf16 | CUTLASS-based blockwise FP8 grouped GEMM with ptr-array problem-size scheduling; integrated CUTLASS MoE path in SGLang. _(confidence: medium)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/moe.py) |
| 3 | grouped_gemm | from grouped_gemm import ops; out = ops.gmm(a, b, batch_sizes, trans_b=False) # batch_sizes[num_experts] tokens-per-expert (CPU int tensor); a=[total_m,k], b=[num_experts,k,n] | GROUPED_GEMM_CUTLASS=1 pip install grouped_gemm | sm80+ (CUTLASS grouped GEMM; cuBLAS fallback) | fp16/bf16 | The classic Megatron/MegaBlocks grouped GEMM with autograd; single kernel launch when GROUPED_GEMM_CUTLASS=1, else per-expert cuBLAS. _(confidence: high)_ | [link](https://github.com/tgale96/grouped_gemm/blob/main/grouped_gemm/ops.py) |

### `moe_grouped_gemm_masked`

Masked grouped expert GEMM for the decode phase with CUDA graphs: M dimension is fixed-capacity per expert and a per-expert masked_m (valid token count) plus expected_m hint drive computation when the host does not know per-expert counts. Avoids host sync / recapture.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | DeepGEMM | import deep_gemm; deep_gemm.m_grouped_fp8_gemm_nt_masked((lhs_fp8, lhs_scales), (rhs_fp8, rhs_scales), out, masked_m, expected_m) # lhs/out shaped [num_experts, max_m, .]; masked_m[num_experts] valid rows; expected_m avg-count perf hint. BF16: deep_gemm.m_grouped_bf16_gemm_nt_masked(...) | git clone --recursive https://github.com/deepseek-ai/DeepGEMM && cd DeepGEMM && ./install.sh | Hopper sm90 (CUDA 12.3+) and Blackwell sm100 (CUDA 12.9+) | fp8 e4m3 block-scaled, bf16, fp4 | Only widely-used masked grouped GEMM; designed for CUDA-graph decode in DeepEP low-latency MoE. Pairs directly with DeepEP low_latency_dispatch output layout. _(confidence: high)_ | [link](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/__init__.py) |

### `moe_unpermute_combine`

Inverse permutation that scatters expert outputs back to original token positions and reduces the top-k contributions weighted by routing weights: out[t] = sum_k w[t,k] * expert_out[pos(t,k)]. Local (non-distributed) combine.

**kernel_set_abi:** `ks_moe_unpermute`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_unpermute; out = moe_unpermute(permuted_hidden_states, topk_weights, inv_perm, ...) # weighted scatter-add back to [num_tokens, hidden]; pairs with moe_permute. Also vllm._custom_ops.moe_sum for the top-k reduction. | pip install vllm | sm80+ (Ampere+); ROCm | hidden fp16/bf16; weights fp32; indices int32 | CUDA unpermute kernel that fuses routing-weight scaling with the inverse gather; companion to vLLM moe_permute. _(confidence: high)_ | [link](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/moe_permute_unpermute/) |
| 2 | megatron-core | from megatron.core.transformer.moe.moe_utils import unpermute; out = unpermute(permuted_tokens, sorted_indices, restore_shape, probs=None, routing_map=None, fused=True, drop_and_pad=False) | pip install megatron-core | sm80+; fused via Transformer Engine | tokens fp16/bf16/fp8; probs fp32 | Training-grade fused unpermute; when probs are folded into the activation, this is a pure scatter (no multiply), saving bandwidth. _(confidence: high)_ | [link](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.moe_utils.html) |
| 3 | sgl-kernel | from sgl_kernel import moe_sum_reduce; moe_sum_reduce(input, output, routed_scaling_factor) # sums top-k expert outputs [T,k,H] -> [T,H] with optional scaling (combine reduction step) | pip install sgl-kernel | sm80+; HIP/ROCm | fp16/bf16; fp32 accumulate | Fused top-k reduction with routed_scaling_factor for the combine stage; low-overhead final-sum kernel. _(confidence: medium)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/moe.py) |

### `fused_moe_full`

Single-call fused MoE expert MLP: takes hidden states + router top-k weights/ids + the two expert weight matrices and internally does align -> permute -> grouped GEMM (gate/up) -> activation (SiLU/SwiGLU) -> grouped GEMM (down) -> weighted unpermute. The whole expert block in one Python call.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts; out = fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, activation=MoEActivation.SILU, apply_router_weight_on_input=False, global_num_experts=-1, expert_map=None, quant_config=None) # high-level fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize, ...) also available | pip install vllm | sm80+ (Triton); fp8 paths sm89/sm90; ROCm | bf16/fp16, fp8 (w8a8/blockwise), int8/int4 (w8a16/w4a16 gptq/awq) | The de-facto Triton fused MoE; modular kernel architecture composes any all2all prepare/finalize with any expert impl (Triton/DeepGEMM/CUTLASS/Marlin). expert_map gives expert-parallel slicing. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) |
| 2 | flashinfer | from flashinfer.fused_moe import trtllm_fp4_block_scale_moe, cutlass_fused_moe, trtllm_fp8_block_scale_moe; out = trtllm_fp4_block_scale_moe(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, ..., gemm2_weights, gemm2_weights_scale, ..., num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset, local_num_experts, routed_scaling_factor, routing_method_type=0, do_finalize=True) | pip install flashinfer-python | Blackwell sm100 (nvfp4/trtllm), Hopper sm90 (fp8 block scale / cutlass) | nvfp4, mxfp8, fp8 e4m3, bf16 | TensorRT-LLM-derived fused MoE with internal routing (incl. DeepSeek group-topk via routing_method_type) for Blackwell FP4/FP8; cutlass_fused_moe covers Hopper and EP all-to-all. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.fused_moe.trtllm_fp4_block_scale_moe.html) |
| 3 | DeepGEMM | import deep_gemm; deep_gemm.fp8_fp4_mega_moe(...) # 'Mega MoE' single persistent kernel fusing EP dispatch + FC1 + SwiGLU + FC2 + combine via SymmBuffer (deep_gemm.get_symm_buffer_for_mega_moe / transform_weights_for_mega_moe) | git clone --recursive https://github.com/deepseek-ai/DeepGEMM && cd DeepGEMM && ./install.sh | Blackwell sm100 (CUDA 12.9+); Hopper sm90 partial | fp8 e4m3, fp4 | Megakernel that fuses the entire EP MoE layer (dispatch, both GEMMs, SwiGLU, combine) into one launch using symmetric NVLink buffers; cutting-edge but newer/less-battle-tested API. _(confidence: medium)_ | [link](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/__init__.py) |

### `ep_dispatch_alltoall`

Expert-parallel all-to-all dispatch: route each token's top-k copies to the GPUs that host its selected experts (high-throughput, NVLink intranode + RDMA internode). Returns received tokens grouped by local expert plus a handle for the symmetric combine. Supports FP8 dispatch.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | DeepEP | from deep_ep import Buffer; buf = Buffer(group, num_nvl_bytes, num_rdma_bytes); num_per_rank, num_rdma_per_rank, is_token_in_rank, num_recv_per_expert, handle, event = buf.get_dispatch_layout(topk_idx, num_experts); recv_x, recv_topk_idx, recv_topk_weights, num_recv_per_expert, handle, event = buf.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights, num_tokens_per_rank=..., is_token_in_rank=..., num_tokens_per_expert=...) # x is bf16 tensor or (fp8_tensor, scales) tuple | pip install nvshmem-cu12 (or build NVSHMEM) ; python setup.py install | Hopper sm90; NVLink (intranode) + RDMA/NVSHMEM (internode) | bf16 and fp8 e4m3 (dispatch with scales) | DeepSeek's reference EP all-to-all; near-NVLink/RDMA-bandwidth dispatch, FP8 dispatch to halve traffic. The standard EP backend wired into vLLM and SGLang. _(confidence: high)_ | [link](https://github.com/deepseek-ai/DeepEP/blob/main/README.md) |
| 2 | flashinfer | from flashinfer.comm import MoeAlltoAll, moe_a2a_dispatch; a2a = MoeAlltoAll(...); recv = moe_a2a_dispatch(...) # workspace-based MoE all-to-all dispatch (moe_a2a_initialize then moe_a2a_dispatch) | pip install flashinfer-python | Hopper sm90 / Blackwell sm100 (NVLink, multi-node) | bf16, fp8 | TensorRT-LLM-derived MoE all-to-all integrated in FlashInfer; alternative to DeepEP especially on Blackwell. _(confidence: medium)_ | [link](https://docs.flashinfer.ai/api/comm.html) |

### `ep_combine_alltoall`

Expert-parallel all-to-all combine: the inverse of dispatch. Gather each token's expert outputs from the hosting GPUs and reduce the top-k contributions (weighted) back to the originating rank/position, using the handle returned by dispatch.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | DeepEP | from deep_ep import Buffer; combined_x, _, event = buf.combine(x, handle, topk_weights=topk_weights) # handle from buf.dispatch; reduces top-k expert outputs back to source tokens | pip install nvshmem-cu12 ; python setup.py install | Hopper sm90; NVLink + RDMA/NVSHMEM | bf16 (combine), fp8 dispatch counterpart | Symmetric reduce-scatter combine matched to dispatch handle; weighted top-k reduction folded in via topk_weights. _(confidence: high)_ | [link](https://github.com/deepseek-ai/DeepEP/blob/main/README.md) |
| 2 | flashinfer | from flashinfer.comm import moe_a2a_combine; out = moe_a2a_combine(...) # inverse of moe_a2a_dispatch, reduces expert outputs to source ranks | pip install flashinfer-python | Hopper sm90 / Blackwell sm100 | bf16, fp8 | Paired combine for FlashInfer MoeAlltoAll; Blackwell-optimized. _(confidence: medium)_ | [link](https://docs.flashinfer.ai/api/comm.html) |

### `ep_low_latency_dispatch_combine`

Latency-optimized (decode) EP dispatch+combine pair designed for CUDA graphs: pure RDMA/IBGDA path with fixed per-rank token capacity (num_max_dispatch_tokens_per_rank), FP8 dispatch, and an optional receive hook for compute-comm overlap. Output layout pairs with masked grouped GEMM.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | DeepEP | from deep_ep import Buffer; recv_x, recv_count, handle, event, hook = buf.low_latency_dispatch(x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts, use_fp8=True, async_finish=False, return_recv_hook=False); combined_x, event, hook = buf.low_latency_combine(x, topk_idx, topk_weights, handle, use_logfmt=False, async_finish=False, return_recv_hook=False) # recv_x is (fp8_tensor, scales) when use_fp8 | pip install nvshmem-cu12 ; python setup.py install | Hopper sm90; RDMA/IBGDA (NVSHMEM) low-latency mode | fp8 e4m3 dispatch (with scales), bf16 combine, optional logfmt | Microsecond-scale decode EP path with CUDA-graph compatibility and recv-hook double-buffering for full comm/compute overlap; pairs with DeepGEMM masked grouped GEMM. Best-in-class for MoE decode. _(confidence: high)_ | [link](https://github.com/deepseek-ai/DeepEP/blob/main/README.md) |

### `moe_tp_allreduce_fused`

Tensor-parallel all-reduce of MoE/expert outputs, optionally fused with residual-add + RMSNorm (and FP4/FP8 quant) to hide the collective behind the epilogue. Used when experts are tensor-parallel-sharded rather than (or in addition to) expert-parallel.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer | from flashinfer.comm import trtllm_allreduce_fusion; trtllm_allreduce_fusion(allreduce_in, world_size, world_rank, token_num, hidden_dim, workspace_ptrs, ..., residual_in=..., rms_gamma=..., rms_eps=..., pattern=...) # fused AllReduce + Residual + RMSNorm (+ optional FP4/FP8 quant) | pip install flashinfer-python | Hopper sm90 / Blackwell sm100 (NVLink one-shot/two-shot) | fp16/bf16; fp8/fp4 quant epilogue | TRT-LLM one-shot/two-shot custom all-reduce with fused norm epilogue; far lower latency than NCCL for the small-message TP all-reduce after the MoE down-projection. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.comm.trtllm_allreduce_fusion.html) |
| 2 | vllm | from vllm.distributed import tensor_model_parallel_all_reduce; out = tensor_model_parallel_all_reduce(hidden_states) # routes through vLLM custom all-reduce (CustomAllreduce) for small TP messages, else NCCL | pip install vllm | sm80+ (custom all-reduce on NVLink); ROCm | fp16/bf16/fp32 | vLLM's CustomAllreduce beats NCCL for the small latency-bound TP reduce after experts; default collective in the MoE TP path. _(confidence: medium)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py) |

---

## Sampling / Logit Processing — `sampling-logitproc`

16 operators, 32 providers.

### `softmax`

Numerically-stable softmax over the last (vocab) dim, with optional temperature scaling. probs[s,v] = exp(z/T) / sum_v exp(z/T).

**kernel_set_abi:** `ks_softmax`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.sampling.softmax(logits, temperature=1.0, enable_pdl=None) -> probs # online safe softmax with temperature; logits [batch, vocab] | pip install flashinfer-python # or build from source per docs | sm75+ (sm80/89/90 tuned; works on Ampere/Ada/Hopper/Blackwell) | fp32/fp16/bf16 (accumulate fp32) | Single-pass online safe softmax fused with temperature; avoids separate max+exp+sum passes. Part of the sorting-free FlashInfer sampling suite. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/sampling.html) |
| 2 | torch | torch.softmax(logits / temperature, dim=-1) # baseline; vLLM/SGLang torch fallback path | pip install torch | any CUDA SM (sm70+); also CPU/ROCm/XPU | fp32/fp16/bf16 | cuDNN/ATen fused softmax; portable baseline used as the native fallback when FlashInfer is unavailable. _(confidence: high)_ | [link](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html) |

### `log_softmax`

log(softmax(logits)) over the vocab dim, computed stably (logz = logsumexp). Used for logprob/cross-entropy reporting.

**kernel_set_abi:** `ks_log_softmax`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torch | torch.log_softmax(logits, dim=-1) # fp32 out recommended; this is the path vLLM/SGLang use for logprobs | pip install torch | any CUDA SM (sm70+); CPU/ROCm/XPU | fp32/fp16/bf16 (out fp32 for stability) | ATen fused log_softmax with internal fp32 accumulation; this is what production engines (vLLM SamplerOutput logprobs, SGLang) actually call for logprob computation — no faster dedicated CUDA kernel is shipped by FlashInfer/SGLang for log_softmax specifically. _(confidence: high)_ | [link](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html) |

### `argmax_greedy`

Greedy decode: out_token[s] = argmax_v logits[s,v]. temperature==0 case.

**kernel_set_abi:** `ks_argmax`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torch | logits.argmax(dim=-1) # vLLM v1 Sampler greedy path: torch.argmax(logits, dim=-1).view(-1) -> int token ids | pip install torch | any CUDA SM (sm70+); CPU/ROCm/XPU | fp32/fp16/bf16 | ATen block-reduce argmax; the standard greedy path in vLLM v1 (vllm/v1/sample/sampler.py) and SGLang. A single argmax over vocab is memory-bound and already optimal; no specialized kernel beats it. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py) |
| 2 | flashinfer-python | flashinfer.sampling.sampling_from_logits(logits, indices=None, deterministic=True, generator=None, seed=None, offset=None) # with temperature->0 behaves greedily via fused path; for strict argmax torch.argmax is preferred | pip install flashinfer-python | sm75+ (Ampere/Ada/Hopper/Blackwell tuned) | fp32/fp16/bf16 | Fused softmax+sample-from-logits; not a pure argmax but covers the greedy regime inside the same kernel used for stochastic sampling, avoiding an extra launch in mixed-batch decoding. _(confidence: medium)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.sampling.sampling_from_logits.html) |

### `temperature_scaling`

Divide logits by per-request temperature before softmax/sampling. T<=0 => greedy.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopK, TopP, Sample; pipe = LogitsPipe([Temperature(), Softmax(), TopK(), TopP(), Sample()]); ids = pipe(logits, temperature=temps, top_k=ks, top_p=ps) # Temperature() fused into the pipeline | pip install flashinfer-python | sm75+ (Ampere/Ada/Hopper/Blackwell) | fp32/fp16/bf16 | Composable LogitsPipe fuses Temperature+Softmax+filters+Sample with compile-time fusion rules, eliminating intermediate tensors. Best-in-class for full per-request sampling pipelines. _(confidence: high)_ | [link](https://docs.flashinfer.ai/api/logits_processor.html) |
| 2 | torch | logits.div_(temperatures.unsqueeze(-1)) # in-place per-row temperature; vLLM v1 apply_temperature path (handles all_greedy mask) | pip install torch | any CUDA SM; CPU/ROCm/XPU | fp32/fp16/bf16 | Elementwise broadcast division; trivially memory-bound and used directly in vLLM/SGLang. No specialized kernel needed. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py) |

### `top_k_filter_mask`

Keep only the k highest-probability tokens, mask the rest to -inf (logits) or renormalize (probs). Per-request k supported.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.sampling.top_k_mask_logits(logits, top_k) # -> masked logits; or flashinfer.sampling.top_k_renorm_probs(probs, top_k) for renormalized probs. top_k: int\|Tensor[batch] | pip install flashinfer-python | sm75+ (Ampere/Ada/Hopper/Blackwell) | fp32/fp16/bf16 | Sorting-free: finds the k-th threshold via parallel pivot search instead of a full O(V log V) sort, so masking/renorm is O(V) memory-bound. >50% faster than torch.sort-based top-k at large batch. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_mask_logits.html) |
| 2 | vllm | from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_only; apply_top_k_only(logits, k) # k: Tensor; uses torch.topk to find kth value then masks below to -inf | pip install vllm | any CUDA SM (sm70+); CPU/ROCm(aiter)/XPU fallbacks | fp32/fp16/bf16 | torch.topk(k) to get threshold (avoids sorting the whole vocab when only k is needed), then boolean mask. Native fallback when FlashInfer is not installed. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py) |

### `top_p_nucleus_filter`

Nucleus filtering: keep smallest set of tokens whose cumulative prob >= p, renormalize/mask the rest. Per-request p.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=False) # renorm to nucleus; top_p: float\|Tensor[batch]. (For direct sampling use top_p_sampling_from_probs.) | pip install flashinfer-python | sm75+ (Ampere/Ada/Hopper/Blackwell) | fp32/fp16/bf16 | Sorting-free nucleus via dual-pivot threshold search; no cumulative-sum-over-sorted-vocab needed. O(V) memory-bound, eliminates the ~20% sort overhead seen in torch-native top-p at high throughput. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_p_renorm_probs.html) |
| 2 | vllm | from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p; apply_top_k_top_p(logits, k, p) # sort-based: sorts logits, cumsum softmax, mask tokens beyond p (and below kth) | pip install vllm | any CUDA SM; CPU/ROCm/XPU fallbacks | fp32/fp16/bf16 | Exact sort-based top-k+top-p (torch.sort + cumsum). Correct and portable but pays O(V log V) sort cost; used as native fallback. Statistically identical, unlike rejection-based kernels. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py) |

### `min_p_filter`

Min-p filtering: keep tokens with prob >= min_p * max_prob (equivalently logit >= max_logit + log(min_p)), mask the rest.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.sampling.min_p_sampling_from_probs(probs, min_p, indices=None, deterministic=True, generator=None, seed=None, offset=None) -> sampled_token_ids[batch]  # min_p: float\|Tensor[batch]; fused filter+sample | pip install flashinfer-python | sm75+ (Ampere/Ada/Hopper/Blackwell) | fp32/fp16/bf16 probs | Single-CUDA-kernel rejection sampling fuses min-p threshold + sampling, no sort. Fastest production min-p path; multiple rejection rounds fused in one launch. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.sampling.min_p_sampling_from_probs.html) |
| 2 | vllm | from vllm.v1.worker.gpu.sample.min_p import apply_min_p; apply_min_p(logits, expanded_idx_mapping, min_p)  # in-place Triton @triton.jit _min_p_kernel: threshold = max_logit + log(min_p), set below to -inf | pip install vllm | CUDA sm70+ (Triton); ROCm via Triton | fp32/fp16/bf16 | Dedicated Triton kernel: two-pass (find max, then threshold-mask) with 1024-wide blocks, fully parallel over vocab. Decouples min-p masking from the sampler; exact (not rejection). _(confidence: high)_ | [link](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu/sample/min_p/) |
| 3 | sgl-kernel | from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob; renorm-by-threshold then sample. (min-p sampling is exposed as min_p_sampling_from_probs on the MUSA build; the CUDA path composes renorm primitives.) | pip install sgl-kernel | sm80+ (Ampere+); HIP/ROCm | fp32 probs | SGLang fused renorm-by-threshold sampling primitives (FlashInfer-derived); min-p realized via the renorm+sample composition. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/sampling.py) |

### `typical_sampling`

Locally-typical sampling (Meister et al.): keep tokens whose surprisal is closest to the conditional entropy; mask atypical tokens.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | transformers | from transformers import TypicalLogitsWarper; warper = TypicalLogitsWarper(mass=0.9, filter_value=-inf, min_tokens_to_keep=1); logits = warper(input_ids, scores) # entropy-based typical filtering | pip install transformers | any CUDA SM; CPU/ROCm/XPU (torch ops) | fp32/fp16/bf16 | Reference torch implementation (sort by \|logp - entropy\|, cumulative mass cutoff). No fused CUDA kernel exists in FlashInfer/SGLang for typical sampling; HF Transformers is the de-facto correct impl that vLLM/SGLang mirror when supported. _(confidence: medium)_ | [link](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py) |

### `repetition_presence_frequency_penalty`

Penalize tokens already seen: repetition_penalty (divide/scale logits), presence_penalty (subtract constant if seen), frequency_penalty (subtract count*coef). Applied over prompt+output tokens.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm.v1.sample.ops.penalties import apply_all_penalties; logits = apply_all_penalties(logits, prompt_token_ids, presence_penalties, frequency_penalties, repetition_penalties, output_token_ids) # output_token_ids: list[list[int]]; delegates to vllm.model_executor.layers.utils.apply_penalties | pip install vllm | any CUDA SM (sm70+); CPU/ROCm/XPU | fp32 logits | Fused bin-count (get_token_bin_counts_and_mask) computes per-token presence+frequency masks once, then applies all three penalties in vectorized torch ops on GPU. Standard production penalty path; correct handling of prompt vs output tokens. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/penalties.py) |
| 2 | sglang | sglang.srt.sampling.penaltylib orchestrator: BatchedRepetitionPenalizer / BatchedPresencePenalizer / BatchedFrequencyPenalizer .apply(logits) # batched penalizers maintained per running batch | pip install 'sglang[all]' | any CUDA SM; ROCm/CPU | fp32 logits | Batched penalizer framework that incrementally maintains token-count tensors across decode steps (avoids recomputing bincounts each step). Strong for long-running batches. _(confidence: medium)_ | [link](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/sampling/penaltylib) |
| 3 | transformers | from transformers import RepetitionPenaltyLogitsProcessor; proc = RepetitionPenaltyLogitsProcessor(penalty=1.1); scores = proc(input_ids, scores) # also (Encoder)NoRepeatNGram, presence/frequency via custom | pip install transformers | any CUDA SM; CPU/ROCm/XPU | fp32/fp16/bf16 | Reference torch implementation; correct but not batched-incremental. Useful as the canonical semantics reference. _(confidence: high)_ | [link](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py) |

### `logit_bias_badwords_mask`

Apply additive per-token logit_bias and hard-ban bad-words token sequences (set last-completing token logit to -inf) and min-tokens (ban EOS until min length).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm.v1.sample.logits_processor.builtin import LogitBiasLogitsProcessor, MinTokensLogitsProcessor; proc.apply(logits) # bad-words via get_bad_words_logits_processors(); builtin processors loaded at engine start | pip install vllm | any CUDA SM; CPU/ROCm/XPU | fp32 logits | Batch-aware builtin LogitsProcessor subclasses (LogitBias, MinTokens, MinP) that scatter -inf/bias into logits in vectorized torch ops; integrated with the v1 logits-processor programming model. Production-grade. _(confidence: high)_ | [link](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/logits_processor/builtin.html) |
| 2 | transformers | from transformers import SequenceBiasLogitsProcessor, NoBadWordsLogitsProcessor, MinNewTokensLengthLogitsProcessor; scores = proc(input_ids, scores) | pip install transformers | any CUDA SM; CPU/ROCm/XPU | fp32/fp16/bf16 | Canonical reference for sequence-bias / bad-words / min-tokens semantics (only ban last completing token of a banned sequence). torch ops, not fused. _(confidence: high)_ | [link](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py) |

### `guided_grammar_mask`

Structured-output constrained decoding: compute a per-step token bitmask (allowed tokens) from a grammar/JSON-schema FSM and apply it to logits (disallowed -> -inf).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | xgrammar | import xgrammar as xgr; bitmask = xgr.allocate_token_bitmask(batch, vocab_size); matcher.fill_next_token_bitmask(bitmask, index=0); xgr.apply_token_bitmask_inplace(logits, bitmask.to('cuda'), vocab_size=V, backend='cuda') # masked logits -> -inf | pip install xgrammar | CUDA sm70+ (cuda/triton backends); cpu/torch_native fallback | fp32/fp16/bf16 logits; int32 packed bitmask (vocab/32) | Default structured-output backend in vLLM and SGLang. Pre-computes context-independent token masks at compile time, validates only context-dependent tokens per step; dedicated CUDA bitmask-apply kernel. Fastest end-to-end for JSON-schema/EBNF. _(confidence: high)_ | [link](https://xgrammar.mlc.ai/docs/api/python/bitmask_ops.html) |
| 2 | llguidance | import llguidance; matcher.compute_mask() / fill_next_token_bitmask(...) -> bitmask; apply to logits (set disallowed -> -inf). Lazy lexer automata; ~50us/step CPU mask for 128k vocab | pip install llguidance | CPU mask compute (host); GPU apply via engine bitmask kernel | logits fp32/fp16/bf16; packed bitmask | Zero startup cost (lazy automata), ~50us single-core mask time for 128k-token vocab. Best when grammars change frequently / startup latency matters; alternative backend in vLLM and SGLang. _(confidence: medium)_ | [link](https://github.com/guidance-ai/llguidance) |
| 3 | outlines-core | from outlines_core import Guide, Index; guide.get_tokens()/advance(token) -> allowed token set -> build mask, apply -inf to logits # Rust FSM-based regex/JSON-schema guidance | pip install outlines-core | CPU FSM compute; GPU apply via engine | logits fp32/fp16/bf16 | Rust-core FSM with index precomputation; mature regex/JSON-schema support. Heavier precompute than llguidance but well-tested; used as a vLLM/SGLang backend option. _(confidence: medium)_ | [link](https://github.com/dottxt-ai/outlines-core) |

### `categorical_sample_from_probs`

Draw one token per row from a categorical distribution (probs already filtered/renormalized) using reproducible counter-based RNG (Gumbel/inverse-CDF).

**kernel_set_abi:** `ks_sample`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.sampling.sampling_from_probs(probs, indices=None, deterministic=True, generator=None, seed=None, offset=None) -> token_ids[batch]  # also sampling_from_logits(logits, ...) for fused softmax+sample | pip install flashinfer-python | sm75+ (Ampere/Ada/Hopper/Blackwell) | fp32/fp16/bf16 probs | Single-kernel inverse-transform sampling with counter-based (philox seed/offset) RNG for reproducibility; no torch.multinomial CPU-GPU sync. The canonical fast categorical sampler. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.sampling.sampling_from_logits.html) |
| 2 | vllm | from vllm.v1.sample.ops.topk_topp_sampler import random_sample; random_sample(probs, generators)  # Gumbel/exponential trick (probs / -log(uniform)).argmax to avoid torch.multinomial sync | pip install vllm | any CUDA SM; CPU/ROCm/XPU | fp32/fp16/bf16 | Exponential-distribution (Gumbel-max) sampling with per-request torch.Generator, deliberately avoiding torch.multinomial's CPU-GPU sync. Native fallback when FlashInfer is off. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py) |
| 3 | sgl-kernel | from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob; combine with categorical sampling. On MUSA: top_p_sampling_from_probs(probs, top_p, indices=None, deterministic=True, generator=None). | pip install sgl-kernel | sm80+ (Ampere+); HIP/ROCm | fp32 probs; samples int32 | SGLang categorical sampling primitives; alignment target for ks_sample. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/sampling.py) |

### `fused_temp_topk_topp_sample`

End-to-end fused sampler: per-request temperature + top-k + top-p (and optionally min-p) filtering followed by categorical sampling, in one kernel/path. Maps directly to ks_sample.

**kernel_set_abi:** `ks_sample`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k, top_p, indices=None, filter_apply_order='top_k_first', deterministic=True, generator=None, seed=None, offset=None)  # or top_k_top_p_sampling_from_probs(probs, top_k, top_p, ...). top_k:int\|Tensor, top_p:float\|Tensor | pip install flashinfer-python | sm75+ (Ampere/Ada/Hopper/Blackwell tuned) | fp32/fp16/bf16 | Dual-pivot sorting-free rejection sampling: fuses filter+sample in a single CUDA kernel, O(log(1/eps)) rounds, per-request k/p tensors. Reduces sampling time >50% vs torch.sort path on H100; near-zero overhead in vLLM v1 (PR #11394). The best-in-class fused sampler. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_probs.html) |
| 2 | sgl-kernel | from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob; top_k_first order: renorm_probs = top_k_renorm_prob(probs, top_k); top_p_renorm_prob(renorm_probs, top_p) then sample. (joint top_k_top_p_sampling_from_probs on the MUSA build.) | pip install sgl-kernel | sm80+ (Ampere+); HIP/ROCm | fp32 probs; top_k int (or tensor); top_p float (or tensor) | SGLang fused temperature/top-k/top-p sampling path used in production serving; alignment target for kernel-set's ks_sample. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/sampling.py) |
| 3 | vllm | from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler; sampler = TopKTopPSampler(); tokens, _ = sampler(logits, generators, k, p)  # forward_cuda dispatches to flashinfer_sample or native apply_top_k_top_p+random_sample | pip install vllm | CUDA (flashinfer/native), CPU, ROCm(aiter), XPU | fp32/fp16/bf16 | Production orchestrator: picks FlashInfer rejection kernel when VLLM_USE_FLASHINFER_SAMPLER + CUDA, else exact sort-based native path; handles per-request generators and logprobs_mode. The integration layer most engines actually call. _(confidence: high)_ | [link](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/ops/topk_topp_sampler/) |
| 4 | sglang | sglang.srt.layers.sampler.Sampler.forward(...) -> flashinfer top_k_top_p_sampling_from_probs / min_p_sampling_from_probs (backend='flashinfer') or top_k_top_p_min_p_sampling_from_probs_torch (backend='pytorch'); renorm via sgl_kernel.top_k_renorm_prob/top_p_renorm_prob | pip install 'sglang[all]' | CUDA sm80+ (flashinfer/sgl_kernel); Ascend NPU path; torch fallback | fp32/fp16/bf16 | Server-args-selected backend (flashinfer/pytorch/ascend). Adds combined top-k+top-p+min-p in one path (torch fallback fuses all three) plus deterministic RL on-policy multinomial_with_seed. Strong full-featured sampler. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/sampler.py) |

### `speculative_verify_rejection_chain`

Chain (linear) speculative decoding verification: rejection-sample draft tokens against target probs, accept/reject per token, sample a correction from the residual distribution, emit a bonus token if all accepted.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.sampling.chain_speculative_sampling(draft_probs, draft_token_ids, target_probs, maybe_output_accepted_token_num=None, maybe_output_emitted_draft_token_num=None, deterministic=True, generator=None, seed=None, offset=None) -> (output_token_ids[batch, num_spec+1], accepted_num[batch], emitted_num[batch]) | pip install flashinfer-python | sm75+ (Ampere/Ada/Hopper/Blackwell) | fp32/fp16/bf16 probs; int32 token ids | Single-kernel theoretically-lossless rejection verification for linear speculative decoding (draft model). Returns acceptance/emission counts for adaptive spec length. Best-in-class for chain (non-tree) verify. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.sampling.chain_speculative_sampling.html) |
| 2 | vllm | vllm.v1.spec_decode rejection sampler -> uses flashinfer.sampling.chain_speculative_sampling under the hood (VLLM_USE_FLASHINFER) or a Triton rejection_sample kernel fallback | pip install vllm | CUDA sm70+ (Triton fallback); flashinfer path sm75+ | fp32/fp16/bf16 | Production rejection sampler with greedy and stochastic verify paths; Triton kernel fallback when FlashInfer absent. Integrated with EAGLE/ngram/medusa proposers. _(confidence: medium)_ | [link](https://github.com/vllm-project/vllm/tree/main/vllm/v1/spec_decode) |

### `speculative_verify_tree`

Tree-based speculative verification (EAGLE/Medusa): verify a token tree against target probs, traverse accepted path via retrieve indices, emit accepted tokens + bonus. Supports stochastic (rejection) and greedy verify.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | sgl-kernel | from sgl_kernel import tree_speculative_sampling_target_only; tree_speculative_sampling_target_only(predicts, accept_index, accept_token_num, candidates, retrive_index, retrive_next_token, retrive_next_sibling, uniform_samples, uniform_samples_for_final_sampling, target_probs, draft_probs, threshold_single=1.0, threshold_acc=1.0, deterministic=True) # in-place; greedy: verify_tree_greedy(predicts, accept_index, accept_token_num, candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict) | pip install sgl-kernel | CUDA sm80+ (Ampere/Hopper/Blackwell); ROCm not yet (falls back to greedy) | fp32/fp16/bf16 probs; int32 indices | Purpose-built EAGLE tree-verify CUDA kernels: traverse the draft token tree (retrive_next_token/sibling), rejection-verify against target_probs in one kernel, returning accept_index/accept_token_num. The reference high-performance tree verify used by SGLang EAGLE/EAGLE3. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/speculative.py) |
| 2 | vllm | vllm.v1.spec_decode.eagle / rejection_sampler tree verify path (EagleProposer + RejectionSampler); flat-tree verify integrated in v1 spec decode | pip install vllm | CUDA sm70+ (Triton); flashinfer chain path sm75+ | fp32/fp16/bf16 | vLLM v1 EAGLE/EAGLE3 + Medusa support with rejection-sampler verify; uses flattened proposal verification. Production-integrated though tree kernels are less specialized than sgl-kernel's dedicated tree-verify. _(confidence: medium)_ | [link](https://github.com/vllm-project/vllm/tree/main/vllm/v1/spec_decode) |

### `topk_topp_renorm_probs`

Renormalize a probability distribution after top-k or top-p truncation (rescale kept mass to sum to 1) without sampling. Used as a building block in fused samplers and spec-decode (sgl_kernel.top_k_renorm_prob / top_p_renorm_prob).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | flashinfer-python | flashinfer.sampling.top_k_renorm_probs(probs, top_k)  # and flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=False) -> renormalized probs | pip install flashinfer-python | sm75+ (Ampere/Ada/Hopper/Blackwell) | fp32/fp16/bf16 | Sorting-free renorm via pivot threshold; single O(V) pass. SGLang re-exports these as sgl_kernel.top_k_renorm_prob/top_p_renorm_prob and uses them in EAGLE verify. Best-in-class. _(confidence: high)_ | [link](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_renorm_probs.html) |
| 2 | sgl-kernel | from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob; top_k_renorm_prob(probs, top_k) -> Tensor; top_p_renorm_prob(probs, top_p) -> Tensor. Renormalizes probs after top-k / top-p thresholding. | pip install sgl-kernel | sm80+ (Ampere+); HIP/ROCm | fp32 probs | SGLang fused top-k / top-p renorm-probs kernels (FlashInfer-derived); compose with sampling for top-k/top-p decoding. _(confidence: high)_ | [link](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/sampling.py) |

---

## Loss / Optimizer / Misc — `loss-optim-misc`

23 operators, 38 providers.

### `cross_entropy_fused`

Fused cross-entropy forward+backward over [num_tokens, vocab] logits with softmax-minus-onehot written in-place; supports label smoothing, ignore_index, z-loss (lse_square_scale), and logit softcapping.

**kernel_set_abi:** `ks_cross_entropy`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | liger-kernel | from liger_kernel.transformers.functional import liger_cross_entropy; loss, z_loss = liger_cross_entropy(logits, target, ignore_index=-100, label_smoothing=0.0, lse_square_scale=0.0, softcap=None, return_z_loss=False, reduction='mean') # or nn.Module: from liger_kernel.transformers import LigerCrossEntropyLoss | pip install liger-kernel | sm70+ (Triton); ROCm supported (torch>=2.5, triton>=3.0) | fp32/bf16/fp16 (fp32 accumulation in-kernel) | In-place gradient (writes softmax-onehot back over logits) avoids a separate grad buffer; ~2-4x faster and large memory savings vs torch.nn.CrossEntropyLoss for big vocab. Native z-loss + softcap. _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py) |
| 2 | torch (eager/compile) | import torch.nn.functional as F; loss = F.cross_entropy(logits, target, ignore_index=-100, label_smoothing=0.0, reduction='mean') # wrap in torch.compile for fusion | pip install torch | any CUDA SM; CPU; ROCm; MPS | fp32/bf16/fp16 | Reference baseline; torch.compile can fuse but still materializes full logits (no z-loss/softcap built in). _(confidence: high)_ | [link](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) |

### `fused_linear_cross_entropy`

Computes CE from hidden states + LM-head weight in token-chunks WITHOUT materializing the full [num_tokens, vocab] logits; returns loss and grads for hidden + weight. The key memory-saver for large-vocab LM training.

**kernel_set_abi:** `ks_fused_linear_cross_entropy`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | cut-cross-entropy | from cut_cross_entropy import linear_cross_entropy; loss = linear_cross_entropy(embeddings, classifier, labels, bias=None, shift=0, reduction='mean', ignore_index=-100, impl='cce', filter_eps=1e-7) # patch HF model: from cut_cross_entropy.transformers import cce_patch; model = cce_patch(model, impl='cce') | pip install 'cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git' # or unsloth fork: pip install 'cut-cross-entropy @ git+https://github.com/unslothai/cut-cross-entropy.git' | sm80+ (Ampere or newer; Triton 3.0+). MacOS falls back to torch_compile impl. | bf16/fp16 with fp32 for unstable ops (logit/log-sum-exp computed on the fly, never stored) | Never materializes logits at all: only computes the correct-token logit + on-the-fly LSE. Reduces loss memory from ~24GB to ~1MB on Gemma-2 (2B), classifier-head training memory 28GB->1GB. Best-in-class for very large vocab. _(confidence: high)_ | [link](https://github.com/apple/ml-cross-entropy) |
| 2 | liger-kernel | from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy; loss, z_loss = liger_fused_linear_cross_entropy(input, weight, target, bias=None, ce_weight=None, ignore_index=-100, lse_square_scale=0.0, label_smoothing=0.0, reduction='mean', softcap=None, return_z_loss=False) # nn.Module: from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss; loss_fn(weight, input, target) | pip install liger-kernel | sm70+ (Triton); ROCm supported | fp32/bf16/fp16 (fp32 accum) | Chunks over the token dim and fuses the LM-head matmul with CE; 60-80% memory reduction vs unfused. Adds z-loss + softcap + label smoothing. Note class call order is (weight, input, target). _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py) |
| 3 | linear_cross_entropy_loss (Geiping) | from linear_cross_entropy import linear_cross_entropy; loss = linear_cross_entropy(x, weight, labels, ignore_index=-100, reduction='mean') | pip install git+https://github.com/JonasGeiping/linear_cross_entropy_loss | sm80+ (Triton) | bf16/fp16/fp32 | Single-file Triton fusion of linear layer + CE; good drop-in but less maintained / fewer features than CCE and Liger. _(confidence: medium)_ | [link](https://github.com/JonasGeiping/linear_cross_entropy_loss) |

### `z_loss`

Auxiliary stabilization loss = lse_square_scale * (logsumexp(logits))^2 added to CE (from PaLM); penalizes large logits. Exposed as a knob inside fused CE rather than a standalone kernel.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | liger-kernel | from liger_kernel.transformers.functional import liger_cross_entropy; loss, z_loss = liger_cross_entropy(logits, target, lse_square_scale=1e-4, return_z_loss=True) | pip install liger-kernel | sm70+ (Triton) | fp32/bf16/fp16 | Computed inside the same CE kernel from the already-computed LSE; zero extra passes. The standard way to get PaLM-style z-loss fused. _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py) |

### `kl_divergence`

KL divergence loss for distillation / RLHF; KL(student || teacher) over log-prob distributions.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | liger-kernel | from liger_kernel.transformers import LigerKLDIVLoss; loss = LigerKLDIVLoss(reduction='batchmean', log_target=False)(student_log_probs, target) # functional: liger_kernel.transformers.functional.liger_kl_div | pip install liger-kernel | sm70+ (Triton) | fp32/bf16/fp16 | Fused Triton KLDiv; drop-in for torch.nn.KLDivLoss with lower memory for large-vocab distillation. _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel) |

### `jsd_distillation`

Jensen-Shannon divergence loss and the fused-linear JSD (computes JSD from hidden + weight without materializing logits) for generalized knowledge distillation (GKD).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | liger-kernel | from liger_kernel.transformers import LigerJSD, LigerFusedLinearJSD; loss = LigerFusedLinearJSD(jsd_beta=0.5, ignore_index=-100, temperature=1.0)(student_input, student_weight, teacher_input, teacher_weight, target) # functional: liger_jsd, liger_fused_linear_jsd | pip install liger-kernel | sm70+ (Triton) | fp32/bf16/fp16 | Fused-linear JSD avoids materializing student/teacher logits — the distillation analogue of FLCE. Best-in-class for GKD-style distillation. _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel) |

### `tvd_loss`

Total Variation Distance loss between distributions (alternative distillation divergence).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | liger-kernel | from liger_kernel.transformers import LigerTVDLoss; loss = LigerTVDLoss(reduction='batchmean')(student, target) # functional: liger_kernel.transformers.functional.liger_tvd | pip install liger-kernel | sm70+ (Triton) | fp32/bf16/fp16 | Fused Triton TVD; rare to find a dedicated kernel for this. _(confidence: medium)_ | [link](https://github.com/linkedin/Liger-Kernel) |

### `dpo_loss`

Direct Preference Optimization preference loss, fused with the linear head so chosen/rejected logits are never fully materialized (chunked).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | liger-kernel | from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss; loss_fn = LigerFusedLinearDPOLoss(ignore_index=-100, beta=0.1); loss = loss_fn(lin_weight, student_input, target, bias=None, ref_input=..., ref_weight=...) | pip install liger-kernel | sm70+ (Triton) | fp32/bf16/fp16 | Chunked fused-linear preference loss: large memory savings on the doubled (chosen+rejected) sequence vs TRL's eager DPO. Best-in-class fused DPO. _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel/tree/main/src/liger_kernel/chunked_loss) |
| 2 | trl | from trl import DPOTrainer, DPOConfig # eager reference impl; can enable use_liger_loss=True in DPOConfig to route to Liger kernels | pip install trl | any (eager torch); Liger path needs Triton GPU | fp32/bf16/fp16 | Canonical reference DPO; integrates Liger fused loss via use_liger_loss flag. _(confidence: high)_ | [link](https://github.com/huggingface/trl) |

### `orpo_loss`

Odds-Ratio Preference Optimization loss, fused with the linear head (chunked, reference-model-free).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | liger-kernel | from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss; loss_fn = LigerFusedLinearORPOLoss(ignore_index=-100, beta=0.1); loss = loss_fn(lin_weight, student_input, target, bias=None) | pip install liger-kernel | sm70+ (Triton) | fp32/bf16/fp16 | Reference-free ORPO fused-linear chunked loss; major memory win vs eager. _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel/tree/main/src/liger_kernel/chunked_loss) |

### `preference_losses_simpo_cpo_kto_grpo`

Other fused-linear chunked preference / RL losses: SimPO, CPO, KTO, GRPO, cosine-similarity. Same chunked-head trick as DPO/ORPO.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | liger-kernel | from liger_kernel.chunked_loss import LigerFusedLinearSimPOLoss, LigerFusedLinearCPOLoss, LigerFusedLinearKTOLoss, LigerFusedLinearGRPOLoss, LigerFusedLinearCosineSimilarityLoss | pip install liger-kernel | sm70+ (Triton) | fp32/bf16/fp16 | Single library covering the full preference/RL loss zoo with the fused-linear chunking memory optimization; GRPO is the RLHF-for-reasoning loss used widely in 2025-26. _(confidence: high)_ | [link](https://github.com/linkedin/Liger-Kernel/tree/main/src/liger_kernel/chunked_loss) |

### `adamw_fused`

Fused AdamW step (decoupled weight decay) updating a parameter tensor in place; multi-tensor batched launch over all params; optional fp32 master weights.

**kernel_set_abi:** `ks_adamw`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | apex | from apex.optimizers import FusedAdam; opt = FusedAdam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, adam_w_mode=True) # backend: amp_C.multi_tensor_adam via multi_tensor_applier | pip install -v --no-build-isolation --config-settings '--build-option=--cpp_ext' --config-settings '--build-option=--cuda_ext' git+https://github.com/NVIDIA/apex.git | sm70+ (CUDA only) | fp32/bf16/fp16 params; fp32 optimizer state | multi_tensor_applier batches the elementwise Adam update across all params into one/few kernel launches; the long-standing fastest fused AdamW. adam_w_mode=True gives decoupled WD. _(confidence: high)_ | [link](https://github.com/NVIDIA/apex/blob/master/apex/optimizers/fused_adam.py) |
| 2 | torch (fused) | import torch; opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9,0.999), weight_decay=0.01, fused=True) # or foreach=True for multi-tensor without a fused CUDA kernel | pip install torch | sm70+ (CUDA fused path); CPU/ROCm via foreach | fp32/bf16/fp16 | Native fused CUDA AdamW (fused=True) competitive with apex and dependency-free; supports capturable for CUDA graphs. _(confidence: high)_ | [link](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) |
| 3 | deepspeed | from deepspeed.ops.adam import FusedAdam; opt = FusedAdam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8, adam_w_mode=True, weight_decay=0.01) # CPU offload: from deepspeed.ops.adam import DeepSpeedCPUAdam | pip install deepspeed | sm70+ (GPU FusedAdam); DeepSpeedCPUAdam is AVX/SIMD CPU (for ZeRO-Offload) | fp32/bf16/fp16 params; fp32 state | DeepSpeedCPUAdam gives 5-7x speedup over torch CPU Adam for ZeRO-Offload; GPU FusedAdam mirrors apex. Best when using ZeRO. _(confidence: high)_ | [link](https://deepspeed.readthedocs.io/en/latest/optimizers.html) |

### `adamw_8bit`

8-bit (block-wise quantized) AdamW optimizer state to cut optimizer memory ~4x vs fp32 while matching fp32 quality.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | bitsandbytes | import bitsandbytes as bnb; opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3, betas=(0.9,0.999), weight_decay=0.01, min_8bit_size=4096) # also PagedAdamW8bit for CPU paging | pip install bitsandbytes | sm60+ (CUDA cc 6.0+) | fp32/bf16/fp16 params; 8-bit block-wise quantized state | Original block-wise 8-bit optimizers; near-zero quality loss at ~25% optimizer memory. PagedAdamW8bit uses CUDA unified memory to page state to CPU on OOM. The de-facto standard. _(confidence: high)_ | [link](https://huggingface.co/docs/bitsandbytes/main/en/optimizers) |
| 2 | torchao | from torchao.optim import AdamW8bit, AdamW4bit, AdamWFp8; opt = AdamW8bit(model.parameters(), lr=1e-3, betas=(0.9,0.999), weight_decay=0.01) | pip install torchao | sm80+ recommended (fp8 path needs sm89/sm90); pure-PyTorch + torch.compile | 8-bit, 4-bit, and fp8 (e4m3) quantized optimizer state | Pure-PyTorch, torch.compiled — competitive with bnb CUDA kernels and adds 4-bit (8x state reduction) and fp8 variants. Also CPUOffloadOptimizer(params, torch.optim.AdamW, fused=True) for ~60% VRAM cut. _(confidence: high)_ | [link](https://github.com/pytorch/ao/tree/main/torchao/optim) |

### `lion_optimizer`

Lion (EvoLved Sign Momentum) optimizer step — sign-based update, lower memory than Adam (single momentum state), and its 8-bit variant.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | bitsandbytes | import bitsandbytes as bnb; opt = bnb.optim.Lion8bit(model.parameters(), lr=1e-4, betas=(0.9,0.99), weight_decay=0.0) # full-precision: bnb.optim.Lion or PagedLion8bit | pip install bitsandbytes | sm60+ | fp32/bf16/fp16 params; 8-bit state | 8-bit Lion: only one momentum buffer, quantized — lowest-memory adaptive optimizer in production use. _(confidence: high)_ | [link](https://huggingface.co/docs/bitsandbytes/main/en/optimizers) |
| 2 | lion-pytorch | from lion_pytorch import Lion; opt = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2, betas=(0.9,0.99)) # use_triton=True for fused Triton kernel | pip install lion-pytorch | any; use_triton path needs Triton GPU | fp32/bf16/fp16 | Reference Lion impl from the paper authors' community; optional fused Triton kernel via use_triton=True. _(confidence: high)_ | [link](https://github.com/lucidrains/lion-pytorch) |

### `adafactor_optimizer`

Adafactor — sublinear-memory adaptive optimizer using factored second-moment estimates (rank-1 row/col stats), avoiding a full second-moment tensor.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | transformers | from transformers.optimization import Adafactor; opt = Adafactor(model.parameters(), lr=None, scale_parameter=True, relative_step=True, warmup_init=True) # for external LR: scale_parameter=False, relative_step=False, lr=... | pip install transformers | any (eager torch) | fp32/bf16/fp16 | The standard production Adafactor; factored second moments => much lower optimizer memory than Adam (no full vsq tensor). _(confidence: high)_ | [link](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py) |
| 2 | torchao | from torchao.optim import _AdamW # torchao focuses on AdamW8bit/4bit/Fp8; use transformers Adafactor for factored second moment | pip install torchao | sm80+ recommended | 8/4-bit/fp8 | Listed only as an alternative low-memory optimizer source; not a true Adafactor — prefer transformers.Adafactor for factored variance. _(confidence: low)_ | [link](https://github.com/pytorch/ao/tree/main/torchao/optim) |

### `muon_optimizer`

Muon — momentum SGD followed by Newton-Schulz orthogonalization of the 2D weight update; for hidden-layer matrices, with AdamW for 1D params/embeddings/head. ~2x compute efficiency vs AdamW at scale.

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | Moonlight (MoonshotAI scalable Muon) | from muon import Muon; opt = Muon(lr=2e-2, wd=0.1, momentum=0.95, muon_params=hidden_2d_params, adamw_params=other_params) # 'Muon is Scalable': adds weight decay + per-param update scaling; class also vendored in Moonlight/examples/toy_train.py | pip install git+https://github.com/MoonshotAI/Moonlight.git # or copy the Muon class from examples/toy_train.py | any CUDA (distributed/memory-optimal variant for multi-GPU) | bf16/fp16/fp32 (Newton-Schulz iterations in bf16) | Production-scaled Muon (Kimi/Moonlight 16B-A3B, 5.7T tokens): weight decay + update-scale fix make it work out-of-the-box at scale, ~2x compute efficiency vs AdamW. Memory-optimal, communication-efficient distributed impl. _(confidence: medium)_ | [link](https://github.com/MoonshotAI/Moonlight) |
| 2 | Muon (Keller Jordan) | from muon import MuonWithAuxAdam; opt = MuonWithAuxAdam([{'params': hidden_weights, 'use_muon': True, 'lr': 0.02}, {'params': other, 'use_muon': False, 'lr': 3e-4, 'betas': (0.9,0.95)}]) | pip install git+https://github.com/KellerJordan/Muon # PyPI: pip install muon-optimizer | any CUDA | bf16/fp16/fp32 | Original Muon; holds NanoGPT & CIFAR-10 speedrun records (1.35x faster NanoGPT). MuonWithAuxAdam handles param-group routing automatically. _(confidence: high)_ | [link](https://github.com/KellerJordan/Muon) |

### `sgd_momentum_fused`

Fused SGD with (optional Nesterov) momentum and weight decay; multi-tensor batched launch.

**kernel_set_abi:** `ks_sgd_momentum`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | apex | from apex.optimizers import FusedSGD; opt = FusedSGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True) | pip install -v --no-build-isolation --config-settings '--build-option=--cuda_ext' git+https://github.com/NVIDIA/apex.git | sm70+ (CUDA) | fp32/bf16/fp16 | multi_tensor_applier SGD; batches updates across all params into one launch. _(confidence: high)_ | [link](https://nvidia.github.io/apex/optimizers.html) |
| 2 | torch (fused) | import torch; opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True, fused=True) | pip install torch | sm70+ (fused CUDA); CPU/ROCm via foreach | fp32/bf16/fp16 | Native fused/foreach SGD; dependency-free and on par with apex. _(confidence: high)_ | [link](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) |

### `global_grad_norm_clip`

Compute global L2 norm across all grad tensors (multi-tensor) and clip in place — sqrt(sum ||g||^2), then scale.

**kernel_set_abi:** `ks_global_grad_norm`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | apex | import amp_C; from apex.multi_tensor_apply import multi_tensor_applier; norm, _ = multi_tensor_applier(amp_C.multi_tensor_l2norm, overflow_buf, [grads], True) # also amp_C.multi_tensor_scale to apply the clip coefficient | pip install -v --no-build-isolation --config-settings '--build-option=--cuda_ext' git+https://github.com/NVIDIA/apex.git | sm70+ (CUDA) | fp32/bf16/fp16 | multi_tensor_l2norm + multi_tensor_scale: single-launch global norm + clip over thousands of tensors; the kernel Megatron uses for grad clipping. _(confidence: high)_ | [link](https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_l2norm_kernel.cu) |
| 2 | torch | import torch; total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0, foreach=True) | pip install torch | any (foreach multi-tensor on CUDA) | fp32/bf16/fp16 | foreach=True uses torch._foreach_norm for a multi-tensor reduction; dependency-free and standard. _(confidence: high)_ | [link](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) |

### `embedding_lookup`

Gather rows of an embedding table by token indices: out[i,:] = table[indices[i],:].

**kernel_set_abi:** `ks_embedding_lookup`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torch | import torch.nn.functional as F; out = F.embedding(indices, table, padding_idx=None) # nn.Embedding wraps this | pip install torch | any (CUDA gather kernel) | fp32/bf16/fp16/int | Highly optimized native gather; the embedding lookup baseline everyone uses. _(confidence: high)_ | [link](https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html) |
| 2 | fbgemm-gpu (TorchRec) | from fbgemm_gpu.split_table_batched_embeddings_ops_training import SplitTableBatchedEmbeddingBagsCodegen; emb = SplitTableBatchedEmbeddingBagsCodegen(embedding_specs=...); out = emb(indices, offsets) | pip install fbgemm-gpu | sm70+ (CUDA); also CPU | fp32/fp16/int8/int4 (quantized embeddings) | Best-in-class for huge batched/sharded embedding tables (recsys-scale): UVM, quantized rows, fused optimizer in the embedding backward. Overkill for plain LM embeddings. _(confidence: high)_ | [link](https://github.com/pytorch/FBGEMM) |

### `embedding_backward_scatter`

Scatter-add gradients back into the embedding table: grad_table[indices[i],:] += grad_out[i,:] (atomic / sorted-segment-reduce).

**kernel_set_abi:** `ks_embedding_backward`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torch (autograd) | torch.ops.aten.embedding_dense_backward(grad_out, indices, num_weights, padding_idx, scale_grad_by_freq) # normally invoked by autograd of F.embedding | pip install torch | any (CUDA) | fp32/bf16/fp16 | Native sorted-segment scatter-add backward; correct and fast for LM-sized vocabs. _(confidence: high)_ | [link](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Embedding.cu) |
| 2 | fbgemm-gpu (TorchRec) | SplitTableBatchedEmbeddingBagsCodegen(...).backward via autograd # fuses the optimizer step into the embedding backward (no separate grad table) | pip install fbgemm-gpu | sm70+ | fp32/fp16/int8 | Fuses optimizer update into the scatter backward, avoiding a materialized grad table for billion-row embeddings. _(confidence: high)_ | [link](https://github.com/pytorch/FBGEMM) |

### `dtype_cast`

Elementwise dtype conversion (e.g. bf16<->fp32, fp32->fp8) with vectorized IO.

**kernel_set_abi:** `ks_cast`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torch | out = x.to(torch.bfloat16) # or torch.float8_e4m3fn; fused under torch.compile | pip install torch | any; fp8 needs sm89/sm90 storage support | fp32/bf16/fp16/fp8(e4m3,e5m2)/int8/int4 | Native vectorized cast; torch.compile fuses casts into surrounding ops. _(confidence: high)_ | [link](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html) |

### `axpby_fused`

Fused out = a*alpha + b*beta (used for residual scaling, EMA, master-weight updates).

**kernel_set_abi:** `ks_axpby`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | torch | torch.add(a, b, alpha=beta, out=out) # for general alpha*a+beta*b use a.mul(alpha).add_(b, alpha=beta) or torch._foreach_add_ for multi-tensor | pip install torch | any | fp32/bf16/fp16 | torch._foreach_* gives a true multi-tensor fused AXPBY for EMA/master-weight loops; torch.compile fuses the scalar case. _(confidence: high)_ | [link](https://pytorch.org/docs/stable/generated/torch._foreach_add_.html) |

### `kv_cache_reshape_and_cache`

Scatter K/V vectors into a paged KV cache by slot mapping, optionally quantizing to fp8 with per-tensor/per-head scales; the inference-time KV write kernel.

**kernel_set_abi:** `ks_reshape_and_cache`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm import _custom_ops as ops; ops.reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale) # non-flash layout: ops.reshape_and_cache(...) | pip install vllm | sm70+; fp8 (e4m3/e5m2) needs CUDA 11.8+ / sm89+ | fp16/bf16 cache; fp8 (e4m3, e5m2) quantized cache with fp32 scales (per-tensor or per-head as of the 2026 fp8-kvcache update) | Production paged-KV write used by vLLM; 2026 update extended it to per-head fp8 scale arrays. Best-in-class for inference KV-cache copy/quant. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |

### `kv_cache_copy_swap_blocks`

Copy/swap KV-cache blocks between buffers/devices for paged-attention block management (copy-on-write, CPU<->GPU swap).

**kernel_set_abi:** _(none)_

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm import _custom_ops as ops; ops.swap_blocks(src, dst, block_mapping) # copy-on-write: ops.copy_blocks(key_caches, value_caches, block_mapping) | pip install vllm | sm70+ | fp16/bf16/fp8 | Block-level KV management kernels behind PagedAttention; swap_blocks for CPU<->GPU offload, copy_blocks for copy-on-write fork. Confirm exact arg order against installed version (signatures shift across releases). _(confidence: medium)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |

### `fp8_convert_quantize`

Convert/quantize a tensor to/from fp8 (e4m3/e5m2) with a scale — used for fp8 KV cache and fp8 activations.

**kernel_set_abi:** `ks_quantize_fp8`

| Rank | Lib | Python call | Install | GPU arch | Dtypes | Perf note | Source |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | vllm | from vllm import _custom_ops as ops; ops.convert_fp8(output, input, scale=1.0, kv_dtype='fp8') # dynamic per-token: ops.scaled_fp8_quant(input, scale=None) | pip install vllm | sm89+ for native fp8 tensor support; sm80 storage-only | fp8 e4m3 / e5m2 with fp32 scale | Pairs with reshape_and_cache for fp8 KV; scaled_fp8_quant gives dynamic per-token scales for fp8 GEMM activations. _(confidence: high)_ | [link](https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py) |
| 2 | torchao | from torchao.float8 import convert_to_float8_training # training-time fp8; or torchao.quantization fp8 APIs for inference | pip install torchao | sm89/sm90 (Hopper/Ada) for fp8 matmul | fp8 e4m3/e5m2 | Best-in-class fp8 *training* (rowwise/tensorwise scaling, delayed scaling) integrated with torch.compile + FSDP2. _(confidence: high)_ | [link](https://github.com/pytorch/ao/tree/main/torchao/float8) |

---
