# Unified FLAT Atomic-Operator Index

Single flat index of **476** atomic operators harvested across three inference-kernel libraries
(`sgl-kernel`, `flashinfer`, `vllm`), each assigned a canonical **logical op** tag and, where one
exists, the matching **kernel-set C ABI** op (`ks_*`). This is the flattened "select-optimal" menu:
for any logical operation, pick the best atomic provider for your arch/dtype.

Source data: `providers/atomic_ops.json` (flat array of `{addr, lib, logical_op, ks_abi, category, what, signature, arch, dtype, source}`).

## Totals

| Library | Address prefix | Atomic ops |
|---|---|---|
| sgl-kernel | `sgl.*` | 167 |
| flashinfer | `flashinfer.*` | 157 |
| vllm | `vllm.*` | 152 |
| **TOTAL** | | **476** |

- **Distinct logical ops:** 187
- **Logical ops with ≥2 libs** (true cross-lib choices): 76
- **Logical ops with all 3 libs:** 17
- **Atomic ops mapped to a kernel-set C ABI op:** 127 / 476

---

## (a) Flat alphabetical table — every atomic op

All `sgl.*` / `flashinfer.*` / `vllm.*` addresses, sorted alphabetically, with logical op, what, and arch.

| addr | logical_op | ks_abi | what | arch |
|---|---|---|---|---|
| `flashinfer.BatchAttention.plan` | `attention.plan` | — | Plan kernel for holistic two-stage persistent paged attention scheduler | sm80+ |
| `flashinfer.BatchAttention.run` | `attention.unified_paged` | — | Holistic/persistent unified paged attention run (fused prefill+decode, two-stage reductio… | sm80+ |
| `flashinfer.BatchDecodeMlaWithPagedKVCacheWrapper.plan` | `attention.mla.plan` | — | Plan kernel for MLA batched decode (CuTe SM80 path) | sm80 |
| `flashinfer.BatchDecodeMlaWithPagedKVCacheWrapper.run` | `attention.mla.decode` | — | MLA batched decode attention over paged compressed-KV (ckv+kpe) cache | sm80 |
| `flashinfer.BatchDecodeWithPagedKVCacheWrapper.plan` | `attention.decode.plan` | — | Plan/scheduling kernel for batched paged decode (split-KV partition) | sm80+ |
| `flashinfer.BatchDecodeWithPagedKVCacheWrapper.run` | `attention.decode` | — | Batched decode attention over paged KV cache (single query step per request) | sm80+ |
| `flashinfer.BatchPrefillWithPagedKVCacheWrapper.paged_run` | `attention.prefill` | — | Batched prefill attention over paged KV cache (FA2), the atomic batch_prefill_paged op | sm80+ |
| `flashinfer.BatchPrefillWithPagedKVCacheWrapper.paged_run_fp8_sm90` | `attention.prefill` | — | Batched paged FP8 prefill attention, Hopper FA3 | sm90 |
| `flashinfer.BatchPrefillWithPagedKVCacheWrapper.paged_run_sm90` | `attention.prefill` | — | Batched paged prefill attention, Hopper FA3 kernel | sm90 |
| `flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan` | `attention.prefill.plan` | — | Plan/scheduling kernel for batched paged prefill (split-KV work partition) | sm80+ |
| `flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan_sm90` | `attention.prefill.plan` | — | Plan kernel for Hopper FA3 batched prefill | sm90 |
| `flashinfer.BatchPrefillWithRaggedKVCacheWrapper.ragged_run` | `attention.prefill` | — | Batched prefill attention over ragged (varlen) contiguous KV (FA2) | sm80+ |
| `flashinfer.BatchPrefillWithRaggedKVCacheWrapper.ragged_run_sm90` | `attention.prefill` | — | Batched ragged prefill attention, Hopper FA3 kernel | sm90 |
| `flashinfer.activation.gelu_and_mul` | `act.gelu_mul` | `ks_gelu_and_mul` | Gated GELU (erf): out = gelu(x[...,:d]) * x[...,d:] | sm80+ |
| `flashinfer.activation.gelu_tanh_and_mul` | `act.gelu_tanh_mul` | `ks_gelu_and_mul` | Gated GELU-tanh approx: out = gelu_tanh(x[...,:d]) * x[...,d:] | sm80+ |
| `flashinfer.activation.silu_and_mul` | `act.silu_mul` | `ks_silu_and_mul` | Gated SiLU: out = silu(x[...,:d]) * x[...,d:] | sm80+ |
| `flashinfer.activation.silu_and_mul_scaled_nvfp4_experts_quantize` | `act.silu_mul_quant` | — | Fused SiLU+mul then per-expert NVFP4 quantization with per-row mask | sm100 |
| `flashinfer.api_log_print_tensor_stats` | `util.debug_stats` | — | Debug kernel: compute and print tensor stats (min/max/mean/nan) by id | sm80+ |
| `flashinfer.batch_pod_with_kv_cache_tensor` | `attention.pod_fused` | — | Batch POD attention: batched paged prefill fused with batched paged decode | sm80+ |
| `flashinfer.cascade.merge_state` | `attention.merge_state` | — | Merge two attention states (v_a/s_a, v_b/s_b) via log-sum-exp into merged v/s | sm80+ |
| `flashinfer.cascade.merge_state_in_place` | `attention.merge_state` | — | In-place merge of another attention state into (v,s) | sm80+ |
| `flashinfer.cascade.merge_states` | `attention.merge_state` | — | Merge N attention states stacked along an axis into one | sm80+ |
| `flashinfer.comm.init_custom_ar` | `comm.init` | — | Initialize vLLM custom all-reduce IPC workspace/handles | sm80+ |
| `flashinfer.comm.mnnvl.allgather` | `comm.allgather` | — | NVSHMEM-based all-gather | sm90+ |
| `flashinfer.comm.mnnvl.allreduce` | `comm.allreduce` | — | NVSHMEM-based all-reduce (mixed_comm) | sm90+ |
| `flashinfer.comm.mnnvl.fused_allreduce_allgather` | `comm.allreduce` | — | Fused all-reduce + all-gather (NVSHMEM) | sm90+ |
| `flashinfer.comm.mnnvl.fused_reducescatter_allreduce` | `comm.allreduce` | — | Fused reduce-scatter + all-reduce (NVSHMEM) | sm90+ |
| `flashinfer.comm.mnnvl.reducescatter` | `comm.reduce_scatter` | — | NVSHMEM-based reduce-scatter | sm90+ |
| `flashinfer.comm.nvshmem_init` | `comm.init` | — | NVSHMEM init/finalize and PE/topology query primitives | sm90+ |
| `flashinfer.comm.trtllm_allreduce_fusion` | `comm.allreduce_rmsnorm` | — | Fused all-reduce + (residual add) + RMSNorm + optional FP8/FP4 quant | sm90/sm100 |
| `flashinfer.comm.trtllm_alltoall.moe_comm` | `comm.moe_alltoall` | — | TRT-LLM MoE expert-parallel all-to-all token dispatch/combine | sm90+ |
| `flashinfer.comm.trtllm_alltoall.moe_comm_prepare_indices` | `comm.moe_a2a_prepare` | — | Prepare send/recv index metadata for MoE all-to-all | sm90+ |
| `flashinfer.comm.trtllm_alltoall.moe_local_gather` | `comm.moe_a2a_gather` | — | Local gather of MoE tokens for all-to-all path | sm90+ |
| `flashinfer.comm.trtllm_alltoall.moe_prepare` | `comm.moe_a2a_prepare` | — | Prepare-stage kernel for MoE all-to-all communication | sm90+ |
| `flashinfer.comm.trtllm_custom_all_reduce` | `comm.allreduce` | — | TRT-LLM custom one-shot/two-shot all-reduce (Lamport buffers) | sm80+ |
| `flashinfer.comm.trtllm_dcp_alltoall.alltoall_dcp_native` | `comm.moe_alltoall` | — | Decode context-parallel (DCP) native all-to-all | sm90+ |
| `flashinfer.comm.trtllm_lamport_initialize` | `comm.init` | — | Initialize Lamport flag buffers for custom all-reduce | sm80+ |
| `flashinfer.comm.trtllm_mnnvl_allreduce_fusion` | `comm.allreduce_rmsnorm` | — | Multi-node NVLink (MNNVL) fused all-reduce + RMSNorm | sm90/sm100 |
| `flashinfer.comm.trtllm_moe_allreduce_fusion` | `comm.moe_allreduce_fusion` | — | Fused MoE all-reduce + residual + norm fusion | sm90/sm100 |
| `flashinfer.comm.trtllm_moe_alltoall.moe_a2a_combine` | `comm.moe_combine` | — | MoE all-to-all combine (gather + weighted reduce expert outputs) | sm90+ |
| `flashinfer.comm.trtllm_moe_alltoall.moe_a2a_dispatch` | `comm.moe_dispatch` | — | MoE all-to-all dispatch (scatter tokens to expert ranks) | sm90+ |
| `flashinfer.comm.trtllm_moe_alltoall.moe_a2a_sanitize_expert_ids` | `comm.moe_a2a_prepare` | — | Sanitize/validate expert ids for MoE all-to-all | sm90+ |
| `flashinfer.comm.trtllm_moe_finalize_allreduce_fusion` | `comm.moe_finalize_allreduce` | — | MoE finalize (weighted expert combine) fused with all-reduce + norm | sm90/sm100 |
| `flashinfer.comm.vllm_all_reduce` | `comm.allreduce` | — | vLLM custom all-reduce (one/two-shot) out-of-place | sm80+ |
| `flashinfer.concat_mla_k` | `kvcache.concat_mla_k` | — | Concatenate MLA K nope and rope parts into full K tensor | sm80+ |
| `flashinfer.cudnn_batch_decode_with_kv_cache` | `attention.decode` | — | cuDNN SDPA decode attention launcher | sm90/sm100 |
| `flashinfer.cudnn_batch_prefill_with_kv_cache` | `attention.prefill` | — | cuDNN SDPA prefill attention launcher | sm90/sm100 |
| `flashinfer.cutlass_mla_paged_attention` | `attention.mla.decode` | — | CUTLASS MLA paged attention (Blackwell) for compressed q_nope_pe vs ckv_kpe cache | sm100 |
| `flashinfer.fast_topk_clusters_exact` | `sparse.topk_select` | — | Exact fast top-k over cluster logits (sparse MLA cluster selection) | sm90/sm100 |
| `flashinfer.fast_topk_clusters_exact_page_table_transform` | `sparse.topk_transform_page` | — | Fused exact cluster top-k + page-table transform | sm90/sm100 |
| `flashinfer.fast_topk_clusters_exact_ragged_transform` | `sparse.topk_transform_ragged` | — | Fused exact cluster top-k + ragged index transform | sm90/sm100 |
| `flashinfer.fmha.blackwell_fmha_plan` | `attention.plan` | — | Blackwell FMHA work-partition plan kernel (tile/head/batch indices) | sm100 |
| `flashinfer.fmha.cutlass_sm100.run` | `attention.prefill` | — | CUTLASS Blackwell FMHA prefill run (varlen, work-partitioned) | sm100 |
| `flashinfer.fmha_reduction` | `attention.merge_state` | — | Cross-CTA softmax-state reduction (LSE merge) for split-KV FMHA outputs | sm100 |
| `flashinfer.fmha_v2.run` | `attention.pod_fused` | — | TRT-LLM fmha_v2 (XQA-family) prefill/decode attention kernel | sm80+ |
| `flashinfer.fused_moe.RoutingMethodType.NoAuxTc` | `moe.gate_grouped` | `ks_moe_gate_sigmoid_group_topk` | DeepSeek-V3 no-aux-loss group-limited expert routing (sigmoid+group topk) | sm90/sm100 |
| `flashinfer.fused_moe.bgmv_moe_expand` | `moe.fused_full` | — | BGMV MoE/LoRA expand GEMM (B-projection, accumulate) | sm80+ |
| `flashinfer.fused_moe.bgmv_moe_shrink` | `moe.fused_full` | — | BGMV (batched gather matrix-vector) MoE/LoRA shrink GEMM (A-projection) | sm80+ |
| `flashinfer.fused_moe.cutlass_fused_moe` | `moe.fused_full` | — | CUTLASS fused MoE: gather + grouped GEMM1 + gated act + grouped GEMM2 + scatter (FusedMoe… | sm90/sm100 |
| `flashinfer.fused_moe.interleave_moe_weights_for_sm90_mixed_gemm` | `moe.fused_full` | — | Interleave MoE weights for SM90 mixed-precision grouped GEMM layout | sm90 |
| `flashinfer.fused_moe.moe_activation` | `moe.fused_full` | — | MoE intermediate gated activation (SwiGLU) between GEMM1 and GEMM2 | sm80+ |
| `flashinfer.fused_moe.moe_output_memset` | `moe.fused_full` | — | Zero/scatter-init MoE output buffer (in-place and out-of-place variants) | sm80+ |
| `flashinfer.fused_moe.moe_permute` | `moe.permute` | `ks_moe_permute` | Permute (gather) tokens into expert-contiguous layout (fp16/bf16/fp8/fp4 variants) | sm80+ |
| `flashinfer.fused_moe.moe_sort` | `moe.sort` | `ks_moe_compute_permutation` | Sort/argsort tokens by selected expert id for grouped MoE GEMM | sm80+ |
| `flashinfer.fused_moe.moe_unpermute` | `moe.unpermute_combine` | `ks_moe_unpermute` | Unpermute + weighted-combine expert outputs back to token order | sm80+ |
| `flashinfer.fused_moe.trtllm_bf16_moe` | `moe.fused_full` | — | TRT-LLM BF16 fused MoE | sm90/sm100 |
| `flashinfer.fused_moe.trtllm_fp4_block_scale_moe` | `moe.fused_full` | — | TRT-LLM FP4 (NVFP4) block-scale fused MoE | sm100 |
| `flashinfer.fused_moe.trtllm_fp8_block_scale_moe` | `moe.fused_full` | — | TRT-LLM FP8 block-scale fused MoE | sm90/sm100 |
| `flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe` | `moe.fused_full` | — | TRT-LLM FP8 per-tensor-scale fused MoE | sm90/sm100 |
| `flashinfer.fused_moe.trtllm_get_valid_moe_configs` | `moe.fused_full` | — | Enumerate valid TRT-LLM fused-MoE tactic configs | sm90/sm100 |
| `flashinfer.fused_moe.trtllm_mxint4_block_scale_moe` | `moe.fused_full` | — | TRT-LLM MXINT4 block-scale fused MoE | sm100 |
| `flashinfer.fused_qk_rmsnorm_rope` | `rope.qk_norm_rope` | — | Fused QK-RMSNorm + RoPE applied to fused QKV (with optional fp8 output quant) | sm80+ |
| `flashinfer.gdn_prefill` | `ssm.gated_delta_rule` | — | Gated DeltaNet (GDN/linear-attention) prefill scan with state output | sm90 |
| `flashinfer.gemm.SegmentGEMMWrapper.cutlass_segment_gemm` | `gemm.grouped` | `ks_moe_grouped_gemm` | CUTLASS grouped/segment GEMM (variable-size problems) for SM80 | sm80+ |
| `flashinfer.gemm.SegmentGEMMWrapper.cutlass_segment_gemm_sm90` | `gemm.grouped` | `ks_moe_grouped_gemm` | CUTLASS grouped/segment GEMM for Hopper | sm90 |
| `flashinfer.gemm.bmm_bf16` | `gemm.batched` | `ks_gemm_batched` | CUTLASS BF16 (batched) GEMM with tactic autotuning | sm90/sm100 |
| `flashinfer.gemm.bmm_fp8` | `gemm.fp8_batched` | — | Batched FP8 GEMM via cuBLASLt with per-tensor scales | sm89+ |
| `flashinfer.gemm.bmm_fp8.get_algos` | `gemm.fp8_batched` | — | Enumerate cuBLASLt algos for an FP8 BMM problem | sm89+ |
| `flashinfer.gemm.bmm_fp8.run_with_algo` | `gemm.fp8_batched` | — | Run FP8 BMM with a chosen cuBLASLt algo index | sm89+ |
| `flashinfer.gemm.dsv3_router_gemm` | `gemm.router` | — | DeepSeek-V3 router GEMM (M<=16, K=7168, N=256, fp32) | sm90/sm100 |
| `flashinfer.gemm.fp8_blockscale_gemm_sm90` | `gemm.fp8_blockwise` | — | FP8 block-scaled GEMM (DeepSeek-style 128x128 block scales) for Hopper | sm90 |
| `flashinfer.gemm.gemm_fp8_nt_groupwise` | `gemm.fp8_blockwise` | — | FP8 NT GEMM with groupwise/blockwise scales (SM100) | sm100 |
| `flashinfer.gemm.gemm_fp8_nt_groupwise_sm120` | `gemm.fp8_blockwise` | — | FP8 NT groupwise-scaled GEMM (SM120) | sm120 |
| `flashinfer.gemm.glm_dsa_router_gemm` | `gemm.router` | — | GLM DSA router GEMM (M<=16, K=6144, N=256, fp32) | sm90/sm100 |
| `flashinfer.gemm.group_gemm_fp8_nt_groupwise` | `gemm.fp8_blockwise` | — | Grouped FP8 NT GEMM with groupwise scales (SM100) | sm100 |
| `flashinfer.gemm.group_gemm_fp8_nt_groupwise_sm120` | `gemm.fp8_blockwise` | — | Grouped FP8 NT groupwise GEMM (SM120) | sm120 |
| `flashinfer.gemm.group_gemm_mxfp4_nt_groupwise` | `gemm.fp4` | — | Grouped MXFP4 NT GEMM with groupwise scales (SM100) | sm100 |
| `flashinfer.gemm.group_gemm_mxfp4_nt_groupwise_sm120` | `gemm.fp4` | — | Grouped MXFP4 NT groupwise GEMM (SM120) | sm120 |
| `flashinfer.gemm.group_gemm_nvfp4_nt_groupwise` | `gemm.fp4` | — | Grouped NVFP4 NT groupwise GEMM (SM120) | sm120 |
| `flashinfer.gemm.ml3_router_gemm` | `gemm.router` | — | Router GEMM variant (M<=16, K=7168, N=128, bf16) | sm90/sm100 |
| `flashinfer.gemm.mm_bf16` | `gemm.dense` | `ks_gemm` | BF16 matrix multiply via cuBLASLt (get_algos + run_with_algo) | sm80+ |
| `flashinfer.gemm.mm_fp4` | `gemm.fp4` | — | CUTLASS NVFP4/FP4 GEMM with block scales and tactic autotuning | sm100 |
| `flashinfer.gemm.mm_fp4.sm103` | `gemm.fp4` | — | FP4 CUTLASS GEMM for SM103 | sm103 |
| `flashinfer.gemm.mm_fp4.sm120` | `gemm.fp4` | — | FP4 CUTLASS GEMM for SM120 | sm120 |
| `flashinfer.gemm.mm_fp8` | `gemm.fp8` | — | CUTLASS FP8 GEMM with tactic autotuning | sm90/sm100 |
| `flashinfer.gemm.mm_mxfp8` | `gemm.fp8` | — | CUTLASS MXFP8 GEMM with block scales and tactic autotuning | sm100 |
| `flashinfer.gemm.mm_mxfp8.sm120` | `gemm.fp8` | — | CUTLASS MXFP8 GEMM for SM120 | sm120 |
| `flashinfer.gemm.tgv_gemm_sm100` | `gemm.dense` | `ks_gemm` | TGV (tile-genvolta) GEMM with optional bias, tactic autotuning (Blackwell) | sm100 |
| `flashinfer.gemm.tinygemm2` | `gemm.dense` | `ks_gemm` | TinyGEMM2 small-M GEMM with bias (mm_M1_16_* shapes) | sm90/sm100 |
| `flashinfer.gemm.tinygemm2_nobias` | `gemm.dense` | `ks_gemm` | TinyGEMM2 small-M GEMM without bias | sm90/sm100 |
| `flashinfer.gemm.trtllm_gemm` | `gemm.fp8` | — | TRT-LLM cubin GEMM runner (FP8/FP4/bf16) with tactic selection | sm90/sm100 |
| `flashinfer.gemm.trtllm_low_latency_gemm` | `gemm.fp8` | — | TRT-LLM low-latency GEMM (small-batch decode GEMM) | sm90/sm100 |
| `flashinfer.mamba.checkpointing_ssu` | `ssm.selective_scan` | — | Mamba multi-token selective-scan with checkpointing (chunked recurrence) | sm90 |
| `flashinfer.mamba.selective_state_update` | `ssm.selective_state_update` | — | Mamba selective state-space update (SSU): per-step state recurrence with optional gating/… | sm80+ |
| `flashinfer.mamba.seq_chunk_cumsum` | `ssm.chunk_cumsum` | — | Per-sequence chunked cumulative-sum (Mamba2 chunk-scan prep) | sm80+ |
| `flashinfer.mla.BatchMLAPagedAttentionWrapper.plan` | `attention.mla.plan` | — | Plan kernel for FlashMLA paged attention (prefill+decode) | sm80+ |
| `flashinfer.mla.BatchMLAPagedAttentionWrapper.plan_sm90` | `attention.mla.plan` | — | Plan kernel for Hopper FlashMLA paged attention | sm90 |
| `flashinfer.mla.BatchMLAPagedAttentionWrapper.run` | `attention.mla.decode` | — | FlashMLA paged attention run (deepseek MLA absorb, ckv+kpe), atomic batch_mla op | sm80+ |
| `flashinfer.mla.BatchMLAPagedAttentionWrapper.run_sm90` | `attention.mla.decode` | — | FlashMLA paged attention run, Hopper kernel | sm90 |
| `flashinfer.norm.fused_add_rmsnorm` | `norm.fused_add_rmsnorm` | `ks_fused_add_rmsnorm` | Fused residual-add then RMSNorm in-place (input+=residual; normalize) | sm80+ |
| `flashinfer.norm.fused_add_rmsnorm_quant` | `norm.fused_add_rmsnorm_quant` | — | Fused residual-add + RMSNorm + FP8 quant | sm80+ |
| `flashinfer.norm.fused_dit_layernorm` | `norm.dit_layernorm` | — | DiT fused layernorm: residual add, gate, scale/shift modulation + optional fp4 scale outp… | sm90+ |
| `flashinfer.norm.fused_rmsnorm_silu` | `norm.rmsnorm_silu` | — | Fused RMSNorm + SiLU activation with per-row output scale | sm90+ |
| `flashinfer.norm.gemma_fused_add_rmsnorm` | `norm.gemma_fused_add_rmsnorm` | — | Gemma-style fused residual-add + RMSNorm | sm80+ |
| `flashinfer.norm.gemma_rmsnorm` | `norm.gemma_rmsnorm` | `ks_gemma_rmsnorm` | Gemma-style RMSNorm (weight uses (1+w)) | sm80+ |
| `flashinfer.norm.layernorm` | `norm.layernorm` | `ks_layernorm` | Standard LayerNorm with gamma/beta | sm80+ |
| `flashinfer.norm.rmsnorm` | `norm.rmsnorm` | `ks_rmsnorm` | RMS normalization out = x/rms(x)*weight | sm80+ |
| `flashinfer.norm.rmsnorm_quant` | `norm.rmsnorm_quant` | — | Fused RMSNorm + FP8 quantization with output scale | sm80+ |
| `flashinfer.page.append_paged_kv_cache` | `kvcache.append` | `ks_reshape_and_cache` | Scatter-append new K/V into a paged KV cache at batch_indices/positions | sm80+ |
| `flashinfer.page.append_paged_mla_kv_cache` | `kvcache.append_mla` | — | Scatter-append compressed MLA ckv/kpe into paged MLA cache | sm80+ |
| `flashinfer.pod_with_kv_cache_tensor` | `attention.pod_fused` | — | POD attention: single prefill fused with batched paged decode in one persistent kernel | sm80+ |
| `flashinfer.quantization.nvfp4_kv_dequantize` | `quant.fp4_dequant` | — | Dequantize NVFP4 KV cache back to fp16/bf16 | sm100 |
| `flashinfer.quantization.nvfp4_kv_quantize` | `quant.fp4_kv` | — | Quantize KV cache tensor to NVFP4 with block scale factors | sm100 |
| `flashinfer.quantization.packbits` | `quant.packbits` | — | Pack a boolean tensor into bits (big/little bitorder) | sm80+ |
| `flashinfer.quantization.segment_packbits` | `quant.packbits` | — | Segment-wise packbits using input/output indptr | sm80+ |
| `flashinfer.radix_topk` | `sparse.topk_select` | — | Radix-select top-k indices/values per row (sparse-attn selection) | sm80+ |
| `flashinfer.radix_topk_page_table_transform` | `sparse.topk_transform_page` | — | Fused radix top-k + page-table transform for sparse paged attention | sm80+ |
| `flashinfer.radix_topk_ragged_transform` | `sparse.topk_transform_ragged` | — | Fused radix top-k + ragged index transform for sparse attention | sm80+ |
| `flashinfer.rope.apply_llama31_rope` | `rope.apply_llama31` | — | Apply Llama-3.1 scaled RoPE to q/k (indptr/offsets) | sm80+ |
| `flashinfer.rope.apply_llama31_rope_pos_ids` | `rope.apply_llama31` | — | Apply Llama-3.1 RoPE addressed by position ids | sm80+ |
| `flashinfer.rope.apply_rope` | `rope.apply` | `ks_rope` | Apply RoPE to q/k using indptr/offsets (ragged), out-of-place | sm80+ |
| `flashinfer.rope.apply_rope_pos_ids` | `rope.apply` | `ks_rope` | Apply RoPE to q/k addressed by explicit position ids | sm80+ |
| `flashinfer.rope.apply_rope_with_cos_sin_cache` | `rope.apply` | `ks_rope` | Apply RoPE to q/k using precomputed cos/sin cache + pos ids | sm80+ |
| `flashinfer.rope.rope_quantize` | `rope.quantize` | — | Apply RoPE (cos/sin cache) to q/k rope+nope parts and FP8-quantize outputs | sm80+ |
| `flashinfer.rope.rope_quantize_append_paged_kv_cache` | `rope.quantize_append_kv` | — | Fused RoPE + FP8-quantize + append into paged KV/MLA cache | sm80+ |
| `flashinfer.sampling.chain_speculative_sampling` | `spec.verify_sampling` | — | Verify draft tokens against target probs and emit accepted/bonus tokens | sm80+ |
| `flashinfer.sampling.min_p_sampling_from_probs` | `sampling.min_p` | — | Min-p rejection sampling from probabilities | sm80+ |
| `flashinfer.sampling.sampling_from_logits` | `sampling.from_logits` | `ks_sample` | Gumbel-style sampling directly from logits | sm80+ |
| `flashinfer.sampling.sampling_from_probs` | `sampling.from_probs` | `ks_sample` | Single-pass categorical sampling from a probability distribution (Philox RNG) | sm80+ |
| `flashinfer.sampling.softmax` | `sampling.softmax` | `ks_softmax` | Online softmax over logits with optional per-row temperature | sm80+ |
| `flashinfer.sampling.top_k_mask_logits` | `sampling.topk_mask` | — | Mask logits to keep only top-k (set rest to -inf) | sm80+ |
| `flashinfer.sampling.top_k_renorm_probs` | `sampling.topk_renorm` | — | Renormalize probabilities after top-k masking | sm80+ |
| `flashinfer.sampling.top_k_sampling_from_probs` | `sampling.topk` | `ks_sample` | Top-k rejection sampling from probabilities | sm80+ |
| `flashinfer.sampling.top_k_top_p_sampling_from_probs` | `sampling.topk_topp` | `ks_sample` | Joint top-k + top-p rejection sampling from probabilities | sm80+ |
| `flashinfer.sampling.top_p_renorm_probs` | `sampling.topp_renorm` | — | Renormalize probabilities after top-p masking | sm80+ |
| `flashinfer.sampling.top_p_sampling_from_probs` | `sampling.topp` | `ks_sample` | Nucleus (top-p) rejection sampling from probabilities | sm80+ |
| `flashinfer.single_decode_with_kv_cache` | `attention.decode` | — | Single-request decode attention (1 query step) over contiguous K/V | sm80+ |
| `flashinfer.single_prefill_with_kv_cache` | `attention.prefill` | — | Single-request prefill/append attention over contiguous Q/K/V (FA2 kernel), optional LSE | sm80+ |
| `flashinfer.single_prefill_with_kv_cache.fp8_sm90` | `attention.prefill` | — | Single-request FP8 prefill attention, Hopper FA3 kernel | sm90 |
| `flashinfer.single_prefill_with_kv_cache_sm90` | `attention.prefill` | — | Single-request prefill attention, Hopper FA3 kernel variant | sm90 |
| `flashinfer.trtllm_batch_context_with_kv_cache` | `attention.prefill` | — | TRT-LLM paged context (prefill) attention | sm90/sm100 |
| `flashinfer.trtllm_batch_decode_with_kv_cache` | `attention.decode` | — | TRT-LLM paged decode attention (XQA/optimized GQA) | sm90/sm100 |
| `flashinfer.trtllm_batch_decode_with_kv_cache_mla` | `attention.mla.sparse_decode` | — | TRT-LLM sparse MLA paged decode (DSv4 sparse) attention | sm100 |
| `flashinfer.trtllm_fmha_v2.run` | `attention.prefill` | — | TRT-LLM fmha_v2 binding run (cubin-based FMHA) | sm90/sm100 |
| `flashinfer.trtllm_ragged_attention` | `attention.prefill` | — | TRT-LLM ragged (varlen contiguous) attention | sm90/sm100 |
| `flashinfer.xqa` | `attention.decode` | — | XQA optimized GQA/MHA paged decode attention (TRT-LLM XQA kernel) | sm90/sm100 |
| `flashinfer.xqa_mla` | `attention.mla.decode` | — | XQA MLA paged decode attention variant | sm90/sm100 |
| `sgl.all_reduce` | `comm.allreduce` | — | Custom multi-GPU all-reduce over IPC buffers | sm80+ |
| `sgl.all_reduce_reg` | `comm.allreduce` | — | ROCm custom all-reduce over registered buffers | ROCm/HIP |
| `sgl.all_reduce_unreg` | `comm.allreduce` | — | ROCm custom all-reduce over unregistered buffer | ROCm/HIP |
| `sgl.apply_rotary_pos_emb_cpu` | `rope.apply` | `ks_rope` | CPU apply precomputed cos/sin rotary embedding | x86/aarch64 CPU |
| `sgl.apply_shuffle_mul_sum` | `moe.shuffle_mul_sum` | — | Shuffle rows by permutation, multiply by factors, then sum | sm80+ |
| `sgl.apply_token_bitmask_inplace_cuda` | `sampling.grammar_mask` | — | Apply grammar token bitmask to logits in-place (-inf masked) | sm80+ |
| `sgl.awq_dequantize` | `quant.awq_dequant` | — | AWQ int4 weight dequantize to fp16 | sm80+ |
| `sgl.biased_grouped_topk_cpu` | `moe.gate_grouped` | `ks_moe_gate_sigmoid_group_topk` | CPU biased grouped top-k MoE gate (with correction bias) | x86/aarch64 CPU |
| `sgl.bmm_cpu` | `gemm.batched` | `ks_gemm_batched` | CPU batched matmul with optional scale | x86/aarch64 CPU |
| `sgl.bmm_fp8` | `gemm.fp8_batched` | — | Batched FP8 matmul via cuBLAS with A/B scales | sm89/sm90 |
| `sgl.build_tree_kernel_efficient` | `spec.build_tree` | — | Build EAGLE speculative draft tree mask/positions/retrieve indices | sm80+ |
| `sgl.causal_conv1d_fwd` | `ssm.conv1d` | — | Mamba causal depthwise conv1d forward (varlen, optional silu) | sm80+ |
| `sgl.causal_conv1d_fwd_cpu` | `ssm.conv1d` | — | CPU mamba causal conv1d forward | x86 CPU |
| `sgl.causal_conv1d_update` | `ssm.conv1d_update` | — | Mamba causal conv1d decode update with conv state cache | sm80+ |
| `sgl.causal_conv1d_update_cpu` | `ssm.conv1d_update` | — | CPU mamba causal conv1d decode update | x86 CPU |
| `sgl.causal_conv1d_weight_pack` | `ssm.conv1d_weight_pack` | — | CPU prepack causal conv1d weights | x86 CPU |
| `sgl.chunk_gated_delta_rule_cpu` | `ssm.gated_delta_rule` | — | CPU chunked gated delta-rule linear attention | x86/aarch64 CPU |
| `sgl.concat_mla_absorb_q` | `kvcache.concat_mla_q` | — | Concatenate two q tensors along last dim for MLA absorb | sm80+ |
| `sgl.concat_mla_k` | `kvcache.concat_mla_k` | — | Concatenate MLA k_nope and k_rope into k buffer | sm80+ |
| `sgl.conv3d_embed_cpu` | `util.conv3d_embed` | — | CPU conv3d fast path for patch embedding | x86/aarch64 CPU |
| `sgl.conv3d_embed_weight_pack` | `util.conv3d_embed` | — | CPU prepack conv3d patch-embed weights | x86/aarch64 CPU |
| `sgl.convert_scale_packed` | `quant.scale_pack` | — | CPU scale pre-pack for mxfp4 | x86 CPU |
| `sgl.convert_vertical_slash_indexes` | `attention.sparse_index_build` | — | Build block_count/offset + column_count/index for vertical-slash sparse attn | sm80+ |
| `sgl.convert_vertical_slash_indexes_mergehead` | `attention.sparse_index_build` | — | Vertical-slash index builder with per-head merged index counts | sm80+ |
| `sgl.convert_weight_packed` | `gemm.dense` | `ks_gemm` | CPU weight pre-pack to VNNI/blocked layout | x86/aarch64 CPU |
| `sgl.convert_weight_packed_scale_zp` | `quant.weight_prepack` | — | CPU int4 weight prepack with packed scales+zeros (awq/gptq) | x86 CPU |
| `sgl.copy_to_gpu_no_ce` | `elementwise.copy` | — | Host->device copy without copy-engine (kernel-driven memcpy) | sm80+ |
| `sgl.create_greenctx_stream_by_value` | `util.greenctx_stream` | — | Create two green-context CUDA streams partitioned by SM count | sm90 |
| `sgl.cutlass_mla_decode` | `attention.mla.decode` | — | CUTLASS MLA decode attention (latent 512 + rope 64) | sm100 |
| `sgl.cutlass_mla_get_workspace_size` | `attention.mla.workspace` | — | Compute workspace bytes for cutlass_mla_decode | sm100 |
| `sgl.cutlass_w4a8_moe_mm` | `moe.grouped_gemm_w4a8` | — | CUTLASS grouped GEMM int4 weights x fp8 activations for MoE | sm90 |
| `sgl.decode_attention_cpu` | `attention.decode` | — | CPU decode attention over paged KV with req_to_token mapping | x86/aarch64 CPU |
| `sgl.deepseek_v4_topk_transform_512` | `sparse.topk_transform_page` | — | DeepSeek-V4 indexer top-k select -> paged physical slot indices (topk<=1024) | sm80+ (ROCm path) |
| `sgl.dense_decode_fwd` | `attention.mla.decode` | — | FlashMLA dense decode attention forward (returns sched meta) | sm90 |
| `sgl.dense_prefill_fwd` | `attention.mla.prefill` | — | CUTLASS SM100 dense MLA prefill FMHA forward | sm100 |
| `sgl.dsv3_fused_a_gemm` | `gemm.dsv3_fused_a` | — | DeepSeek-V3 fused 'a' projection GEMM | sm90 |
| `sgl.dsv3_router_gemm` | `gemm.router` | — | DeepSeek-V3 MoE router GEMM (bf16 or float out) | sm90 |
| `sgl.dsv4_fused_k_norm_rope_flashmla` | `rope.dsv4_k_norm_rope_cache` | — | DeepSeek-V4 fused K RMSNorm + RoPE + FlashMLA FP8 cache store | sm90 |
| `sgl.dsv4_fused_q_indexer_rope_hadamard_quant` | `quant.fp8` | `ks_quantize_fp8` | DeepSeek-V4 Q indexer: RoPE + Hadamard transform + FP8 quant | sm90 |
| `sgl.dsv4_fused_q_norm_rope` | `rope.dsv4_q_norm_rope` | — | DeepSeek-V4 fused Q RMSNorm (no weight) + RoPE | sm90 |
| `sgl.es_fp8_blockwise_scaled_grouped_mm` | `moe.grouped_gemm_fp8` | — | Expert-specialization FP8 block-scaled grouped GEMM | sm90 |
| `sgl.es_sm100_mxfp8_blockscaled_grouped_mm` | `moe.grouped_gemm_fp8` | — | SM100 MXFP8 block-scaled grouped GEMM (expert specialization) | sm100 |
| `sgl.es_sm100_mxfp8_blockscaled_grouped_quant` | `quant.mxfp8` | — | SM100 grouped MXFP8 block-scaled quantization per expert | sm100 |
| `sgl.extend_attention_cpu` | `attention.prefill` | — | CPU extend/prefill attention with KV buffers | x86/aarch64 CPU |
| `sgl.fast_topk` | `sparse.topk_select` | — | Fast top-k indices over ragged/paged score rows (DSv3.2 topk=2048) | sm80+ |
| `sgl.fast_topk_transform_fused` | `sparse.topk_transform_page` | — | Top-k then transform indices to page-table (page_size=1) slots | sm80+ |
| `sgl.fast_topk_transform_ragged_fused` | `sparse.topk_transform_ragged` | — | Top-k then transform indices into ragged (non-paged) KV layout | sm80+ |
| `sgl.flash_attn_varlen_func` | `attention.prefill` | — | CPU flash attention varlen forward | x86/aarch64 CPU |
| `sgl.fp8_blockwise_scaled_grouped_mm` | `moe.grouped_gemm_fp8` | — | Grouped (per-expert) FP8 block-scaled GEMM for MoE | sm90 |
| `sgl.fp8_blockwise_scaled_mm` | `gemm.fp8_blockwise` | — | FP8 GEMM with block-wise scaling factors | sm90 |
| `sgl.fp8_scaled_mm` | `gemm.fp8_scaled` | — | FP8 x FP8 -> out_dtype GEMM with per-tensor scales + opt bias | sm89/sm90 |
| `sgl.fp8_scaled_mm_cpu` | `gemm.fp8_scaled` | — | CPU FP8 scaled GEMM with block_size scales | x86 CPU |
| `sgl.fused_add_layernorm_cpu` | `norm.fused_add_layernorm` | — | CPU fused residual-add + layernorm | x86/aarch64 CPU |
| `sgl.fused_add_rmsnorm` | `norm.fused_add_rmsnorm` | `ks_fused_add_rmsnorm` | Fused residual-add then RMS norm in-place | sm80+ |
| `sgl.fused_add_rmsnorm_cpu` | `norm.fused_add_rmsnorm` | `ks_fused_add_rmsnorm` | CPU fused residual-add + RMS norm in-place | x86/aarch64 CPU |
| `sgl.fused_experts_cpu` | `moe.fused_full` | — | CPU fused MoE experts (grouped GEMM + activation + combine) | x86/aarch64 CPU |
| `sgl.fused_gdn_gating_cpu` | `ssm.gated_delta_gating` | — | CPU fused gated-delta-net gating computation | x86 CPU |
| `sgl.fused_linear_sigmoid_mul` | `gemm.fused_linear_sigmoid` | — | CPU fused linear -> sigmoid -> elementwise mul with post matrix | x86/aarch64 CPU |
| `sgl.fused_qk_norm_rope` | `rope.qk_norm_rope` | — | Fused QK RMSNorm + RoPE on packed QKV in-place | sm80+ |
| `sgl.fused_qkvzba_split_reshape_cat_contiguous_cpu` | `ssm.qkvzba_split` | — | CPU contiguous variant of QKVZBA split/reshape/concat | x86 CPU |
| `sgl.fused_qkvzba_split_reshape_cat_cpu` | `ssm.qkvzba_split` | — | CPU split/reshape/concat of mixed QKVZ + BA tensors (gated delta net) | x86 CPU |
| `sgl.fused_rmsnorm_gated_cpu` | `norm.gated_rmsnorm` | — | CPU Qwen3-next gated RMS norm (rmsnorm then gate mul) | x86/aarch64 CPU |
| `sgl.fused_sigmoid_gating_delta_rule_update_cpu` | `ssm.gated_delta_update` | — | CPU fused sigmoid-gating delta-rule recurrent state update | x86 CPU |
| `sgl.fwd` | `attention.prefill` | — | FlashAttention-3 forward (varlen/paged/kvcache/GQA/rope/fp8) | sm80+/sm90a |
| `sgl.fwd_kvcache_mla` | `attention.mla.sparse_decode` | — | FlashMLA decode attention with paged KV cache (dense or sparse) | sm90 |
| `sgl.fwd_kvcache_mla_fp8` | `attention.mla.decode` | — | FlashMLA decode attention with FP8 KV cache + descale | sm90 |
| `sgl.fwd_sparse` | `attention.sparse_prefill` | — | Sparse flash attention fwd with vertical+slash block sparsity | sm80+ |
| `sgl.gelu_and_mul` | `act.gelu_mul` | `ks_gelu_and_mul` | GELU(erf exact) gate then multiply | sm80+ |
| `sgl.gelu_and_mul_cpu` | `act.gelu_mul` | `ks_gelu_and_mul` | CPU GELU(erf)-and-mul activation | x86/aarch64 CPU |
| `sgl.gelu_quick` | `act.gelu_quick` | `ks_gelu` | QuickGELU: y = x*sigmoid(1.702*x) (ROCm/HIP only) | ROCm/HIP |
| `sgl.gelu_tanh_and_mul` | `act.gelu_tanh_mul` | `ks_gelu_and_mul` | GELU(tanh approx) gate then multiply | sm80+ |
| `sgl.gelu_tanh_and_mul_cpu` | `act.gelu_tanh_mul` | `ks_gelu_and_mul` | CPU GELU(tanh)-and-mul activation | x86/aarch64 CPU |
| `sgl.gemma3_rmsnorm_cpu` | `norm.gemma_rmsnorm` | `ks_gemma_rmsnorm` | CPU Gemma3 RMS norm variant | x86/aarch64 CPU |
| `sgl.gemma4_rmsnorm_cpu` | `norm.gemma_rmsnorm` | `ks_gemma_rmsnorm` | CPU Gemma4 RMS norm with scale_shift/with_scale | x86/aarch64 CPU |
| `sgl.gemma_fused_add_rmsnorm` | `norm.gemma_fused_add_rmsnorm` | — | Gemma fused residual-add then RMS norm in-place | sm80+ |
| `sgl.gemma_fused_add_rmsnorm_cpu` | `norm.gemma_fused_add_rmsnorm` | — | CPU Gemma fused residual-add + RMS norm in-place | x86/aarch64 CPU |
| `sgl.gemma_rmsnorm` | `norm.gemma_rmsnorm` | `ks_gemma_rmsnorm` | Gemma-style RMS norm: out = (x/RMS(x))*(weight+1) | sm80+ |
| `sgl.gemma_rmsnorm_cpu` | `norm.gemma_rmsnorm` | `ks_gemma_rmsnorm` | CPU Gemma RMS norm (weight+1) | x86/aarch64 CPU |
| `sgl.get_cutlass_w4a8_moe_mm_data` | `moe.prepare_input` | `ks_moe_compute_permutation` | Build expert offsets/problem-sizes/permutations for W4A8 MoE MM | sm90 |
| `sgl.get_mla_decoding_metadata` | `attention.mla.plan` | — | FlashMLA tile-scheduler metadata + num_splits for decode | sm90 |
| `sgl.get_mla_decoding_metadata_dense_fp8` | `attention.mla.plan` | — | FlashMLA dense-FP8 decode tile-scheduler metadata | sm90 |
| `sgl.get_scheduler_metadata` | `attention.plan` | — | Precompute FA3 tile-scheduler metadata for a batch | sm90 |
| `sgl.ggml_dequantize` | `quant.gguf_dequant` | — | GGUF/GGML dequantize quantized weight to dtype | sm80+ |
| `sgl.ggml_moe_a8` | `moe.gguf_grouped_gemm` | — | GGML quantized MoE matmul (a8) with sorted token/expert ids | sm80+ |
| `sgl.ggml_moe_a8_vec` | `moe.gguf_gemv` | — | GGML quantized MoE matrix-vector multiply (a8) | sm80+ |
| `sgl.ggml_moe_get_block_size` | `quant.gguf_meta` | — | Return GGML quant block size for a given type | sm80+ |
| `sgl.ggml_mul_mat_a8` | `gemm.gguf` | — | GGML quantized matrix-matrix multiply (a8 activations) | sm80+ |
| `sgl.ggml_mul_mat_vec_a8` | `gemm.gguf_gemv` | — | GGML quantized matrix-vector multiply (a8 activations) | sm80+ |
| `sgl.gptq_gemm` | `gemm.w4a16` | `ks_gemm_w4a16` | GPTQ quantized weight GEMM (2/3/4/8-bit) with optional shuffle | sm80+ |
| `sgl.gptq_shuffle` | `quant.gptq_shuffle` | — | GPTQ weight permutation/shuffle in-place for fast GEMM | sm80+ |
| `sgl.grouped_topk_cpu` | `moe.gate_grouped` | `ks_moe_gate_sigmoid_group_topk` | CPU grouped top-k MoE gate | x86/aarch64 CPU |
| `sgl.image_preprocess_cpu` | `util.image_preprocess` | — | CPU image preprocessor (resize/rescale/normalize/patchify) | x86/aarch64 CPU |
| `sgl.init_custom_ar` | `comm.init` | — | Initialize custom IPC all-reduce context | sm80+ |
| `sgl.int4_scaled_mm_cpu` | `gemm.w4a16` | `ks_gemm_w4a16` | CPU INT4 weight scaled GEMM with zeros/scales | x86 CPU |
| `sgl.int8_scaled_mm` | `gemm.w8a8` | `ks_gemm_w8a8` | INT8 x INT8 -> out_dtype GEMM with per-row/col scales + opt bias | sm80+ |
| `sgl.int8_scaled_mm_cpu` | `gemm.w8a8` | `ks_gemm_w8a8` | CPU INT8 scaled GEMM with per-row/col scales | x86/aarch64 CPU |
| `sgl.int8_scaled_mm_with_quant` | `gemm.int8_fused_quant` | `ks_gemm_w8a8` | CPU fused per-token int8 quant + scaled GEMM | x86/aarch64 CPU |
| `sgl.kimi_k2_moe_fused_gate` | `moe.gate_grouped` | `ks_moe_gate_sigmoid_group_topk` | Kimi-K2 single-group fused MoE gate (no grouped logic) | sm80+ |
| `sgl.l2norm_cpu` | `norm.l2norm` | `ks_layernorm` | CPU L2 normalization | x86/aarch64 CPU |
| `sgl.layernorm_cpu` | `norm.layernorm` | `ks_layernorm` | CPU layer normalization with optional bias | x86/aarch64 CPU |
| `sgl.merge_state_v2` | `attention.merge_state` | — | Merge two attention states (V,LSE) for split-KV attention | sm80+ |
| `sgl.min_p_sampling_from_probs` | `sampling.min_p` | — | Min-p sampling from probability distribution (MUSA) | MUSA |
| `sgl.moe_align_block_size` | `moe.align_block` | `ks_moe_compute_permutation` | Sort tokens by expert and pad to block_size for grouped MoE GEMM | sm80+ |
| `sgl.moe_fused_gate` | `moe.gate_grouped` | `ks_moe_gate_sigmoid_group_topk` | Hierarchical grouped top-k expert gate (group then in-group topk) | sm80+ |
| `sgl.moe_sum` | `moe.sum_reduce` | `ks_moe_unpermute` | Sum reduction of top-k expert outputs into output | sm80+ |
| `sgl.moe_sum_reduce` | `moe.sum_reduce` | `ks_moe_unpermute` | Sum-reduce expert outputs with routed scaling factor | sm80+ |
| `sgl.mscclpp_allreduce` | `comm.allreduce` | — | MSCCL++ based multi-GPU all-reduce | sm90 |
| `sgl.mscclpp_init_context` | `comm.allreduce` | — | Initialize MSCCL++ all-reduce communication context | sm90 |
| `sgl.multimodal_rotary_embedding_cpu` | `rope.apply_mrope` | — | CPU multimodal (M-RoPE) rotary embedding with sections | x86/aarch64 CPU |
| `sgl.musa_batched_rotary_embedding_contiguous` | `rope.apply` | `ks_rope` | MUSA batched rotary embedding with per-row cache offsets | MUSA |
| `sgl.musa_fused_gemv` | `gemm.fp8` | — | MUSA fused GEMV (fp8/w4a16/general) with opt swiglu/rmsnorm | MUSA |
| `sgl.musa_fused_moe_gemv` | `moe.fused_full` | — | MUSA fused MoE GEMV with routed weights + opt swiglu | MUSA |
| `sgl.musa_fused_mul_add` | `elementwise.mul_add` | `ks_axpby` | MUSA fused out = self*scale + bias | MUSA |
| `sgl.musa_rotary_embedding_contiguous` | `rope.apply` | `ks_rope` | MUSA rotary embedding on contiguous query/key | MUSA |
| `sgl.musa_top_k_top_p_sampling_from_probs` | `sampling.topk_topp` | `ks_sample` | Joint top-k+top-p sampling from probabilities (MUSA) | MUSA |
| `sgl.mxfp4_scaled_mm_cpu` | `gemm.fp4` | — | CPU MXFP4 scaled GEMM | x86 CPU |
| `sgl.per_token_quant_int8_cpu` | `quant.int8_per_token` | `ks_quantize_int8` | CPU per-token int8 quantization with scales | x86/aarch64 CPU |
| `sgl.prepare_moe_input` | `moe.prepare_input` | `ks_moe_compute_permutation` | Compute expert offsets, problem sizes and permutations from topk_ids | sm80+ |
| `sgl.qkv_proj_with_rope` | `gemm.qkv_proj_rope_fused` | — | CPU fused QKV projection + RoPE with weight absorption (MLA) | x86 CPU |
| `sgl.qkv_proj_with_rope_fused_weight` | `gemm.qkv_proj_rope_fused` | — | CPU fused QKV proj + RoPE with fused qkv_a_proj weight (MLA) | x86 CPU |
| `sgl.qr_all_reduce` | `comm.allreduce` | — | Quick all-reduce (ROCm) one-shot reduce | ROCm/HIP |
| `sgl.qserve_w4a8_per_chn_gemm` | `gemm.w4a8` | — | QServe W4A8 per-channel GEMM (int4 weight, int8 act) | sm80+ |
| `sgl.qserve_w4a8_per_group_gemm` | `gemm.w4a8` | — | QServe W4A8 per-group GEMM with zeros + i8 scales | sm80+ |
| `sgl.reconstruct_indices_from_tree_mask` | `spec.reconstruct_tree` | — | Reconstruct retrieve indices/positions from a tree attention mask | sm80+ |
| `sgl.rmsnorm` | `norm.rmsnorm` | `ks_rmsnorm` | RMS normalization: out = (x / RMS(x)) * weight | sm80+ |
| `sgl.rmsnorm_cpu` | `norm.rmsnorm` | `ks_rmsnorm` | CPU RMS normalization | x86/aarch64 CPU |
| `sgl.rope_pool_fused` | `rope.apply_pool_fused` | — | Metal: NeoX RoPE on Q/K + scatter K/V into MLX KV pool | Apple Metal |
| `sgl.rotary_embedding` | `rope.apply` | `ks_rope` | Apply rotary position embedding to query/key in-place | sm80+ |
| `sgl.rotary_embedding_cpu` | `rope.apply` | `ks_rope` | CPU rotary position embedding on query/key | x86/aarch64 CPU |
| `sgl.segment_packbits` | `spec.packbits` | — | Segmented bit-packing of boolean tensor by indptr segments | sm80+ |
| `sgl.sgl_per_token_group_quant_8bit` | `quant.fp8_per_token_group` | `ks_quantize_fp8` | Per-token group quantize activations to fp8/int8 with scales | sm80+ |
| `sgl.sgl_per_token_group_quant_8bit_v2` | `quant.fp8_per_token_group` | `ks_quantize_fp8` | v2 per-token group quant with optional fused silu_and_mul + masked_m | sm90 |
| `sgl.sgl_per_token_quant_fp8` | `quant.fp8_per_token` | `ks_quantize_fp8` | Per-token (whole-row) FP8 quantization with scales | sm89/sm90 |
| `sgl.shared_expert_cpu` | `moe.fused_full` | — | CPU shared-expert MLP with routed scaling/combine | x86 CPU |
| `sgl.shm_allgather` | `comm.allgather` | — | CPU shared-memory all-gather along dim | x86/aarch64 CPU |
| `sgl.shm_allreduce` | `comm.allreduce` | — | CPU shared-memory all-reduce | x86/aarch64 CPU |
| `sgl.shuffle_rows` | `moe.shuffle_rows` | `ks_moe_permute` | Gather/scatter rows by dst->src map (MoE permutation) | sm80+ |
| `sgl.silu_and_mul` | `act.silu_mul` | `ks_silu_and_mul` | SiLU gate then multiply: out = silu(x[:d])*x[d:] | sm80+ |
| `sgl.silu_and_mul_cpu` | `act.silu_mul` | `ks_silu_and_mul` | CPU SiLU-and-mul activation | x86/aarch64 CPU |
| `sgl.sparse_decode_fwd` | `attention.mla.sparse_decode` | — | FlashMLA sparse decode attention forward (topk indices) | sm90 |
| `sgl.sparse_prefill_fwd` | `attention.mla.sparse_prefill` | — | FlashMLA sparse attention prefill (topk indices, d_v=512) | sm90 |
| `sgl.store_cache_cpu` | `kvcache.store` | `ks_reshape_and_cache` | CPU store K/V into paged cache by indices | x86/aarch64 CPU |
| `sgl.top_k_renorm_probs` | `sampling.topk_renorm` | — | Renormalize probabilities keeping only top-k entries | sm80+ |
| `sgl.top_p_renorm_probs` | `sampling.topp_renorm` | — | Renormalize probabilities keeping top-p nucleus mass | sm80+ |
| `sgl.top_p_sampling_from_probs` | `sampling.topp` | `ks_sample` | Top-p (nucleus) sampling from probabilities (MUSA) | MUSA |
| `sgl.topk_sigmoid` | `moe.gate_sigmoid` | `ks_moe_gate_sigmoid_group_topk` | MoE top-k sigmoid routing with optional correction bias | sm80+ |
| `sgl.topk_sigmoid_cpu` | `moe.gate_sigmoid` | `ks_moe_gate_sigmoid_group_topk` | CPU MoE top-k sigmoid routing | x86/aarch64 CPU |
| `sgl.topk_softmax` | `moe.gate_softmax` | `ks_moe_gate_softmax_topk` | MoE top-k softmax routing with optional softcap + correction bias | sm80+ |
| `sgl.topk_softmax_cpu` | `moe.gate_softmax` | `ks_moe_gate_softmax_topk` | CPU MoE top-k softmax routing | x86/aarch64 CPU |
| `sgl.transfer_kv_all_layer` | `kvcache.transfer` | — | All-layer K/V cache transfer between buffers by index | sm80+ |
| `sgl.transfer_kv_all_layer_direct_lf_pf` | `kvcache.transfer` | — | All-layer direct KV transfer layer-first->page-first via ptr lists | sm80+ |
| `sgl.transfer_kv_all_layer_lf_pf` | `kvcache.transfer` | — | All-layer KV transfer layer-first src -> page-first dst | sm80+ |
| `sgl.transfer_kv_all_layer_lf_ph` | `kvcache.transfer` | — | All-layer KV transfer layer-first src -> page+head dst | sm80+ |
| `sgl.transfer_kv_all_layer_mla` | `kvcache.transfer_mla` | — | All-layer MLA cache transfer by index | sm80+ |
| `sgl.transfer_kv_all_layer_mla_lf_pf` | `kvcache.transfer_mla` | — | All-layer MLA transfer layer-first src -> page-first dst | sm80+ |
| `sgl.transfer_kv_direct` | `kvcache.transfer` | — | Direct multi-layer KV transfer via tensor-list pointers | sm80+ |
| `sgl.transfer_kv_per_layer` | `kvcache.transfer` | — | Copy per-layer K/V cache entries between buffers by index | sm80+ |
| `sgl.transfer_kv_per_layer_direct_pf_lf` | `kvcache.transfer` | — | Per-layer direct KV transfer page-first->layer-first via ptr lists | sm80+ |
| `sgl.transfer_kv_per_layer_mla` | `kvcache.transfer_mla` | — | Per-layer MLA (single combined cache) transfer by index | sm80+ |
| `sgl.transfer_kv_per_layer_mla_pf_lf` | `kvcache.transfer_mla` | — | Per-layer MLA transfer page-first src -> layer-first dst | sm80+ |
| `sgl.transfer_kv_per_layer_pf_lf` | `kvcache.transfer` | — | Per-layer KV transfer page-first src -> layer-first dst | sm80+ |
| `sgl.transfer_kv_per_layer_ph_lf` | `kvcache.transfer` | — | Per-layer KV transfer page+head src -> layer-first dst | sm80+ |
| `sgl.tree_speculative_sampling_target_only` | `spec.verify_sampling` | — | Tree speculative decoding sampling, target-distribution verification | sm80+ |
| `sgl.varlen_fwd_sparse` | `attention.sparse_prefill` | — | Varlen sparse flash attention fwd (vertical+slash sparsity) | sm80+ |
| `sgl.verify_tree_greedy` | `spec.verify_greedy` | — | Greedy verification of speculative draft tree against target argmax | sm80+ |
| `sgl.weak_ref_tensor` | `util.weak_ref` | — | Create a non-owning weak reference tensor sharing storage | sm80+ |
| `sgl.weight_packed_linear` | `gemm.dense` | `ks_gemm` | CPU packed-weight linear (matmul + opt bias) | x86/aarch64 CPU |
| `vllm.LLMM1` | `gemm.skinny` | — | ROCm custom matrix-vector multiplication GEMM | ROCm |
| `vllm.all_reduce` | `comm.allreduce` | — | Custom one-shot/two-shot GPU all-reduce over registered IPC buffers | sm70+ (NVLink) |
| `vllm.allocate_shared_buffer_and_handle` | `comm.init` | — | Allocate a shared GPU buffer and return its pointer and IPC handle | any |
| `vllm.allspark_w8a16_gemm` | `gemm.w8a16` | — | AllSpark Ampere W8A16 fused GEMM (int8 weight, fp16/bf16 activation) | sm80 |
| `vllm.apply_repetition_penalties_` | `sampling.repetition_penalty` | — | Apply repetition penalties to logits in-place using prompt/output token masks | sm70+ / ROCm |
| `vllm.awq_dequantize` | `quant.awq_dequant` | — | Dequantize AWQ int4 weights back to fp16 | sm75+ |
| `vllm.awq_gemm` | `gemm.w4a16` | `ks_gemm_w4a16` | AWQ quantized GEMM (int4 weight, fp16 activation) with split-k | sm75+ |
| `vllm.awq_marlin_repack` | `quant.marlin_repack` | — | Repack AWQ-format quantized weights into Marlin layout | sm80+ |
| `vllm.batched_moe_align_block_size` | `moe.align_block` | `ks_moe_compute_permutation` | moe_align_block_size for batched-expert format using per-expert token counts | sm70+ |
| `vllm.concat_and_cache_mla` | `kvcache.concat_cache_mla` | — | Concat kv_c and k_pe and write into MLA paged cache at slot_mapping | sm70+ |
| `vllm.concat_and_cache_mla_rope_fused` | `kvcache.concat_cache_mla_rope` | — | Fused RoPE on q_pe/k_pe then concat with kv_c and write to MLA cache | sm70+ |
| `vllm.concat_mla_q` | `kvcache.concat_mla_q` | — | Concatenate ql_nope and q_pe into a single MLA query tensor | sm70+ |
| `vllm.convert_fp8` | `kvcache.convert_fp8` | — | Convert key/value cache to/from fp8 data type with scale | sm89+ |
| `vllm.cp_gather_and_upconvert_fp8_kv_cache` | `kvcache.convert_fp8` | — | Context-parallel gather + upconvert fp8 KV cache into higher precision dst | sm89+ |
| `vllm.cp_gather_cache` | `kvcache.gather` | — | Context-parallel gather of cache blocks into contiguous dst by block_table | sm70+ |
| `vllm.cp_gather_indexer_k_quant_cache` | `kvcache.gather_indexer` | — | Context-parallel gather of quantized indexer K cache, outputting K and scales | sm89+ |
| `vllm.cutlass_encode_and_reorder_int4b` | `quant.int4_reorder` | — | Encode and reorder int4 weight matrix into CUTLASS W4A8 layout | sm90 |
| `vllm.cutlass_encode_and_reorder_int4b_grouped` | `quant.int4_reorder` | — | Encode and reorder grouped int4 weight tensors into CUTLASS W4A8 grouped layout | sm90 |
| `vllm.cutlass_fp4_group_mm` | `moe.grouped_gemm_fp4` | — | CUTLASS NVFP4 block-scaled grouped GEMM for MoE | sm100 |
| `vllm.cutlass_group_gemm_supported` | `quant.capability_query` | — | Query whether CUTLASS grouped GEMM is supported for a given device capability | any |
| `vllm.cutlass_mla_decode` | `attention.mla.decode` | — | CUTLASS Multi-head Latent Attention (MLA) decode against compressed kv_c+k_pe paged cache | sm90 |
| `vllm.cutlass_moe_mm` | `moe.grouped_gemm` | `ks_moe_grouped_gemm` | CUTLASS w8a8 grouped (per-expert) GEMM for fused MoE | sm90+ |
| `vllm.cutlass_mxfp4_group_mm` | `moe.grouped_gemm_fp4` | — | CUTLASS MXFP4 x MXFP4 block-scaled grouped GEMM for MoE | sm100 |
| `vllm.cutlass_mxfp8_grouped_mm` | `moe.grouped_gemm_fp8` | — | Expert-specialization MXFP8 blockscaled grouped GEMM for MoE | sm100+ |
| `vllm.cutlass_pack_scale_fp8` | `quant.fp8` | `ks_quantize_fp8` | Pack FP8 scales into the layout expected by CUTLASS W4A8 kernels | sm90 |
| `vllm.cutlass_scaled_fp4_mm` | `gemm.fp4` | — | CUTLASS NVFP4 block-scaled GEMM: out = alpha*(a@b) with per-block scales | sm100 |
| `vllm.cutlass_scaled_mm` | `gemm.w8a8` | `ks_gemm_w8a8` | CUTLASS w8a8 scaled GEMM (symmetric per-tensor or per-row/col), optional bias: out = scal… | sm80+ |
| `vllm.cutlass_scaled_mm_azp` | `gemm.w8a8_azp` | `ks_gemm_w8a8` | CUTLASS w8a8 scaled GEMM with asymmetric (zero-point) quantization correction | sm80+ |
| `vllm.cutlass_scaled_mm_supports_block_fp8` | `quant.capability_query` | — | Query whether CUTLASS scaled_mm supports block-fp8 quantization (DeepSeekV3) | any |
| `vllm.cutlass_scaled_mm_supports_fp4` | `quant.capability_query` | — | Query whether cutlass_scaled_mm_fp4 is supported for a given device capability | any |
| `vllm.cutlass_scaled_mm_supports_fp8` | `quant.capability_query` | — | Query whether CUTLASS scaled_mm fp8 path is supported for a given device capability | any |
| `vllm.cutlass_w4a8_mm` | `gemm.fp8` | — | CUTLASS W4A8 mixed-precision GEMM (int4 weight, fp8 activation) with group/channel/token … | sm90 |
| `vllm.cutlass_w4a8_moe_mm` | `moe.grouped_gemm_w4a8` | — | CUTLASS W4A8 grouped (per-expert) GEMM for MoE | sm90 |
| `vllm.dispose` | `comm.init` | — | Dispose a custom all-reduce context | any |
| `vllm.dsv3_fused_a_gemm` | `gemm.dsv3_fused_a` | — | DeepSeek-V3 fused A (q/kv down-projection) GEMM, bf16 only, 1-16 tokens | sm90+ |
| `vllm.dsv3_router_gemm` | `moe.misc` | — | DeepSeek-V3 optimized router GEMM (gating projection) for SM90+ | sm90+ |
| `vllm.dynamic_per_token_scaled_fp8_quant` | `quant.fp8_per_token` | `ks_quantize_fp8` | Dynamic per-token FP8 quantization with optional scale upper-bound | sm89+ / ROCm |
| `vllm.dynamic_scaled_fp8_quant` | `quant.fp8_dynamic` | `ks_quantize_fp8` | Dynamic per-tensor FP8 quantization computing scale on the fly | sm89+ / ROCm |
| `vllm.dynamic_scaled_int8_quant` | `quant.int8_dynamic` | `ks_quantize_int8` | INT8 quantization computing scale dynamically, optional asymmetric zero-point | sm70+ / ROCm |
| `vllm.fatrelu_and_mul` | `act.fatrelu_mul` | — | FATReLU gated activation: FATReLU(x[:d], threshold) * x[d:] | sm70+ / ROCm |
| `vllm.fp32_router_gemm` | `gemm.router` | — | BF16/FP32 x FP32 -> FP32 router GEMM for H=3072, E=256, M<=32 | sm90+ |
| `vllm.free_shared_buffer` | `comm.init` | — | Free a shared GPU buffer by pointer | any |
| `vllm.fused_add_rms_norm` | `norm.fused_add_rmsnorm` | `ks_fused_add_rmsnorm` | In-place fused residual-add + RMS normalization | sm70+ / ROCm |
| `vllm.fused_add_rms_norm_static_fp8_quant` | `norm.fused_add_rmsnorm_quant` | — | Fused residual-add + RMS norm + static FP8 quantization | sm89+ / ROCm |
| `vllm.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert` | `rope.dsv4_k_norm_rope_cache` | — | FlashInfer-V4 DeepSeek-V4 fused QK-norm+RoPE writing Q in-place (bf16) and KV into 512-wi… | sm90+ |
| `vllm.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert` | `rope.dsv4_k_norm_rope_cache` | — | FlashInfer-V4 DeepSeek-V4 fused QK-norm+RoPE writing Q to separate FP8 tensor and KV into… | sm90+ |
| `vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` | `rope.dsv4_k_norm_rope_cache` | — | DeepSeek-V4 MLA fused: per-head Q RMSNorm + GPT-J RoPE for Q, and RoPE + UE8M0 FP8 quant … | sm90+ |
| `vllm.fused_qk_norm_rope` | `rope.qk_norm_rope` | — | Fused per-head Q/K RMS norm + RoPE applied to a packed QKV tensor | sm80+ |
| `vllm.gather_and_maybe_dequant_cache` | `kvcache.gather_dequant` | — | Gather paged cache blocks to a contiguous dst, dequantizing fp8 cache if needed | sm70+ |
| `vllm.gelu_and_mul` | `act.gelu_mul` | `ks_gelu_and_mul` | GeGLU with exact (none) GELU approximation: GELU(x[:d]) * x[d:] | sm70+ / ROCm |
| `vllm.gelu_fast` | `act.gelu_fast` | `ks_gelu` | Fast approximate GELU elementwise activation | sm70+ / ROCm |
| `vllm.gelu_new` | `act.gelu_new` | `ks_gelu` | GPT-2 style new GELU elementwise activation | sm70+ / ROCm |
| `vllm.gelu_quick` | `act.gelu_quick` | `ks_gelu` | Quick GELU (x*sigmoid(1.702x)) elementwise activation | sm70+ / ROCm |
| `vllm.gelu_tanh_and_mul` | `act.gelu_tanh_mul` | `ks_gelu_and_mul` | GeGLU with tanh GELU approximation: GELU_tanh(x[:d]) * x[d:] | sm70+ / ROCm |
| `vllm.get_cuda_view_from_cpu_tensor` | `util.cuda_view` | — | Create a CUDA-accessible view from a (pinned) CPU tensor | any CUDA |
| `vllm.get_cutlass_batched_moe_mm_data` | `moe.prepare_input` | `ks_moe_compute_permutation` | Compute expert_offsets and problem sizes for batched-expert-format CUTLASS grouped MoE | sm90+ |
| `vllm.get_cutlass_moe_mm_data` | `moe.prepare_input` | `ks_moe_compute_permutation` | Compute expert_offsets, problem_sizes and input/output permutations for CUTLASS grouped M… | sm90+ |
| `vllm.get_cutlass_moe_mm_problem_sizes_from_expert_offsets` | `moe.prepare_input` | `ks_moe_compute_permutation` | Compute per-expert problem sizes from expert_first_token_offset (from moe_permute) | sm90+ |
| `vllm.get_device_attribute` | `util.device_query` | — | Query a CUDA device attribute by id | any CUDA |
| `vllm.get_graph_buffer_ipc_meta` | `comm.init` | — | Get IPC metadata for CUDA-graph-captured all-reduce buffers | any |
| `vllm.get_max_shared_memory_per_block_device_attribute` | `util.device_query` | — | Query the maximum shared memory per block for a device | any CUDA |
| `vllm.ggml_dequantize` | `quant.gguf_dequant` | — | Dequantize a GGML/GGUF quantized weight to a float dtype | sm60+ |
| `vllm.ggml_moe_a8` | `moe.gguf_grouped_gemm` | — | GGML MoE grouped GEMM (a8) using sorted token ids and expert ids | sm60+ |
| `vllm.ggml_moe_a8_vec` | `moe.gguf_gemv` | — | GGML MoE matrix-vector kernel (a8) for decode using topk_ids | sm60+ |
| `vllm.ggml_moe_get_block_size` | `quant.gguf_meta` | — | Return GGML block size for a given quantization type | any |
| `vllm.ggml_mul_mat_a8` | `gemm.gguf` | — | GGML mmq matrix-matrix kernel (a8 activation) over quantized weight | sm60+ |
| `vllm.ggml_mul_mat_vec_a8` | `gemm.gguf_gemv` | — | GGML mmvq matrix-vector kernel (a8 activation) over quantized weight | sm60+ |
| `vllm.gptq_gemm` | `gemm.w4a16` | `ks_gemm_w4a16` | GPTQ quantized GEMM via exllama/exllamav2 kernels (int4/int8 weight) | sm60+ |
| `vllm.gptq_gemm_rdna3` | `gemm.w4a16` | `ks_gemm_w4a16` | W4A16 GPTQ GEMM for AMD RDNA3 (gfx1100) | ROCm gfx1100 |
| `vllm.gptq_gemm_rdna3_wmma` | `gemm.w4a16` | `ks_gemm_w4a16` | W4A16 GPTQ GEMM (WMMA) for AMD RDNA3 (gfx1100) | ROCm gfx1100 |
| `vllm.gptq_marlin_repack` | `quant.marlin_repack` | — | Repack GPTQ-format quantized weights into Marlin layout | sm80+ |
| `vllm.gptq_shuffle` | `quant.gptq_shuffle` | — | Post-process/shuffle GPTQ quantized weights in place | sm60+ |
| `vllm.grouped_topk` | `moe.gate_grouped` | `ks_moe_gate_sigmoid_group_topk` | Grouped top-k expert routing (DeepSeek-style group-limited routing) returning weights and… | sm70+ |
| `vllm.hadacore_transform` | `util.hadamard` | — | Fast Hadamard transform (Hadacore) over input, optionally in place | sm80+ |
| `vllm.indexer_k_quant_and_cache` | `kvcache.indexer_quant` | — | Quantize K (per quant_block_size) and write into indexer KV cache at slot_mapping | sm89+ |
| `vllm.init_custom_ar` | `comm.init` | — | Initialize custom all-reduce context over IPC tensors | sm70+ (NVLink) |
| `vllm.init_custom_qr` | `comm.init` | — | Initialize ROCm Quick Reduce all-reduce context | ROCm |
| `vllm.machete_mm` | `gemm.fp8` | — | Machete mixed-precision GEMM for Hopper: A@B with quantized B (group/channel/token scales… | sm90 |
| `vllm.machete_prepack_B` | `quant.weight_prepack` | — | Prepack/reorder weight B into Machete kernel layout | sm90 |
| `vllm.machete_supported_schedules` | `gemm.w4a16_machete` | `ks_gemm_w4a16` | List supported Machete kernel schedules for given type combination | sm90 |
| `vllm.marlin_gemm` | `gemm.fp8` | — | Marlin optimized quantized GEMM supporting GPTQ/AWQ/FP8/NVFP4/MXFP4 with optional bias, z… | sm80+ |
| `vllm.marlin_int4_fp8_preprocess` | `quant.fp8` | `ks_quantize_fp8` | Preprocess W-int4 / A-fp8 weight (and optional zeros) for Marlin kernel | sm89+ |
| `vllm.merge_attn_states` | `attention.merge_state` | — | Merge two partial attention results (prefix+suffix) with their LSEs into a single output … | sm70+ |
| `vllm.meta_size` | `comm.init` | — | Return metadata size for custom all-reduce signal pad | any |
| `vllm.minimax_allreduce_rms` | `comm.allreduce_rmsnorm` | — | Fused all-reduce + RMS norm (MiniMax) across nranks | sm80+ (CUDA) |
| `vllm.minimax_allreduce_rms_qk` | `comm.allreduce_rmsnorm` | — | Fused all-reduce + separate Q and K RMS norm (MiniMax) returning normed Q,K | sm80+ (CUDA) |
| `vllm.moe_align_block_size` | `moe.align_block` | `ks_moe_compute_permutation` | Align/sort tokens per expert so each expert block count is divisible by block_size for Mo… | sm70+ |
| `vllm.moe_lora_align_block_size` | `moe.align_block_lora` | — | MoE+LoRA aware token alignment producing sorted ids, expert ids, adapter/lora id maps | sm70+ |
| `vllm.moe_permute` | `moe.permute` | `ks_moe_permute` | Permute/scatter MoE input rows by expert assignment, producing permuted input and expert … | sm80+ |
| `vllm.moe_permute_sort_workspace_size` | `moe.permute_meta` | — | Compute scratch workspace size for moe_permute sorting | any |
| `vllm.moe_permute_unpermute_supported` | `moe.unpermute_combine` | `ks_moe_unpermute` | Query whether moe_permute/unpermute kernels are supported on current build | any |
| `vllm.moe_permute_with_scratch` | `moe.permute` | `ks_moe_permute` | moe_permute variant using explicit scratch buffers for sorting | sm80+ |
| `vllm.moe_sum` | `moe.sum_reduce` | `ks_moe_unpermute` | Sum per-expert partial MoE results into a single output tensor | sm70+ |
| `vllm.moe_unpermute` | `moe.unpermute_combine` | `ks_moe_unpermute` | Unpermute/gather MoE expert outputs back to token order with topk weight reduction | sm80+ |
| `vllm.moe_wna16_gemm` | `moe.grouped_gemm_w4a16` | — | MoE WNA16 (weight-only int4/int8, fp16 act) grouped GEMM using sorted token/expert ids | sm80+ |
| `vllm.moe_wna16_marlin_gemm` | `moe.grouped_gemm_w4a16_marlin` | — | Marlin-based MoE WNA16 grouped GEMM with topk weight fusion and full Marlin tuning params | sm80+ |
| `vllm.mul_and_silu` | `act.silu_mul` | `ks_silu_and_mul` | Gated activation x[:d] * SiLU(x[d:]) (mul-then-silu order) | sm70+ / ROCm |
| `vllm.mxfp4_experts_quant` | `quant.fp4_experts` | — | Per-expert MXFP4 quantization (32-element blocks, E8M0 scale factors) | sm100 |
| `vllm.mxfp8_experts_quant` | `quant.mxfp8` | — | Expert-specialization MXFP8 blockscaled grouped quantization using problem/expert/blocksc… | sm100+ |
| `vllm.open_mem_handle` | `comm.init` | — | Open a remote shared-memory IPC handle, returning a device pointer | any |
| `vllm.paged_attention` | `attention.decode` | — | ROCm custom PagedAttention (MFMA-based) with optional fp8 output and query_start_loc | ROCm (gfx9) |
| `vllm.paged_attention_v1` | `attention.decode` | — | PagedAttention V1: single-kernel attention of query against paged KV cache (whole-sequenc… | sm70+ |
| `vllm.paged_attention_v2` | `attention.decode` | — | PagedAttention V2: split-KV partitioned attention with separate reduction (exp_sums/max_l… | sm70+ |
| `vllm.per_token_group_fp8_quant` | `quant.fp8_per_token_group` | `ks_quantize_fp8` | Per-token-group FP8 quantization producing quantized tensor and group scales (fusable wit… | sm89+ |
| `vllm.per_token_group_fp8_quant_packed` | `quant.fp8_per_token_group` | `ks_quantize_fp8` | Per-token-group 8-bit quant producing UE8M0-packed, TMA-aligned scales for DeepGEMM | sm90+ |
| `vllm.per_token_group_quant_int8` | `quant.fp8_per_token_group` | `ks_quantize_fp8` | Per-token-group INT8 quantization producing quantized tensor and group scales | sm70+ |
| `vllm.permute_cols` | `quant.permute_cols` | — | Permute columns of matrix A by perm (weight reordering helper) | sm80+ |
| `vllm.persistent_masked_m_silu_mul_quant` | `act.silu_mul_quant` | — | Persistent masked grouped (per-expert) SiLU*Mul + per-group FP8 quant for MoE; counts mas… | sm90+ |
| `vllm.persistent_topk` | `sparse.topk_select` | — | Persistent-kernel top-k selection over variable-length logits | sm70+ / ROCm |
| `vllm.qr_all_reduce` | `comm.allreduce` | — | ROCm Quick Reduce all-reduce with optional quantization level and bf16->half cast | ROCm |
| `vllm.qr_destroy` | `comm.init` | — | Destroy a ROCm Quick Reduce context | ROCm |
| `vllm.qr_get_handle` | `comm.init` | — | Get IPC handle for ROCm Quick Reduce buffer | ROCm |
| `vllm.qr_max_size` | `comm.init` | — | Return max supported buffer size for ROCm Quick Reduce | ROCm |
| `vllm.qr_open_handles` | `comm.init` | — | Open peer IPC handles for ROCm Quick Reduce | ROCm |
| `vllm.rearrange_kn_weight_as_n32k16_order` | `quant.weight_reorder` | — | Reorder weight/scales/zeros into N32K16 layout for AllSpark W8A16 kernel | sm80 |
| `vllm.register_buffer` | `comm.init` | — | Register IPC buffers with a custom all-reduce context | any |
| `vllm.register_graph_buffers` | `comm.init` | — | Register CUDA-graph buffers (handles/offsets) for custom all-reduce | any |
| `vllm.reshape_and_cache` | `kvcache.reshape_and_cache` | `ks_reshape_and_cache` | Reshape key/value and write into paged KV cache at slot_mapping (with kv fp8 scaling) | sm70+ |
| `vllm.reshape_and_cache_flash` | `kvcache.reshape_and_cache_flash` | `ks_reshape_and_cache` | Reshape key/value and write into FlashAttention-layout paged KV cache (with kv fp8 scalin… | sm70+ |
| `vllm.rms_norm` | `norm.rmsnorm` | `ks_rmsnorm` | Root Mean Square (RMS) normalization of input tensor with weight | sm70+ / ROCm |
| `vllm.rms_norm_dynamic_per_token_quant` | `norm.rmsnorm_per_token_quant` | — | Fused RMS norm + dynamic per-token quant (fp8/int8), optional residual add and scale uppe… | sm89+ / ROCm |
| `vllm.rms_norm_per_block_quant` | `norm.rmsnorm_per_block_quant` | — | Fused RMS norm + per-block (group_size) quantization, optional residual and transposed-sc… | sm89+ |
| `vllm.rms_norm_static_fp8_quant` | `norm.rmsnorm_quant` | — | RMS norm fused with static (precomputed-scale) FP8 output quantization | sm89+ / ROCm |
| `vllm.rotary_embedding` | `rope.apply` | `ks_rope` | Apply GPT-NeoX or GPT-J style rotary positional embedding to query and key in-place | sm70+ / ROCm |
| `vllm.scaled_fp4_experts_quant` | `quant.fp4_experts` | — | Per-expert NVFP4 block quantization using expert offset arrays | sm100 |
| `vllm.scaled_fp4_quant` | `quant.fp4` | — | Quantize to NVFP4 (e2m1) block-scaled tensor, returning packed data and block scales | sm100 |
| `vllm.scaled_fp4_quant.out` | `quant.fp4` | — | Out-variant of scaled_fp4_quant writing into preallocated output and output_scale | sm100 |
| `vllm.selective_scan_fwd` | `ssm.selective_scan` | — | Mamba selective state-space scan forward (u,delta,A,B,C -> output) with chunked/paged SSM… | sm70+ |
| `vllm.shuffle_rows` | `moe.shuffle_rows` | `ks_moe_permute` | Row shuffle/gather for MoE using a dst-to-src index map | sm70+ |
| `vllm.silu_and_mul` | `act.silu_mul` | `ks_silu_and_mul` | SwiGLU: SiLU(x[:d]) * x[d:] gated activation | sm70+ / ROCm |
| `vllm.silu_and_mul_mxfp4_experts_quant` | `quant.fp4_experts` | — | Fused SiLU+Mul + per-expert MXFP4 quantization for MoE | sm100 |
| `vllm.silu_and_mul_nvfp4_quant` | `quant.fp4` | — | Fused SiLU+Mul + NVFP4 block quantization (single tensor) | sm100 |
| `vllm.silu_and_mul_per_block_quant` | `act.silu_mul_quant` | — | Fused SiLU+Mul + per-block (group_size) quantization with optional scale_ub and transpose… | sm89+ |
| `vllm.silu_and_mul_quant` | `act.silu_mul_quant` | — | Fused SiLU+Mul (SwiGLU) with FP8 quantization to result using given scale | sm89+ |
| `vllm.silu_and_mul_scaled_fp4_experts_quant` | `quant.fp4_experts` | — | Fused SiLU+Mul + per-expert NVFP4 block quantization for MoE | sm100 |
| `vllm.silu_and_mul_with_clamp` | `act.silu_mul` | `ks_silu_and_mul` | SwiGLU activation with input clamping to [-limit, limit] | sm70+ / ROCm |
| `vllm.sm100_cutlass_mla_decode` | `attention.mla.decode` | — | SM100 CUTLASS MLA decode with KV-splits, writes out + LSE | sm100 |
| `vllm.sm100_cutlass_mla_get_workspace_size` | `attention.mla.workspace` | — | Compute workspace byte size required by sm100_cutlass_mla_decode | sm100 |
| `vllm.static_scaled_fp8_quant` | `quant.fp8_static` | `ks_quantize_fp8` | FP8 quantization with given static scale; per-tensor/channel/token/2D-group via group_sha… | sm89+ / ROCm |
| `vllm.static_scaled_int8_quant` | `quant.int8_static` | `ks_quantize_int8` | INT8 quantization with given (static) per-tensor/channel scale, optional asymmetric zero-… | sm70+ / ROCm |
| `vllm.swap_blocks` | `kvcache.swap_blocks` | — | Copy/swap cache blocks between src and dst tensors per block_mapping | sm70+ |
| `vllm.swap_blocks_batch` | `kvcache.swap_blocks` | — | Batch block swap submitting all block copies in a single driver call | CPU dispatch |
| `vllm.swigluoai_and_mul` | `act.silu_mul` | `ks_silu_and_mul` | OpenAI-style SwiGLU gated activation with alpha and clamp limit | sm70+ / ROCm |
| `vllm.top_k_per_row_decode` | `sparse.topk_select` | — | Optimized per-row top-k selection over logits for decode | sm70+ / ROCm |
| `vllm.top_k_per_row_prefill` | `sparse.topk_select` | — | Optimized per-row top-k selection over logits for prefill (variable row ranges) | sm70+ / ROCm |
| `vllm.topk_sigmoid` | `moe.gate_sigmoid` | `ks_moe_gate_sigmoid_group_topk` | Apply top-k sigmoid to gating output for MoE routing | sm70+ |
| `vllm.topk_softmax` | `moe.gate_softmax` | `ks_moe_gate_softmax_topk` | Apply top-k softmax to gating output to produce expert weights/indices for MoE routing | sm70+ |
| `vllm.topk_softplus_sqrt` | `moe.gate_softplus` | — | Apply top-k softplus-sqrt routing with routed scaling factor for MoE | sm70+ |
| `vllm.weak_ref_tensor` | `util.weak_ref` | — | Create a weak reference (alias) tensor from a CUDA tensor's raw data pointer | any CUDA |
| `vllm.wvSplitK` | `gemm.skinny` | — | ROCm skinny matrix-matrix multiplication GEMM with optional bias | ROCm |
| `vllm.wvSplitKQ` | `gemm.skinny` | — | ROCm wvSplitK skinny GEMM for fp8 with per-tensor scales | ROCm |
| `vllm.wvSplitKrc` | `gemm.skinny` | — | ROCm skinny matrix-matrix multiplication GEMM (rc variant) with optional bias | ROCm |

---

## (b) Grouped by logical op — the flattened "select-optimal" menu

Per logical op: every atomic provider across the 3 libs side by side, plus the kernel-set op it maps to.
Logical ops are ordered by family, then by number of providing libs (most-shared first).
Rows marked **[multi-lib]** offer a real cross-library choice.

### attention

#### `attention.decode` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.decode_attention_cpu` | CPU decode attention over paged KV with req_to_token mapping | x86/aarch64 CPU | fp16/bf16/fp32 |
| flashinfer | `flashinfer.BatchDecodeWithPagedKVCacheWrapper.run` | Batched decode attention over paged KV cache (single query step per request) | sm80+ | fp16/bf16/fp8 |
| flashinfer | `flashinfer.cudnn_batch_decode_with_kv_cache` | cuDNN SDPA decode attention launcher | sm90/sm100 | fp16/bf16/fp8 |
| flashinfer | `flashinfer.single_decode_with_kv_cache` | Single-request decode attention (1 query step) over contiguous K/V | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.trtllm_batch_decode_with_kv_cache` | TRT-LLM paged decode attention (XQA/optimized GQA) | sm90/sm100 | fp16/bf16/fp8/fp4 |
| flashinfer | `flashinfer.xqa` | XQA optimized GQA/MHA paged decode attention (TRT-LLM XQA kernel) | sm90/sm100 | fp16/bf16/fp8 |
| vllm | `vllm.paged_attention` | ROCm custom PagedAttention (MFMA-based) with optional fp8 output and query_start_loc | ROCm (gfx9) | fp16/bf16, kv fp8 |
| vllm | `vllm.paged_attention_v1` | PagedAttention V1: single-kernel attention of query against paged KV cache (whole-se… | sm70+ | fp16/bf16, kv cache fp8 supported |
| vllm | `vllm.paged_attention_v2` | PagedAttention V2: split-KV partitioned attention with separate reduction (exp_sums/… | sm70+ | fp16/bf16, kv cache fp8 supported |

#### `attention.merge_state` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.merge_state_v2` | Merge two attention states (V,LSE) for split-KV attention | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.cascade.merge_state` | Merge two attention states (v_a/s_a, v_b/s_b) via log-sum-exp into merged v/s | sm80+ | fp16/bf16/fp32 |
| flashinfer | `flashinfer.cascade.merge_state_in_place` | In-place merge of another attention state into (v,s) | sm80+ | fp16/bf16/fp32 |
| flashinfer | `flashinfer.cascade.merge_states` | Merge N attention states stacked along an axis into one | sm80+ | fp16/bf16/fp32 |
| flashinfer | `flashinfer.fmha_reduction` | Cross-CTA softmax-state reduction (LSE merge) for split-KV FMHA outputs | sm100 | fp16/bf16/fp32 |
| vllm | `vllm.merge_attn_states` | Merge two partial attention results (prefix+suffix) with their LSEs into a single ou… | sm70+ | fp16/bf16/fp32 |

#### `attention.mla.decode` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.cutlass_mla_decode` | CUTLASS MLA decode attention (latent 512 + rope 64) | sm100 | fp16/bf16 |
| sgl | `sgl.dense_decode_fwd` | FlashMLA dense decode attention forward (returns sched meta) | sm90 | fp16/bf16 |
| sgl | `sgl.fwd_kvcache_mla_fp8` | FlashMLA decode attention with FP8 KV cache + descale | sm90 | fp8_e4m3 |
| flashinfer | `flashinfer.BatchDecodeMlaWithPagedKVCacheWrapper.run` | MLA batched decode attention over paged compressed-KV (ckv+kpe) cache | sm80 | fp16/bf16 |
| flashinfer | `flashinfer.cutlass_mla_paged_attention` | CUTLASS MLA paged attention (Blackwell) for compressed q_nope_pe vs ckv_kpe cache | sm100 | fp16/bf16/fp8 |
| flashinfer | `flashinfer.mla.BatchMLAPagedAttentionWrapper.run` | FlashMLA paged attention run (deepseek MLA absorb, ckv+kpe), atomic batch_mla op | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.mla.BatchMLAPagedAttentionWrapper.run_sm90` | FlashMLA paged attention run, Hopper kernel | sm90 | fp16/bf16 |
| flashinfer | `flashinfer.xqa_mla` | XQA MLA paged decode attention variant | sm90/sm100 | fp16/bf16/fp8 |
| vllm | `vllm.cutlass_mla_decode` | CUTLASS Multi-head Latent Attention (MLA) decode against compressed kv_c+k_pe paged … | sm90 | fp16/bf16 |
| vllm | `vllm.sm100_cutlass_mla_decode` | SM100 CUTLASS MLA decode with KV-splits, writes out + LSE | sm100 | fp16/bf16 |

#### `attention.mla.plan` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.get_mla_decoding_metadata` | FlashMLA tile-scheduler metadata + num_splits for decode | sm90 | int32 |
| sgl | `sgl.get_mla_decoding_metadata_dense_fp8` | FlashMLA dense-FP8 decode tile-scheduler metadata | sm90 | int32 |
| flashinfer | `flashinfer.BatchDecodeMlaWithPagedKVCacheWrapper.plan` | Plan kernel for MLA batched decode (CuTe SM80 path) | sm80 | int32 metadata |
| flashinfer | `flashinfer.mla.BatchMLAPagedAttentionWrapper.plan` | Plan kernel for FlashMLA paged attention (prefill+decode) | sm80+ | int32 metadata |
| flashinfer | `flashinfer.mla.BatchMLAPagedAttentionWrapper.plan_sm90` | Plan kernel for Hopper FlashMLA paged attention | sm90 | int32 metadata |

#### `attention.mla.sparse_decode` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fwd_kvcache_mla` | FlashMLA decode attention with paged KV cache (dense or sparse) | sm90 | fp16/bf16/fp8 |
| sgl | `sgl.sparse_decode_fwd` | FlashMLA sparse decode attention forward (topk indices) | sm90 | fp8/bf16 |
| flashinfer | `flashinfer.trtllm_batch_decode_with_kv_cache_mla` | TRT-LLM sparse MLA paged decode (DSv4 sparse) attention | sm100 | fp8/bf16 |

#### `attention.mla.workspace` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.cutlass_mla_get_workspace_size` | Compute workspace bytes for cutlass_mla_decode | sm100 | n/a |
| vllm | `vllm.sm100_cutlass_mla_get_workspace_size` | Compute workspace byte size required by sm100_cutlass_mla_decode | sm100 | n/a |

#### `attention.plan` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.get_scheduler_metadata` | Precompute FA3 tile-scheduler metadata for a batch | sm90 | n/a |
| flashinfer | `flashinfer.BatchAttention.plan` | Plan kernel for holistic two-stage persistent paged attention scheduler | sm80+ | int32 metadata |
| flashinfer | `flashinfer.fmha.blackwell_fmha_plan` | Blackwell FMHA work-partition plan kernel (tile/head/batch indices) | sm100 | int32 metadata |

#### `attention.prefill` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.extend_attention_cpu` | CPU extend/prefill attention with KV buffers | x86/aarch64 CPU | fp16/bf16/fp32 |
| sgl | `sgl.flash_attn_varlen_func` | CPU flash attention varlen forward | x86/aarch64 CPU | fp16/bf16/fp32 |
| sgl | `sgl.fwd` | FlashAttention-3 forward (varlen/paged/kvcache/GQA/rope/fp8) | sm80+/sm90a | fp16/bf16/fp8 |
| flashinfer | `flashinfer.BatchPrefillWithPagedKVCacheWrapper.paged_run` | Batched prefill attention over paged KV cache (FA2), the atomic batch_prefill_paged … | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.BatchPrefillWithPagedKVCacheWrapper.paged_run_fp8_sm90` | Batched paged FP8 prefill attention, Hopper FA3 | sm90 | fp8e4m3/fp8e5m2 |
| flashinfer | `flashinfer.BatchPrefillWithPagedKVCacheWrapper.paged_run_sm90` | Batched paged prefill attention, Hopper FA3 kernel | sm90 | fp16/bf16 |
| flashinfer | `flashinfer.BatchPrefillWithRaggedKVCacheWrapper.ragged_run` | Batched prefill attention over ragged (varlen) contiguous KV (FA2) | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.BatchPrefillWithRaggedKVCacheWrapper.ragged_run_sm90` | Batched ragged prefill attention, Hopper FA3 kernel | sm90 | fp16/bf16 |
| flashinfer | `flashinfer.cudnn_batch_prefill_with_kv_cache` | cuDNN SDPA prefill attention launcher | sm90/sm100 | fp16/bf16/fp8 |
| flashinfer | `flashinfer.fmha.cutlass_sm100.run` | CUTLASS Blackwell FMHA prefill run (varlen, work-partitioned) | sm100 | fp16/bf16/fp8 |
| flashinfer | `flashinfer.single_prefill_with_kv_cache` | Single-request prefill/append attention over contiguous Q/K/V (FA2 kernel), optional… | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.single_prefill_with_kv_cache.fp8_sm90` | Single-request FP8 prefill attention, Hopper FA3 kernel | sm90 | fp8e4m3/fp8e5m2 |
| flashinfer | `flashinfer.single_prefill_with_kv_cache_sm90` | Single-request prefill attention, Hopper FA3 kernel variant | sm90 | fp16/bf16 |
| flashinfer | `flashinfer.trtllm_batch_context_with_kv_cache` | TRT-LLM paged context (prefill) attention | sm90/sm100 | fp16/bf16/fp8/fp4 |
| flashinfer | `flashinfer.trtllm_fmha_v2.run` | TRT-LLM fmha_v2 binding run (cubin-based FMHA) | sm90/sm100 | fp16/bf16/fp8 |
| flashinfer | `flashinfer.trtllm_ragged_attention` | TRT-LLM ragged (varlen contiguous) attention | sm90/sm100 | fp16/bf16/fp8 |

#### `attention.decode.plan`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.BatchDecodeWithPagedKVCacheWrapper.plan` | Plan/scheduling kernel for batched paged decode (split-KV partition) | sm80+ | int32 metadata |

#### `attention.mla.prefill`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.dense_prefill_fwd` | CUTLASS SM100 dense MLA prefill FMHA forward | sm100 | fp16/bf16 |

#### `attention.mla.sparse_prefill`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.sparse_prefill_fwd` | FlashMLA sparse attention prefill (topk indices, d_v=512) | sm90 | bf16 |

#### `attention.pod_fused`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.batch_pod_with_kv_cache_tensor` | Batch POD attention: batched paged prefill fused with batched paged decode | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.fmha_v2.run` | TRT-LLM fmha_v2 (XQA-family) prefill/decode attention kernel | sm80+ | fp16/bf16/fp8 |
| flashinfer | `flashinfer.pod_with_kv_cache_tensor` | POD attention: single prefill fused with batched paged decode in one persistent kern… | sm80+ | fp16/bf16 |

#### `attention.prefill.plan`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan` | Plan/scheduling kernel for batched paged prefill (split-KV work partition) | sm80+ | int32 metadata |
| flashinfer | `flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan_sm90` | Plan kernel for Hopper FA3 batched prefill | sm90 | int32 metadata |

#### `attention.sparse_index_build`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.convert_vertical_slash_indexes` | Build block_count/offset + column_count/index for vertical-slash sparse attn | sm80+ | int32 |
| sgl | `sgl.convert_vertical_slash_indexes_mergehead` | Vertical-slash index builder with per-head merged index counts | sm80+ | int32 |

#### `attention.sparse_prefill`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fwd_sparse` | Sparse flash attention fwd with vertical+slash block sparsity | sm80+ | fp16/bf16 |
| sgl | `sgl.varlen_fwd_sparse` | Varlen sparse flash attention fwd (vertical+slash sparsity) | sm80+ | fp16/bf16 |

#### `attention.unified_paged`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.BatchAttention.run` | Holistic/persistent unified paged attention run (fused prefill+decode, two-stage red… | sm80+ | fp16/bf16 |

### norm

#### `norm.fused_add_rmsnorm` **[multi-lib]**
- kernel-set ABI: `ks_fused_add_rmsnorm`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fused_add_rmsnorm` | Fused residual-add then RMS norm in-place | sm80+ | fp16/bf16 |
| sgl | `sgl.fused_add_rmsnorm_cpu` | CPU fused residual-add + RMS norm in-place | x86/aarch64 CPU | fp16/bf16/fp32 |
| flashinfer | `flashinfer.norm.fused_add_rmsnorm` | Fused residual-add then RMSNorm in-place (input+=residual; normalize) | sm80+ | fp16/bf16 |
| vllm | `vllm.fused_add_rms_norm` | In-place fused residual-add + RMS normalization | sm70+ / ROCm | fp16/bf16/fp32 |

#### `norm.rmsnorm` **[multi-lib]**
- kernel-set ABI: `ks_rmsnorm`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.rmsnorm` | RMS normalization: out = (x / RMS(x)) * weight | sm80+ | fp16/bf16 |
| sgl | `sgl.rmsnorm_cpu` | CPU RMS normalization | x86/aarch64 CPU | fp16/bf16/fp32 |
| flashinfer | `flashinfer.norm.rmsnorm` | RMS normalization out = x/rms(x)*weight | sm80+ | fp16/bf16 |
| vllm | `vllm.rms_norm` | Root Mean Square (RMS) normalization of input tensor with weight | sm70+ / ROCm | fp16/bf16/fp32 |

#### `norm.fused_add_rmsnorm_quant` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.norm.fused_add_rmsnorm_quant` | Fused residual-add + RMSNorm + FP8 quant | sm80+ | out fp8 |
| vllm | `vllm.fused_add_rms_norm_static_fp8_quant` | Fused residual-add + RMS norm + static FP8 quantization | sm89+ / ROCm | in fp16/bf16, out fp8 |

#### `norm.gemma_fused_add_rmsnorm` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.gemma_fused_add_rmsnorm` | Gemma fused residual-add then RMS norm in-place | sm80+ | fp16/bf16 |
| sgl | `sgl.gemma_fused_add_rmsnorm_cpu` | CPU Gemma fused residual-add + RMS norm in-place | x86/aarch64 CPU | fp16/bf16/fp32 |
| flashinfer | `flashinfer.norm.gemma_fused_add_rmsnorm` | Gemma-style fused residual-add + RMSNorm | sm80+ | fp16/bf16 |

#### `norm.gemma_rmsnorm` **[multi-lib]**
- kernel-set ABI: `ks_gemma_rmsnorm`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.gemma3_rmsnorm_cpu` | CPU Gemma3 RMS norm variant | x86/aarch64 CPU | fp16/bf16/fp32 |
| sgl | `sgl.gemma4_rmsnorm_cpu` | CPU Gemma4 RMS norm with scale_shift/with_scale | x86/aarch64 CPU | fp16/bf16/fp32 |
| sgl | `sgl.gemma_rmsnorm` | Gemma-style RMS norm: out = (x/RMS(x))*(weight+1) | sm80+ | fp16/bf16 |
| sgl | `sgl.gemma_rmsnorm_cpu` | CPU Gemma RMS norm (weight+1) | x86/aarch64 CPU | fp16/bf16/fp32 |
| flashinfer | `flashinfer.norm.gemma_rmsnorm` | Gemma-style RMSNorm (weight uses (1+w)) | sm80+ | fp16/bf16 |

#### `norm.layernorm` **[multi-lib]**
- kernel-set ABI: `ks_layernorm`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.layernorm_cpu` | CPU layer normalization with optional bias | x86/aarch64 CPU | fp16/bf16/fp32 |
| flashinfer | `flashinfer.norm.layernorm` | Standard LayerNorm with gamma/beta | sm80+ | fp16/bf16 |

#### `norm.rmsnorm_quant` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.norm.rmsnorm_quant` | Fused RMSNorm + FP8 quantization with output scale | sm80+ | in fp16/bf16, out fp8 |
| vllm | `vllm.rms_norm_static_fp8_quant` | RMS norm fused with static (precomputed-scale) FP8 output quantization | sm89+ / ROCm | in fp16/bf16, out fp8 |

#### `norm.dit_layernorm`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.norm.fused_dit_layernorm` | DiT fused layernorm: residual add, gate, scale/shift modulation + optional fp4 scale… | sm90+ | fp16/bf16/fp8/fp4 |

#### `norm.fused_add_layernorm`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fused_add_layernorm_cpu` | CPU fused residual-add + layernorm | x86/aarch64 CPU | fp16/bf16/fp32 |

#### `norm.gated_rmsnorm`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fused_rmsnorm_gated_cpu` | CPU Qwen3-next gated RMS norm (rmsnorm then gate mul) | x86/aarch64 CPU | fp16/bf16/fp32 |

#### `norm.l2norm`
- kernel-set ABI: `ks_layernorm`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.l2norm_cpu` | CPU L2 normalization | x86/aarch64 CPU | fp16/bf16/fp32 |

#### `norm.rmsnorm_per_block_quant`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.rms_norm_per_block_quant` | Fused RMS norm + per-block (group_size) quantization, optional residual and transpos… | sm89+ | in fp16/bf16, out fp8 |

#### `norm.rmsnorm_per_token_quant`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.rms_norm_dynamic_per_token_quant` | Fused RMS norm + dynamic per-token quant (fp8/int8), optional residual add and scale… | sm89+ / ROCm | in fp16/bf16, out fp8/int8 |

#### `norm.rmsnorm_silu`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.norm.fused_rmsnorm_silu` | Fused RMSNorm + SiLU activation with per-row output scale | sm90+ | fp16/bf16, out fp8 scaled |

### act

#### `act.gelu_mul` **[multi-lib]**
- kernel-set ABI: `ks_gelu_and_mul`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.gelu_and_mul` | GELU(erf exact) gate then multiply | sm80+ | fp16/bf16 |
| sgl | `sgl.gelu_and_mul_cpu` | CPU GELU(erf)-and-mul activation | x86/aarch64 CPU | fp16/bf16/fp32 |
| flashinfer | `flashinfer.activation.gelu_and_mul` | Gated GELU (erf): out = gelu(x[...,:d]) * x[...,d:] | sm80+ | fp16/bf16 |
| vllm | `vllm.gelu_and_mul` | GeGLU with exact (none) GELU approximation: GELU(x[:d]) * x[d:] | sm70+ / ROCm | fp16/bf16/fp32 |

#### `act.gelu_tanh_mul` **[multi-lib]**
- kernel-set ABI: `ks_gelu_and_mul`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.gelu_tanh_and_mul` | GELU(tanh approx) gate then multiply | sm80+ | fp16/bf16 |
| sgl | `sgl.gelu_tanh_and_mul_cpu` | CPU GELU(tanh)-and-mul activation | x86/aarch64 CPU | fp16/bf16/fp32 |
| flashinfer | `flashinfer.activation.gelu_tanh_and_mul` | Gated GELU-tanh approx: out = gelu_tanh(x[...,:d]) * x[...,d:] | sm80+ | fp16/bf16 |
| vllm | `vllm.gelu_tanh_and_mul` | GeGLU with tanh GELU approximation: GELU_tanh(x[:d]) * x[d:] | sm70+ / ROCm | fp16/bf16/fp32 |

#### `act.silu_mul` **[multi-lib]**
- kernel-set ABI: `ks_silu_and_mul`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.silu_and_mul` | SiLU gate then multiply: out = silu(x[:d])*x[d:] | sm80+ | fp16/bf16 |
| sgl | `sgl.silu_and_mul_cpu` | CPU SiLU-and-mul activation | x86/aarch64 CPU | fp16/bf16/fp32 |
| flashinfer | `flashinfer.activation.silu_and_mul` | Gated SiLU: out = silu(x[...,:d]) * x[...,d:] | sm80+ | fp16/bf16 |
| vllm | `vllm.mul_and_silu` | Gated activation x[:d] * SiLU(x[d:]) (mul-then-silu order) | sm70+ / ROCm | fp16/bf16/fp32 |
| vllm | `vllm.silu_and_mul` | SwiGLU: SiLU(x[:d]) * x[d:] gated activation | sm70+ / ROCm | fp16/bf16/fp32 |
| vllm | `vllm.silu_and_mul_with_clamp` | SwiGLU activation with input clamping to [-limit, limit] | sm70+ / ROCm | fp16/bf16/fp32 |
| vllm | `vllm.swigluoai_and_mul` | OpenAI-style SwiGLU gated activation with alpha and clamp limit | sm70+ / ROCm | fp16/bf16/fp32 |

#### `act.gelu_quick` **[multi-lib]**
- kernel-set ABI: `ks_gelu`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.gelu_quick` | QuickGELU: y = x*sigmoid(1.702*x) (ROCm/HIP only) | ROCm/HIP | fp16/bf16 |
| vllm | `vllm.gelu_quick` | Quick GELU (x*sigmoid(1.702x)) elementwise activation | sm70+ / ROCm | fp16/bf16/fp32 |

#### `act.silu_mul_quant` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.activation.silu_and_mul_scaled_nvfp4_experts_quantize` | Fused SiLU+mul then per-expert NVFP4 quantization with per-row mask | sm100 | in fp16/bf16, out nvfp4 |
| vllm | `vllm.persistent_masked_m_silu_mul_quant` | Persistent masked grouped (per-expert) SiLU*Mul + per-group FP8 quant for MoE; count… | sm90+ | in bf16, out fp8 |
| vllm | `vllm.silu_and_mul_per_block_quant` | Fused SiLU+Mul + per-block (group_size) quantization with optional scale_ub and tran… | sm89+ | in fp16/bf16, out fp8 |
| vllm | `vllm.silu_and_mul_quant` | Fused SiLU+Mul (SwiGLU) with FP8 quantization to result using given scale | sm89+ | in fp16/bf16, out fp8 |

#### `act.fatrelu_mul`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.fatrelu_and_mul` | FATReLU gated activation: FATReLU(x[:d], threshold) * x[d:] | sm70+ / ROCm | fp16/bf16/fp32 |

#### `act.gelu_fast`
- kernel-set ABI: `ks_gelu`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.gelu_fast` | Fast approximate GELU elementwise activation | sm70+ / ROCm | fp16/bf16/fp32 |

#### `act.gelu_new`
- kernel-set ABI: `ks_gelu`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.gelu_new` | GPT-2 style new GELU elementwise activation | sm70+ / ROCm | fp16/bf16/fp32 |

### rope

#### `rope.apply` **[multi-lib]**
- kernel-set ABI: `ks_rope`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.apply_rotary_pos_emb_cpu` | CPU apply precomputed cos/sin rotary embedding | x86/aarch64 CPU | fp16/bf16/fp32 |
| sgl | `sgl.musa_batched_rotary_embedding_contiguous` | MUSA batched rotary embedding with per-row cache offsets | MUSA | fp16/bf16 |
| sgl | `sgl.musa_rotary_embedding_contiguous` | MUSA rotary embedding on contiguous query/key | MUSA | fp16/bf16 |
| sgl | `sgl.rotary_embedding` | Apply rotary position embedding to query/key in-place | sm80+ | fp16/bf16 |
| sgl | `sgl.rotary_embedding_cpu` | CPU rotary position embedding on query/key | x86/aarch64 CPU | fp16/bf16/fp32 |
| flashinfer | `flashinfer.rope.apply_rope` | Apply RoPE to q/k using indptr/offsets (ragged), out-of-place | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.rope.apply_rope_pos_ids` | Apply RoPE to q/k addressed by explicit position ids | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.rope.apply_rope_with_cos_sin_cache` | Apply RoPE to q/k using precomputed cos/sin cache + pos ids | sm80+ | fp16/bf16 |
| vllm | `vllm.rotary_embedding` | Apply GPT-NeoX or GPT-J style rotary positional embedding to query and key in-place | sm70+ / ROCm | fp16/bf16/fp32 |

#### `rope.qk_norm_rope` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fused_qk_norm_rope` | Fused QK RMSNorm + RoPE on packed QKV in-place | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.fused_qk_rmsnorm_rope` | Fused QK-RMSNorm + RoPE applied to fused QKV (with optional fp8 output quant) | sm80+ | fp16/bf16, out fp8 |
| vllm | `vllm.fused_qk_norm_rope` | Fused per-head Q/K RMS norm + RoPE applied to a packed QKV tensor | sm80+ | fp16/bf16 |

#### `rope.dsv4_k_norm_rope_cache` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.dsv4_fused_k_norm_rope_flashmla` | DeepSeek-V4 fused K RMSNorm + RoPE + FlashMLA FP8 cache store | sm90 | bf16->fp8 |
| vllm | `vllm.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert` | FlashInfer-V4 DeepSeek-V4 fused QK-norm+RoPE writing Q in-place (bf16) and KV into 5… | sm90+ | bf16 |
| vllm | `vllm.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert` | FlashInfer-V4 DeepSeek-V4 fused QK-norm+RoPE writing Q to separate FP8 tensor and KV… | sm90+ | bf16 in, fp8 out |
| vllm | `vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` | DeepSeek-V4 MLA fused: per-head Q RMSNorm + GPT-J RoPE for Q, and RoPE + UE8M0 FP8 q… | sm90+ | bf16 in, fp8 cache |

#### `rope.apply_llama31`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.rope.apply_llama31_rope` | Apply Llama-3.1 scaled RoPE to q/k (indptr/offsets) | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.rope.apply_llama31_rope_pos_ids` | Apply Llama-3.1 RoPE addressed by position ids | sm80+ | fp16/bf16 |

#### `rope.apply_mrope`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.multimodal_rotary_embedding_cpu` | CPU multimodal (M-RoPE) rotary embedding with sections | x86/aarch64 CPU | fp16/bf16/fp32 |

#### `rope.apply_pool_fused`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.rope_pool_fused` | Metal: NeoX RoPE on Q/K + scatter K/V into MLX KV pool | Apple Metal | fp16/bf16 |

#### `rope.dsv4_q_norm_rope`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.dsv4_fused_q_norm_rope` | DeepSeek-V4 fused Q RMSNorm (no weight) + RoPE | sm90 | bf16 |

#### `rope.quantize`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.rope.rope_quantize` | Apply RoPE (cos/sin cache) to q/k rope+nope parts and FP8-quantize outputs | sm80+ | in fp16/bf16, out fp8 |

#### `rope.quantize_append_kv`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.rope.rope_quantize_append_paged_kv_cache` | Fused RoPE + FP8-quantize + append into paged KV/MLA cache | sm80+ | in fp16/bf16, cache fp8 |

### gemm

#### `gemm.fp4` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.mxfp4_scaled_mm_cpu` | CPU MXFP4 scaled GEMM | x86 CPU | mxfp4->bf16 |
| flashinfer | `flashinfer.gemm.group_gemm_mxfp4_nt_groupwise` | Grouped MXFP4 NT GEMM with groupwise scales (SM100) | sm100 | mxfp4 |
| flashinfer | `flashinfer.gemm.group_gemm_mxfp4_nt_groupwise_sm120` | Grouped MXFP4 NT groupwise GEMM (SM120) | sm120 | mxfp4 |
| flashinfer | `flashinfer.gemm.group_gemm_nvfp4_nt_groupwise` | Grouped NVFP4 NT groupwise GEMM (SM120) | sm120 | nvfp4 |
| flashinfer | `flashinfer.gemm.mm_fp4` | CUTLASS NVFP4/FP4 GEMM with block scales and tactic autotuning | sm100 | nvfp4, out fp16/bf16 |
| flashinfer | `flashinfer.gemm.mm_fp4.sm103` | FP4 CUTLASS GEMM for SM103 | sm103 | nvfp4 |
| flashinfer | `flashinfer.gemm.mm_fp4.sm120` | FP4 CUTLASS GEMM for SM120 | sm120 | nvfp4 |
| vllm | `vllm.cutlass_scaled_fp4_mm` | CUTLASS NVFP4 block-scaled GEMM: out = alpha*(a@b) with per-block scales | sm100 | nvfp4 in, fp16/bf16 out |

#### `gemm.fp8` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.musa_fused_gemv` | MUSA fused GEMV (fp8/w4a16/general) with opt swiglu/rmsnorm | MUSA | fp8/int4/fp16 |
| flashinfer | `flashinfer.gemm.mm_fp8` | CUTLASS FP8 GEMM with tactic autotuning | sm90/sm100 | fp8e4m3 |
| flashinfer | `flashinfer.gemm.mm_mxfp8` | CUTLASS MXFP8 GEMM with block scales and tactic autotuning | sm100 | mxfp8 |
| flashinfer | `flashinfer.gemm.mm_mxfp8.sm120` | CUTLASS MXFP8 GEMM for SM120 | sm120 | mxfp8 |
| flashinfer | `flashinfer.gemm.trtllm_gemm` | TRT-LLM cubin GEMM runner (FP8/FP4/bf16) with tactic selection | sm90/sm100 | fp8/fp4/bf16 |
| flashinfer | `flashinfer.gemm.trtllm_low_latency_gemm` | TRT-LLM low-latency GEMM (small-batch decode GEMM) | sm90/sm100 | fp8/bf16 |
| vllm | `vllm.cutlass_w4a8_mm` | CUTLASS W4A8 mixed-precision GEMM (int4 weight, fp8 activation) with group/channel/t… | sm90 | int4 w / fp8 a |
| vllm | `vllm.machete_mm` | Machete mixed-precision GEMM for Hopper: A@B with quantized B (group/channel/token s… | sm90 | fp16/bf16/fp8 A, int4/int8 B |
| vllm | `vllm.marlin_gemm` | Marlin optimized quantized GEMM supporting GPTQ/AWQ/FP8/NVFP4/MXFP4 with optional bi… | sm80+ | int4/int8/fp8/fp4 w, fp16/bf16 a |

#### `gemm.router` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.dsv3_router_gemm` | DeepSeek-V3 MoE router GEMM (bf16 or float out) | sm90 | bf16/fp32 out |
| flashinfer | `flashinfer.gemm.dsv3_router_gemm` | DeepSeek-V3 router GEMM (M<=16, K=7168, N=256, fp32) | sm90/sm100 | fp32 |
| flashinfer | `flashinfer.gemm.glm_dsa_router_gemm` | GLM DSA router GEMM (M<=16, K=6144, N=256, fp32) | sm90/sm100 | fp32 |
| flashinfer | `flashinfer.gemm.ml3_router_gemm` | Router GEMM variant (M<=16, K=7168, N=128, bf16) | sm90/sm100 | bf16 |
| vllm | `vllm.fp32_router_gemm` | BF16/FP32 x FP32 -> FP32 router GEMM for H=3072, E=256, M<=32 | sm90+ | bf16/fp32 in, fp32 out |

#### `gemm.batched` **[multi-lib]**
- kernel-set ABI: `ks_gemm_batched`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.bmm_cpu` | CPU batched matmul with optional scale | x86/aarch64 CPU | bf16/fp16 |
| flashinfer | `flashinfer.gemm.bmm_bf16` | CUTLASS BF16 (batched) GEMM with tactic autotuning | sm90/sm100 | bf16 |

#### `gemm.dense` **[multi-lib]**
- kernel-set ABI: `ks_gemm`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.convert_weight_packed` | CPU weight pre-pack to VNNI/blocked layout | x86/aarch64 CPU | int8/fp8/bf16 |
| sgl | `sgl.weight_packed_linear` | CPU packed-weight linear (matmul + opt bias) | x86/aarch64 CPU | bf16/fp16 |
| flashinfer | `flashinfer.gemm.mm_bf16` | BF16 matrix multiply via cuBLASLt (get_algos + run_with_algo) | sm80+ | bf16 |
| flashinfer | `flashinfer.gemm.tgv_gemm_sm100` | TGV (tile-genvolta) GEMM with optional bias, tactic autotuning (Blackwell) | sm100 | bf16/fp16 |
| flashinfer | `flashinfer.gemm.tinygemm2` | TinyGEMM2 small-M GEMM with bias (mm_M1_16_* shapes) | sm90/sm100 | bf16/fp16 |
| flashinfer | `flashinfer.gemm.tinygemm2_nobias` | TinyGEMM2 small-M GEMM without bias | sm90/sm100 | bf16/fp16 |

#### `gemm.dsv3_fused_a` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.dsv3_fused_a_gemm` | DeepSeek-V3 fused 'a' projection GEMM | sm90 | bf16 |
| vllm | `vllm.dsv3_fused_a_gemm` | DeepSeek-V3 fused A (q/kv down-projection) GEMM, bf16 only, 1-16 tokens | sm90+ | bf16 |

#### `gemm.fp8_batched` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.bmm_fp8` | Batched FP8 matmul via cuBLAS with A/B scales | sm89/sm90 | fp8_e4m3->fp16/bf16 |
| flashinfer | `flashinfer.gemm.bmm_fp8` | Batched FP8 GEMM via cuBLASLt with per-tensor scales | sm89+ | fp8e4m3/e5m2, out fp16/bf16 |
| flashinfer | `flashinfer.gemm.bmm_fp8.get_algos` | Enumerate cuBLASLt algos for an FP8 BMM problem | sm89+ | fp8 |
| flashinfer | `flashinfer.gemm.bmm_fp8.run_with_algo` | Run FP8 BMM with a chosen cuBLASLt algo index | sm89+ | fp8 |

#### `gemm.fp8_blockwise` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fp8_blockwise_scaled_mm` | FP8 GEMM with block-wise scaling factors | sm90 | fp8_e4m3->bf16 |
| flashinfer | `flashinfer.gemm.fp8_blockscale_gemm_sm90` | FP8 block-scaled GEMM (DeepSeek-style 128x128 block scales) for Hopper | sm90 | fp8e4m3 |
| flashinfer | `flashinfer.gemm.gemm_fp8_nt_groupwise` | FP8 NT GEMM with groupwise/blockwise scales (SM100) | sm100 | fp8e4m3, out fp16/bf16 |
| flashinfer | `flashinfer.gemm.gemm_fp8_nt_groupwise_sm120` | FP8 NT groupwise-scaled GEMM (SM120) | sm120 | fp8e4m3 |
| flashinfer | `flashinfer.gemm.group_gemm_fp8_nt_groupwise` | Grouped FP8 NT GEMM with groupwise scales (SM100) | sm100 | fp8e4m3 |
| flashinfer | `flashinfer.gemm.group_gemm_fp8_nt_groupwise_sm120` | Grouped FP8 NT groupwise GEMM (SM120) | sm120 | fp8e4m3 |

#### `gemm.gguf` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.ggml_mul_mat_a8` | GGML quantized matrix-matrix multiply (a8 activations) | sm80+ | gguf/int8 |
| vllm | `vllm.ggml_mul_mat_a8` | GGML mmq matrix-matrix kernel (a8 activation) over quantized weight | sm60+ | gguf w, int8 a |

#### `gemm.gguf_gemv` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.ggml_mul_mat_vec_a8` | GGML quantized matrix-vector multiply (a8 activations) | sm80+ | gguf/int8 |
| vllm | `vllm.ggml_mul_mat_vec_a8` | GGML mmvq matrix-vector kernel (a8 activation) over quantized weight | sm60+ | gguf w, int8 a |

#### `gemm.w4a16` **[multi-lib]**
- kernel-set ABI: `ks_gemm_w4a16`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.gptq_gemm` | GPTQ quantized weight GEMM (2/3/4/8-bit) with optional shuffle | sm80+ | int2/3/4/8->fp16 |
| sgl | `sgl.int4_scaled_mm_cpu` | CPU INT4 weight scaled GEMM with zeros/scales | x86 CPU | int4->bf16 |
| vllm | `vllm.awq_gemm` | AWQ quantized GEMM (int4 weight, fp16 activation) with split-k | sm75+ | int4 w, fp16 a |
| vllm | `vllm.gptq_gemm` | GPTQ quantized GEMM via exllama/exllamav2 kernels (int4/int8 weight) | sm60+ | int4/int8 w, fp16 a |
| vllm | `vllm.gptq_gemm_rdna3` | W4A16 GPTQ GEMM for AMD RDNA3 (gfx1100) | ROCm gfx1100 | int4 w, fp16 a |
| vllm | `vllm.gptq_gemm_rdna3_wmma` | W4A16 GPTQ GEMM (WMMA) for AMD RDNA3 (gfx1100) | ROCm gfx1100 | int4 w, fp16 a |

#### `gemm.w8a8` **[multi-lib]**
- kernel-set ABI: `ks_gemm_w8a8`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.int8_scaled_mm` | INT8 x INT8 -> out_dtype GEMM with per-row/col scales + opt bias | sm80+ | int8->fp16/bf16 |
| sgl | `sgl.int8_scaled_mm_cpu` | CPU INT8 scaled GEMM with per-row/col scales | x86/aarch64 CPU | int8->fp16/bf16 |
| vllm | `vllm.cutlass_scaled_mm` | CUTLASS w8a8 scaled GEMM (symmetric per-tensor or per-row/col), optional bias: out =… | sm80+ | int8/fp8 in, fp16/bf16 out |

#### `gemm.fp8_scaled`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fp8_scaled_mm` | FP8 x FP8 -> out_dtype GEMM with per-tensor scales + opt bias | sm89/sm90 | fp8_e4m3->fp16/bf16 |
| sgl | `sgl.fp8_scaled_mm_cpu` | CPU FP8 scaled GEMM with block_size scales | x86 CPU | fp8->fp16/bf16 |

#### `gemm.fused_linear_sigmoid`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fused_linear_sigmoid_mul` | CPU fused linear -> sigmoid -> elementwise mul with post matrix | x86/aarch64 CPU | bf16/fp16 |

#### `gemm.grouped`
- kernel-set ABI: `ks_moe_grouped_gemm`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.gemm.SegmentGEMMWrapper.cutlass_segment_gemm` | CUTLASS grouped/segment GEMM (variable-size problems) for SM80 | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.gemm.SegmentGEMMWrapper.cutlass_segment_gemm_sm90` | CUTLASS grouped/segment GEMM for Hopper | sm90 | fp16/bf16 |

#### `gemm.int8_fused_quant`
- kernel-set ABI: `ks_gemm_w8a8`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.int8_scaled_mm_with_quant` | CPU fused per-token int8 quant + scaled GEMM | x86/aarch64 CPU | fp16/bf16->int8->out |

#### `gemm.qkv_proj_rope_fused`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.qkv_proj_with_rope` | CPU fused QKV projection + RoPE with weight absorption (MLA) | x86 CPU | bf16/int8/fp8 |
| sgl | `sgl.qkv_proj_with_rope_fused_weight` | CPU fused QKV proj + RoPE with fused qkv_a_proj weight (MLA) | x86 CPU | bf16/int8/fp8 |

#### `gemm.skinny`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.LLMM1` | ROCm custom matrix-vector multiplication GEMM | ROCm | fp16/bf16 |
| vllm | `vllm.wvSplitK` | ROCm skinny matrix-matrix multiplication GEMM with optional bias | ROCm | fp16/bf16 |
| vllm | `vllm.wvSplitKQ` | ROCm wvSplitK skinny GEMM for fp8 with per-tensor scales | ROCm | fp8 in, fp16/bf16 out |
| vllm | `vllm.wvSplitKrc` | ROCm skinny matrix-matrix multiplication GEMM (rc variant) with optional bias | ROCm | fp16/bf16 |

#### `gemm.w4a16_machete`
- kernel-set ABI: `ks_gemm_w4a16`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.machete_supported_schedules` | List supported Machete kernel schedules for given type combination | sm90 | n/a |

#### `gemm.w4a8`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.qserve_w4a8_per_chn_gemm` | QServe W4A8 per-channel GEMM (int4 weight, int8 act) | sm80+ | int4/int8->fp16 |
| sgl | `sgl.qserve_w4a8_per_group_gemm` | QServe W4A8 per-group GEMM with zeros + i8 scales | sm80+ | int4/int8->fp16 |

#### `gemm.w8a16`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.allspark_w8a16_gemm` | AllSpark Ampere W8A16 fused GEMM (int8 weight, fp16/bf16 activation) | sm80 | int8 w, fp16/bf16 a |

#### `gemm.w8a8_azp`
- kernel-set ABI: `ks_gemm_w8a8`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.cutlass_scaled_mm_azp` | CUTLASS w8a8 scaled GEMM with asymmetric (zero-point) quantization correction | sm80+ | int8 in, fp16/bf16 out |

### moe

#### `moe.gate_grouped` **[multi-lib]**
- kernel-set ABI: `ks_moe_gate_sigmoid_group_topk`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.biased_grouped_topk_cpu` | CPU biased grouped top-k MoE gate (with correction bias) | x86/aarch64 CPU | fp32 |
| sgl | `sgl.grouped_topk_cpu` | CPU grouped top-k MoE gate | x86/aarch64 CPU | fp32 |
| sgl | `sgl.kimi_k2_moe_fused_gate` | Kimi-K2 single-group fused MoE gate (no grouped logic) | sm80+ | fp32 |
| sgl | `sgl.moe_fused_gate` | Hierarchical grouped top-k expert gate (group then in-group topk) | sm80+ | fp32 |
| flashinfer | `flashinfer.fused_moe.RoutingMethodType.NoAuxTc` | DeepSeek-V3 no-aux-loss group-limited expert routing (sigmoid+group topk) | sm90/sm100 | fp32/bf16 |
| vllm | `vllm.grouped_topk` | Grouped top-k expert routing (DeepSeek-style group-limited routing) returning weight… | sm70+ | fp16/bf16/fp32 |

#### `moe.align_block` **[multi-lib]**
- kernel-set ABI: `ks_moe_compute_permutation`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.moe_align_block_size` | Sort tokens by expert and pad to block_size for grouped MoE GEMM | sm80+ | int32 |
| vllm | `vllm.batched_moe_align_block_size` | moe_align_block_size for batched-expert format using per-expert token counts | sm70+ | int32 indices |
| vllm | `vllm.moe_align_block_size` | Align/sort tokens per expert so each expert block count is divisible by block_size f… | sm70+ | int32 indices |

#### `moe.fused_full` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fused_experts_cpu` | CPU fused MoE experts (grouped GEMM + activation + combine) | x86/aarch64 CPU | bf16/int8/fp8/mxfp4 |
| sgl | `sgl.musa_fused_moe_gemv` | MUSA fused MoE GEMV with routed weights + opt swiglu | MUSA | fp8/int4/fp16 |
| sgl | `sgl.shared_expert_cpu` | CPU shared-expert MLP with routed scaling/combine | x86 CPU | bf16/int8/fp8 |
| flashinfer | `flashinfer.fused_moe.bgmv_moe_expand` | BGMV MoE/LoRA expand GEMM (B-projection, accumulate) | sm80+ | fp16/bf16/fp32 |
| flashinfer | `flashinfer.fused_moe.bgmv_moe_shrink` | BGMV (batched gather matrix-vector) MoE/LoRA shrink GEMM (A-projection) | sm80+ | fp16/bf16/fp32 |
| flashinfer | `flashinfer.fused_moe.cutlass_fused_moe` | CUTLASS fused MoE: gather + grouped GEMM1 + gated act + grouped GEMM2 + scatter (Fus… | sm90/sm100 | fp16/bf16/fp8/fp4/mxfp4 |
| flashinfer | `flashinfer.fused_moe.interleave_moe_weights_for_sm90_mixed_gemm` | Interleave MoE weights for SM90 mixed-precision grouped GEMM layout | sm90 | fp8/int4 |
| flashinfer | `flashinfer.fused_moe.moe_activation` | MoE intermediate gated activation (SwiGLU) between GEMM1 and GEMM2 | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.fused_moe.moe_output_memset` | Zero/scatter-init MoE output buffer (in-place and out-of-place variants) | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.fused_moe.trtllm_bf16_moe` | TRT-LLM BF16 fused MoE | sm90/sm100 | bf16 |
| flashinfer | `flashinfer.fused_moe.trtllm_fp4_block_scale_moe` | TRT-LLM FP4 (NVFP4) block-scale fused MoE | sm100 | nvfp4 |
| flashinfer | `flashinfer.fused_moe.trtllm_fp8_block_scale_moe` | TRT-LLM FP8 block-scale fused MoE | sm90/sm100 | fp8e4m3 |
| flashinfer | `flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe` | TRT-LLM FP8 per-tensor-scale fused MoE | sm90/sm100 | fp8e4m3 |
| flashinfer | `flashinfer.fused_moe.trtllm_get_valid_moe_configs` | Enumerate valid TRT-LLM fused-MoE tactic configs | sm90/sm100 | meta |
| flashinfer | `flashinfer.fused_moe.trtllm_mxint4_block_scale_moe` | TRT-LLM MXINT4 block-scale fused MoE | sm100 | mxint4 |

#### `moe.gate_sigmoid` **[multi-lib]**
- kernel-set ABI: `ks_moe_gate_sigmoid_group_topk`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.topk_sigmoid` | MoE top-k sigmoid routing with optional correction bias | sm80+ | fp32 |
| sgl | `sgl.topk_sigmoid_cpu` | CPU MoE top-k sigmoid routing | x86/aarch64 CPU | fp32 |
| vllm | `vllm.topk_sigmoid` | Apply top-k sigmoid to gating output for MoE routing | sm70+ | fp16/bf16/fp32 |

#### `moe.gate_softmax` **[multi-lib]**
- kernel-set ABI: `ks_moe_gate_softmax_topk`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.topk_softmax` | MoE top-k softmax routing with optional softcap + correction bias | sm80+ | fp32 |
| sgl | `sgl.topk_softmax_cpu` | CPU MoE top-k softmax routing | x86/aarch64 CPU | fp32 |
| vllm | `vllm.topk_softmax` | Apply top-k softmax to gating output to produce expert weights/indices for MoE routi… | sm70+ | fp16/bf16/fp32 |

#### `moe.gguf_gemv` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.ggml_moe_a8_vec` | GGML quantized MoE matrix-vector multiply (a8) | sm80+ | gguf/int8 |
| vllm | `vllm.ggml_moe_a8_vec` | GGML MoE matrix-vector kernel (a8) for decode using topk_ids | sm60+ | gguf w, int8 a |

#### `moe.gguf_grouped_gemm` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.ggml_moe_a8` | GGML quantized MoE matmul (a8) with sorted token/expert ids | sm80+ | gguf/int8 |
| vllm | `vllm.ggml_moe_a8` | GGML MoE grouped GEMM (a8) using sorted token ids and expert ids | sm60+ | gguf w, int8 a |

#### `moe.grouped_gemm_fp8` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.es_fp8_blockwise_scaled_grouped_mm` | Expert-specialization FP8 block-scaled grouped GEMM | sm90 | fp8_e4m3->bf16 |
| sgl | `sgl.es_sm100_mxfp8_blockscaled_grouped_mm` | SM100 MXFP8 block-scaled grouped GEMM (expert specialization) | sm100 | mxfp8->bf16 |
| sgl | `sgl.fp8_blockwise_scaled_grouped_mm` | Grouped (per-expert) FP8 block-scaled GEMM for MoE | sm90 | fp8_e4m3->bf16 |
| vllm | `vllm.cutlass_mxfp8_grouped_mm` | Expert-specialization MXFP8 blockscaled grouped GEMM for MoE | sm100+ | mxfp8 in, bf16 out |

#### `moe.grouped_gemm_w4a8` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.cutlass_w4a8_moe_mm` | CUTLASS grouped GEMM int4 weights x fp8 activations for MoE | sm90 | int4/fp8->bf16 |
| vllm | `vllm.cutlass_w4a8_moe_mm` | CUTLASS W4A8 grouped (per-expert) GEMM for MoE | sm90 | int4 w / fp8 a |

#### `moe.permute` **[multi-lib]**
- kernel-set ABI: `ks_moe_permute`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.fused_moe.moe_permute` | Permute (gather) tokens into expert-contiguous layout (fp16/bf16/fp8/fp4 variants) | sm80+ | fp16/bf16/fp8/fp4 |
| vllm | `vllm.moe_permute` | Permute/scatter MoE input rows by expert assignment, producing permuted input and ex… | sm80+ | fp16/bf16 |
| vllm | `vllm.moe_permute_with_scratch` | moe_permute variant using explicit scratch buffers for sorting | sm80+ | fp16/bf16 |

#### `moe.prepare_input` **[multi-lib]**
- kernel-set ABI: `ks_moe_compute_permutation`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.get_cutlass_w4a8_moe_mm_data` | Build expert offsets/problem-sizes/permutations for W4A8 MoE MM | sm90 | int32 |
| sgl | `sgl.prepare_moe_input` | Compute expert offsets, problem sizes and permutations from topk_ids | sm80+ | int32 |
| vllm | `vllm.get_cutlass_batched_moe_mm_data` | Compute expert_offsets and problem sizes for batched-expert-format CUTLASS grouped M… | sm90+ | int32 indices |
| vllm | `vllm.get_cutlass_moe_mm_data` | Compute expert_offsets, problem_sizes and input/output permutations for CUTLASS grou… | sm90+ | int32 indices |
| vllm | `vllm.get_cutlass_moe_mm_problem_sizes_from_expert_offsets` | Compute per-expert problem sizes from expert_first_token_offset (from moe_permute) | sm90+ | int32 indices |

#### `moe.shuffle_rows` **[multi-lib]**
- kernel-set ABI: `ks_moe_permute`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.shuffle_rows` | Gather/scatter rows by dst->src map (MoE permutation) | sm80+ | any |
| vllm | `vllm.shuffle_rows` | Row shuffle/gather for MoE using a dst-to-src index map | sm70+ | fp16/bf16 |

#### `moe.sum_reduce` **[multi-lib]**
- kernel-set ABI: `ks_moe_unpermute`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.moe_sum` | Sum reduction of top-k expert outputs into output | sm80+ | fp16/bf16 |
| sgl | `sgl.moe_sum_reduce` | Sum-reduce expert outputs with routed scaling factor | sm80+ | fp16/bf16 |
| vllm | `vllm.moe_sum` | Sum per-expert partial MoE results into a single output tensor | sm70+ | fp16/bf16/fp32 |

#### `moe.unpermute_combine` **[multi-lib]**
- kernel-set ABI: `ks_moe_unpermute`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.fused_moe.moe_unpermute` | Unpermute + weighted-combine expert outputs back to token order | sm80+ | fp16/bf16 |
| vllm | `vllm.moe_permute_unpermute_supported` | Query whether moe_permute/unpermute kernels are supported on current build | any | n/a |
| vllm | `vllm.moe_unpermute` | Unpermute/gather MoE expert outputs back to token order with topk weight reduction | sm80+ | fp16/bf16 |

#### `moe.align_block_lora`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.moe_lora_align_block_size` | MoE+LoRA aware token alignment producing sorted ids, expert ids, adapter/lora id maps | sm70+ | int32 indices |

#### `moe.gate_softplus`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.topk_softplus_sqrt` | Apply top-k softplus-sqrt routing with routed scaling factor for MoE | sm70+ | fp16/bf16/fp32 |

#### `moe.grouped_gemm`
- kernel-set ABI: `ks_moe_grouped_gemm`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.cutlass_moe_mm` | CUTLASS w8a8 grouped (per-expert) GEMM for fused MoE | sm90+ | int8/fp8 in, fp16/bf16 out |

#### `moe.grouped_gemm_fp4`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.cutlass_fp4_group_mm` | CUTLASS NVFP4 block-scaled grouped GEMM for MoE | sm100 | nvfp4 in, fp16/bf16 out |
| vllm | `vllm.cutlass_mxfp4_group_mm` | CUTLASS MXFP4 x MXFP4 block-scaled grouped GEMM for MoE | sm100 | mxfp4 in, fp16/bf16 out |

#### `moe.grouped_gemm_w4a16`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.moe_wna16_gemm` | MoE WNA16 (weight-only int4/int8, fp16 act) grouped GEMM using sorted token/expert i… | sm80+ | int4/int8 w, fp16 a |

#### `moe.grouped_gemm_w4a16_marlin`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.moe_wna16_marlin_gemm` | Marlin-based MoE WNA16 grouped GEMM with topk weight fusion and full Marlin tuning p… | sm80+ | int4/int8/fp8 w, fp16/bf16 a |

#### `moe.misc`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.dsv3_router_gemm` | DeepSeek-V3 optimized router GEMM (gating projection) for SM90+ | sm90+ | bf16/fp16 |

#### `moe.permute_meta`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.moe_permute_sort_workspace_size` | Compute scratch workspace size for moe_permute sorting | any | n/a |

#### `moe.shuffle_mul_sum`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.apply_shuffle_mul_sum` | Shuffle rows by permutation, multiply by factors, then sum | sm80+ | fp16/bf16 |

#### `moe.sort`
- kernel-set ABI: `ks_moe_compute_permutation`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.fused_moe.moe_sort` | Sort/argsort tokens by selected expert id for grouped MoE GEMM | sm80+ | int32 |

### quant

#### `quant.awq_dequant` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.awq_dequantize` | AWQ int4 weight dequantize to fp16 | sm80+ | int4->fp16 |
| vllm | `vllm.awq_dequantize` | Dequantize AWQ int4 weights back to fp16 | sm75+ | int4 -> fp16 |

#### `quant.fp8` **[multi-lib]**
- kernel-set ABI: `ks_quantize_fp8`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.dsv4_fused_q_indexer_rope_hadamard_quant` | DeepSeek-V4 Q indexer: RoPE + Hadamard transform + FP8 quant | sm90 | bf16->fp8_e4m3 |
| vllm | `vllm.cutlass_pack_scale_fp8` | Pack FP8 scales into the layout expected by CUTLASS W4A8 kernels | sm90 | fp8 scales |
| vllm | `vllm.marlin_int4_fp8_preprocess` | Preprocess W-int4 / A-fp8 weight (and optional zeros) for Marlin kernel | sm89+ | int4 w / fp8 a |

#### `quant.fp8_per_token` **[multi-lib]**
- kernel-set ABI: `ks_quantize_fp8`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.sgl_per_token_quant_fp8` | Per-token (whole-row) FP8 quantization with scales | sm89/sm90 | fp16/bf16->fp8_e4m3 |
| vllm | `vllm.dynamic_per_token_scaled_fp8_quant` | Dynamic per-token FP8 quantization with optional scale upper-bound | sm89+ / ROCm | in fp16/bf16/fp32, out fp8 |

#### `quant.fp8_per_token_group` **[multi-lib]**
- kernel-set ABI: `ks_quantize_fp8`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.sgl_per_token_group_quant_8bit` | Per-token group quantize activations to fp8/int8 with scales | sm80+ | fp16/bf16->fp8/int8 |
| sgl | `sgl.sgl_per_token_group_quant_8bit_v2` | v2 per-token group quant with optional fused silu_and_mul + masked_m | sm90 | fp16/bf16->fp8/int8 |
| vllm | `vllm.per_token_group_fp8_quant` | Per-token-group FP8 quantization producing quantized tensor and group scales (fusabl… | sm89+ | in fp16/bf16, out fp8 |
| vllm | `vllm.per_token_group_fp8_quant_packed` | Per-token-group 8-bit quant producing UE8M0-packed, TMA-aligned scales for DeepGEMM | sm90+ | in fp16/bf16, out fp8 |
| vllm | `vllm.per_token_group_quant_int8` | Per-token-group INT8 quantization producing quantized tensor and group scales | sm70+ | in fp16/bf16, out int8 |

#### `quant.gguf_dequant` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.ggml_dequantize` | GGUF/GGML dequantize quantized weight to dtype | sm80+ | gguf->fp16/bf16 |
| vllm | `vllm.ggml_dequantize` | Dequantize a GGML/GGUF quantized weight to a float dtype | sm60+ | gguf -> fp16/bf16/fp32 |

#### `quant.gguf_meta` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.ggml_moe_get_block_size` | Return GGML quant block size for a given type | sm80+ | n/a |
| vllm | `vllm.ggml_moe_get_block_size` | Return GGML block size for a given quantization type | any | n/a |

#### `quant.gptq_shuffle` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.gptq_shuffle` | GPTQ weight permutation/shuffle in-place for fast GEMM | sm80+ | intN packed |
| vllm | `vllm.gptq_shuffle` | Post-process/shuffle GPTQ quantized weights in place | sm60+ | int4/int8 |

#### `quant.mxfp8` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.es_sm100_mxfp8_blockscaled_grouped_quant` | SM100 grouped MXFP8 block-scaled quantization per expert | sm100 | bf16->mxfp8 |
| vllm | `vllm.mxfp8_experts_quant` | Expert-specialization MXFP8 blockscaled grouped quantization using problem/expert/bl… | sm100+ | in bf16, out mxfp8 |

#### `quant.weight_prepack` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.convert_weight_packed_scale_zp` | CPU int4 weight prepack with packed scales+zeros (awq/gptq) | x86 CPU | int4 |
| vllm | `vllm.machete_prepack_B` | Prepack/reorder weight B into Machete kernel layout | sm90 | int4/int8 |

#### `quant.capability_query`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.cutlass_group_gemm_supported` | Query whether CUTLASS grouped GEMM is supported for a given device capability | any | n/a |
| vllm | `vllm.cutlass_scaled_mm_supports_block_fp8` | Query whether CUTLASS scaled_mm supports block-fp8 quantization (DeepSeekV3) | any | n/a |
| vllm | `vllm.cutlass_scaled_mm_supports_fp4` | Query whether cutlass_scaled_mm_fp4 is supported for a given device capability | any | n/a |
| vllm | `vllm.cutlass_scaled_mm_supports_fp8` | Query whether CUTLASS scaled_mm fp8 path is supported for a given device capability | any | n/a |

#### `quant.fp4`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.scaled_fp4_quant` | Quantize to NVFP4 (e2m1) block-scaled tensor, returning packed data and block scales | sm100 | in fp16/bf16, out nvfp4 |
| vllm | `vllm.scaled_fp4_quant.out` | Out-variant of scaled_fp4_quant writing into preallocated output and output_scale | sm100 | in fp16/bf16, out nvfp4 |
| vllm | `vllm.silu_and_mul_nvfp4_quant` | Fused SiLU+Mul + NVFP4 block quantization (single tensor) | sm100 | in fp16/bf16, out nvfp4 |

#### `quant.fp4_dequant`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.quantization.nvfp4_kv_dequantize` | Dequantize NVFP4 KV cache back to fp16/bf16 | sm100 | in nvfp4, out fp16/bf16 |

#### `quant.fp4_experts`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.mxfp4_experts_quant` | Per-expert MXFP4 quantization (32-element blocks, E8M0 scale factors) | sm100 | in bf16, out mxfp4 |
| vllm | `vllm.scaled_fp4_experts_quant` | Per-expert NVFP4 block quantization using expert offset arrays | sm100 | in fp16/bf16, out nvfp4 |
| vllm | `vllm.silu_and_mul_mxfp4_experts_quant` | Fused SiLU+Mul + per-expert MXFP4 quantization for MoE | sm100 | in bf16, out mxfp4 |
| vllm | `vllm.silu_and_mul_scaled_fp4_experts_quant` | Fused SiLU+Mul + per-expert NVFP4 block quantization for MoE | sm100 | in fp16/bf16, out nvfp4 |

#### `quant.fp4_kv`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.quantization.nvfp4_kv_quantize` | Quantize KV cache tensor to NVFP4 with block scale factors | sm100 | in fp16/bf16, out nvfp4 |

#### `quant.fp8_dynamic`
- kernel-set ABI: `ks_quantize_fp8`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.dynamic_scaled_fp8_quant` | Dynamic per-tensor FP8 quantization computing scale on the fly | sm89+ / ROCm | in fp16/bf16/fp32, out fp8 |

#### `quant.fp8_static`
- kernel-set ABI: `ks_quantize_fp8`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.static_scaled_fp8_quant` | FP8 quantization with given static scale; per-tensor/channel/token/2D-group via grou… | sm89+ / ROCm | in fp16/bf16/fp32, out fp8 |

#### `quant.int4_reorder`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.cutlass_encode_and_reorder_int4b` | Encode and reorder int4 weight matrix into CUTLASS W4A8 layout | sm90 | int4 |
| vllm | `vllm.cutlass_encode_and_reorder_int4b_grouped` | Encode and reorder grouped int4 weight tensors into CUTLASS W4A8 grouped layout | sm90 | int4 |

#### `quant.int8_dynamic`
- kernel-set ABI: `ks_quantize_int8`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.dynamic_scaled_int8_quant` | INT8 quantization computing scale dynamically, optional asymmetric zero-point | sm70+ / ROCm | in fp16/bf16/fp32, out int8 |

#### `quant.int8_per_token`
- kernel-set ABI: `ks_quantize_int8`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.per_token_quant_int8_cpu` | CPU per-token int8 quantization with scales | x86/aarch64 CPU | fp32->int8 |

#### `quant.int8_static`
- kernel-set ABI: `ks_quantize_int8`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.static_scaled_int8_quant` | INT8 quantization with given (static) per-tensor/channel scale, optional asymmetric … | sm70+ / ROCm | in fp16/bf16/fp32, out int8 |

#### `quant.marlin_repack`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.awq_marlin_repack` | Repack AWQ-format quantized weights into Marlin layout | sm80+ | int4/int8 |
| vllm | `vllm.gptq_marlin_repack` | Repack GPTQ-format quantized weights into Marlin layout | sm80+ | int4/int8 |

#### `quant.packbits`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.quantization.packbits` | Pack a boolean tensor into bits (big/little bitorder) | sm80+ | bool->uint8 |
| flashinfer | `flashinfer.quantization.segment_packbits` | Segment-wise packbits using input/output indptr | sm80+ | bool->uint8 |

#### `quant.permute_cols`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.permute_cols` | Permute columns of matrix A by perm (weight reordering helper) | sm80+ | any |

#### `quant.scale_pack`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.convert_scale_packed` | CPU scale pre-pack for mxfp4 | x86 CPU | mxfp4 scales |

#### `quant.weight_reorder`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.rearrange_kn_weight_as_n32k16_order` | Reorder weight/scales/zeros into N32K16 layout for AllSpark W8A16 kernel | sm80 | int8 |

### sampling

#### `sampling.min_p` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.min_p_sampling_from_probs` | Min-p sampling from probability distribution (MUSA) | MUSA | fp32 |
| flashinfer | `flashinfer.sampling.min_p_sampling_from_probs` | Min-p rejection sampling from probabilities | sm80+ | fp32 |

#### `sampling.topk_renorm` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.top_k_renorm_probs` | Renormalize probabilities keeping only top-k entries | sm80+ | fp32 |
| flashinfer | `flashinfer.sampling.top_k_renorm_probs` | Renormalize probabilities after top-k masking | sm80+ | fp32 |

#### `sampling.topk_topp` **[multi-lib]**
- kernel-set ABI: `ks_sample`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.musa_top_k_top_p_sampling_from_probs` | Joint top-k+top-p sampling from probabilities (MUSA) | MUSA | fp32 |
| flashinfer | `flashinfer.sampling.top_k_top_p_sampling_from_probs` | Joint top-k + top-p rejection sampling from probabilities | sm80+ | fp32 |

#### `sampling.topp` **[multi-lib]**
- kernel-set ABI: `ks_sample`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.top_p_sampling_from_probs` | Top-p (nucleus) sampling from probabilities (MUSA) | MUSA | fp32 |
| flashinfer | `flashinfer.sampling.top_p_sampling_from_probs` | Nucleus (top-p) rejection sampling from probabilities | sm80+ | fp32 |

#### `sampling.topp_renorm` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.top_p_renorm_probs` | Renormalize probabilities keeping top-p nucleus mass | sm80+ | fp32 |
| flashinfer | `flashinfer.sampling.top_p_renorm_probs` | Renormalize probabilities after top-p masking | sm80+ | fp32 |

#### `sampling.from_logits`
- kernel-set ABI: `ks_sample`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.sampling.sampling_from_logits` | Gumbel-style sampling directly from logits | sm80+ | fp32/fp16 |

#### `sampling.from_probs`
- kernel-set ABI: `ks_sample`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.sampling.sampling_from_probs` | Single-pass categorical sampling from a probability distribution (Philox RNG) | sm80+ | fp32 |

#### `sampling.grammar_mask`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.apply_token_bitmask_inplace_cuda` | Apply grammar token bitmask to logits in-place (-inf masked) | sm80+ | fp16/bf16/fp32 |

#### `sampling.repetition_penalty`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.apply_repetition_penalties_` | Apply repetition penalties to logits in-place using prompt/output token masks | sm70+ / ROCm | fp32 |

#### `sampling.softmax`
- kernel-set ABI: `ks_softmax`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.sampling.softmax` | Online softmax over logits with optional per-row temperature | sm80+ | fp32/fp16 |

#### `sampling.topk`
- kernel-set ABI: `ks_sample`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.sampling.top_k_sampling_from_probs` | Top-k rejection sampling from probabilities | sm80+ | fp32 |

#### `sampling.topk_mask`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.sampling.top_k_mask_logits` | Mask logits to keep only top-k (set rest to -inf) | sm80+ | fp32/fp16 |

### spec

#### `spec.verify_sampling` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.tree_speculative_sampling_target_only` | Tree speculative decoding sampling, target-distribution verification | sm80+ | fp32 probs |
| flashinfer | `flashinfer.sampling.chain_speculative_sampling` | Verify draft tokens against target probs and emit accepted/bonus tokens | sm80+ | fp32 |

#### `spec.build_tree`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.build_tree_kernel_efficient` | Build EAGLE speculative draft tree mask/positions/retrieve indices | sm80+ | int |

#### `spec.packbits`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.segment_packbits` | Segmented bit-packing of boolean tensor by indptr segments | sm80+ | bool->uint8 |

#### `spec.reconstruct_tree`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.reconstruct_indices_from_tree_mask` | Reconstruct retrieve indices/positions from a tree attention mask | sm80+ | int |

#### `spec.verify_greedy`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.verify_tree_greedy` | Greedy verification of speculative draft tree against target argmax | sm80+ | int |

### kvcache

#### `kvcache.concat_mla_k` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.concat_mla_k` | Concatenate MLA k_nope and k_rope into k buffer | sm80+ | fp16/bf16 |
| flashinfer | `flashinfer.concat_mla_k` | Concatenate MLA K nope and rope parts into full K tensor | sm80+ | fp16/bf16 |

#### `kvcache.concat_mla_q` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.concat_mla_absorb_q` | Concatenate two q tensors along last dim for MLA absorb | sm80+ | fp16/bf16 |
| vllm | `vllm.concat_mla_q` | Concatenate ql_nope and q_pe into a single MLA query tensor | sm70+ | fp16/bf16 |

#### `kvcache.append`
- kernel-set ABI: `ks_reshape_and_cache`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.page.append_paged_kv_cache` | Scatter-append new K/V into a paged KV cache at batch_indices/positions | sm80+ | fp16/bf16/fp8 |

#### `kvcache.append_mla`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.page.append_paged_mla_kv_cache` | Scatter-append compressed MLA ckv/kpe into paged MLA cache | sm80+ | fp16/bf16 |

#### `kvcache.concat_cache_mla`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.concat_and_cache_mla` | Concat kv_c and k_pe and write into MLA paged cache at slot_mapping | sm70+ | fp16/bf16, fp8 cache |

#### `kvcache.concat_cache_mla_rope`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.concat_and_cache_mla_rope_fused` | Fused RoPE on q_pe/k_pe then concat with kv_c and write to MLA cache | sm70+ | fp16/bf16, fp8 cache |

#### `kvcache.convert_fp8`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.convert_fp8` | Convert key/value cache to/from fp8 data type with scale | sm89+ | fp16/bf16 <-> fp8 |
| vllm | `vllm.cp_gather_and_upconvert_fp8_kv_cache` | Context-parallel gather + upconvert fp8 KV cache into higher precision dst | sm89+ | fp8 -> fp16/bf16 |

#### `kvcache.gather`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.cp_gather_cache` | Context-parallel gather of cache blocks into contiguous dst by block_table | sm70+ | any |

#### `kvcache.gather_dequant`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.gather_and_maybe_dequant_cache` | Gather paged cache blocks to a contiguous dst, dequantizing fp8 cache if needed | sm70+ | fp8 -> fp16/bf16 |

#### `kvcache.gather_indexer`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.cp_gather_indexer_k_quant_cache` | Context-parallel gather of quantized indexer K cache, outputting K and scales | sm89+ | fp8 cache |

#### `kvcache.indexer_quant`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.indexer_k_quant_and_cache` | Quantize K (per quant_block_size) and write into indexer KV cache at slot_mapping | sm89+ | in fp16/bf16, fp8 cache |

#### `kvcache.reshape_and_cache`
- kernel-set ABI: `ks_reshape_and_cache`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.reshape_and_cache` | Reshape key/value and write into paged KV cache at slot_mapping (with kv fp8 scaling) | sm70+ | fp16/bf16, fp8 cache |

#### `kvcache.reshape_and_cache_flash`
- kernel-set ABI: `ks_reshape_and_cache`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.reshape_and_cache_flash` | Reshape key/value and write into FlashAttention-layout paged KV cache (with kv fp8 s… | sm70+ | fp16/bf16, fp8 cache |

#### `kvcache.store`
- kernel-set ABI: `ks_reshape_and_cache`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.store_cache_cpu` | CPU store K/V into paged cache by indices | x86/aarch64 CPU | fp16/bf16/fp32 |

#### `kvcache.swap_blocks`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.swap_blocks` | Copy/swap cache blocks between src and dst tensors per block_mapping | sm70+ | any |
| vllm | `vllm.swap_blocks_batch` | Batch block swap submitting all block copies in a single driver call | CPU dispatch | any |

#### `kvcache.transfer`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.transfer_kv_all_layer` | All-layer K/V cache transfer between buffers by index | sm80+ | any |
| sgl | `sgl.transfer_kv_all_layer_direct_lf_pf` | All-layer direct KV transfer layer-first->page-first via ptr lists | sm80+ | any |
| sgl | `sgl.transfer_kv_all_layer_lf_pf` | All-layer KV transfer layer-first src -> page-first dst | sm80+ | any |
| sgl | `sgl.transfer_kv_all_layer_lf_ph` | All-layer KV transfer layer-first src -> page+head dst | sm80+ | any |
| sgl | `sgl.transfer_kv_direct` | Direct multi-layer KV transfer via tensor-list pointers | sm80+ | any |
| sgl | `sgl.transfer_kv_per_layer` | Copy per-layer K/V cache entries between buffers by index | sm80+ | any |
| sgl | `sgl.transfer_kv_per_layer_direct_pf_lf` | Per-layer direct KV transfer page-first->layer-first via ptr lists | sm80+ | any |
| sgl | `sgl.transfer_kv_per_layer_pf_lf` | Per-layer KV transfer page-first src -> layer-first dst | sm80+ | any |
| sgl | `sgl.transfer_kv_per_layer_ph_lf` | Per-layer KV transfer page+head src -> layer-first dst | sm80+ | any |

#### `kvcache.transfer_mla`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.transfer_kv_all_layer_mla` | All-layer MLA cache transfer by index | sm80+ | any |
| sgl | `sgl.transfer_kv_all_layer_mla_lf_pf` | All-layer MLA transfer layer-first src -> page-first dst | sm80+ | any |
| sgl | `sgl.transfer_kv_per_layer_mla` | Per-layer MLA (single combined cache) transfer by index | sm80+ | any |
| sgl | `sgl.transfer_kv_per_layer_mla_pf_lf` | Per-layer MLA transfer page-first src -> layer-first dst | sm80+ | any |

### ssm

#### `ssm.gated_delta_rule` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.chunk_gated_delta_rule_cpu` | CPU chunked gated delta-rule linear attention | x86/aarch64 CPU | fp16/bf16/fp32 |
| flashinfer | `flashinfer.gdn_prefill` | Gated DeltaNet (GDN/linear-attention) prefill scan with state output | sm90 | fp16/bf16 |

#### `ssm.selective_scan` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.mamba.checkpointing_ssu` | Mamba multi-token selective-scan with checkpointing (chunked recurrence) | sm90 | fp16/bf16/fp32 |
| vllm | `vllm.selective_scan_fwd` | Mamba selective state-space scan forward (u,delta,A,B,C -> output) with chunked/page… | sm70+ | fp16/bf16/fp32 |

#### `ssm.chunk_cumsum`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.mamba.seq_chunk_cumsum` | Per-sequence chunked cumulative-sum (Mamba2 chunk-scan prep) | sm80+ | fp32 |

#### `ssm.conv1d`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.causal_conv1d_fwd` | Mamba causal depthwise conv1d forward (varlen, optional silu) | sm80+ | fp16/bf16 |
| sgl | `sgl.causal_conv1d_fwd_cpu` | CPU mamba causal conv1d forward | x86 CPU | bf16/fp16 |

#### `ssm.conv1d_update`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.causal_conv1d_update` | Mamba causal conv1d decode update with conv state cache | sm80+ | fp16/bf16 |
| sgl | `sgl.causal_conv1d_update_cpu` | CPU mamba causal conv1d decode update | x86 CPU | bf16/fp16 |

#### `ssm.conv1d_weight_pack`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.causal_conv1d_weight_pack` | CPU prepack causal conv1d weights | x86 CPU | bf16/fp16 |

#### `ssm.gated_delta_gating`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fused_gdn_gating_cpu` | CPU fused gated-delta-net gating computation | x86 CPU | fp32/bf16 |

#### `ssm.gated_delta_update`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fused_sigmoid_gating_delta_rule_update_cpu` | CPU fused sigmoid-gating delta-rule recurrent state update | x86 CPU | fp32/bf16 |

#### `ssm.qkvzba_split`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fused_qkvzba_split_reshape_cat_contiguous_cpu` | CPU contiguous variant of QKVZBA split/reshape/concat | x86 CPU | bf16/fp16 |
| sgl | `sgl.fused_qkvzba_split_reshape_cat_cpu` | CPU split/reshape/concat of mixed QKVZ + BA tensors (gated delta net) | x86 CPU | bf16/fp16 |

#### `ssm.selective_state_update`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.mamba.selective_state_update` | Mamba selective state-space update (SSU): per-step state recurrence with optional ga… | sm80+ | fp16/bf16/fp32 |

### comm

#### `comm.allreduce` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.all_reduce` | Custom multi-GPU all-reduce over IPC buffers | sm80+ | fp16/bf16 |
| sgl | `sgl.all_reduce_reg` | ROCm custom all-reduce over registered buffers | ROCm/HIP | fp16/bf16 |
| sgl | `sgl.all_reduce_unreg` | ROCm custom all-reduce over unregistered buffer | ROCm/HIP | fp16/bf16 |
| sgl | `sgl.mscclpp_allreduce` | MSCCL++ based multi-GPU all-reduce | sm90 | fp16/bf16 |
| sgl | `sgl.mscclpp_init_context` | Initialize MSCCL++ all-reduce communication context | sm90 | n/a |
| sgl | `sgl.qr_all_reduce` | Quick all-reduce (ROCm) one-shot reduce | ROCm/HIP | fp16/bf16 |
| sgl | `sgl.shm_allreduce` | CPU shared-memory all-reduce | x86/aarch64 CPU | fp32/bf16 |
| flashinfer | `flashinfer.comm.mnnvl.allreduce` | NVSHMEM-based all-reduce (mixed_comm) | sm90+ | fp16/bf16/fp32 |
| flashinfer | `flashinfer.comm.mnnvl.fused_allreduce_allgather` | Fused all-reduce + all-gather (NVSHMEM) | sm90+ | fp16/bf16/fp32 |
| flashinfer | `flashinfer.comm.mnnvl.fused_reducescatter_allreduce` | Fused reduce-scatter + all-reduce (NVSHMEM) | sm90+ | fp16/bf16/fp32 |
| flashinfer | `flashinfer.comm.trtllm_custom_all_reduce` | TRT-LLM custom one-shot/two-shot all-reduce (Lamport buffers) | sm80+ | fp16/bf16/fp32 |
| flashinfer | `flashinfer.comm.vllm_all_reduce` | vLLM custom all-reduce (one/two-shot) out-of-place | sm80+ | fp16/bf16/fp32 |
| vllm | `vllm.all_reduce` | Custom one-shot/two-shot GPU all-reduce over registered IPC buffers | sm70+ (NVLink) | fp16/bf16/fp32 |
| vllm | `vllm.qr_all_reduce` | ROCm Quick Reduce all-reduce with optional quantization level and bf16->half cast | ROCm | fp16/bf16 |

#### `comm.init` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.init_custom_ar` | Initialize custom IPC all-reduce context | sm80+ | n/a |
| flashinfer | `flashinfer.comm.init_custom_ar` | Initialize vLLM custom all-reduce IPC workspace/handles | sm80+ | meta |
| flashinfer | `flashinfer.comm.nvshmem_init` | NVSHMEM init/finalize and PE/topology query primitives | sm90+ | meta |
| flashinfer | `flashinfer.comm.trtllm_lamport_initialize` | Initialize Lamport flag buffers for custom all-reduce | sm80+ | uint8/flag |
| vllm | `vllm.allocate_shared_buffer_and_handle` | Allocate a shared GPU buffer and return its pointer and IPC handle | any | n/a |
| vllm | `vllm.dispose` | Dispose a custom all-reduce context | any | n/a |
| vllm | `vllm.free_shared_buffer` | Free a shared GPU buffer by pointer | any | n/a |
| vllm | `vllm.get_graph_buffer_ipc_meta` | Get IPC metadata for CUDA-graph-captured all-reduce buffers | any | n/a |
| vllm | `vllm.init_custom_ar` | Initialize custom all-reduce context over IPC tensors | sm70+ (NVLink) | n/a |
| vllm | `vllm.init_custom_qr` | Initialize ROCm Quick Reduce all-reduce context | ROCm | n/a |
| vllm | `vllm.meta_size` | Return metadata size for custom all-reduce signal pad | any | n/a |
| vllm | `vllm.open_mem_handle` | Open a remote shared-memory IPC handle, returning a device pointer | any | n/a |
| vllm | `vllm.qr_destroy` | Destroy a ROCm Quick Reduce context | ROCm | n/a |
| vllm | `vllm.qr_get_handle` | Get IPC handle for ROCm Quick Reduce buffer | ROCm | n/a |
| vllm | `vllm.qr_max_size` | Return max supported buffer size for ROCm Quick Reduce | ROCm | n/a |
| vllm | `vllm.qr_open_handles` | Open peer IPC handles for ROCm Quick Reduce | ROCm | n/a |
| vllm | `vllm.register_buffer` | Register IPC buffers with a custom all-reduce context | any | n/a |
| vllm | `vllm.register_graph_buffers` | Register CUDA-graph buffers (handles/offsets) for custom all-reduce | any | n/a |

#### `comm.allgather` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.shm_allgather` | CPU shared-memory all-gather along dim | x86/aarch64 CPU | any |
| flashinfer | `flashinfer.comm.mnnvl.allgather` | NVSHMEM-based all-gather | sm90+ | any |

#### `comm.allreduce_rmsnorm` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.comm.trtllm_allreduce_fusion` | Fused all-reduce + (residual add) + RMSNorm + optional FP8/FP4 quant | sm90/sm100 | fp16/bf16, out fp8/fp4 |
| flashinfer | `flashinfer.comm.trtllm_mnnvl_allreduce_fusion` | Multi-node NVLink (MNNVL) fused all-reduce + RMSNorm | sm90/sm100 | fp16/bf16 |
| vllm | `vllm.minimax_allreduce_rms` | Fused all-reduce + RMS norm (MiniMax) across nranks | sm80+ (CUDA) | fp16/bf16 |
| vllm | `vllm.minimax_allreduce_rms_qk` | Fused all-reduce + separate Q and K RMS norm (MiniMax) returning normed Q,K | sm80+ (CUDA) | fp16/bf16 |

#### `comm.moe_a2a_gather`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.comm.trtllm_alltoall.moe_local_gather` | Local gather of MoE tokens for all-to-all path | sm90+ | fp16/bf16/fp8 |

#### `comm.moe_a2a_prepare`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.comm.trtllm_alltoall.moe_comm_prepare_indices` | Prepare send/recv index metadata for MoE all-to-all | sm90+ | int32 |
| flashinfer | `flashinfer.comm.trtllm_alltoall.moe_prepare` | Prepare-stage kernel for MoE all-to-all communication | sm90+ | int32 |
| flashinfer | `flashinfer.comm.trtllm_moe_alltoall.moe_a2a_sanitize_expert_ids` | Sanitize/validate expert ids for MoE all-to-all | sm90+ | int32 |

#### `comm.moe_allreduce_fusion`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.comm.trtllm_moe_allreduce_fusion` | Fused MoE all-reduce + residual + norm fusion | sm90/sm100 | fp16/bf16 |

#### `comm.moe_alltoall`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.comm.trtllm_alltoall.moe_comm` | TRT-LLM MoE expert-parallel all-to-all token dispatch/combine | sm90+ | fp16/bf16/fp8 |
| flashinfer | `flashinfer.comm.trtllm_dcp_alltoall.alltoall_dcp_native` | Decode context-parallel (DCP) native all-to-all | sm90+ | fp16/bf16/fp8 |

#### `comm.moe_combine`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.comm.trtllm_moe_alltoall.moe_a2a_combine` | MoE all-to-all combine (gather + weighted reduce expert outputs) | sm90+ | fp16/bf16 |

#### `comm.moe_dispatch`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.comm.trtllm_moe_alltoall.moe_a2a_dispatch` | MoE all-to-all dispatch (scatter tokens to expert ranks) | sm90+ | fp16/bf16/fp8 |

#### `comm.moe_finalize_allreduce`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.comm.trtllm_moe_finalize_allreduce_fusion` | MoE finalize (weighted expert combine) fused with all-reduce + norm | sm90/sm100 | fp16/bf16 |

#### `comm.reduce_scatter`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.comm.mnnvl.reducescatter` | NVSHMEM-based reduce-scatter | sm90+ | fp16/bf16/fp32 |

### sparse

#### `sparse.topk_select` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fast_topk` | Fast top-k indices over ragged/paged score rows (DSv3.2 topk=2048) | sm80+ | fp32 score |
| flashinfer | `flashinfer.fast_topk_clusters_exact` | Exact fast top-k over cluster logits (sparse MLA cluster selection) | sm90/sm100 | fp16/bf16/fp32 |
| flashinfer | `flashinfer.radix_topk` | Radix-select top-k indices/values per row (sparse-attn selection) | sm80+ | fp16/bf16/fp32 |
| vllm | `vllm.persistent_topk` | Persistent-kernel top-k selection over variable-length logits | sm70+ / ROCm | fp32 |
| vllm | `vllm.top_k_per_row_decode` | Optimized per-row top-k selection over logits for decode | sm70+ / ROCm | fp32 |
| vllm | `vllm.top_k_per_row_prefill` | Optimized per-row top-k selection over logits for prefill (variable row ranges) | sm70+ / ROCm | fp32 |

#### `sparse.topk_transform_page` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.deepseek_v4_topk_transform_512` | DeepSeek-V4 indexer top-k select -> paged physical slot indices (topk<=1024) | sm80+ (ROCm path) | fp32 scores |
| sgl | `sgl.fast_topk_transform_fused` | Top-k then transform indices to page-table (page_size=1) slots | sm80+ | fp32 score |
| flashinfer | `flashinfer.fast_topk_clusters_exact_page_table_transform` | Fused exact cluster top-k + page-table transform | sm90/sm100 | fp16/bf16/fp32 |
| flashinfer | `flashinfer.radix_topk_page_table_transform` | Fused radix top-k + page-table transform for sparse paged attention | sm80+ | fp16/bf16/fp32 |

#### `sparse.topk_transform_ragged` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.fast_topk_transform_ragged_fused` | Top-k then transform indices into ragged (non-paged) KV layout | sm80+ | fp32 score |
| flashinfer | `flashinfer.fast_topk_clusters_exact_ragged_transform` | Fused exact cluster top-k + ragged index transform | sm90/sm100 | fp16/bf16/fp32 |
| flashinfer | `flashinfer.radix_topk_ragged_transform` | Fused radix top-k + ragged index transform for sparse attention | sm80+ | fp16/bf16/fp32 |

### elementwise

#### `elementwise.copy`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.copy_to_gpu_no_ce` | Host->device copy without copy-engine (kernel-driven memcpy) | sm80+ | any |

#### `elementwise.mul_add`
- kernel-set ABI: `ks_axpby`

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.musa_fused_mul_add` | MUSA fused out = self*scale + bias | MUSA | fp16/bf16 |

### util

#### `util.weak_ref` **[multi-lib]**
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.weak_ref_tensor` | Create a non-owning weak reference tensor sharing storage | sm80+ | any |
| vllm | `vllm.weak_ref_tensor` | Create a weak reference (alias) tensor from a CUDA tensor's raw data pointer | any CUDA | any |

#### `util.conv3d_embed`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.conv3d_embed_cpu` | CPU conv3d fast path for patch embedding | x86/aarch64 CPU | bf16/fp16 |
| sgl | `sgl.conv3d_embed_weight_pack` | CPU prepack conv3d patch-embed weights | x86/aarch64 CPU | bf16/fp16 |

#### `util.cuda_view`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.get_cuda_view_from_cpu_tensor` | Create a CUDA-accessible view from a (pinned) CPU tensor | any CUDA | any |

#### `util.debug_stats`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| flashinfer | `flashinfer.api_log_print_tensor_stats` | Debug kernel: compute and print tensor stats (min/max/mean/nan) by id | sm80+ | any |

#### `util.device_query`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.get_device_attribute` | Query a CUDA device attribute by id | any CUDA | n/a |
| vllm | `vllm.get_max_shared_memory_per_block_device_attribute` | Query the maximum shared memory per block for a device | any CUDA | n/a |

#### `util.greenctx_stream`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.create_greenctx_stream_by_value` | Create two green-context CUDA streams partitioned by SM count | sm90 | n/a |

#### `util.hadamard`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| vllm | `vllm.hadacore_transform` | Fast Hadamard transform (Hadacore) over input, optionally in place | sm80+ | fp16/bf16/fp32 |

#### `util.image_preprocess`
- kernel-set ABI: — (no kernel-set ABI op)

| lib | addr | what | arch | dtype |
|---|---|---|---|---|
| sgl | `sgl.image_preprocess_cpu` | CPU image preprocessor (resize/rescale/normalize/patchify) | x86/aarch64 CPU | uint8->fp32/bf16 |

---

## kernel-set C ABI ops referenced

Logical ops above map onto these kernel-set ABI operators (from `providers/registry.json`):

- `ks_axpby`
- `ks_fused_add_rmsnorm`
- `ks_gelu`
- `ks_gelu_and_mul`
- `ks_gemm`
- `ks_gemm_batched`
- `ks_gemm_w4a16`
- `ks_gemm_w8a8`
- `ks_gemma_rmsnorm`
- `ks_layernorm`
- `ks_moe_compute_permutation`
- `ks_moe_gate_sigmoid_group_topk`
- `ks_moe_gate_softmax_topk`
- `ks_moe_grouped_gemm`
- `ks_moe_permute`
- `ks_moe_unpermute`
- `ks_quantize_fp8`
- `ks_quantize_int8`
- `ks_reshape_and_cache`
- `ks_rmsnorm`
- `ks_rope`
- `ks_sample`
- `ks_silu_and_mul`
- `ks_softmax`

Logical ops in families **attention, ssm, comm, kvcache (transfer/gather), spec, sparse, util** and the
newer quant formats (fp4/mxfp8/mxfp4/w4a8) have **no native kernel-set C ABI op** today (`ks_abi = null`);
they are exposed only through their library-specific atomic providers.
