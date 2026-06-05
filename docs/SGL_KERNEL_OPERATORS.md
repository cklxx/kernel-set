# SGLang `sgl-kernel` — operator inventory

Decomposed from the vendored source `third_party/sglang/sgl-kernel/` (commit in `SOURCE.md`). **95 public operators** across 17 modules; CUDA sources in `csrc/` (122 files).

This is the **menu we select the optimal kernel from** for the hard ops. The kernel-set dispatch layer (`bindings/python/kernel_set/backends/`) routes compute-bound ops to these (see `docs/OPTIMAL_SELECTION.md`); our own kernels stay as the portable C-ABI fallback.

## csrc categories

| dir | source files |
|---|--:|
| `csrc/allreduce/` | 8 |
| `csrc/attention/` | 3 |
| `csrc/cpu/` | 29 |
| `csrc/cutlass_extensions/` | 3 |
| `csrc/elementwise/` | 10 |
| `csrc/expert_specialization/` | 10 |
| `csrc/gemm/` | 24 |
| `csrc/grammar/` | 1 |
| `csrc/kvcacheio/` | 1 |
| `csrc/mamba/` | 1 |
| `csrc/memory/` | 1 |
| `csrc/metal/` | 1 |
| `csrc/moe/` | 17 |
| `csrc/quantization/` | 7 |
| `csrc/spatial/` | 1 |
| `csrc/speculative/` | 5 |

## Public Python operators (by module)

### Attention  ·  `sgl_kernel/attention.py`

_paged/ragged attention, KV-cache append_

**kernel-set mapping:** attention_decode/prefill — adopt

```
sgl_kernel.cutlass_mla_decode
sgl_kernel.cutlass_mla_get_workspace_size
sgl_kernel.merge_state_v2
```

### MoE · CUTLASS  ·  `sgl_kernel/cutlass_moe.py`

_cutlass grouped/fused MoE_

**kernel-set mapping:** moe_grouped_gemm — adopt

```
sgl_kernel.cutlass_w4a8_moe_mm
sgl_kernel.get_cutlass_w4a8_moe_mm_data
```

### Elementwise / Norm / RoPE / Activation  ·  `sgl_kernel/elementwise.py`

_rmsnorm, fused_add_rmsnorm, gemma_rmsnorm, silu_and_mul, rotary, etc._

**kernel-set mapping:** rmsnorm/rope/swiglu — competitive (kernel-set ~SOTA); adopt where faster

```
sgl_kernel.concat_mla_absorb_q
sgl_kernel.concat_mla_k
sgl_kernel.copy_to_gpu_no_ce
sgl_kernel.dsv4_fused_k_norm_rope_flashmla
sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant
sgl_kernel.dsv4_fused_q_norm_rope
sgl_kernel.fused_add_rmsnorm
sgl_kernel.gelu_and_mul
sgl_kernel.gelu_tanh_and_mul
sgl_kernel.gemma_fused_add_rmsnorm
sgl_kernel.gemma_rmsnorm
sgl_kernel.rmsnorm
sgl_kernel.rotary_embedding
sgl_kernel.silu_and_mul
```

### MoE · Expert-Specialization  ·  `sgl_kernel/expert_specialization.py`

_EP expert balancing / specialization_

**kernel-set mapping:** moe EP (future)

```
sgl_kernel.es_fp8_blockwise_scaled_grouped_mm
sgl_kernel.es_sm100_mxfp8_blockscaled_grouped_mm
sgl_kernel.es_sm100_mxfp8_blockscaled_grouped_quant
```

### Attention · FlashAttention  ·  `sgl_kernel/flash_attn.py`

_FA2/FA3 varlen + kvcache_

**kernel-set mapping:** attention_prefill/decode — adopt (rank-1)

```
sgl_kernel.flash_attn_varlen_func
sgl_kernel.flash_attn_with_kvcache
sgl_kernel.get_scheduler_metadata
sgl_kernel.is_fa3_supported
sgl_kernel.maybe_contiguous
```

### Attention · MLA  ·  `sgl_kernel/flash_mla.py`

_DeepSeek FlashMLA decode (sm90)_

**kernel-set mapping:** mla_decode — adopt on sm90 / hybrid

```
sgl_kernel.flash_mla_sparse_fwd
sgl_kernel.flash_mla_with_kvcache
sgl_kernel.get_mla_metadata
```

### GEMM  ·  `sgl_kernel/gemm.py`

_fp8/int8/bf16 (cutlass) scaled mm, bmm_

**kernel-set mapping:** gemm/gemm_fp8/w8a8 — adopt

```
sgl_kernel.awq_dequantize
sgl_kernel.bmm_fp8
sgl_kernel.dsv3_fused_a_gemm
sgl_kernel.dsv3_router_gemm
sgl_kernel.fp8_blockwise_scaled_mm
sgl_kernel.fp8_scaled_mm
sgl_kernel.gptq_gemm
sgl_kernel.gptq_shuffle
sgl_kernel.int8_scaled_mm
sgl_kernel.qserve_w4a8_per_chn_gemm
sgl_kernel.qserve_w4a8_per_group_gemm
sgl_kernel.sgl_per_token_group_quant_8bit
sgl_kernel.sgl_per_token_quant_fp8
sgl_kernel.shuffle_rows
```

### Grammar / Guided  ·  `sgl_kernel/grammar.py`

_bitmask apply for guided decoding_

**kernel-set mapping:** logit-proc (future)

```
sgl_kernel.apply_token_bitmask_inplace_cuda
```

### Memory · KV-cache IO  ·  `sgl_kernel/kvcacheio.py`

_kv-cache load/store_

**kernel-set mapping:** kv-cache mgmt (future)

```
sgl_kernel.is_hip
sgl_kernel.transfer_kv_all_layer
sgl_kernel.transfer_kv_all_layer_direct_lf_pf
sgl_kernel.transfer_kv_all_layer_lf_pf
sgl_kernel.transfer_kv_all_layer_lf_ph
sgl_kernel.transfer_kv_all_layer_mla
sgl_kernel.transfer_kv_all_layer_mla_lf_pf
sgl_kernel.transfer_kv_direct
sgl_kernel.transfer_kv_per_layer
sgl_kernel.transfer_kv_per_layer_direct_pf_lf
sgl_kernel.transfer_kv_per_layer_mla
sgl_kernel.transfer_kv_per_layer_mla_pf_lf
sgl_kernel.transfer_kv_per_layer_pf_lf
sgl_kernel.transfer_kv_per_layer_ph_lf
```

### State-Space (Mamba)  ·  `sgl_kernel/mamba.py`

_selective scan / causal-conv1d / chunk scan_

**kernel-set mapping:** (self-develop gap — no ks SSM ABI yet)

```
sgl_kernel.causal_conv1d_fn_cpu
sgl_kernel.causal_conv1d_fwd
sgl_kernel.causal_conv1d_update
sgl_kernel.causal_conv1d_update_cpu
sgl_kernel.chunk_gated_delta_rule_cpu
```

### Memory / KV-cache  ·  `sgl_kernel/memory.py`

_kv-cache copy/transfer/layout_

**kernel-set mapping:** kv-cache mgmt (future)

```
sgl_kernel.weak_ref_tensor
```

### MoE  ·  `sgl_kernel/moe.py`

_fused gate (moe_fused_gate), topk_softmax, align, fused experts_

**kernel-set mapping:** moe_gate/moe — adopt (rank-1 gate)

```
sgl_kernel.apply_shuffle_mul_sum
sgl_kernel.fp8_blockwise_scaled_grouped_mm
sgl_kernel.fused_qk_norm_rope
sgl_kernel.kimi_k2_moe_fused_gate
sgl_kernel.moe_align_block_size
sgl_kernel.moe_fused_gate
sgl_kernel.moe_sum
sgl_kernel.moe_sum_reduce
sgl_kernel.prepare_moe_input
sgl_kernel.topk_sigmoid
sgl_kernel.topk_softmax
```

### Sampling  ·  `sgl_kernel/sampling.py`

_top-k/top-p/min-p renorm + sampling-from-probs_

**kernel-set mapping:** sampling — adopt

```
sgl_kernel.top_k_renorm_probs
sgl_kernel.top_p_renorm_probs
```

### Attention · Sparse  ·  `sgl_kernel/sparse_flash_attn.py`

_block-sparse / NSA attention_

**kernel-set mapping:** (future) sparse attention

```
sgl_kernel.convert_vertical_slash_indexes
sgl_kernel.convert_vertical_slash_indexes_mergehead
sgl_kernel.maybe_contiguous
sgl_kernel.sparse_attn_func
sgl_kernel.sparse_attn_varlen_func
```

### Spatial  ·  `sgl_kernel/spatial.py`

_spatial/2D helpers_

**kernel-set mapping:** misc

```
sgl_kernel.create_greenctx_stream_by_value
sgl_kernel.get_sm_available
```

### Speculative Decoding  ·  `sgl_kernel/speculative.py`

_EAGLE/tree verify, draft helpers_

**kernel-set mapping:** spec-decode (future)

```
sgl_kernel.build_tree_kernel_efficient
sgl_kernel.reconstruct_indices_from_tree_mask
sgl_kernel.segment_packbits
sgl_kernel.tree_speculative_sampling_target_only
sgl_kernel.verify_tree_greedy
```

### Sampling · top-k  ·  `sgl_kernel/top_k.py`

_fast top-k helpers_

**kernel-set mapping:** sampling — adopt

```
sgl_kernel.deepseek_v4_topk_transform_512
sgl_kernel.fast_topk
sgl_kernel.fast_topk_transform_fused
sgl_kernel.fast_topk_transform_ragged_fused
sgl_kernel.fast_topk_v2
```

---
*Generated from the vendored sgl-kernel package source (AST of the public API).*
