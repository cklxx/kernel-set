/**
 * kernel-set — low-level FFI layer.
 *
 * This module locates and loads the shared library (libkernel_set.so / .dylib /
 * kernel_set.dll) with koffi and declares EVERY function in the C ABI with an
 * exact koffi signature. It performs no validation and applies no ergonomics —
 * see ./index.ts for typed, throwing wrappers.
 *
 * Calling convention notes that hold for the whole ABI:
 *   - Device pointers are raw integer addresses passed as `void*`. koffi accepts
 *     a JS Number or BigInt for a pointer-typed argument, so callers can pass
 *     either a `number` (small/aligned host or device address) or a `bigint`
 *     (full 64-bit address). We declare these params as the koffi alias
 *     `ks_devptr` (= `void *`).
 *   - Streams are `ks_stream_t` (opaque `void*`); 0 / null == the default stream.
 *   - Every kernel returns `ks_status_t` (an int; 0 == KS_SUCCESS).
 *   - `int64_t` dimensions are declared as koffi `int64` and accept Number or
 *     BigInt; koffi returns BigInt for any int64 *output* parameter.
 */
import koffi from 'koffi';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

// ---------------------------------------------------------------------------
// Shared-library discovery & loading
// ---------------------------------------------------------------------------

/** Platform-specific default file names for the shared library. */
function defaultLibNames(): string[] {
  switch (process.platform) {
    case 'win32':
      return ['kernel_set.dll', 'libkernel_set.dll'];
    case 'darwin':
      return ['libkernel_set.dylib', 'kernel_set.dylib'];
    default:
      return ['libkernel_set.so', 'kernel_set.so'];
  }
}

/**
 * Candidate directories searched (in order) when KERNEL_SET_LIB is not an
 * absolute path to an existing file. Mirrors typical install/build layouts.
 */
function candidateDirs(): string[] {
  const dirs: string[] = [];
  const env = process.env.KERNEL_SET_LIB_DIR;
  if (env) dirs.push(...env.split(path.delimiter).filter(Boolean));

  // Relative to this binding: bindings/ts/src -> repo build trees.
  const here = __dirname;
  dirs.push(
    path.resolve(here, '..', '..', '..', 'build'),
    path.resolve(here, '..', '..', '..', 'build', 'lib'),
    path.resolve(here, '..', '..', '..', 'build', 'Release'),
    path.resolve(here, '..', '..', '..', 'build', 'Debug'),
    path.resolve(here, '..', '..', '..'),
  );

  // Standard system locations.
  if (process.platform === 'win32') {
    if (process.env.SystemRoot) {
      dirs.push(path.join(process.env.SystemRoot, 'System32'));
    }
  } else {
    dirs.push('/usr/local/lib', '/usr/lib', '/opt/kernel_set/lib');
    if (process.platform === 'darwin') {
      dirs.push('/opt/homebrew/lib', '/usr/local/opt/kernel_set/lib');
    }
  }
  return dirs;
}

/**
 * Resolve the path to the shared library.
 *
 * Resolution order:
 *   1. `KERNEL_SET_LIB` — if it points to an existing file, use it verbatim.
 *      Otherwise it is treated as a library *name* and searched on the candidate
 *      directories below (and finally handed to the OS loader as-is).
 *   2. Platform default names under each candidate directory.
 *   3. Bare default name (lets the OS loader use LD_LIBRARY_PATH / PATH / rpath).
 */
export function resolveLibraryPath(): string {
  const override = process.env.KERNEL_SET_LIB;
  if (override) {
    if (fs.existsSync(override)) return override;
    // Treat as a name and let it fall through the search dirs / loader.
    for (const dir of candidateDirs()) {
      const p = path.join(dir, override);
      if (fs.existsSync(p)) return p;
    }
    return override; // hand to loader (may be on the system search path)
  }

  const names = defaultLibNames();
  for (const dir of candidateDirs()) {
    for (const name of names) {
      const p = path.join(dir, name);
      if (fs.existsSync(p)) return p;
    }
  }
  // Last resort: bare name; rely on the OS dynamic loader search path.
  return names[0];
}

let _lib: koffi.IKoffiLib | null = null;

/** The loaded koffi library handle (lazily opened, memoized). */
export function getLib(): koffi.IKoffiLib {
  if (_lib) return _lib;
  const libPath = resolveLibraryPath();
  try {
    _lib = koffi.load(libPath);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(
      `kernel-set: failed to load shared library "${libPath}". ` +
        `Set KERNEL_SET_LIB to the absolute path of libkernel_set` +
        `${process.platform === 'win32' ? '.dll' : process.platform === 'darwin' ? '.dylib' : '.so'}. ` +
        `Underlying error: ${msg}`,
    );
  }
  return _lib;
}

// ---------------------------------------------------------------------------
// koffi type aliases mirroring kernel_set/types.h
// ---------------------------------------------------------------------------

// All device pointers and the stream handle are plain `void *` on the wire.
// koffi accepts Number | BigInt | null for void* arguments.
koffi.alias('ks_devptr', 'void *');
koffi.alias('ks_stream_t', 'void *');

// Enums are plain C ints over FFI.
koffi.alias('ks_status_t', 'int');
koffi.alias('ks_dtype_t', 'int');
koffi.alias('ks_activation_t', 'int');
koffi.alias('ks_quant_mode_t', 'int');
koffi.alias('ks_memcpy_kind_t', 'int');

/** ks_device_properties_t — mirrors runtime.h exactly (field order matters). */
export const KsDeviceProperties = koffi.struct('ks_device_properties_t', {
  name: koffi.array('char', 256),
  compute_major: 'int',
  compute_minor: 'int',
  multiprocessor_count: 'int',
  max_threads_per_block: 'int',
  max_shared_memory_per_block: 'int',
  warp_size: 'int',
  total_global_memory: 'size_t',
  supports_bf16: 'int',
  supports_fp8: 'int',
  supports_tf32: 'int',
});

// ---------------------------------------------------------------------------
// Function declarations — the full C ABI (~60 entry points).
//
// koffi.IKoffiRegisteredCallback style: lib.func('ret name(args)') returns a
// callable. We keep the C-like prototype strings for fidelity with the headers.
// `_out` struct/scalar params use koffi's `_Out_` direction marker.
// ---------------------------------------------------------------------------

const L = getLib();

/* ============================== runtime.h =============================== */

export const ks_version = L.func('const char *ks_version(void)');
export const ks_status_string = L.func(
  'const char *ks_status_string(ks_status_t status)',
);
export const ks_dtype_size_bits = L.func(
  'int ks_dtype_size_bits(ks_dtype_t dtype)',
);
export const ks_dtype_name = L.func('const char *ks_dtype_name(ks_dtype_t dtype)');
export const ks_backend_name = L.func('const char *ks_backend_name(void)');
export const ks_last_error_string = L.func('const char *ks_last_error_string(void)');

export const ks_device_count = L.func('ks_status_t ks_device_count(_Out_ int *out_count)');
export const ks_set_device = L.func('ks_status_t ks_set_device(int device)');
export const ks_get_device = L.func('ks_status_t ks_get_device(_Out_ int *out_device)');
export const ks_get_device_properties = L.func(
  'ks_status_t ks_get_device_properties(int device, _Out_ ks_device_properties_t *out_props)',
);

export const ks_stream_create = L.func(
  'ks_status_t ks_stream_create(_Out_ ks_stream_t *out_stream)',
);
export const ks_stream_destroy = L.func(
  'ks_status_t ks_stream_destroy(ks_stream_t stream)',
);
export const ks_stream_synchronize = L.func(
  'ks_status_t ks_stream_synchronize(ks_stream_t stream)',
);

export const ks_malloc_device = L.func(
  'ks_status_t ks_malloc_device(_Out_ ks_devptr *out_ptr, size_t bytes)',
);
export const ks_free_device = L.func('ks_status_t ks_free_device(ks_devptr ptr)');
export const ks_memcpy = L.func(
  'ks_status_t ks_memcpy(ks_devptr dst, ks_devptr src, size_t bytes, ks_memcpy_kind_t kind, ks_stream_t stream)',
);
export const ks_memset_device = L.func(
  'ks_status_t ks_memset_device(ks_devptr dst, int value, size_t bytes, ks_stream_t stream)',
);

/* ============================== norm.h ================================= */

export const ks_rms_norm = L.func(
  'ks_status_t ks_rms_norm(ks_devptr out, ks_devptr input, ks_devptr weight, int64 rows, int64 cols, float eps, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_rms_norm_residual = L.func(
  'ks_status_t ks_rms_norm_residual(ks_devptr out, ks_devptr residual_out, ks_devptr input, ks_devptr residual, ks_devptr weight, int64 rows, int64 cols, float eps, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_layer_norm = L.func(
  'ks_status_t ks_layer_norm(ks_devptr out, ks_devptr input, ks_devptr weight, ks_devptr bias, int64 rows, int64 cols, float eps, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_rms_norm_backward = L.func(
  'ks_status_t ks_rms_norm_backward(ks_devptr grad_input, ks_devptr grad_weight_fp32, ks_devptr grad_out, ks_devptr input, ks_devptr weight, int64 rows, int64 cols, float eps, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_layer_norm_backward = L.func(
  'ks_status_t ks_layer_norm_backward(ks_devptr grad_input, ks_devptr grad_weight_fp32, ks_devptr grad_bias_fp32, ks_devptr grad_out, ks_devptr input, ks_devptr weight, int64 rows, int64 cols, float eps, ks_dtype_t dtype, ks_stream_t stream)',
);

/* ============================== activation.h ========================== */

export const ks_silu = L.func(
  'ks_status_t ks_silu(ks_devptr out, ks_devptr input, int64 n, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_gelu = L.func(
  'ks_status_t ks_gelu(ks_devptr out, ks_devptr input, int64 n, int tanh_approx, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_relu = L.func(
  'ks_status_t ks_relu(ks_devptr out, ks_devptr input, int64 n, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_swiglu = L.func(
  'ks_status_t ks_swiglu(ks_devptr out, ks_devptr gate, ks_devptr up, int64 rows, int64 inter, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_swiglu_packed = L.func(
  'ks_status_t ks_swiglu_packed(ks_devptr out, ks_devptr input, int64 rows, int64 inter, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_geglu = L.func(
  'ks_status_t ks_geglu(ks_devptr out, ks_devptr gate, ks_devptr up, int64 rows, int64 inter, int tanh_approx, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_swiglu_backward = L.func(
  'ks_status_t ks_swiglu_backward(ks_devptr grad_gate, ks_devptr grad_up, ks_devptr grad_out, ks_devptr gate, ks_devptr up, int64 rows, int64 inter, ks_dtype_t dtype, ks_stream_t stream)',
);

/* ============================== attention.h =========================== */

export const ks_flash_attn_varlen = L.func(
  'ks_status_t ks_flash_attn_varlen(ks_devptr out, ks_devptr softmax_lse, ks_devptr q, ks_devptr k, ks_devptr v, ks_devptr cu_seqlens_q, ks_devptr cu_seqlens_k, int batch, int max_seqlen_q, int max_seqlen_k, int num_heads, int num_kv_heads, int head_dim, float softmax_scale, int causal, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_flash_attn = L.func(
  'ks_status_t ks_flash_attn(ks_devptr out, ks_devptr softmax_lse, ks_devptr q, ks_devptr k, ks_devptr v, int batch, int seqlen_q, int seqlen_k, int num_heads, int num_kv_heads, int head_dim, float softmax_scale, int causal, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_paged_attn_decode = L.func(
  'ks_status_t ks_paged_attn_decode(ks_devptr out, ks_devptr q, ks_devptr k_cache, ks_devptr v_cache, ks_devptr block_tables, ks_devptr seq_lens, int num_seqs, int num_heads, int num_kv_heads, int head_dim, int block_size, int max_blocks_per_seq, float softmax_scale, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_reshape_and_cache = L.func(
  'ks_status_t ks_reshape_and_cache(ks_devptr k_cache, ks_devptr v_cache, ks_devptr key, ks_devptr value, ks_devptr slot_mapping, int num_tokens, int num_kv_heads, int head_dim, int block_size, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_mla_decode = L.func(
  'ks_status_t ks_mla_decode(ks_devptr out, ks_devptr q_nope, ks_devptr q_pe, ks_devptr kv_cache, ks_devptr block_tables, ks_devptr seq_lens, int num_seqs, int num_heads, int kv_lora_rank, int rope_dim, int block_size, int max_blocks_per_seq, float softmax_scale, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_flash_attn_backward = L.func(
  'ks_status_t ks_flash_attn_backward(ks_devptr grad_q, ks_devptr grad_k, ks_devptr grad_v, ks_devptr grad_out, ks_devptr q, ks_devptr k, ks_devptr v, ks_devptr out, ks_devptr softmax_lse, int batch, int seqlen_q, int seqlen_k, int num_heads, int num_kv_heads, int head_dim, float softmax_scale, int causal, ks_dtype_t dtype, ks_stream_t stream)',
);

/* ============================== gemm.h ================================ */

export const ks_gemm = L.func(
  'ks_status_t ks_gemm(ks_devptr c, ks_devptr a, ks_devptr b, int64 m, int64 n, int64 k, int trans_a, int trans_b, int64 lda, int64 ldb, int64 ldc, float alpha, float beta, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_gemm_bias_act = L.func(
  'ks_status_t ks_gemm_bias_act(ks_devptr d, ks_devptr a, ks_devptr b, ks_devptr bias, int64 m, int64 n, int64 k, float alpha, ks_activation_t act, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_gemm_batched = L.func(
  'ks_status_t ks_gemm_batched(ks_devptr c, ks_devptr a, ks_devptr b, int64 batch, int64 m, int64 n, int64 k, int trans_a, int trans_b, int64 stride_a, int64 stride_b, int64 stride_c, float alpha, float beta, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_gemm_w8a8 = L.func(
  'ks_status_t ks_gemm_w8a8(ks_devptr c, ks_devptr a_i8, ks_devptr b_i8, ks_devptr a_scale, ks_devptr b_scale, ks_devptr bias, int64 m, int64 n, int64 k, ks_quant_mode_t a_mode, ks_quant_mode_t b_mode, ks_dtype_t out_dtype, ks_stream_t stream)',
);
export const ks_gemm_w4a16 = L.func(
  'ks_status_t ks_gemm_w4a16(ks_devptr c, ks_devptr a, ks_devptr b_packed, ks_devptr scales, ks_devptr zeros, ks_devptr bias, int64 m, int64 n, int64 k, int group_size, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_gemm_fp8 = L.func(
  'ks_status_t ks_gemm_fp8(ks_devptr out, ks_devptr a_fp8, ks_devptr b_fp8, ks_devptr a_scale, ks_devptr b_scale, int64 m, int64 n, int64 k, ks_quant_mode_t a_mode, ks_quant_mode_t b_mode, ks_dtype_t fp8_dtype, ks_dtype_t out_dtype, ks_stream_t stream)',
);

/* ============================== ssm.h ================================= */

export const ks_causal_conv1d = L.func(
  'ks_status_t ks_causal_conv1d(ks_devptr out, ks_devptr x, ks_devptr weight, ks_devptr bias, int batch, int dim, int seqlen, int width, int silu, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_selective_scan = L.func(
  'ks_status_t ks_selective_scan(ks_devptr out, ks_devptr x, ks_devptr dt, ks_devptr A, ks_devptr B, ks_devptr C, ks_devptr D, ks_devptr z, ks_devptr dt_bias, int delta_softplus, int batch, int dim, int seqlen, int dstate, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_selective_scan_update = L.func(
  'ks_status_t ks_selective_scan_update(ks_devptr state, ks_devptr out, ks_devptr x, ks_devptr dt, ks_devptr A, ks_devptr B, ks_devptr C, ks_devptr D, ks_devptr z, ks_devptr dt_bias, int delta_softplus, int batch, int dim, int dstate, ks_dtype_t dtype, ks_stream_t stream)',
);

/* ============================== moe.h ================================= */

export const ks_moe_gate_softmax_topk = L.func(
  'ks_status_t ks_moe_gate_softmax_topk(ks_devptr out_weights, ks_devptr out_indices, ks_devptr logits, int64 num_tokens, int num_experts, int top_k, int renormalize, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_moe_gate_sigmoid_group_topk = L.func(
  'ks_status_t ks_moe_gate_sigmoid_group_topk(ks_devptr out_weights, ks_devptr out_indices, ks_devptr logits, ks_devptr correction_bias, int64 num_tokens, int num_experts, int n_group, int topk_group, int top_k, int renormalize, float routed_scaling_factor, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_moe_compute_permutation = L.func(
  'ks_status_t ks_moe_compute_permutation(ks_devptr sorted_token_ids, ks_devptr expert_offsets, ks_devptr topk_indices, int64 num_tokens, int num_experts, int top_k, ks_stream_t stream)',
);
export const ks_moe_permute = L.func(
  'ks_status_t ks_moe_permute(ks_devptr permuted, ks_devptr input, ks_devptr sorted_token_ids, int64 num_tokens, int top_k, int64 hidden, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_moe_unpermute = L.func(
  'ks_status_t ks_moe_unpermute(ks_devptr out, ks_devptr permuted, ks_devptr sorted_token_ids, ks_devptr routing_weights, int64 num_tokens, int top_k, int64 hidden, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_moe_grouped_gemm = L.func(
  'ks_status_t ks_moe_grouped_gemm(ks_devptr c, ks_devptr a, ks_devptr b, ks_devptr expert_offsets, int num_experts, int64 total_rows, int64 n, int64 k, ks_dtype_t dtype, ks_stream_t stream)',
);

/* ============================== rope.h ================================ */

export const ks_rope_inplace = L.func(
  'ks_status_t ks_rope_inplace(ks_devptr q, ks_devptr k, ks_devptr cos, ks_devptr sin, int64 num_tokens, int num_q_heads, int num_kv_heads, int head_dim, int interleaved, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_rope = L.func(
  'ks_status_t ks_rope(ks_devptr q_out, ks_devptr k_out, ks_devptr q, ks_devptr k, ks_devptr cos, ks_devptr sin, int64 num_tokens, int num_q_heads, int num_kv_heads, int head_dim, int interleaved, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_rope_gather = L.func(
  'ks_status_t ks_rope_gather(ks_devptr q, ks_devptr k, ks_devptr cos_cache, ks_devptr sin_cache, ks_devptr positions, int64 num_tokens, int num_q_heads, int num_kv_heads, int head_dim, int interleaved, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_rope_backward = L.func(
  'ks_status_t ks_rope_backward(ks_devptr grad_q, ks_devptr grad_k, ks_devptr cos, ks_devptr sin, int64 num_tokens, int num_q_heads, int num_kv_heads, int head_dim, int interleaved, ks_dtype_t dtype, ks_stream_t stream)',
);

/* ============================== quant.h =============================== */

export const ks_quantize_fp8 = L.func(
  'ks_status_t ks_quantize_fp8(ks_devptr out, ks_devptr scale, ks_devptr input, int64 rows, int64 cols, ks_dtype_t in_dtype, ks_dtype_t fp8_dtype, ks_quant_mode_t mode, ks_stream_t stream)',
);
export const ks_dequantize_fp8 = L.func(
  'ks_status_t ks_dequantize_fp8(ks_devptr out, ks_devptr input, ks_devptr scale, int64 rows, int64 cols, ks_dtype_t out_dtype, ks_dtype_t fp8_dtype, ks_quant_mode_t mode, ks_stream_t stream)',
);
export const ks_quantize_int8 = L.func(
  'ks_status_t ks_quantize_int8(ks_devptr out, ks_devptr scale, ks_devptr input, int64 rows, int64 cols, ks_dtype_t in_dtype, ks_quant_mode_t mode, ks_stream_t stream)',
);
export const ks_dequantize_int8 = L.func(
  'ks_status_t ks_dequantize_int8(ks_devptr out, ks_devptr input, ks_devptr scale, int64 rows, int64 cols, ks_dtype_t out_dtype, ks_quant_mode_t mode, ks_stream_t stream)',
);
export const ks_dequantize_int4 = L.func(
  'ks_status_t ks_dequantize_int4(ks_devptr out, ks_devptr qweight_packed, ks_devptr scales, ks_devptr zeros, int64 k, int64 n, int group_size, ks_dtype_t out_dtype, ks_stream_t stream)',
);

/* ============================== sampling.h =========================== */

export const ks_softmax = L.func(
  'ks_status_t ks_softmax(ks_devptr out, ks_devptr input, int64 rows, int64 cols, float temperature, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_log_softmax = L.func(
  'ks_status_t ks_log_softmax(ks_devptr out, ks_devptr input, int64 rows, int64 cols, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_argmax = L.func(
  'ks_status_t ks_argmax(ks_devptr out_tokens, ks_devptr logits, int64 num_seqs, int64 vocab_size, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_sample = L.func(
  'ks_status_t ks_sample(ks_devptr out_tokens, ks_devptr out_probs, ks_devptr logits, ks_devptr temperatures, ks_devptr top_ks, ks_devptr top_ps, int64 num_seqs, int64 vocab_size, uint64 seed, uint64 philox_offset, ks_dtype_t dtype, ks_stream_t stream)',
);

/* ============================== embedding.h ========================== */

export const ks_embedding_lookup = L.func(
  'ks_status_t ks_embedding_lookup(ks_devptr out, ks_devptr table, ks_devptr indices, int indices_i64, int64 num_tokens, int64 embed_dim, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_embedding_backward = L.func(
  'ks_status_t ks_embedding_backward(ks_devptr grad_table_fp32, ks_devptr grad_out, ks_devptr indices, int indices_i64, int64 num_tokens, int64 embed_dim, ks_dtype_t dtype, ks_stream_t stream)',
);

/* ============================== elementwise.h ======================= */

export const ks_add = L.func(
  'ks_status_t ks_add(ks_devptr out, ks_devptr a, ks_devptr b, int64 n, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_mul = L.func(
  'ks_status_t ks_mul(ks_devptr out, ks_devptr a, ks_devptr b, int64 n, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_add_residual = L.func(
  'ks_status_t ks_add_residual(ks_devptr residual, ks_devptr x, int64 n, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_scale = L.func(
  'ks_status_t ks_scale(ks_devptr out, ks_devptr x, float scale, int64 n, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_cast = L.func(
  'ks_status_t ks_cast(ks_devptr out, ks_dtype_t dst_dtype, ks_devptr in, ks_dtype_t src_dtype, int64 n, ks_stream_t stream)',
);
export const ks_axpby = L.func(
  'ks_status_t ks_axpby(ks_devptr out, ks_devptr a, float alpha, ks_devptr b, float beta, int64 n, ks_dtype_t dtype, ks_stream_t stream)',
);

/* ============================== loss.h =============================== */

export const ks_cross_entropy = L.func(
  'ks_status_t ks_cross_entropy(ks_devptr losses, ks_devptr grad_logits, ks_devptr logits, ks_devptr targets, int targets_i64, int64 num_tokens, int64 vocab, int64 ignore_index, float label_smoothing, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_fused_linear_cross_entropy = L.func(
  'ks_status_t ks_fused_linear_cross_entropy(ks_devptr losses, ks_devptr grad_hidden, ks_devptr grad_weight_fp32, ks_devptr hidden, ks_devptr weight, ks_devptr targets, int targets_i64, int64 num_tokens, int64 hidden_dim, int64 vocab, int64 ignore_index, float label_smoothing, int chunk_size, ks_dtype_t dtype, ks_stream_t stream)',
);

/* ============================== optimizer.h ========================= */

export const ks_adamw = L.func(
  'ks_status_t ks_adamw(ks_devptr param, ks_devptr master_param, ks_devptr grad, ks_devptr exp_avg, ks_devptr exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay, int64 step, float grad_scale, int64 n, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_sgd_momentum = L.func(
  'ks_status_t ks_sgd_momentum(ks_devptr param, ks_devptr master_param, ks_devptr grad, ks_devptr momentum, float lr, float momentum_factor, float weight_decay, int nesterov, float grad_scale, int64 n, ks_dtype_t dtype, ks_stream_t stream)',
);
export const ks_global_grad_norm = L.func(
  'ks_status_t ks_global_grad_norm(ks_devptr out_norm, ks_devptr grads, ks_devptr sizes, int num_tensors, ks_dtype_t dtype, ks_stream_t stream)',
);

// Re-export koffi so the high-level layer can build typed pointers / arrays.
export { koffi };

// Silence unused-import lints for helpers kept for completeness.
void os;
