//! Raw, hand-written FFI declarations for the **frozen** `kernel_set` C ABI.
//!
//! This module is a 1:1 transcription of the headers under
//! `include/kernel_set/*.h`. Everything here is `unsafe` to call and uses raw
//! pointers; prefer the safe wrappers in the crate root ([`crate`]). Nothing in
//! this module allocates, validates, or interprets data — it is pure linkage.
//!
//! ## Determinism vs. bindgen
//!
//! These declarations are written by hand so the generated symbols, enum
//! discriminants, and struct layout are deterministic and reviewable, and so
//! the crate builds with **no** build-time dependency on libclang. If you would
//! rather generate them, the equivalent `build.rs` is:
//!
//! ```ignore
//! // Cargo.toml: [build-dependencies] bindgen = "0.69"
//! let bindings = bindgen::Builder::default()
//!     .header("../../include/kernel_set/kernel_set.h")
//!     .clang_arg("-I../../include")
//!     .allowlist_function("ks_.*")
//!     .allowlist_type("ks_.*")
//!     .default_enum_style(bindgen::EnumVariation::Rust { non_exhaustive: false })
//!     .generate()
//!     .expect("bindgen");
//! bindings
//!     .write_to_file(std::path::PathBuf::from(std::env::var("OUT_DIR")?).join("bindings.rs"))?;
//! ```
//! and then `include!(concat!(env!("OUT_DIR"), "/bindings.rs"));` in place of
//! this file. The hand-written version below is kept authoritative.
//!
//! ## ABI conventions (from the headers)
//!
//! * Device pointers are raw addresses passed as `*mut c_void` / `*const c_void`.
//! * Streams are `ks_stream_t` (an opaque `*mut c_void`; null == default stream).
//! * Every entry point returns a [`ks_status_t`].
//! * Integer dimensions are `i64` (`int64_t`) unless the header uses `int`.

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(clippy::missing_safety_doc)]
// The raw FFI surface is documented at the module level and 1:1 with the C
// headers; individual `extern "C"` items deliberately repeat no docs.
#![allow(missing_docs)]

use core::ffi::c_void;
use libc::{c_char, c_int};

/// Opaque GPU stream handle (`cudaStream_t` / `hipStream_t`). Null == default.
pub type ks_stream_t = *mut c_void;

// ---------------------------------------------------------------------------
// Enums (#[repr(C)] — match the C `enum` discriminants exactly).
// ---------------------------------------------------------------------------

/// Status / error codes returned by every public entry point.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ks_status_t {
    KS_SUCCESS = 0,
    KS_ERROR_INVALID_ARGUMENT = 1,
    KS_ERROR_UNSUPPORTED_DTYPE = 2,
    KS_ERROR_UNSUPPORTED_SHAPE = 3,
    KS_ERROR_CUDA = 4,
    KS_ERROR_NOT_IMPLEMENTED = 5,
    KS_ERROR_OUT_OF_MEMORY = 6,
    KS_ERROR_ARCH_UNSUPPORTED = 7,
    KS_ERROR_INTERNAL = 8,
}

/// Numeric element types understood across the ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ks_dtype_t {
    KS_DTYPE_F32 = 0,
    KS_DTYPE_F16 = 1,
    KS_DTYPE_BF16 = 2,
    KS_DTYPE_F8E4M3 = 3,
    KS_DTYPE_F8E5M2 = 4,
    KS_DTYPE_F64 = 5,
    KS_DTYPE_I64 = 6,
    KS_DTYPE_I32 = 7,
    KS_DTYPE_I8 = 8,
    KS_DTYPE_U8 = 9,
    KS_DTYPE_I4 = 10,
}

/// Activation functions referenced by fused epilogues.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ks_activation_t {
    KS_ACT_NONE = 0,
    KS_ACT_RELU = 1,
    KS_ACT_GELU = 2,
    KS_ACT_GELU_TANH = 3,
    KS_ACT_SILU = 4,
}

/// Quantization granularity for `quant.h` / `gemm.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ks_quant_mode_t {
    KS_QUANT_PER_TENSOR = 0,
    KS_QUANT_PER_TOKEN = 1,
    KS_QUANT_PER_CHANNEL = 2,
    KS_QUANT_GROUPWISE = 3,
}

/// Direction argument for [`ks_memcpy`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ks_memcpy_kind_t {
    KS_MEMCPY_HOST_TO_DEVICE = 0,
    KS_MEMCPY_DEVICE_TO_HOST = 1,
    KS_MEMCPY_DEVICE_TO_DEVICE = 2,
}

// ---------------------------------------------------------------------------
// Structs (#[repr(C)]).
// ---------------------------------------------------------------------------

/// Device properties, mirror of `ks_device_properties_t`.
///
/// `name` is a fixed 256-byte NUL-terminated C string buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ks_device_properties_t {
    pub name: [c_char; 256],
    pub compute_major: c_int,
    pub compute_minor: c_int,
    pub multiprocessor_count: c_int,
    pub max_threads_per_block: c_int,
    pub max_shared_memory_per_block: c_int,
    pub warp_size: c_int,
    pub total_global_memory: usize, // size_t
    pub supports_bf16: c_int,
    pub supports_fp8: c_int,
    pub supports_tf32: c_int,
}

impl Default for ks_device_properties_t {
    fn default() -> Self {
        // SAFETY: a struct of integers and a fixed-size byte array is valid
        // when zero-initialized.
        unsafe { core::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// extern "C" declarations.
//
// Linkage is handled by build.rs (cargo:rustc-link-lib=dylib=kernel_set).
// ---------------------------------------------------------------------------

extern "C" {
    // ===== runtime.h: library / build introspection ========================

    pub fn ks_version() -> *const c_char;
    pub fn ks_status_string(status: ks_status_t) -> *const c_char;
    pub fn ks_dtype_size_bits(dtype: ks_dtype_t) -> c_int;
    pub fn ks_dtype_name(dtype: ks_dtype_t) -> *const c_char;
    pub fn ks_backend_name() -> *const c_char;

    // ===== runtime.h: device queries =======================================

    pub fn ks_device_count(out_count: *mut c_int) -> ks_status_t;
    pub fn ks_set_device(device: c_int) -> ks_status_t;
    pub fn ks_get_device(out_device: *mut c_int) -> ks_status_t;
    pub fn ks_get_device_properties(
        device: c_int,
        out_props: *mut ks_device_properties_t,
    ) -> ks_status_t;
    pub fn ks_last_error_string() -> *const c_char;

    // ===== runtime.h: streams ==============================================

    pub fn ks_stream_create(out_stream: *mut ks_stream_t) -> ks_status_t;
    pub fn ks_stream_destroy(stream: ks_stream_t) -> ks_status_t;
    pub fn ks_stream_synchronize(stream: ks_stream_t) -> ks_status_t;

    // ===== runtime.h: device memory ========================================

    pub fn ks_malloc_device(out_ptr: *mut *mut c_void, bytes: usize) -> ks_status_t;
    pub fn ks_free_device(ptr: *mut c_void) -> ks_status_t;
    pub fn ks_memcpy(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: ks_memcpy_kind_t,
        stream: ks_stream_t,
    ) -> ks_status_t;
    pub fn ks_memset_device(
        dst: *mut c_void,
        value: c_int,
        bytes: usize,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== norm.h ==========================================================

    pub fn ks_rms_norm(
        out: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        rows: i64,
        cols: i64,
        eps: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_rms_norm_residual(
        out: *mut c_void,
        residual_out: *mut c_void,
        input: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        rows: i64,
        cols: i64,
        eps: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_layer_norm(
        out: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        rows: i64,
        cols: i64,
        eps: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_rms_norm_backward(
        grad_input: *mut c_void,
        grad_weight_fp32: *mut c_void,
        grad_out: *const c_void,
        input: *const c_void,
        weight: *const c_void,
        rows: i64,
        cols: i64,
        eps: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_layer_norm_backward(
        grad_input: *mut c_void,
        grad_weight_fp32: *mut c_void,
        grad_bias_fp32: *mut c_void,
        grad_out: *const c_void,
        input: *const c_void,
        weight: *const c_void,
        rows: i64,
        cols: i64,
        eps: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_fused_rmsnorm_gated(
        out: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        gate: *const c_void,
        rows: i64,
        cols: i64,
        gate_act: c_int,
        eps: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== activation.h ====================================================

    pub fn ks_silu(
        out: *mut c_void,
        input: *const c_void,
        n: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_gelu(
        out: *mut c_void,
        input: *const c_void,
        n: i64,
        tanh_approx: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_relu(
        out: *mut c_void,
        input: *const c_void,
        n: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_swiglu(
        out: *mut c_void,
        gate: *const c_void,
        up: *const c_void,
        rows: i64,
        inter: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_swiglu_packed(
        out: *mut c_void,
        input: *const c_void,
        rows: i64,
        inter: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_geglu(
        out: *mut c_void,
        gate: *const c_void,
        up: *const c_void,
        rows: i64,
        inter: i64,
        tanh_approx: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_swiglu_backward(
        grad_gate: *mut c_void,
        grad_up: *mut c_void,
        grad_out: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        rows: i64,
        inter: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== attention.h =====================================================

    pub fn ks_flash_attn_varlen(
        out: *mut c_void,
        softmax_lse: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        cu_seqlens_q: *const i32,
        cu_seqlens_k: *const i32,
        batch: c_int,
        max_seqlen_q: c_int,
        max_seqlen_k: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        softmax_scale: f32,
        causal: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_flash_attn(
        out: *mut c_void,
        softmax_lse: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        batch: c_int,
        seqlen_q: c_int,
        seqlen_k: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        softmax_scale: f32,
        causal: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_paged_attn_decode(
        out: *mut c_void,
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        block_tables: *const i32,
        seq_lens: *const i32,
        num_seqs: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        block_size: c_int,
        max_blocks_per_seq: c_int,
        softmax_scale: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_paged_attn_decode_split_k(
        out: *mut c_void,
        partial_out: *mut c_void,
        partial_lse: *mut f32,
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        block_tables: *const i32,
        seq_lens: *const i32,
        num_seqs: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        block_size: c_int,
        max_blocks_per_seq: c_int,
        num_splits: c_int,
        softmax_scale: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_reshape_and_cache(
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        key: *const c_void,
        value: *const c_void,
        slot_mapping: *const i32,
        num_tokens: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        block_size: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_mla_decode(
        out: *mut c_void,
        q_nope: *const c_void,
        q_pe: *const c_void,
        kv_cache: *const c_void,
        block_tables: *const i32,
        seq_lens: *const i32,
        num_seqs: c_int,
        num_heads: c_int,
        kv_lora_rank: c_int,
        rope_dim: c_int,
        block_size: c_int,
        max_blocks_per_seq: c_int,
        softmax_scale: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_mla_decode_split_k(
        out: *mut c_void,
        partial_out: *mut c_void,
        partial_lse: *mut f32,
        q_nope: *const c_void,
        q_pe: *const c_void,
        kv_cache: *const c_void,
        block_tables: *const i32,
        seq_lens: *const i32,
        num_seqs: c_int,
        num_heads: c_int,
        kv_lora_rank: c_int,
        rope_dim: c_int,
        block_size: c_int,
        max_blocks_per_seq: c_int,
        num_splits: c_int,
        softmax_scale: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_attention_state_merge(
        out: *mut c_void,
        lse: *mut f32,
        out_a: *const c_void,
        lse_a: *const f32,
        out_b: *const c_void,
        lse_b: *const f32,
        n_rows: i64,
        v_dim: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_dsa_topk_select(
        indices: *mut i32,
        scores: *const c_void,
        n_rows: i64,
        n_cols: i64,
        topk: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_flash_attn_backward(
        grad_q: *mut c_void,
        grad_k: *mut c_void,
        grad_v: *mut c_void,
        grad_out: *const c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *const c_void,
        softmax_lse: *const c_void,
        batch: c_int,
        seqlen_q: c_int,
        seqlen_k: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        softmax_scale: f32,
        causal: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== gemm.h ==========================================================

    pub fn ks_gemm(
        c: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        m: i64,
        n: i64,
        k: i64,
        trans_a: c_int,
        trans_b: c_int,
        lda: i64,
        ldb: i64,
        ldc: i64,
        alpha: f32,
        beta: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_gemm_bias_act(
        d: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        bias: *const c_void,
        m: i64,
        n: i64,
        k: i64,
        alpha: f32,
        act: ks_activation_t,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_gemm_batched(
        c: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        batch: i64,
        m: i64,
        n: i64,
        k: i64,
        trans_a: c_int,
        trans_b: c_int,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        alpha: f32,
        beta: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_gemm_w8a8(
        c: *mut c_void,
        a_i8: *const c_void,
        b_i8: *const c_void,
        a_scale: *const f32,
        b_scale: *const f32,
        bias: *const c_void,
        m: i64,
        n: i64,
        k: i64,
        a_mode: ks_quant_mode_t,
        b_mode: ks_quant_mode_t,
        out_dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_gemm_w4a16(
        c: *mut c_void,
        a: *const c_void,
        b_packed: *const c_void,
        scales: *const c_void,
        zeros: *const c_void,
        bias: *const c_void,
        m: i64,
        n: i64,
        k: i64,
        group_size: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_gemm_nvfp4(
        c: *mut c_void,
        a_fp4: *const c_void,
        b_fp4: *const c_void,
        a_scale: *const c_void,
        b_scale: *const c_void,
        alpha: f32,
        m: i64,
        n: i64,
        k: i64,
        out_dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_gemm_fp8(
        out: *mut c_void,
        a_fp8: *const c_void,
        b_fp8: *const c_void,
        a_scale: *const f32,
        b_scale: *const f32,
        m: i64,
        n: i64,
        k: i64,
        a_mode: ks_quant_mode_t,
        b_mode: ks_quant_mode_t,
        fp8_dtype: ks_dtype_t,
        out_dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_gemm_fp8_blockwise(
        out: *mut c_void,
        a_fp8: *const c_void,
        b_fp8: *const c_void,
        a_scale: *const f32,
        b_scale: *const f32,
        m: i64,
        n: i64,
        k: i64,
        block_n: c_int,
        block_k: c_int,
        fp8_dtype: ks_dtype_t,
        out_dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== ssm.h ===========================================================

    pub fn ks_causal_conv1d(
        out: *mut c_void,
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        batch: c_int,
        dim: c_int,
        seqlen: c_int,
        width: c_int,
        silu: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_selective_scan(
        out: *mut c_void,
        x: *const c_void,
        dt: *const c_void,
        A: *const c_void,
        B: *const c_void,
        C: *const c_void,
        D: *const c_void,
        z: *const c_void,
        dt_bias: *const c_void,
        delta_softplus: c_int,
        batch: c_int,
        dim: c_int,
        seqlen: c_int,
        dstate: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_selective_scan_update(
        state: *mut c_void,
        out: *mut c_void,
        x: *const c_void,
        dt: *const c_void,
        A: *const c_void,
        B: *const c_void,
        C: *const c_void,
        D: *const c_void,
        z: *const c_void,
        dt_bias: *const c_void,
        delta_softplus: c_int,
        batch: c_int,
        dim: c_int,
        dstate: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== linear_attn.h ===================================================

    pub fn ks_gated_delta_rule(
        out: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        batch: i64,
        seqlen: i64,
        heads: i64,
        k_dim: i64,
        v_dim: i64,
        g_is_vector: c_int,
        use_qk_l2norm: c_int,
        scale: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_gated_linear_attn(
        out: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const c_void,
        head_decay: *const f32,
        batch: i64,
        seqlen: i64,
        heads: i64,
        k_dim: i64,
        v_dim: i64,
        gate_mode: c_int,
        scale: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_rwkv_wkv7(
        out: *mut c_void,
        r: *const c_void,
        w: *const c_void,
        k: *const c_void,
        v: *const c_void,
        a: *const c_void,
        b: *const c_void,
        batch: i64,
        seqlen: i64,
        heads: i64,
        k_dim: i64,
        v_dim: i64,
        scale: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== moe.h ===========================================================

    pub fn ks_moe_gate_softmax_topk(
        out_weights: *mut f32,
        out_indices: *mut i32,
        logits: *const c_void,
        num_tokens: i64,
        num_experts: c_int,
        top_k: c_int,
        renormalize: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_moe_gate_sigmoid_group_topk(
        out_weights: *mut f32,
        out_indices: *mut i32,
        logits: *const c_void,
        correction_bias: *const c_void,
        num_tokens: i64,
        num_experts: c_int,
        n_group: c_int,
        topk_group: c_int,
        top_k: c_int,
        renormalize: c_int,
        routed_scaling_factor: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_moe_compute_permutation(
        sorted_token_ids: *mut i32,
        expert_offsets: *mut i32,
        topk_indices: *const i32,
        num_tokens: i64,
        num_experts: c_int,
        top_k: c_int,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_moe_permute(
        permuted: *mut c_void,
        input: *const c_void,
        sorted_token_ids: *const i32,
        num_tokens: i64,
        top_k: c_int,
        hidden: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_moe_unpermute(
        out: *mut c_void,
        permuted: *const c_void,
        sorted_token_ids: *const i32,
        routing_weights: *const f32,
        num_tokens: i64,
        top_k: c_int,
        hidden: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_moe_grouped_gemm(
        c: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        expert_offsets: *const i32,
        num_experts: c_int,
        total_rows: i64,
        n: i64,
        k: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== rope.h ==========================================================

    pub fn ks_rope_inplace(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        num_tokens: i64,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        interleaved: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_rope(
        q_out: *mut c_void,
        k_out: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        num_tokens: i64,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        interleaved: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_rope_gather(
        q: *mut c_void,
        k: *mut c_void,
        cos_cache: *const c_void,
        sin_cache: *const c_void,
        positions: *const i32,
        num_tokens: i64,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        interleaved: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_rope_backward(
        grad_q: *mut c_void,
        grad_k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        num_tokens: i64,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        interleaved: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== quant.h =========================================================

    pub fn ks_quantize_fp8(
        out: *mut c_void,
        scale: *mut f32,
        input: *const c_void,
        rows: i64,
        cols: i64,
        in_dtype: ks_dtype_t,
        fp8_dtype: ks_dtype_t,
        mode: ks_quant_mode_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_quantize_fp8_group(
        out: *mut c_void,
        scale: *mut f32,
        input: *const c_void,
        rows: i64,
        cols: i64,
        group_size: c_int,
        in_dtype: ks_dtype_t,
        fp8_dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_dequantize_fp8(
        out: *mut c_void,
        input: *const c_void,
        scale: *const f32,
        rows: i64,
        cols: i64,
        out_dtype: ks_dtype_t,
        fp8_dtype: ks_dtype_t,
        mode: ks_quant_mode_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_quantize_int8(
        out: *mut c_void,
        scale: *mut f32,
        input: *const c_void,
        rows: i64,
        cols: i64,
        in_dtype: ks_dtype_t,
        mode: ks_quant_mode_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_dequantize_int8(
        out: *mut c_void,
        input: *const c_void,
        scale: *const f32,
        rows: i64,
        cols: i64,
        out_dtype: ks_dtype_t,
        mode: ks_quant_mode_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_dequantize_int4(
        out: *mut c_void,
        qweight_packed: *const c_void,
        scales: *const c_void,
        zeros: *const c_void,
        k: i64,
        n: i64,
        group_size: c_int,
        out_dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_repack_int4(
        out_packed: *mut c_void,
        qweight: *const c_void,
        perm: *const c_void,
        size_k: i64,
        size_n: i64,
        num_bits: c_int,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_quantize_nvfp4(
        out_fp4: *mut c_void,
        out_scales: *mut c_void,
        input: *const c_void,
        global_scale: f32,
        rows: i64,
        cols: i64,
        in_dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== sampling.h ======================================================

    pub fn ks_softmax(
        out: *mut c_void,
        input: *const c_void,
        rows: i64,
        cols: i64,
        temperature: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_log_softmax(
        out: *mut c_void,
        input: *const c_void,
        rows: i64,
        cols: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_argmax(
        out_tokens: *mut i32,
        logits: *const c_void,
        num_seqs: i64,
        vocab_size: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_sample(
        out_tokens: *mut i32,
        out_probs: *mut f32,
        logits: *const c_void,
        temperatures: *const f32,
        top_ks: *const i32,
        top_ps: *const f32,
        num_seqs: i64,
        vocab_size: i64,
        seed: u64,
        philox_offset: u64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== embedding.h =====================================================

    pub fn ks_embedding_lookup(
        out: *mut c_void,
        table: *const c_void,
        indices: *const c_void,
        indices_i64: c_int,
        num_tokens: i64,
        embed_dim: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_embedding_backward(
        grad_table_fp32: *mut c_void,
        grad_out: *const c_void,
        indices: *const c_void,
        indices_i64: c_int,
        num_tokens: i64,
        embed_dim: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== elementwise.h ===================================================

    pub fn ks_add(
        out: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        n: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_mul(
        out: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        n: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_add_residual(
        residual: *mut c_void,
        x: *const c_void,
        n: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_scale(
        out: *mut c_void,
        x: *const c_void,
        scale: f32,
        n: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_cast(
        out: *mut c_void,
        dst_dtype: ks_dtype_t,
        input: *const c_void,
        src_dtype: ks_dtype_t,
        n: i64,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_axpby(
        out: *mut c_void,
        a: *const c_void,
        alpha: f32,
        b: *const c_void,
        beta: f32,
        n: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== loss.h ==========================================================

    pub fn ks_cross_entropy(
        losses: *mut f32,
        grad_logits: *mut c_void,
        logits: *const c_void,
        targets: *const c_void,
        targets_i64: c_int,
        num_tokens: i64,
        vocab: i64,
        ignore_index: i64,
        label_smoothing: f32,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_fused_linear_cross_entropy(
        losses: *mut f32,
        grad_hidden: *mut c_void,
        grad_weight_fp32: *mut c_void,
        hidden: *const c_void,
        weight: *const c_void,
        targets: *const c_void,
        targets_i64: c_int,
        num_tokens: i64,
        hidden_dim: i64,
        vocab: i64,
        ignore_index: i64,
        label_smoothing: f32,
        chunk_size: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    // ===== optimizer.h =====================================================

    pub fn ks_adamw(
        param: *mut c_void,
        master_param: *mut c_void,
        grad: *const c_void,
        exp_avg: *mut f32,
        exp_avg_sq: *mut f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: i64,
        grad_scale: f32,
        n: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_sgd_momentum(
        param: *mut c_void,
        master_param: *mut c_void,
        grad: *const c_void,
        momentum: *mut f32,
        lr: f32,
        momentum_factor: f32,
        weight_decay: f32,
        nesterov: c_int,
        grad_scale: f32,
        n: i64,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;

    pub fn ks_global_grad_norm(
        out_norm: *mut f32,
        grads: *const *const c_void,
        sizes: *const i64,
        num_tensors: c_int,
        dtype: ks_dtype_t,
        stream: ks_stream_t,
    ) -> ks_status_t;
}
