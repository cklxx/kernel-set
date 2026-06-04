//! FFI verification driver for the Rust binding against the CPU stub.
//!
//! Exercises linkage + marshalling without a GPU:
//!   * safe API: version(), backend_name(), Dtype helpers, device_count(),
//!     get_device_properties()
//!   * raw `sys` layer: a few ops called with null/dummy pointers to confirm
//!     symbol resolution, calling convention, and that a ks_status_t comes back.
//!
//! Run with the stub discoverable via KERNEL_SET_LIB:
//!
//! ```sh
//! cd bindings/rust
//! KERNEL_SET_LIB=/abs/path/to/bindings/verify cargo run --example verify
//! ```

use core::ffi::c_void;
use core::ptr;

use kernel_set as ks;
use kernel_set::sys;

fn main() -> Result<(), ks::Error> {
    // ---- safe API: introspection ----------------------------------------
    let v = ks::version();
    println!("[rust] version()      = {v:?}");
    assert_eq!(v, "0.0.0-stub", "unexpected version: {v:?}");

    let bn = ks::backend_name();
    println!("[rust] backend_name() = {bn:?}");
    assert_eq!(bn, "stub", "unexpected backend: {bn:?}");

    let bits = ks::Dtype::Bf16.size_bits();
    let name = ks::Dtype::Bf16.name();
    println!("[rust] Dtype::Bf16 size_bits={bits} name={name:?}");
    assert_eq!(bits, 16, "stub should report 16 bits");

    // ---- safe API: device queries (out-params) --------------------------
    let n = ks::device_count()?;
    println!("[rust] device_count() = {n}");
    assert_eq!(n, 1, "stub should report 1 device");

    let props = ks::get_device_properties(0)?;
    println!(
        "[rust] device 0: name={:?} warp={} max_threads={}",
        props.name, props.warp_size, props.max_threads_per_block
    );
    assert_eq!(props.name, "stub-device");
    assert_eq!(props.warp_size, 32);

    // ---- raw sys layer: ops with dummy/null pointers --------------------
    // These confirm the extern "C" decls resolve and marshal correctly. The
    // stub never dereferences the pointers, so null/dangling is safe here.
    unsafe {
        // elementwise add: out, a, b, n, dtype, stream
        let st = sys::ks_add(
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            16,
            sys::ks_dtype_t::KS_DTYPE_F16,
            ptr::null_mut(),
        );
        println!("[rust] sys::ks_add(null...) -> {st:?}");
        assert_eq!(st, sys::ks_status_t::KS_SUCCESS);

        // rms_norm: out, input, weight, rows, cols, eps, dtype, stream
        let st = sys::ks_rms_norm(
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            2,
            8,
            1e-6,
            sys::ks_dtype_t::KS_DTYPE_F16,
            ptr::null_mut(),
        );
        println!("[rust] sys::ks_rms_norm(null...) -> {st:?}");
        assert_eq!(st, sys::ks_status_t::KS_SUCCESS);

        // gemm: many i64 + 2 f32 + enums (exercises the calling convention)
        let st = sys::ks_gemm(
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            4,
            4,
            4, // m, n, k
            0,
            0, // trans_a, trans_b
            4,
            4,
            4, // lda, ldb, ldc
            1.0,
            0.0, // alpha, beta
            sys::ks_dtype_t::KS_DTYPE_F16,
            ptr::null_mut(),
        );
        println!("[rust] sys::ks_gemm(null...) -> {st:?}");
        assert_eq!(st, sys::ks_status_t::KS_SUCCESS);

        // sample: exercises u64 marshalling and many pointer args
        let st = sys::ks_sample(
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            4,
            32000, // num_seqs, vocab_size
            1234,
            0, // seed, philox_offset (u64)
            sys::ks_dtype_t::KS_DTYPE_F32,
            ptr::null_mut(),
        );
        println!("[rust] sys::ks_sample(null..., u64 seed) -> {st:?}");
        assert_eq!(st, sys::ks_status_t::KS_SUCCESS);

        // global_grad_norm: writes an out-param fp32 scalar
        let mut norm: f32 = -1.0;
        let grads: *const *const c_void = ptr::null();
        let st = sys::ks_global_grad_norm(
            &mut norm as *mut f32,
            grads,
            ptr::null(),
            0,
            sys::ks_dtype_t::KS_DTYPE_F32,
            ptr::null_mut(),
        );
        println!("[rust] sys::ks_global_grad_norm -> {st:?}, out_norm={norm}");
        assert_eq!(st, sys::ks_status_t::KS_SUCCESS);
        assert_eq!(norm, 0.0, "stub should zero the out-param");
    }

    println!("[rust] PASS");
    Ok(())
}
