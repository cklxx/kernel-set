//! # kernel-set
//!
//! Idiomatic Rust bindings for **kernel-set**, a high-performance LLM inference
//! and training kernel library exposed as a frozen C ABI.
//!
//! The crate has two layers:
//!
//! * [`sys`] — raw, `unsafe`, hand-written `extern "C"` declarations and
//!   `#[repr(C)]` enums/structs that mirror `include/kernel_set/*.h` 1:1.
//! * The crate root (this module) — safe, ergonomic wrappers that take typed
//!   enums and [`DevicePtr`] handles, return [`Result<T, Error>`], and translate
//!   the C status codes into Rust errors.
//!
//! Device pointers are **raw integer addresses** under the hood. The ABI never
//! dereferences them on the host, so this crate treats them as opaque
//! [`DevicePtr`] / [`DevicePtrMut`] handles. You get one of three ways to obtain
//! them:
//!
//! 1. The built-in runtime allocator: [`malloc_device`] / [`DeviceBuffer`].
//! 2. From a raw integer/`*mut c_void` you already hold (e.g. a tensor lib's
//!    `data_ptr()`): [`DevicePtr::from_raw`] / [`DevicePtrMut::from_raw`].
//! 3. From an external GPU stack via the `cust` / `tch` feature notes in the
//!    crate README.
//!
//! ## Quick start
//!
//! ```no_run
//! use kernel_set::{self as ks, Dtype, Stream};
//!
//! # fn main() -> Result<(), ks::Error> {
//! println!("kernel-set {} ({} backend)", ks::version(), ks::backend_name());
//!
//! let rows = 8i64;
//! let cols = 4096i64;
//! let bytes = (rows * cols) as usize * Dtype::Bf16.size_bytes();
//!
//! // Allocate device buffers via the built-in runtime allocator.
//! let x = ks::DeviceBuffer::new(bytes)?;
//! let w = ks::DeviceBuffer::new(cols as usize * Dtype::Bf16.size_bytes())?;
//! let mut out = ks::DeviceBuffer::new(bytes)?;
//!
//! // ... fill x / w on the device (memcpy from host, another kernel, etc.) ...
//!
//! let stream = Stream::default(); // 0 == default stream
//! ks::rms_norm(out.ptr_mut(), x.ptr(), w.ptr(), rows, cols, 1e-5, Dtype::Bf16, stream)?;
//! ks::stream_synchronize(stream)?;
//! # Ok(())
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

pub mod sys;

use core::ffi::c_void;
use core::fmt;
use std::ffi::CStr;

// Re-export the raw enums/structs for callers who want to drop to `sys`.
pub use sys::ks_device_properties_t;

// ===========================================================================
// Errors
// ===========================================================================

/// A non-success status returned by a kernel-set entry point.
///
/// Wraps the C [`sys::ks_status_t`] and exposes its human-readable description
/// (from `ks_status_string`) plus, when available, the thread-local backend
/// error string (`ks_last_error_string`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// The raw status code returned by the ABI.
    pub status: Status,
    /// The backend's thread-local error string captured at failure time, if any.
    pub backend_message: Option<String>,
}

impl Error {
    fn from_status(status: sys::ks_status_t) -> Self {
        let status = Status::from(status);
        // Capture the thread-local backend message immediately; it is only
        // guaranteed valid until the next kernel-set call on this thread.
        let backend_message = last_error_string().filter(|s| !s.is_empty());
        Error {
            status,
            backend_message,
        }
    }

    /// The static, human-readable name of the status code.
    pub fn message(&self) -> &'static str {
        self.status.message()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.backend_message {
            Some(m) => write!(f, "{} ({:?}): {}", self.status.message(), self.status, m),
            None => write!(f, "{} ({:?})", self.status.message(), self.status),
        }
    }
}

impl std::error::Error for Error {}

/// A typed mirror of [`sys::ks_status_t`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    /// Operation succeeded (`KS_SUCCESS`).
    Success,
    /// A NULL pointer or out-of-range value (`KS_ERROR_INVALID_ARGUMENT`).
    InvalidArgument,
    /// dtype not implemented for this kernel (`KS_ERROR_UNSUPPORTED_DTYPE`).
    UnsupportedDtype,
    /// shape/alignment not supported (`KS_ERROR_UNSUPPORTED_SHAPE`).
    UnsupportedShape,
    /// Underlying CUDA/HIP runtime failure (`KS_ERROR_CUDA`).
    Cuda,
    /// Declared but no backend yet (`KS_ERROR_NOT_IMPLEMENTED`).
    NotImplemented,
    /// Device or workspace allocation failed (`KS_ERROR_OUT_OF_MEMORY`).
    OutOfMemory,
    /// GPU SM/arch lacks a required feature (`KS_ERROR_ARCH_UNSUPPORTED`).
    ArchUnsupported,
    /// Invariant violated; please file a bug (`KS_ERROR_INTERNAL`).
    Internal,
    /// An unrecognized status code (forward compatibility).
    Unknown(i32),
}

impl Status {
    /// The library's static description for this status (`ks_status_string`).
    pub fn message(self) -> &'static str {
        // Prefer the library's own string; fall back to a local description if
        // we hold an Unknown code the library doesn't recognize.
        let raw: sys::ks_status_t = match self.try_into_raw() {
            Some(r) => r,
            None => return "unknown status",
        };
        // SAFETY: ks_status_string never returns NULL and returns a 'static str.
        unsafe {
            let p = sys::ks_status_string(raw);
            if p.is_null() {
                "unknown status"
            } else {
                CStr::from_ptr(p)
                    .to_str()
                    .unwrap_or("invalid status string")
            }
        }
    }

    fn try_into_raw(self) -> Option<sys::ks_status_t> {
        use sys::ks_status_t::*;
        Some(match self {
            Status::Success => KS_SUCCESS,
            Status::InvalidArgument => KS_ERROR_INVALID_ARGUMENT,
            Status::UnsupportedDtype => KS_ERROR_UNSUPPORTED_DTYPE,
            Status::UnsupportedShape => KS_ERROR_UNSUPPORTED_SHAPE,
            Status::Cuda => KS_ERROR_CUDA,
            Status::NotImplemented => KS_ERROR_NOT_IMPLEMENTED,
            Status::OutOfMemory => KS_ERROR_OUT_OF_MEMORY,
            Status::ArchUnsupported => KS_ERROR_ARCH_UNSUPPORTED,
            Status::Internal => KS_ERROR_INTERNAL,
            Status::Unknown(_) => return None,
        })
    }
}

impl From<sys::ks_status_t> for Status {
    fn from(s: sys::ks_status_t) -> Self {
        use sys::ks_status_t::*;
        match s {
            KS_SUCCESS => Status::Success,
            KS_ERROR_INVALID_ARGUMENT => Status::InvalidArgument,
            KS_ERROR_UNSUPPORTED_DTYPE => Status::UnsupportedDtype,
            KS_ERROR_UNSUPPORTED_SHAPE => Status::UnsupportedShape,
            KS_ERROR_CUDA => Status::Cuda,
            KS_ERROR_NOT_IMPLEMENTED => Status::NotImplemented,
            KS_ERROR_OUT_OF_MEMORY => Status::OutOfMemory,
            KS_ERROR_ARCH_UNSUPPORTED => Status::ArchUnsupported,
            KS_ERROR_INTERNAL => Status::Internal,
        }
    }
}

/// Result alias used throughout the safe API.
pub type Result<T> = core::result::Result<T, Error>;

/// Convert a raw status into `Result<()>`, capturing the backend message on error.
#[inline]
fn check(status: sys::ks_status_t) -> Result<()> {
    if status == sys::ks_status_t::KS_SUCCESS {
        Ok(())
    } else {
        Err(Error::from_status(status))
    }
}

// ===========================================================================
// Enums (typed mirrors with From/Into to the sys repr-C enums)
// ===========================================================================

/// Numeric element type understood across the ABI ([`sys::ks_dtype_t`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// IEEE float32.
    F32,
    /// IEEE float16 (half).
    F16,
    /// bfloat16.
    Bf16,
    /// FP8 e4m3 (OCP).
    F8E4M3,
    /// FP8 e5m2 (OCP).
    F8E5M2,
    /// IEEE float64.
    F64,
    /// int64 (token ids, indices).
    I64,
    /// int32.
    I32,
    /// int8 (quantized weights/acts).
    I8,
    /// uint8.
    U8,
    /// packed int4 (two per byte).
    I4,
}

impl Dtype {
    /// Element size in bits, as reported by the library (`ks_dtype_size_bits`).
    pub fn size_bits(self) -> i32 {
        // SAFETY: pure value query, no pointers.
        unsafe { sys::ks_dtype_size_bits(self.into()) }
    }

    /// Element size in bytes (rounded up; e.g. I4 -> 1).
    ///
    /// For sub-byte types this rounds up to a whole byte, which is rarely what
    /// you want for buffer sizing of *packed* tensors — size those by their
    /// packed layout (e.g. `k * n / 2` bytes for packed int4).
    pub fn size_bytes(self) -> usize {
        let bits = self.size_bits();
        if bits <= 0 {
            0
        } else {
            ((bits as usize) + 7) / 8
        }
    }

    /// Short token for the dtype, e.g. `"bf16"` (`ks_dtype_name`).
    pub fn name(self) -> &'static str {
        // SAFETY: ks_dtype_name never returns NULL and returns a 'static str.
        unsafe {
            let p = sys::ks_dtype_name(self.into());
            if p.is_null() {
                "unknown"
            } else {
                CStr::from_ptr(p).to_str().unwrap_or("invalid")
            }
        }
    }
}

impl From<Dtype> for sys::ks_dtype_t {
    fn from(d: Dtype) -> Self {
        use sys::ks_dtype_t::*;
        match d {
            Dtype::F32 => KS_DTYPE_F32,
            Dtype::F16 => KS_DTYPE_F16,
            Dtype::Bf16 => KS_DTYPE_BF16,
            Dtype::F8E4M3 => KS_DTYPE_F8E4M3,
            Dtype::F8E5M2 => KS_DTYPE_F8E5M2,
            Dtype::F64 => KS_DTYPE_F64,
            Dtype::I64 => KS_DTYPE_I64,
            Dtype::I32 => KS_DTYPE_I32,
            Dtype::I8 => KS_DTYPE_I8,
            Dtype::U8 => KS_DTYPE_U8,
            Dtype::I4 => KS_DTYPE_I4,
        }
    }
}

/// Activation function for fused epilogues ([`sys::ks_activation_t`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activation {
    /// Identity (no activation).
    None,
    /// ReLU.
    Relu,
    /// Exact (erf) GELU.
    Gelu,
    /// tanh-approximation GELU.
    GeluTanh,
    /// SiLU = x * sigmoid(x).
    Silu,
}

impl From<Activation> for sys::ks_activation_t {
    fn from(a: Activation) -> Self {
        use sys::ks_activation_t::*;
        match a {
            Activation::None => KS_ACT_NONE,
            Activation::Relu => KS_ACT_RELU,
            Activation::Gelu => KS_ACT_GELU,
            Activation::GeluTanh => KS_ACT_GELU_TANH,
            Activation::Silu => KS_ACT_SILU,
        }
    }
}

/// Quantization granularity ([`sys::ks_quant_mode_t`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantMode {
    /// One scale for the whole tensor.
    PerTensor,
    /// One scale per token (row).
    PerToken,
    /// One scale per channel (column).
    PerChannel,
    /// One scale per group of columns.
    Groupwise,
}

impl From<QuantMode> for sys::ks_quant_mode_t {
    fn from(m: QuantMode) -> Self {
        use sys::ks_quant_mode_t::*;
        match m {
            QuantMode::PerTensor => KS_QUANT_PER_TENSOR,
            QuantMode::PerToken => KS_QUANT_PER_TOKEN,
            QuantMode::PerChannel => KS_QUANT_PER_CHANNEL,
            QuantMode::Groupwise => KS_QUANT_GROUPWISE,
        }
    }
}

/// Memory copy direction ([`sys::ks_memcpy_kind_t`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemcpyKind {
    /// Host -> Device.
    HostToDevice,
    /// Device -> Host.
    DeviceToHost,
    /// Device -> Device.
    DeviceToDevice,
}

impl From<MemcpyKind> for sys::ks_memcpy_kind_t {
    fn from(k: MemcpyKind) -> Self {
        use sys::ks_memcpy_kind_t::*;
        match k {
            MemcpyKind::HostToDevice => KS_MEMCPY_HOST_TO_DEVICE,
            MemcpyKind::DeviceToHost => KS_MEMCPY_DEVICE_TO_HOST,
            MemcpyKind::DeviceToDevice => KS_MEMCPY_DEVICE_TO_DEVICE,
        }
    }
}

// ===========================================================================
// Stream
// ===========================================================================

/// An opaque GPU stream handle (`cudaStream_t` / `hipStream_t`).
///
/// A null stream is the **default** stream. [`Stream`] is `Copy` and does not
/// own the underlying stream — create/destroy it explicitly with
/// [`stream_create`] / [`stream_destroy`], or use [`OwnedStream`] for RAII.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Stream(pub sys::ks_stream_t);

impl Stream {
    /// The default stream (null handle).
    pub const DEFAULT: Stream = Stream(core::ptr::null_mut());

    /// Wrap a raw stream handle (e.g. a `cudaStream_t` from another library).
    ///
    /// # Safety
    /// The caller asserts the handle is a valid stream for this process's GPU
    /// context, or null for the default stream.
    #[inline]
    pub const unsafe fn from_raw(handle: sys::ks_stream_t) -> Stream {
        Stream(handle)
    }

    /// The underlying raw handle.
    #[inline]
    pub const fn as_raw(self) -> sys::ks_stream_t {
        self.0
    }

    /// Block the host until all work on this stream has completed.
    #[inline]
    pub fn synchronize(self) -> Result<()> {
        stream_synchronize(self)
    }
}

impl Default for Stream {
    #[inline]
    fn default() -> Self {
        Stream::DEFAULT
    }
}

// Stream is just an address; it is safe to send/share like a pointer wrapper.
// The underlying GPU runtime is responsible for thread-safety of the handle.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

/// An owned stream created via [`stream_create`] that is destroyed on drop.
#[derive(Debug)]
pub struct OwnedStream {
    stream: Stream,
}

impl OwnedStream {
    /// Create a new owned stream.
    pub fn new() -> Result<OwnedStream> {
        Ok(OwnedStream {
            stream: stream_create()?,
        })
    }

    /// Borrow the stream handle (does not transfer ownership).
    #[inline]
    pub fn stream(&self) -> Stream {
        self.stream
    }

    /// Block until all work on this stream completes.
    #[inline]
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Consume the wrapper and return the raw [`Stream`] without destroying it.
    #[inline]
    pub fn into_inner(self) -> Stream {
        let s = self.stream;
        core::mem::forget(self);
        s
    }
}

impl Drop for OwnedStream {
    fn drop(&mut self) {
        // Best-effort: ignore errors during drop.
        let _ = stream_destroy(self.stream);
    }
}

// ===========================================================================
// Device pointers
// ===========================================================================

/// An immutable device pointer (raw GPU address, never dereferenced on host).
///
/// Construct from the runtime allocator ([`DeviceBuffer::ptr`]), from a raw
/// address ([`DevicePtr::from_raw`] / [`DevicePtr::from_addr`]), or from an
/// external tensor lib's `data_ptr()` (see the README interop section).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct DevicePtr(pub *const c_void);

impl DevicePtr {
    /// A null device pointer.
    pub const NULL: DevicePtr = DevicePtr(core::ptr::null());

    /// Wrap a raw const pointer.
    #[inline]
    pub const fn from_raw(p: *const c_void) -> DevicePtr {
        DevicePtr(p)
    }

    /// Wrap a raw integer device address (e.g. from another FFI layer).
    #[inline]
    pub const fn from_addr(addr: usize) -> DevicePtr {
        DevicePtr(addr as *const c_void)
    }

    /// The underlying raw pointer.
    #[inline]
    pub const fn as_ptr(self) -> *const c_void {
        self.0
    }

    /// The device address as an integer.
    #[inline]
    pub fn addr(self) -> usize {
        self.0 as usize
    }

    /// True if this is the null pointer.
    #[inline]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }

    /// Offset the pointer by `bytes` (caller must keep it in-bounds).
    #[inline]
    pub fn byte_offset(self, bytes: isize) -> DevicePtr {
        // SAFETY: pointer arithmetic only; not dereferenced here.
        DevicePtr(unsafe { (self.0 as *const u8).offset(bytes) as *const c_void })
    }
}

impl Default for DevicePtr {
    #[inline]
    fn default() -> Self {
        DevicePtr::NULL
    }
}

impl From<DevicePtrMut> for DevicePtr {
    #[inline]
    fn from(p: DevicePtrMut) -> Self {
        DevicePtr(p.0 as *const c_void)
    }
}

unsafe impl Send for DevicePtr {}
unsafe impl Sync for DevicePtr {}

/// A mutable device pointer (raw GPU address, never dereferenced on host).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct DevicePtrMut(pub *mut c_void);

impl DevicePtrMut {
    /// A null device pointer.
    pub const NULL: DevicePtrMut = DevicePtrMut(core::ptr::null_mut());

    /// Wrap a raw mut pointer.
    #[inline]
    pub const fn from_raw(p: *mut c_void) -> DevicePtrMut {
        DevicePtrMut(p)
    }

    /// Wrap a raw integer device address.
    #[inline]
    pub const fn from_addr(addr: usize) -> DevicePtrMut {
        DevicePtrMut(addr as *mut c_void)
    }

    /// The underlying raw pointer.
    #[inline]
    pub const fn as_ptr(self) -> *mut c_void {
        self.0
    }

    /// The device address as an integer.
    #[inline]
    pub fn addr(self) -> usize {
        self.0 as usize
    }

    /// True if this is the null pointer.
    #[inline]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }

    /// Reborrow as an immutable [`DevicePtr`].
    #[inline]
    pub const fn as_const(self) -> DevicePtr {
        DevicePtr(self.0 as *const c_void)
    }

    /// Offset the pointer by `bytes` (caller must keep it in-bounds).
    #[inline]
    pub fn byte_offset(self, bytes: isize) -> DevicePtrMut {
        // SAFETY: pointer arithmetic only; not dereferenced here.
        DevicePtrMut(unsafe { (self.0 as *mut u8).offset(bytes) as *mut c_void })
    }
}

impl Default for DevicePtrMut {
    #[inline]
    fn default() -> Self {
        DevicePtrMut::NULL
    }
}

unsafe impl Send for DevicePtrMut {}
unsafe impl Sync for DevicePtrMut {}

/// Trait for things convertible into a `*const c_void` device address.
///
/// Implemented for [`DevicePtr`], [`DevicePtrMut`], `usize` (raw address), and
/// raw pointers, so the kernel wrappers accept whichever you have on hand.
pub trait AsDevicePtr {
    /// The const device pointer.
    fn as_device_ptr(&self) -> *const c_void;
}

/// Trait for things convertible into a `*mut c_void` device address.
pub trait AsDevicePtrMut {
    /// The mutable device pointer.
    fn as_device_ptr_mut(&self) -> *mut c_void;
}

impl AsDevicePtr for DevicePtr {
    #[inline]
    fn as_device_ptr(&self) -> *const c_void {
        self.0
    }
}
impl AsDevicePtr for DevicePtrMut {
    #[inline]
    fn as_device_ptr(&self) -> *const c_void {
        self.0 as *const c_void
    }
}
impl AsDevicePtrMut for DevicePtrMut {
    #[inline]
    fn as_device_ptr_mut(&self) -> *mut c_void {
        self.0
    }
}
impl AsDevicePtr for usize {
    #[inline]
    fn as_device_ptr(&self) -> *const c_void {
        *self as *const c_void
    }
}
impl AsDevicePtrMut for usize {
    #[inline]
    fn as_device_ptr_mut(&self) -> *mut c_void {
        *self as *mut c_void
    }
}

// ===========================================================================
// Runtime: introspection
// ===========================================================================

/// Semantic version string of the loaded library, e.g. `"0.1.0"`.
pub fn version() -> String {
    // SAFETY: ks_version never returns NULL.
    cstr_to_string(unsafe { sys::ks_version() })
}

/// The backend this build targets: `"cuda"`, `"hip"`, or `"none"`.
pub fn backend_name() -> String {
    // SAFETY: ks_backend_name never returns NULL.
    cstr_to_string(unsafe { sys::ks_backend_name() })
}

/// The thread-local last backend error string, if any.
///
/// Valid only until the next kernel-set call on this thread; the safe wrappers
/// capture it into [`Error`] automatically.
pub fn last_error_string() -> Option<String> {
    // SAFETY: ks_last_error_string never returns NULL.
    let p = unsafe { sys::ks_last_error_string() };
    if p.is_null() {
        None
    } else {
        Some(cstr_to_string(p))
    }
}

#[inline]
fn cstr_to_string(p: *const libc::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: p is a valid NUL-terminated C string for the duration of this call.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

// ===========================================================================
// Runtime: device queries
// ===========================================================================

/// Number of visible GPU devices.
pub fn device_count() -> Result<i32> {
    let mut count: libc::c_int = 0;
    check(unsafe { sys::ks_device_count(&mut count) })?;
    Ok(count)
}

/// Set the active device for the current thread.
pub fn set_device(device: i32) -> Result<()> {
    check(unsafe { sys::ks_set_device(device) })
}

/// Get the active device for the current thread.
pub fn get_device() -> Result<i32> {
    let mut device: libc::c_int = 0;
    check(unsafe { sys::ks_get_device(&mut device) })?;
    Ok(device)
}

/// Properties of a device, with the C-string `name` field decoded to Rust.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Device name.
    pub name: String,
    /// CUDA SM major / HIP gcnArch major.
    pub compute_major: i32,
    /// CUDA SM minor / HIP gcnArch minor.
    pub compute_minor: i32,
    /// Number of streaming multiprocessors.
    pub multiprocessor_count: i32,
    /// Max threads per block.
    pub max_threads_per_block: i32,
    /// Max (opt-in/static) shared memory per block, in bytes.
    pub max_shared_memory_per_block: i32,
    /// Warp size.
    pub warp_size: i32,
    /// Total global memory, in bytes.
    pub total_global_memory: usize,
    /// Whether bf16 is supported.
    pub supports_bf16: bool,
    /// Whether FP8 (e4m3/e5m2) tensor cores are supported.
    pub supports_fp8: bool,
    /// Whether TF32 is supported.
    pub supports_tf32: bool,
}

impl From<ks_device_properties_t> for DeviceProperties {
    fn from(p: ks_device_properties_t) -> Self {
        // Decode the fixed-size, NUL-terminated name buffer.
        let name = {
            let bytes: &[u8] =
                unsafe { core::slice::from_raw_parts(p.name.as_ptr() as *const u8, p.name.len()) };
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            String::from_utf8_lossy(&bytes[..end]).into_owned()
        };
        DeviceProperties {
            name,
            compute_major: p.compute_major,
            compute_minor: p.compute_minor,
            multiprocessor_count: p.multiprocessor_count,
            max_threads_per_block: p.max_threads_per_block,
            max_shared_memory_per_block: p.max_shared_memory_per_block,
            warp_size: p.warp_size,
            total_global_memory: p.total_global_memory,
            supports_bf16: p.supports_bf16 != 0,
            supports_fp8: p.supports_fp8 != 0,
            supports_tf32: p.supports_tf32 != 0,
        }
    }
}

/// Query properties of the given device.
pub fn get_device_properties(device: i32) -> Result<DeviceProperties> {
    let mut props = ks_device_properties_t::default();
    check(unsafe { sys::ks_get_device_properties(device, &mut props) })?;
    Ok(props.into())
}

// ===========================================================================
// Runtime: streams
// ===========================================================================

/// Create a new GPU stream.
pub fn stream_create() -> Result<Stream> {
    let mut handle: sys::ks_stream_t = core::ptr::null_mut();
    check(unsafe { sys::ks_stream_create(&mut handle) })?;
    Ok(Stream(handle))
}

/// Destroy a previously created stream. (Do not pass the default stream.)
pub fn stream_destroy(stream: Stream) -> Result<()> {
    check(unsafe { sys::ks_stream_destroy(stream.as_raw()) })
}

/// Block the host until all work on `stream` has completed.
pub fn stream_synchronize(stream: Stream) -> Result<()> {
    check(unsafe { sys::ks_stream_synchronize(stream.as_raw()) })
}

// ===========================================================================
// Runtime: device memory
// ===========================================================================

/// Allocate `bytes` of device memory; returns a raw mutable device pointer.
///
/// Prefer [`DeviceBuffer`] for RAII management.
pub fn malloc_device(bytes: usize) -> Result<DevicePtrMut> {
    let mut ptr: *mut c_void = core::ptr::null_mut();
    check(unsafe { sys::ks_malloc_device(&mut ptr, bytes) })?;
    Ok(DevicePtrMut(ptr))
}

/// Free device memory previously returned by [`malloc_device`].
///
/// # Safety
/// `ptr` must have come from [`malloc_device`] and not already been freed.
pub unsafe fn free_device(ptr: DevicePtrMut) -> Result<()> {
    check(sys::ks_free_device(ptr.as_ptr()))
}

/// Copy `bytes` from `src` to `dst` with the given direction on `stream`.
///
/// # Safety
/// `dst`/`src` must be valid for `bytes` in the appropriate memory space for
/// `kind`, and the copy may be asynchronous on a non-default stream.
pub unsafe fn memcpy(
    dst: impl AsDevicePtrMut,
    src: impl AsDevicePtr,
    bytes: usize,
    kind: MemcpyKind,
    stream: Stream,
) -> Result<()> {
    check(sys::ks_memcpy(
        dst.as_device_ptr_mut(),
        src.as_device_ptr(),
        bytes,
        kind.into(),
        stream.as_raw(),
    ))
}

/// Copy `bytes` from a host slice into device memory (Host -> Device).
///
/// # Safety
/// `dst` must be a valid device allocation of at least `len * size_of::<T>()`
/// bytes; the copy may be async on a non-default stream.
pub unsafe fn memcpy_h2d<T: Copy>(
    dst: impl AsDevicePtrMut,
    src: &[T],
    stream: Stream,
) -> Result<()> {
    let bytes = core::mem::size_of_val(src);
    check(sys::ks_memcpy(
        dst.as_device_ptr_mut(),
        src.as_ptr() as *const c_void,
        bytes,
        MemcpyKind::HostToDevice.into(),
        stream.as_raw(),
    ))
}

/// Copy device memory into a host slice (Device -> Host).
///
/// # Safety
/// `src` must be a valid device allocation of at least `dst.len()*size_of::<T>()`
/// bytes; on a non-default stream you must synchronize before reading `dst`.
pub unsafe fn memcpy_d2h<T: Copy>(
    dst: &mut [T],
    src: impl AsDevicePtr,
    stream: Stream,
) -> Result<()> {
    let bytes = core::mem::size_of_val(dst);
    check(sys::ks_memcpy(
        dst.as_mut_ptr() as *mut c_void,
        src.as_device_ptr(),
        bytes,
        MemcpyKind::DeviceToHost.into(),
        stream.as_raw(),
    ))
}

/// Set `bytes` of device memory to the byte value `value` on `stream`.
///
/// # Safety
/// `dst` must be a valid device allocation of at least `bytes`.
pub unsafe fn memset_device(
    dst: impl AsDevicePtrMut,
    value: i32,
    bytes: usize,
    stream: Stream,
) -> Result<()> {
    check(sys::ks_memset_device(
        dst.as_device_ptr_mut(),
        value,
        bytes,
        stream.as_raw(),
    ))
}

/// An owned device allocation managed by the kernel-set runtime allocator.
///
/// Frees its memory on drop. Use [`DeviceBuffer::ptr`] / [`DeviceBuffer::ptr_mut`]
/// to feed the kernels.
#[derive(Debug)]
pub struct DeviceBuffer {
    ptr: DevicePtrMut,
    bytes: usize,
}

impl DeviceBuffer {
    /// Allocate `bytes` of device memory.
    pub fn new(bytes: usize) -> Result<DeviceBuffer> {
        Ok(DeviceBuffer {
            ptr: malloc_device(bytes)?,
            bytes,
        })
    }

    /// Allocate room for `count` elements of `dtype` (sub-byte types rounded up).
    pub fn for_elems(count: usize, dtype: Dtype) -> Result<DeviceBuffer> {
        DeviceBuffer::new(count * dtype.size_bytes())
    }

    /// Allocation size in bytes.
    #[inline]
    pub fn len_bytes(&self) -> usize {
        self.bytes
    }

    /// Immutable device pointer to the start of the buffer.
    #[inline]
    pub fn ptr(&self) -> DevicePtr {
        self.ptr.as_const()
    }

    /// Mutable device pointer to the start of the buffer.
    #[inline]
    pub fn ptr_mut(&self) -> DevicePtrMut {
        self.ptr
    }

    /// Copy a host slice into this buffer (Host -> Device) on `stream`.
    pub fn copy_from_host<T: Copy>(&self, src: &[T], stream: Stream) -> Result<()> {
        assert!(
            core::mem::size_of_val(src) <= self.bytes,
            "copy_from_host: src ({} bytes) larger than buffer ({} bytes)",
            core::mem::size_of_val(src),
            self.bytes
        );
        // SAFETY: bounds checked above; ptr is a valid device allocation.
        unsafe { memcpy_h2d(self.ptr_mut(), src, stream) }
    }

    /// Copy this buffer into a host slice (Device -> Host) on `stream`.
    pub fn copy_to_host<T: Copy>(&self, dst: &mut [T], stream: Stream) -> Result<()> {
        assert!(
            core::mem::size_of_val(dst) <= self.bytes,
            "copy_to_host: dst ({} bytes) larger than buffer ({} bytes)",
            core::mem::size_of_val(dst),
            self.bytes
        );
        // SAFETY: bounds checked above; ptr is a valid device allocation.
        unsafe { memcpy_d2h(dst, self.ptr(), stream) }
    }

    /// Leak the buffer, returning its raw device pointer without freeing.
    pub fn into_raw(self) -> DevicePtrMut {
        let p = self.ptr;
        core::mem::forget(self);
        p
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from malloc_device and is freed exactly once.
            let _ = unsafe { free_device(self.ptr) };
        }
    }
}

// ===========================================================================
// norm.h
// ===========================================================================

/// `out = (x / rms(x)) * weight`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm(
    out: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    weight: impl AsDevicePtr,
    rows: i64,
    cols: i64,
    eps: f32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_rms_norm(
            out.as_device_ptr_mut(),
            input.as_device_ptr(),
            weight.as_device_ptr(),
            rows,
            cols,
            eps,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Fused add-RMSNorm: `residual_out = input + residual; out = rms_norm(...) * weight`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_residual(
    out: impl AsDevicePtrMut,
    residual_out: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    residual: impl AsDevicePtr,
    weight: impl AsDevicePtr,
    rows: i64,
    cols: i64,
    eps: f32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_rms_norm_residual(
            out.as_device_ptr_mut(),
            residual_out.as_device_ptr_mut(),
            input.as_device_ptr(),
            residual.as_device_ptr(),
            weight.as_device_ptr(),
            rows,
            cols,
            eps,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// `out = (x - mean) / sqrt(var + eps) * weight + bias`. `bias` may be [`DevicePtr::NULL`].
#[allow(clippy::too_many_arguments)]
pub fn layer_norm(
    out: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    weight: impl AsDevicePtr,
    bias: impl AsDevicePtr,
    rows: i64,
    cols: i64,
    eps: f32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_layer_norm(
            out.as_device_ptr_mut(),
            input.as_device_ptr(),
            weight.as_device_ptr(),
            bias.as_device_ptr(),
            rows,
            cols,
            eps,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// RMSNorm backward. `grad_weight_fp32` is a `[cols]` fp32 buffer.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward(
    grad_input: impl AsDevicePtrMut,
    grad_weight_fp32: impl AsDevicePtrMut,
    grad_out: impl AsDevicePtr,
    input: impl AsDevicePtr,
    weight: impl AsDevicePtr,
    rows: i64,
    cols: i64,
    eps: f32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_rms_norm_backward(
            grad_input.as_device_ptr_mut(),
            grad_weight_fp32.as_device_ptr_mut(),
            grad_out.as_device_ptr(),
            input.as_device_ptr(),
            weight.as_device_ptr(),
            rows,
            cols,
            eps,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// LayerNorm backward. `grad_weight_fp32`/`grad_bias_fp32` are `[cols]` fp32.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward(
    grad_input: impl AsDevicePtrMut,
    grad_weight_fp32: impl AsDevicePtrMut,
    grad_bias_fp32: impl AsDevicePtrMut,
    grad_out: impl AsDevicePtr,
    input: impl AsDevicePtr,
    weight: impl AsDevicePtr,
    rows: i64,
    cols: i64,
    eps: f32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_layer_norm_backward(
            grad_input.as_device_ptr_mut(),
            grad_weight_fp32.as_device_ptr_mut(),
            grad_bias_fp32.as_device_ptr_mut(),
            grad_out.as_device_ptr(),
            input.as_device_ptr(),
            weight.as_device_ptr(),
            rows,
            cols,
            eps,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// activation.h
// ===========================================================================

/// `out = silu(x)` over `n` elements.
pub fn silu(
    out: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    n: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_silu(
            out.as_device_ptr_mut(),
            input.as_device_ptr(),
            n,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// `out = gelu(x)` over `n` elements; `tanh_approx` selects the tanh form.
pub fn gelu(
    out: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    n: i64,
    tanh_approx: bool,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_gelu(
            out.as_device_ptr_mut(),
            input.as_device_ptr(),
            n,
            tanh_approx as libc::c_int,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// `out = relu(x)` over `n` elements.
pub fn relu(
    out: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    n: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_relu(
            out.as_device_ptr_mut(),
            input.as_device_ptr(),
            n,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// SwiGLU: `out = silu(gate) * up`, gate/up each `[rows, inter]`.
pub fn swiglu(
    out: impl AsDevicePtrMut,
    gate: impl AsDevicePtr,
    up: impl AsDevicePtr,
    rows: i64,
    inter: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_swiglu(
            out.as_device_ptr_mut(),
            gate.as_device_ptr(),
            up.as_device_ptr(),
            rows,
            inter,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Packed SwiGLU: `input` is `[rows, 2*inter]` (gate||up); `out` is `[rows, inter]`.
pub fn swiglu_packed(
    out: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    rows: i64,
    inter: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_swiglu_packed(
            out.as_device_ptr_mut(),
            input.as_device_ptr(),
            rows,
            inter,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// GeGLU: `out = gelu(gate) * up`; `tanh_approx` selects the approximation.
#[allow(clippy::too_many_arguments)]
pub fn geglu(
    out: impl AsDevicePtrMut,
    gate: impl AsDevicePtr,
    up: impl AsDevicePtr,
    rows: i64,
    inter: i64,
    tanh_approx: bool,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_geglu(
            out.as_device_ptr_mut(),
            gate.as_device_ptr(),
            up.as_device_ptr(),
            rows,
            inter,
            tanh_approx as libc::c_int,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// SwiGLU backward: produces `grad_gate`/`grad_up` from `grad_out`.
#[allow(clippy::too_many_arguments)]
pub fn swiglu_backward(
    grad_gate: impl AsDevicePtrMut,
    grad_up: impl AsDevicePtrMut,
    grad_out: impl AsDevicePtr,
    gate: impl AsDevicePtr,
    up: impl AsDevicePtr,
    rows: i64,
    inter: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_swiglu_backward(
            grad_gate.as_device_ptr_mut(),
            grad_up.as_device_ptr_mut(),
            grad_out.as_device_ptr(),
            gate.as_device_ptr(),
            up.as_device_ptr(),
            rows,
            inter,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// attention.h
// ===========================================================================

/// FlashAttention-2 variable-length packed prefill forward.
///
/// `softmax_lse` may be [`DevicePtrMut::NULL`] if not training. `cu_seqlens_*`
/// are device pointers to `[batch+1]` int32 prefix sums (use a `DevicePtr`).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen(
    out: impl AsDevicePtrMut,
    softmax_lse: impl AsDevicePtrMut,
    q: impl AsDevicePtr,
    k: impl AsDevicePtr,
    v: impl AsDevicePtr,
    cu_seqlens_q: impl AsDevicePtr,
    cu_seqlens_k: impl AsDevicePtr,
    batch: i32,
    max_seqlen_q: i32,
    max_seqlen_k: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    softmax_scale: f32,
    causal: bool,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_flash_attn_varlen(
            out.as_device_ptr_mut(),
            softmax_lse.as_device_ptr_mut(),
            q.as_device_ptr(),
            k.as_device_ptr(),
            v.as_device_ptr(),
            cu_seqlens_q.as_device_ptr() as *const i32,
            cu_seqlens_k.as_device_ptr() as *const i32,
            batch,
            max_seqlen_q,
            max_seqlen_k,
            num_heads,
            num_kv_heads,
            head_dim,
            softmax_scale,
            causal as libc::c_int,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Dense (non-varlen) FlashAttention prefill for uniform `[batch, seqlen]`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn(
    out: impl AsDevicePtrMut,
    softmax_lse: impl AsDevicePtrMut,
    q: impl AsDevicePtr,
    k: impl AsDevicePtr,
    v: impl AsDevicePtr,
    batch: i32,
    seqlen_q: i32,
    seqlen_k: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    softmax_scale: f32,
    causal: bool,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_flash_attn(
            out.as_device_ptr_mut(),
            softmax_lse.as_device_ptr_mut(),
            q.as_device_ptr(),
            k.as_device_ptr(),
            v.as_device_ptr(),
            batch,
            seqlen_q,
            seqlen_k,
            num_heads,
            num_kv_heads,
            head_dim,
            softmax_scale,
            causal as libc::c_int,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Paged KV-cache decode (FlashDecoding, single query position per sequence).
#[allow(clippy::too_many_arguments)]
pub fn paged_attn_decode(
    out: impl AsDevicePtrMut,
    q: impl AsDevicePtr,
    k_cache: impl AsDevicePtr,
    v_cache: impl AsDevicePtr,
    block_tables: impl AsDevicePtr,
    seq_lens: impl AsDevicePtr,
    num_seqs: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks_per_seq: i32,
    softmax_scale: f32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_paged_attn_decode(
            out.as_device_ptr_mut(),
            q.as_device_ptr(),
            k_cache.as_device_ptr(),
            v_cache.as_device_ptr(),
            block_tables.as_device_ptr() as *const i32,
            seq_lens.as_device_ptr() as *const i32,
            num_seqs,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
            softmax_scale,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Write new K/V into a paged cache at the given slots.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache(
    k_cache: impl AsDevicePtrMut,
    v_cache: impl AsDevicePtrMut,
    key: impl AsDevicePtr,
    value: impl AsDevicePtr,
    slot_mapping: impl AsDevicePtr,
    num_tokens: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_reshape_and_cache(
            k_cache.as_device_ptr_mut(),
            v_cache.as_device_ptr_mut(),
            key.as_device_ptr(),
            value.as_device_ptr(),
            slot_mapping.as_device_ptr() as *const i32,
            num_tokens,
            num_kv_heads,
            head_dim,
            block_size,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// DeepSeek Multi-head Latent Attention decode over a compressed KV cache.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode(
    out: impl AsDevicePtrMut,
    q_nope: impl AsDevicePtr,
    q_pe: impl AsDevicePtr,
    kv_cache: impl AsDevicePtr,
    block_tables: impl AsDevicePtr,
    seq_lens: impl AsDevicePtr,
    num_seqs: i32,
    num_heads: i32,
    kv_lora_rank: i32,
    rope_dim: i32,
    block_size: i32,
    max_blocks_per_seq: i32,
    softmax_scale: f32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_mla_decode(
            out.as_device_ptr_mut(),
            q_nope.as_device_ptr(),
            q_pe.as_device_ptr(),
            kv_cache.as_device_ptr(),
            block_tables.as_device_ptr() as *const i32,
            seq_lens.as_device_ptr() as *const i32,
            num_seqs,
            num_heads,
            kv_lora_rank,
            rope_dim,
            block_size,
            max_blocks_per_seq,
            softmax_scale,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// FlashAttention backward. Requires the forward `out` and `softmax_lse`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_backward(
    grad_q: impl AsDevicePtrMut,
    grad_k: impl AsDevicePtrMut,
    grad_v: impl AsDevicePtrMut,
    grad_out: impl AsDevicePtr,
    q: impl AsDevicePtr,
    k: impl AsDevicePtr,
    v: impl AsDevicePtr,
    out: impl AsDevicePtr,
    softmax_lse: impl AsDevicePtr,
    batch: i32,
    seqlen_q: i32,
    seqlen_k: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    softmax_scale: f32,
    causal: bool,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_flash_attn_backward(
            grad_q.as_device_ptr_mut(),
            grad_k.as_device_ptr_mut(),
            grad_v.as_device_ptr_mut(),
            grad_out.as_device_ptr(),
            q.as_device_ptr(),
            k.as_device_ptr(),
            v.as_device_ptr(),
            out.as_device_ptr(),
            softmax_lse.as_device_ptr(),
            batch,
            seqlen_q,
            seqlen_k,
            num_heads,
            num_kv_heads,
            head_dim,
            softmax_scale,
            causal as libc::c_int,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// gemm.h
// ===========================================================================

/// `C = alpha * op(A) @ op(B) + beta * C`.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    c: impl AsDevicePtrMut,
    a: impl AsDevicePtr,
    b: impl AsDevicePtr,
    m: i64,
    n: i64,
    k: i64,
    trans_a: bool,
    trans_b: bool,
    lda: i64,
    ldb: i64,
    ldc: i64,
    alpha: f32,
    beta: f32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_gemm(
            c.as_device_ptr_mut(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            m,
            n,
            k,
            trans_a as libc::c_int,
            trans_b as libc::c_int,
            lda,
            ldb,
            ldc,
            alpha,
            beta,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// `D = act(alpha * A @ B + bias)`. `bias` is `[N]` or [`DevicePtr::NULL`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_bias_act(
    d: impl AsDevicePtrMut,
    a: impl AsDevicePtr,
    b: impl AsDevicePtr,
    bias: impl AsDevicePtr,
    m: i64,
    n: i64,
    k: i64,
    alpha: f32,
    act: Activation,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_gemm_bias_act(
            d.as_device_ptr_mut(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            bias.as_device_ptr(),
            m,
            n,
            k,
            alpha,
            act.into(),
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Strided-batched GEMM. Each batch `i` uses `base + i*stride_*`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched(
    c: impl AsDevicePtrMut,
    a: impl AsDevicePtr,
    b: impl AsDevicePtr,
    batch: i64,
    m: i64,
    n: i64,
    k: i64,
    trans_a: bool,
    trans_b: bool,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
    alpha: f32,
    beta: f32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_gemm_batched(
            c.as_device_ptr_mut(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            batch,
            m,
            n,
            k,
            trans_a as libc::c_int,
            trans_b as libc::c_int,
            stride_a,
            stride_b,
            stride_c,
            alpha,
            beta,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// W8A8 GEMM: int8 A/B -> `out_dtype` C with per-token/channel/tensor scales.
///
/// `a_scale`/`b_scale` are fp32 device pointers; `bias` may be [`DevicePtr::NULL`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_w8a8(
    c: impl AsDevicePtrMut,
    a_i8: impl AsDevicePtr,
    b_i8: impl AsDevicePtr,
    a_scale: impl AsDevicePtr,
    b_scale: impl AsDevicePtr,
    bias: impl AsDevicePtr,
    m: i64,
    n: i64,
    k: i64,
    a_mode: QuantMode,
    b_mode: QuantMode,
    out_dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_gemm_w8a8(
            c.as_device_ptr_mut(),
            a_i8.as_device_ptr(),
            b_i8.as_device_ptr(),
            a_scale.as_device_ptr() as *const f32,
            b_scale.as_device_ptr() as *const f32,
            bias.as_device_ptr(),
            m,
            n,
            k,
            a_mode.into(),
            b_mode.into(),
            out_dtype.into(),
            stream.as_raw(),
        )
    })
}

/// W4A16 GEMM: fp16/bf16 activations x packed int4 weights (AWQ/GPTQ layout).
#[allow(clippy::too_many_arguments)]
pub fn gemm_w4a16(
    c: impl AsDevicePtrMut,
    a: impl AsDevicePtr,
    b_packed: impl AsDevicePtr,
    scales: impl AsDevicePtr,
    zeros: impl AsDevicePtr,
    bias: impl AsDevicePtr,
    m: i64,
    n: i64,
    k: i64,
    group_size: i32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_gemm_w4a16(
            c.as_device_ptr_mut(),
            a.as_device_ptr(),
            b_packed.as_device_ptr(),
            scales.as_device_ptr(),
            zeros.as_device_ptr(),
            bias.as_device_ptr(),
            m,
            n,
            k,
            group_size,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// FP8 GEMM: fp8 A/B -> `out_dtype` C, dequantized by fp32 scales.
///
/// Both operands are FP8 (`fp8_dtype` selects e4m3 / e5m2). `a_scale`/`b_scale`
/// are fp32 device pointers (per-token `[M]`/per-channel `[N]` or `[1]` for
/// per-tensor). Row-major, non-transposed (A row stride K, B row stride N).
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8(
    out: impl AsDevicePtrMut,
    a_fp8: impl AsDevicePtr,
    b_fp8: impl AsDevicePtr,
    a_scale: impl AsDevicePtr,
    b_scale: impl AsDevicePtr,
    m: i64,
    n: i64,
    k: i64,
    a_mode: QuantMode,
    b_mode: QuantMode,
    fp8_dtype: Dtype,
    out_dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_gemm_fp8(
            out.as_device_ptr_mut(),
            a_fp8.as_device_ptr(),
            b_fp8.as_device_ptr(),
            a_scale.as_device_ptr() as *const f32,
            b_scale.as_device_ptr() as *const f32,
            m,
            n,
            k,
            a_mode.into(),
            b_mode.into(),
            fp8_dtype.into(),
            out_dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// moe.h
// ===========================================================================

/// Softmax over experts then top-k select.
///
/// `out_weights` is fp32 `[num_tokens, top_k]`; `out_indices` is int32 `[..]`.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_softmax_topk(
    out_weights: impl AsDevicePtrMut,
    out_indices: impl AsDevicePtrMut,
    logits: impl AsDevicePtr,
    num_tokens: i64,
    num_experts: i32,
    top_k: i32,
    renormalize: bool,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_moe_gate_softmax_topk(
            out_weights.as_device_ptr_mut() as *mut f32,
            out_indices.as_device_ptr_mut() as *mut i32,
            logits.as_device_ptr(),
            num_tokens,
            num_experts,
            top_k,
            renormalize as libc::c_int,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Sigmoid + group-limited top-k gating (DeepSeek-V3 style).
///
/// `correction_bias` may be [`DevicePtr::NULL`].
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_sigmoid_group_topk(
    out_weights: impl AsDevicePtrMut,
    out_indices: impl AsDevicePtrMut,
    logits: impl AsDevicePtr,
    correction_bias: impl AsDevicePtr,
    num_tokens: i64,
    num_experts: i32,
    n_group: i32,
    topk_group: i32,
    top_k: i32,
    renormalize: bool,
    routed_scaling_factor: f32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_moe_gate_sigmoid_group_topk(
            out_weights.as_device_ptr_mut() as *mut f32,
            out_indices.as_device_ptr_mut() as *mut i32,
            logits.as_device_ptr(),
            correction_bias.as_device_ptr(),
            num_tokens,
            num_experts,
            n_group,
            topk_group,
            top_k,
            renormalize as libc::c_int,
            routed_scaling_factor,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Build the permutation that groups tokens by expert (CSR-style offsets).
#[allow(clippy::too_many_arguments)]
pub fn moe_compute_permutation(
    sorted_token_ids: impl AsDevicePtrMut,
    expert_offsets: impl AsDevicePtrMut,
    topk_indices: impl AsDevicePtr,
    num_tokens: i64,
    num_experts: i32,
    top_k: i32,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_moe_compute_permutation(
            sorted_token_ids.as_device_ptr_mut() as *mut i32,
            expert_offsets.as_device_ptr_mut() as *mut i32,
            topk_indices.as_device_ptr() as *const i32,
            num_tokens,
            num_experts,
            top_k,
            stream.as_raw(),
        )
    })
}

/// Gather rows of `input` into `permuted` following `sorted_token_ids`.
#[allow(clippy::too_many_arguments)]
pub fn moe_permute(
    permuted: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    sorted_token_ids: impl AsDevicePtr,
    num_tokens: i64,
    top_k: i32,
    hidden: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_moe_permute(
            permuted.as_device_ptr_mut(),
            input.as_device_ptr(),
            sorted_token_ids.as_device_ptr() as *const i32,
            num_tokens,
            top_k,
            hidden,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Scatter-add expert outputs back, weighted by routing weights.
#[allow(clippy::too_many_arguments)]
pub fn moe_unpermute(
    out: impl AsDevicePtrMut,
    permuted: impl AsDevicePtr,
    sorted_token_ids: impl AsDevicePtr,
    routing_weights: impl AsDevicePtr,
    num_tokens: i64,
    top_k: i32,
    hidden: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_moe_unpermute(
            out.as_device_ptr_mut(),
            permuted.as_device_ptr(),
            sorted_token_ids.as_device_ptr() as *const i32,
            routing_weights.as_device_ptr() as *const f32,
            num_tokens,
            top_k,
            hidden,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Grouped GEMM: per-expert `C[off_e:off_{e+1}] = A[..] @ B_e`.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_gemm(
    c: impl AsDevicePtrMut,
    a: impl AsDevicePtr,
    b: impl AsDevicePtr,
    expert_offsets: impl AsDevicePtr,
    num_experts: i32,
    total_rows: i64,
    n: i64,
    k: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_moe_grouped_gemm(
            c.as_device_ptr_mut(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            expert_offsets.as_device_ptr() as *const i32,
            num_experts,
            total_rows,
            n,
            k,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// rope.h
// ===========================================================================

/// In-place RoPE applied to q and k. `k` may be [`DevicePtrMut::NULL`] to skip.
#[allow(clippy::too_many_arguments)]
pub fn rope_inplace(
    q: impl AsDevicePtrMut,
    k: impl AsDevicePtrMut,
    cos: impl AsDevicePtr,
    sin: impl AsDevicePtr,
    num_tokens: i64,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    interleaved: bool,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_rope_inplace(
            q.as_device_ptr_mut(),
            k.as_device_ptr_mut(),
            cos.as_device_ptr(),
            sin.as_device_ptr(),
            num_tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            interleaved as libc::c_int,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Out-of-place RoPE writing to `q_out`/`k_out`.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    q_out: impl AsDevicePtrMut,
    k_out: impl AsDevicePtrMut,
    q: impl AsDevicePtr,
    k: impl AsDevicePtr,
    cos: impl AsDevicePtr,
    sin: impl AsDevicePtr,
    num_tokens: i64,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    interleaved: bool,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_rope(
            q_out.as_device_ptr_mut(),
            k_out.as_device_ptr_mut(),
            q.as_device_ptr(),
            k.as_device_ptr(),
            cos.as_device_ptr(),
            sin.as_device_ptr(),
            num_tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            interleaved as libc::c_int,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Gathered RoPE: cos/sin caches `[max_pos, head_dim/2]` indexed by `positions`.
#[allow(clippy::too_many_arguments)]
pub fn rope_gather(
    q: impl AsDevicePtrMut,
    k: impl AsDevicePtrMut,
    cos_cache: impl AsDevicePtr,
    sin_cache: impl AsDevicePtr,
    positions: impl AsDevicePtr,
    num_tokens: i64,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    interleaved: bool,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_rope_gather(
            q.as_device_ptr_mut(),
            k.as_device_ptr_mut(),
            cos_cache.as_device_ptr(),
            sin_cache.as_device_ptr(),
            positions.as_device_ptr() as *const i32,
            num_tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            interleaved as libc::c_int,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// RoPE backward: rotates gradients by the conjugate rotation.
#[allow(clippy::too_many_arguments)]
pub fn rope_backward(
    grad_q: impl AsDevicePtrMut,
    grad_k: impl AsDevicePtrMut,
    cos: impl AsDevicePtr,
    sin: impl AsDevicePtr,
    num_tokens: i64,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    interleaved: bool,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_rope_backward(
            grad_q.as_device_ptr_mut(),
            grad_k.as_device_ptr_mut(),
            cos.as_device_ptr(),
            sin.as_device_ptr(),
            num_tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            interleaved as libc::c_int,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// quant.h
// ===========================================================================

/// Dynamic FP8 quantization. `scale` is fp32 (per-tensor `[1]` or per-token `[rows]`).
#[allow(clippy::too_many_arguments)]
pub fn quantize_fp8(
    out: impl AsDevicePtrMut,
    scale: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    rows: i64,
    cols: i64,
    in_dtype: Dtype,
    fp8_dtype: Dtype,
    mode: QuantMode,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_quantize_fp8(
            out.as_device_ptr_mut(),
            scale.as_device_ptr_mut() as *mut f32,
            input.as_device_ptr(),
            rows,
            cols,
            in_dtype.into(),
            fp8_dtype.into(),
            mode.into(),
            stream.as_raw(),
        )
    })
}

/// Dequantize FP8 to `out_dtype`.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_fp8(
    out: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    scale: impl AsDevicePtr,
    rows: i64,
    cols: i64,
    out_dtype: Dtype,
    fp8_dtype: Dtype,
    mode: QuantMode,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_dequantize_fp8(
            out.as_device_ptr_mut(),
            input.as_device_ptr(),
            scale.as_device_ptr() as *const f32,
            rows,
            cols,
            out_dtype.into(),
            fp8_dtype.into(),
            mode.into(),
            stream.as_raw(),
        )
    })
}

/// Dynamic symmetric INT8 quantization (out int8 + fp32 scale).
#[allow(clippy::too_many_arguments)]
pub fn quantize_int8(
    out: impl AsDevicePtrMut,
    scale: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    rows: i64,
    cols: i64,
    in_dtype: Dtype,
    mode: QuantMode,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_quantize_int8(
            out.as_device_ptr_mut(),
            scale.as_device_ptr_mut() as *mut f32,
            input.as_device_ptr(),
            rows,
            cols,
            in_dtype.into(),
            mode.into(),
            stream.as_raw(),
        )
    })
}

/// Dequantize INT8 to `out_dtype`.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_int8(
    out: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    scale: impl AsDevicePtr,
    rows: i64,
    cols: i64,
    out_dtype: Dtype,
    mode: QuantMode,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_dequantize_int8(
            out.as_device_ptr_mut(),
            input.as_device_ptr(),
            scale.as_device_ptr() as *const f32,
            rows,
            cols,
            out_dtype.into(),
            mode.into(),
            stream.as_raw(),
        )
    })
}

/// Dequantize group-wise packed INT4 weights (AWQ/GPTQ) to `out_dtype`.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_int4(
    out: impl AsDevicePtrMut,
    qweight_packed: impl AsDevicePtr,
    scales: impl AsDevicePtr,
    zeros: impl AsDevicePtr,
    k: i64,
    n: i64,
    group_size: i32,
    out_dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_dequantize_int4(
            out.as_device_ptr_mut(),
            qweight_packed.as_device_ptr(),
            scales.as_device_ptr(),
            zeros.as_device_ptr(),
            k,
            n,
            group_size,
            out_dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// sampling.h
// ===========================================================================

/// `out = softmax(input / temperature)` along the last dim.
pub fn softmax(
    out: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    rows: i64,
    cols: i64,
    temperature: f32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_softmax(
            out.as_device_ptr_mut(),
            input.as_device_ptr(),
            rows,
            cols,
            temperature,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// `out = log_softmax(input)` along the last dim.
pub fn log_softmax(
    out: impl AsDevicePtrMut,
    input: impl AsDevicePtr,
    rows: i64,
    cols: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_log_softmax(
            out.as_device_ptr_mut(),
            input.as_device_ptr(),
            rows,
            cols,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Greedy argmax: `out_tokens[s] = argmax_v logits[s, v]`.
pub fn argmax(
    out_tokens: impl AsDevicePtrMut,
    logits: impl AsDevicePtr,
    num_seqs: i64,
    vocab_size: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_argmax(
            out_tokens.as_device_ptr_mut() as *mut i32,
            logits.as_device_ptr(),
            num_seqs,
            vocab_size,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Combined temperature + top-k + top-p sampling.
///
/// `temperatures`/`top_ks`/`top_ps`/`out_probs` may be NULL device pointers.
#[allow(clippy::too_many_arguments)]
pub fn sample(
    out_tokens: impl AsDevicePtrMut,
    out_probs: impl AsDevicePtrMut,
    logits: impl AsDevicePtr,
    temperatures: impl AsDevicePtr,
    top_ks: impl AsDevicePtr,
    top_ps: impl AsDevicePtr,
    num_seqs: i64,
    vocab_size: i64,
    seed: u64,
    philox_offset: u64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_sample(
            out_tokens.as_device_ptr_mut() as *mut i32,
            out_probs.as_device_ptr_mut() as *mut f32,
            logits.as_device_ptr(),
            temperatures.as_device_ptr() as *const f32,
            top_ks.as_device_ptr() as *const i32,
            top_ps.as_device_ptr() as *const f32,
            num_seqs,
            vocab_size,
            seed,
            philox_offset,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// embedding.h
// ===========================================================================

/// `out[i, :] = table[indices[i], :]`. `indices_i64` selects int64 indices.
#[allow(clippy::too_many_arguments)]
pub fn embedding_lookup(
    out: impl AsDevicePtrMut,
    table: impl AsDevicePtr,
    indices: impl AsDevicePtr,
    indices_i64: bool,
    num_tokens: i64,
    embed_dim: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_embedding_lookup(
            out.as_device_ptr_mut(),
            table.as_device_ptr(),
            indices.as_device_ptr(),
            indices_i64 as libc::c_int,
            num_tokens,
            embed_dim,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Embedding backward: atomic scatter-add into fp32 `grad_table`.
#[allow(clippy::too_many_arguments)]
pub fn embedding_backward(
    grad_table_fp32: impl AsDevicePtrMut,
    grad_out: impl AsDevicePtr,
    indices: impl AsDevicePtr,
    indices_i64: bool,
    num_tokens: i64,
    embed_dim: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_embedding_backward(
            grad_table_fp32.as_device_ptr_mut(),
            grad_out.as_device_ptr(),
            indices.as_device_ptr(),
            indices_i64 as libc::c_int,
            num_tokens,
            embed_dim,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// elementwise.h
// ===========================================================================

/// `out = a + b` over `n` elements.
pub fn add(
    out: impl AsDevicePtrMut,
    a: impl AsDevicePtr,
    b: impl AsDevicePtr,
    n: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_add(
            out.as_device_ptr_mut(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            n,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// `out = a * b` (Hadamard) over `n` elements.
pub fn mul(
    out: impl AsDevicePtrMut,
    a: impl AsDevicePtr,
    b: impl AsDevicePtr,
    n: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_mul(
            out.as_device_ptr_mut(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            n,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// In-place residual accumulation: `residual <- residual + x`.
pub fn add_residual(
    residual: impl AsDevicePtrMut,
    x: impl AsDevicePtr,
    n: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_add_residual(
            residual.as_device_ptr_mut(),
            x.as_device_ptr(),
            n,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// `out = x * scale` over `n` elements.
pub fn scale(
    out: impl AsDevicePtrMut,
    x: impl AsDevicePtr,
    scale: f32,
    n: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_scale(
            out.as_device_ptr_mut(),
            x.as_device_ptr(),
            scale,
            n,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// `out[i] = cast<dst>(in[i])` — dtype conversion over `n` elements.
pub fn cast(
    out: impl AsDevicePtrMut,
    dst_dtype: Dtype,
    input: impl AsDevicePtr,
    src_dtype: Dtype,
    n: i64,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_cast(
            out.as_device_ptr_mut(),
            dst_dtype.into(),
            input.as_device_ptr(),
            src_dtype.into(),
            n,
            stream.as_raw(),
        )
    })
}

/// `out = a*alpha + b*beta` (fused AXPBY) over `n` elements.
#[allow(clippy::too_many_arguments)]
pub fn axpby(
    out: impl AsDevicePtrMut,
    a: impl AsDevicePtr,
    alpha: f32,
    b: impl AsDevicePtr,
    beta: f32,
    n: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_axpby(
            out.as_device_ptr_mut(),
            a.as_device_ptr(),
            alpha,
            b.as_device_ptr(),
            beta,
            n,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// loss.h
// ===========================================================================

/// Fused cross-entropy forward+backward. `grad_logits` may alias `logits`.
#[allow(clippy::too_many_arguments)]
pub fn cross_entropy(
    losses: impl AsDevicePtrMut,
    grad_logits: impl AsDevicePtrMut,
    logits: impl AsDevicePtr,
    targets: impl AsDevicePtr,
    targets_i64: bool,
    num_tokens: i64,
    vocab: i64,
    ignore_index: i64,
    label_smoothing: f32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_cross_entropy(
            losses.as_device_ptr_mut() as *mut f32,
            grad_logits.as_device_ptr_mut(),
            logits.as_device_ptr(),
            targets.as_device_ptr(),
            targets_i64 as libc::c_int,
            num_tokens,
            vocab,
            ignore_index,
            label_smoothing,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Fused-linear-cross-entropy from hidden states and the LM head weight.
#[allow(clippy::too_many_arguments)]
pub fn fused_linear_cross_entropy(
    losses: impl AsDevicePtrMut,
    grad_hidden: impl AsDevicePtrMut,
    grad_weight_fp32: impl AsDevicePtrMut,
    hidden: impl AsDevicePtr,
    weight: impl AsDevicePtr,
    targets: impl AsDevicePtr,
    targets_i64: bool,
    num_tokens: i64,
    hidden_dim: i64,
    vocab: i64,
    ignore_index: i64,
    label_smoothing: f32,
    chunk_size: i32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_fused_linear_cross_entropy(
            losses.as_device_ptr_mut() as *mut f32,
            grad_hidden.as_device_ptr_mut(),
            grad_weight_fp32.as_device_ptr_mut(),
            hidden.as_device_ptr(),
            weight.as_device_ptr(),
            targets.as_device_ptr(),
            targets_i64 as libc::c_int,
            num_tokens,
            hidden_dim,
            vocab,
            ignore_index,
            label_smoothing,
            chunk_size,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// optimizer.h
// ===========================================================================

/// AdamW (decoupled weight decay) fused step.
///
/// `master_param` may be [`DevicePtrMut::NULL`]; `step` is 1-based.
#[allow(clippy::too_many_arguments)]
pub fn adamw(
    param: impl AsDevicePtrMut,
    master_param: impl AsDevicePtrMut,
    grad: impl AsDevicePtr,
    exp_avg: impl AsDevicePtrMut,
    exp_avg_sq: impl AsDevicePtrMut,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: i64,
    grad_scale: f32,
    n: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_adamw(
            param.as_device_ptr_mut(),
            master_param.as_device_ptr_mut(),
            grad.as_device_ptr(),
            exp_avg.as_device_ptr_mut() as *mut f32,
            exp_avg_sq.as_device_ptr_mut() as *mut f32,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            grad_scale,
            n,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// SGD with optional Nesterov momentum and weight decay.
#[allow(clippy::too_many_arguments)]
pub fn sgd_momentum(
    param: impl AsDevicePtrMut,
    master_param: impl AsDevicePtrMut,
    grad: impl AsDevicePtr,
    momentum: impl AsDevicePtrMut,
    lr: f32,
    momentum_factor: f32,
    weight_decay: f32,
    nesterov: bool,
    grad_scale: f32,
    n: i64,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_sgd_momentum(
            param.as_device_ptr_mut(),
            master_param.as_device_ptr_mut(),
            grad.as_device_ptr(),
            momentum.as_device_ptr_mut() as *mut f32,
            lr,
            momentum_factor,
            weight_decay,
            nesterov as libc::c_int,
            grad_scale,
            n,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Compute the global L2 norm of a set of grad tensors (for clipping).
///
/// `grads` is a host slice of [`DevicePtr`] handles; `sizes` are their element
/// counts. `out_norm` is a fp32 `[1]` **device** scalar.
pub fn global_grad_norm(
    out_norm: impl AsDevicePtrMut,
    grads: &[DevicePtr],
    sizes: &[i64],
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    assert_eq!(
        grads.len(),
        sizes.len(),
        "global_grad_norm: grads.len() ({}) != sizes.len() ({})",
        grads.len(),
        sizes.len()
    );
    // The C ABI wants a host array of `const void*` pointers. DevicePtr is
    // #[repr(transparent)] over *const c_void, so the slice is layout-compatible.
    let grads_ptr = grads.as_ptr() as *const *const c_void;
    check(unsafe {
        sys::ks_global_grad_norm(
            out_norm.as_device_ptr_mut() as *mut f32,
            grads_ptr,
            sizes.as_ptr(),
            grads.len() as libc::c_int,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// ssm.h
// ===========================================================================

/// Depthwise causal 1-D convolution (+ optional SiLU), Mamba `(B, D, L)` layout.
///
/// `x`/`out` are `[batch, dim, seqlen]` and `weight` is `[dim, width]` in `dtype`.
/// `bias` is an fp32 `[dim]` device pointer or [`DevicePtr::NULL`]. When `silu`
/// is true a SiLU activation is applied to the convolution result.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d(
    out: impl AsDevicePtrMut,
    x: impl AsDevicePtr,
    weight: impl AsDevicePtr,
    bias: impl AsDevicePtr,
    batch: i32,
    dim: i32,
    seqlen: i32,
    width: i32,
    silu: bool,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_causal_conv1d(
            out.as_device_ptr_mut(),
            x.as_device_ptr(),
            weight.as_device_ptr(),
            bias.as_device_ptr(),
            batch,
            dim,
            seqlen,
            width,
            silu as libc::c_int,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Mamba selective scan (forward) over `[batch, dim, seqlen]` activations.
///
/// `A` (`[dim, dstate]`), `D` (`[dim]`) and `dt_bias` (`[dim]`) are fp32 device
/// pointers; `D`/`z`/`dt_bias` may be [`DevicePtr::NULL`]. `delta_softplus`
/// enables the softplus on `dt`. `B`/`C` are `[batch, dstate, seqlen]`.
#[allow(clippy::too_many_arguments)]
pub fn selective_scan(
    out: impl AsDevicePtrMut,
    x: impl AsDevicePtr,
    dt: impl AsDevicePtr,
    a: impl AsDevicePtr,
    b: impl AsDevicePtr,
    c: impl AsDevicePtr,
    d: impl AsDevicePtr,
    z: impl AsDevicePtr,
    dt_bias: impl AsDevicePtr,
    delta_softplus: bool,
    batch: i32,
    dim: i32,
    seqlen: i32,
    dstate: i32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_selective_scan(
            out.as_device_ptr_mut(),
            x.as_device_ptr(),
            dt.as_device_ptr(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            c.as_device_ptr(),
            d.as_device_ptr(),
            z.as_device_ptr(),
            dt_bias.as_device_ptr(),
            delta_softplus as libc::c_int,
            batch,
            dim,
            seqlen,
            dstate,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

/// Single-step selective-scan decode update (seqlen == 1).
///
/// Reads and writes the recurrent `state` (`[batch, dim, dstate]`, fp32) in
/// place. `x`/`dt`/`out` are `[batch, dim]`; `B`/`C` are `[batch, dstate]`. `A`,
/// `D` and `dt_bias` are fp32 device pointers; `D`/`z`/`dt_bias` may be
/// [`DevicePtr::NULL`].
#[allow(clippy::too_many_arguments)]
pub fn selective_scan_update(
    state: impl AsDevicePtrMut,
    out: impl AsDevicePtrMut,
    x: impl AsDevicePtr,
    dt: impl AsDevicePtr,
    a: impl AsDevicePtr,
    b: impl AsDevicePtr,
    c: impl AsDevicePtr,
    d: impl AsDevicePtr,
    z: impl AsDevicePtr,
    dt_bias: impl AsDevicePtr,
    delta_softplus: bool,
    batch: i32,
    dim: i32,
    dstate: i32,
    dtype: Dtype,
    stream: Stream,
) -> Result<()> {
    check(unsafe {
        sys::ks_selective_scan_update(
            state.as_device_ptr_mut(),
            out.as_device_ptr_mut(),
            x.as_device_ptr(),
            dt.as_device_ptr(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            c.as_device_ptr(),
            d.as_device_ptr(),
            z.as_device_ptr(),
            dt_bias.as_device_ptr(),
            delta_softplus as libc::c_int,
            batch,
            dim,
            dstate,
            dtype.into(),
            stream.as_raw(),
        )
    })
}

// ===========================================================================
// Tests (host-only; do not require a GPU or even the linked library to compile).
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_roundtrips() {
        // Dtype -> sys -> discriminant sanity.
        assert_eq!(sys::ks_dtype_t::from(Dtype::Bf16) as i32, 2);
        assert_eq!(sys::ks_dtype_t::from(Dtype::I4) as i32, 10);
        assert_eq!(sys::ks_activation_t::from(Activation::Silu) as i32, 4);
        assert_eq!(sys::ks_quant_mode_t::from(QuantMode::Groupwise) as i32, 3);
        assert_eq!(
            sys::ks_memcpy_kind_t::from(MemcpyKind::DeviceToDevice) as i32,
            2
        );
    }

    #[test]
    fn status_mapping() {
        assert_eq!(
            Status::from(sys::ks_status_t::KS_ERROR_OUT_OF_MEMORY),
            Status::OutOfMemory
        );
        assert_eq!(
            Status::Success.try_into_raw(),
            Some(sys::ks_status_t::KS_SUCCESS)
        );
        assert_eq!(Status::Unknown(99).try_into_raw(), None);
    }

    #[test]
    fn device_ptr_helpers() {
        let p = DevicePtrMut::from_addr(0x1000);
        assert_eq!(p.addr(), 0x1000);
        assert!(!p.is_null());
        assert_eq!(p.as_const().addr(), 0x1000);
        assert!(DevicePtr::NULL.is_null());
        let off = p.byte_offset(0x10);
        assert_eq!(off.addr(), 0x1010);
    }

    #[test]
    fn stream_default_is_null() {
        assert!(Stream::default().as_raw().is_null());
        assert_eq!(Stream::DEFAULT, Stream(core::ptr::null_mut()));
    }

    #[test]
    fn device_props_repr_size() {
        // name buffer must be exactly 256 bytes.
        assert_eq!(core::mem::size_of::<[libc::c_char; 256]>(), 256);
        // Ensure DevicePtr is pointer-sized (needed for the grad_norm cast).
        assert_eq!(
            core::mem::size_of::<DevicePtr>(),
            core::mem::size_of::<*const c_void>()
        );
    }
}
