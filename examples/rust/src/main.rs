//! kernel-set Rust example: RMSNorm end to end.
//!
//! Allocates device buffers with the built-in allocator, uploads bf16 input and
//! weight, calls `ks::rms_norm` (`ks_rms_norm`), copies the result back, and
//! prints it.
//!
//! Run (point the build at the shared library):
//! ```sh
//! cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j
//! KERNEL_SET_LIB="$PWD/build" cargo run --manifest-path examples/rust/Cargo.toml
//! ```

use kernel_set::{self as ks, DeviceBuffer, Dtype, Stream};

/// Round-trip f32 -> bf16 by truncating the low 16 mantissa/exponent bits.
fn f32_to_bf16_bits(x: f32) -> u16 {
    // Round-to-nearest-even bf16 from an f32 bit pattern.
    let bits = x.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn main() -> Result<(), ks::Error> {
    println!("kernel-set {} ({} backend)", ks::version(), ks::backend_name());

    let (rows, cols) = (4i64, 8i64);
    let eps = 1e-5f32;
    let n = (rows * cols) as usize;
    let bytes = n * Dtype::Bf16.size_bytes();

    // Host data: a simple ramp for x, all-ones weight.
    let x_host: Vec<u16> = (0..n).map(|i| f32_to_bf16_bits(i as f32 * 0.1)).collect();
    let w_host: Vec<u16> = (0..cols as usize).map(|_| f32_to_bf16_bits(1.0)).collect();

    // Device buffers (freed on drop).
    let x = DeviceBuffer::new(bytes)?;
    let w = DeviceBuffer::for_elems(cols as usize, Dtype::Bf16)?;
    let out = DeviceBuffer::new(bytes)?;

    let stream = Stream::DEFAULT; // null == default stream
    x.copy_from_host(&x_host, stream)?;
    w.copy_from_host(&w_host, stream)?;

    // The kernel call.
    ks::rms_norm(out.ptr_mut(), x.ptr(), w.ptr(), rows, cols, eps, Dtype::Bf16, stream)?;
    ks::stream_synchronize(stream)?;

    // Copy the result back and print row 0.
    let mut out_host = vec![0u16; n];
    out.copy_to_host(&mut out_host, stream)?;
    ks::stream_synchronize(stream)?;

    let row0: Vec<f32> = out_host[..cols as usize].iter().map(|&b| bf16_bits_to_f32(b)).collect();
    println!("rms_norm(out)[0] = {row0:?}");
    Ok(())
}
