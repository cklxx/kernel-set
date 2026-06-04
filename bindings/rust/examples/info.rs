//! Print library/build info and (if a GPU is present) the first device's props.
//!
//! Run with the shared library discoverable via `KERNEL_SET_LIB`:
//!
//! ```sh
//! KERNEL_SET_LIB=/path/to/libdir cargo run --example info
//! ```

fn main() -> Result<(), kernel_set::Error> {
    println!("kernel-set version : {}", kernel_set::version());
    println!("backend            : {}", kernel_set::backend_name());

    println!(
        "bf16 size          : {} bits / {} bytes",
        kernel_set::Dtype::Bf16.size_bits(),
        kernel_set::Dtype::Bf16.size_bytes()
    );
    println!("bf16 name          : {}", kernel_set::Dtype::Bf16.name());

    match kernel_set::device_count() {
        Ok(n) => {
            println!("device count       : {n}");
            for d in 0..n {
                match kernel_set::get_device_properties(d) {
                    Ok(p) => println!(
                        "  device {d}: {} (sm{}{}), {} SMs, {:.1} GiB",
                        p.name,
                        p.compute_major,
                        p.compute_minor,
                        p.multiprocessor_count,
                        p.total_global_memory as f64 / (1024.0 * 1024.0 * 1024.0)
                    ),
                    Err(e) => println!("  device {d}: <error: {e}>"),
                }
            }
        }
        Err(e) => println!("device count       : <error: {e}>"),
    }

    Ok(())
}
