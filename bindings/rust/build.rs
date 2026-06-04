//! Build script for the `kernel-set` crate.
//!
//! Responsibilities:
//!   * Tell Cargo to link the system shared library `kernel_set`
//!     (libkernel_set.so / libkernel_set.dylib / kernel_set.dll).
//!   * Add a library search path from the `KERNEL_SET_LIB` environment variable
//!     (a directory, or the full path to the shared object). This is honored at
//!     both build/link time and — via an emitted rpath on Unix — at run time.
//!
//! Override knobs (environment variables, read at build time):
//!
//! * `KERNEL_SET_LIB` — directory containing the shared lib, OR the full path to
//!   the shared object file itself.
//! * `KERNEL_SET_NO_LINK` — if set (to anything), skip emitting the link
//!   directive entirely. Useful when you `dlopen` the library yourself or only
//!   need the type/enum definitions from `sys`.
//!
//! docs.rs builds with `--cfg docsrs` and has no library to link, so linking is
//! skipped there automatically.

use std::env;
use std::path::Path;

fn main() {
    // Re-run if any of our knobs change.
    println!("cargo:rerun-if-env-changed=KERNEL_SET_LIB");
    println!("cargo:rerun-if-env-changed=KERNEL_SET_NO_LINK");
    println!("cargo:rerun-if-changed=build.rs");

    // Expose the resolved search dir to dependent crates as DEP_KERNEL_SET_LIB.
    // (Cargo prefixes these with `DEP_<links>_` for crates with `links`.)
    if let Some(dir) = resolve_search_dir() {
        println!("cargo:rustc-link-search=native={dir}");
        // Bake an rpath so the produced binary/test can find the lib at runtime
        // without requiring LD_LIBRARY_PATH / DYLD_LIBRARY_PATH to be set.
        emit_rpath(&dir);
        // Make the directory discoverable by downstream build scripts.
        println!("cargo:lib={dir}");
    }

    // docs.rs (and anyone passing --cfg docsrs) can't link the native lib.
    let docsrs = env::var_os("CARGO_CFG_DOCSRS").is_some()
        || env::var_os("DOCS_RS").is_some()
        || cfg_docsrs();

    let skip_link = env::var_os("KERNEL_SET_NO_LINK").is_some() || docsrs;

    if skip_link {
        println!("cargo:warning=kernel-set: skipping link of `kernel_set` (docs.rs or KERNEL_SET_NO_LINK).");
        return;
    }

    // Link the shared library. The base name is `kernel_set`; the platform
    // toolchain maps it to libkernel_set.{so,dylib} / kernel_set.dll|.lib.
    println!("cargo:rustc-link-lib=dylib=kernel_set");
}

/// Resolve the directory to add to the link search path from `KERNEL_SET_LIB`.
/// Accepts either a directory or a full path to the shared object file.
fn resolve_search_dir() -> Option<String> {
    let raw = env::var("KERNEL_SET_LIB").ok()?;
    if raw.is_empty() {
        return None;
    }
    let p = Path::new(&raw);
    if p.is_file() {
        // Full path to the .so/.dylib/.dll — search its parent directory.
        p.parent()
            .map(|d| d.to_string_lossy().into_owned())
            .or(Some(raw))
    } else {
        // Treat as a directory (it need not exist yet at build time).
        Some(raw)
    }
}

/// Emit a platform-appropriate rpath linker arg so the runtime loader can find
/// libkernel_set next to where we linked it.
fn emit_rpath(dir: &str) {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" | "ios" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
        "windows" => {
            // Windows has no rpath; the DLL must be on PATH or beside the .exe.
            // Nothing to emit here.
        }
        _ => {
            // Linux / *BSD / etc.
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
}

/// Best-effort detection of the `docsrs` cfg passed through to the build script.
fn cfg_docsrs() -> bool {
    // Cargo does not forward arbitrary --cfg to build scripts, but docs.rs sets
    // DOCS_RS; this is a defensive secondary check.
    env::var_os("CARGO_CFG_DOCSRS").is_some()
}
