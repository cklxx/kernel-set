/*
 * kernel-set — runtime / device management ABI.
 *
 * Thin, vendor-neutral wrappers over the GPU runtime so that language bindings
 * which have no GPU library of their own (Go, raw Rust, Node) can still query
 * devices, allocate device memory, move data, and create streams. Tensor-aware
 * callers (PyTorch, candle, tch-rs) typically skip these and pass their own
 * device pointers + streams straight into the kernel entry points.
 */
#ifndef KERNEL_SET_RUNTIME_H_
#define KERNEL_SET_RUNTIME_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* ---- Library / build introspection ------------------------------------- */

/* Semantic version string, e.g. "0.1.0". */
KS_API const char* ks_version(void);

/* Human-readable name for a status code (never NULL). */
KS_API const char* ks_status_string(ks_status_t status);

/* Element size in bits (e.g. KS_DTYPE_I4 -> 4, KS_DTYPE_F32 -> 32). */
KS_API int ks_dtype_size_bits(ks_dtype_t dtype);

/* Short token for a dtype, e.g. "bf16" (never NULL). */
KS_API const char* ks_dtype_name(ks_dtype_t dtype);

/* The backend this build targets: "cuda", "hip", or "none". */
KS_API const char* ks_backend_name(void);

/* ---- Device queries ----------------------------------------------------- */

typedef struct ks_device_properties_t {
  char name[256];
  int compute_major;        /* CUDA SM major / HIP gcnArch major              */
  int compute_minor;
  int multiprocessor_count;
  int max_threads_per_block;
  int max_shared_memory_per_block;   /* bytes (opt-in/static limit)           */
  int warp_size;
  size_t total_global_memory;        /* bytes                                 */
  int supports_bf16;
  int supports_fp8;                  /* e4m3/e5m2 tensor-core support          */
  int supports_tf32;
} ks_device_properties_t;

KS_API ks_status_t ks_device_count(int* out_count);
KS_API ks_status_t ks_set_device(int device);
KS_API ks_status_t ks_get_device(int* out_device);
KS_API ks_status_t ks_get_device_properties(int device,
                                            ks_device_properties_t* out_props);

/* Returns a thread-local string describing the last backend runtime error.
 * Valid until the next kernel-set call on the same thread. Never NULL. */
KS_API const char* ks_last_error_string(void);

/* ---- Streams ------------------------------------------------------------ */

KS_API ks_status_t ks_stream_create(ks_stream_t* out_stream);
KS_API ks_status_t ks_stream_destroy(ks_stream_t stream);
KS_API ks_status_t ks_stream_synchronize(ks_stream_t stream);

/* ---- Device memory (convenience for bindings without a tensor lib) ------ */

KS_API ks_status_t ks_malloc_device(void** out_ptr, size_t bytes);
KS_API ks_status_t ks_free_device(void* ptr);

typedef enum ks_memcpy_kind_t {
  KS_MEMCPY_HOST_TO_DEVICE = 0,
  KS_MEMCPY_DEVICE_TO_HOST = 1,
  KS_MEMCPY_DEVICE_TO_DEVICE = 2,
} ks_memcpy_kind_t;

KS_API ks_status_t ks_memcpy(void* dst, const void* src, size_t bytes,
                             ks_memcpy_kind_t kind, ks_stream_t stream);
KS_API ks_status_t ks_memset_device(void* dst, int value, size_t bytes,
                                    ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_RUNTIME_H_ */
