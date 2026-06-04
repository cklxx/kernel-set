// kernel-set — implementation of runtime.h + shared error plumbing.
#include <string>

#include "kernel_set/runtime.h"
#include "common/platform.cuh"
#include "common/dispatch.cuh"

#ifndef KERNEL_SET_VERSION
#define KERNEL_SET_VERSION "0.1.0"
#endif

namespace ks {

// Thread-local last error backing ks_last_error_string().
static thread_local std::string g_last_error;

const char* last_error_cstr() { return g_last_error.c_str(); }
void set_last_error(const std::string& msg) { g_last_error = msg; }

}  // namespace ks

using ks::check;
using ks::to_stream;

extern "C" {

const char* ks_version(void) { return KERNEL_SET_VERSION; }

const char* ks_backend_name(void) {
#if defined(KS_PLATFORM_HIP)
  return "hip";
#elif defined(KS_PLATFORM_CUDA)
  return "cuda";
#else
  return "none";
#endif
}

const char* ks_last_error_string(void) { return ks::last_error_cstr(); }

const char* ks_status_string(ks_status_t status) {
  switch (status) {
    case KS_SUCCESS: return "KS_SUCCESS";
    case KS_ERROR_INVALID_ARGUMENT: return "KS_ERROR_INVALID_ARGUMENT";
    case KS_ERROR_UNSUPPORTED_DTYPE: return "KS_ERROR_UNSUPPORTED_DTYPE";
    case KS_ERROR_UNSUPPORTED_SHAPE: return "KS_ERROR_UNSUPPORTED_SHAPE";
    case KS_ERROR_CUDA: return "KS_ERROR_CUDA";
    case KS_ERROR_NOT_IMPLEMENTED: return "KS_ERROR_NOT_IMPLEMENTED";
    case KS_ERROR_OUT_OF_MEMORY: return "KS_ERROR_OUT_OF_MEMORY";
    case KS_ERROR_ARCH_UNSUPPORTED: return "KS_ERROR_ARCH_UNSUPPORTED";
    case KS_ERROR_INTERNAL: return "KS_ERROR_INTERNAL";
    default: return "KS_ERROR_UNKNOWN";
  }
}

int ks_dtype_size_bits(ks_dtype_t dtype) {
  switch (dtype) {
    case KS_DTYPE_F64:
    case KS_DTYPE_I64: return 64;
    case KS_DTYPE_F32:
    case KS_DTYPE_I32: return 32;
    case KS_DTYPE_F16:
    case KS_DTYPE_BF16: return 16;
    case KS_DTYPE_F8E4M3:
    case KS_DTYPE_F8E5M2:
    case KS_DTYPE_I8:
    case KS_DTYPE_U8: return 8;
    case KS_DTYPE_I4: return 4;
    default: return 0;
  }
}

const char* ks_dtype_name(ks_dtype_t dtype) {
  switch (dtype) {
    case KS_DTYPE_F32: return "f32";
    case KS_DTYPE_F16: return "f16";
    case KS_DTYPE_BF16: return "bf16";
    case KS_DTYPE_F8E4M3: return "f8e4m3";
    case KS_DTYPE_F8E5M2: return "f8e5m2";
    case KS_DTYPE_F64: return "f64";
    case KS_DTYPE_I64: return "i64";
    case KS_DTYPE_I32: return "i32";
    case KS_DTYPE_I8: return "i8";
    case KS_DTYPE_U8: return "u8";
    case KS_DTYPE_I4: return "i4";
    default: return "unknown";
  }
}

ks_status_t ks_device_count(int* out_count) {
  KS_CHECK_PTR(out_count);
  return check(ks::gpuGetDeviceCount(out_count), "ks_device_count");
}

ks_status_t ks_set_device(int device) {
  return check(ks::gpuSetDevice(device), "ks_set_device");
}

ks_status_t ks_get_device(int* out_device) {
  KS_CHECK_PTR(out_device);
  return check(ks::gpuGetDevice(out_device), "ks_get_device");
}

ks_status_t ks_get_device_properties(int device,
                                     ks_device_properties_t* out_props) {
  KS_CHECK_PTR(out_props);
  ks::gpuDeviceProp_t prop;
  ks_status_t s = check(ks::gpuGetDeviceProperties(&prop, device),
                        "ks_get_device_properties");
  if (s != KS_SUCCESS) return s;

  for (int i = 0; i < 256; ++i) {
    out_props->name[i] = prop.name[i];
    if (prop.name[i] == '\0') break;
  }
  out_props->name[255] = '\0';
  out_props->compute_major = prop.major;
  out_props->compute_minor = prop.minor;
  out_props->multiprocessor_count = prop.multiProcessorCount;
  out_props->max_threads_per_block = prop.maxThreadsPerBlock;
  out_props->max_shared_memory_per_block =
      static_cast<int>(prop.sharedMemPerBlock);
  out_props->warp_size = prop.warpSize;
  out_props->total_global_memory = prop.totalGlobalMem;
  const int sm = prop.major * 10 + prop.minor;
  out_props->supports_bf16 = (sm >= 80) ? 1 : 0;
  out_props->supports_fp8 = (sm >= 89) ? 1 : 0;
  out_props->supports_tf32 = (sm >= 80) ? 1 : 0;
  return KS_SUCCESS;
}

ks_status_t ks_stream_create(ks_stream_t* out_stream) {
  KS_CHECK_PTR(out_stream);
  ks::gpuStream_t s;
  ks_status_t st = check(ks::gpuStreamCreate(&s), "ks_stream_create");
  if (st != KS_SUCCESS) return st;
  *out_stream = static_cast<ks_stream_t>(s);
  return KS_SUCCESS;
}

ks_status_t ks_stream_destroy(ks_stream_t stream) {
  return check(ks::gpuStreamDestroy(to_stream(stream)), "ks_stream_destroy");
}

ks_status_t ks_stream_synchronize(ks_stream_t stream) {
  return check(ks::gpuStreamSynchronize(to_stream(stream)),
               "ks_stream_synchronize");
}

ks_status_t ks_malloc_device(void** out_ptr, size_t bytes) {
  KS_CHECK_PTR(out_ptr);
  ks::gpuError_t e = ks::gpuMalloc(out_ptr, bytes);
  if (e != ks::gpuSuccess) {
    ks::set_last_error(std::string("ks_malloc_device: ") +
                       ks::gpuGetErrorString(e));
    return KS_ERROR_OUT_OF_MEMORY;
  }
  return KS_SUCCESS;
}

ks_status_t ks_free_device(void* ptr) {
  return check(ks::gpuFree(ptr), "ks_free_device");
}

ks_status_t ks_memcpy(void* dst, const void* src, size_t bytes,
                      ks_memcpy_kind_t kind, ks_stream_t stream) {
  KS_CHECK_PTR(dst);
  KS_CHECK_PTR(src);
  auto k = ks::gpuMemcpyDeviceToDevice;
  if (kind == KS_MEMCPY_HOST_TO_DEVICE) k = ks::gpuMemcpyHostToDevice;
  else if (kind == KS_MEMCPY_DEVICE_TO_HOST) k = ks::gpuMemcpyDeviceToHost;
  return check(ks::gpuMemcpyAsync(dst, src, bytes, k, to_stream(stream)),
               "ks_memcpy");
}

ks_status_t ks_memset_device(void* dst, int value, size_t bytes,
                             ks_stream_t stream) {
  KS_CHECK_PTR(dst);
  return check(ks::gpuMemsetAsync(dst, value, bytes, to_stream(stream)),
               "ks_memset_device");
}

}  // extern "C"
