// kernel-set — runtime dtype dispatch + status/error plumbing.
//
// KS_DISPATCH_FLOATING_TYPES turns a runtime ks_dtype_t into a compile-time
// scalar type bound to `scalar_t` inside the body, so a single templated kernel
// launcher serves f32/f16/bf16. Errors from the backend runtime are funneled
// through ks::set_last_error / ks::check so the C ABI can return ks_status_t and
// expose ks_last_error_string().
#pragma once

#include <string>

#include "kernel_set/types.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"

namespace ks {

// Thread-local last-error message backing ks_last_error_string().
const char* last_error_cstr();
void set_last_error(const std::string& msg);

// Translate a backend error into a ks_status_t, recording the message.
inline ks_status_t check(gpuError_t e, const char* where) {
  if (e == gpuSuccess) return KS_SUCCESS;
  set_last_error(std::string(where) + ": " + gpuGetErrorString(e));
  return KS_ERROR_CUDA;
}

}  // namespace ks

// Record a message and return a status from a C ABI function.
#define KS_RETURN_ERROR(status, msg) \
  do {                               \
    ::ks::set_last_error(msg);       \
    return (status);                 \
  } while (0)

// Validate a pointer argument.
#define KS_CHECK_PTR(ptr)                                            \
  do {                                                               \
    if ((ptr) == nullptr)                                            \
      KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "null arg: " #ptr); \
  } while (0)

// Check a kernel launch + return on failure.
#define KS_CHECK_LAUNCH()                                          \
  do {                                                            \
    ks_status_t _s = ::ks::check(::ks::gpuGetLastError(), __func__); \
    if (_s != KS_SUCCESS) return _s;                              \
  } while (0)

// Dispatch over the floating element types supported by most kernels.
#define KS_DISPATCH_FLOATING_TYPES(DTYPE, NAME, ...)                       \
  switch (DTYPE) {                                                         \
    case KS_DTYPE_F32: {                                                   \
      using scalar_t = float;                                             \
      __VA_ARGS__;                                                         \
      break;                                                              \
    }                                                                     \
    case KS_DTYPE_F16: {                                                   \
      using scalar_t = __half;                                            \
      __VA_ARGS__;                                                         \
      break;                                                              \
    }                                                                     \
    case KS_DTYPE_BF16: {                                                  \
      using scalar_t = __nv_bfloat16;                                     \
      __VA_ARGS__;                                                         \
      break;                                                              \
    }                                                                     \
    default:                                                              \
      KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,                         \
                      std::string(NAME) + ": unsupported dtype");         \
  }

// Half-precision only (f16/bf16), e.g. tensor-core paths.
#define KS_DISPATCH_HALF_TYPES(DTYPE, NAME, ...)                  \
  switch (DTYPE) {                                                \
    case KS_DTYPE_F16: {                                          \
      using scalar_t = __half;                                    \
      __VA_ARGS__;                                                \
      break;                                                      \
    }                                                             \
    case KS_DTYPE_BF16: {                                         \
      using scalar_t = __nv_bfloat16;                             \
      __VA_ARGS__;                                                \
      break;                                                      \
    }                                                             \
    default:                                                      \
      KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,                 \
                      std::string(NAME) + ": needs f16/bf16");    \
  }
