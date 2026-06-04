// kernel-set — elementwise activations: SiLU, GELU (erf + tanh), ReLU.
//
// Each entry point runs over `n` contiguous elements via the shared vectorized
// launcher (128-bit wide IO when the buffers are aligned, scalar tail otherwise)
// and accumulates in fp32 regardless of the storage dtype. See activation_ops.cuh
// for the scalar numerics; this file is just dtype dispatch + the C ABI.
#include "kernel_set/activation.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/vec.cuh"
#include "activation_ops.cuh"

using namespace ks;

extern "C" {

ks_status_t ks_silu(void* out, const void* input, int64_t n, ks_dtype_t dtype,
                    ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  if (n < 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_silu: n must be >= 0");
  if (n == 0) return KS_SUCCESS;  // nothing to do; not an error.

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_silu", {
    activation::launch_elementwise<scalar_t>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(input), n,
        activation::SiluOp{}, s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_gelu(void* out, const void* input, int64_t n, int tanh_approx,
                    ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  if (n < 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_gelu: n must be >= 0");
  if (n == 0) return KS_SUCCESS;

  auto s = to_stream(stream);
  if (tanh_approx != 0) {
    KS_DISPATCH_FLOATING_TYPES(dtype, "ks_gelu", {
      activation::launch_elementwise<scalar_t>(
          static_cast<scalar_t*>(out), static_cast<const scalar_t*>(input), n,
          activation::GeluTanhOp{}, s);
    });
  } else {
    KS_DISPATCH_FLOATING_TYPES(dtype, "ks_gelu", {
      activation::launch_elementwise<scalar_t>(
          static_cast<scalar_t*>(out), static_cast<const scalar_t*>(input), n,
          activation::GeluErfOp{}, s);
    });
  }
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_relu(void* out, const void* input, int64_t n, ks_dtype_t dtype,
                    ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  if (n < 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_relu: n must be >= 0");
  if (n == 0) return KS_SUCCESS;

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_relu", {
    activation::launch_elementwise<scalar_t>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(input), n,
        activation::ReluOp{}, s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
