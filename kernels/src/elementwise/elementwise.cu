// kernel-set — pointwise elementwise ops: add, mul, add_residual, scale, axpby.
//
// Every op is a thin host wrapper around the shared grid-stride launchers in
// elementwise_common.cuh. They process VecWidth<scalar_t> elements per thread
// via 128-bit loads/stores when alignment permits and fall back to a scalar
// tail (and a scalar path for unaligned buffers). All arithmetic accumulates in
// fp32 through ks::to_float / ks::from_float, exactly like norm/rms_norm.cu.
//
// Further tuning (documented, not yet applied): for very large tensors a
// vectorized path with VEC*ILP unrolling and cp.async-staged prefetch can edge
// closer to peak HBM bandwidth; for these memory-bound ops the single wide
// load/store loop already hits the roofline on sm_80/89/90.
#include "kernel_set/elementwise.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/vec.cuh"
#include "elementwise/elementwise_common.cuh"

using namespace ks;

extern "C" {

ks_status_t ks_add(void* out, const void* a, const void* b, int64_t n,
                   ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(a);
  KS_CHECK_PTR(b);
  if (n < 0) KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_add: n < 0");
  if (n == 0) return KS_SUCCESS;

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_add", {
    elementwise::launch_binary<scalar_t>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(a),
        static_cast<const scalar_t*>(b), n, elementwise::AddOp{}, s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_mul(void* out, const void* a, const void* b, int64_t n,
                   ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(a);
  KS_CHECK_PTR(b);
  if (n < 0) KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_mul: n < 0");
  if (n == 0) return KS_SUCCESS;

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_mul", {
    elementwise::launch_binary<scalar_t>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(a),
        static_cast<const scalar_t*>(b), n, elementwise::MulOp{}, s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_add_residual(void* residual, const void* x, int64_t n,
                            ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(residual);
  KS_CHECK_PTR(x);
  if (n < 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_add_residual: n < 0");
  if (n == 0) return KS_SUCCESS;

  auto s = to_stream(stream);
  // In-place: residual <- residual + x. The in-place launcher uses kernels that
  // do not mark `dst` __restrict__, so the read-modify-write is well-defined.
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_add_residual", {
    elementwise::launch_inplace<scalar_t>(
        static_cast<scalar_t*>(residual), static_cast<const scalar_t*>(x), n,
        elementwise::AddOp{}, s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_scale(void* out, const void* x, float scale, int64_t n,
                     ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(x);
  if (n < 0) KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_scale: n < 0");
  if (n == 0) return KS_SUCCESS;

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_scale", {
    elementwise::launch_unary<scalar_t>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(x), n,
        elementwise::ScaleOp{scale}, s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_axpby(void* out, const void* a, float alpha, const void* b,
                     float beta, int64_t n, ks_dtype_t dtype,
                     ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(a);
  KS_CHECK_PTR(b);
  if (n < 0) KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_axpby: n < 0");
  if (n == 0) return KS_SUCCESS;

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_axpby", {
    elementwise::launch_binary<scalar_t>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(a),
        static_cast<const scalar_t*>(b), n, elementwise::AxpbyOp{alpha, beta},
        s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
