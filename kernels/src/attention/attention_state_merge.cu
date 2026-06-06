// kernel-set — merge two partial attention states by log-sum-exp.
//
// The associative combine that lets KV be split across kernels/chunks/devices
// (cascade / chunked-prefill / ring attention). For each row r and value dim j:
//   m   = max(lse_a, lse_b)
//   wa  = exp(lse_a - m);  wb = exp(lse_b - m)
//   out = (out_a*wa + out_b*wb) / (wa + wb)
//   lse = m + log(wa + wb)
// out/out_a/out_b: [n_rows, v_dim] (model dtype); lse*: [n_rows] fp32.
// `out` may alias out_a and `lse` may alias lse_a.
#include "kernel_set/attention.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace attention {

constexpr int kMergeBlock = 256;

template <typename scalar_t>
KS_GLOBAL void state_merge_kernel(scalar_t* __restrict__ out,
                                  float* __restrict__ lse,
                                  const scalar_t* __restrict__ out_a,
                                  const float* __restrict__ lse_a,
                                  const scalar_t* __restrict__ out_b,
                                  const float* __restrict__ lse_b,
                                  int64_t v_dim) {
  const int64_t row = blockIdx.x;
  const float la = lse_a[row];
  const float lb = lse_b[row];
  const float m = fmaxf(la, lb);
  // Both -inf (empty) => avoid NaN: emit 0 / -inf.
  const bool empty = !isfinite(m);
  const float wa = empty ? 0.f : __expf(la - m);
  const float wb = empty ? 0.f : __expf(lb - m);
  const float denom = wa + wb;
  const float inv = denom > 0.f ? 1.f / denom : 0.f;

  const scalar_t* a = out_a + row * v_dim;
  const scalar_t* b = out_b + row * v_dim;
  scalar_t* y = out + row * v_dim;
  for (int64_t j = threadIdx.x; j < v_dim; j += blockDim.x)
    y[j] = from_float<scalar_t>((to_float(a[j]) * wa + to_float(b[j]) * wb) * inv);

  if (threadIdx.x == 0)
    lse[row] = empty ? -INFINITY : (m + __logf(denom));
}

}  // namespace attention
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_attention_state_merge(void* out, float* lse, const void* out_a,
                                     const float* lse_a, const void* out_b,
                                     const float* lse_b, int64_t n_rows,
                                     int64_t v_dim, ks_dtype_t dtype,
                                     ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(lse);
  KS_CHECK_PTR(out_a);
  KS_CHECK_PTR(lse_a);
  KS_CHECK_PTR(out_b);
  KS_CHECK_PTR(lse_b);
  if (n_rows <= 0 || v_dim <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_attention_state_merge: shape");
  if (n_rows > 2147483647LL)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_attention_state_merge: n_rows > grid limit");

  const dim3 grid(static_cast<unsigned>(n_rows));
  const dim3 block(attention::kMergeBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_attention_state_merge", {
    attention::state_merge_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(out), lse, static_cast<const scalar_t*>(out_a),
        lse_a, static_cast<const scalar_t*>(out_b), lse_b, v_dim);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
