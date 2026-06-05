// kernel-set — RWKV-7 "Goose" WKV (free DPLR generalized delta rule) portable
// recurrent fallback. See include/kernel_set/linear_attn.h.
//
// Correctness-first O(T) recurrence (the chunked DPLR fast path is the FLA
// provider). One block per (batch, head); each thread owns one state column vv
// of S[K,V]. Per step, per head:
//   S = S (Diag(exp(w_t)) - a_t b_t^T) + v_t k_t^T ;   o_t = (scale * r_t)^T S
// where the transition acts on the K dimension. For column vv:
//   aS = sum_jj a_jj S[jj,vv]                       (using the OLD state)
//   S[kk,vv] = exp(w_kk) S[kk,vv] - b_kk * aS + k_kk * v_vv
//   o_vv = sum_kk (scale*r_kk) S[kk,vv]
#include "kernel_set/linear_attn.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace linear_attn {

template <typename scalar_t>
KS_GLOBAL void rwkv7_kernel(scalar_t* __restrict__ out,
                            const scalar_t* __restrict__ r,
                            const scalar_t* __restrict__ w,
                            const scalar_t* __restrict__ k,
                            const scalar_t* __restrict__ v,
                            const scalar_t* __restrict__ a,
                            const scalar_t* __restrict__ b_in,
                            float* __restrict__ Sg, int64_t T, int H, int K,
                            int V, float scale) {
  const int bh = blockIdx.x;
  const int h = bh % H;
  const int64_t b = bh / H;
  float* S = Sg + static_cast<int64_t>(bh) * K * V;
  extern __shared__ float sh[];
  float* sr = sh;            // [K] receptance (query)
  float* sw = sh + K;        // [K] exp(w) decay
  float* sk = sh + 2 * K;    // [K]
  float* sa = sh + 3 * K;    // [K]
  float* sb = sh + 4 * K;    // [K]
  const float qscale = scale > 0.f ? scale : 1.0f;
  const int tid = threadIdx.x, nthr = blockDim.x;

  for (int64_t t = 0; t < T; ++t) {
    const int64_t bth = (b * T + t) * H + h;  // [B,T,H,*] base
    for (int i = tid; i < K; i += nthr) {
      sr[i] = to_float(r[bth * K + i]);
      sw[i] = expf(to_float(w[bth * K + i]));
      sk[i] = to_float(k[bth * K + i]);
      sa[i] = to_float(a[bth * K + i]);
      sb[i] = to_float(b_in[bth * K + i]);
    }
    __syncthreads();
    for (int vv = tid; vv < V; vv += nthr) {
      const float vval = to_float(v[bth * V + vv]);
      // aS = a^T S(:,vv) over the OLD state.
      float aS = 0.f;
      for (int kk = 0; kk < K; ++kk)
        aS += sa[kk] * S[static_cast<int64_t>(kk) * V + vv];
      // update + output.
      float o = 0.f;
      for (int kk = 0; kk < K; ++kk) {
        const int64_t idx = static_cast<int64_t>(kk) * V + vv;
        const float s = sw[kk] * S[idx] - sb[kk] * aS + sk[kk] * vval;
        S[idx] = s;
        o += sr[kk] * qscale * s;
      }
      out[bth * V + vv] = from_float<scalar_t>(o);
    }
    __syncthreads();
  }
}

}  // namespace linear_attn
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_rwkv_wkv7(void* out, const void* r, const void* w,
                         const void* k, const void* v, const void* a,
                         const void* b, int64_t batch, int64_t seqlen,
                         int64_t heads, int64_t k_dim, int64_t v_dim,
                         float scale, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(r);
  KS_CHECK_PTR(w);
  KS_CHECK_PTR(k);
  KS_CHECK_PTR(v);
  KS_CHECK_PTR(a);
  KS_CHECK_PTR(b);
  if (batch <= 0 || seqlen <= 0 || heads <= 0 || k_dim <= 0 || v_dim <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_rwkv_wkv7: bad shape");

  const int64_t bh = batch * heads;
  constexpr int64_t kI64Max = 9223372036854775807LL;
  if (v_dim > kI64Max / k_dim || bh > kI64Max / (k_dim * v_dim))
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_rwkv_wkv7: state size exceeds int64 range");
  const size_t nstate = static_cast<size_t>(bh) * k_dim * v_dim;

  float* S = nullptr;
  if (ks::gpuMalloc(reinterpret_cast<void**>(&S), nstate * sizeof(float)) !=
      ks::gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_OUT_OF_MEMORY,
                    "ks_rwkv_wkv7: state scratch alloc failed");
  auto s = to_stream(stream);
  ks::gpuMemsetAsync(S, 0, nstate * sizeof(float), s);

  const unsigned block = 256;
  const dim3 grid(static_cast<unsigned>(bh));
  const size_t smem = static_cast<size_t>(5) * k_dim * sizeof(float);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_rwkv_wkv7", {
    linear_attn::rwkv7_kernel<scalar_t><<<grid, block, smem, s>>>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(r),
        static_cast<const scalar_t*>(w), static_cast<const scalar_t*>(k),
        static_cast<const scalar_t*>(v), static_cast<const scalar_t*>(a),
        static_cast<const scalar_t*>(b), S, seqlen, static_cast<int>(heads),
        static_cast<int>(k_dim), static_cast<int>(v_dim), scale);
  });
  ks::gpuStreamSynchronize(s);
  ks::gpuFree(S);
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
