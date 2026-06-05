// kernel-set — gated delta-rule linear attention (gated DeltaNet + Kimi-Delta)
// portable recurrent fallback. See include/kernel_set/linear_attn.h.
//
// Correctness-first O(T) recurrence (the chunked WY-transform fast path is the
// FLA provider). One block per (batch, head); each thread owns one state column
// vv of S[K,V] (columns independent). Per step, per head:
//   decay = exp(g)  (scalar g_is_vector=0, or per-k diagonal g_is_vector=1)
//   S = Diag(decay) S
//   u_t = v_t - S^T k_t                 (delta error, using the decayed S)
//   S = S + beta_t * k_t u_t^T          (tied-key rank-1 write)
//   o_t = (scale * q_t)^T S
// Optional per-(token,head) L2-norm of q,k (use_qk_l2norm).
#include "kernel_set/linear_attn.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace linear_attn {

template <typename scalar_t>
KS_GLOBAL void gated_delta_kernel(scalar_t* __restrict__ out,
                                  const scalar_t* __restrict__ q,
                                  const scalar_t* __restrict__ k,
                                  const scalar_t* __restrict__ v,
                                  const scalar_t* __restrict__ g,
                                  const scalar_t* __restrict__ beta,
                                  float* __restrict__ Sg, int64_t T, int H,
                                  int K, int V, int g_is_vector,
                                  int use_qk_l2norm, float scale) {
  const int bh = blockIdx.x;
  const int h = bh % H;
  const int64_t b = bh / H;
  float* S = Sg + static_cast<int64_t>(bh) * K * V;
  extern __shared__ float sh[];
  float* sq = sh;          // [K]
  float* sk = sh + K;      // [K]
  float* sd = sh + 2 * K;  // [K] decay
  float* scl = sh + 3 * K; // [2] norms
  const float qscale = scale > 0.f ? scale : 1.0f;
  const int tid = threadIdx.x, nthr = blockDim.x;

  for (int64_t t = 0; t < T; ++t) {
    const int64_t bth = (b * T + t) * H + h;  // [B,T,H,*] base
    for (int i = tid; i < K; i += nthr) {
      sq[i] = to_float(q[bth * K + i]);
      sk[i] = to_float(k[bth * K + i]);
      sd[i] = expf(g_is_vector ? to_float(g[bth * K + i]) : to_float(g[bth]));
    }
    __syncthreads();
    if (use_qk_l2norm) {
      if (tid == 0) {
        float nq = 0.f, nk = 0.f;
        for (int i = 0; i < K; ++i) {
          nq += sq[i] * sq[i];
          nk += sk[i] * sk[i];
        }
        scl[0] = rsqrtf(nq + 1e-12f);
        scl[1] = rsqrtf(nk + 1e-12f);
      }
      __syncthreads();
      const float iq = scl[0], ik = scl[1];
      for (int i = tid; i < K; i += nthr) {
        sq[i] *= iq;
        sk[i] *= ik;
      }
      __syncthreads();
    }
    const float beta_t = to_float(beta[bth]);
    for (int vv = tid; vv < V; vv += nthr) {
      const float vval = to_float(v[bth * V + vv]);
      // 1) apply decay, 2) compute k^T (decayed S) for this column.
      float kdot = 0.f;
      for (int kk = 0; kk < K; ++kk) {
        const int64_t idx = static_cast<int64_t>(kk) * V + vv;
        const float s = sd[kk] * S[idx];
        S[idx] = s;
        kdot += sk[kk] * s;
      }
      const float u = vval - kdot;  // delta error
      // 3) rank-1 write, 4) output q^T S.
      float o = 0.f;
      for (int kk = 0; kk < K; ++kk) {
        const int64_t idx = static_cast<int64_t>(kk) * V + vv;
        const float s = S[idx] + beta_t * sk[kk] * u;
        S[idx] = s;
        o += sq[kk] * qscale * s;
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

ks_status_t ks_gated_delta_rule(void* out, const void* q, const void* k,
                                const void* v, const void* g, const void* beta,
                                int64_t batch, int64_t seqlen, int64_t heads,
                                int64_t k_dim, int64_t v_dim, int g_is_vector,
                                int use_qk_l2norm, float scale, ks_dtype_t dtype,
                                ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(q);
  KS_CHECK_PTR(k);
  KS_CHECK_PTR(v);
  KS_CHECK_PTR(g);
  KS_CHECK_PTR(beta);
  if (batch <= 0 || seqlen <= 0 || heads <= 0 || k_dim <= 0 || v_dim <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_gated_delta_rule: bad shape");

  const int64_t bh = batch * heads;
  constexpr int64_t kI64Max = 9223372036854775807LL;
  if (v_dim > kI64Max / k_dim || bh > kI64Max / (k_dim * v_dim))
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_gated_delta_rule: state size exceeds int64 range");
  const size_t nstate = static_cast<size_t>(bh) * k_dim * v_dim;

  float* S = nullptr;
  if (ks::gpuMalloc(reinterpret_cast<void**>(&S), nstate * sizeof(float)) !=
      ks::gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_OUT_OF_MEMORY,
                    "ks_gated_delta_rule: state scratch alloc failed");
  auto s = to_stream(stream);
  ks::gpuMemsetAsync(S, 0, nstate * sizeof(float), s);

  const unsigned block = 256;
  const dim3 grid(static_cast<unsigned>(bh));
  const size_t smem = static_cast<size_t>(3 * k_dim + 2) * sizeof(float);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_gated_delta_rule", {
    linear_attn::gated_delta_kernel<scalar_t><<<grid, block, smem, s>>>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(q),
        static_cast<const scalar_t*>(k), static_cast<const scalar_t*>(v),
        static_cast<const scalar_t*>(g), static_cast<const scalar_t*>(beta), S,
        seqlen, static_cast<int>(heads), static_cast<int>(k_dim),
        static_cast<int>(v_dim), g_is_vector, use_qk_l2norm, scale);
  });
  ks::gpuStreamSynchronize(s);
  ks::gpuFree(S);
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
