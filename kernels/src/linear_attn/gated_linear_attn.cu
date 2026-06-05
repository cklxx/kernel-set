// kernel-set — gated linear attention (GLA / simple-GLA / lightning) portable
// recurrent fallback. See include/kernel_set/linear_attn.h.
//
// Correctness-first O(T) recurrence (NOT the chunked-parallel fast path — that
// is the FLA provider). One thread-block per (batch, head); each thread owns one
// output column vv of the per-head state S[K,V] (columns are independent, so no
// intra-timestep races). The recurrence over t is sequential. State S is held in
// fp32 global scratch; per-timestep vectors are staged in shared memory.
//
//   S = Diag(alpha_t) S + v_t k_t^T ;   o_t = (scale * q_t)^T S
#include "kernel_set/linear_attn.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace linear_attn {

template <typename scalar_t>
KS_GLOBAL void gla_kernel(scalar_t* __restrict__ out,
                          const scalar_t* __restrict__ q,
                          const scalar_t* __restrict__ k,
                          const scalar_t* __restrict__ v,
                          const scalar_t* __restrict__ g,
                          const float* __restrict__ head_decay,
                          float* __restrict__ Sg, int64_t T, int H, int K, int V,
                          int gate_mode, float scale) {
  const int bh = blockIdx.x;
  const int h = bh % H;
  const int64_t b = bh / H;
  float* S = Sg + static_cast<int64_t>(bh) * K * V;
  extern __shared__ float sh[];
  float* sq = sh;           // [K]
  float* sk = sh + K;       // [K]
  float* sa = sh + 2 * K;   // [K] alpha
  const float qscale = scale > 0.f ? scale : 1.0f;
  const int tid = threadIdx.x, nthr = blockDim.x;

  for (int64_t t = 0; t < T; ++t) {
    const int64_t bth = (b * T + t) * H + h;  // [B,T,H,*] row base
    for (int i = tid; i < K; i += nthr) {
      sq[i] = to_float(q[bth * K + i]);
      sk[i] = to_float(k[bth * K + i]);
      if (gate_mode == 0)
        sa[i] = expf(to_float(g[bth * K + i]));        // data-dep diagonal
      else if (gate_mode == 1)
        sa[i] = expf(to_float(g[bth]));                // scalar per-head
      else
        sa[i] = expf(-(head_decay ? head_decay[h] : 0.f));  // fixed slope
    }
    __syncthreads();
    for (int vv = tid; vv < V; vv += nthr) {
      const float vval = to_float(v[bth * V + vv]);
      float o = 0.f;
      for (int kk = 0; kk < K; ++kk) {
        const int64_t idx = static_cast<int64_t>(kk) * V + vv;
        const float s = sa[kk] * S[idx] + sk[kk] * vval;
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

ks_status_t ks_gated_linear_attn(void* out, const void* q, const void* k,
                                 const void* v, const void* g,
                                 const float* head_decay, int64_t batch,
                                 int64_t seqlen, int64_t heads, int64_t k_dim,
                                 int64_t v_dim, int gate_mode, float scale,
                                 ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(q);
  KS_CHECK_PTR(k);
  KS_CHECK_PTR(v);
  if (batch <= 0 || seqlen <= 0 || heads <= 0 || k_dim <= 0 || v_dim <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_gated_linear_attn: bad shape");
  if ((gate_mode == 0 || gate_mode == 1) && g == nullptr)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_gated_linear_attn: g required for gate_mode 0/1");
  if (gate_mode == 2 && head_decay == nullptr)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_gated_linear_attn: head_decay required for gate_mode 2");

  const int64_t bh = batch * heads;
  constexpr int64_t kI64Max = 9223372036854775807LL;
  if (v_dim > kI64Max / k_dim || bh > kI64Max / (k_dim * v_dim))
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_gated_linear_attn: state size exceeds int64 range");
  const size_t nstate = static_cast<size_t>(bh) * k_dim * v_dim;

  float* S = nullptr;
  if (ks::gpuMalloc(reinterpret_cast<void**>(&S), nstate * sizeof(float)) !=
      ks::gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_OUT_OF_MEMORY,
                    "ks_gated_linear_attn: state scratch alloc failed");
  auto s = to_stream(stream);
  ks::gpuMemsetAsync(S, 0, nstate * sizeof(float), s);

  const unsigned block = 256;
  const dim3 grid(static_cast<unsigned>(bh));
  const size_t smem = static_cast<size_t>(3) * k_dim * sizeof(float);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_gated_linear_attn", {
    linear_attn::gla_kernel<scalar_t><<<grid, block, smem, s>>>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(q),
        static_cast<const scalar_t*>(k), static_cast<const scalar_t*>(v),
        static_cast<const scalar_t*>(g), head_decay, S, seqlen,
        static_cast<int>(heads), static_cast<int>(k_dim),
        static_cast<int>(v_dim), gate_mode, scale);
  });
  ks::gpuStreamSynchronize(s);
  ks::gpuFree(S);
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
