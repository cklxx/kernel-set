// kernel-set — Mamba selective (data-dependent) state-space scan.
//
// One CUDA block owns one (batch, channel=dim) pair and runs the full sequential
// recurrence over the L positions, exactly as the math in ssm.h. Within a block
// the threads partition the `dstate` hidden dimension: thread `j` owns states
// n = j, j+blockDim, ... and keeps each h[n] live in a register across the whole
// scan. The only cross-thread communication is the per-step output projection
//   y_t = sum_n C_t[n] * h_t[n]
// which is a block reduction over the state axis.
//
// Per time step t (all fp32):
//   dt' = dt[t] (+ dt_bias[d]);  dt' = softplus(dt') if requested
//   dA_n = exp(dt' * A[d,n])
//   h_n  = dA_n * h_n + dt' * B_t[n] * x_t           (state update)
//   y   += C_t[n] * h_n                              (output, reduced over n)
//   y_t  = y + D[d] * x_t ;  if z: y_t *= silu(z_t)
//
// h_{-1} = 0. B and C are shared across all `dim` channels (single-group
// selective scan), laid out [batch, dstate, seqlen] with seqlen contiguous, so
// they are reloaded per channel-block but read coalesced along t.
//
// Accumulation is fp32 throughout (storage dtype converted via ks::to_float /
// ks::from_float<scalar_t>); A / D / dt_bias are fp32 device buffers per the
// ssm.h dtype policy.
//
// Optimized-path note: this is the numerically-faithful *sequential* scan — its
// latency is O(L) per block. The documented perf upgrade is the chunked
// parallel scan (Blelloch / Heinsen associative scan): factor the recurrence
// into the associative pair (a_t, b_t) = (exp(dt'*A), dt'*B*x) and run a CUB
// block-wide InclusiveScan with SSMScanOp((a0,b0),(a1,b1)) = (a1*a0, a1*b0+b1)
// across L tiles, which is what the upstream Mamba kernel does. Correctness is
// identical; only the parallelization over L changes.
#include "kernel_set/ssm.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"

namespace ks {
namespace ssm {

constexpr int kScanBlock = 128;
// Largest selective-scan state we support (mirrors the reference MAX_DSTATE).
constexpr int kMaxDState = 256;
// With kScanBlock threads, each thread owns at most this many states.
constexpr int kMaxStatesPerThread =
    (kMaxDState + kScanBlock - 1) / kScanBlock;  // = 2

// Numerically-guarded softplus, matching the reference: pass through for large
// inputs to avoid exp overflow, otherwise log1p(exp(v)).
KS_DI float softplus_f(float v) {
  return v > 20.f ? v : log1pf(__expf(v));
}

// x * sigmoid(x); logistic written via the negative-exp branch for stability.
KS_DI float silu_f(float x) { return x / (1.f + __expf(-x)); }

// grid = (batch, dim); block = kScanBlock threads cooperating over dstate.
template <typename scalar_t>
KS_GLOBAL void selective_scan_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ dt, const float* __restrict__ A,
    const scalar_t* __restrict__ B, const scalar_t* __restrict__ C,
    const float* __restrict__ D, const scalar_t* __restrict__ z,
    const float* __restrict__ dt_bias, bool delta_softplus, int dim,
    int seqlen, int dstate) {
  const int batch_id = blockIdx.x;
  const int channel_id = blockIdx.y;  // == dim index
  const int tid = threadIdx.x;

  // Sequence rows for this (batch, channel): [batch, dim, seqlen], L contiguous.
  const int64_t seq_row =
      (static_cast<int64_t>(batch_id) * dim + channel_id) *
      static_cast<int64_t>(seqlen);
  const scalar_t* xrow = x + seq_row;
  const scalar_t* dtrow = dt + seq_row;
  scalar_t* yrow = out + seq_row;
  const scalar_t* zrow = (z != nullptr) ? z + seq_row : nullptr;

  // A: [dim, dstate], contiguous over dstate.
  const float* Arow = A + static_cast<int64_t>(channel_id) * dstate;
  const float d_val = (D != nullptr) ? D[channel_id] : 0.f;
  const float dt_bias_val = (dt_bias != nullptr) ? dt_bias[channel_id] : 0.f;

  // B, C: [batch, dstate, seqlen], L contiguous, shared across `dim`.
  const int64_t bc_base = static_cast<int64_t>(batch_id) * dstate * seqlen;

  // Each thread owns states n = tid, tid + blockDim, ... Hold A[n] and the
  // running h[n] in registers for the whole scan.
  float A_local[kMaxStatesPerThread];
  float h_local[kMaxStatesPerThread];
#pragma unroll
  for (int s = 0; s < kMaxStatesPerThread; ++s) {
    const int n = tid + s * blockDim.x;
    A_local[s] = (n < dstate) ? Arow[n] : 0.f;
    h_local[s] = 0.f;  // h_{-1} = 0
  }

  __shared__ float smem[kScanBlock / KS_WARP_SIZE];

  for (int t = 0; t < seqlen; ++t) {
    const float x_val = to_float(xrow[t]);
    float dtp = to_float(dtrow[t]) + dt_bias_val;
    if (delta_softplus) dtp = softplus_f(dtp);
    const float dt_x = dtp * x_val;  // dt' * x_t, shared across states

    // State update + partial output projection over this thread's states.
    float y_partial = 0.f;
#pragma unroll
    for (int s = 0; s < kMaxStatesPerThread; ++s) {
      const int n = tid + s * blockDim.x;
      if (n < dstate) {
        const float b_val =
            to_float(B[bc_base + static_cast<int64_t>(n) * seqlen + t]);
        const float c_val =
            to_float(C[bc_base + static_cast<int64_t>(n) * seqlen + t]);
        const float dA = __expf(dtp * A_local[s]);
        const float h = dA * h_local[s] + dt_x * b_val;
        h_local[s] = h;
        y_partial += c_val * h;
      }
    }

    // y_t = sum_n C_t[n] * h_t[n] + D[d] * x_t, reduced across the block.
    float y = block_reduce_sum(y_partial, smem);
    if (tid == 0) {
      y += d_val * x_val;
      if (zrow != nullptr) y *= silu_f(to_float(zrow[t]));
      yrow[t] = from_float<scalar_t>(y);
    }
    __syncthreads();  // protect smem reuse before the next step's reduction
  }
}

// Single-step decode update (one position). grid = (batch, dim); block
// cooperates over dstate as above, but the state lives in the `state` buffer
// ([batch, dim, dstate], fp32) instead of registers, so it persists across
// decode calls.
template <typename scalar_t>
KS_GLOBAL void selective_scan_update_kernel(
    float* __restrict__ state, scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ dt,
    const float* __restrict__ A, const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ C, const float* __restrict__ D,
    const scalar_t* __restrict__ z, const float* __restrict__ dt_bias,
    bool delta_softplus, int dim, int dstate) {
  const int batch_id = blockIdx.x;
  const int channel_id = blockIdx.y;
  const int tid = threadIdx.x;

  // x / dt / out / z: [batch, dim] (one step).
  const int64_t bc_chan = static_cast<int64_t>(batch_id) * dim + channel_id;
  const float x_val = to_float(x[bc_chan]);
  float dtp =
      to_float(dt[bc_chan]) + (dt_bias != nullptr ? dt_bias[channel_id] : 0.f);
  if (delta_softplus) dtp = softplus_f(dtp);
  const float dt_x = dtp * x_val;

  // state: [batch, dim, dstate]; A: [dim, dstate]; B/C: [batch, dstate].
  float* h_row = state + (static_cast<int64_t>(batch_id) * dim + channel_id) *
                             static_cast<int64_t>(dstate);
  const float* Arow = A + static_cast<int64_t>(channel_id) * dstate;
  const int64_t bc_base = static_cast<int64_t>(batch_id) * dstate;

  float y_partial = 0.f;
  for (int n = tid; n < dstate; n += blockDim.x) {
    const float b_val = to_float(B[bc_base + n]);
    const float c_val = to_float(C[bc_base + n]);
    const float dA = __expf(dtp * Arow[n]);
    const float h = dA * h_row[n] + dt_x * b_val;
    h_row[n] = h;  // persist updated state
    y_partial += c_val * h;
  }

  __shared__ float smem[kScanBlock / KS_WARP_SIZE];
  float y = block_reduce_sum(y_partial, smem);
  if (tid == 0) {
    y += (D != nullptr ? D[channel_id] : 0.f) * x_val;
    if (z != nullptr) y *= silu_f(to_float(z[bc_chan]));
    out[bc_chan] = from_float<scalar_t>(y);
  }
}

}  // namespace ssm
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_selective_scan(void* out, const void* x, const void* dt,
                              const void* A, const void* B, const void* C,
                              const void* D, const void* z, const void* dt_bias,
                              int delta_softplus, int batch, int dim, int seqlen,
                              int dstate, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(x);
  KS_CHECK_PTR(dt);
  KS_CHECK_PTR(A);
  KS_CHECK_PTR(B);
  KS_CHECK_PTR(C);
  // D, z, dt_bias are optional (may be NULL).
  if (batch <= 0 || dim <= 0 || seqlen <= 0 || dstate <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_selective_scan: bad shape");
  if (dstate > ssm::kMaxDState)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_selective_scan: dstate exceeds supported maximum");
  // grid = (batch, dim); both axes are bounded by 65535 on supported arches.
  if (static_cast<unsigned>(batch) > 65535u || static_cast<unsigned>(dim) > 65535u)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_selective_scan: batch/dim exceeds grid limit");

  const dim3 block(ssm::kScanBlock);
  const dim3 grid(static_cast<unsigned>(batch), static_cast<unsigned>(dim));
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_selective_scan", {
    ssm::selective_scan_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(x),
        static_cast<const scalar_t*>(dt), static_cast<const float*>(A),
        static_cast<const scalar_t*>(B), static_cast<const scalar_t*>(C),
        static_cast<const float*>(D), static_cast<const scalar_t*>(z),
        static_cast<const float*>(dt_bias), delta_softplus != 0, dim, seqlen,
        dstate);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_selective_scan_update(void* state, void* out, const void* x,
                                     const void* dt, const void* A,
                                     const void* B, const void* C,
                                     const void* D, const void* z,
                                     const void* dt_bias, int delta_softplus,
                                     int batch, int dim, int dstate,
                                     ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(state);
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(x);
  KS_CHECK_PTR(dt);
  KS_CHECK_PTR(A);
  KS_CHECK_PTR(B);
  KS_CHECK_PTR(C);
  // D, z, dt_bias are optional (may be NULL).
  if (batch <= 0 || dim <= 0 || dstate <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_selective_scan_update: bad shape");
  if (dstate > ssm::kMaxDState)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_selective_scan_update: dstate exceeds supported maximum");
  // grid = (batch, dim); both axes are bounded by 65535 on supported arches.
  if (static_cast<unsigned>(batch) > 65535u || static_cast<unsigned>(dim) > 65535u)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_selective_scan_update: batch/dim exceeds grid limit");

  const dim3 block(ssm::kScanBlock);
  const dim3 grid(static_cast<unsigned>(batch), static_cast<unsigned>(dim));
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_selective_scan_update", {
    ssm::selective_scan_update_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<float*>(state), static_cast<scalar_t*>(out),
        static_cast<const scalar_t*>(x), static_cast<const scalar_t*>(dt),
        static_cast<const float*>(A), static_cast<const scalar_t*>(B),
        static_cast<const scalar_t*>(C), static_cast<const float*>(D),
        static_cast<const scalar_t*>(z), static_cast<const float*>(dt_bias),
        delta_softplus != 0, dim, dstate);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
