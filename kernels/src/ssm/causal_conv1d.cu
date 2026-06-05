// kernel-set — depthwise causal 1-D convolution (Mamba conv block).
//
// Each (batch, channel) pair owns an independent depthwise filter, so the kernel
// maps one thread to one (batch, channel) lane and walks the sequence axis once,
// keeping a `width`-wide sliding window of inputs in registers:
//
//   out[t] = bias + sum_{k=0..W-1} weight[k] * x[t - (W-1) + k]
//
// with implicit zero left-padding (x[<0] == 0) which makes the conv causal:
// out[t] only sees inputs at positions <= t. An optional SiLU follows.
//
// Layout is [batch, dim, seqlen] with seqlen innermost (stride 1), so the per-
// lane walk over t is a unit-stride stream — fully coalesced across the threads
// of a block, which all share the same t but differ in channel.
//
// All arithmetic is fp32 (storage dtype converted via ks::to_float /
// ks::from_float<scalar_t>), matching the house accumulation rule. `bias` is a
// fp32 buffer (see ssm.h dtype policy); weight uses the activation dtype.
//
// Optimized-path note: width is tiny (~4) and known only at runtime here, so the
// inner taps loop is a short runtime loop. The documented perf upgrade is to
// template on kWidth (BOOL/INT switch over 2/3/4) so the taps unroll fully and
// `x_window` lives in named registers; for a tile of channels one can also stage
// the shared weight row through smem. Correctness is identical either way.
#include "kernel_set/ssm.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace ssm {

constexpr int kConvBlock = 256;
// Upper bound on the conv width we keep a register window for. Mamba uses 4;
// guard the ABI so a wider request fails cleanly instead of overrunning.
constexpr int kMaxConvWidth = 8;

// One thread per (batch, channel). grid.x = batch, grid.y tiles the channels.
template <typename scalar_t>
KS_GLOBAL void causal_conv1d_kernel(scalar_t* __restrict__ out,
                                    const scalar_t* __restrict__ x,
                                    const scalar_t* __restrict__ weight,
                                    const float* __restrict__ bias, int batch,
                                    int dim, int seqlen, int width,
                                    bool silu) {
  const int batch_id = blockIdx.x;
  const int channel_id = blockIdx.y * blockDim.x + threadIdx.x;
  if (channel_id >= dim) return;

  // x / out: [batch, dim, seqlen] with seqlen contiguous.
  const int64_t row = (static_cast<int64_t>(batch_id) * dim + channel_id) *
                      static_cast<int64_t>(seqlen);
  const scalar_t* xrow = x + row;
  scalar_t* yrow = out + row;
  // weight: [dim, width], contiguous over width.
  const scalar_t* wrow = weight + static_cast<int64_t>(channel_id) * width;

  float w[kMaxConvWidth];
#pragma unroll
  for (int k = 0; k < kMaxConvWidth; ++k) {
    w[k] = (k < width) ? to_float(wrow[k]) : 0.f;
  }
  const float bias_val = (bias != nullptr) ? bias[channel_id] : 0.f;

  // Sliding window of the last `width` inputs, window[width-1] == current x[t].
  // Seeded to zero so the first (width-1) positions use causal zero-padding.
  float window[kMaxConvWidth];
#pragma unroll
  for (int k = 0; k < kMaxConvWidth; ++k) window[k] = 0.f;

  for (int t = 0; t < seqlen; ++t) {
    // Shift the window left by one and append the new sample at the head slot.
#pragma unroll
    for (int k = 0; k < kMaxConvWidth - 1; ++k) window[k] = window[k + 1];
    window[width - 1] = to_float(xrow[t]);

    float acc = bias_val;
    for (int k = 0; k < width; ++k) acc += w[k] * window[k];

    if (silu) acc = acc / (1.f + __expf(-acc));  // x * sigmoid(x)
    yrow[t] = from_float<scalar_t>(acc);
  }
}

}  // namespace ssm
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_causal_conv1d(void* out, const void* x, const void* weight,
                             const void* bias, int batch, int dim, int seqlen,
                             int width, int silu, ks_dtype_t dtype,
                             ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(x);
  KS_CHECK_PTR(weight);
  // `bias` may be NULL (no bias term).
  if (batch <= 0 || dim <= 0 || seqlen <= 0 || width <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_causal_conv1d: bad shape");
  if (width > ssm::kMaxConvWidth)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_causal_conv1d: width exceeds supported maximum");

  // The device row offset is (batch_id*dim + channel_id)*seqlen in int64; reject
  // shapes whose total element count cannot be represented so that offset (and
  // any read/write derived from it) can never overflow. batch,dim,seqlen are > 0.
  constexpr int64_t kI64Max = 9223372036854775807LL;  // INT64_MAX
  const int64_t batch64 = batch;
  const int64_t dim64 = dim;
  const int64_t seqlen64 = seqlen;
  if (dim64 > kI64Max / batch64 ||
      seqlen64 > kI64Max / (batch64 * dim64))
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_causal_conv1d: batch*dim*seqlen exceeds int64 range");

  // Compute grid.y from int64 so dim+kConvBlock-1 cannot overflow before the cast.
  const int64_t grid_y64 = (dim64 + ssm::kConvBlock - 1) / ssm::kConvBlock;
  // grid.x/grid.y are bounded by 65535 on all supported arches; fail cleanly
  // instead of letting the launch return an opaque InvalidConfiguration.
  if (batch64 > 65535 || grid_y64 > 65535)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_causal_conv1d: batch/dim exceeds grid limit");
  const unsigned grid_y = static_cast<unsigned>(grid_y64);

  const dim3 block(ssm::kConvBlock);
  const dim3 grid(static_cast<unsigned>(batch), grid_y);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_causal_conv1d", {
    ssm::causal_conv1d_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(x),
        static_cast<const scalar_t*>(weight),
        static_cast<const float*>(bias), batch, dim, seqlen, width,
        silu != 0);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
