// kernel-set — softmax, log_softmax, and argmax over the last dim.
//
// House style mirrored from norm/rms_norm.cu:
//   * one CUDA block per row, blockDim.x threads cooperatively reduce the row
//   * fp32 accumulation regardless of storage dtype (online max + sum)
//   * a single templated kernel serves f32/f16/bf16 via KS_DISPATCH_FLOATING_TYPES
//
// We use the numerically stable two-statistic form: subtract the row max before
// exp so no term overflows, then divide by the exp-sum (the standard "online"
// safe-softmax; see Milakov & Gimelshein "Online normalizer calculation for
// softmax"). Both statistics are produced with a block_reduce so a row of any
// width is handled by a fixed-size block.
#include "kernel_set/sampling.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"

namespace ks {
namespace sampling {

constexpr int kBlock = 256;

// out = softmax(input * inv_temp) along a row of `cols`.
template <typename scalar_t>
KS_GLOBAL void softmax_kernel(scalar_t* __restrict__ out,
                              const scalar_t* __restrict__ input, int64_t cols,
                              float inv_temp) {
  const int64_t row = blockIdx.x;
  const scalar_t* x = input + row * cols;
  scalar_t* y = out + row * cols;

  __shared__ float smem[kBlock / KS_WARP_SIZE];

  // Pass 1: row max of the temperature-scaled logits.
  float local_max = -3.4028235e38f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(x[i]) * inv_temp;
    local_max = v > local_max ? v : local_max;
  }
  const float row_max = block_reduce_max(local_max, smem);
  __syncthreads();

  // Pass 2: sum of exp(v - row_max).
  float local_sum = 0.f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(x[i]) * inv_temp;
    local_sum += __expf(v - row_max);
  }
  const float row_sum = block_reduce_sum(local_sum, smem);
  const float inv_sum = 1.0f / row_sum;

  // Pass 3: normalize.
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(x[i]) * inv_temp;
    y[i] = from_float<scalar_t>(__expf(v - row_max) * inv_sum);
  }
}

// out = log_softmax(input) = (input - row_max) - log(sum exp(input - row_max)).
template <typename scalar_t>
KS_GLOBAL void log_softmax_kernel(scalar_t* __restrict__ out,
                                  const scalar_t* __restrict__ input,
                                  int64_t cols) {
  const int64_t row = blockIdx.x;
  const scalar_t* x = input + row * cols;
  scalar_t* y = out + row * cols;

  __shared__ float smem[kBlock / KS_WARP_SIZE];

  float local_max = -3.4028235e38f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(x[i]);
    local_max = v > local_max ? v : local_max;
  }
  const float row_max = block_reduce_max(local_max, smem);
  __syncthreads();

  float local_sum = 0.f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    local_sum += __expf(to_float(x[i]) - row_max);
  }
  const float row_sum = block_reduce_sum(local_sum, smem);
  const float log_denom = row_max + logf(row_sum);  // logsumexp

  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    y[i] = from_float<scalar_t>(to_float(x[i]) - log_denom);
  }
}

// out_tokens[row] = argmax_v input[row, v]. Ties resolved to the lowest index
// (matches PyTorch/NumPy argmax semantics) by keeping the smaller index when
// values are equal during reduction.
template <typename scalar_t>
KS_GLOBAL void argmax_kernel(int32_t* __restrict__ out_tokens,
                             const scalar_t* __restrict__ input, int64_t cols) {
  const int64_t row = blockIdx.x;
  const scalar_t* x = input + row * cols;

  float local_max = -3.4028235e38f;
  int local_idx = 0;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(x[i]);
    if (v > local_max) {
      local_max = v;
      local_idx = static_cast<int>(i);
    }
  }

  // Block-wide argmax: reduce (value, index) pairs, breaking ties to lower idx.
  __shared__ float s_val[kBlock];
  __shared__ int s_idx[kBlock];
  s_val[threadIdx.x] = local_max;
  s_idx[threadIdx.x] = local_idx;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float ov = s_val[threadIdx.x + stride];
      const int oi = s_idx[threadIdx.x + stride];
      const float cv = s_val[threadIdx.x];
      const int ci = s_idx[threadIdx.x];
      if (ov > cv || (ov == cv && oi < ci)) {
        s_val[threadIdx.x] = ov;
        s_idx[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) out_tokens[row] = s_idx[0];
}

}  // namespace sampling
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_softmax(void* out, const void* input, int64_t rows, int64_t cols,
                       float temperature, ks_dtype_t dtype,
                       ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_softmax: bad shape");

  // temperature <= 0 => identity scaling (per the ABI contract).
  const float inv_temp = (temperature > 0.f) ? (1.0f / temperature) : 1.0f;

  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(sampling::kBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_softmax", {
    sampling::softmax_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(input), cols,
        inv_temp);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_log_softmax(void* out, const void* input, int64_t rows,
                           int64_t cols, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_log_softmax: bad shape");

  const dim3 grid(static_cast<unsigned>(rows));
  const dim3 block(sampling::kBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_log_softmax", {
    sampling::log_softmax_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(input), cols);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_argmax(int32_t* out_tokens, const void* logits,
                      int64_t num_seqs, int64_t vocab_size, ks_dtype_t dtype,
                      ks_stream_t stream) {
  KS_CHECK_PTR(out_tokens);
  KS_CHECK_PTR(logits);
  if (num_seqs <= 0 || vocab_size <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_argmax: bad shape");

  const dim3 grid(static_cast<unsigned>(num_seqs));
  const dim3 block(sampling::kBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_argmax", {
    sampling::argmax_kernel<scalar_t><<<grid, block, 0, s>>>(
        out_tokens, static_cast<const scalar_t*>(logits), vocab_size);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
