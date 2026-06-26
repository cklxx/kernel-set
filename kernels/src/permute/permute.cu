#include "kernel_set/permute.h"
#include "common/dispatch.cuh"
#include <cuda_runtime.h>

namespace kernel_set {

template <typename T>
__global__ void transpose_2d_kernel(T *__restrict__ out, const T *__restrict__ in,
                                     int M, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * K;
  if (idx >= total) return;
  int row = idx / K;
  int col = idx % K;
  out[col * M + row] = in[row * K + col];
}

template <typename T>
__global__ void nchw_to_nhwc_kernel(T *__restrict__ out, const T *__restrict__ in,
                                     int C, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = C * H * W;
  if (idx >= total) return;
  int c = idx / (H * W);
  int r = idx % (H * W);
  int h = r / W;
  int w = r % W;
  out[h * (W * C) + w * C + c] = in[c * (H * W) + h * W + w];
}

template <typename T>
__global__ void nhwc_to_nchw_kernel(T *__restrict__ out, const T *__restrict__ in,
                                     int H, int W, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = H * W * C;
  if (idx >= total) return;
  int r = idx / C;
  int c = idx % C;
  int h = r / W;
  int w = r % W;
  out[c * (H * W) + h * W + w] = in[h * (W * C) + w * C + c];
}

template <typename T>
__global__ void upsample_nearest_2x_kernel(T *__restrict__ out, const T *__restrict__ in,
                                            int C, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = C * H * W;
  if (idx >= total) return;
  int c = idx / (H * W);
  int r = idx % (H * W);
  int h = r / W;
  int w = r % W;
  T val = in[c * (H * W) + h * W + w];
  int OH = 2 * H, OW = 2 * W;
  out[c * (OH * OW) + (2 * h) * OW + (2 * w)] = val;
  out[c * (OH * OW) + (2 * h) * OW + (2 * w + 1)] = val;
  out[c * (OH * OW) + (2 * h + 1) * OW + (2 * w)] = val;
  out[c * (OH * OW) + (2 * h + 1) * OW + (2 * w + 1)] = val;
}

} // namespace kernel_set

// ---- Public API ----

ks_status_t ks_transpose_2d(void *out, const void *in, int M, int K,
                             ks_dtype_t dtype, ks_stream_t stream) {
  if (!out || !in || M <= 0 || K <= 0) return KS_ERROR_INVALID_ARGUMENT;
  int total = M * K;
  int block = 256;
  int grid = (total + block - 1) / block;
  auto s = reinterpret_cast<cudaStream_t>(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_transpose_2d", {
    kernel_set::transpose_2d_kernel<<<grid, block, 0, s>>>(
        static_cast<scalar_t *>(out), static_cast<const scalar_t *>(in), M, K);
  });
  return KS_SUCCESS;
}

ks_status_t ks_nchw_to_nhwc(void *out, const void *in, int N, int C, int H, int W,
                             ks_dtype_t dtype, ks_stream_t stream) {
  if (!out || !in || N <= 0 || C <= 0 || H <= 0 || W <= 0) return KS_ERROR_INVALID_ARGUMENT;
  int total = N * C * H * W;
  int block = 256;
  int grid = (total + block - 1) / block;
  auto s = reinterpret_cast<cudaStream_t>(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_nchw_to_nhwc", {
    kernel_set::nchw_to_nhwc_kernel<<<grid, block, 0, s>>>(
        static_cast<scalar_t *>(out), static_cast<const scalar_t *>(in), C, H, W);
  });
  return KS_SUCCESS;
}

ks_status_t ks_nhwc_to_nchw(void *out, const void *in, int N, int H, int W, int C,
                             ks_dtype_t dtype, ks_stream_t stream) {
  if (!out || !in || N <= 0 || H <= 0 || W <= 0 || C <= 0) return KS_ERROR_INVALID_ARGUMENT;
  int total = N * H * W * C;
  int block = 256;
  int grid = (total + block - 1) / block;
  auto s = reinterpret_cast<cudaStream_t>(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_nhwc_to_nchw", {
    kernel_set::nhwc_to_nchw_kernel<<<grid, block, 0, s>>>(
        static_cast<scalar_t *>(out), static_cast<const scalar_t *>(in), H, W, C);
  });
  return KS_SUCCESS;
}

ks_status_t ks_upsample_nearest_2x(void *out, const void *in, int N, int C, int H, int W,
                                    ks_dtype_t dtype, ks_stream_t stream) {
  if (!out || !in || N <= 0 || C <= 0 || H <= 0 || W <= 0) return KS_ERROR_INVALID_ARGUMENT;
  int total = N * C * H * W;
  int block = 256;
  int grid = (total + block - 1) / block;
  auto s = reinterpret_cast<cudaStream_t>(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_upsample_nearest_2x", {
    kernel_set::upsample_nearest_2x_kernel<<<grid, block, 0, s>>>(
        static_cast<scalar_t *>(out), static_cast<const scalar_t *>(in), C, H, W);
  });
  return KS_SUCCESS;
}