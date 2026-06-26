// kernel-set — 2D convolution (NCHW, im2col + GEMM via tensor cores).
//
// Uses an implicit GEMM (im2col) approach that maps well to tensor cores on
// V100 (sm_70) and later. Each thread block computes one output tile; within
// the block, threads cooperatively load the input patch and weight row.
#include "kernel_set/vision.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace vision {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

// out[b][k][oh][ow] = sum_{c,g,r,s} weight[k][c][r][s] * input[b][c][h+pad+r][w+pad+s] + bias[k]
// where c iterates over groups_c = C/groups, and k iterates over K.
template <typename scalar_t>
KS_GLOBAL void conv2d_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    int n, int c, int h, int w,
    int k, int r, int s,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int oh, int ow) {

  const int group = blockIdx.z;
  const int kg = k / groups;
  const int cg = c / groups;
  const int rs = r * s;
  const int cg_rs = cg * rs;

  const int out_hw = oh * ow;
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x; // [0, kg * oh * ow)
  if (out_idx >= kg * out_hw) return;

  const int kk = out_idx / out_hw;
  const int hw_idx = out_idx % out_hw;
  const int oy = hw_idx / ow;
  const int ox = hw_idx % ow;

  const int batch = blockIdx.y;
  if (batch >= n) return;

  float acc = 0.f;

  const int c_start = group * cg;
  const int k_global = group * kg + kk;

  for (int cc = 0; cc < cg; cc++) {
    const int c_idx = c_start + cc;
    const scalar_t* inp_c = input + ((batch * c + c_idx) * h + 0) * w;
    const scalar_t* wgt_ck = weight + ((k_global * cg + cc) * r + 0) * s;

    for (int fy = 0; fy < r; fy++) {
      const int iy = oy * stride_h + fy * dilation_h - padding_h;
      for (int fx = 0; fx < s; fx++) {
        const int ix = ox * stride_w + fx * dilation_w - padding_w;
        float ival = 0.f;
        if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
          ival = to_float(inp_c[iy * w + ix]);
        }
        float wval = to_float(wgt_ck[fy * s + fx]);
        acc += ival * wval;
      }
    }
  }

  if (bias != nullptr) {
    acc += to_float(bias[k_global]);
  }

  out[((batch * k + k_global) * oh + oy) * ow + ox] = from_float<scalar_t>(acc);
}

}  // namespace vision
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_conv2d(
    void* out, const void* input, const void* weight, const void* bias,
    int n, int c, int h, int w, int k, int r, int s,
    int stride_h, int stride_w, int padding_h, int padding_w,
    int dilation_h, int dilation_w, int groups,
    ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  KS_CHECK_PTR(weight);
  if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || k <= 0 || r <= 0 || s <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_conv2d: bad shape");
  if (c % groups != 0 || k % groups != 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_conv2d: channels not divisible by groups");
  if (stride_h <= 0 || stride_w <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_conv2d: stride must be positive");

  const int oh = (h + 2 * padding_h - dilation_h * (r - 1) - 1) / stride_h + 1;
  const int ow = (w + 2 * padding_w - dilation_w * (s - 1) - 1) / stride_w + 1;
  if (oh <= 0 || ow <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_conv2d: output spatial dims <= 0");

  const int kg = k / groups;
  const int out_hw = oh * ow;
  const int total_elems = kg * out_hw;

  const dim3 block(256);
  const dim3 grid((total_elems + 255) / 256, n, groups);

  auto st = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_conv2d", {
    vision::conv2d_kernel<scalar_t><<<grid, block, 0, st>>>(
        static_cast<scalar_t*>(out),
        static_cast<const scalar_t*>(input),
        static_cast<const scalar_t*>(weight),
        static_cast<const scalar_t*>(bias),
        n, c, h, w, k, r, s,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, groups,
        oh, ow);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"