// kernel-set — correctness test: ks_rms_norm and ks_layer_norm vs a host
// reference. Mirrors the kernels' semantics exactly:
//   rms_norm:   out = x / sqrt(mean(x^2) + eps) * weight
//   layer_norm: out = (x - mean) / sqrt(var + eps) * weight + bias
//               with biased variance (divide by cols), matching layer_norm.cu.
#include "test_common.cuh"

#include "kernel_set/norm.h"

using namespace kst;

// Host reference for RMSNorm. Inputs are already narrowed to the storage dtype
// (so `x`/`weight` are the exact values the kernel reads); accumulation is fp32.
static std::vector<float> ref_rms_norm(const std::vector<float>& x,
                                       const std::vector<float>& weight,
                                       int64_t rows, int64_t cols, float eps,
                                       ks_dtype_t dt) {
  std::vector<float> out(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r) {
    double sumsq = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      const float v = x[static_cast<size_t>(r * cols + c)];
      sumsq += static_cast<double>(v) * v;
    }
    const float inv_rms =
        1.0f / std::sqrt(static_cast<float>(sumsq / cols) + eps);
    for (int64_t c = 0; c < cols; ++c) {
      const float v = x[static_cast<size_t>(r * cols + c)] * inv_rms *
                      weight[static_cast<size_t>(c)];
      out[static_cast<size_t>(r * cols + c)] = quantize(v, dt);
    }
  }
  return out;
}

static std::vector<float> ref_layer_norm(const std::vector<float>& x,
                                         const std::vector<float>& weight,
                                         const std::vector<float>& bias,
                                         bool has_bias, int64_t rows,
                                         int64_t cols, float eps,
                                         ks_dtype_t dt) {
  std::vector<float> out(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r) {
    double sum = 0.0, sumsq = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      const float v = x[static_cast<size_t>(r * cols + c)];
      sum += v;
      sumsq += static_cast<double>(v) * v;
    }
    const float mean = static_cast<float>(sum / cols);
    const float var = static_cast<float>(sumsq / cols) - mean * mean;
    const float inv_std = 1.0f / std::sqrt(var + eps);
    for (int64_t c = 0; c < cols; ++c) {
      float v = (x[static_cast<size_t>(r * cols + c)] - mean) * inv_std *
                weight[static_cast<size_t>(c)];
      if (has_bias) v += bias[static_cast<size_t>(c)];
      out[static_cast<size_t>(r * cols + c)] = quantize(v, dt);
    }
  }
  return out;
}

static void test_rms_norm(ks_dtype_t dt) {
  const int64_t rows = 7, cols = 257;  // odd cols exercises the ragged tail
  const float eps = 1e-6f;
  const Tol t = tol_for(dt);

  std::vector<float> hx = rand_host(rows * cols, 11, -3.f, 3.f);
  std::vector<float> hw = rand_host(cols, 22, 0.5f, 1.5f);
  // Narrow inputs to the storage dtype so the host reference reads the same
  // values the kernel does.
  for (auto& v : hx) v = quantize(v, dt);
  for (auto& v : hw) v = quantize(v, dt);

  DeviceTensor x, w, out;
  x.alloc(rows * cols, dt);
  w.alloc(cols, dt);
  out.alloc(rows * cols, dt);
  x.upload(hx);
  w.upload(hw);

  CHECK_KS(ks_rms_norm(out.dptr, x.dptr, w.dptr, rows, cols, eps, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> got = out.download();
  std::vector<float> ref = ref_rms_norm(hx, hw, rows, cols, eps, dt);
  const std::string name = std::string("rms_norm[") + dt_name(dt) + "]";
  check_allclose(name.c_str(), got, ref, t.atol, t.rtol);
}

static void test_layer_norm(ks_dtype_t dt, bool with_bias) {
  const int64_t rows = 5, cols = 130;
  const float eps = 1e-5f;
  const Tol t = tol_for(dt);

  std::vector<float> hx = rand_host(rows * cols, 33, -4.f, 4.f);
  std::vector<float> hw = rand_host(cols, 44, 0.5f, 1.5f);
  std::vector<float> hb = rand_host(cols, 55, -0.5f, 0.5f);
  for (auto& v : hx) v = quantize(v, dt);
  for (auto& v : hw) v = quantize(v, dt);
  for (auto& v : hb) v = quantize(v, dt);

  DeviceTensor x, w, b, out;
  x.alloc(rows * cols, dt);
  w.alloc(cols, dt);
  out.alloc(rows * cols, dt);
  x.upload(hx);
  w.upload(hw);
  if (with_bias) {
    b.alloc(cols, dt);
    b.upload(hb);
  }

  CHECK_KS(ks_layer_norm(out.dptr, x.dptr, w.dptr,
                         with_bias ? b.dptr : nullptr, rows, cols, eps, dt,
                         nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> got = out.download();
  std::vector<float> ref =
      ref_layer_norm(hx, hw, hb, with_bias, rows, cols, eps, dt);
  const std::string name = std::string("layer_norm[") + dt_name(dt) +
                           (with_bias ? ",bias]" : ",nobias]");
  check_allclose(name.c_str(), got, ref, t.atol, t.rtol);
}

int main() {
  const char* kName = "test_norm";
  if (!has_cuda_device(kName)) return 0;
  for (ks_dtype_t dt : float_dtypes()) {
    test_rms_norm(dt);
    test_layer_norm(dt, /*with_bias=*/true);
    test_layer_norm(dt, /*with_bias=*/false);
  }
  return finish(kName);
}
