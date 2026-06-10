// kernel-set — correctness test: ks_quantize_int8 / ks_dequantize_int8 vs a
// host reference. Symmetric dynamic INT8:
//   scale = max(amax / 127, eps);  q = round_to_nearest_even(x / scale) clamped
//           to [-127, 127];  dequant: x' = q * scale
// Covered for per-token (one scale per row) and per-tensor (single scale).
#include "test_common.cuh"

#include <algorithm>
#include <cmath>

#include "kernel_set/quant.h"

using namespace kst;

// Round-to-nearest-even, matching __float2int_rn / the kernel.
static int rne(float v) {
  float r = std::nearbyint(v);  // honors the current (default = to-nearest) mode
  return static_cast<int>(r);
}

static int8_t qclamp(float x, float scale) {
  const float q = rne(x / scale);
  const float c = std::min(127.f, std::max(-127.f, q));
  return static_cast<int8_t>(c);
}

// Read back an int8 device buffer as host int8 values.
static std::vector<int8_t> download_i8(void* dptr, size_t n) {
  std::vector<int8_t> h(n);
  CHECK_CUDA(cudaMemcpy(h.data(), dptr, n, cudaMemcpyDeviceToHost));
  return h;
}

static void test_int8(ks_dtype_t dt, ks_quant_mode_t mode) {
  const int64_t rows = 6, cols = 130;
  const float eps = 1e-12f;
  const bool per_token = (mode == KS_QUANT_PER_TOKEN);

  std::vector<float> hx = rand_host(rows * cols, 11, -5.f, 5.f);
  for (auto& v : hx) v = quantize(v, dt);

  DeviceTensor x;
  x.alloc(rows * cols, dt);
  x.upload(hx);

  int8_t* dq = nullptr;
  CHECK_CUDA(cudaMalloc(&dq, static_cast<size_t>(rows * cols)));
  const int64_t num_scales = per_token ? rows : 1;
  float* dscale = nullptr;
  CHECK_CUDA(cudaMalloc(&dscale, static_cast<size_t>(num_scales) * sizeof(float)));

  CHECK_KS(ks_quantize_int8(dq, dscale, x.dptr, rows, cols, dt, mode, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Host reference scales.
  std::vector<float> ref_scale(static_cast<size_t>(num_scales));
  if (per_token) {
    for (int64_t r = 0; r < rows; ++r) {
      float amax = 0.f;
      for (int64_t c = 0; c < cols; ++c)
        amax = std::max(amax, std::fabs(hx[static_cast<size_t>(r * cols + c)]));
      ref_scale[static_cast<size_t>(r)] = std::max(amax / 127.f, eps);
    }
  } else {
    float amax = 0.f;
    for (float v : hx) amax = std::max(amax, std::fabs(v));
    ref_scale[0] = std::max(amax / 127.f, eps);
  }

  // Compare scales.
  std::vector<float> got_scale(static_cast<size_t>(num_scales));
  CHECK_CUDA(cudaMemcpy(got_scale.data(), dscale,
                        static_cast<size_t>(num_scales) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const std::string mode_s = per_token ? "per_token" : "per_tensor";
  check_allclose((std::string("int8.scale[") + dt_name(dt) + "," + mode_s + "]")
                     .c_str(),
                 got_scale, ref_scale, 1e-5f, 1e-4f);

  // Compare the quantized codes (exact integer match expected).
  std::vector<int8_t> got_q = download_i8(dq, static_cast<size_t>(rows * cols));
  int bad = -1;
  for (int64_t r = 0; r < rows && bad < 0; ++r) {
    const float sc = per_token ? ref_scale[static_cast<size_t>(r)]
                               : ref_scale[0];
    for (int64_t c = 0; c < cols; ++c) {
      const int8_t ref =
          qclamp(hx[static_cast<size_t>(r * cols + c)], sc);
      const int8_t got = got_q[static_cast<size_t>(r * cols + c)];
      // Allow off-by-one on the round (tie direction differences vs the GPU).
      if (std::abs(static_cast<int>(got) - static_cast<int>(ref)) > 1) {
        bad = static_cast<int>(r * cols + c);
        KST_FAILF("int8.q[%s,%s] mismatch at [%d] got=%d ref=%d",
                  dt_name(dt), mode_s.c_str(), bad, static_cast<int>(got),
                  static_cast<int>(ref));
        break;
      }
    }
  }
  if (bad < 0)
    std::printf("[ok] int8.q[%s,%s] (n=%lld)\n", dt_name(dt), mode_s.c_str(),
                static_cast<long long>(rows * cols));

  // Round-trip dequant: x' = q * scale, compared back to the kernel's quant of
  // x (so the comparison reference uses the SAME codes the kernel produced).
  DeviceTensor deq;
  deq.alloc(rows * cols, dt);
  CHECK_KS(ks_dequantize_int8(deq.dptr, dq, dscale, rows, cols, dt, mode,
                              nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<float> got_deq = deq.download();
  std::vector<float> ref_deq(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r) {
    const float sc = per_token ? got_scale[static_cast<size_t>(r)]
                               : got_scale[0];
    for (int64_t c = 0; c < cols; ++c) {
      const float v =
          static_cast<float>(got_q[static_cast<size_t>(r * cols + c)]) * sc;
      ref_deq[static_cast<size_t>(r * cols + c)] = quantize(v, dt);
    }
  }
  const Tol t = tol_for(dt);
  check_allclose(
      (std::string("int8.dequant[") + dt_name(dt) + "," + mode_s + "]").c_str(),
      got_deq, ref_deq, t.atol, t.rtol);

  cudaFree(dq);
  cudaFree(dscale);
}

int main() {
  const char* kName = "test_quant";
  if (!has_cuda_device(kName)) return 0;
  for (ks_dtype_t dt : float_dtypes()) {
    test_int8(dt, KS_QUANT_PER_TOKEN);
    test_int8(dt, KS_QUANT_PER_TENSOR);
  }
  return finish(kName);
}
