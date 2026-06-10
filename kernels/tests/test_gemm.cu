// kernel-set — correctness test: ks_gemm and ks_gemm_bias_act vs a host
// reference. Row-major, fp32 accumulation (the kernel uses tensor cores for the
// f16/bf16 paths and TF32/fp32 for the f32 path; all accumulate in fp32).
//
//   ks_gemm:          C = alpha * op(A) @ op(B) + beta * C
//   ks_gemm_bias_act: D = act(alpha * A @ B + bias)   (row-major, no transpose)
#include "test_common.cuh"

#include "kernel_set/gemm.h"

using namespace kst;

// Host GEMM with the same (trans_a, trans_b, leading-dim) convention as the
// kernel. Inputs are pre-narrowed to the storage dtype; accumulation is fp32 and
// the result is narrowed back to `dt` so the reference matches the kernel's
// read/write rounding.
static std::vector<float> ref_gemm(const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   const std::vector<float>& c_in, int64_t m,
                                   int64_t n, int64_t k, bool trans_a,
                                   bool trans_b, int64_t lda, int64_t ldb,
                                   int64_t ldc, float alpha, float beta,
                                   ks_dtype_t dt) {
  std::vector<float> out(static_cast<size_t>(m * n));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t p = 0; p < k; ++p) {
        // A is [M,K] row-major (lda=K) unless transposed -> [K,M] (lda=M).
        const float av = trans_a ? a[static_cast<size_t>(p * lda + i)]
                                 : a[static_cast<size_t>(i * lda + p)];
        // B is [K,N] row-major (ldb=N) unless transposed -> [N,K] (ldb=K).
        const float bv = trans_b ? b[static_cast<size_t>(j * ldb + p)]
                                 : b[static_cast<size_t>(p * ldb + j)];
        acc += static_cast<double>(av) * static_cast<double>(bv);
      }
      float v = alpha * static_cast<float>(acc);
      if (beta != 0.f) v += beta * c_in[static_cast<size_t>(i * ldc + j)];
      out[static_cast<size_t>(i * n + j)] = quantize(v, dt);
    }
  }
  return out;
}

static float act_apply(float v, ks_activation_t act) {
  switch (act) {
    case KS_ACT_RELU:
      return v > 0.f ? v : 0.f;
    case KS_ACT_GELU:
      return 0.5f * v * (1.f + std::erf(v * 0.70710678f));
    case KS_ACT_GELU_TANH: {
      const float c = 0.7978845608f;  // sqrt(2/pi)
      const float inner = c * (v + 0.044715f * v * v * v);
      return 0.5f * v * (1.f + std::tanh(inner));
    }
    case KS_ACT_SILU:
      return v / (1.f + std::exp(-v));
    default:
      return v;
  }
}

static void test_gemm(ks_dtype_t dt) {
  // Sizes deliberately not multiples of common tile sizes to exercise tails.
  const int64_t m = 33, n = 40, k = 24;
  const float alpha = 1.25f, beta = 0.5f;
  const Tol t = tol_for(dt);

  std::vector<float> ha = rand_host(m * k, 101, -1.f, 1.f);
  std::vector<float> hb = rand_host(k * n, 202, -1.f, 1.f);
  std::vector<float> hc = rand_host(m * n, 303, -1.f, 1.f);
  for (auto& v : ha) v = quantize(v, dt);
  for (auto& v : hb) v = quantize(v, dt);
  for (auto& v : hc) v = quantize(v, dt);

  DeviceTensor a, b, c;
  a.alloc(m * k, dt);
  b.alloc(k * n, dt);
  c.alloc(m * n, dt);
  a.upload(ha);
  b.upload(hb);
  c.upload(hc);

  CHECK_KS(ks_gemm(c.dptr, a.dptr, b.dptr, m, n, k, /*trans_a=*/0,
                   /*trans_b=*/0, /*lda=*/k, /*ldb=*/n, /*ldc=*/n, alpha, beta,
                   dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> got = c.download();
  std::vector<float> ref =
      ref_gemm(ha, hb, hc, m, n, k, false, false, k, n, n, alpha, beta, dt);
  const std::string name = std::string("gemm[") + dt_name(dt) + "]";
  check_allclose(name.c_str(), got, ref, t.atol, t.rtol);
}

static void test_gemm_bias_act(ks_dtype_t dt, ks_activation_t act,
                               const char* act_name) {
  const int64_t m = 17, n = 32, k = 20;
  const float alpha = 1.0f;
  const Tol t = tol_for(dt);

  std::vector<float> ha = rand_host(m * k, 404, -1.f, 1.f);
  std::vector<float> hb = rand_host(k * n, 505, -1.f, 1.f);
  std::vector<float> hbias = rand_host(n, 606, -0.5f, 0.5f);
  for (auto& v : ha) v = quantize(v, dt);
  for (auto& v : hb) v = quantize(v, dt);
  for (auto& v : hbias) v = quantize(v, dt);

  DeviceTensor a, b, bias, d;
  a.alloc(m * k, dt);
  b.alloc(k * n, dt);
  bias.alloc(n, dt);
  d.alloc(m * n, dt);
  a.upload(ha);
  b.upload(hb);
  bias.upload(hbias);

  CHECK_KS(ks_gemm_bias_act(d.dptr, a.dptr, b.dptr, bias.dptr, m, n, k, alpha,
                            act, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> got = d.download();
  std::vector<float> ref(static_cast<size_t>(m * n));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t p = 0; p < k; ++p)
        acc += static_cast<double>(ha[static_cast<size_t>(i * k + p)]) *
               static_cast<double>(hb[static_cast<size_t>(p * n + j)]);
      float v = alpha * static_cast<float>(acc) + hbias[static_cast<size_t>(j)];
      v = act_apply(v, act);
      ref[static_cast<size_t>(i * n + j)] = quantize(v, dt);
    }
  }
  const std::string name =
      std::string("gemm_bias_act[") + dt_name(dt) + "," + act_name + "]";
  check_allclose(name.c_str(), got, ref, t.atol, t.rtol);
}

int main() {
  const char* kName = "test_gemm";
  if (!has_cuda_device(kName)) return 0;
  for (ks_dtype_t dt : float_dtypes()) {
    test_gemm(dt);
    test_gemm_bias_act(dt, KS_ACT_NONE, "none");
    test_gemm_bias_act(dt, KS_ACT_RELU, "relu");
    test_gemm_bias_act(dt, KS_ACT_SILU, "silu");
  }
  return finish(kName);
}
