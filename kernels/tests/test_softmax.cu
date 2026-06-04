// kernel-set — correctness test: ks_softmax (with temperature), ks_log_softmax,
// and ks_argmax vs numerically-stable host references over the last dim.
#include "test_common.cuh"

#include "kernel_set/sampling.h"

using namespace kst;

// Stable host softmax of a row of `cols`, scaled by inv_temp, narrowed to `dt`.
static std::vector<float> ref_softmax(const std::vector<float>& x, int64_t rows,
                                      int64_t cols, float inv_temp,
                                      ks_dtype_t dt) {
  std::vector<float> out(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r) {
    float m = -3.4028235e38f;
    for (int64_t c = 0; c < cols; ++c)
      m = std::max(m, x[static_cast<size_t>(r * cols + c)] * inv_temp);
    double denom = 0.0;
    for (int64_t c = 0; c < cols; ++c)
      denom += std::exp(x[static_cast<size_t>(r * cols + c)] * inv_temp - m);
    const float inv = 1.0f / static_cast<float>(denom);
    for (int64_t c = 0; c < cols; ++c) {
      const float p =
          std::exp(x[static_cast<size_t>(r * cols + c)] * inv_temp - m) * inv;
      out[static_cast<size_t>(r * cols + c)] = quantize(p, dt);
    }
  }
  return out;
}

static std::vector<float> ref_log_softmax(const std::vector<float>& x,
                                          int64_t rows, int64_t cols,
                                          ks_dtype_t dt) {
  std::vector<float> out(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r) {
    float m = -3.4028235e38f;
    for (int64_t c = 0; c < cols; ++c)
      m = std::max(m, x[static_cast<size_t>(r * cols + c)]);
    double denom = 0.0;
    for (int64_t c = 0; c < cols; ++c)
      denom += std::exp(x[static_cast<size_t>(r * cols + c)] - m);
    const float log_denom = m + std::log(static_cast<float>(denom));
    for (int64_t c = 0; c < cols; ++c)
      out[static_cast<size_t>(r * cols + c)] =
          quantize(x[static_cast<size_t>(r * cols + c)] - log_denom, dt);
  }
  return out;
}

static void test_softmax(ks_dtype_t dt, float temperature) {
  const int64_t rows = 6, cols = 200;
  const float inv_temp = temperature > 0.f ? 1.0f / temperature : 1.0f;
  const Tol t = tol_for(dt);
  std::vector<float> hx = rand_host(rows * cols, 401, -6.f, 6.f);
  for (auto& v : hx) v = quantize(v, dt);

  DeviceTensor x, out;
  x.alloc(rows * cols, dt);
  out.alloc(rows * cols, dt);
  x.upload(hx);

  CHECK_KS(ks_softmax(out.dptr, x.dptr, rows, cols, temperature, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> ref = ref_softmax(hx, rows, cols, inv_temp, dt);
  std::vector<float> got = out.download();
  char buf[64];
  std::snprintf(buf, sizeof(buf), "softmax[%s,T=%.2f]", dt_name(dt),
                temperature);
  check_allclose(buf, got, ref, t.atol, t.rtol);

  // Each row of the softmax must sum to ~1.
  for (int64_t r = 0; r < rows; ++r) {
    double s = 0.0;
    for (int64_t c = 0; c < cols; ++c)
      s += got[static_cast<size_t>(r * cols + c)];
    CHECK(close(static_cast<float>(s), 1.0f, 5 * t.atol, 5 * t.rtol));
  }
}

static void test_log_softmax(ks_dtype_t dt) {
  const int64_t rows = 4, cols = 150;
  const Tol t = tol_for(dt);
  std::vector<float> hx = rand_host(rows * cols, 402, -6.f, 6.f);
  for (auto& v : hx) v = quantize(v, dt);

  DeviceTensor x, out;
  x.alloc(rows * cols, dt);
  out.alloc(rows * cols, dt);
  x.upload(hx);

  CHECK_KS(ks_log_softmax(out.dptr, x.dptr, rows, cols, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> ref = ref_log_softmax(hx, rows, cols, dt);
  const std::string name = std::string("log_softmax[") + dt_name(dt) + "]";
  check_allclose(name.c_str(), out.download(), ref, t.atol, t.rtol);
}

static void test_argmax(ks_dtype_t dt) {
  const int64_t seqs = 5, vocab = 311;
  std::vector<float> hl = rand_host(seqs * vocab, 403, -10.f, 10.f);
  for (auto& v : hl) v = quantize(v, dt);

  DeviceTensor logits;
  logits.alloc(seqs * vocab, dt);
  logits.upload(hl);

  int32_t* d_tok = nullptr;
  CHECK_CUDA(cudaMalloc(&d_tok, seqs * sizeof(int32_t)));
  CHECK_KS(ks_argmax(d_tok, logits.dptr, seqs, vocab, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<int32_t> got(static_cast<size_t>(seqs));
  CHECK_CUDA(cudaMemcpy(got.data(), d_tok, seqs * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));

  // Reference argmax with lowest-index tie-breaking (matches the kernel).
  for (int64_t r = 0; r < seqs; ++r) {
    float best = -3.4028235e38f;
    int best_i = 0;
    for (int64_t c = 0; c < vocab; ++c) {
      const float v = hl[static_cast<size_t>(r * vocab + c)];
      if (v > best) {
        best = v;
        best_i = static_cast<int>(c);
      }
    }
    if (got[static_cast<size_t>(r)] != best_i)
      KST_FAILF("argmax[%s] row %lld: got %d ref %d", dt_name(dt),
                static_cast<long long>(r), got[static_cast<size_t>(r)], best_i);
  }
  std::printf("[ok] argmax[%s]\n", dt_name(dt));
  cudaFree(d_tok);
}

int main() {
  const char* kName = "test_softmax";
  if (!has_cuda_device(kName)) return 0;
  for (ks_dtype_t dt : float_dtypes()) {
    test_softmax(dt, 1.0f);
    test_softmax(dt, 0.7f);
    test_log_softmax(dt);
    test_argmax(dt);
  }
  return finish(kName);
}
