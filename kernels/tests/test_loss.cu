// kernel-set — correctness test: ks_cross_entropy (fused forward+backward) vs a
// host reference mirroring cross_entropy.cu / Liger semantics:
//   lse = logsumexp(row); nll = lse - x_y
//   loss = (1-ls)*nll + (ls*lse - (ls/V)*sum(logits))
//   grad[j] = softmax[j] - (1-ls)*onehot[j] - ls/V   (zeroed for ignore_index)
// Reduction over tokens is left to the caller (the kernel writes per-token
// losses and per-element grads only).
#include "test_common.cuh"

#include <algorithm>

#include "kernel_set/loss.h"

using namespace kst;

static void test_cross_entropy(ks_dtype_t dt, float label_smoothing) {
  const int64_t num_tokens = 5;
  const int64_t vocab = 257;  // odd, > block, exercises the streaming passes
  const int64_t ignore_index = -100;
  const Tol t = tol_for(dt);

  std::vector<float> hlogits =
      rand_host(num_tokens * vocab, 11, -3.f, 3.f);
  for (auto& v : hlogits) v = quantize(v, dt);
  // Targets: one row uses ignore_index to exercise the masked path.
  std::vector<int32_t> htargets = {3, static_cast<int32_t>(ignore_index), 100,
                                   256, 0};

  DeviceTensor logits, grad;
  logits.alloc(num_tokens * vocab, dt);
  grad.alloc(num_tokens * vocab, dt);
  logits.upload(hlogits);

  int32_t* dtargets = nullptr;
  CHECK_CUDA(cudaMalloc(&dtargets, htargets.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMemcpy(dtargets, htargets.data(),
                        htargets.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  float* dlosses = nullptr;
  CHECK_CUDA(cudaMalloc(&dlosses, num_tokens * sizeof(float)));

  CHECK_KS(ks_cross_entropy(dlosses, grad.dptr, logits.dptr, dtargets,
                            /*targets_i64=*/0, num_tokens, vocab, ignore_index,
                            label_smoothing, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> got_losses(static_cast<size_t>(num_tokens));
  CHECK_CUDA(cudaMemcpy(got_losses.data(), dlosses,
                        num_tokens * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<float> got_grad = grad.download();

  // Host reference.
  const float ls = label_smoothing;
  const float eps = ls / static_cast<float>(vocab);
  std::vector<float> ref_losses(static_cast<size_t>(num_tokens));
  std::vector<float> ref_grad(static_cast<size_t>(num_tokens * vocab));
  for (int64_t r = 0; r < num_tokens; ++r) {
    const int64_t y = htargets[static_cast<size_t>(r)];
    const float* x = &hlogits[static_cast<size_t>(r * vocab)];
    float* dxr = &ref_grad[static_cast<size_t>(r * vocab)];
    if (y == ignore_index) {
      for (int64_t j = 0; j < vocab; ++j) dxr[j] = quantize(0.f, dt);
      ref_losses[static_cast<size_t>(r)] = 0.f;
      continue;
    }
    float row_max = -1e30f;
    for (int64_t j = 0; j < vocab; ++j) row_max = std::max(row_max, x[j]);
    double denom = 0.0, sum_logits = 0.0;
    for (int64_t j = 0; j < vocab; ++j) {
      denom += std::exp(static_cast<double>(x[j] - row_max));
      sum_logits += x[j];
    }
    const float lse = row_max + std::log(static_cast<float>(denom));
    const float x_y = x[y];
    const float nll = lse - x_y;
    float loss = nll;
    if (ls > 0.f)
      loss = (1.f - ls) * nll +
             (ls * lse - eps * static_cast<float>(sum_logits));
    ref_losses[static_cast<size_t>(r)] = loss;
    for (int64_t j = 0; j < vocab; ++j) {
      const float p =
          std::exp(x[j] - row_max) / static_cast<float>(denom);
      float g = p - eps;
      if (j == y) g -= (1.f - ls);
      dxr[j] = quantize(g, dt);
    }
  }

  const std::string suffix =
      std::string("[") + dt_name(dt) +
      (label_smoothing > 0.f ? ",ls]" : ",nols]");
  // Loss tolerance is a touch looser: lse accumulates over the whole vocab.
  check_allclose((std::string("ce.loss") + suffix).c_str(), got_losses,
                 ref_losses, std::max(t.atol, 5e-3f), std::max(t.rtol, 5e-3f));
  check_allclose((std::string("ce.grad") + suffix).c_str(), got_grad, ref_grad,
                 t.atol, t.rtol);

  cudaFree(dtargets);
  cudaFree(dlosses);
}

int main() {
  const char* kName = "test_loss";
  if (!has_cuda_device(kName)) return 0;
  for (ks_dtype_t dt : float_dtypes()) {
    test_cross_entropy(dt, /*label_smoothing=*/0.f);
    test_cross_entropy(dt, /*label_smoothing=*/0.1f);
  }
  return finish(kName);
}
