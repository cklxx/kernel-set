// kernel-set — correctness test: ks_adamw / ks_sgd_momentum /
// ks_global_grad_norm vs host references mirroring the kernel math exactly.
//
//   AdamW (decoupled decay):
//     g=grad/grad_scale; m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g
//     p-=lr*wd*p; p-=step_size*(m/(sqrt(v*inv_bc2)+eps))
//     step_size=lr/(1-b1^step), inv_bc2=1/(1-b2^step)
//   SGD: g=grad/gs; g+=wd*p; buf=mu*buf+g; d=nesterov?(g+mu*buf):buf; p-=lr*d
//   global_grad_norm: sqrt(sum g_i^2) over all tensors.
//
// State (exp_avg / exp_avg_sq / momentum) is fp32. Param/grad use the storage
// dtype; all math is fp32 and the result is narrowed back to `dt`.
#include "test_common.cuh"

#include "kernel_set/optimizer.h"

using namespace kst;

// A device fp32 buffer (state tensors are always fp32).
struct DeviceF32 {
  float* dptr = nullptr;
  size_t n = 0;
  void alloc(const std::vector<float>& host) {
    n = host.size();
    CHECK_CUDA(cudaMalloc(&dptr, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dptr, host.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  std::vector<float> download() const {
    std::vector<float> h(n);
    CHECK_CUDA(cudaMemcpy(h.data(), dptr, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return h;
  }
  ~DeviceF32() {
    if (dptr) cudaFree(dptr);
  }
};

static void test_adamw(ks_dtype_t dt) {
  const int64_t n = 1024 + 5;  // +5 forces the scalar (ragged) tail path
  const float lr = 1e-2f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;
  const float grad_scale = 1.f;
  const int64_t step = 7;
  const Tol t = tol_for(dt);

  std::vector<float> hp = rand_host(n, 11, -1.f, 1.f);
  std::vector<float> hg = rand_host(n, 22, -1.f, 1.f);
  std::vector<float> hm = rand_host(n, 33, -0.1f, 0.1f);
  std::vector<float> hv = rand_host(n, 44, 0.f, 0.1f);
  for (auto& v : hp) v = quantize(v, dt);
  for (auto& v : hg) v = quantize(v, dt);

  DeviceTensor p, g;
  p.alloc(n, dt);
  g.alloc(n, dt);
  p.upload(hp);
  g.upload(hg);
  DeviceF32 m, v;
  m.alloc(hm);
  v.alloc(hv);

  CHECK_KS(ks_adamw(p.dptr, /*master=*/nullptr, g.dptr, m.dptr, v.dptr, lr, b1,
                    b2, eps, wd, step, grad_scale, n, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Host reference.
  const float bc1 = 1.f - std::pow(b1, static_cast<float>(step));
  const float bc2 = 1.f - std::pow(b2, static_cast<float>(step));
  const float step_size = lr / bc1;
  const float inv_bc2 = 1.f / bc2;
  std::vector<float> refp(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    float pp = hp[static_cast<size_t>(i)];
    float gg = hg[static_cast<size_t>(i)];
    float mm = hm[static_cast<size_t>(i)];
    float vv = hv[static_cast<size_t>(i)];
    mm = b1 * mm + (1.f - b1) * gg;
    vv = b2 * vv + (1.f - b2) * gg * gg;
    pp -= lr * wd * pp;
    pp -= step_size * (mm / (std::sqrt(vv * inv_bc2) + eps));
    refp[static_cast<size_t>(i)] = quantize(pp, dt);
  }
  const std::string name = std::string("adamw[") + dt_name(dt) + "]";
  check_allclose(name.c_str(), p.download(), refp, t.atol, t.rtol);
}

static void test_sgd(ks_dtype_t dt, bool nesterov) {
  const int64_t n = 512 + 3;
  const float lr = 0.1f, mu = 0.9f, wd = 0.001f, grad_scale = 1.f;
  const Tol t = tol_for(dt);

  std::vector<float> hp = rand_host(n, 55, -1.f, 1.f);
  std::vector<float> hg = rand_host(n, 66, -1.f, 1.f);
  std::vector<float> hbuf = rand_host(n, 77, -0.1f, 0.1f);
  for (auto& v : hp) v = quantize(v, dt);
  for (auto& v : hg) v = quantize(v, dt);

  DeviceTensor p, g;
  p.alloc(n, dt);
  g.alloc(n, dt);
  p.upload(hp);
  g.upload(hg);
  DeviceF32 buf;
  buf.alloc(hbuf);

  CHECK_KS(ks_sgd_momentum(p.dptr, /*master=*/nullptr, g.dptr, buf.dptr, lr, mu,
                           wd, nesterov ? 1 : 0, grad_scale, n, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> refp(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    float pp = hp[static_cast<size_t>(i)];
    float gg = hg[static_cast<size_t>(i)];
    float bb = hbuf[static_cast<size_t>(i)];
    if (wd != 0.f) gg += wd * pp;
    bb = mu * bb + gg;
    const float d = nesterov ? (gg + mu * bb) : bb;
    pp -= lr * d;
    refp[static_cast<size_t>(i)] = quantize(pp, dt);
  }
  const std::string name = std::string("sgd[") + dt_name(dt) +
                           (nesterov ? ",nesterov]" : ",plain]");
  check_allclose(name.c_str(), p.download(), refp, t.atol, t.rtol);
}

static void test_global_grad_norm(ks_dtype_t dt) {
  const Tol t = tol_for(dt);
  std::vector<int64_t> sizes = {300, 1, 777};
  std::vector<DeviceTensor*> tensors;
  std::vector<float> all;  // concatenated for the host reference
  std::vector<const void*> hptrs;
  for (size_t s = 0; s < sizes.size(); ++s) {
    auto* dtv = new DeviceTensor();
    std::vector<float> h = rand_host(static_cast<size_t>(sizes[s]),
                                     static_cast<unsigned>(101 + s), -1.f, 1.f);
    for (auto& v : h) v = quantize(v, dt);
    dtv->alloc(static_cast<size_t>(sizes[s]), dt);
    dtv->upload(h);
    tensors.push_back(dtv);
    hptrs.push_back(dtv->dptr);
    all.insert(all.end(), h.begin(), h.end());
  }

  // The ABI takes a HOST array of device pointers (see global_grad_norm.cu),
  // so hptrs is passed directly — no device-side pointer table needed.
  float* dnorm = nullptr;
  CHECK_CUDA(cudaMalloc(&dnorm, sizeof(float)));

  CHECK_KS(ks_global_grad_norm(dnorm, hptrs.data(), sizes.data(),
                               static_cast<int>(sizes.size()), dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  float got = 0.f;
  CHECK_CUDA(cudaMemcpy(&got, dnorm, sizeof(float), cudaMemcpyDeviceToHost));

  double sumsq = 0.0;
  for (float v : all) sumsq += static_cast<double>(v) * v;
  const float ref = std::sqrt(static_cast<float>(sumsq));
  std::vector<float> g = {got}, r = {ref};
  const std::string name = std::string("global_grad_norm[") + dt_name(dt) + "]";
  check_allclose(name.c_str(), g, r, 1e-2f, 1e-3f);

  cudaFree(dnorm);
  for (auto* p : tensors) delete p;
}

int main() {
  const char* kName = "test_optimizer";
  if (!has_cuda_device(kName)) return 0;
  for (ks_dtype_t dt : float_dtypes()) {
    test_adamw(dt);
    test_sgd(dt, /*nesterov=*/false);
    test_sgd(dt, /*nesterov=*/true);
    test_global_grad_norm(dt);
  }
  return finish(kName);
}
