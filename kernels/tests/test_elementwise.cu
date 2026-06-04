// kernel-set — correctness test: elementwise ops (add, mul, scale, axpby,
// add_residual) vs host references. fp32 math with storage-dtype rounding,
// matching elementwise.cu.
#include "test_common.cuh"

#include "kernel_set/elementwise.h"

using namespace kst;

static void test_add(ks_dtype_t dt) {
  const int64_t n = 777;
  const Tol t = tol_for(dt);
  std::vector<float> ha = rand_host(n, 1, -3.f, 3.f);
  std::vector<float> hb = rand_host(n, 2, -3.f, 3.f);
  for (auto& v : ha) v = quantize(v, dt);
  for (auto& v : hb) v = quantize(v, dt);

  DeviceTensor a, b, out;
  a.alloc(n, dt);
  b.alloc(n, dt);
  out.alloc(n, dt);
  a.upload(ha);
  b.upload(hb);

  CHECK_KS(ks_add(out.dptr, a.dptr, b.dptr, n, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> ref(static_cast<size_t>(n));
  for (size_t i = 0; i < ref.size(); ++i) ref[i] = quantize(ha[i] + hb[i], dt);
  const std::string name = std::string("add[") + dt_name(dt) + "]";
  check_allclose(name.c_str(), out.download(), ref, t.atol, t.rtol);
}

static void test_mul(ks_dtype_t dt) {
  const int64_t n = 640;
  const Tol t = tol_for(dt);
  std::vector<float> ha = rand_host(n, 3, -2.f, 2.f);
  std::vector<float> hb = rand_host(n, 4, -2.f, 2.f);
  for (auto& v : ha) v = quantize(v, dt);
  for (auto& v : hb) v = quantize(v, dt);

  DeviceTensor a, b, out;
  a.alloc(n, dt);
  b.alloc(n, dt);
  out.alloc(n, dt);
  a.upload(ha);
  b.upload(hb);

  CHECK_KS(ks_mul(out.dptr, a.dptr, b.dptr, n, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> ref(static_cast<size_t>(n));
  for (size_t i = 0; i < ref.size(); ++i) ref[i] = quantize(ha[i] * hb[i], dt);
  const std::string name = std::string("mul[") + dt_name(dt) + "]";
  check_allclose(name.c_str(), out.download(), ref, t.atol, t.rtol);
}

static void test_scale(ks_dtype_t dt) {
  const int64_t n = 513;
  const float s = 1.75f;
  const Tol t = tol_for(dt);
  std::vector<float> hx = rand_host(n, 5, -3.f, 3.f);
  for (auto& v : hx) v = quantize(v, dt);

  DeviceTensor x, out;
  x.alloc(n, dt);
  out.alloc(n, dt);
  x.upload(hx);

  CHECK_KS(ks_scale(out.dptr, x.dptr, s, n, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> ref(static_cast<size_t>(n));
  for (size_t i = 0; i < ref.size(); ++i) ref[i] = quantize(hx[i] * s, dt);
  const std::string name = std::string("scale[") + dt_name(dt) + "]";
  check_allclose(name.c_str(), out.download(), ref, t.atol, t.rtol);
}

static void test_axpby(ks_dtype_t dt) {
  const int64_t n = 900;
  const float alpha = 0.5f, beta = -1.25f;
  const Tol t = tol_for(dt);
  std::vector<float> ha = rand_host(n, 6, -3.f, 3.f);
  std::vector<float> hb = rand_host(n, 7, -3.f, 3.f);
  for (auto& v : ha) v = quantize(v, dt);
  for (auto& v : hb) v = quantize(v, dt);

  DeviceTensor a, b, out;
  a.alloc(n, dt);
  b.alloc(n, dt);
  out.alloc(n, dt);
  a.upload(ha);
  b.upload(hb);

  CHECK_KS(ks_axpby(out.dptr, a.dptr, alpha, b.dptr, beta, n, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> ref(static_cast<size_t>(n));
  for (size_t i = 0; i < ref.size(); ++i)
    ref[i] = quantize(ha[i] * alpha + hb[i] * beta, dt);
  const std::string name = std::string("axpby[") + dt_name(dt) + "]";
  check_allclose(name.c_str(), out.download(), ref, t.atol, t.rtol);
}

static void test_add_residual(ks_dtype_t dt) {
  const int64_t n = 321;
  const Tol t = tol_for(dt);
  std::vector<float> hr = rand_host(n, 8, -3.f, 3.f);
  std::vector<float> hx = rand_host(n, 9, -3.f, 3.f);
  for (auto& v : hr) v = quantize(v, dt);
  for (auto& v : hx) v = quantize(v, dt);

  DeviceTensor res, x;
  res.alloc(n, dt);
  x.alloc(n, dt);
  res.upload(hr);
  x.upload(hx);

  CHECK_KS(ks_add_residual(res.dptr, x.dptr, n, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> ref(static_cast<size_t>(n));
  for (size_t i = 0; i < ref.size(); ++i) ref[i] = quantize(hr[i] + hx[i], dt);
  const std::string name = std::string("add_residual[") + dt_name(dt) + "]";
  check_allclose(name.c_str(), res.download(), ref, t.atol, t.rtol);
}

int main() {
  const char* kName = "test_elementwise";
  if (!has_cuda_device(kName)) return 0;
  for (ks_dtype_t dt : float_dtypes()) {
    test_add(dt);
    test_mul(dt);
    test_scale(dt);
    test_axpby(dt);
    test_add_residual(dt);
  }
  return finish(kName);
}
