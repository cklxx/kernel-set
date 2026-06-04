// kernel-set — correctness test: activations (silu, gelu erf/tanh, relu) and
// the gated SwiGLU fusion vs host references. The scalar numerics mirror
// activation_ops.cuh exactly (fp32 math, storage-dtype rounding on store).
#include "test_common.cuh"

#include "kernel_set/activation.h"

using namespace kst;

// ---- Scalar references (fp32), matching activation_ops.cuh ------------------
static float r_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static float r_silu(float x) { return x * r_sigmoid(x); }
static float r_relu(float x) { return x > 0.f ? x : 0.f; }
static float r_gelu_erf(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f));
}
static float r_gelu_tanh(float x) {
  const float inner =
      0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

template <typename Fn>
static std::vector<float> map_unary(const std::vector<float>& x, Fn fn,
                                    ks_dtype_t dt) {
  std::vector<float> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = quantize(fn(x[i]), dt);
  return out;
}

// elementwise activation under test, selected by a small enum.
enum class Act { Silu, GeluErf, GeluTanh, Relu };

static void test_unary(Act act, ks_dtype_t dt) {
  const int64_t n = 1031;  // prime -> exercises the vectorized + scalar tail
  const Tol t = tol_for(dt);
  std::vector<float> hx = rand_host(n, 101, -5.f, 5.f);
  for (auto& v : hx) v = quantize(v, dt);

  DeviceTensor x, out;
  x.alloc(n, dt);
  out.alloc(n, dt);
  x.upload(hx);

  std::vector<float> ref;
  std::string name;
  switch (act) {
    case Act::Silu:
      CHECK_KS(ks_silu(out.dptr, x.dptr, n, dt, nullptr));
      ref = map_unary(hx, r_silu, dt);
      name = "silu";
      break;
    case Act::GeluErf:
      CHECK_KS(ks_gelu(out.dptr, x.dptr, n, /*tanh_approx=*/0, dt, nullptr));
      ref = map_unary(hx, r_gelu_erf, dt);
      name = "gelu_erf";
      break;
    case Act::GeluTanh:
      CHECK_KS(ks_gelu(out.dptr, x.dptr, n, /*tanh_approx=*/1, dt, nullptr));
      ref = map_unary(hx, r_gelu_tanh, dt);
      name = "gelu_tanh";
      break;
    case Act::Relu:
      CHECK_KS(ks_relu(out.dptr, x.dptr, n, dt, nullptr));
      ref = map_unary(hx, r_relu, dt);
      name = "relu";
      break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<float> got = out.download();
  name = name + "[" + dt_name(dt) + "]";
  check_allclose(name.c_str(), got, ref, t.atol, t.rtol);
}

// SwiGLU: out = silu(gate) * up.
static void test_swiglu(ks_dtype_t dt) {
  const int64_t rows = 9, inter = 128;
  const int64_t n = rows * inter;
  const Tol t = tol_for(dt);
  std::vector<float> hg = rand_host(n, 202, -4.f, 4.f);
  std::vector<float> hu = rand_host(n, 303, -4.f, 4.f);
  for (auto& v : hg) v = quantize(v, dt);
  for (auto& v : hu) v = quantize(v, dt);

  DeviceTensor g, u, out;
  g.alloc(n, dt);
  u.alloc(n, dt);
  out.alloc(n, dt);
  g.upload(hg);
  u.upload(hu);

  CHECK_KS(ks_swiglu(out.dptr, g.dptr, u.dptr, rows, inter, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> ref(static_cast<size_t>(n));
  for (size_t i = 0; i < ref.size(); ++i)
    ref[i] = quantize(r_silu(hg[i]) * hu[i], dt);

  std::vector<float> got = out.download();
  const std::string name = std::string("swiglu[") + dt_name(dt) + "]";
  check_allclose(name.c_str(), got, ref, t.atol, t.rtol);
}

int main() {
  const char* kName = "test_activation";
  if (!has_cuda_device(kName)) return 0;
  for (ks_dtype_t dt : float_dtypes()) {
    test_unary(Act::Silu, dt);
    test_unary(Act::GeluErf, dt);
    test_unary(Act::GeluTanh, dt);
    test_unary(Act::Relu, dt);
    test_swiglu(dt);
  }
  return finish(kName);
}
