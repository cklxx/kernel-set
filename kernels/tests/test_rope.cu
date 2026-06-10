// kernel-set — correctness test: ks_rope / ks_rope_inplace / ks_rope_gather /
// ks_rope_backward vs a host reference. Mirrors rope_common.cuh exactly:
//   NeoX  (interleaved==0): pair j is (x[j], x[j+half])
//   GPT-J (interleaved!=0): pair j is (x[2j], x[2j+1])
//   forward : x0' = x0*cos - x1*sin ; x1' = x1*cos + x0*sin
//   backward: conjugate rotation (sin -> -sin).
#include "test_common.cuh"

#include "kernel_set/rope.h"

using namespace kst;

// Rotate one [num_tokens, num_heads, head_dim] tensor in place on the host.
// `cs_row[t]` selects the cos/sin row for token t (== t for the non-gather
// path; == positions[t] for gather). sign: +1 forward, -1 backward.
static void ref_rope_apply(std::vector<float>& x, const std::vector<float>& cosr,
                           const std::vector<float>& sinr,
                           const std::vector<int>& cs_row, int64_t num_tokens,
                           int num_heads, int head_dim, bool interleaved,
                           float sign, ks_dtype_t dt) {
  const int half = head_dim / 2;
  for (int64_t t = 0; t < num_tokens; ++t) {
    const int row = cs_row[static_cast<size_t>(t)];
    for (int h = 0; h < num_heads; ++h) {
      const int64_t base =
          (t * num_heads + h) * static_cast<int64_t>(head_dim);
      for (int j = 0; j < half; ++j) {
        int i0, i1;
        if (interleaved) {
          i0 = 2 * j;
          i1 = 2 * j + 1;
        } else {
          i0 = j;
          i1 = j + half;
        }
        const float c = cosr[static_cast<size_t>(row * half + j)];
        const float s = sign * sinr[static_cast<size_t>(row * half + j)];
        const float x0 = x[static_cast<size_t>(base + i0)];
        const float x1 = x[static_cast<size_t>(base + i1)];
        x[static_cast<size_t>(base + i0)] = quantize(x0 * c - x1 * s, dt);
        x[static_cast<size_t>(base + i1)] = quantize(x1 * c + x0 * s, dt);
      }
    }
  }
}

static void test_rope_oop(ks_dtype_t dt, bool interleaved) {
  const int64_t num_tokens = 6;
  const int num_q_heads = 4, num_kv_heads = 2, head_dim = 64;
  const int half = head_dim / 2;
  const Tol t = tol_for(dt);

  std::vector<float> hq =
      rand_host(num_tokens * num_q_heads * head_dim, 11, -2.f, 2.f);
  std::vector<float> hk =
      rand_host(num_tokens * num_kv_heads * head_dim, 22, -2.f, 2.f);
  std::vector<float> hcos = rand_host(num_tokens * half, 33, -1.f, 1.f);
  std::vector<float> hsin = rand_host(num_tokens * half, 44, -1.f, 1.f);
  for (auto& v : hq) v = quantize(v, dt);
  for (auto& v : hk) v = quantize(v, dt);
  for (auto& v : hcos) v = quantize(v, dt);
  for (auto& v : hsin) v = quantize(v, dt);

  DeviceTensor q, k, qo, ko, cos, sin;
  q.alloc(hq.size(), dt);
  k.alloc(hk.size(), dt);
  qo.alloc(hq.size(), dt);
  ko.alloc(hk.size(), dt);
  cos.alloc(hcos.size(), dt);
  sin.alloc(hsin.size(), dt);
  q.upload(hq);
  k.upload(hk);
  cos.upload(hcos);
  sin.upload(hsin);

  CHECK_KS(ks_rope(qo.dptr, ko.dptr, q.dptr, k.dptr, cos.dptr, sin.dptr,
                   num_tokens, num_q_heads, num_kv_heads, head_dim,
                   interleaved ? 1 : 0, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<int> cs_row(static_cast<size_t>(num_tokens));
  for (int64_t i = 0; i < num_tokens; ++i) cs_row[static_cast<size_t>(i)] =
      static_cast<int>(i);
  std::vector<float> refq = hq, refk = hk;
  ref_rope_apply(refq, hcos, hsin, cs_row, num_tokens, num_q_heads, head_dim,
                 interleaved, 1.f, dt);
  ref_rope_apply(refk, hcos, hsin, cs_row, num_tokens, num_kv_heads, head_dim,
                 interleaved, 1.f, dt);

  const std::string base =
      std::string("[") + dt_name(dt) + (interleaved ? ",gptj]" : ",neox]");
  check_allclose((std::string("rope.q") + base).c_str(), qo.download(), refq,
                 t.atol, t.rtol);
  check_allclose((std::string("rope.k") + base).c_str(), ko.download(), refk,
                 t.atol, t.rtol);
}

static void test_rope_backward(ks_dtype_t dt) {
  const int64_t num_tokens = 5;
  const int num_q_heads = 3, num_kv_heads = 0, head_dim = 32;
  const int half = head_dim / 2;
  const Tol t = tol_for(dt);

  std::vector<float> hg =
      rand_host(num_tokens * num_q_heads * head_dim, 55, -2.f, 2.f);
  std::vector<float> hcos = rand_host(num_tokens * half, 66, -1.f, 1.f);
  std::vector<float> hsin = rand_host(num_tokens * half, 77, -1.f, 1.f);
  for (auto& v : hg) v = quantize(v, dt);
  for (auto& v : hcos) v = quantize(v, dt);
  for (auto& v : hsin) v = quantize(v, dt);

  DeviceTensor g, cos, sin;
  g.alloc(hg.size(), dt);
  cos.alloc(hcos.size(), dt);
  sin.alloc(hsin.size(), dt);
  g.upload(hg);
  cos.upload(hcos);
  sin.upload(hsin);

  CHECK_KS(ks_rope_backward(g.dptr, nullptr, cos.dptr, sin.dptr, num_tokens,
                            num_q_heads, num_kv_heads, head_dim,
                            /*interleaved=*/0, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<int> cs_row(static_cast<size_t>(num_tokens));
  for (int64_t i = 0; i < num_tokens; ++i) cs_row[static_cast<size_t>(i)] =
      static_cast<int>(i);
  std::vector<float> ref = hg;
  ref_rope_apply(ref, hcos, hsin, cs_row, num_tokens, num_q_heads, head_dim,
                 /*interleaved=*/false, /*sign=*/-1.f, dt);
  const std::string name = std::string("rope.backward[") + dt_name(dt) + "]";
  check_allclose(name.c_str(), g.download(), ref, t.atol, t.rtol);
}

static void test_rope_gather(ks_dtype_t dt) {
  const int64_t num_tokens = 5;
  const int num_q_heads = 2, num_kv_heads = 1, head_dim = 32;
  const int half = head_dim / 2;
  const int max_pos = 16;
  const Tol t = tol_for(dt);

  std::vector<float> hq =
      rand_host(num_tokens * num_q_heads * head_dim, 88, -2.f, 2.f);
  std::vector<float> hk =
      rand_host(num_tokens * num_kv_heads * head_dim, 99, -2.f, 2.f);
  std::vector<float> hcos = rand_host(max_pos * half, 111, -1.f, 1.f);
  std::vector<float> hsin = rand_host(max_pos * half, 222, -1.f, 1.f);
  for (auto& v : hq) v = quantize(v, dt);
  for (auto& v : hk) v = quantize(v, dt);
  for (auto& v : hcos) v = quantize(v, dt);
  for (auto& v : hsin) v = quantize(v, dt);
  // Arbitrary positions into the [max_pos, half] cache.
  std::vector<int> hpos = {3, 0, 9, 15, 7};

  DeviceTensor q, k, cos, sin;
  q.alloc(hq.size(), dt);
  k.alloc(hk.size(), dt);
  cos.alloc(hcos.size(), dt);
  sin.alloc(hsin.size(), dt);
  q.upload(hq);
  k.upload(hk);
  cos.upload(hcos);
  sin.upload(hsin);

  int32_t* dpos = nullptr;
  CHECK_CUDA(cudaMalloc(&dpos, hpos.size() * sizeof(int32_t)));
  std::vector<int32_t> hpos32(hpos.begin(), hpos.end());
  CHECK_CUDA(cudaMemcpy(dpos, hpos32.data(), hpos32.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));

  CHECK_KS(ks_rope_gather(q.dptr, k.dptr, cos.dptr, sin.dptr, dpos, num_tokens,
                          num_q_heads, num_kv_heads, head_dim,
                          /*interleaved=*/0, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> refq = hq, refk = hk;
  ref_rope_apply(refq, hcos, hsin, hpos, num_tokens, num_q_heads, head_dim,
                 false, 1.f, dt);
  ref_rope_apply(refk, hcos, hsin, hpos, num_tokens, num_kv_heads, head_dim,
                 false, 1.f, dt);
  const std::string base = std::string("[") + dt_name(dt) + "]";
  check_allclose((std::string("rope.gather.q") + base).c_str(), q.download(),
                 refq, t.atol, t.rtol);
  check_allclose((std::string("rope.gather.k") + base).c_str(), k.download(),
                 refk, t.atol, t.rtol);
  cudaFree(dpos);
}

int main() {
  const char* kName = "test_rope";
  if (!has_cuda_device(kName)) return 0;
  for (ks_dtype_t dt : float_dtypes()) {
    test_rope_oop(dt, /*interleaved=*/false);
    test_rope_oop(dt, /*interleaved=*/true);
    test_rope_backward(dt);
    test_rope_gather(dt);
  }
  return finish(kName);
}
