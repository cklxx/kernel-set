// kernel-set — correctness test: ks_embedding_lookup gather vs host reference,
// for both int32 and int64 index buffers. out[i,:] = table[idx[i],:].
#include "test_common.cuh"

#include "kernel_set/embedding.h"

using namespace kst;

static void test_lookup(ks_dtype_t dt, bool i64) {
  const int64_t vocab = 50, embed = 96, tokens = 17;
  const Tol t = tol_for(dt);

  std::vector<float> htable = rand_host(vocab * embed, 71, -2.f, 2.f);
  for (auto& v : htable) v = quantize(v, dt);

  // Deterministic indices, including repeats (which the kernel must handle).
  std::vector<int32_t> idx32(static_cast<size_t>(tokens));
  std::vector<int64_t> idx64(static_cast<size_t>(tokens));
  for (int64_t i = 0; i < tokens; ++i) {
    const int32_t id = static_cast<int32_t>((i * 7 + 3) % vocab);
    idx32[static_cast<size_t>(i)] = id;
    idx64[static_cast<size_t>(i)] = id;
  }

  DeviceTensor table, out;
  table.alloc(vocab * embed, dt);
  out.alloc(tokens * embed, dt);
  table.upload(htable);

  void* didx = nullptr;
  if (i64) {
    CHECK_CUDA(cudaMalloc(&didx, idx64.size() * sizeof(int64_t)));
    CHECK_CUDA(cudaMemcpy(didx, idx64.data(), idx64.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
  } else {
    CHECK_CUDA(cudaMalloc(&didx, idx32.size() * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(didx, idx32.data(), idx32.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
  }

  CHECK_KS(ks_embedding_lookup(out.dptr, table.dptr, didx, i64 ? 1 : 0, tokens,
                               embed, dt, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> ref(static_cast<size_t>(tokens * embed));
  for (int64_t i = 0; i < tokens; ++i) {
    const int64_t row = idx64[static_cast<size_t>(i)];
    for (int64_t c = 0; c < embed; ++c) {
      ref[static_cast<size_t>(i * embed + c)] =
          htable[static_cast<size_t>(row * embed + c)];
    }
  }

  std::vector<float> got = out.download();
  const std::string name = std::string("embedding_lookup[") + dt_name(dt) +
                           (i64 ? ",i64]" : ",i32]");
  check_allclose(name.c_str(), got, ref, t.atol, t.rtol);

  cudaFree(didx);
}

int main() {
  const char* kName = "test_embedding";
  if (!has_cuda_device(kName)) return 0;
  for (ks_dtype_t dt : float_dtypes()) {
    test_lookup(dt, /*i64=*/false);
    test_lookup(dt, /*i64=*/true);
  }
  return finish(kName);
}
