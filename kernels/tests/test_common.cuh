// kernel-set — tiny dependency-free C++ test harness for the kernel suite.
//
// These tests launch a public C ABI kernel on the GPU and compare the result to
// a plain-C++ reference computed on the host, within a tolerance. There is no
// gtest dependency on purpose (the repo must build with only nvcc + CMake): we
// provide a minimal CHECK / CHECK_CLOSE macro set, a couple of host<->device
// memory helpers, and a fp16/bf16 host reference that mirrors the kernels'
// "accumulate in fp32, store in storage dtype" convention.
//
// Each test file defines `int main()` that runs one or more cases and returns
// non-zero on the first failure so CTest reports it. If no CUDA device is
// present at runtime the test prints a notice and returns 0 (skips) — the build
// still proves the kernels compile and link against the public ABI.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kernel_set/types.h"
#include "kernel_set/runtime.h"

namespace kst {

// ---------------------------------------------------------------------------
// Failure plumbing. A failing CHECK records the message and aborts the test
// process with a non-zero status so CTest marks the case failed.
// ---------------------------------------------------------------------------
inline int& fail_count() {
  static int n = 0;
  return n;
}

#define KST_FAILF(...)                                       \
  do {                                                       \
    std::fprintf(stderr, "[FAIL] %s:%d: ", __FILE__, __LINE__); \
    std::fprintf(stderr, __VA_ARGS__);                       \
    std::fprintf(stderr, "\n");                              \
    ++::kst::fail_count();                                   \
  } while (0)

// Boolean predicate.
#define CHECK(cond)                            \
  do {                                         \
    if (!(cond)) KST_FAILF("CHECK(%s)", #cond); \
  } while (0)

// CUDA API call must return cudaSuccess.
#define CHECK_CUDA(expr)                                              \
  do {                                                               \
    cudaError_t _e = (expr);                                         \
    if (_e != cudaSuccess)                                           \
      KST_FAILF("CUDA error %s: %s", #expr, cudaGetErrorString(_e)); \
  } while (0)

// kernel-set C ABI call must return KS_SUCCESS.
#define CHECK_KS(expr)                                                       \
  do {                                                                       \
    ks_status_t _s = (expr);                                                 \
    if (_s != KS_SUCCESS)                                                    \
      KST_FAILF("ks status %s = %s (%s)", #expr, ks_status_string(_s),       \
                ks_last_error_string());                                     \
  } while (0)

// Scalar near-equality with absolute+relative tolerance.
inline bool close(float a, float b, float atol, float rtol) {
  if (std::isnan(a) || std::isnan(b)) return false;
  const float diff = std::fabs(a - b);
  return diff <= atol + rtol * std::fabs(b);
}

// Compare a host result vector against a host reference, elementwise.
inline void check_allclose(const char* what, const std::vector<float>& got,
                           const std::vector<float>& ref, float atol,
                           float rtol) {
  if (got.size() != ref.size()) {
    KST_FAILF("%s: size mismatch got=%zu ref=%zu", what, got.size(),
              ref.size());
    return;
  }
  float max_abs = 0.f;
  int bad = -1;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float d = std::fabs(got[i] - ref[i]);
    if (d > max_abs) max_abs = d;
    if (!close(got[i], ref[i], atol, rtol) && bad < 0) bad = static_cast<int>(i);
  }
  if (bad >= 0) {
    KST_FAILF("%s: mismatch at [%d] got=%.6g ref=%.6g (max_abs=%.6g, atol=%g, "
              "rtol=%g)",
              what, bad, got[static_cast<size_t>(bad)],
              ref[static_cast<size_t>(bad)], max_abs, atol, rtol);
  } else {
    std::printf("[ok] %s: max_abs=%.3g (n=%zu)\n", what, max_abs, ref.size());
  }
}

// ---------------------------------------------------------------------------
// Storage-dtype host round-trip. Tests build inputs in fp32, optionally narrow
// them to the kernel's storage dtype on the host (matching what the GPU sees),
// and read results back through the same conversion so the reference and the
// kernel agree bit-for-bit on the rounding step.
// ---------------------------------------------------------------------------
inline float quantize(float v, ks_dtype_t dt) {
  switch (dt) {
    case KS_DTYPE_F16:
      return __half2float(__float2half_rn(v));
    case KS_DTYPE_BF16:
      return __bfloat162float(__float2bfloat16_rn(v));
    default:
      return v;  // f32 stored exactly
  }
}

inline size_t dtype_bytes(ks_dtype_t dt) {
  switch (dt) {
    case KS_DTYPE_F32:
      return 4;
    case KS_DTYPE_F16:
    case KS_DTYPE_BF16:
      return 2;
    default:
      return 4;
  }
}

// Pack a host fp32 vector into the storage dtype's byte layout.
inline std::vector<uint8_t> pack(const std::vector<float>& host, ks_dtype_t dt) {
  const size_t n = host.size();
  std::vector<uint8_t> raw(n * dtype_bytes(dt));
  if (dt == KS_DTYPE_F32) {
    std::memcpy(raw.data(), host.data(), n * sizeof(float));
  } else if (dt == KS_DTYPE_F16) {
    auto* p = reinterpret_cast<__half*>(raw.data());
    for (size_t i = 0; i < n; ++i) p[i] = __float2half_rn(host[i]);
  } else if (dt == KS_DTYPE_BF16) {
    auto* p = reinterpret_cast<__nv_bfloat16*>(raw.data());
    for (size_t i = 0; i < n; ++i) p[i] = __float2bfloat16_rn(host[i]);
  }
  return raw;
}

// Unpack the storage dtype's byte layout back into host fp32.
inline std::vector<float> unpack(const std::vector<uint8_t>& raw, ks_dtype_t dt,
                                 size_t n) {
  std::vector<float> host(n);
  if (dt == KS_DTYPE_F32) {
    std::memcpy(host.data(), raw.data(), n * sizeof(float));
  } else if (dt == KS_DTYPE_F16) {
    auto* p = reinterpret_cast<const __half*>(raw.data());
    for (size_t i = 0; i < n; ++i) host[i] = __half2float(p[i]);
  } else if (dt == KS_DTYPE_BF16) {
    auto* p = reinterpret_cast<const __nv_bfloat16*>(raw.data());
    for (size_t i = 0; i < n; ++i) host[i] = __bfloat162float(p[i]);
  }
  return host;
}

// A device buffer holding `n` elements of `dt`, initialized from a host fp32
// vector (narrowed to `dt`). RAII frees on scope exit.
struct DeviceTensor {
  void* dptr = nullptr;
  size_t nbytes = 0;
  ks_dtype_t dt = KS_DTYPE_F32;
  size_t n = 0;

  DeviceTensor() = default;

  void alloc(size_t count, ks_dtype_t dtype) {
    dt = dtype;
    n = count;
    nbytes = count * dtype_bytes(dtype);
    CHECK_CUDA(cudaMalloc(&dptr, nbytes));
  }

  void upload(const std::vector<float>& host) {
    std::vector<uint8_t> raw = pack(host, dt);
    CHECK_CUDA(cudaMemcpy(dptr, raw.data(), raw.size(), cudaMemcpyHostToDevice));
  }

  std::vector<float> download() const {
    std::vector<uint8_t> raw(nbytes);
    CHECK_CUDA(cudaMemcpy(raw.data(), dptr, nbytes, cudaMemcpyDeviceToHost));
    return unpack(raw, dt, n);
  }

  ~DeviceTensor() {
    if (dptr) cudaFree(dptr);
  }

  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
};

// Deterministic pseudo-random fp32 host data in a comfortable range so the
// reference vs. kernel comparison is stable across runs (no GPU RNG involved).
inline std::vector<float> rand_host(size_t n, unsigned seed, float lo = -2.f,
                                    float hi = 2.f) {
  std::vector<float> v(n);
  // xorshift32 — small, deterministic, no <random> locale surprises.
  uint32_t s = seed ? seed : 0x9e3779b9u;
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const float u = static_cast<float>(s) / 4294967295.0f;  // [0,1]
    v[i] = lo + u * (hi - lo);
  }
  return v;
}

// Returns false (and prints a skip notice) when no CUDA device is usable, so a
// test can early-return 0. This keeps `ctest` green on machines without a GPU
// while still exercising compile + link of the kernels under test.
inline bool has_cuda_device(const char* test_name) {
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess || count == 0) {
    std::printf("[skip] %s: no CUDA device (%s)\n", test_name,
                e == cudaSuccess ? "count==0" : cudaGetErrorString(e));
    return false;
  }
  return true;
}

// Standard tail for a test's main(): print and return the right code.
inline int finish(const char* test_name) {
  if (fail_count() == 0) {
    std::printf("[PASS] %s\n", test_name);
    return 0;
  }
  std::printf("[FAILED] %s: %d failure(s)\n", test_name, fail_count());
  return 1;
}

// Tolerances by storage dtype: fp32 is tight; fp16/bf16 are loose enough for
// the storage rounding the kernel applies on read and write.
struct Tol {
  float atol;
  float rtol;
};
inline Tol tol_for(ks_dtype_t dt) {
  switch (dt) {
    case KS_DTYPE_F32:
      return {1e-4f, 1e-4f};
    case KS_DTYPE_F16:
      return {3e-3f, 3e-3f};
    case KS_DTYPE_BF16:
      return {2e-2f, 2e-2f};
    default:
      return {1e-3f, 1e-3f};
  }
}

// The set of floating storage dtypes every kernel under test must support.
inline const std::vector<ks_dtype_t>& float_dtypes() {
  static const std::vector<ks_dtype_t> v = {KS_DTYPE_F32, KS_DTYPE_F16,
                                            KS_DTYPE_BF16};
  return v;
}

inline const char* dt_name(ks_dtype_t dt) { return ks_dtype_name(dt); }

}  // namespace kst
