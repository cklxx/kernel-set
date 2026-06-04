// kernel-set — vectorized (128-bit) load/store helpers.
//
// Coalesced wide memory access for elementwise/normalization kernels. `Vec<T,N>`
// packs N scalars into an aligned struct that the compiler lowers to a single
// 128-bit transaction when alignment permits.
#pragma once

#include "common/platform.cuh"

namespace ks {

template <typename T, int N>
struct alignas(sizeof(T) * N) Vec {
  T data[N];
  KS_HDI T& operator[](int i) { return data[i]; }
  KS_HDI const T& operator[](int i) const { return data[i]; }
};

// Number of T that fit in a 128-bit vector (min 1).
template <typename T>
struct VecWidth {
  static constexpr int value = (16 / sizeof(T)) > 0 ? (16 / sizeof(T)) : 1;
};

template <typename T, int N>
KS_DI Vec<T, N> load_vec(const T* ptr) {
  return *reinterpret_cast<const Vec<T, N>*>(ptr);
}

template <typename T, int N>
KS_DI void store_vec(T* ptr, const Vec<T, N>& v) {
  *reinterpret_cast<Vec<T, N>*>(ptr) = v;
}

// True if `ptr` and `count` are both aligned to a width-N vector of T.
template <typename T, int N>
KS_HDI bool is_aligned(const void* ptr, int64_t count) {
  return (reinterpret_cast<uintptr_t>(ptr) % (sizeof(T) * N) == 0) &&
         (count % N == 0);
}

}  // namespace ks
