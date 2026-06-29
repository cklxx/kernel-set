/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_FASTDIV_CUH_
#define FLASHINFER_FASTDIV_CUH_
#include <cstdint>

namespace flashinfer {

// Self-contained fast_mod_div for CUDA 11.x compatibility.
// cuda::fast_mod_div is only available in CTK 12.0+.
struct fast_mod_div {
  uint32_t d_, m_, s_;
  __host__ __device__ fast_mod_div() : d_(1), m_(0), s_(0) {}
  __host__ fast_mod_div(uint32_t d) : d_(d ? d : 1) {
    // Compute m, s such that n / d == (n * m) >> (32 + s) for n in [0, 2^32).
    // Standard libdivide-like algorithm for 32-bit unsigned.
    if (d_ == 1) { m_ = 0; s_ = 0; return; }
    s_ = 31;
    uint64_t one = 1;
    uint64_t n_hi = ((one << 32) + d_ - 1) / d_ - 1;
    while (true) {
      uint64_t m_candidate = (n_hi + 1) << 1;
      if (m_candidate >= (one << 32) || m_candidate * d_ >= (one << 32)) {
        m_ = static_cast<uint32_t>(m_candidate);
        break;
      }
      ++s_;
      n_hi = n_hi * 2 + 1;
    }
  }
  __host__ __device__ __forceinline__ uint32_t div(uint32_t n) const {
    if (d_ == 1) return n;
    uint64_t hi = (static_cast<uint64_t>(m_) * n) >> 32;
    return static_cast<uint32_t>(hi >> s_);
  }
  __host__ __device__ __forceinline__ uint32_t mod(uint32_t n) const {
    return n - div(n) * d_;
  }
};

// API-compatible wrapper around fast_mod_div<uint32_t>.
// Preserves the default constructor, implicit conversions, and divmod()
// method expected by existing call sites throughout the attention kernels.
struct uint_fastdiv {
  __host__ __device__ uint_fastdiv() : impl_(1), d_(0) {}

  __host__ uint_fastdiv(uint32_t d) : impl_(d ? d : 1), d_(d) {}

  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }

  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    q = impl_.div(n);
    r = n - q * d_;
  }

 private:
  fast_mod_div impl_;
  uint32_t d_;
};

__host__ __device__ __forceinline__ uint32_t operator/(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return q;
}

__host__ __device__ __forceinline__ uint32_t operator%(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return r;
}

}  // namespace flashinfer

#endif  // FLASHINFER_FASTDIV_CUH_
