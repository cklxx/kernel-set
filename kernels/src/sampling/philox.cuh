// kernel-set — counter-based Philox 4x32-10 RNG for reproducible sampling.
//
// A stateless, counter-based generator (Salmon et al., "Parallel Random Numbers:
// As Easy as 1, 2, 3"), the same scheme CUDA's curandStatePhilox4_32_10 and
// PyTorch/FlashInfer/vLLM use for GPU sampling. Being counter-based means the
// n-th draw of a stream is a pure function of (seed, offset, counter): identical
// (seed, philox_offset) inputs reproduce identical tokens regardless of how the
// batch is tiled across blocks/threads — exactly the reproducibility the ABI
// promises. We seed key = {lo(seed), hi(seed)} and counter with the per-sequence
// index plus philox_offset so each sequence gets an independent sub-stream.
#pragma once

#include "common/platform.cuh"

namespace ks {
namespace sampling {

// Philox bijection round constants (from the reference paper / curand).
KS_DI unsigned mulhilo32(unsigned a, unsigned b, unsigned* hip) {
  const unsigned long long product =
      static_cast<unsigned long long>(a) * static_cast<unsigned long long>(b);
  *hip = static_cast<unsigned>(product >> 32);
  return static_cast<unsigned>(product);
}

// One Philox-4x32 round. ctr is the 128-bit counter (4x u32), key the 64-bit key
// (2x u32). Ten rounds give a well-mixed bijection.
struct Philox4x32 {
  unsigned ctr[4];
  unsigned key[2];

  KS_DI void single_round() {
    const unsigned kPhiloxM0 = 0xD2511F53u;
    const unsigned kPhiloxM1 = 0xCD9E8D57u;
    unsigned hi0, hi1;
    const unsigned lo0 = mulhilo32(kPhiloxM0, ctr[0], &hi0);
    const unsigned lo1 = mulhilo32(kPhiloxM1, ctr[2], &hi1);
    unsigned out[4];
    out[0] = hi1 ^ ctr[1] ^ key[0];
    out[1] = lo1;
    out[2] = hi0 ^ ctr[3] ^ key[1];
    out[3] = lo0;
    ctr[0] = out[0];
    ctr[1] = out[1];
    ctr[2] = out[2];
    ctr[3] = out[3];
  }

  KS_DI void bump_key() {
    const unsigned kPhiloxW0 = 0x9E3779B9u;  // golden ratio
    const unsigned kPhiloxW1 = 0xBB67AE85u;  // sqrt(3) - 1
    key[0] += kPhiloxW0;
    key[1] += kPhiloxW1;
  }

  // Produce four 32-bit random words from the current counter/key.
  KS_DI void rand4(unsigned out[4]) {
    unsigned saved[4] = {ctr[0], ctr[1], ctr[2], ctr[3]};
    unsigned saved_key[2] = {key[0], key[1]};
#pragma unroll
    for (int i = 0; i < 10; ++i) {
      single_round();
      if (i < 9) bump_key();
    }
    out[0] = ctr[0];
    out[1] = ctr[1];
    out[2] = ctr[2];
    out[3] = ctr[3];
    // Restore so the object can be reused with an advanced counter if desired.
    ctr[0] = saved[0];
    ctr[1] = saved[1];
    ctr[2] = saved[2];
    ctr[3] = saved[3];
    key[0] = saved_key[0];
    key[1] = saved_key[1];
  }
};

// Construct a generator seeded by (seed, offset) with a per-subsequence id in the
// high counter words. seed -> key (lo/hi), subsequence -> ctr[2..3], draw index
// (offset) -> ctr[0..1].
KS_DI Philox4x32 make_philox(unsigned long long seed,
                             unsigned long long offset,
                             unsigned long long subsequence) {
  Philox4x32 p;
  p.key[0] = static_cast<unsigned>(seed);
  p.key[1] = static_cast<unsigned>(seed >> 32);
  p.ctr[0] = static_cast<unsigned>(offset);
  p.ctr[1] = static_cast<unsigned>(offset >> 32);
  p.ctr[2] = static_cast<unsigned>(subsequence);
  p.ctr[3] = static_cast<unsigned>(subsequence >> 32);
  return p;
}

// Convert a 32-bit word to a uniform float in [0, 1). Matches curand's scheme:
// 24 mantissa bits -> [0,1). Strictly < 1 so it can index a CDF safely.
KS_DI float u32_to_uniform(unsigned x) {
  // 2^-24, leaving the result in [0, 1) with 24 bits of resolution.
  const float kScale = 1.0f / 16777216.0f;  // 1 / 2^24
  return (static_cast<float>(x >> 8) + 0.5f) * kScale;
}

// One uniform float in [0,1) for a given (seed, offset, subsequence). We only
// need a single draw per sequence for inverse-CDF sampling.
KS_DI float philox_uniform(unsigned long long seed, unsigned long long offset,
                           unsigned long long subsequence) {
  Philox4x32 p = make_philox(seed, offset, subsequence);
  unsigned out[4];
  p.rand4(out);
  return u32_to_uniform(out[0]);
}

}  // namespace sampling
}  // namespace ks
