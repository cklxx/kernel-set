// kernel-set — rotary position embedding (RoPE) shared device helpers.
//
// All four public RoPE entry points (in-place, out-of-place, gathered, and
// backward) reduce to one elementwise rotation of (cos, sin) pairs over the
// head_dim axis. This header factors out the two pairing conventions and the
// per-pair complex rotation so every .cu file applies *identical* math:
//
//   NeoX / "rotate_half" (interleaved == 0):
//     pair j (0 <= j < rot/2) is (x[j], x[j + rot/2])
//   GPT-J / interleaved (interleaved != 0):
//     pair j is (x[2j], x[2j + 1])
//
//   forward rotation by angle theta_j (cos_j = cos theta_j, sin_j = sin theta_j):
//     x0' = x0 * cos_j - x1 * sin_j
//     x1' = x1 * cos_j + x0 * sin_j
//
//   backward / gradient propagation is the conjugate rotation (sin -> -sin):
//     g0' = g0 * cos_j + g1 * sin_j
//     g1' = g1 * cos_j - g0 * sin_j
//   which the callers obtain by passing a negated `sign` to apply_pair().
//
// cos/sin are stored as [*, rot/2] (one entry per pair) and are *shared across
// heads* for a given token, matching HF transformers / vLLM / FlashAttention.
//
// Notes on generality:
//   * Only the first `rot` (== head_dim here, but kept distinct for clarity and
//     future partial-rotary support) dimensions are rotated; any tail is copied.
//   * All arithmetic accumulates in fp32 via ks::to_float / ks::from_float so
//     f16/bf16 storage matches the fp32 reference to within rounding.
#pragma once

#include "common/platform.cuh"
#include "common/dtype.cuh"

namespace ks {
namespace rope {

// Locate the two element offsets (within a head) and the cos/sin pair index for
// pair `j` under the selected convention.
struct PairLayout {
  int idx0;  // offset of the first  element of the pair within the head
  int idx1;  // offset of the second element of the pair within the head
  int cs;    // index into the [rot/2] cos/sin row
};

KS_DI PairLayout pair_layout(int j, int half, bool interleaved) {
  PairLayout p;
  if (interleaved) {
    // GPT-J: consecutive elements (2j, 2j+1).
    p.idx0 = 2 * j;
    p.idx1 = 2 * j + 1;
  } else {
    // NeoX / rotate_half: (j, j + half).
    p.idx0 = j;
    p.idx1 = j + half;
  }
  p.cs = j;
  return p;
}

// Apply the rotation to one pair read from `in_head`, writing to `out_head`.
//   sign = +1.0f  -> forward rotation
//   sign = -1.0f  -> conjugate rotation (backward)
// `cos_j`/`sin_j` are the (cos, sin) for pair `j` of this token.
template <typename scalar_t>
KS_DI void apply_pair(scalar_t* __restrict__ out_head,
                      const scalar_t* __restrict__ in_head, int j, int half,
                      bool interleaved, float cos_j, float sin_j, float sign) {
  const PairLayout p = pair_layout(j, half, interleaved);
  const float x0 = to_float(in_head[p.idx0]);
  const float x1 = to_float(in_head[p.idx1]);
  const float s = sign * sin_j;
  out_head[p.idx0] = from_float<scalar_t>(x0 * cos_j - x1 * s);
  out_head[p.idx1] = from_float<scalar_t>(x1 * cos_j + x0 * s);
}

}  // namespace rope
}  // namespace ks
