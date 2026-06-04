// kernel-set — combined temperature + top-k + top-p (nucleus) token sampling.
//
// One CUDA block samples one sequence's row of `vocab_size` logits. The pipeline
// mirrors FlashInfer / vLLM's fused sampler:
//
//   1. temperature scale:  z = logit / T          (T<=0 => greedy-ish, T=1)
//   2. optional top-k:      keep only the k largest z (threshold by value)
//   3. safe softmax:        p = softmax(z) over the surviving candidates
//   4. optional top-p:      keep the smallest nucleus whose mass >= top_p
//   5. inverse-CDF sample:  draw u ~ U[0,1), pick first index whose renormalized
//                           cumulative prob crosses u.
//
// top-k and top-p are realized with *threshold binary searches* rather than a
// full sort, which is the standard way to do fused single-pass nucleus sampling
// on GPU without a per-sequence workspace (FlashInfer "sampling from probability
// thresholds"). Each refinement pass is a block reduction over the row; we cap
// the number of passes so the cost is O(vocab * log(range)) with a small const.
//
// Randomness is a counter-based Philox stream keyed by (seed, philox_offset) and
// indexed by the sequence id, so the same inputs reproduce the same tokens no
// matter how the batch is tiled (see philox.cuh).
#include "kernel_set/sampling.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"
#include "philox.cuh"

namespace ks {
namespace sampling {

constexpr int kSampleBlock = 256;
// Binary-search passes for the top-k / top-p thresholds. fp32 has 24 mantissa
// bits; ~40 bisections drive the relative bracket far below fp32 resolution so
// the threshold is exact to floating point for any realistic vocab.
constexpr int kThreshIters = 40;

// Shared state used to broadcast scalars between the cooperating threads.
struct SampleScratch {
  float reduce[kSampleBlock / KS_WARP_SIZE];
  float scan[kSampleBlock];
  float bcast;     // generic scalar broadcast slot
  int count;       // candidate counter for top-k bracket search
  int token;       // chosen token (written by the crossing thread)
  float token_p;   // its renormalized probability
};

// Block inclusive prefix sum over `val`, returning this thread's inclusive sum.
// `block_total` (out) receives the sum across the whole block. Uses the scan
// scratch (one float per thread). Deterministic and order-stable.
KS_DI float block_inclusive_scan(float val, float* scan, float* block_total) {
  const int tid = threadIdx.x;
  scan[tid] = val;
  __syncthreads();
  // Hillis-Steele inclusive scan.
  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    float add = 0.f;
    if (tid >= offset) add = scan[tid - offset];
    __syncthreads();
    scan[tid] += add;
    __syncthreads();
  }
  *block_total = scan[blockDim.x - 1];
  const float inclusive = scan[tid];
  __syncthreads();
  return inclusive;
}

template <typename scalar_t>
KS_GLOBAL void sample_kernel(int32_t* __restrict__ out_tokens,
                             float* __restrict__ out_probs,
                             const scalar_t* __restrict__ logits,
                             const float* __restrict__ temperatures,
                             const int32_t* __restrict__ top_ks,
                             const float* __restrict__ top_ps, int64_t vocab,
                             unsigned long long seed,
                             unsigned long long philox_offset) {
  const int64_t seq = blockIdx.x;
  const scalar_t* row = logits + seq * vocab;

  __shared__ SampleScratch sh;

  // --- Per-sequence parameters -------------------------------------------
  const float temp = temperatures ? temperatures[seq] : 1.0f;
  const float inv_temp = (temp > 0.f) ? (1.0f / temp) : 1.0f;
  const int top_k = top_ks ? top_ks[seq] : 0;            // <=0 => disabled
  const float top_p = top_ps ? top_ps[seq] : 0.0f;       // <=0 => disabled
  const bool use_topk = (top_k > 0 && top_k < vocab);
  const bool use_topp = (top_p > 0.0f && top_p < 1.0f);

  // --- Row max & min of temperature-scaled logits ------------------------
  float local_max = -3.4028235e38f;
  float local_min = 3.4028235e38f;
  for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float z = to_float(row[i]) * inv_temp;
    local_max = z > local_max ? z : local_max;
    local_min = z < local_min ? z : local_min;
  }
  const float row_max = block_reduce_max(local_max, sh.reduce);
  __syncthreads();
  const float row_min = -block_reduce_max(-local_min, sh.reduce);
  __syncthreads();

  // --- top-k threshold (a logit cutoff that admits >= k candidates) ------
  // Bisect a logit threshold t in [row_min, row_max]: count z >= t. We want the
  // largest t with count >= top_k, i.e. the k-th largest value (ties keep all
  // tied entries, matching FlashInfer). If top-k disabled, cutoff = -inf.
  float topk_cut = -3.4028235e38f;
  if (use_topk) {
    float lo = row_min;
    float hi = row_max;
    for (int it = 0; it < kThreshIters; ++it) {
      const float mid = 0.5f * (lo + hi);
      int local_cnt = 0;
      for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float z = to_float(row[i]) * inv_temp;
        if (z >= mid) ++local_cnt;
      }
      // Block-sum the count via the float reduce buffer.
      const float cntf = block_reduce_sum(static_cast<float>(local_cnt),
                                          sh.reduce);
      __syncthreads();
      const int cnt = static_cast<int>(cntf);
      if (cnt >= top_k) {
        lo = mid;  // can raise the bar and still satisfy >= k
      } else {
        hi = mid;  // too strict, lower the bar
      }
    }
    topk_cut = lo;  // include z >= topk_cut
  }

  // --- softmax denominator over the top-k-surviving candidates -----------
  float local_sum = 0.f;
  for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float z = to_float(row[i]) * inv_temp;
    if (z >= topk_cut) local_sum += __expf(z - row_max);
  }
  float denom = block_reduce_sum(local_sum, sh.reduce);
  __syncthreads();
  const float inv_denom = 1.0f / denom;

  // --- top-p (nucleus) probability cutoff --------------------------------
  // Among the surviving probabilities p_i = exp(z-row_max)/denom, find the
  // largest probability pivot `pc` such that sum_{p_i >= pc} p_i >= top_p.
  // Masking p_i < pc keeps the smallest set of highest-prob tokens whose mass
  // covers top_p (the nucleus). Disabled => pivot 0 (keep everything).
  float topp_cut = 0.0f;
  if (use_topp) {
    float lo = 0.0f;  // keeps all -> mass == 1 >= top_p
    float hi = 1.0f;  // keeps (almost) nothing
    for (int it = 0; it < kThreshIters; ++it) {
      const float mid = 0.5f * (lo + hi);
      float local_mass = 0.f;
      for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float z = to_float(row[i]) * inv_temp;
        if (z >= topk_cut) {
          const float p = __expf(z - row_max) * inv_denom;
          if (p >= mid) local_mass += p;
        }
      }
      const float mass = block_reduce_sum(local_mass, sh.reduce);
      __syncthreads();
      if (mass >= top_p) {
        lo = mid;  // can tighten (raise pivot) and still cover top_p
      } else {
        hi = mid;  // too tight, relax
      }
    }
    topp_cut = lo;  // include p >= topp_cut
  }

  // --- renormalize over the final candidate set --------------------------
  // Candidate iff z >= topk_cut AND p >= topp_cut.
  float local_keep = 0.f;
  for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float z = to_float(row[i]) * inv_temp;
    if (z >= topk_cut) {
      const float p = __expf(z - row_max) * inv_denom;
      if (p >= topp_cut) local_keep += p;
    }
  }
  float keep_mass = block_reduce_sum(local_keep, sh.reduce);
  __syncthreads();
  // Guard against degenerate empty set (shouldn't happen: argmax always passes).
  if (keep_mass <= 0.f) keep_mass = 1.0f;
  const float inv_keep = 1.0f / keep_mass;

  // --- draw target = u in [0, 1) over the renormalized candidate CDF -----
  if (threadIdx.x == 0) {
    sh.bcast = philox_uniform(seed, philox_offset,
                              static_cast<unsigned long long>(seq));
    sh.token = -1;
    sh.token_p = 0.f;
  }
  __syncthreads();
  const float u = sh.bcast;

  // --- tiled inverse-CDF scan --------------------------------------------
  // Walk the vocab in tiles of blockDim.x, maintaining a running prefix of the
  // renormalized candidate mass. The first index whose inclusive prefix crosses
  // u is the sampled token. Deterministic (index order), single uniform draw.
  float running = 0.f;  // mass before this tile (replicated in all threads)
  for (int64_t base = 0; base < vocab; base += blockDim.x) {
    const int64_t i = base + threadIdx.x;
    float pi = 0.f;
    if (i < vocab) {
      const float z = to_float(row[i]) * inv_temp;
      if (z >= topk_cut) {
        const float p = __expf(z - row_max) * inv_denom;
        if (p >= topp_cut) pi = p * inv_keep;
      }
    }
    float tile_total = 0.f;
    const float incl = block_inclusive_scan(pi, sh.scan, &tile_total);
    const float prefix = running + incl;   // inclusive CDF up to and incl. i
    const float excl = prefix - pi;        // CDF strictly before i
    // i is selected iff excl <= u < prefix and i is a live candidate.
    if (i < vocab && pi > 0.f && sh.token < 0 && u >= excl && u < prefix) {
      // Multiple threads can't both satisfy this for the same u (disjoint
      // intervals), so a plain write is race-free across the block.
      sh.token = static_cast<int>(i);
      sh.token_p = pi * keep_mass;  // report the *unrenormalized* candidate prob
    }
    __syncthreads();
    running += tile_total;
    if (sh.token >= 0) break;  // found it; all threads agree via the sync above
  }

  // --- finalize: fallback to the last candidate if rounding left token<0 -
  if (threadIdx.x == 0) {
    int tok = sh.token;
    float tp = sh.token_p;
    if (tok < 0) {
      // u landed in the tail past the accumulated mass (fp rounding). Fall back
      // to the highest-prob token: argmax of the scaled logits.
      tok = 0;
      // Re-derive argmax cheaply is overkill here; use index of row_max via a
      // last single-thread scan over candidates. vocab is bounded per call.
      float best = -3.4028235e38f;
      for (int64_t i = 0; i < vocab; ++i) {
        const float z = to_float(row[i]) * inv_temp;
        if (z > best) {
          best = z;
          tok = static_cast<int>(i);
        }
      }
      const float p = __expf(best - row_max) * inv_denom;
      tp = p;
    }
    out_tokens[seq] = tok;
    if (out_probs) out_probs[seq] = tp;
  }
}

}  // namespace sampling
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_sample(int32_t* out_tokens, float* out_probs, const void* logits,
                      const float* temperatures, const int32_t* top_ks,
                      const float* top_ps, int64_t num_seqs, int64_t vocab_size,
                      uint64_t seed, uint64_t philox_offset, ks_dtype_t dtype,
                      ks_stream_t stream) {
  KS_CHECK_PTR(out_tokens);
  KS_CHECK_PTR(logits);
  // out_probs, temperatures, top_ks, top_ps are all optional (NULL allowed).
  if (num_seqs <= 0 || vocab_size <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_sample: bad shape");

  const dim3 grid(static_cast<unsigned>(num_seqs));
  const dim3 block(sampling::kSampleBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_sample", {
    sampling::sample_kernel<scalar_t><<<grid, block, 0, s>>>(
        out_tokens, out_probs, static_cast<const scalar_t*>(logits),
        temperatures, top_ks, top_ps, vocab_size,
        static_cast<unsigned long long>(seed),
        static_cast<unsigned long long>(philox_offset));
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
