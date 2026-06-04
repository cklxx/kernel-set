// kernel-set — rotary position embeddings (RoPE).
//
// Implements ks_rope_inplace, ks_rope, ks_rope_gather, ks_rope_backward.
//
// Conventions (see rope_common.cuh for the math):
//   * Tensors are [num_tokens, num_heads, head_dim], contiguous over head_dim.
//   * cos/sin are [num_tokens, head_dim/2] (forward / backward) or caches
//     [max_pos, head_dim/2] gathered by an int32 `positions` array (gather).
//   * `interleaved != 0` selects GPT-J pairing (2j, 2j+1); otherwise NeoX
//     rotate_half pairing (j, j + head_dim/2).
//   * GQA: q has num_q_heads, k has num_kv_heads; both rotate with the *same*
//     per-token cos/sin (positions are per token, not per head).
//   * Backward rotates the incoming gradients by the conjugate rotation
//     (sin -> -sin), which is the transpose of the forward rotation matrix.
//
// Launch strategy
// ---------------
// One thread cooperatively rotates one (head, pair) for one token. We launch a
// 2D grid:
//   grid.x = num_tokens
//   grid.y = ceil(total_heads * (head_dim/2) / blockDim.x)
// and flatten (head, pair) into a single index inside the block so q and k are
// processed by one launch (k heads are appended after q heads). cos/sin for the
// token are loaded once per thread from the shared per-token row; for the common
// case head_dim <= 256 the whole row fits in L1/L2 and is reused across heads.
//
// Further tuning (documented, not implemented to keep the kernel branch-light):
//   * Cache the cos/sin row in shared memory once per block when many heads map
//     to the same token (reduces redundant global loads for large head counts).
//   * Vectorize the NeoX path with 128-bit loads of the two halves when
//     head_dim is a multiple of the vector width (cos/sin would also be
//     vectorized). Left as a follow-up; correctness-first here.
#include <string>

#include "kernel_set/rope.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

#include "rope_common.cuh"

namespace ks {
namespace rope {

constexpr int kBlock = 256;

// Unified RoPE kernel.
//
//   q_out/q_in : [num_tokens, num_q_heads, head_dim]   (q_in may alias q_out)
//   k_out/k_in : [num_tokens, num_kv_heads, head_dim]  (NULL => skip k)
//   cos/sin    : [stride_cs rows, half]  indexed by `positions[t]` if positions
//                != NULL, else by the token index t directly.
//   sign       : +1 forward, -1 backward (conjugate).
//
// Each thread handles one (head, pair) of one token. `flat` enumerates
// (q heads then k heads) x (half pairs).
template <typename scalar_t>
KS_GLOBAL void rope_kernel(scalar_t* __restrict__ q_out,
                           const scalar_t* __restrict__ q_in,
                           scalar_t* __restrict__ k_out,
                           const scalar_t* __restrict__ k_in,
                           const scalar_t* __restrict__ cos,
                           const scalar_t* __restrict__ sin,
                           const int32_t* __restrict__ positions,
                           int64_t num_tokens, int num_q_heads, int num_kv_heads,
                           int head_dim, int half, bool interleaved,
                           float sign) {
  const int64_t token = blockIdx.x;
  if (token >= num_tokens) return;

  const int total_heads = num_q_heads + num_kv_heads;
  const int64_t work_per_token = static_cast<int64_t>(total_heads) * half;

  // Per-token cos/sin row (shared across heads). For gather, the row is chosen
  // by positions[token]; otherwise it is the token row itself. Hoisted out of
  // the grid-stride loop since it is constant for the whole token.
  const int64_t cs_row =
      positions ? static_cast<int64_t>(positions[token]) : token;
  const scalar_t* cos_row = cos + cs_row * half;
  const scalar_t* sin_row = sin + cs_row * half;

  // Grid-stride over the (head, pair) work of this token. Striding over the y
  // dimension keeps the launch correct even when total_heads * half overflows
  // the 65535-block CUDA y/z grid limit (grid.y is capped on the host).
  const int64_t stride = static_cast<int64_t>(gridDim.y) * blockDim.x;
  for (int64_t flat =
           static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
       flat < work_per_token; flat += stride) {
    // Decompose flat -> (head, pair).
    const int head = static_cast<int>(flat / half);
    const int j = static_cast<int>(flat - static_cast<int64_t>(head) * half);

    const float cos_j = to_float(cos_row[j]);
    const float sin_j = to_float(sin_row[j]);

    // Select q vs k slice. q heads occupy [0, num_q_heads), k heads follow.
    scalar_t* out_head;
    const scalar_t* in_head;
    if (head < num_q_heads) {
      const int64_t base =
          (token * num_q_heads + head) * static_cast<int64_t>(head_dim);
      out_head = q_out + base;
      in_head = q_in + base;
    } else {
      const int kh = head - num_q_heads;
      const int64_t base =
          (token * num_kv_heads + kh) * static_cast<int64_t>(head_dim);
      out_head = k_out + base;
      in_head = k_in + base;
    }

    apply_pair<scalar_t>(out_head, in_head, j, half, interleaved, cos_j, sin_j,
                         sign);
  }
}

// Copy any non-rotated tail [rot, head_dim) for the out-of-place path. With
// rot == head_dim (the only case the ABI exposes today) this is a no-op launch
// and is skipped by the host, so it exists only for forward-compat / clarity.
template <typename scalar_t>
KS_GLOBAL void rope_copy_tail_kernel(scalar_t* __restrict__ q_out,
                                     const scalar_t* __restrict__ q_in,
                                     scalar_t* __restrict__ k_out,
                                     const scalar_t* __restrict__ k_in,
                                     int64_t num_tokens, int num_q_heads,
                                     int num_kv_heads, int head_dim, int rot) {
  const int tail = head_dim - rot;
  if (tail <= 0) return;
  const int64_t token = blockIdx.x;
  if (token >= num_tokens) return;

  const int total_heads = num_q_heads + num_kv_heads;
  const int64_t work = static_cast<int64_t>(total_heads) * tail;
  const int64_t stride = static_cast<int64_t>(gridDim.y) * blockDim.x;
  for (int64_t flat =
           static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
       flat < work; flat += stride) {
    const int head = static_cast<int>(flat / tail);
    const int d =
        rot + static_cast<int>(flat - static_cast<int64_t>(head) * tail);

    if (head < num_q_heads) {
      const int64_t base =
          (token * num_q_heads + head) * static_cast<int64_t>(head_dim);
      q_out[base + d] = q_in[base + d];
    } else {
      const int kh = head - num_q_heads;
      const int64_t base =
          (token * num_kv_heads + kh) * static_cast<int64_t>(head_dim);
      k_out[base + d] = k_in[base + d];
    }
  }
}

// Shared host launcher for every public entry point. q_in/k_in may alias the
// outputs (in-place / backward). `cos`/`sin`/`positions` are passed straight
// through. `sign` selects forward (+1) vs conjugate/backward (-1).
template <typename scalar_t>
static void launch_rope(scalar_t* q_out, const scalar_t* q_in, scalar_t* k_out,
                        const scalar_t* k_in, const scalar_t* cos,
                        const scalar_t* sin, const int32_t* positions,
                        int64_t num_tokens, int num_q_heads, int num_kv_heads,
                        int head_dim, bool interleaved, float sign,
                        ks::gpuStream_t stream) {
  const int half = head_dim / 2;
  const int active_kv = (k_out != nullptr) ? num_kv_heads : 0;
  const int total_heads = num_q_heads + active_kv;
  const int64_t work_per_token = static_cast<int64_t>(total_heads) * half;

  // grid.y is limited to 65535 on every CUDA arch; the kernel grid-strides over
  // the y dimension, so capping here stays correct for huge head counts.
  constexpr int64_t kMaxGridY = 65535;
  int64_t blocks_y = (work_per_token + kBlock - 1) / kBlock;
  if (blocks_y > kMaxGridY) blocks_y = kMaxGridY;
  if (blocks_y < 1) blocks_y = 1;
  const dim3 grid(static_cast<unsigned>(num_tokens),
                  static_cast<unsigned>(blocks_y));
  const dim3 block(kBlock);

  rope_kernel<scalar_t><<<grid, block, 0, stream>>>(
      q_out, q_in, k_out, k_in, cos, sin, positions, num_tokens, num_q_heads,
      active_kv, head_dim, half, interleaved, sign);
}

}  // namespace rope
}  // namespace ks

using namespace ks;

namespace {

// Common shape validation shared by every entry point. Returns KS_SUCCESS when
// args are valid, otherwise records a message and returns the error status.
ks_status_t validate_rope_args(const void* q, int64_t num_tokens,
                               int num_q_heads, int num_kv_heads, int head_dim,
                               bool k_present, const char* name) {
  if (q == nullptr) {
    ks::set_last_error(std::string(name) + ": null arg: q");
    return KS_ERROR_INVALID_ARGUMENT;
  }
  if (num_tokens <= 0 || num_q_heads <= 0 || head_dim <= 0) {
    ks::set_last_error(std::string(name) + ": bad shape");
    return KS_ERROR_INVALID_ARGUMENT;
  }
  if (k_present && num_kv_heads <= 0) {
    ks::set_last_error(std::string(name) +
                       ": num_kv_heads must be > 0 when k is provided");
    return KS_ERROR_INVALID_ARGUMENT;
  }
  if ((head_dim & 1) != 0) {
    ks::set_last_error(std::string(name) + ": head_dim must be even");
    return KS_ERROR_UNSUPPORTED_SHAPE;
  }
  // grid.x carries num_tokens; guard the 2^31-1 CUDA grid limit.
  if (num_tokens > 2147483647LL) {
    ks::set_last_error(std::string(name) + ": num_tokens exceeds grid limit");
    return KS_ERROR_UNSUPPORTED_SHAPE;
  }
  return KS_SUCCESS;
}

}  // namespace

extern "C" {

ks_status_t ks_rope_inplace(void* q, void* k, const void* cos, const void* sin,
                            int64_t num_tokens, int num_q_heads,
                            int num_kv_heads, int head_dim, int interleaved,
                            ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(cos);
  KS_CHECK_PTR(sin);
  const bool has_k = (k != nullptr);
  ks_status_t vs = validate_rope_args(q, num_tokens, num_q_heads, num_kv_heads,
                                      head_dim, has_k, "ks_rope_inplace");
  if (vs != KS_SUCCESS) return vs;

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_rope_inplace", {
    auto* qp = static_cast<scalar_t*>(q);
    auto* kp = static_cast<scalar_t*>(k);  // nullptr if has_k == false
    rope::launch_rope<scalar_t>(
        qp, qp, kp, kp, static_cast<const scalar_t*>(cos),
        static_cast<const scalar_t*>(sin), /*positions=*/nullptr, num_tokens,
        num_q_heads, num_kv_heads, head_dim, interleaved != 0, /*sign=*/1.0f, s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_rope(void* q_out, void* k_out, const void* q, const void* k,
                    const void* cos, const void* sin, int64_t num_tokens,
                    int num_q_heads, int num_kv_heads, int head_dim,
                    int interleaved, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(q_out);
  KS_CHECK_PTR(cos);
  KS_CHECK_PTR(sin);
  const bool has_k = (k_out != nullptr) || (k != nullptr);
  if (has_k) {
    // Both k_out and k must be supplied together.
    KS_CHECK_PTR(k_out);
    KS_CHECK_PTR(k);
  }
  ks_status_t vs = validate_rope_args(q, num_tokens, num_q_heads, num_kv_heads,
                                      head_dim, has_k, "ks_rope");
  if (vs != KS_SUCCESS) return vs;

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_rope", {
    rope::launch_rope<scalar_t>(
        static_cast<scalar_t*>(q_out), static_cast<const scalar_t*>(q),
        static_cast<scalar_t*>(k_out), static_cast<const scalar_t*>(k),
        static_cast<const scalar_t*>(cos), static_cast<const scalar_t*>(sin),
        /*positions=*/nullptr, num_tokens, num_q_heads, num_kv_heads, head_dim,
        interleaved != 0, /*sign=*/1.0f, s);
  });
  KS_CHECK_LAUNCH();
  // rot == head_dim today, so there is no tail to copy. (rope_copy_tail_kernel
  // is provided for forward-compat with partial rotary.)
  return KS_SUCCESS;
}

ks_status_t ks_rope_gather(void* q, void* k, const void* cos_cache,
                           const void* sin_cache, const int32_t* positions,
                           int64_t num_tokens, int num_q_heads, int num_kv_heads,
                           int head_dim, int interleaved, ks_dtype_t dtype,
                           ks_stream_t stream) {
  KS_CHECK_PTR(cos_cache);
  KS_CHECK_PTR(sin_cache);
  KS_CHECK_PTR(positions);
  const bool has_k = (k != nullptr);
  ks_status_t vs = validate_rope_args(q, num_tokens, num_q_heads, num_kv_heads,
                                      head_dim, has_k, "ks_rope_gather");
  if (vs != KS_SUCCESS) return vs;

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_rope_gather", {
    auto* qp = static_cast<scalar_t*>(q);
    auto* kp = static_cast<scalar_t*>(k);
    rope::launch_rope<scalar_t>(
        qp, qp, kp, kp, static_cast<const scalar_t*>(cos_cache),
        static_cast<const scalar_t*>(sin_cache), positions, num_tokens,
        num_q_heads, num_kv_heads, head_dim, interleaved != 0, /*sign=*/1.0f, s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_rope_backward(void* grad_q, void* grad_k, const void* cos,
                             const void* sin, int64_t num_tokens,
                             int num_q_heads, int num_kv_heads, int head_dim,
                             int interleaved, ks_dtype_t dtype,
                             ks_stream_t stream) {
  KS_CHECK_PTR(cos);
  KS_CHECK_PTR(sin);
  const bool has_k = (grad_k != nullptr);
  ks_status_t vs = validate_rope_args(grad_q, num_tokens, num_q_heads,
                                      num_kv_heads, head_dim, has_k,
                                      "ks_rope_backward");
  if (vs != KS_SUCCESS) return vs;

  auto s = to_stream(stream);
  // Backward = conjugate rotation (sin -> -sin), applied in place to the grads.
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_rope_backward", {
    auto* gq = static_cast<scalar_t*>(grad_q);
    auto* gk = static_cast<scalar_t*>(grad_k);
    rope::launch_rope<scalar_t>(
        gq, gq, gk, gk, static_cast<const scalar_t*>(cos),
        static_cast<const scalar_t*>(sin), /*positions=*/nullptr, num_tokens,
        num_q_heads, num_kv_heads, head_dim, interleaved != 0, /*sign=*/-1.0f,
        s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
