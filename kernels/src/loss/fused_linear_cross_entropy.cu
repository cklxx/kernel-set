// kernel-set — fused-linear-cross-entropy (FLCE).
//
// Modeled on Liger-Kernel's `fused_linear_cross_entropy_forward`
// (src/liger_kernel/ops/fused_linear_cross_entropy.py). Computes cross-entropy
// directly from hidden states and the LM-head weight WITHOUT ever materializing
// the full [num_tokens, vocab] logit matrix. Tokens are processed in chunks of
// `chunk_size`; only a [chunk_size, vocab] fp32 scratch buffer is allocated.
//
// For each chunk:
//   1) logits_chunk = hidden_chunk @ weight^T            (scratch, fp32)
//   2) CE per row over logits_chunk -> losses[], and
//      grad of logits written back in-place into logits_chunk
//      (grad[j] = softmax[j] - (1-ls)*onehot[j] - ls/V)
//   3) grad_hidden_chunk = grad_logits_chunk @ weight    (shape [cs, H])
//   4) grad_weight_fp32  += grad_logits_chunk^T @ hidden_chunk  ([V, H], fp32)
//
// Reduction over non-ignored tokens is left to the caller (no 1/n scaling).
//
// Implementation notes / further tuning:
//   * The three matmuls use simple shared-memory tiled GEMM kernels (no cuBLAS
//     dependency so this stays a single self-contained .cu). For peak perf one
//     would dispatch to CUTLASS / cuBLASLt or a wmma/tensor-core epilogue, and
//     fuse the logit GEMM with the CE pass (cp.async double-buffer over K). The
//     tiled kernels here are correct and coalesced; documented as the tuning
//     hook.
//   * grad_weight_fp32 is zeroed once up front and accumulated across chunks
//     (matching Liger's `grad_weight = zeros_like(...)`).
#include "kernel_set/loss.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"
#include "loss/loss_common.cuh"

namespace ks {
namespace loss {

constexpr int kTile = 16;          // tiled-GEMM tile dim
constexpr int kCeRowBlock = 256;   // threads per row for the CE pass

// ---------------------------------------------------------------------------
// 1) logits_chunk[cs, V] = hidden_chunk[cs, H] @ weight[V, H]^T
//    (weight is row-major [V, H]; logits[t, v] = sum_h hidden[t,h]*weight[v,h])
//    Output is fp32 scratch.
// ---------------------------------------------------------------------------
template <typename scalar_t>
KS_GLOBAL void logits_gemm_kernel(float* __restrict__ logits,    // [cs, V]
                                  const scalar_t* __restrict__ hidden,  // [cs, H]
                                  const scalar_t* __restrict__ weight,  // [V, H]
                                  int cs, int64_t vocab, int64_t hidden_dim) {
  __shared__ float sh[kTile][kTile];  // hidden tile  [row, k]
  __shared__ float sw[kTile][kTile];  // weight tile  [col(v), k]

  const int row = blockIdx.y * kTile + threadIdx.y;  // token index in chunk
  const int col = blockIdx.x * kTile + threadIdx.x;  // vocab index

  float acc = 0.f;
  const int n_k_tiles = static_cast<int>((hidden_dim + kTile - 1) / kTile);
  for (int kt = 0; kt < n_k_tiles; ++kt) {
    const int k_h = kt * kTile + threadIdx.x;  // k for hidden load
    const int k_w = kt * kTile + threadIdx.y;  // k for weight load

    sh[threadIdx.y][threadIdx.x] =
        (row < cs && k_h < hidden_dim)
            ? to_float(hidden[static_cast<int64_t>(row) * hidden_dim + k_h])
            : 0.f;
    // weight[col, k_w] with col=vocab, contiguous in H.
    sw[threadIdx.y][threadIdx.x] =
        (col < vocab && k_w < hidden_dim)
            ? to_float(weight[static_cast<int64_t>(col) * hidden_dim + k_w])
            : 0.f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < kTile; ++k) acc += sh[threadIdx.y][k] * sw[k][threadIdx.x];
    __syncthreads();
  }

  if (row < cs && col < vocab)
    logits[static_cast<int64_t>(row) * vocab + col] = acc;
}

// ---------------------------------------------------------------------------
// 2) Cross-entropy over a chunk of fp32 logits, grad written in-place.
//    One block per row of the chunk. Mirrors cross_entropy.cu but on fp32
//    scratch (logits) and indexed by global token row for losses/targets.
// ---------------------------------------------------------------------------
KS_GLOBAL void ce_chunk_kernel(float* __restrict__ losses,       // [num_tokens]
                               float* __restrict__ logits,       // [cs, V] in/out
                               const void* __restrict__ targets, // [num_tokens]
                               int targets_i64, int cs, int64_t chunk_start,
                               int64_t vocab, int64_t ignore_index,
                               float label_smoothing) {
  const int local_row = blockIdx.x;
  if (local_row >= cs) return;
  const int64_t global_row = chunk_start + local_row;
  float* x = logits + static_cast<int64_t>(local_row) * vocab;

  const int64_t y = load_target_id(targets, targets_i64, global_row);

  __shared__ float smem[kCeRowBlock / KS_WARP_SIZE];

  if (y == ignore_index) {
    for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) x[i] = 0.f;
    if (threadIdx.x == 0) losses[global_row] = 0.f;
    return;
  }

  // Pass 1: online (max, denom) + sum(logits).
  float thread_max = neg_inf<float>();
  float thread_denom = 0.f;
  float thread_sum = 0.f;
  for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float v = x[i];
    thread_sum += v;
    if (v > thread_max) {
      thread_denom = thread_denom * __expf(thread_max - v) + 1.f;
      thread_max = v;
    } else {
      thread_denom += __expf(v - thread_max);
    }
  }

  const float row_max = block_reduce_max(thread_max, smem);
  __syncthreads();
  float rebased_denom =
      (thread_max == neg_inf<float>())
          ? 0.f
          : thread_denom * __expf(thread_max - row_max);
  const float denom = block_reduce_sum(rebased_denom, smem);
  __syncthreads();
  const float sum_logits = block_reduce_sum(thread_sum, smem);

  const float lse = row_max + __logf(denom);

  __shared__ float s_xy;
  if (y >= 0 && y < vocab) {
    const int owner = static_cast<int>(y % blockDim.x);
    if (threadIdx.x == owner) s_xy = x[y];
  }
  __syncthreads();
  const float x_y = (y >= 0 && y < vocab) ? s_xy : 0.f;

  const float ls = label_smoothing;
  const float eps = (vocab > 0) ? ls / static_cast<float>(vocab) : 0.f;
  if (threadIdx.x == 0) {
    const float nll = lse - x_y;
    float loss = nll;
    if (ls > 0.f) {
      const float smooth_loss = ls * lse - eps * sum_logits;
      loss = (1.f - ls) * nll + smooth_loss;
    }
    losses[global_row] = loss;
  }

  // Pass 2: grad = softmax - (1-ls)*onehot - eps, in-place.
  const float inv_denom = 1.f / denom;
  for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float p = __expf(x[i] - row_max) * inv_denom;
    float g = p - eps;
    if (i == y) g -= (1.f - ls);
    x[i] = g;
  }
}

// ---------------------------------------------------------------------------
// 3) grad_hidden_chunk[cs, H] = grad_logits_chunk[cs, V] @ weight[V, H]
//    grad_hidden[t, h] = sum_v glog[t, v] * weight[v, h]
//    (glog is the fp32 scratch produced by ce_chunk_kernel; output dtype scalar)
// ---------------------------------------------------------------------------
template <typename scalar_t>
KS_GLOBAL void grad_hidden_gemm_kernel(scalar_t* __restrict__ grad_hidden,  // [cs,H]
                                       const float* __restrict__ glog,      // [cs,V]
                                       const scalar_t* __restrict__ weight, // [V,H]
                                       int cs, int64_t vocab,
                                       int64_t hidden_dim) {
  __shared__ float sg[kTile][kTile];  // glog tile   [row, k=v]
  __shared__ float sw[kTile][kTile];  // weight tile [k=v, col=h]

  const int row = blockIdx.y * kTile + threadIdx.y;  // token in chunk
  const int col = blockIdx.x * kTile + threadIdx.x;  // hidden index

  float acc = 0.f;
  const int n_k_tiles = static_cast<int>((vocab + kTile - 1) / kTile);
  for (int kt = 0; kt < n_k_tiles; ++kt) {
    const int k_g = kt * kTile + threadIdx.x;  // v for glog load
    const int k_w = kt * kTile + threadIdx.y;  // v for weight load

    sg[threadIdx.y][threadIdx.x] =
        (row < cs && k_g < vocab)
            ? glog[static_cast<int64_t>(row) * vocab + k_g]
            : 0.f;
    // weight[v=k_w, h=col]
    sw[threadIdx.y][threadIdx.x] =
        (k_w < vocab && col < hidden_dim)
            ? to_float(weight[static_cast<int64_t>(k_w) * hidden_dim + col])
            : 0.f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < kTile; ++k) acc += sg[threadIdx.y][k] * sw[k][threadIdx.x];
    __syncthreads();
  }

  if (row < cs && col < hidden_dim)
    grad_hidden[static_cast<int64_t>(row) * hidden_dim + col] =
        from_float<scalar_t>(acc);
}

// ---------------------------------------------------------------------------
// 4) grad_weight_fp32[V, H] += grad_logits_chunk[cs, V]^T @ hidden_chunk[cs, H]
//    grad_weight[v, h] += sum_t glog[t, v] * hidden[t, h]
//    Accumulated across chunks (fp32 output, += semantics).
// ---------------------------------------------------------------------------
template <typename scalar_t>
KS_GLOBAL void grad_weight_gemm_kernel(float* __restrict__ grad_weight,    // [V,H]
                                       const float* __restrict__ glog,     // [cs,V]
                                       const scalar_t* __restrict__ hidden, // [cs,H]
                                       int cs, int64_t vocab,
                                       int64_t hidden_dim) {
  __shared__ float sg[kTile][kTile];  // glog^T tile: [k=t, row=v]
  __shared__ float sh[kTile][kTile];  // hidden tile: [k=t, col=h]

  const int row = blockIdx.y * kTile + threadIdx.y;  // vocab index
  const int col = blockIdx.x * kTile + threadIdx.x;  // hidden index

  float acc = 0.f;
  const int n_k_tiles = (cs + kTile - 1) / kTile;
  for (int kt = 0; kt < n_k_tiles; ++kt) {
    const int k_g = kt * kTile + threadIdx.x;  // t for glog load
    const int k_h = kt * kTile + threadIdx.y;  // t for hidden load

    // glog[t=k_g, v=row]
    sg[threadIdx.y][threadIdx.x] =
        (k_g < cs && row < vocab)
            ? glog[static_cast<int64_t>(k_g) * vocab + row]
            : 0.f;
    // hidden[t=k_h, h=col]
    sh[threadIdx.y][threadIdx.x] =
        (k_h < cs && col < hidden_dim)
            ? to_float(hidden[static_cast<int64_t>(k_h) * hidden_dim + col])
            : 0.f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < kTile; ++k)
      acc += sg[threadIdx.y][k] * sh[k][threadIdx.x];
    __syncthreads();
  }

  if (row < vocab && col < hidden_dim)
    grad_weight[static_cast<int64_t>(row) * hidden_dim + col] += acc;
}

}  // namespace loss
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_fused_linear_cross_entropy(
    float* losses, void* grad_hidden, void* grad_weight_fp32,
    const void* hidden, const void* weight, const void* targets,
    int targets_i64, int64_t num_tokens, int64_t hidden_dim, int64_t vocab,
    int64_t ignore_index, float label_smoothing, int chunk_size,
    ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(losses);
  KS_CHECK_PTR(grad_hidden);
  KS_CHECK_PTR(grad_weight_fp32);
  KS_CHECK_PTR(hidden);
  KS_CHECK_PTR(weight);
  KS_CHECK_PTR(targets);
  if (num_tokens <= 0 || hidden_dim <= 0 || vocab <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_fused_linear_cross_entropy: bad shape");
  if (label_smoothing < 0.f || label_smoothing >= 1.f)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_fused_linear_cross_entropy: label_smoothing in [0,1)");
  // Validate dtype up front so the scratch allocation below cannot leak through
  // the dispatch macro's early-returning default case.
  if (dtype != KS_DTYPE_F32 && dtype != KS_DTYPE_F16 && dtype != KS_DTYPE_BF16)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                    "ks_fused_linear_cross_entropy: unsupported dtype");

  // Pick a chunk size if the caller passed <= 0. Liger sizes the chunk so the
  // [chunk, vocab] scratch matches the hidden activation footprint:
  //   chunk = ceil(num_tokens / ceil(vocab / hidden_dim)), clamped to tokens.
  int cs_max = chunk_size;
  if (cs_max <= 0) {
    const int64_t inc = (vocab + hidden_dim - 1) / hidden_dim;  // >= 1
    int64_t c = (num_tokens + inc - 1) / inc;
    if (c < 1) c = 1;
    if (c > num_tokens) c = num_tokens;
    cs_max = static_cast<int>(c);
  }
  if (static_cast<int64_t>(cs_max) > num_tokens)
    cs_max = static_cast<int>(num_tokens);

  auto s = to_stream(stream);

  // Scratch for one chunk of logits / grad-logits: [cs_max, vocab] fp32.
  float* d_logits = nullptr;
  const size_t scratch_bytes =
      static_cast<size_t>(cs_max) * static_cast<size_t>(vocab) * sizeof(float);
  {
    ks_status_t st = ks::check(ks::gpuMalloc(reinterpret_cast<void**>(&d_logits),
                                             scratch_bytes),
                               "ks_fused_linear_cross_entropy: gpuMalloc");
    if (st != KS_SUCCESS) return st;
  }

  // Zero grad_weight_fp32 ([vocab, hidden_dim]) — accumulated across chunks.
  {
    const size_t gw_bytes = static_cast<size_t>(vocab) *
                            static_cast<size_t>(hidden_dim) * sizeof(float);
    ks_status_t st = ks::check(
        ks::gpuMemsetAsync(grad_weight_fp32, 0, gw_bytes, s),
        "ks_fused_linear_cross_entropy: memset grad_weight");
    if (st != KS_SUCCESS) {
      ks::gpuFree(d_logits);
      return st;
    }
  }

  const dim3 tile(loss::kTile, loss::kTile);

  ks_status_t launch_status = KS_SUCCESS;
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_fused_linear_cross_entropy", {
    for (int64_t start = 0; start < num_tokens; start += cs_max) {
      const int64_t remaining = num_tokens - start;
      const int cs = static_cast<int>(
          (static_cast<int64_t>(cs_max) < remaining) ? cs_max : remaining);

      const scalar_t* hid_chunk =
          static_cast<const scalar_t*>(hidden) + start * hidden_dim;
      scalar_t* ghid_chunk =
          static_cast<scalar_t*>(grad_hidden) + start * hidden_dim;

      // 1) logits_chunk = hidden_chunk @ weight^T  -> d_logits[cs, vocab]
      const dim3 g1(static_cast<unsigned>((vocab + loss::kTile - 1) / loss::kTile),
                    static_cast<unsigned>((cs + loss::kTile - 1) / loss::kTile));
      loss::logits_gemm_kernel<scalar_t><<<g1, tile, 0, s>>>(
          d_logits, hid_chunk, static_cast<const scalar_t*>(weight), cs, vocab,
          hidden_dim);

      // 2) CE over the chunk; grad of logits written in-place into d_logits.
      const dim3 g2(static_cast<unsigned>(cs));
      const dim3 b2(loss::kCeRowBlock);
      loss::ce_chunk_kernel<<<g2, b2, 0, s>>>(
          losses, d_logits, targets, targets_i64, cs, start, vocab,
          ignore_index, label_smoothing);

      // 3) grad_hidden_chunk = grad_logits_chunk @ weight
      const dim3 g3(
          static_cast<unsigned>((hidden_dim + loss::kTile - 1) / loss::kTile),
          static_cast<unsigned>((cs + loss::kTile - 1) / loss::kTile));
      loss::grad_hidden_gemm_kernel<scalar_t><<<g3, tile, 0, s>>>(
          ghid_chunk, d_logits, static_cast<const scalar_t*>(weight), cs, vocab,
          hidden_dim);

      // 4) grad_weight_fp32 += grad_logits_chunk^T @ hidden_chunk
      const dim3 g4(
          static_cast<unsigned>((hidden_dim + loss::kTile - 1) / loss::kTile),
          static_cast<unsigned>((vocab + loss::kTile - 1) / loss::kTile));
      loss::grad_weight_gemm_kernel<scalar_t><<<g4, tile, 0, s>>>(
          static_cast<float*>(grad_weight_fp32), d_logits, hid_chunk, cs, vocab,
          hidden_dim);
    }
  });

  // Surface any launch error first.
  launch_status = ks::check(ks::gpuGetLastError(), __func__);

  // The scratch buffer is consumed by async kernels on `s`; synchronize before
  // freeing it to avoid a use-after-free. (FLCE owns a device allocation per
  // call, so it is inherently a synchronizing op — a caller wanting full async
  // behavior would pre-allocate a persistent workspace; documented as a tuning
  // hook.) Synchronize even on launch error so outstanding work drains.
  if (launch_status == KS_SUCCESS)
    launch_status = ks::check(ks::gpuStreamSynchronize(s), __func__);
  else
    ks::gpuStreamSynchronize(s);

  ks::gpuFree(d_logits);
  if (launch_status != KS_SUCCESS) return launch_status;
  return KS_SUCCESS;
}

}  // extern "C"
