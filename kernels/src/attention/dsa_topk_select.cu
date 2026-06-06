// kernel-set — DeepSeek sparse-attention row-wise top-k KV selection.
//
// indices[r, 0..topk) = column indices of the topk largest scores[r, :],
// largest first, ties -> lower column, padded with -1 when n_cols < topk.
// Correctness-first reference: one block per row, k passes of block-wide arg-max
// over a mutable fp32 copy of the row (selected entries set to -inf). The fast
// path is a radix top-k provider (flashinfer.top_k).
#include "kernel_set/attention.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"

namespace ks {
namespace attention {

constexpr int kTopkBlock = 256;

// Block-wide arg-max returning (value, index); ties resolve to the lower index.
struct ArgMax {
  float val;
  int idx;
};

template <typename scalar_t>
KS_GLOBAL void dsa_topk_kernel(int32_t* __restrict__ indices,
                               const scalar_t* __restrict__ scores,
                               float* __restrict__ scratch, int64_t n_cols,
                               int topk) {
  const int64_t row = blockIdx.x;
  const scalar_t* s = scores + row * n_cols;
  float* buf = scratch + row * n_cols;        // mutable working copy
  int32_t* out = indices + row * static_cast<int64_t>(topk);

  for (int64_t i = threadIdx.x; i < n_cols; i += blockDim.x)
    buf[i] = to_float(s[i]);
  __syncthreads();

  __shared__ float sval[kTopkBlock / KS_WARP_SIZE];
  __shared__ int sidx[kTopkBlock / KS_WARP_SIZE];

  for (int t = 0; t < topk; ++t) {
    if (t >= n_cols) {                         // pad
      if (threadIdx.x == 0) out[t] = -1;
      continue;
    }
    // block-wide arg-max over buf (ties -> lower index)
    float best = -INFINITY;
    int besti = -1;
    for (int64_t i = threadIdx.x; i < n_cols; i += blockDim.x) {
      const float v = buf[i];
      if (v > best || (v == best && static_cast<int>(i) < besti)) {
        best = v;
        besti = static_cast<int>(i);
      }
    }
    const int lane = threadIdx.x % KS_WARP_SIZE;
    const int warp = threadIdx.x / KS_WARP_SIZE;
    for (int off = KS_WARP_SIZE / 2; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, best, off);
      const int oi = __shfl_down_sync(0xffffffffu, besti, off);
      if (ov > best || (ov == best && oi >= 0 && (besti < 0 || oi < besti))) {
        best = ov;
        besti = oi;
      }
    }
    if (lane == 0) {
      sval[warp] = best;
      sidx[warp] = besti;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const int nwarp = (blockDim.x + KS_WARP_SIZE - 1) / KS_WARP_SIZE;
      float bv = -INFINITY;
      int bi = -1;
      for (int w = 0; w < nwarp; ++w) {
        if (sval[w] > bv || (sval[w] == bv && sidx[w] >= 0 &&
                             (bi < 0 || sidx[w] < bi))) {
          bv = sval[w];
          bi = sidx[w];
        }
      }
      out[t] = bi;
      if (bi >= 0) buf[bi] = -INFINITY;        // remove for next pass
      sidx[0] = bi;                            // broadcast picked index
    }
    __syncthreads();
  }
}

}  // namespace attention
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_dsa_topk_select(int32_t* indices, const void* scores,
                               int64_t n_rows, int64_t n_cols, int topk,
                               ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(indices);
  KS_CHECK_PTR(scores);
  if (n_rows <= 0 || n_cols <= 0 || topk <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_dsa_topk_select: bad shape");
  if (n_rows > 2147483647LL)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_dsa_topk_select: n_rows > grid limit");

  const size_t nscratch = static_cast<size_t>(n_rows) * n_cols;
  float* scratch = nullptr;
  if (ks::gpuMalloc(reinterpret_cast<void**>(&scratch),
                    nscratch * sizeof(float)) != ks::gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_OUT_OF_MEMORY,
                    "ks_dsa_topk_select: scratch alloc failed");

  const dim3 grid(static_cast<unsigned>(n_rows));
  const dim3 block(attention::kTopkBlock);
  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_dsa_topk_select", {
    attention::dsa_topk_kernel<scalar_t><<<grid, block, 0, s>>>(
        indices, static_cast<const scalar_t*>(scores), scratch, n_cols, topk);
  });
  ks::gpuStreamSynchronize(s);
  ks::gpuFree(scratch);
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
