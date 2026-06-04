// kernel-set — warp- and block-level reductions.
#pragma once

#include "common/platform.cuh"

namespace ks {

template <typename T>
struct SumOp {
  KS_DI T operator()(T a, T b) const { return a + b; }
  static KS_DI T identity() { return T(0); }
};
template <typename T>
struct MaxOp {
  KS_DI T operator()(T a, T b) const { return a > b ? a : b; }
  static KS_DI T identity() { return T(-3.4028235e38f); }
};

// Reduce across a warp; lane 0 holds the result (others hold partials).
template <typename T, typename Op>
KS_DI T warp_reduce(T val, Op op) {
#pragma unroll
  for (int offset = KS_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val = op(val, shfl_xor(val, offset));
  }
  return val;
}

template <typename T>
KS_DI T warp_reduce_sum(T v) { return warp_reduce(v, SumOp<T>{}); }
template <typename T>
KS_DI T warp_reduce_max(T v) { return warp_reduce(v, MaxOp<T>{}); }

// Block reduction via shared memory. `smem` must hold at least
// (blockDim.x / KS_WARP_SIZE) elements. Result is broadcast to all threads.
template <typename T, typename Op>
KS_DI T block_reduce(T val, Op op, T* smem) {
  const int lane = threadIdx.x % KS_WARP_SIZE;
  const int wid = threadIdx.x / KS_WARP_SIZE;
  const int num_warps = (blockDim.x + KS_WARP_SIZE - 1) / KS_WARP_SIZE;

  val = warp_reduce(val, op);
  if (lane == 0) smem[wid] = val;
  __syncthreads();

  val = (threadIdx.x < num_warps) ? smem[lane] : Op::identity();
  if (wid == 0) val = warp_reduce(val, op);
  if (threadIdx.x == 0) smem[0] = val;
  __syncthreads();
  return smem[0];
}

template <typename T>
KS_DI T block_reduce_sum(T v, T* smem) {
  return block_reduce(v, SumOp<T>{}, smem);
}
template <typename T>
KS_DI T block_reduce_max(T v, T* smem) {
  return block_reduce(v, MaxOp<T>{}, smem);
}

}  // namespace ks
