// kernel-set — shared helpers for the elementwise category.
//
// A single grid-stride loop drives every pointwise op. When the element pointers
// and count are 128-bit aligned we process VecWidth<T> elements per iteration
// with one wide load/store; the unaligned remainder (and any partial tail) is
// handled by a scalar grid-stride loop. All math is done in fp32 via
// ks::to_float / ks::from_float, matching the house style in norm/rms_norm.cu.
#pragma once

#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/vec.cuh"

namespace ks {
namespace elementwise {

constexpr int kBlock = 256;

// Launch geometry for a 1-D grid-stride kernel over `n` vector-groups (or
// scalars). Caps grid at a generous fixed size so a single launch saturates the
// device while staying within the 1-D grid limit for huge tensors.
inline dim3 grid_for(int64_t work_items) {
  // Aim for a few waves; clamp to [1, 65535] which is safe for the 1-D grid
  // dimension on every supported arch (the grid-stride loop covers the rest).
  int64_t blocks = (work_items + kBlock - 1) / kBlock;
  if (blocks < 1) blocks = 1;
  if (blocks > 65535) blocks = 65535;
  return dim3(static_cast<unsigned>(blocks));
}

// ---------------------------------------------------------------------------
// Binary elementwise: out[i] = Op(a[i], b[i]) with fp32 math.
// `Op` is a functor: float operator()(float, float) const.
//
// `out` is deliberately NOT marked __restrict__: callers commonly run these
// ops in place (e.g. ks_axpby as a[i] = alpha*a[i] + beta*b[i], i.e. out==a),
// so promising the compiler that `out` is disjoint from `a`/`b` would be a
// false aliasing guarantee (UB). Each thread reads its own index into registers
// before writing the same index, so a same-buffer read-modify-write stays
// well-defined. `a`/`b` keep __restrict__ for read-reuse on the common path.
// ---------------------------------------------------------------------------
template <typename scalar_t, int VEC, typename Op>
KS_GLOBAL void binary_vec_kernel(scalar_t* out,
                                 const scalar_t* __restrict__ a,
                                 const scalar_t* __restrict__ b,
                                 int64_t n_vec, Op op) {
  using V = Vec<scalar_t, VEC>;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_vec;
       idx += stride) {
    V va = load_vec<scalar_t, VEC>(a + idx * VEC);
    V vb = load_vec<scalar_t, VEC>(b + idx * VEC);
    V vo;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      vo[j] = from_float<scalar_t>(op(to_float(va[j]), to_float(vb[j])));
    }
    store_vec<scalar_t, VEC>(out + idx * VEC, vo);
  }
}

template <typename scalar_t, typename Op>
KS_GLOBAL void binary_scalar_kernel(scalar_t* out,
                                    const scalar_t* __restrict__ a,
                                    const scalar_t* __restrict__ b,
                                    int64_t begin, int64_t n, Op op) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = begin + blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    out[i] = from_float<scalar_t>(op(to_float(a[i]), to_float(b[i])));
  }
}

// In-place binary: dst <- Op(dst, b). `dst` is read+written and may legally
// alias nothing else, so it is NOT marked __restrict__ (the non-in-place
// kernels above ARE, for the disjoint-buffer fast path). `b` stays restrict.
// Each thread touches only its own index, so the read-modify-write is race-free.
template <typename scalar_t, int VEC, typename Op>
KS_GLOBAL void inplace_vec_kernel(scalar_t* dst, const scalar_t* __restrict__ b,
                                  int64_t n_vec, Op op) {
  using V = Vec<scalar_t, VEC>;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_vec;
       idx += stride) {
    V vd = load_vec<scalar_t, VEC>(dst + idx * VEC);
    V vb = load_vec<scalar_t, VEC>(b + idx * VEC);
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      vd[j] = from_float<scalar_t>(op(to_float(vd[j]), to_float(vb[j])));
    }
    store_vec<scalar_t, VEC>(dst + idx * VEC, vd);
  }
}

template <typename scalar_t, typename Op>
KS_GLOBAL void inplace_scalar_kernel(scalar_t* dst,
                                     const scalar_t* __restrict__ b,
                                     int64_t begin, int64_t n, Op op) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = begin + blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    dst[i] = from_float<scalar_t>(op(to_float(dst[i]), to_float(b[i])));
  }
}

// ---------------------------------------------------------------------------
// Unary elementwise: out[i] = Op(x[i]) with fp32 math. `out` MAY alias `x`
// (in-place scale is common), so `out` is deliberately NOT __restrict__: each
// thread reads x[idx] into registers before writing the same idx, so a same-
// buffer read-modify-write is well-defined. `x` stays restrict for read reuse.
// `Op` is a functor: float operator()(float) const.
// ---------------------------------------------------------------------------
template <typename scalar_t, int VEC, typename Op>
KS_GLOBAL void unary_vec_kernel(scalar_t* out,
                                const scalar_t* __restrict__ x, int64_t n_vec,
                                Op op) {
  using V = Vec<scalar_t, VEC>;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_vec;
       idx += stride) {
    V vx = load_vec<scalar_t, VEC>(x + idx * VEC);
    V vo;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      vo[j] = from_float<scalar_t>(op(to_float(vx[j])));
    }
    store_vec<scalar_t, VEC>(out + idx * VEC, vo);
  }
}

template <typename scalar_t, typename Op>
KS_GLOBAL void unary_scalar_kernel(scalar_t* out,
                                   const scalar_t* __restrict__ x, int64_t begin,
                                   int64_t n, Op op) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = begin + blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    out[i] = from_float<scalar_t>(op(to_float(x[i])));
  }
}

// ---------------------------------------------------------------------------
// Host-side dispatchers: split `n` into a 128-bit-aligned vectorized body plus a
// scalar tail. We require ALL participating pointers to share the same vector
// alignment; otherwise we fall back to a fully scalar launch (still correct,
// just less bandwidth-optimal).
// ---------------------------------------------------------------------------
template <typename scalar_t, typename Op>
inline void launch_binary(scalar_t* out, const scalar_t* a, const scalar_t* b,
                          int64_t n, Op op, gpuStream_t stream) {
  constexpr int VEC = VecWidth<scalar_t>::value;
  const bool vectorize =
      VEC > 1 && (reinterpret_cast<uintptr_t>(out) % (sizeof(scalar_t) * VEC) == 0) &&
      (reinterpret_cast<uintptr_t>(a) % (sizeof(scalar_t) * VEC) == 0) &&
      (reinterpret_cast<uintptr_t>(b) % (sizeof(scalar_t) * VEC) == 0);

  if (vectorize) {
    const int64_t n_vec = n / VEC;
    const int64_t tail_begin = n_vec * VEC;
    if (n_vec > 0) {
      binary_vec_kernel<scalar_t, VEC, Op>
          <<<grid_for(n_vec), kBlock, 0, stream>>>(out, a, b, n_vec, op);
    }
    if (tail_begin < n) {
      binary_scalar_kernel<scalar_t, Op>
          <<<grid_for(n - tail_begin), kBlock, 0, stream>>>(out, a, b,
                                                            tail_begin, n, op);
    }
  } else {
    binary_scalar_kernel<scalar_t, Op>
        <<<grid_for(n), kBlock, 0, stream>>>(out, a, b, 0, n, op);
  }
}

// In-place: dst <- Op(dst, b). Uses the non-restrict in-place kernels so the
// dst==first-operand aliasing is well-defined (used by ks_add_residual).
template <typename scalar_t, typename Op>
inline void launch_inplace(scalar_t* dst, const scalar_t* b, int64_t n, Op op,
                           gpuStream_t stream) {
  constexpr int VEC = VecWidth<scalar_t>::value;
  const bool vectorize =
      VEC > 1 && (reinterpret_cast<uintptr_t>(dst) % (sizeof(scalar_t) * VEC) == 0) &&
      (reinterpret_cast<uintptr_t>(b) % (sizeof(scalar_t) * VEC) == 0);

  if (vectorize) {
    const int64_t n_vec = n / VEC;
    const int64_t tail_begin = n_vec * VEC;
    if (n_vec > 0) {
      inplace_vec_kernel<scalar_t, VEC, Op>
          <<<grid_for(n_vec), kBlock, 0, stream>>>(dst, b, n_vec, op);
    }
    if (tail_begin < n) {
      inplace_scalar_kernel<scalar_t, Op>
          <<<grid_for(n - tail_begin), kBlock, 0, stream>>>(dst, b, tail_begin,
                                                            n, op);
    }
  } else {
    inplace_scalar_kernel<scalar_t, Op>
        <<<grid_for(n), kBlock, 0, stream>>>(dst, b, 0, n, op);
  }
}

template <typename scalar_t, typename Op>
inline void launch_unary(scalar_t* out, const scalar_t* x, int64_t n, Op op,
                         gpuStream_t stream) {
  constexpr int VEC = VecWidth<scalar_t>::value;
  const bool vectorize =
      VEC > 1 && (reinterpret_cast<uintptr_t>(out) % (sizeof(scalar_t) * VEC) == 0) &&
      (reinterpret_cast<uintptr_t>(x) % (sizeof(scalar_t) * VEC) == 0);

  if (vectorize) {
    const int64_t n_vec = n / VEC;
    const int64_t tail_begin = n_vec * VEC;
    if (n_vec > 0) {
      unary_vec_kernel<scalar_t, VEC, Op>
          <<<grid_for(n_vec), kBlock, 0, stream>>>(out, x, n_vec, op);
    }
    if (tail_begin < n) {
      unary_scalar_kernel<scalar_t, Op>
          <<<grid_for(n - tail_begin), kBlock, 0, stream>>>(out, x, tail_begin,
                                                            n, op);
    }
  } else {
    unary_scalar_kernel<scalar_t, Op>
        <<<grid_for(n), kBlock, 0, stream>>>(out, x, 0, n, op);
  }
}

// ---------------------------------------------------------------------------
// fp32 elementwise functors.
// ---------------------------------------------------------------------------
struct AddOp {
  KS_DI float operator()(float x, float y) const { return x + y; }
};
struct MulOp {
  KS_DI float operator()(float x, float y) const { return x * y; }
};
// out = a*alpha + b*beta  (captured scalars).
struct AxpbyOp {
  float alpha, beta;
  KS_DI float operator()(float x, float y) const { return x * alpha + y * beta; }
};
struct ScaleOp {
  float s;
  KS_DI float operator()(float x) const { return x * s; }
};

}  // namespace elementwise
}  // namespace ks
