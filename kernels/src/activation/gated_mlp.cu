// kernel-set — gated-MLP fusions: SwiGLU, GeGLU, packed SwiGLU, SwiGLU backward.
//
// These implement the Llama-family gated feed-forward activation
//   out = act(gate) * up
// fused into a single elementwise pass (no intermediate act(gate) buffer). All
// arithmetic is fp32; storage is f32/f16/bf16 via KS_DISPATCH_FLOATING_TYPES.
//
// Memory layout:
//   * ks_swiglu / ks_geglu:    gate[rows, inter], up[rows, inter] contiguous.
//                              The op is purely elementwise over rows*inter, so
//                              we flatten to a single 1-D launch and vectorize
//                              128-bit IO when all three buffers are aligned.
//   * ks_swiglu_packed:        input[rows, 2*inter] with gate = input[:, :inter]
//                              and up = input[:, inter:]. gate/up are NOT
//                              contiguous as a flat array (each row interleaves
//                              the two halves), so we index per (row, col) and
//                              vectorize within a row's half when aligned.
//   * ks_swiglu_backward:      d_gate = d_out * up * silu'(gate)
//                              d_up   = d_out * silu(gate)
//                              Flattened elementwise like the forward.
//
// Further tuning (documented, not done here to stay simple + correct): the
// gate/up GEMMs that produce these tensors can be fused with this epilogue via a
// CUTLASS elementwise epilogue, and the packed kernel can use cp.async to stage
// rows; both are bandwidth-bound so the win is modest over the wide IO here.
#include "kernel_set/activation.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/vec.cuh"
#include "activation_ops.cuh"

namespace ks {
namespace activation {

// ---- Flat elementwise gated forward: out = act(gate) * up -----------------
//
// `Act` is one of the activation functors (SiluOp / GeluErfOp / GeluTanhOp).
template <typename scalar_t, typename Act, int VEC>
KS_GLOBAL void gated_vec_kernel(scalar_t* __restrict__ out,
                                const scalar_t* __restrict__ gate,
                                const scalar_t* __restrict__ up,
                                int64_t n_vec, Act act) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t v = idx; v < n_vec; v += stride) {
    Vec<scalar_t, VEC> g = load_vec<scalar_t, VEC>(gate + v * VEC);
    Vec<scalar_t, VEC> u = load_vec<scalar_t, VEC>(up + v * VEC);
    Vec<scalar_t, VEC> o;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      o[j] = from_float<scalar_t>(act(to_float(g[j])) * to_float(u[j]));
    }
    store_vec<scalar_t, VEC>(out + v * VEC, o);
  }
}

template <typename scalar_t, typename Act>
KS_GLOBAL void gated_scalar_kernel(scalar_t* __restrict__ out,
                                   const scalar_t* __restrict__ gate,
                                   const scalar_t* __restrict__ up, int64_t n,
                                   Act act) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    out[i] = from_float<scalar_t>(act(to_float(gate[i])) * to_float(up[i]));
  }
}

template <typename scalar_t, typename Act>
void launch_gated(scalar_t* out, const scalar_t* gate, const scalar_t* up,
                  int64_t n, Act act, gpuStream_t stream) {
  constexpr int VEC = VecWidth<scalar_t>::value;
  if (VEC > 1 && is_aligned<scalar_t, VEC>(gate, n) &&
      is_aligned<scalar_t, VEC>(up, n) && is_aligned<scalar_t, VEC>(out, n)) {
    const int64_t n_vec = n / VEC;
    const dim3 grid(grid_for(n_vec));
    const dim3 block(kElemBlock);
    gated_vec_kernel<scalar_t, Act, VEC>
        <<<grid, block, 0, stream>>>(out, gate, up, n_vec, act);
  } else {
    const dim3 grid(grid_for(n));
    const dim3 block(kElemBlock);
    gated_scalar_kernel<scalar_t, Act>
        <<<grid, block, 0, stream>>>(out, gate, up, n, act);
  }
}

// ---- Packed SwiGLU: gate/up are the two halves of each input row ----------
//
// input[row] = [ gate(inter) | up(inter) ]. We index by (row, col) so the two
// halves are read from disjoint offsets; one thread handles VEC consecutive
// columns within a row. When `inter`, the row base, and the buffers are aligned
// to VEC, each half is read with a single wide load.
template <typename scalar_t, int VEC>
KS_GLOBAL void swiglu_packed_vec_kernel(scalar_t* __restrict__ out,
                                        const scalar_t* __restrict__ input,
                                        int64_t rows, int64_t inter) {
  const int64_t inter_vec = inter / VEC;
  const int64_t total_vec = rows * inter_vec;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t t = idx; t < total_vec; t += stride) {
    const int64_t row = t / inter_vec;
    const int64_t cv = t % inter_vec;             // vector column within a half
    const int64_t col = cv * VEC;
    const scalar_t* in_row = input + row * (2 * inter);
    Vec<scalar_t, VEC> g = load_vec<scalar_t, VEC>(in_row + col);
    Vec<scalar_t, VEC> u = load_vec<scalar_t, VEC>(in_row + inter + col);
    Vec<scalar_t, VEC> o;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      o[j] = from_float<scalar_t>(silu_f(to_float(g[j])) * to_float(u[j]));
    }
    store_vec<scalar_t, VEC>(out + row * inter + col, o);
  }
}

template <typename scalar_t>
KS_GLOBAL void swiglu_packed_scalar_kernel(scalar_t* __restrict__ out,
                                           const scalar_t* __restrict__ input,
                                           int64_t rows, int64_t inter) {
  const int64_t total = rows * inter;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t t = idx; t < total; t += stride) {
    const int64_t row = t / inter;
    const int64_t col = t % inter;
    const scalar_t* in_row = input + row * (2 * inter);
    const float g = to_float(in_row[col]);
    const float u = to_float(in_row[inter + col]);
    out[t] = from_float<scalar_t>(silu_f(g) * u);
  }
}

// ---- SwiGLU backward ------------------------------------------------------
// Forward:  y = silu(g) * u
//   d_gate = d_out * u * silu'(g)
//   d_up   = d_out * silu(g)
template <typename scalar_t, int VEC>
KS_GLOBAL void swiglu_backward_vec_kernel(scalar_t* __restrict__ d_gate,
                                          scalar_t* __restrict__ d_up,
                                          const scalar_t* __restrict__ d_out,
                                          const scalar_t* __restrict__ gate,
                                          const scalar_t* __restrict__ up,
                                          int64_t n_vec) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t v = idx; v < n_vec; v += stride) {
    Vec<scalar_t, VEC> go = load_vec<scalar_t, VEC>(d_out + v * VEC);
    Vec<scalar_t, VEC> g = load_vec<scalar_t, VEC>(gate + v * VEC);
    Vec<scalar_t, VEC> u = load_vec<scalar_t, VEC>(up + v * VEC);
    Vec<scalar_t, VEC> dg, du;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float gv = to_float(g[j]);
      const float uv = to_float(u[j]);
      const float gov = to_float(go[j]);
      dg[j] = from_float<scalar_t>(gov * uv * silu_grad_f(gv));
      du[j] = from_float<scalar_t>(gov * silu_f(gv));
    }
    store_vec<scalar_t, VEC>(d_gate + v * VEC, dg);
    store_vec<scalar_t, VEC>(d_up + v * VEC, du);
  }
}

template <typename scalar_t>
KS_GLOBAL void swiglu_backward_scalar_kernel(scalar_t* __restrict__ d_gate,
                                             scalar_t* __restrict__ d_up,
                                             const scalar_t* __restrict__ d_out,
                                             const scalar_t* __restrict__ gate,
                                             const scalar_t* __restrict__ up,
                                             int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    const float gv = to_float(gate[i]);
    const float uv = to_float(up[i]);
    const float gov = to_float(d_out[i]);
    d_gate[i] = from_float<scalar_t>(gov * uv * silu_grad_f(gv));
    d_up[i] = from_float<scalar_t>(gov * silu_f(gv));
  }
}

template <typename scalar_t>
void launch_swiglu_backward(scalar_t* d_gate, scalar_t* d_up,
                            const scalar_t* d_out, const scalar_t* gate,
                            const scalar_t* up, int64_t n, gpuStream_t stream) {
  constexpr int VEC = VecWidth<scalar_t>::value;
  if (VEC > 1 && is_aligned<scalar_t, VEC>(d_gate, n) &&
      is_aligned<scalar_t, VEC>(d_up, n) &&
      is_aligned<scalar_t, VEC>(d_out, n) &&
      is_aligned<scalar_t, VEC>(gate, n) && is_aligned<scalar_t, VEC>(up, n)) {
    const int64_t n_vec = n / VEC;
    const dim3 grid(grid_for(n_vec));
    const dim3 block(kElemBlock);
    swiglu_backward_vec_kernel<scalar_t, VEC>
        <<<grid, block, 0, stream>>>(d_gate, d_up, d_out, gate, up, n_vec);
  } else {
    const dim3 grid(grid_for(n));
    const dim3 block(kElemBlock);
    swiglu_backward_scalar_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(d_gate, d_up, d_out, gate, up, n);
  }
}

}  // namespace activation
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_swiglu(void* out, const void* gate, const void* up, int64_t rows,
                      int64_t inter, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(gate);
  KS_CHECK_PTR(up);
  if (rows < 0 || inter < 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_swiglu: bad shape");
  const int64_t n = rows * inter;
  if (n == 0) return KS_SUCCESS;

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_swiglu", {
    activation::launch_gated<scalar_t>(
        static_cast<scalar_t*>(out), static_cast<const scalar_t*>(gate),
        static_cast<const scalar_t*>(up), n, activation::SiluOp{}, s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_geglu(void* out, const void* gate, const void* up, int64_t rows,
                     int64_t inter, int tanh_approx, ks_dtype_t dtype,
                     ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(gate);
  KS_CHECK_PTR(up);
  if (rows < 0 || inter < 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_geglu: bad shape");
  const int64_t n = rows * inter;
  if (n == 0) return KS_SUCCESS;

  auto s = to_stream(stream);
  if (tanh_approx != 0) {
    KS_DISPATCH_FLOATING_TYPES(dtype, "ks_geglu", {
      activation::launch_gated<scalar_t>(
          static_cast<scalar_t*>(out), static_cast<const scalar_t*>(gate),
          static_cast<const scalar_t*>(up), n, activation::GeluTanhOp{}, s);
    });
  } else {
    KS_DISPATCH_FLOATING_TYPES(dtype, "ks_geglu", {
      activation::launch_gated<scalar_t>(
          static_cast<scalar_t*>(out), static_cast<const scalar_t*>(gate),
          static_cast<const scalar_t*>(up), n, activation::GeluErfOp{}, s);
    });
  }
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_swiglu_packed(void* out, const void* input, int64_t rows,
                             int64_t inter, ks_dtype_t dtype,
                             ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  if (rows < 0 || inter < 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_swiglu_packed: bad shape");
  const int64_t n = rows * inter;
  if (n == 0) return KS_SUCCESS;

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_swiglu_packed", {
    constexpr int VEC = VecWidth<scalar_t>::value;
    auto* outp = static_cast<scalar_t*>(out);
    auto* inp = static_cast<const scalar_t*>(input);
    // Vectorize only when each half (length `inter`), the output rows, and the
    // packed input rows (length 2*inter) all start on a VEC boundary. Since the
    // base ptr is checked plus `inter % VEC == 0`, every row's gate/up half is
    // aligned too.
    const bool can_vec =
        VEC > 1 && (inter % VEC == 0) &&
        is_aligned<scalar_t, VEC>(inp, rows * 2 * inter) &&
        is_aligned<scalar_t, VEC>(outp, n);
    if (can_vec) {
      const int64_t total_vec = rows * (inter / VEC);
      const dim3 grid(activation::grid_for(total_vec));
      const dim3 block(activation::kElemBlock);
      activation::swiglu_packed_vec_kernel<scalar_t, VEC>
          <<<grid, block, 0, s>>>(outp, inp, rows, inter);
    } else {
      const dim3 grid(activation::grid_for(n));
      const dim3 block(activation::kElemBlock);
      activation::swiglu_packed_scalar_kernel<scalar_t>
          <<<grid, block, 0, s>>>(outp, inp, rows, inter);
    }
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_swiglu_backward(void* grad_gate, void* grad_up,
                               const void* grad_out, const void* gate,
                               const void* up, int64_t rows, int64_t inter,
                               ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(grad_gate);
  KS_CHECK_PTR(grad_up);
  KS_CHECK_PTR(grad_out);
  KS_CHECK_PTR(gate);
  KS_CHECK_PTR(up);
  if (rows < 0 || inter < 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_swiglu_backward: bad shape");
  const int64_t n = rows * inter;
  if (n == 0) return KS_SUCCESS;

  auto s = to_stream(stream);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_swiglu_backward", {
    activation::launch_swiglu_backward<scalar_t>(
        static_cast<scalar_t*>(grad_gate), static_cast<scalar_t*>(grad_up),
        static_cast<const scalar_t*>(grad_out),
        static_cast<const scalar_t*>(gate), static_cast<const scalar_t*>(up), n,
        s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
