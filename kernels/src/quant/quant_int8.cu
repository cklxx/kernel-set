// kernel-set — dynamic symmetric INT8 quantize / dequantize.
//
// quantize:  out_int8 = round(in / scale),  scale = amax / 127
//   * KS_QUANT_PER_TOKEN : one scale per row, one CUDA block per row reduces
//     the row's amax (mirrors rms_norm's one-block-per-row layout).
//   * KS_QUANT_PER_TENSOR: a single scale; pass 1 reduces the global amax with
//     an atomicMax over the fp32 bit-pattern, pass 2 turns amax into a scale,
//     pass 3 quantizes. Sequential launches on the same stream order the passes.
// dequantize: out = in_int8 * scale  (symmetric, zero-point free).
#include "kernel_set/quant.h"
#include "quant/quant_common.cuh"

namespace ks {
namespace quant {

// atomicMax on a non-negative float via its int bit-pattern. For v >= 0 the
// IEEE-754 ordering matches the signed-int ordering, so this is a valid max.
KS_DI void atomic_max_nonneg(float* addr, float v) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

// ---- Per-token quantize: one block per row ----
template <typename scalar_t>
KS_GLOBAL void int8_quant_per_token_kernel(int8_t* __restrict__ out,
                                           float* __restrict__ scale,
                                           const scalar_t* __restrict__ input,
                                           int64_t cols) {
  const int64_t row = blockIdx.x;
  const scalar_t* x = input + row * cols;
  int8_t* y = out + row * cols;

  float local_amax = 0.f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    local_amax = fmaxf(local_amax, fabsf(to_float(x[i])));
  }

  __shared__ float smem[kBlock / KS_WARP_SIZE];
  const float amax = block_reduce_max(local_amax, smem);
  const float sc = fmaxf(amax / kInt8Max, kEpsScale);
  if (threadIdx.x == 0) scale[row] = sc;
  const float inv_sc = 1.0f / sc;

  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const int q = round_to_int(clampf(to_float(x[i]) * inv_sc, -kInt8Max, kInt8Max));
    y[i] = static_cast<int8_t>(q);
  }
}

// ---- Per-tensor pass 1: reduce global amax (atomicMax on fp32 bits) ----
template <typename scalar_t>
KS_GLOBAL void int8_amax_kernel(float* __restrict__ amax_out,
                                const scalar_t* __restrict__ input,
                                int64_t numel) {
  float local_amax = 0.f;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < numel;
       i += (int64_t)gridDim.x * blockDim.x) {
    local_amax = fmaxf(local_amax, fabsf(to_float(input[i])));
  }
  __shared__ float smem[kBlock / KS_WARP_SIZE];
  local_amax = block_reduce_max(local_amax, smem);
  if (threadIdx.x == 0) atomic_max_nonneg(amax_out, local_amax);
}

// ---- Per-tensor pass 2: amax -> scale = amax/qmax (single thread) ----
KS_GLOBAL void amax_to_scale_kernel(float* __restrict__ scale, float qmax) {
  const float amax = *scale;  // amax was accumulated in-place into scale[0]
  *scale = fmaxf(amax / qmax, kEpsScale);
}

// ---- Per-tensor pass 3: quantize using the finalized scale ----
template <typename scalar_t>
KS_GLOBAL void int8_quant_per_tensor_kernel(int8_t* __restrict__ out,
                                            const float* __restrict__ scale,
                                            const scalar_t* __restrict__ input,
                                            int64_t numel) {
  const float inv_sc = 1.0f / (*scale);
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < numel;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int q =
        round_to_int(clampf(to_float(input[i]) * inv_sc, -kInt8Max, kInt8Max));
    out[i] = static_cast<int8_t>(q);
  }
}

// ---- Dequantize: out = in_int8 * scale ----
template <typename scalar_t>
KS_GLOBAL void int8_dequant_kernel(scalar_t* __restrict__ out,
                                   const int8_t* __restrict__ input,
                                   const float* __restrict__ scale,
                                   int64_t rows, int64_t cols,
                                   bool per_token) {
  const int64_t numel = rows * cols;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < numel;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t row = i / cols;
    const float sc = per_token ? scale[row] : scale[0];
    out[i] = from_float<scalar_t>(static_cast<float>(input[i]) * sc);
  }
}

}  // namespace quant
}  // namespace ks

using namespace ks;

namespace {
// A reasonable grid cap for grid-stride elementwise passes.
inline unsigned grid_for(int64_t numel, int block) {
  int64_t g = (numel + block - 1) / block;
  if (g < 1) g = 1;
  if (g > 65535) g = 65535;
  return static_cast<unsigned>(g);
}
}  // namespace

extern "C" {

ks_status_t ks_quantize_int8(void* out, float* scale, const void* input,
                             int64_t rows, int64_t cols, ks_dtype_t in_dtype,
                             ks_quant_mode_t mode, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(scale);
  KS_CHECK_PTR(input);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_quantize_int8: bad shape");
  if (mode != KS_QUANT_PER_TENSOR && mode != KS_QUANT_PER_TOKEN)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_quantize_int8: mode must be per-tensor or per-token");

  const int64_t numel = rows * cols;
  auto s = to_stream(stream);
  auto* out8 = static_cast<int8_t*>(out);

  if (mode == KS_QUANT_PER_TOKEN) {
    if (rows > 2147483647LL)
      KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                      "ks_quantize_int8: rows exceeds grid limit");
    const dim3 grid(static_cast<unsigned>(rows));
    const dim3 block(quant::kBlock);
    KS_DISPATCH_FLOATING_TYPES(in_dtype, "ks_quantize_int8", {
      quant::int8_quant_per_token_kernel<scalar_t><<<grid, block, 0, s>>>(
          out8, scale, static_cast<const scalar_t*>(input), cols);
    });
    KS_CHECK_LAUNCH();
    return KS_SUCCESS;
  }

  // Per-tensor: zero the scale slot (used as the amax accumulator), reduce the
  // global amax, convert it to a scale, then quantize.
  ks_status_t z = ks::check(
      ks::gpuMemsetAsync(scale, 0, sizeof(float), s), "ks_quantize_int8");
  if (z != KS_SUCCESS) return z;

  const dim3 block(quant::kBlock);
  const dim3 grid(grid_for(numel, quant::kBlock));
  KS_DISPATCH_FLOATING_TYPES(in_dtype, "ks_quantize_int8", {
    quant::int8_amax_kernel<scalar_t><<<grid, block, 0, s>>>(
        scale, static_cast<const scalar_t*>(input), numel);
  });
  KS_CHECK_LAUNCH();

  quant::amax_to_scale_kernel<<<1, 1, 0, s>>>(scale, quant::kInt8Max);
  KS_CHECK_LAUNCH();

  KS_DISPATCH_FLOATING_TYPES(in_dtype, "ks_quantize_int8", {
    quant::int8_quant_per_tensor_kernel<scalar_t><<<grid, block, 0, s>>>(
        out8, scale, static_cast<const scalar_t*>(input), numel);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_dequantize_int8(void* out, const void* input,
                               const float* scale, int64_t rows, int64_t cols,
                               ks_dtype_t out_dtype, ks_quant_mode_t mode,
                               ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(input);
  KS_CHECK_PTR(scale);
  if (rows <= 0 || cols <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_dequantize_int8: bad shape");
  if (mode != KS_QUANT_PER_TENSOR && mode != KS_QUANT_PER_TOKEN)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_dequantize_int8: mode must be per-tensor or per-token");

  const int64_t numel = rows * cols;
  const bool per_token = (mode == KS_QUANT_PER_TOKEN);
  auto s = to_stream(stream);
  const dim3 block(quant::kBlock);
  const dim3 grid(grid_for(numel, quant::kBlock));

  KS_DISPATCH_FLOATING_TYPES(out_dtype, "ks_dequantize_int8", {
    quant::int8_dequant_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(out), static_cast<const int8_t*>(input), scale,
        rows, cols, per_token);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
