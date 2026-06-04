// kernel-set — W4A16 GEMM: fp16/bf16 A [M,K] x int4 weights B [K,N], group-wise.
//
//   C = A @ dequant(B) + bias
//   dequant(B)[k,n] = (q[k,n] - zero[g,n]) * scale[g,n],  g = k / group_size
//
// Layouts (documented contract for this reference kernel):
//   * b_packed : two int4 per byte, packed along K. Logical weight is [K,N];
//     the byte at b_packed[(k/2)*N + n] holds:
//        low  nibble (bits 0..3) -> q[k = 2*(k/2),   n]
//        high nibble (bits 4..7) -> q[k = 2*(k/2)+1, n]
//     i.e. even-K in the low nibble, odd-K in the high nibble. Quant values are
//     unsigned 0..15 (asymmetric: the zero-point recenters them). Total packed
//     size is (K/2)*N bytes (K assumed even).
//   * scales : [K/group_size, N] in `dtype` (same as activations).
//   * zeros  : [K/group_size, N] in `dtype`, the per-group zero-point as a real
//     value (already dequantized: w = (q - zero)*scale). NULL => zero = 0.
//   * bias   : [N] in `dtype` or NULL.
//   * group_size : number of consecutive K that share one (scale,zero). Must
//     divide K. group_size <= 0 is treated as a single group (= K).
//
// This is the *correct, auditable* reference. The production path is a Marlin /
// Machete style kernel: int4 weights are repacked into a tensor-core friendly
// interleave, dequantized in registers to fp16 just before mma.sync, with a
// cp.async multistage pipeline and group-scale application fused into the
// epilogue. We instead dequantize each weight to fp32 in the inner loop and run
// the same register-tiled SIMT GEMM used by the f32 path.
#include "kernel_set/gemm.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace gemm {

constexpr int kW4TileM = 64;
constexpr int kW4TileN = 64;
constexpr int kW4TileK = 16;
constexpr int kW4ThrM = 4;
constexpr int kW4ThrN = 4;
constexpr int kW4BlockThreads = (kW4TileM / kW4ThrM) * (kW4TileN / kW4ThrN);

// Unpack the int4 at logical (k,n) from the K-packed byte stream.
KS_DI int unpack_q4(const uint8_t* __restrict__ b_packed, int64_t k, int64_t n,
                    int64_t cols_n) {
  const uint8_t byte = b_packed[(k >> 1) * cols_n + n];
  return (k & 1) ? int((byte >> 4) & 0xF) : int(byte & 0xF);
}

template <typename scalar_t>
KS_GLOBAL void gemm_w4a16_kernel(scalar_t* __restrict__ c,
                                 const scalar_t* __restrict__ a,
                                 const uint8_t* __restrict__ b_packed,
                                 const scalar_t* __restrict__ scales,
                                 const scalar_t* __restrict__ zeros,
                                 const scalar_t* __restrict__ bias, int64_t m,
                                 int64_t n, int64_t k, int group_size) {
  __shared__ float as[kW4TileM][kW4TileK];
  __shared__ float bs[kW4TileK][kW4TileN];  // dequantized weight tile (fp32)

  const int block_row = blockIdx.y * kW4TileM;
  const int block_col = blockIdx.x * kW4TileN;
  const int64_t groups_per_col = (k + group_size - 1) / group_size;  // K/gs

  const int tid = threadIdx.x;
  const int tr = tid / (kW4TileN / kW4ThrN);
  const int tc = tid % (kW4TileN / kW4ThrN);

  float acc[kW4ThrM][kW4ThrN];
#pragma unroll
  for (int i = 0; i < kW4ThrM; ++i)
#pragma unroll
    for (int j = 0; j < kW4ThrN; ++j) acc[i][j] = 0.0f;

  for (int64_t k0 = 0; k0 < k; k0 += kW4TileK) {
    // Stage A (fp16/bf16 -> fp32).
    for (int idx = tid; idx < kW4TileM * kW4TileK; idx += kW4BlockThreads) {
      const int r = idx / kW4TileK;
      const int cc = idx % kW4TileK;
      const int64_t gm = block_row + r;
      const int64_t gk = k0 + cc;
      as[r][cc] = (gm < m && gk < k) ? to_float(a[gm * k + gk]) : 0.0f;
    }
    // Stage B: unpack int4, apply group-wise (scale, zero) -> fp32 weight.
    for (int idx = tid; idx < kW4TileK * kW4TileN; idx += kW4BlockThreads) {
      const int r = idx / kW4TileN;
      const int cc = idx % kW4TileN;
      const int64_t gk = k0 + r;
      const int64_t gn = block_col + cc;
      float w = 0.0f;
      if (gk < k && gn < n) {
        const int q = unpack_q4(b_packed, gk, gn, n);
        const int64_t g = gk / group_size;
        const float sc = to_float(scales[g * n + gn]);
        const float zp = zeros ? to_float(zeros[g * n + gn]) : 0.0f;
        w = (static_cast<float>(q) - zp) * sc;
      }
      bs[r][cc] = w;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kW4TileK; ++kk) {
      float areg[kW4ThrM], breg[kW4ThrN];
#pragma unroll
      for (int i = 0; i < kW4ThrM; ++i) areg[i] = as[tr * kW4ThrM + i][kk];
#pragma unroll
      for (int j = 0; j < kW4ThrN; ++j) breg[j] = bs[kk][tc * kW4ThrN + j];
#pragma unroll
      for (int i = 0; i < kW4ThrM; ++i)
#pragma unroll
        for (int j = 0; j < kW4ThrN; ++j) acc[i][j] += areg[i] * breg[j];
    }
    __syncthreads();
  }
  (void)groups_per_col;

#pragma unroll
  for (int i = 0; i < kW4ThrM; ++i) {
    const int64_t gm = block_row + tr * kW4ThrM + i;
    if (gm >= m) continue;
#pragma unroll
    for (int j = 0; j < kW4ThrN; ++j) {
      const int64_t gn = block_col + tc * kW4ThrN + j;
      if (gn >= n) continue;
      float v = acc[i][j];
      if (bias) v += to_float(bias[gn]);
      c[gm * n + gn] = from_float<scalar_t>(v);
    }
  }
}

template <typename scalar_t>
void launch_w4a16(scalar_t* c, const scalar_t* a, const uint8_t* b_packed,
                  const scalar_t* scales, const scalar_t* zeros,
                  const scalar_t* bias, int64_t m, int64_t n, int64_t k,
                  int group_size, gpuStream_t s) {
  dim3 grid(static_cast<unsigned>((n + kW4TileN - 1) / kW4TileN),
            static_cast<unsigned>((m + kW4TileM - 1) / kW4TileM));
  dim3 block(kW4BlockThreads);
  gemm_w4a16_kernel<scalar_t><<<grid, block, 0, s>>>(
      c, a, b_packed, scales, zeros, bias, m, n, k, group_size);
}

}  // namespace gemm
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_gemm_w4a16(void* c, const void* a, const void* b_packed,
                          const void* scales, const void* zeros,
                          const void* bias, int64_t m, int64_t n, int64_t k,
                          int group_size, ks_dtype_t dtype,
                          ks_stream_t stream) {
  KS_CHECK_PTR(c);
  KS_CHECK_PTR(a);
  KS_CHECK_PTR(b_packed);
  KS_CHECK_PTR(scales);  // zeros and bias may be NULL.
  if (m <= 0 || n <= 0 || k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_gemm_w4a16: bad shape");
  if (k % 2 != 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_gemm_w4a16: K must be even (two int4 per byte)");
  // A single group spans all of K when group_size is unset/invalid.
  const int gs = (group_size > 0) ? group_size : static_cast<int>(k);
  if (k % gs != 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_gemm_w4a16: group_size must divide K");

  auto s = to_stream(stream);
  KS_DISPATCH_HALF_TYPES(dtype, "ks_gemm_w4a16", {
    gemm::launch_w4a16<scalar_t>(
        static_cast<scalar_t*>(c), static_cast<const scalar_t*>(a),
        static_cast<const uint8_t*>(b_packed),
        static_cast<const scalar_t*>(scales),
        static_cast<const scalar_t*>(zeros), static_cast<const scalar_t*>(bias),
        m, n, k, gs, s);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
