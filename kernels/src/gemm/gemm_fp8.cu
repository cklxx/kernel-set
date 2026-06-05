// kernel-set — FP8 GEMM: fp8 A [M,K] x fp8 B [K,N] -> fp32 acc -> dequant.
//
//   C = (a_scale (x) b_scale) * (A_fp8 @ B_fp8)
//     a_scale: per-token  [M]  (KS_QUANT_PER_TOKEN)   or [1] (PER_TENSOR)
//     b_scale: per-channel [N] (KS_QUANT_PER_CHANNEL) or [1] (PER_TENSOR)
//
// Row-major, non-transposed (A row stride K, B row stride N) — the standard
// quantized-linear layout (B is the transposed-and-quantized weight, already
// stored [K,N]).
//
// ===========================================================================
// CORRECTNESS-FIRST DESIGN (what ships, on EVERY arch)
// ===========================================================================
// The default and only-enabled compute path is a tiled SIMT GEMM:
//   * load fp8 (__nv_fp8_e4m3 / __nv_fp8_e5m2) operands,
//   * convert each element to fp32 with ks::to_float (a SOFTWARE conversion on
//     pre-sm89, a hardware conversion on Ada/Hopper — but always *available*
//     because <cuda_fp8.h> is included on every nvcc pass via KS_HAS_FP8_TYPES),
//   * accumulate the inner product in fp32,
//   * apply the per-token/per-tensor A scale (x) per-channel/per-tensor B scale
//     and write `out_dtype` (bf16 / fp16 / fp32).
// Because the fp8->fp32 conversion is available on all targeted arches, this
// path compiles and runs correctly on sm_80 (A100, no hardware fp8) as well as
// sm_89 (L4/Ada) and sm_90 (Hopper). It is verifiable by inspection and mirrors
// the validated SIMT path of gemm.cu / gemm_w8a8.cu (same 64x64 block tile, 4x4
// register micro-tile, K streamed in 16-wide slabs through shared memory).
//
// ===========================================================================
// PERF UPGRADE (NOT enabled — documented, gated, intentionally inert)
// ===========================================================================
// On sm_89+ the hardware fp8 tensor cores (mma.sync.aligned.*.f8 via CUTLASS /
// DeepGEMM, or wmma fp8 fragments) deliver multiples of this SIMT throughput.
// That path is the production replacement, but — exactly like the WMMA fp16
// path in gemm.cu, which was found to return wrong results on real L4 hardware —
// it is NOT shipped as the default and is NOT trusted until verified on a GPU.
// It is therefore left entirely behind KS_ENABLE_FP8_MMA (off) and is a no-op
// stub; the SIMT dequant path is what every public call runs. Should a hardware
// fp8-mma path be enabled in the future, ks_gemm_fp8 must additionally
// runtime-gate on device_has_fp8() and return KS_ERROR_ARCH_UNSUPPORTED when the
// device lacks sm_89+. The SIMT path needs no such gate (it works everywhere).
#include "kernel_set/gemm.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

// FP8 conversion TYPES (__nv_fp8_e4m3 / __nv_fp8_e5m2 + ks::to_float overloads)
// are available wherever <cuda_fp8.h> was pulled in — i.e. on every nvcc pass
// (host + all device arches), guarded by KS_HAS_FP8_TYPES in platform.cuh. The
// kernel symbols must exist for EVERY targeted arch or an sm_80-only build would
// leave them undefined and the .so would fail to dlopen on A100/T4. Mirror the
// guard used by quant_fp8.cu so the gating is consistent across categories.
#if defined(KS_HAS_FP8_TYPES)
#define KS_GEMM_FP8_AVAILABLE 1
#endif

namespace ks {
namespace gemm {

#if defined(KS_GEMM_FP8_AVAILABLE)

// Tiling mirrors the SIMT f32 GEMM and the W8A8 GEMM: 64x64 block tile, 4x4
// register micro-tile, K streamed in 16-wide slabs through shared memory. We
// stage A/B into shared memory as fp32 (converted on the cooperative load) so
// the inner loop is plain fp32 FMA — identical in structure to the verified
// gemm_simt_kernel, which is what makes this auditable without a GPU.
constexpr int kF8TileM = 64;
constexpr int kF8TileN = 64;
constexpr int kF8TileK = 16;
constexpr int kF8ThrM = 4;
constexpr int kF8ThrN = 4;
constexpr int kF8BlockThreads = (kF8TileM / kF8ThrM) * (kF8TileN / kF8ThrN);  // 256

// fp8_t is __nv_fp8_e4m3 or __nv_fp8_e5m2; out_t is float / __half / bf16.
template <typename fp8_t, typename out_t>
KS_GLOBAL void gemm_fp8_kernel(out_t* __restrict__ c,
                               const fp8_t* __restrict__ a,
                               const fp8_t* __restrict__ b,
                               const float* __restrict__ a_scale,
                               const float* __restrict__ b_scale, int64_t m,
                               int64_t n, int64_t k, int a_per_token,
                               int b_per_channel) {
  // Shared fp32 tiles. A: [kF8TileM][kF8TileK], B: [kF8TileK][kF8TileN].
  // Staging as fp32 (not fp8) keeps the inner product a clean fp32 FMA loop and
  // lets one staged value feed kF8ThrM/kF8ThrN reuse out of registers.
  __shared__ float as[kF8TileM][kF8TileK];
  __shared__ float bs[kF8TileK][kF8TileN];

  const int block_row = blockIdx.y * kF8TileM;  // first output row of this tile
  const int block_col = blockIdx.x * kF8TileN;  // first output col of this tile

  // Linear thread id -> (thread row, thread col) inside the 16x16 thread grid.
  const int tid = threadIdx.x;
  const int tr = tid / (kF8TileN / kF8ThrN);  // 0..15
  const int tc = tid % (kF8TileN / kF8ThrN);  // 0..15

  float acc[kF8ThrM][kF8ThrN];
#pragma unroll
  for (int i = 0; i < kF8ThrM; ++i)
#pragma unroll
    for (int j = 0; j < kF8ThrN; ++j) acc[i][j] = 0.0f;

  for (int64_t k0 = 0; k0 < k; k0 += kF8TileK) {
    // Cooperatively stage A tile [kF8TileM x kF8TileK]: 256 threads load 1024
    // elements -> 4 each. Convert fp8 -> fp32 at load; zero-fill past the edge.
    for (int idx = tid; idx < kF8TileM * kF8TileK; idx += kF8BlockThreads) {
      const int r = idx / kF8TileK;
      const int cc = idx % kF8TileK;
      const int64_t gm = block_row + r;
      const int64_t gk = k0 + cc;
      as[r][cc] = (gm < m && gk < k)
                      ? to_float(a[gm * k + gk])   // A [M,K] row-major
                      : 0.0f;
    }
    // Stage B tile [kF8TileK x kF8TileN].
    for (int idx = tid; idx < kF8TileK * kF8TileN; idx += kF8BlockThreads) {
      const int r = idx / kF8TileN;
      const int cc = idx % kF8TileN;
      const int64_t gk = k0 + r;
      const int64_t gn = block_col + cc;
      bs[r][cc] = (gk < k && gn < n)
                      ? to_float(b[gk * n + gn])   // B [K,N] row-major
                      : 0.0f;
    }
    __syncthreads();

    // Inner product over the K slab (fp32 FMA, register-blocked 4x4).
#pragma unroll
    for (int kk = 0; kk < kF8TileK; ++kk) {
      float areg[kF8ThrM], breg[kF8ThrN];
#pragma unroll
      for (int i = 0; i < kF8ThrM; ++i) areg[i] = as[tr * kF8ThrM + i][kk];
#pragma unroll
      for (int j = 0; j < kF8ThrN; ++j) breg[j] = bs[kk][tc * kF8ThrN + j];
#pragma unroll
      for (int i = 0; i < kF8ThrM; ++i)
#pragma unroll
        for (int j = 0; j < kF8ThrN; ++j) acc[i][j] += areg[i] * breg[j];
    }
    __syncthreads();
  }

  // Dequant epilogue: C = (a_scale (x) b_scale) * acc, written as out_dtype.
#pragma unroll
  for (int i = 0; i < kF8ThrM; ++i) {
    const int64_t gm = block_row + tr * kF8ThrM + i;
    if (gm >= m) continue;
    const float as_scale = a_per_token ? a_scale[gm] : a_scale[0];
#pragma unroll
    for (int j = 0; j < kF8ThrN; ++j) {
      const int64_t gn = block_col + tc * kF8ThrN + j;
      if (gn >= n) continue;
      const float bs_scale = b_per_channel ? b_scale[gn] : b_scale[0];
      const float v = acc[i][j] * as_scale * bs_scale;
      c[gm * n + gn] = from_float<out_t>(v);
    }
  }
}

template <typename fp8_t, typename out_t>
void launch_fp8(out_t* c, const fp8_t* a, const fp8_t* b, const float* a_scale,
                const float* b_scale, int64_t m, int64_t n, int64_t k,
                int a_per_token, int b_per_channel, gpuStream_t s) {
  dim3 grid(static_cast<unsigned>((n + kF8TileN - 1) / kF8TileN),
            static_cast<unsigned>((m + kF8TileM - 1) / kF8TileM));
  dim3 block(kF8BlockThreads);
  gemm_fp8_kernel<fp8_t, out_t><<<grid, block, 0, s>>>(
      c, a, b, a_scale, b_scale, m, n, k, a_per_token, b_per_channel);
}

// Dispatch over the fp8 storage type, then the output type. Keeps the two enum
// switches out of the public function body so the structure matches gemm_w8a8.
template <typename fp8_t>
ks_status_t dispatch_out(void* c, const fp8_t* a, const fp8_t* b,
                         const float* a_scale, const float* b_scale, int64_t m,
                         int64_t n, int64_t k, int a_per_token, int b_per_channel,
                         ks_dtype_t out_dtype, gpuStream_t s) {
  switch (out_dtype) {
    case KS_DTYPE_F32:
      launch_fp8<fp8_t, float>(static_cast<float*>(c), a, b, a_scale, b_scale, m,
                               n, k, a_per_token, b_per_channel, s);
      return KS_SUCCESS;
    case KS_DTYPE_F16:
      launch_fp8<fp8_t, __half>(static_cast<__half*>(c), a, b, a_scale, b_scale,
                                m, n, k, a_per_token, b_per_channel, s);
      return KS_SUCCESS;
    case KS_DTYPE_BF16:
      launch_fp8<fp8_t, __nv_bfloat16>(static_cast<__nv_bfloat16*>(c), a, b,
                                       a_scale, b_scale, m, n, k, a_per_token,
                                       b_per_channel, s);
      return KS_SUCCESS;
    default:
      KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                      "ks_gemm_fp8: out_dtype must be f32/f16/bf16");
  }
}

#endif  // KS_GEMM_FP8_AVAILABLE

}  // namespace gemm
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_gemm_fp8(void* out, const void* a_fp8, const void* b_fp8,
                        const float* a_scale, const float* b_scale, int64_t m,
                        int64_t n, int64_t k, ks_quant_mode_t a_mode,
                        ks_quant_mode_t b_mode, ks_dtype_t fp8_dtype,
                        ks_dtype_t out_dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(a_fp8);
  KS_CHECK_PTR(b_fp8);
  KS_CHECK_PTR(a_scale);
  KS_CHECK_PTR(b_scale);
  if (m <= 0 || n <= 0 || k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_gemm_fp8: bad shape");

  // The kernel indexes A as gm*k+gk, B as gk*n+gn, C as gm*n+gn (all int64). For
  // an int64 ABI those products can signed-overflow on pathological shapes, which
  // would silently turn a valid-looking call into OOB access. Reject any shape
  // whose index products cannot be represented in int64. (m,n,k are all > 0.)
  constexpr int64_t kI64Max = 9223372036854775807LL;  // INT64_MAX
  if (m > kI64Max / k || k > kI64Max / n || m > kI64Max / n)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_gemm_fp8: M*K / K*N / M*N exceeds int64 range");

  // Tile counts feed the CUDA grid (grid.x = N tiles, grid.y = M tiles). Compute
  // them in uint64 (so m+63 / n+63 cannot overflow) and reject anything past the
  // conservative grid limits before the dim3 cast in launch_fp8.
  constexpr uint64_t kMaxGridX = 2147483647ULL;  // grid.x bound (conservative)
  constexpr uint64_t kMaxGridY = 65535ULL;       // grid.y bound
  const uint64_t tiles_n = (static_cast<uint64_t>(n) + 63) / 64;
  const uint64_t tiles_m = (static_cast<uint64_t>(m) + 63) / 64;
  if (tiles_n > kMaxGridX || tiles_m > kMaxGridY)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_gemm_fp8: tile grid exceeds CUDA grid limits");

  // a_scale is per-token [M] for PER_TOKEN, else [1]; b_scale is per-channel [N]
  // for PER_CHANNEL, else [1]. Reject granularities this kernel can't honor
  // (mirrors ks_gemm_w8a8).
  if (a_mode != KS_QUANT_PER_TENSOR && a_mode != KS_QUANT_PER_TOKEN)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_gemm_fp8: a_mode must be per-tensor or per-token");
  if (b_mode != KS_QUANT_PER_TENSOR && b_mode != KS_QUANT_PER_CHANNEL)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_gemm_fp8: b_mode must be per-tensor or per-channel");
  if (fp8_dtype != KS_DTYPE_F8E4M3 && fp8_dtype != KS_DTYPE_F8E5M2)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                    "ks_gemm_fp8: fp8_dtype must be e4m3 or e5m2");
  if (out_dtype != KS_DTYPE_F32 && out_dtype != KS_DTYPE_F16 &&
      out_dtype != KS_DTYPE_BF16)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                    "ks_gemm_fp8: out_dtype must be f32/f16/bf16");

  const int a_per_token = (a_mode == KS_QUANT_PER_TOKEN) ? 1 : 0;
  const int b_per_channel = (b_mode == KS_QUANT_PER_CHANNEL) ? 1 : 0;

#if defined(KS_GEMM_FP8_AVAILABLE)
  // NOTE on the runtime arch gate: the SIMT dequant path below converts fp8 in
  // SOFTWARE and works on every device (sm_80+), so it deliberately does NOT
  // return KS_ERROR_ARCH_UNSUPPORTED. The hardware fp8-mma path is intentionally
  // not enabled (see header comment); were it enabled, this is where a
  // device_has_fp8() check would gate it to sm_89+ and fall back to SIMT (or
  // return KS_ERROR_ARCH_UNSUPPORTED) on older GPUs.
  auto s = to_stream(stream);
  ks_status_t st;
  if (fp8_dtype == KS_DTYPE_F8E4M3) {
    st = gemm::dispatch_out<__nv_fp8_e4m3>(
        out, static_cast<const __nv_fp8_e4m3*>(a_fp8),
        static_cast<const __nv_fp8_e4m3*>(b_fp8), a_scale, b_scale, m, n, k,
        a_per_token, b_per_channel, out_dtype, s);
  } else {
    st = gemm::dispatch_out<__nv_fp8_e5m2>(
        out, static_cast<const __nv_fp8_e5m2*>(a_fp8),
        static_cast<const __nv_fp8_e5m2*>(b_fp8), a_scale, b_scale, m, n, k,
        a_per_token, b_per_channel, out_dtype, s);
  }
  if (st != KS_SUCCESS) return st;
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
#else
  // FP8 conversion types unavailable in this build (no nvcc / <cuda_fp8.h>): the
  // .so should still link, so report the missing capability defensively rather
  // than fail to compile. On a normal CUDA 12.x build KS_GEMM_FP8_AVAILABLE is
  // always defined (the host pass pulls in <cuda_fp8.h>).
  KS_RETURN_ERROR(KS_ERROR_ARCH_UNSUPPORTED,
                  "ks_gemm_fp8: built without FP8 type support");
#endif
}

}  // extern "C"
