// kernel-set — group-wise INT4 weight dequantization (AWQ / GPTQ style).
//
// Layout (matches the quant.h contract):
//   qweight_packed : int32  [K/8, N]   8 signed-less nibbles per word, packed
//                                       along K. Nibble j (j = 0..7) of word
//                                       [ko, n] is the 4-bit code for
//                                       k = ko*8 + j:
//                                         q = (word >> (4*j)) & 0xF   in [0,15].
//   scales         : out_dtype [K/group_size, N]
//   zeros          : out_dtype [K/group_size, N]   zero-point in the quantized
//                                       (code) domain, stored as a floating
//                                       value (integer-valued, e.g. 8.0f for a
//                                       symmetric 4-bit grid).
//   out            : out_dtype [K, N]
//
// Dequant:  out[k, n] = (float(q) - float(zero[g, n])) * scale[g, n],
//           with group index g = k / group_size.
//
// This is the affine de-quant used by AWQ/GPTQ inference kernels. The classic
// AWQ format also packs the zero-points as int4 and uses a column-interleaved
// nibble order for its fused GEMM; here the ABI hands us a plain (de-packed)
// zeros tensor and the natural sequential nibble order, which is the simplest
// correct mapping. Further tuning (best-effort note): a fused dequant+GEMM
// (CUTLASS mixed-input or the AWQ interleaved layout) avoids materializing the
// [K, N] fp tensor entirely; this standalone dequant is for correctness and
// for paths that need an explicit weight.
#include "kernel_set/quant.h"
#include "quant/quant_common.cuh"

namespace ks {
namespace quant {

// One thread per (k, n) output element. q is unpacked from its nibble; the
// group row picks the scale/zero. Grid-stride over the full [K, N] domain.
template <typename scalar_t>
KS_GLOBAL void dequant_int4_kernel(scalar_t* __restrict__ out,
                                   const int32_t* __restrict__ qweight,
                                   const scalar_t* __restrict__ scales,
                                   const scalar_t* __restrict__ zeros,
                                   int64_t k, int64_t n, int group_size) {
  const int64_t numel = k * n;
  for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
       idx < numel; idx += (int64_t)gridDim.x * blockDim.x) {
    const int64_t kk = idx / n;  // row in [0, K)
    const int64_t nn = idx % n;  // col in [0, N)

    // Unpack the 4-bit code: 8 nibbles per int32 word along K.
    const int64_t word_row = kk >> 3;        // kk / 8
    const int shift = static_cast<int>((kk & 7) << 2);  // (kk % 8) * 4
    const int32_t word = qweight[word_row * n + nn];
    const int q = (word >> shift) & 0xF;     // code in [0, 15]

    // Group-wise scale / zero.
    const int64_t g = kk / group_size;
    const float scale = to_float(scales[g * n + nn]);
    const float zero = to_float(zeros[g * n + nn]);

    out[idx] = from_float<scalar_t>((static_cast<float>(q) - zero) * scale);
  }
}

}  // namespace quant
}  // namespace ks

using namespace ks;

namespace {
inline unsigned grid_for_int4(int64_t numel, int block) {
  int64_t g = (numel + block - 1) / block;
  if (g < 1) g = 1;
  if (g > 65535) g = 65535;
  return static_cast<unsigned>(g);
}
}  // namespace

extern "C" {

ks_status_t ks_dequantize_int4(void* out, const void* qweight_packed,
                               const void* scales, const void* zeros,
                               int64_t k, int64_t n, int group_size,
                               ks_dtype_t out_dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(qweight_packed);
  KS_CHECK_PTR(scales);
  KS_CHECK_PTR(zeros);
  if (k <= 0 || n <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_dequantize_int4: bad shape");
  if (group_size <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_dequantize_int4: group_size must be > 0");
  if (k % 8 != 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_dequantize_int4: K must be a multiple of 8 (8 codes/word)");
  if (k % group_size != 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_dequantize_int4: K must be a multiple of group_size");

  const int64_t numel = k * n;
  auto s = to_stream(stream);
  const dim3 block(quant::kBlock);
  const dim3 grid(grid_for_int4(numel, quant::kBlock));

  KS_DISPATCH_FLOATING_TYPES(out_dtype, "ks_dequantize_int4", {
    quant::dequant_int4_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(out),
        static_cast<const int32_t*>(qweight_packed),
        static_cast<const scalar_t*>(scales),
        static_cast<const scalar_t*>(zeros), k, n, group_size);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
