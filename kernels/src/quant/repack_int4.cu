#include "kernel_set/quant.h"
#include "quant/quant_common.cuh"

namespace ks {
namespace quant {

// GPTQ/AWQ to ks_gemm_w4a16 layout.
// Input: int32 [K/8, N]. Nibble j of word is at (word >> 4j) & 0xF.
// Output: uint8 [K/2, N]. Byte = (q_even & 0xF) | ((q_odd & 0xF) << 4).
KS_GLOBAL void repack_int4_kernel(uint8_t* __restrict__ out,
                                  const int32_t* __restrict__ in,
                                  int64_t k, int64_t n) {
  const int64_t numel = (k / 2) * n;
  for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
       idx < numel; idx += (int64_t)gridDim.x * blockDim.x) {
    const int64_t out_row = idx / n;  // in [0, K/2)
    const int64_t col = idx % n;      // in [0, N)

    const int64_t k_even = out_row * 2;
    const int64_t k_odd = k_even + 1;

    // Read even nibble
    const int64_t word_idx_even = (k_even / 8) * n + col;
    const int shift_even = (k_even % 8) * 4;
    const int q_even = (in[word_idx_even] >> shift_even) & 0xF;

    // Read odd nibble
    const int64_t word_idx_odd = (k_odd / 8) * n + col;
    const int shift_odd = (k_odd % 8) * 4;
    const int q_odd = (in[word_idx_odd] >> shift_odd) & 0xF;

    out[idx] = static_cast<uint8_t>(q_even | (q_odd << 4));
  }
}

}  // namespace quant
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_repack_int4(void* out_packed, const void* qweight,
                           const void* perm, int64_t size_k, int64_t size_n,
                           int num_bits, ks_stream_t stream) {
  KS_CHECK_PTR(out_packed);
  KS_CHECK_PTR(qweight);
  if (size_k <= 0 || size_n <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_repack_int4: bad shape");
  if (num_bits != 4)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_repack_int4: only 4-bit supported");
  if (perm != nullptr)
    KS_RETURN_ERROR(KS_ERROR_NOT_IMPLEMENTED, "ks_repack_int4: act-order perm not yet supported");

  auto s = to_stream(stream);
  const int64_t numel = (size_k / 2) * size_n;
  
  int64_t g = (numel + quant::kBlock - 1) / quant::kBlock;
  if (g < 1) g = 1;
  if (g > 65535) g = 65535;
  const dim3 grid(static_cast<unsigned>(g));
  const dim3 block(quant::kBlock);

  quant::repack_int4_kernel<<<grid, block, 0, s>>>(
      static_cast<uint8_t*>(out_packed),
      static_cast<const int32_t*>(qweight),
      size_k, size_n);

  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
