#include "kernel_set/quant.h"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include <cmath>

namespace ks {
namespace quant {

// Naive correctness fallback
KS_GLOBAL void quantize_nvfp4_kernel(uint8_t* __restrict__ out_fp4,
                                     void* __restrict__ out_scales,
                                     const float* __restrict__ input,
                                     float global_scale, int64_t rows, int64_t cols) {
  // NOT IMPLEMENTED for now as true quantization to e2m1 requires block scale computation
}

} // namespace quant
} // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_quantize_nvfp4(void* out_fp4, void* out_scales,
                              const void* input, float global_scale,
                              int64_t rows, int64_t cols,
                              ks_dtype_t in_dtype, ks_stream_t stream) {
  return KS_ERROR_NOT_IMPLEMENTED;
}

} // extern "C"
