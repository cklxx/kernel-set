#pragma once

#include "kernel_set/kernel_set.h"

#ifdef __cplusplus
extern "C" {
#endif

// Transpose a 2D tensor [M, K] -> [K, M] (fp16 only).
// out: [K, M] device ptr, in: [M, K] device ptr.
KS_API ks_status_t ks_transpose_2d(
    void *out, const void *in, int M, int K, ks_dtype_t dtype, ks_stream_t stream);

// Permute NCHW [N, C, H, W] -> NHWC [N, H, W, C] (fp16 only).
KS_API ks_status_t ks_nchw_to_nhwc(
    void *out, const void *in, int N, int C, int H, int W, ks_dtype_t dtype, ks_stream_t stream);

// Permute NHWC [N, H, W, C] -> NCHW [N, C, H, W] (fp16 only).
KS_API ks_status_t ks_nhwc_to_nchw(
    void *out, const void *in, int N, int H, int W, int C, ks_dtype_t dtype, ks_stream_t stream);

// Nearest-neighbor 2x upsample: in [N, C, H, W] -> out [N, C, 2H, 2W] (fp16 only).
KS_API ks_status_t ks_upsample_nearest_2x(
    void *out, const void *in, int N, int C, int H, int W, ks_dtype_t dtype, ks_stream_t stream);

#ifdef __cplusplus
}
#endif