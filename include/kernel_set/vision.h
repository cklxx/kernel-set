/*
 * kernel-set — vision ops (conv2d, group norm) for ViT / VQ decoder backbones.
 *
 * All tensors are NCHW device pointers. Dtype is a runtime ks_dtype_t;
 * accumulation is always in fp32.
 */
#ifndef KERNEL_SET_VISION_H_
#define KERNEL_SET_VISION_H_

#include "kernel_set/types.h"

KS_BEGIN_EXTERN_C

/* 2D convolution (NCHW layout).
 *   input:  [N, C, H, W]
 *   weight: [K, C/groups, R, S]
 *   bias:   [K] (may be NULL)
 *   output: [N, K, OH, OW]
 *
 * padding_h/padding_w apply symmetric zero-padding before the convolution.
 * dilation > 1 is supported (no performance penalty on the im2col path). */
KS_API ks_status_t ks_conv2d(
    void* out, const void* input, const void* weight, const void* bias,
    int n, int c, int h, int w, int k, int r, int s,
    int stride_h, int stride_w, int padding_h, int padding_w,
    int dilation_h, int dilation_w, int groups,
    ks_dtype_t dtype, ks_stream_t stream);

/* Group Normalization (NCHW layout).
 *   input:  [N, C, H, W]
 *   weight: [C] (may be NULL, defaults to 1)
 *   bias:   [C] (may be NULL, defaults to 0)
 *   output: [N, C, H, W]
 *
 * num_groups divides C evenly. Statistics are computed per (N, group) slice. */
KS_API ks_status_t ks_group_norm(
    void* out, const void* input, const void* weight, const void* bias,
    int n, int c, int hw, int num_groups, float eps,
    ks_dtype_t dtype, ks_stream_t stream);

KS_END_EXTERN_C

#endif /* KERNEL_SET_VISION_H_ */