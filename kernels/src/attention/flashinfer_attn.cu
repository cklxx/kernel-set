// kernel-set — FlashInfer prefill attention dispatch (sm_75+).
//
// Wraps FlashInfer's header-only SinglePrefillWithKVCacheDispatched into the
// kernel-set C ABI. The template is instantiated for the dtype/head_dim
// combinations we support and dispatched at runtime.
//
// Architectures:
//   sm_75+ (Turing, Ampere, Ada, Hopper): FlashInfer tensor-core prefill
//   sm_70  (Volta / V100):            falls back to the existing ks_flash_attn
//                                      kernel (mma/ldmatrix not available on sm_70)
//
// Guarded by __CUDA_ARCH__ >= 750 so the CUDA compiler skips device code
// generation for unsupported architectures. The host entry point is always
// compiled and does a runtime SM check.
#include "kernel_set/attention.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "attention_common.cuh"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750

#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>

namespace ks {
namespace attention {

template <typename scalar_t, int HEAD_DIM, flashinfer::MaskMode MASK_MODE>
ks_status_t fi_prefill_launch(void* out, float* lse, const void* q,
                               const void* k, const void* v, int batch,
                               int seqlen_q, int seqlen_k, int num_heads,
                               int num_kv_heads, float scale, gpuStream_t s) {
  using DType = scalar_t;
  using Params = flashinfer::SinglePrefillParams<DType, DType, DType>;
  using AttentionVariant =
      flashinfer::DefaultAttention</*use_custom_mask=*/false,
                                   /*use_sliding_window=*/false,
                                   /*use_logits_soft_cap=*/false,
                                   /*use_alibi=*/false>;

  for (int b = 0; b < batch; ++b) {
    const int64_t q_rows = static_cast<int64_t>(seqlen_q) * num_heads;
    const int64_t kv_rows = static_cast<int64_t>(seqlen_k) * num_kv_heads;
    const int64_t q_off = static_cast<int64_t>(b) * q_rows * HEAD_DIM;
    const int64_t k_off = static_cast<int64_t>(b) * kv_rows * HEAD_DIM;
    const int64_t o_off = static_cast<int64_t>(b) * q_rows * HEAD_DIM;

    Params params;
    params.q = const_cast<DType*>(static_cast<const DType*>(q)) + q_off;
    params.k = const_cast<DType*>(static_cast<const DType*>(k)) + k_off;
    params.v = const_cast<DType*>(static_cast<const DType*>(v)) + k_off;
    params.o = static_cast<DType*>(out) + o_off;
    params.lse = lse ? (lse + b * q_rows) : nullptr;
    params.num_qo_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.group_size = flashinfer::uint_fastdiv(num_heads / num_kv_heads);
    params.qo_len = seqlen_q;
    params.kv_len = seqlen_k;
    params.q_stride_n = num_heads * HEAD_DIM;
    params.q_stride_h = HEAD_DIM;
    params.k_stride_n = num_kv_heads * HEAD_DIM;
    params.k_stride_h = HEAD_DIM;
    params.v_stride_n = num_kv_heads * HEAD_DIM;
    params.v_stride_h = HEAD_DIM;
    params.window_left = -1;
    params.partition_kv = false;
    params.head_dim = HEAD_DIM;
    params.sm_scale = scale;
    params.logits_soft_cap = 0.f;
    params.rope_rcp_scale = 1.f;
    params.rope_rcp_theta = 1.f;
    params.maybe_custom_mask = nullptr;
    params.maybe_alibi_slopes = nullptr;

    constexpr bool USE_FP16_QK_REDUCTION = false;
    constexpr auto POS_ENCODING = flashinfer::PosEncodingMode::kNone;

    cudaError_t err = flashinfer::SinglePrefillWithKVCacheDispatched<
        HEAD_DIM, HEAD_DIM, POS_ENCODING, USE_FP16_QK_REDUCTION, MASK_MODE,
        AttentionVariant>(params, /*tmp=*/nullptr, s);
    if (err != cudaSuccess) {
      KS_RETURN_ERROR(KS_ERROR_CUDA, "FlashInfer prefill launch failed");
    }
  }
  return KS_SUCCESS;
}

static ks_status_t fi_flash_attn(void* out, void* softmax_lse, const void* q,
                                  const void* k, const void* v, int batch,
                                  int seqlen_q, int seqlen_k, int num_heads,
                                  int num_kv_heads, int head_dim, float scale,
                                  int causal, ks_dtype_t dtype, gpuStream_t s) {
  // FlashInfer requires fp16/bf16 (sizeof(DTypeQ) == 2). fp32 falls back to ks_flash_attn.
  if (dtype == KS_DTYPE_F32) {
    return KS_ERROR_UNSUPPORTED_DTYPE;  // caller will fall back
  }
  const auto mask = causal ? flashinfer::MaskMode::kCausal
                            : flashinfer::MaskMode::kNone;

  if (head_dim == 64) {
    if (mask == flashinfer::MaskMode::kCausal) {
      KS_DISPATCH_HALF_TYPES(dtype, "ks_flashinfer_attn", {
        return fi_prefill_launch<scalar_t, 64, flashinfer::MaskMode::kCausal>(
            out, static_cast<float*>(softmax_lse), q, k, v, batch, seqlen_q,
            seqlen_k, num_heads, num_kv_heads, scale, s);
      });
    } else {
      KS_DISPATCH_HALF_TYPES(dtype, "ks_flashinfer_attn", {
        return fi_prefill_launch<scalar_t, 64, flashinfer::MaskMode::kNone>(
            out, static_cast<float*>(softmax_lse), q, k, v, batch, seqlen_q,
            seqlen_k, num_heads, num_kv_heads, scale, s);
      });
    }
  } else if (head_dim == 128) {
    if (mask == flashinfer::MaskMode::kCausal) {
      KS_DISPATCH_HALF_TYPES(dtype, "ks_flashinfer_attn", {
        return fi_prefill_launch<scalar_t, 128, flashinfer::MaskMode::kCausal>(
            out, static_cast<float*>(softmax_lse), q, k, v, batch, seqlen_q,
            seqlen_k, num_heads, num_kv_heads, scale, s);
      });
    } else {
      KS_DISPATCH_HALF_TYPES(dtype, "ks_flashinfer_attn", {
        return fi_prefill_launch<scalar_t, 128, flashinfer::MaskMode::kNone>(
            out, static_cast<float*>(softmax_lse), q, k, v, batch, seqlen_q,
            seqlen_k, num_heads, num_kv_heads, scale, s);
      });
    }
  } else {
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "FlashInfer: head_dim must be 64 or 128");
  }
  return KS_SUCCESS;
}

}  // namespace attention
}  // namespace ks

#endif  // !__CUDA_ARCH__ || __CUDA_ARCH__ >= 750

using namespace ks;

extern "C" {

ks_status_t ks_flashinfer_attn(void* out, void* softmax_lse, const void* q,
                                const void* k, const void* v, int batch,
                                int seqlen_q, int seqlen_k, int num_heads,
                                int num_kv_heads, int head_dim,
                                float softmax_scale, int causal,
                                ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(q);
  KS_CHECK_PTR(k);
  KS_CHECK_PTR(v);
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_flashinfer_attn: bad shape");

  // FlashInfer requires sm_75+ for ldmatrix / mma.
  // On sm_70 (V100), fall back to the existing ks_flash_attn kernel.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
  int dev = 0;
  cudaGetDevice(&dev);
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  if (major * 10 + minor >= 75) {
    const float scale = attention::resolve_scale(softmax_scale, head_dim);
    auto s = to_stream(stream);
    ks_status_t st = attention::fi_flash_attn(
        out, softmax_lse, q, k, v, batch, seqlen_q, seqlen_k, num_heads,
        num_kv_heads, head_dim, scale, causal, dtype, s);
    if (st == KS_SUCCESS) return st;
    // fp32 or unsupported config: fall through to ks_flash_attn
  }
#endif

  // Fallback: use the existing kernel-set flash attention kernel.
  return ks_flash_attn(out, softmax_lse, q, k, v, batch, seqlen_q, seqlen_k,
                        num_heads, num_kv_heads, head_dim, softmax_scale,
                        causal, dtype, stream);
}

}  // extern "C"