// kernel-set — FlashAttention backward (training).
//
// Recompute-based, atomic-free backward that consumes the forward `out` and
// `softmax_lse` (lse_i = m_i + log l_i). With the row probabilities recovered as
//   P_ij = exp(scale * q_i . k_j - lse_i)        (masked entries -> 0)
// the gradients are the standard FlashAttention backward identities:
//   D_i  = sum_d O_id * dO_id                     (per query row)
//   dP_ij = dO_i . v_j                            (head_dim dot)
//   dS_ij = P_ij * (dP_ij - D_i)
//   dQ_i = scale * sum_j dS_ij * k_j
//   dK_j = scale * sum_i dS_ij * q_i
//   dV_j = sum_i P_ij * dO_i
// Everything accumulates in fp32. To avoid atomics we partition work so that
// each output row is owned by exactly one block:
//   * delta kernel : one block per (batch, head, q_row)   -> D_i
//   * dQ kernel    : one block per (batch, head, q_row)   -> dQ_i
//   * dKV kernel   : one block per (batch, kv_head, k_row) -> dK_j, dV_j,
//                    summing over every query head that shares this kv head
//                    (GQA/MQA) and every query row that attends this key.
// This is the "recompute per tile/row using softmax_lse" path the task allows;
// it is numerically exact rather than a fused single-pass backward. The fused
// FA2 backward (dQ via atomics or a second loop over key tiles) is the
// documented perf follow-up.
//
// blockDim.x == head_dim throughout; thread d owns dim d of its output row and
// participates in the head_dim-wide dot-product block reductions.
#include "kernel_set/attention.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/reduce.cuh"
#include "attention_common.cuh"

namespace ks {
namespace attention {

// Dense (non-varlen) indexing helpers. Tensors are
//   q/out/grad_out : [batch, seqlen_q, num_heads, head_dim]
//   k/v            : [batch, seqlen_k, num_kv_heads, head_dim]
//   lse / delta    : [num_heads, batch*seqlen_q]  (row-major; matches forward)

// delta[i] = sum_d out_id * grad_out_id, for one (batch, head, q_row) per block.
template <typename scalar_t>
KS_GLOBAL void bwd_delta_kernel(float* __restrict__ delta,
                                const scalar_t* __restrict__ out,
                                const scalar_t* __restrict__ grad_out,
                                int seqlen_q, int num_heads, int head_dim) {
  const int d = threadIdx.x;
  const int qi = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;

  const int64_t row = (static_cast<int64_t>(batch) * seqlen_q + qi);
  const int64_t base = (row * num_heads + head) * head_dim;
  const float prod = to_float(out[base + d]) * to_float(grad_out[base + d]);

  extern __shared__ float red[];
  const float di = block_reduce_sum(prod, red);
  if (d == 0) {
    // delta layout matches lse: [num_heads, batch*seqlen_q] row-major.
    const int64_t total_q = static_cast<int64_t>(gridDim.z) * seqlen_q;
    delta[static_cast<int64_t>(head) * total_q + row] = di;
  }
}

// dQ for one (batch, head, q_row) per block.
template <typename scalar_t>
KS_GLOBAL void bwd_dq_kernel(scalar_t* __restrict__ grad_q,
                             const scalar_t* __restrict__ q,
                             const scalar_t* __restrict__ k,
                             const scalar_t* __restrict__ v,
                             const scalar_t* __restrict__ grad_out,
                             const float* __restrict__ lse,
                             const float* __restrict__ delta, int seqlen_q,
                             int seqlen_k, int num_heads, int num_kv_heads,
                             int head_dim, float scale, int causal) {
  const int d = threadIdx.x;
  const int qi = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int kv_head = head / gqa_group_size(num_heads, num_kv_heads);

  const int64_t total_q = static_cast<int64_t>(gridDim.z) * seqlen_q;
  const int64_t q_row = static_cast<int64_t>(batch) * seqlen_q + qi;
  const scalar_t* q_ptr =
      q + (q_row * num_heads + head) * head_dim;
  const scalar_t* do_ptr =
      grad_out + (q_row * num_heads + head) * head_dim;

  const float q_reg = to_float(q_ptr[d]);
  const float do_reg = to_float(do_ptr[d]);
  const float lse_i = lse[static_cast<int64_t>(head) * total_q + q_row];
  const float d_i = delta[static_cast<int64_t>(head) * total_q + q_row];

  extern __shared__ float red[];
  float dq_acc = 0.f;

  const int causal_offset = seqlen_k - seqlen_q;
  const int kv_max = causal ? (causal_offset + qi + 1) : seqlen_k;

  for (int j = 0; j < kv_max; ++j) {
    const int64_t kv_row = static_cast<int64_t>(batch) * seqlen_k + j;
    const scalar_t* k_ptr = k + (kv_row * num_kv_heads + kv_head) * head_dim;
    const scalar_t* v_ptr = v + (kv_row * num_kv_heads + kv_head) * head_dim;
    const float k_reg = to_float(k_ptr[d]);
    const float v_reg = to_float(v_ptr[d]);

    const float s = block_reduce_sum(q_reg * k_reg, red) * scale;
    __syncthreads();
    const float dp = block_reduce_sum(do_reg * v_reg, red);
    __syncthreads();

    const float p = __expf(s - lse_i);   // masked rows: lse handles range; j<=kv_max
    const float ds = p * (dp - d_i);
    dq_acc += scale * ds * k_reg;
  }

  scalar_t* dq_ptr = grad_q + (q_row * num_heads + head) * head_dim;
  dq_ptr[d] = from_float<scalar_t>(dq_acc);
}

// dK and dV for one (batch, kv_head, k_row) per block; sums over the GQA group
// of query heads and every query row that attends this key.
template <typename scalar_t>
KS_GLOBAL void bwd_dkv_kernel(scalar_t* __restrict__ grad_k,
                              scalar_t* __restrict__ grad_v,
                              const scalar_t* __restrict__ q,
                              const scalar_t* __restrict__ k,
                              const scalar_t* __restrict__ v,
                              const scalar_t* __restrict__ grad_out,
                              const float* __restrict__ lse,
                              const float* __restrict__ delta, int seqlen_q,
                              int seqlen_k, int num_heads, int num_kv_heads,
                              int head_dim, float scale, int causal) {
  const int d = threadIdx.x;
  const int kj = blockIdx.x;        // key row within this sequence
  const int kv_head = blockIdx.y;   // kv head
  const int batch = blockIdx.z;
  const int group = gqa_group_size(num_heads, num_kv_heads);

  const int64_t total_q = static_cast<int64_t>(gridDim.z) * seqlen_q;
  const int64_t kv_row = static_cast<int64_t>(batch) * seqlen_k + kj;
  const scalar_t* k_ptr = k + (kv_row * num_kv_heads + kv_head) * head_dim;
  const scalar_t* v_ptr = v + (kv_row * num_kv_heads + kv_head) * head_dim;
  const float k_reg = to_float(k_ptr[d]);
  const float v_reg = to_float(v_ptr[d]);

  extern __shared__ float red[];
  float dk_acc = 0.f;
  float dv_acc = 0.f;

  const int causal_offset = seqlen_k - seqlen_q;

  // Iterate the query heads sharing this kv head, then the query rows.
  for (int g = 0; g < group; ++g) {
    const int head = kv_head * group + g;
    for (int qi = 0; qi < seqlen_q; ++qi) {
      // Causal: key kj is visible to query qi iff kj <= causal_offset + qi.
      if (causal && kj > causal_offset + qi) continue;

      const int64_t q_row = static_cast<int64_t>(batch) * seqlen_q + qi;
      const scalar_t* q_ptr = q + (q_row * num_heads + head) * head_dim;
      const scalar_t* do_ptr = grad_out + (q_row * num_heads + head) * head_dim;
      const float q_reg = to_float(q_ptr[d]);
      const float do_reg = to_float(do_ptr[d]);
      const float lse_i = lse[static_cast<int64_t>(head) * total_q + q_row];
      const float d_i = delta[static_cast<int64_t>(head) * total_q + q_row];

      const float s = block_reduce_sum(q_reg * k_reg, red) * scale;
      __syncthreads();
      const float dp = block_reduce_sum(do_reg * v_reg, red);
      __syncthreads();

      const float p = __expf(s - lse_i);
      const float ds = p * (dp - d_i);
      dv_acc += p * do_reg;
      dk_acc += scale * ds * q_reg;
    }
  }

  scalar_t* dk_ptr = grad_k + (kv_row * num_kv_heads + kv_head) * head_dim;
  scalar_t* dv_ptr = grad_v + (kv_row * num_kv_heads + kv_head) * head_dim;
  dk_ptr[d] = from_float<scalar_t>(dk_acc);
  dv_ptr[d] = from_float<scalar_t>(dv_acc);
}

}  // namespace attention
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_flash_attn_backward(
    void* grad_q, void* grad_k, void* grad_v, const void* grad_out,
    const void* q, const void* k, const void* v, const void* out,
    const void* softmax_lse, int batch, int seqlen_q, int seqlen_k,
    int num_heads, int num_kv_heads, int head_dim, float softmax_scale,
    int causal, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(grad_q);
  KS_CHECK_PTR(grad_k);
  KS_CHECK_PTR(grad_v);
  KS_CHECK_PTR(grad_out);
  KS_CHECK_PTR(q);
  KS_CHECK_PTR(k);
  KS_CHECK_PTR(v);
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(softmax_lse);
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || num_heads <= 0 ||
      num_kv_heads <= 0 || head_dim <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_flash_attn_backward: shape");
  if (num_kv_heads > num_heads || (num_heads % num_kv_heads) != 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_flash_attn_backward: num_heads % num_kv_heads != 0");
  if (head_dim > attention::kMaxHeadDim)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_flash_attn_backward: head_dim too large");
  if ((head_dim & 31) != 0)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_SHAPE,
                    "ks_flash_attn_backward: head_dim must be a multiple of 32");
  if (dtype != KS_DTYPE_F32 && dtype != KS_DTYPE_F16 && dtype != KS_DTYPE_BF16)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                    "ks_flash_attn_backward: needs f32/f16/bf16");

  const float scale = attention::resolve_scale(softmax_scale, head_dim);
  auto s = to_stream(stream);

  // delta[i] = sum_d out * grad_out, stored fp32 [num_heads, batch*seqlen_q].
  float* delta = nullptr;
  const size_t delta_n =
      static_cast<size_t>(num_heads) * batch * seqlen_q * sizeof(float);
  gpuError_t e = gpuMalloc(reinterpret_cast<void**>(&delta), delta_n);
  if (e != gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_OUT_OF_MEMORY, "ks_flash_attn_backward: delta");

  const dim3 block(static_cast<unsigned>(head_dim));
  const size_t red_smem = sizeof(float) * static_cast<size_t>(head_dim);

  const dim3 grid_q(static_cast<unsigned>(seqlen_q),
                    static_cast<unsigned>(num_heads),
                    static_cast<unsigned>(batch));
  const dim3 grid_kv(static_cast<unsigned>(seqlen_k),
                     static_cast<unsigned>(num_kv_heads),
                     static_cast<unsigned>(batch));

  const float* lse_f = static_cast<const float*>(softmax_lse);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_flash_attn_backward", {
    attention::bwd_delta_kernel<scalar_t><<<grid_q, block, red_smem, s>>>(
        delta, static_cast<const scalar_t*>(out),
        static_cast<const scalar_t*>(grad_out), seqlen_q, num_heads, head_dim);

    attention::bwd_dq_kernel<scalar_t><<<grid_q, block, red_smem, s>>>(
        static_cast<scalar_t*>(grad_q), static_cast<const scalar_t*>(q),
        static_cast<const scalar_t*>(k), static_cast<const scalar_t*>(v),
        static_cast<const scalar_t*>(grad_out), lse_f, delta, seqlen_q,
        seqlen_k, num_heads, num_kv_heads, head_dim, scale, causal);

    attention::bwd_dkv_kernel<scalar_t><<<grid_kv, block, red_smem, s>>>(
        static_cast<scalar_t*>(grad_k), static_cast<scalar_t*>(grad_v),
        static_cast<const scalar_t*>(q), static_cast<const scalar_t*>(k),
        static_cast<const scalar_t*>(v),
        static_cast<const scalar_t*>(grad_out), lse_f, delta, seqlen_q,
        seqlen_k, num_heads, num_kv_heads, head_dim, scale, causal);
  });

  e = gpuStreamSynchronize(s);  // keep `delta` alive until all 3 kernels finish
  gpuFree(delta);
  if (e != gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_CUDA, "ks_flash_attn_backward: launch/sync");
  return KS_SUCCESS;
}

}  // extern "C"
