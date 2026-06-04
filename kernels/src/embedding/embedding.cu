// kernel-set — token embedding lookup (gather) and backward (scatter-add).
//
// Two entry points, mirroring the house style established by norm/rms_norm.cu:
//   * ks_embedding_lookup    out[i,:]            = table[idx[i],:]   (gather)
//   * ks_embedding_backward  grad_table[idx[i],:] += grad_out[i,:]   (scatter)
//
// Layout: one CUDA block per token, blockDim.x threads stride the embedding
// dimension cooperatively. Each token's row is a contiguous span of `embed_dim`
// scalars, so a block-stride copy is fully coalesced. When the table/out/grad
// pointers and embed_dim are 128-bit aligned we widen the copy to ks::Vec for
// peak bandwidth (one 16-byte transaction per thread per step), falling back to
// a scalar tail otherwise.
//
// Indices are int32 or int64 selected by the `indices_i64` flag (matching the
// ABI). A negative or out-of-range padding index is *not* special-cased here:
// PyTorch's nn.Embedding padding is handled by the caller zeroing the relevant
// table row, so any valid index in [0, vocab) is a plain gather. We do clamp
// nothing and trust the caller's bounds (same convention as torch's embedding
// CUDA kernel, which is UB on OOB ids); shapes are validated on the host side.
//
// Backward accumulates into an fp32 grad_table via atomicAdd. Because many
// tokens can map to the same vocab row (e.g. repeated/padding ids in a batch),
// the scatter must be atomic; we convert the (possibly half/bf16) grad_out to
// fp32 first so accumulation never loses precision. This matches the reference
// "index_add_"/embedding_backward path used by PyTorch and FasterTransformer.
//
// Further tuning (documented, not implemented): for very large num_tokens with
// heavy id duplication, a sort-by-id + segmented-reduce backward (as in
// PyTorch's embedding_dense_backward) avoids atomic contention; and cp.async
// double-buffering could hide table latency on Hopper. The atomic scatter here
// is correct and simple, which the contract prioritizes.
#include "kernel_set/embedding.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"
#include "common/vec.cuh"

namespace ks {
namespace embedding {

constexpr int kBlock = 256;

// ---------------------------------------------------------------------------
// Index access: unify int32 / int64 indices behind one device accessor.
// ---------------------------------------------------------------------------
KS_DI int64_t load_index(const void* indices, int indices_i64, int64_t i) {
  if (indices_i64) {
    return static_cast<const int64_t*>(indices)[i];
  }
  return static_cast<int64_t>(static_cast<const int32_t*>(indices)[i]);
}

// ---------------------------------------------------------------------------
// Forward: gather.  out[token, :] = table[idx[token], :]
//
// VEC > 1 selects the vectorized path; the host only launches it when both the
// table and out base pointers plus embed_dim satisfy 128-bit alignment, so the
// reinterpret_cast loads/stores are always legal here. `vec_cols` is embed_dim
// expressed in units of Vec<scalar_t, VEC>.
// ---------------------------------------------------------------------------
template <typename scalar_t, int VEC>
KS_GLOBAL void embedding_lookup_kernel(scalar_t* __restrict__ out,
                                       const scalar_t* __restrict__ table,
                                       const void* indices, int indices_i64,
                                       int64_t num_tokens, int64_t embed_dim) {
  // Grid-stride over tokens so a bounded grid handles any num_tokens.
  for (int64_t token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    const int64_t row = load_index(indices, indices_i64, token);
    const scalar_t* src = table + row * embed_dim;
    scalar_t* dst = out + token * embed_dim;

    if (VEC > 1) {
      using V = Vec<scalar_t, VEC>;
      const int64_t vec_cols = embed_dim / VEC;
      const V* src_v = reinterpret_cast<const V*>(src);
      V* dst_v = reinterpret_cast<V*>(dst);
      for (int64_t i = threadIdx.x; i < vec_cols; i += blockDim.x) {
        dst_v[i] = src_v[i];
      }
    } else {
      for (int64_t i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        dst[i] = src[i];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Backward: scatter-add.  grad_table[idx[token], :] += grad_out[token, :]
//
// grad_table is always fp32 (per the ABI); grad_out is the storage dtype. We
// convert each element to fp32 and atomicAdd into the destination row. atomicAdd
// is element-wise (there is no 128-bit float atomic), so the grad_out read is
// the only part we vectorize — and even that buys little, so we keep the simple
// scalar grid-stride which is already coalesced for both load and atomic store.
// ---------------------------------------------------------------------------
template <typename scalar_t>
KS_GLOBAL void embedding_backward_kernel(float* __restrict__ grad_table,
                                         const scalar_t* __restrict__ grad_out,
                                         const void* indices, int indices_i64,
                                         int64_t num_tokens, int64_t embed_dim) {
  for (int64_t token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    const int64_t row = load_index(indices, indices_i64, token);
    float* dst = grad_table + row * embed_dim;
    const scalar_t* src = grad_out + token * embed_dim;
    for (int64_t i = threadIdx.x; i < embed_dim; i += blockDim.x) {
      atomicAdd(&dst[i], to_float(src[i]));
    }
  }
}

// Cap the grid so a single kernel handles arbitrarily many tokens via the
// grid-stride loop while keeping enough blocks to saturate large GPUs.
KS_HDI unsigned grid_for(int64_t num_tokens) {
  constexpr int64_t kMaxBlocks = 65535;
  const int64_t n = num_tokens < kMaxBlocks ? num_tokens : kMaxBlocks;
  return static_cast<unsigned>(n < 1 ? 1 : n);
}

}  // namespace embedding
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_embedding_lookup(void* out, const void* table,
                                const void* indices, int indices_i64,
                                int64_t num_tokens, int64_t embed_dim,
                                ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(table);
  KS_CHECK_PTR(indices);
  if (num_tokens < 0 || embed_dim <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_embedding_lookup: bad shape");
  if (num_tokens == 0) return KS_SUCCESS;  // nothing to gather

  const dim3 grid(embedding::grid_for(num_tokens));
  const dim3 block(embedding::kBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_embedding_lookup", {
    constexpr int VEC = VecWidth<scalar_t>::value;
    // Use the 128-bit copy only when out, table, and embed_dim are all aligned
    // to a width-VEC vector; otherwise the scalar path stays correct.
    const bool vec_ok = is_aligned<scalar_t, VEC>(out, embed_dim) &&
                        is_aligned<scalar_t, VEC>(table, embed_dim);
    if (vec_ok) {
      embedding::embedding_lookup_kernel<scalar_t, VEC><<<grid, block, 0, s>>>(
          static_cast<scalar_t*>(out), static_cast<const scalar_t*>(table),
          indices, indices_i64, num_tokens, embed_dim);
    } else {
      embedding::embedding_lookup_kernel<scalar_t, 1><<<grid, block, 0, s>>>(
          static_cast<scalar_t*>(out), static_cast<const scalar_t*>(table),
          indices, indices_i64, num_tokens, embed_dim);
    }
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_embedding_backward(void* grad_table_fp32, const void* grad_out,
                                  const void* indices, int indices_i64,
                                  int64_t num_tokens, int64_t embed_dim,
                                  ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(grad_table_fp32);
  KS_CHECK_PTR(grad_out);
  KS_CHECK_PTR(indices);
  if (num_tokens < 0 || embed_dim <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_embedding_backward: bad shape");
  if (num_tokens == 0) return KS_SUCCESS;  // no gradients to scatter

  const dim3 grid(embedding::grid_for(num_tokens));
  const dim3 block(embedding::kBlock);
  auto s = to_stream(stream);

  // Caller is responsible for zero-initializing grad_table_fp32 before the
  // scatter-add (same convention as the norm backward grad_weight buffers).
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_embedding_backward", {
    embedding::embedding_backward_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<float*>(grad_table_fp32),
        static_cast<const scalar_t*>(grad_out), indices, indices_i64,
        num_tokens, embed_dim);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
