// kernel-set — write new K/V into a paged KV cache (vLLM reshape_and_cache).
//
// During prefill and decode-append the freshly computed K/V for `num_tokens`
// positions must be scattered into the paged cache at the slots named by
// `slot_mapping`. A flat slot decomposes as
//   block_id = slot / block_size,  block_offset = slot % block_size
// and the destination cache has layout
//   [num_blocks, num_kv_heads, block_size, head_dim].
// A slot of -1 marks a padding token that should be skipped (matches vLLM).
//
// Parallelization: grid.x == num_tokens, grid.y == num_kv_heads; threads stride
// over head_dim. Pure copy with dtype-preserving conversion through fp32 so the
// path also works if storage dtypes ever differ.
#include "kernel_set/attention.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace attention {

// key,value : [num_tokens, num_kv_heads, head_dim]
// k_cache,v_cache : [num_blocks, num_kv_heads, block_size, head_dim]
// slot_mapping : [num_tokens]
template <typename scalar_t>
KS_GLOBAL void reshape_and_cache_kernel(
    scalar_t* __restrict__ k_cache, scalar_t* __restrict__ v_cache,
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    const int32_t* __restrict__ slot_mapping, int num_kv_heads, int head_dim,
    int block_size) {
  const int token = blockIdx.x;
  const int kv_head = blockIdx.y;

  const int32_t slot = slot_mapping[token];
  if (slot < 0) return;  // padding token: skip

  const int block_id = slot / block_size;
  const int block_off = slot % block_size;

  const int64_t src_base =
      (static_cast<int64_t>(token) * num_kv_heads + kv_head) * head_dim;
  // Destination index inside [num_blocks, num_kv_heads, block_size, head_dim].
  const int64_t dst_base =
      ((static_cast<int64_t>(block_id) * num_kv_heads + kv_head) * block_size +
       block_off) *
      head_dim;

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    k_cache[dst_base + d] =
        from_float<scalar_t>(to_float(key[src_base + d]));
    v_cache[dst_base + d] =
        from_float<scalar_t>(to_float(value[src_base + d]));
  }
}

}  // namespace attention
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_reshape_and_cache(
    void* k_cache, void* v_cache, const void* key, const void* value,
    const int32_t* slot_mapping, int num_tokens, int num_kv_heads,
    int head_dim, int block_size, ks_dtype_t dtype, ks_stream_t stream) {
  KS_CHECK_PTR(k_cache);
  KS_CHECK_PTR(v_cache);
  KS_CHECK_PTR(key);
  KS_CHECK_PTR(value);
  KS_CHECK_PTR(slot_mapping);
  if (num_tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0 || block_size <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_reshape_and_cache: shape");

  auto s = to_stream(stream);
  const dim3 grid(static_cast<unsigned>(num_tokens),
                  static_cast<unsigned>(num_kv_heads));
  const int threads = head_dim < 256 ? head_dim : 256;
  const dim3 block(static_cast<unsigned>(threads));

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_reshape_and_cache", {
    attention::reshape_and_cache_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(k_cache), static_cast<scalar_t*>(v_cache),
        static_cast<const scalar_t*>(key), static_cast<const scalar_t*>(value),
        slot_mapping, num_kv_heads, head_dim, block_size);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
