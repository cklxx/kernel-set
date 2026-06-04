// kernel-set — MoE token<->expert permutation: build groups, gather, scatter.
//
// Three public ops implement the data-movement around the grouped expert GEMM,
// modeled on vLLM / SGLang fused-MoE (`moe_align_block_size` + permute) and
// Megatron's token permutation:
//
//   ks_moe_compute_permutation : counting-sort the (token,k) -> expert pairs
//       into per-expert contiguous groups. Produces `expert_offsets`
//       (CSR-style [num_experts+1] boundaries) and `sorted_token_ids`
//       (flattened pair ids grouped by expert, length num_tokens*top_k).
//   ks_moe_permute            : gather token rows into the sorted layout so each
//       expert sees a contiguous block of rows.
//   ks_moe_unpermute          : weighted scatter-add of expert outputs back to
//       per-token rows (sum over the token's top_k slots).
//
// The flattened pair id is `pair = token * top_k + k`; the source token row is
// therefore `pair / top_k` and the routing slot is `pair % top_k`.
#include "kernel_set/moe.h"
#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/dispatch.cuh"

namespace ks {
namespace moe {

constexpr int kPermBlock = 256;

// -------------------------------------------------------------------------
// compute_permutation: counting sort over experts.
// -------------------------------------------------------------------------
// Pass 1: histogram of pairs per expert (global atomics into `counts`).
KS_GLOBAL void perm_count_kernel(int32_t* __restrict__ counts,
                                 const int32_t* __restrict__ topk_indices,
                                 int64_t num_pairs, int num_experts) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_pairs) return;
  const int e = topk_indices[i];
  if (e >= 0 && e < num_experts) atomicAdd(&counts[e], 1);
}

// Pass 2: exclusive prefix sum of `counts` -> `expert_offsets[num_experts+1]`,
// and seed a write `cursor` (copy of the offsets) for the scatter. Single block;
// num_experts is small (<= a few hundred), so a serial scan by thread 0 is fine.
KS_GLOBAL void perm_scan_kernel(int32_t* __restrict__ expert_offsets,
                                int32_t* __restrict__ cursor,
                                const int32_t* __restrict__ counts,
                                int num_experts) {
  if (threadIdx.x != 0) return;
  int32_t running = 0;
  for (int e = 0; e < num_experts; ++e) {
    expert_offsets[e] = running;
    cursor[e] = running;
    running += counts[e];
  }
  expert_offsets[num_experts] = running;  // total = num routed pairs
}

// Pass 3: scatter each pair into its expert's region using an atomic cursor.
// Groups are correct; intra-group order follows atomic arrival (unspecified),
// matching the reference fused-MoE sort contract. Pairs whose expert id is out
// of range are dropped (defensive; valid gating never emits them).
KS_GLOBAL void perm_scatter_kernel(int32_t* __restrict__ sorted_token_ids,
                                   int32_t* __restrict__ cursor,
                                   const int32_t* __restrict__ topk_indices,
                                   int64_t num_pairs, int num_experts) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_pairs) return;
  const int e = topk_indices[i];
  if (e < 0 || e >= num_experts) return;
  const int pos = atomicAdd(&cursor[e], 1);
  sorted_token_ids[pos] = static_cast<int32_t>(i);
}

// -------------------------------------------------------------------------
// permute: gather token rows into sorted order. One block per output row.
// -------------------------------------------------------------------------
template <typename scalar_t>
KS_GLOBAL void permute_kernel(scalar_t* __restrict__ permuted,
                              const scalar_t* __restrict__ input,
                              const int32_t* __restrict__ sorted_token_ids,
                              int top_k, int64_t hidden) {
  const int64_t out_row = blockIdx.x;
  const int32_t pair = sorted_token_ids[out_row];
  const int64_t src_token = static_cast<int64_t>(pair) / top_k;

  const scalar_t* src = input + src_token * hidden;
  scalar_t* dst = permuted + out_row * hidden;
  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}

// -------------------------------------------------------------------------
// unpermute: build the inverse map then gather+accumulate per token.
// -------------------------------------------------------------------------
// inverse[pair] = permuted_row, i.e. where (token,slot) landed after the sort.
// Built by scattering from sorted_token_ids (which maps permuted_row -> pair).
KS_GLOBAL void unpermute_inverse_kernel(
    int32_t* __restrict__ inverse,
    const int32_t* __restrict__ sorted_token_ids, int64_t num_rows) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  const int32_t pair = sorted_token_ids[row];
  inverse[pair] = static_cast<int32_t>(row);
}

// One block per output token. Accumulates the weighted sum of the token's top_k
// expert outputs in fp32 registers (no atomics) and writes the row once.
template <typename scalar_t>
KS_GLOBAL void unpermute_gather_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ permuted,
    const int32_t* __restrict__ inverse,
    const float* __restrict__ routing_weights, int top_k, int64_t hidden) {
  const int64_t token = blockIdx.x;
  scalar_t* dst = out + token * hidden;

  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) {
      const int64_t pair = token * top_k + k;
      const int32_t prow = inverse[pair];
      const float w = routing_weights[pair];
      acc += to_float(permuted[static_cast<int64_t>(prow) * hidden + i]) * w;
    }
    dst[i] = from_float<scalar_t>(acc);
  }
}

}  // namespace moe
}  // namespace ks

using namespace ks;

extern "C" {

ks_status_t ks_moe_compute_permutation(int32_t* sorted_token_ids,
                                       int32_t* expert_offsets,
                                       const int32_t* topk_indices,
                                       int64_t num_tokens, int num_experts,
                                       int top_k, ks_stream_t stream) {
  KS_CHECK_PTR(sorted_token_ids);
  KS_CHECK_PTR(expert_offsets);
  KS_CHECK_PTR(topk_indices);
  if (num_tokens <= 0 || num_experts <= 0 || top_k <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT,
                    "ks_moe_compute_permutation: bad shape");

  const int64_t num_pairs = num_tokens * static_cast<int64_t>(top_k);
  auto s = to_stream(stream);

  // Internal scratch: per-expert counts (reused as a write cursor). num_experts
  // is small, so this allocation is negligible. Freed before returning.
  int32_t* scratch = nullptr;  // [2 * num_experts]: counts | cursor
  const size_t scratch_bytes =
      static_cast<size_t>(2) * num_experts * sizeof(int32_t);
  ks::gpuError_t me =
      ks::gpuMalloc(reinterpret_cast<void**>(&scratch), scratch_bytes);
  if (me != ks::gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_OUT_OF_MEMORY,
                    "ks_moe_compute_permutation: scratch alloc");
  int32_t* counts = scratch;
  int32_t* cursor = scratch + num_experts;

  ks::gpuError_t ze = ks::gpuMemsetAsync(
      counts, 0, static_cast<size_t>(num_experts) * sizeof(int32_t), s);
  if (ze != ks::gpuSuccess) {
    ks::gpuFree(scratch);
    KS_RETURN_ERROR(KS_ERROR_CUDA, "ks_moe_compute_permutation: memset");
  }

  const dim3 block(moe::kPermBlock);
  const dim3 grid(static_cast<unsigned>(
      (num_pairs + moe::kPermBlock - 1) / moe::kPermBlock));

  moe::perm_count_kernel<<<grid, block, 0, s>>>(counts, topk_indices, num_pairs,
                                                num_experts);
  moe::perm_scan_kernel<<<1, moe::kPermBlock, 0, s>>>(expert_offsets, cursor,
                                                      counts, num_experts);
  moe::perm_scatter_kernel<<<grid, block, 0, s>>>(
      sorted_token_ids, cursor, topk_indices, num_pairs, num_experts);

  // The scatter is asynchronous; cursor must outlive it. Synchronize on this
  // stream before freeing the scratch so the free does not race the kernel.
  ks::gpuError_t se = ks::gpuStreamSynchronize(s);
  ks::gpuFree(scratch);
  if (se != ks::gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_CUDA, "ks_moe_compute_permutation: sync");

  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_moe_permute(void* permuted, const void* input,
                           const int32_t* sorted_token_ids, int64_t num_tokens,
                           int top_k, int64_t hidden, ks_dtype_t dtype,
                           ks_stream_t stream) {
  KS_CHECK_PTR(permuted);
  KS_CHECK_PTR(input);
  KS_CHECK_PTR(sorted_token_ids);
  if (num_tokens <= 0 || top_k <= 0 || hidden <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_moe_permute: bad shape");

  const int64_t out_rows = num_tokens * static_cast<int64_t>(top_k);
  const dim3 grid(static_cast<unsigned>(out_rows));
  const dim3 block(moe::kPermBlock);
  auto s = to_stream(stream);

  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_moe_permute", {
    moe::permute_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(permuted),
        static_cast<const scalar_t*>(input), sorted_token_ids, top_k, hidden);
  });
  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

ks_status_t ks_moe_unpermute(void* out, const void* permuted,
                             const int32_t* sorted_token_ids,
                             const float* routing_weights, int64_t num_tokens,
                             int top_k, int64_t hidden, ks_dtype_t dtype,
                             ks_stream_t stream) {
  KS_CHECK_PTR(out);
  KS_CHECK_PTR(permuted);
  KS_CHECK_PTR(sorted_token_ids);
  KS_CHECK_PTR(routing_weights);
  if (num_tokens <= 0 || top_k <= 0 || hidden <= 0)
    KS_RETURN_ERROR(KS_ERROR_INVALID_ARGUMENT, "ks_moe_unpermute: bad shape");
  // Validate the dtype up front so the unsupported-dtype path of
  // KS_DISPATCH_FLOATING_TYPES (which returns early) cannot leak the `inverse`
  // scratch allocated below.
  if (dtype != KS_DTYPE_F32 && dtype != KS_DTYPE_F16 && dtype != KS_DTYPE_BF16)
    KS_RETURN_ERROR(KS_ERROR_UNSUPPORTED_DTYPE,
                    "ks_moe_unpermute: unsupported dtype");

  const int64_t num_rows = num_tokens * static_cast<int64_t>(top_k);
  auto s = to_stream(stream);

  // Scratch holds the inverse permutation (pair -> permuted_row).
  int32_t* inverse = nullptr;
  ks::gpuError_t me = ks::gpuMalloc(
      reinterpret_cast<void**>(&inverse),
      static_cast<size_t>(num_rows) * sizeof(int32_t));
  if (me != ks::gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_OUT_OF_MEMORY, "ks_moe_unpermute: scratch alloc");

  const dim3 inv_block(moe::kPermBlock);
  const dim3 inv_grid(static_cast<unsigned>(
      (num_rows + moe::kPermBlock - 1) / moe::kPermBlock));
  moe::unpermute_inverse_kernel<<<inv_grid, inv_block, 0, s>>>(
      inverse, sorted_token_ids, num_rows);

  const dim3 grid(static_cast<unsigned>(num_tokens));
  const dim3 block(moe::kPermBlock);
  KS_DISPATCH_FLOATING_TYPES(dtype, "ks_moe_unpermute", {
    moe::unpermute_gather_kernel<scalar_t><<<grid, block, 0, s>>>(
        static_cast<scalar_t*>(out),
        static_cast<const scalar_t*>(permuted), inverse, routing_weights, top_k,
        hidden);
  });

  ks::gpuError_t se = ks::gpuStreamSynchronize(s);
  ks::gpuFree(inverse);
  if (se != ks::gpuSuccess)
    KS_RETURN_ERROR(KS_ERROR_CUDA, "ks_moe_unpermute: sync");

  KS_CHECK_LAUNCH();
  return KS_SUCCESS;
}

}  // extern "C"
