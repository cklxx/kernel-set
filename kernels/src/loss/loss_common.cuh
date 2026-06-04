// kernel-set — shared helpers for the loss category (target load, CE math).
#pragma once

#include "common/platform.cuh"
#include "common/dtype.cuh"
#include "common/reduce.cuh"

namespace ks {
namespace loss {

// Fetch the (possibly int32 or int64) target id for a row.
KS_DI int64_t load_target_id(const void* targets, int targets_i64,
                             int64_t row) {
  if (targets_i64)
    return reinterpret_cast<const int64_t*>(targets)[row];
  return static_cast<int64_t>(reinterpret_cast<const int32_t*>(targets)[row]);
}

}  // namespace loss
}  // namespace ks
