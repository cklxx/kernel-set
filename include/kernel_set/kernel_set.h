/*
 * kernel-set — high-performance LLM inference & training kernels.
 *
 * Single umbrella include for the whole C ABI. Link against `kernel_set`
 * (libkernel_set.so / .dylib / kernel_set.dll). Every entry point returns a
 * ks_status_t; pass device pointers and a ks_stream_t (NULL = default stream).
 *
 *   #include "kernel_set/kernel_set.h"
 *
 * The ABI is pure C and free of CUDA headers, so it is safe to include from any
 * FFI host (Python ctypes, Go cgo, Rust bindgen, Node N-API/FFI).
 */
#ifndef KERNEL_SET_KERNEL_SET_H_
#define KERNEL_SET_KERNEL_SET_H_

#include "kernel_set/types.h"
#include "kernel_set/runtime.h"

#include "kernel_set/activation.h"
#include "kernel_set/attention.h"
#include "kernel_set/elementwise.h"
#include "kernel_set/embedding.h"
#include "kernel_set/gemm.h"
#include "kernel_set/loss.h"
#include "kernel_set/moe.h"
#include "kernel_set/norm.h"
#include "kernel_set/optimizer.h"
#include "kernel_set/quant.h"
#include "kernel_set/rope.h"
#include "kernel_set/sampling.h"
#include "kernel_set/ssm.h"

#endif /* KERNEL_SET_KERNEL_SET_H_ */
