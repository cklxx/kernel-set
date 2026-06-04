// kernel-set — GPU platform abstraction layer.
//
// All device code includes this header instead of <cuda_runtime.h> directly.
// CUDA is the active backend today; HIP is wired in behind KS_PLATFORM_HIP so
// that porting to ROCm is a build-flag change, not a source rewrite. Host-side
// runtime calls go through the ks:: aliases below; kernels use the KS_* macros.
#pragma once

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------
#if defined(__HIPCC__) || defined(KS_PLATFORM_HIP)
#  define KS_PLATFORM_HIP 1
#  include <hip/hip_runtime.h>
#  include <hip/hip_fp16.h>
#  include <hip/hip_bf16.h>
#else
#  ifndef KS_PLATFORM_CUDA
#    define KS_PLATFORM_CUDA 1
#  endif
#  include <cuda_runtime.h>
#  include <cuda_fp16.h>
#  include <cuda_bf16.h>
#  if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#    define KS_HAS_FP8 1
#    include <cuda_fp8.h>
#  elif !defined(__CUDA_ARCH__) && defined(__CUDACC__)
// Host pass of nvcc: expose the fp8 types for size/dispatch logic.
#    include <cuda_fp8.h>
#  endif
#endif

// ---------------------------------------------------------------------------
// Function qualifiers
// ---------------------------------------------------------------------------
#define KS_DEVICE __device__
#define KS_HOST __host__
#define KS_HOST_DEVICE __host__ __device__
#define KS_GLOBAL __global__
#define KS_FORCEINLINE __forceinline__
#define KS_DI KS_DEVICE KS_FORCEINLINE
#define KS_HDI KS_HOST_DEVICE KS_FORCEINLINE

#ifndef KS_WARP_SIZE
#  if defined(KS_PLATFORM_HIP)
#    define KS_WARP_SIZE warpSize  // 32 (RDNA) or 64 (CDNA), resolved at runtime
#  else
#    define KS_WARP_SIZE 32
#  endif
#endif
#define KS_FULL_MASK 0xffffffffu

// ---------------------------------------------------------------------------
// Host runtime aliases (cuda* <-> hip*)
// ---------------------------------------------------------------------------
namespace ks {

#if defined(KS_PLATFORM_HIP)
using gpuStream_t = hipStream_t;
using gpuError_t = hipError_t;
using gpuDeviceProp_t = hipDeviceProp_t;
constexpr gpuError_t gpuSuccess = hipSuccess;

inline gpuError_t gpuGetLastError() { return hipGetLastError(); }
inline const char* gpuGetErrorString(gpuError_t e) { return hipGetErrorString(e); }
inline gpuError_t gpuGetDeviceCount(int* c) { return hipGetDeviceCount(c); }
inline gpuError_t gpuSetDevice(int d) { return hipSetDevice(d); }
inline gpuError_t gpuGetDevice(int* d) { return hipGetDevice(d); }
inline gpuError_t gpuGetDeviceProperties(gpuDeviceProp_t* p, int d) {
  return hipGetDeviceProperties(p, d);
}
inline gpuError_t gpuStreamCreate(gpuStream_t* s) { return hipStreamCreate(s); }
inline gpuError_t gpuStreamDestroy(gpuStream_t s) { return hipStreamDestroy(s); }
inline gpuError_t gpuStreamSynchronize(gpuStream_t s) {
  return hipStreamSynchronize(s);
}
inline gpuError_t gpuMalloc(void** p, size_t n) { return hipMalloc(p, n); }
inline gpuError_t gpuFree(void* p) { return hipFree(p); }
inline gpuError_t gpuMemsetAsync(void* p, int v, size_t n, gpuStream_t s) {
  return hipMemsetAsync(p, v, n, s);
}
inline gpuError_t gpuMemcpyAsync(void* d, const void* s, size_t n,
                                 hipMemcpyKind k, gpuStream_t st) {
  return hipMemcpyAsync(d, s, n, k, st);
}
constexpr hipMemcpyKind gpuMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr hipMemcpyKind gpuMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr hipMemcpyKind gpuMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
#else
using gpuStream_t = cudaStream_t;
using gpuError_t = cudaError_t;
using gpuDeviceProp_t = cudaDeviceProp;
constexpr gpuError_t gpuSuccess = cudaSuccess;

inline gpuError_t gpuGetLastError() { return cudaGetLastError(); }
inline const char* gpuGetErrorString(gpuError_t e) { return cudaGetErrorString(e); }
inline gpuError_t gpuGetDeviceCount(int* c) { return cudaGetDeviceCount(c); }
inline gpuError_t gpuSetDevice(int d) { return cudaSetDevice(d); }
inline gpuError_t gpuGetDevice(int* d) { return cudaGetDevice(d); }
inline gpuError_t gpuGetDeviceProperties(gpuDeviceProp_t* p, int d) {
  return cudaGetDeviceProperties(p, d);
}
inline gpuError_t gpuStreamCreate(gpuStream_t* s) { return cudaStreamCreate(s); }
inline gpuError_t gpuStreamDestroy(gpuStream_t s) { return cudaStreamDestroy(s); }
inline gpuError_t gpuStreamSynchronize(gpuStream_t s) {
  return cudaStreamSynchronize(s);
}
inline gpuError_t gpuMalloc(void** p, size_t n) { return cudaMalloc(p, n); }
inline gpuError_t gpuFree(void* p) { return cudaFree(p); }
inline gpuError_t gpuMemsetAsync(void* p, int v, size_t n, gpuStream_t s) {
  return cudaMemsetAsync(p, v, n, s);
}
inline gpuError_t gpuMemcpyAsync(void* d, const void* s, size_t n,
                                 cudaMemcpyKind k, gpuStream_t st) {
  return cudaMemcpyAsync(d, s, n, k, st);
}
constexpr cudaMemcpyKind gpuMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr cudaMemcpyKind gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
constexpr cudaMemcpyKind gpuMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
#endif

// Cast the ABI's opaque ks_stream_t (void*) to the backend stream type.
KS_HDI gpuStream_t to_stream(void* s) { return static_cast<gpuStream_t>(s); }

// ---------------------------------------------------------------------------
// Warp primitives (mask-correct on CUDA, plain on HIP)
// ---------------------------------------------------------------------------
template <typename T>
KS_DI T shfl_xor(T v, int laneMask, int width = KS_WARP_SIZE) {
#if defined(KS_PLATFORM_HIP)
  return __shfl_xor(v, laneMask, width);
#else
  return __shfl_xor_sync(KS_FULL_MASK, v, laneMask, width);
#endif
}

template <typename T>
KS_DI T shfl_down(T v, unsigned delta, int width = KS_WARP_SIZE) {
#if defined(KS_PLATFORM_HIP)
  return __shfl_down(v, delta, width);
#else
  return __shfl_down_sync(KS_FULL_MASK, v, delta, width);
#endif
}

template <typename T>
KS_DI T shfl(T v, int srcLane, int width = KS_WARP_SIZE) {
#if defined(KS_PLATFORM_HIP)
  return __shfl(v, srcLane, width);
#else
  return __shfl_sync(KS_FULL_MASK, v, srcLane, width);
#endif
}

}  // namespace ks
