/***************************************************************************************************
 * cuBLASLt INT8 GEMM Benchmark
 *
 * D = A * B (no alpha/beta scaling, no C input)
 * A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
 * Input: INT8, Accumulator: INT32, Output: INT32
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include "../../common.cuh"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLASLt error in " << __FILE__ << " line " << __LINE__ << ": " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

static constexpr int warmup_iters = 500;
static constexpr int profiling_iters = 100;

///////////////////////////////////////////////////////////////////////////////////////////////////
// cuBLASLt GEMM: D = A * B
// A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
// Input: INT8, Accumulator: INT32, Output: INT32
///////////////////////////////////////////////////////////////////////////////////////////////////

struct CublasLtGemm {
  cublasLtHandle_t handle;
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t layoutA, layoutB, layoutD;
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulHeuristicResult_t heuristic;
  void* workspace;
  size_t workspaceSize;

  void init(int M, int N, int K) {
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // INT32 compute with INT32 scale type (alpha/beta are int32_t)
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

    // D[m,n] = sum_k A[m,k] * B[n,k]
    // A: RowMajor MxK = ColMajor KxM, B: ColMajor NxK = ColMajor KxN (transposed)
    // D: RowMajor MxN = ColMajor NxM
    // In col-major: D' = B'^T * A' where B' is KxN, A' is KxM, D' is NxM
    cublasOperation_t transA = CUBLAS_OP_T;  // B' (KxN) transposed gives NxK
    cublasOperation_t transB = CUBLAS_OP_N;  // A' (KxM) as-is gives KxM
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Layout for B (cuBLAS "A"): RowMajor NxK = ColMajor KxN, ld=K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, K, N, K));
    // Layout for A (cuBLAS "B"): RowMajor MxK = ColMajor KxM, ld=K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, K, M, K));
    // Layout for D: RowMajor MxN = ColMajor NxM, ld=N
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_32I, N, M, N));

    // Workspace
    workspaceSize = 32 * 1024 * 1024;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    // Preference
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspaceSize, sizeof(workspaceSize)));

    // Get best algorithm
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, layoutA, layoutB, layoutD, layoutD,
                                                 preference, 1, &heuristic, &returnedResults));
    if (returnedResults == 0) {
      std::cerr << "No algorithm found!" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  void run(int8_t const* A, int8_t const* B, int32_t* D, cudaStream_t stream = nullptr) {
    const int32_t alpha = 1;
    const int32_t beta = 0;
    // Note: B is first arg, A is second arg (for the transpose trick)
    CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc, &alpha,
                                 B, layoutA,   // "A" in cublasLt = our B
                                 A, layoutB,   // "B" in cublasLt = our A
                                 &beta,
                                 D, layoutD,
                                 D, layoutD,
                                 &heuristic.algo, workspace, workspaceSize, stream));
  }

  void destroy() {
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutA));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutB));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutD));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc));
    CHECK_CUBLAS(cublasLtDestroy(handle));
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Benchmark function
///////////////////////////////////////////////////////////////////////////////////////////////////

void benchmark(int M, int N, int K) {
  // Cooldown between configurations
  sleep_ms(500);

  std::cout << "\n----------------------------------------" << std::endl;
  std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

  // L2 cache eviction - multiple buffer groups
  int l2_cache_size;
  cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
  // INT8 inputs are 1 byte each, INT32 output is 4 bytes
  const size_t arg_size = size_t(M) * K + size_t(N) * K + 4 * size_t(M) * N;
  const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
  const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

  // Allocate buffer groups
  std::vector<int8_t*> blocks_A(arg_group_count);
  std::vector<int8_t*> blocks_B(arg_group_count);
  std::vector<int32_t*> blocks_D(arg_group_count);
  int32_t* block_D_ref;

  size_t size_A = size_t(M) * K;
  size_t size_B = size_t(K) * N;
  size_t size_D = size_t(M) * N;

  CHECK_CUDA(cudaMalloc(&block_D_ref, size_D * sizeof(int32_t)));

  uint64_t seed = 2024;
  for (int i = 0; i < arg_group_count; ++i) {
    CHECK_CUDA(cudaMalloc(&blocks_A[i], size_A * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&blocks_B[i], size_B * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&blocks_D[i], size_D * sizeof(int32_t)));

    // Small range keeps float reference accumulation exact for correctness checking
    fill<int8_t, FillMode::RANDOM>(blocks_A[i], size_A, seed + i * 100, -128.0f, 127.0f);
    fill<int8_t, FillMode::RANDOM>(blocks_B[i], size_B, seed + i * 100 + 1, -128.0f, 127.0f);
    fill<int32_t, FillMode::CONSTANT>(blocks_D[i], size_D, 0.0f);
  }
  fill<int32_t, FillMode::CONSTANT>(block_D_ref, size_D, 0.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Compute reference GEMM (float accumulation, int32 output)
  reference_gemm<int8_t, int32_t>(block_D_ref, blocks_A[0], blocks_B[0], M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Initialize cuBLASLt
  CublasLtGemm gemm;
  gemm.init(M, N, K);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Warmup
  for (int i = 0; i < warmup_iters; ++i) {
    int idx = i % arg_group_count;
    gemm.run(blocks_A[idx], blocks_B[idx], blocks_D[idx], stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < profiling_iters; ++i) {
    int idx = i % arg_group_count;
    gemm.run(blocks_A[idx], blocks_B[idx], blocks_D[idx], stream);
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

  double runtime_ms = static_cast<double>(milliseconds) / profiling_iters;
  double runtime_s = runtime_ms / 1000.0;
  int64_t ops = int64_t(2) * M * N * K;
  double tops = (double(ops) / 1e12) / runtime_s;

  std::cout << "Average runtime: " << runtime_ms << " ms" << std::endl;
  std::cout << "Performance: " << tops << " TOP/s" << std::endl;

  // Verify correctness
  fill<int32_t, FillMode::CONSTANT>(blocks_D[0], size_D, 0.0f);
  gemm.run(blocks_A[0], blocks_B[0], blocks_D[0], stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  check_correctness(blocks_D[0], block_D_ref, size_D);

  // Cleanup
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  gemm.destroy();

  for (int i = 0; i < arg_group_count; ++i) {
    CHECK_CUDA(cudaFree(blocks_A[i]));
    CHECK_CUDA(cudaFree(blocks_B[i]));
    CHECK_CUDA(cudaFree(blocks_D[i]));
  }
  CHECK_CUDA(cudaFree(block_D_ref));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
  std::cout << "cuBLASLt INT8 GEMM Profiler" << std::endl;
  std::cout << "D = A * B, A: RowMajor (MxK), B: ColMajor (NxK), D: RowMajor (MxN)" << std::endl;
  std::cout << "Input: INT8, Accumulator: INT32, Output: INT32" << std::endl;
  std::cout << "Warmup: " << warmup_iters << ", Profiling: " << profiling_iters << std::endl;

  benchmark(1024, 1024, 1024);
  benchmark(2048, 2048, 2048);
  benchmark(4096, 4096, 4096);
  benchmark(8192, 8192, 8192);
  benchmark(16384, 16384, 16384);

  return 0;
}
