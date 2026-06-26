cmake_minimum_required(VERSION 3.24)

# ---------------------------------------------------------------------------
# kernel-set: high-performance LLM inference & training kernels
# ---------------------------------------------------------------------------
project(kernel_set VERSION 0.2.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- Options --------------------------------------------------------------
option(KS_ENABLE_CUDA "Build the CUDA backend" ON)
option(KS_ENABLE_HIP  "Build the HIP/ROCm backend (reserved)" OFF)
option(KS_ENABLE_TESTS "Build C++ correctness tests" OFF)
option(KS_ENABLE_WMMA_GEMM "Enable WMMA tensor-core GEMM fast path (half/bf16)" OFF)
option(KS_FAST_MATH "Enable --use_fast_math (lower precision transcendentals)" OFF)
# Default arch list covers: T4(75) A100(80) A10/3090(86) L4/4090(89) H100(90).
# Colab: L4 -> 89, A100 -> 80. Override with -DCMAKE_CUDA_ARCHITECTURES=89.
set(KS_CUDA_ARCHS "70;75;80;86;89;90" CACHE STRING "CUDA architectures to target")

# ---- Collect sources (glob so new kernels need no CMake edits) ------------
file(GLOB_RECURSE KS_GPU_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/kernels/src/*.cu")
file(GLOB_RECURSE KS_CPP_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/kernels/src/*.cpp")

if(KS_ENABLE_HIP AND NOT KS_ENABLE_CUDA)
  # ---- HIP/ROCm path (reserved; hipify-perl or native HIP) ----------------
  enable_language(HIP)
  set_source_files_properties(${KS_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
  add_library(kernel_set SHARED ${KS_GPU_SOURCES} ${KS_CPP_SOURCES})
  target_compile_definitions(kernel_set PRIVATE KS_PLATFORM_HIP=1)
else()
  # ---- CUDA path (default) ------------------------------------------------
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${KS_CUDA_ARCHS})
  endif()
  add_library(kernel_set SHARED ${KS_GPU_SOURCES} ${KS_CPP_SOURCES})
  set_target_properties(kernel_set PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(kernel_set PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  if(KS_FAST_MATH)
    target_compile_options(kernel_set PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  endif()
endif()

target_include_directories(kernel_set
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/src)

target_compile_definitions(kernel_set PRIVATE
    KERNEL_SET_BUILD=1
    KERNEL_SET_VERSION="${PROJECT_VERSION}")

if(KS_ENABLE_WMMA_GEMM)
  target_compile_definitions(kernel_set PRIVATE KS_ENABLE_WMMA_GEMM=1)
endif()

# Link cuBLAS for the GEMM fast path (ks_gemm dispatches to cublasGemmEx).
find_library(CUBLAS_LIBRARY cublas PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
if(CUBLAS_LIBRARY)
  target_link_libraries(kernel_set PRIVATE ${CUBLAS_LIBRARY})
else()
  target_link_libraries(kernel_set PRIVATE cublas)
endif()

set_target_properties(kernel_set PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})

# ---- Install --------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS kernel_set
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ---- Tests ----------------------------------------------------------------
if(KS_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(kernels/tests)
endif()

message(STATUS "kernel-set ${PROJECT_VERSION}")
message(STATUS "  GPU sources : ${KS_GPU_SOURCES}")
message(STATUS "  CUDA archs  : ${CMAKE_CUDA_ARCHITECTURES}")
