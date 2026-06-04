# ---------------------------------------------------------------------------
# kernel-set — C++ correctness tests (gated by KS_ENABLE_TESTS).
#
# The top-level CMakeLists adds this directory only inside `if(KS_ENABLE_TESTS)`
# and after `enable_testing()`. Each test is a single .cu translation unit that
# launches a public C ABI kernel and compares the GPU result to a host
# reference (see test_common.cuh). No gtest: a tiny CHECK macro set lives in
# the harness header. Tests are registered with CTest via add_test.
#
# Build & run:
#   cmake -S . -B build -DKS_ENABLE_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure
#
# Each test SKIPS (returns 0 after printing a notice) when no CUDA device is
# present at runtime, so `ctest` stays green on CI hosts without a GPU while
# still proving the kernels compile and link against the frozen ABI.
# ---------------------------------------------------------------------------

# Tests are CUDA programs; the language is already enabled by the parent for the
# CUDA build path. Guard so a HIP-only configuration doesn't try to add them.
if(NOT KS_ENABLE_CUDA)
  message(STATUS "kernel-set tests: skipped (KS_ENABLE_CUDA is OFF)")
  return()
endif()

# Inherit the parent's CUDA architectures; fall back to a broad default so the
# tests can be configured standalone too.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
endif()

# Every .cu in this directory is one test executable. Globbing keeps adding a
# new test a zero-CMake-edit operation, matching the kernel source convention.
file(GLOB KS_TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu")

foreach(test_src IN LISTS KS_TEST_SOURCES)
  get_filename_component(test_name "${test_src}" NAME_WE)

  add_executable(${test_name} "${test_src}")

  target_link_libraries(${test_name} PRIVATE kernel_set)

  # Public ABI headers + this directory for the shared test harness header.
  target_include_directories(${test_name} PRIVATE
      "${CMAKE_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}")

  set_target_properties(${test_name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON)

  target_compile_options(${test_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

  add_test(NAME ${test_name} COMMAND ${test_name})

  # A clean run prints "[PASS] ..."; treat a missing-GPU skip as a pass too so
  # CI without a device is green. Failures return non-zero and fail the test.
  set_tests_properties(${test_name} PROPERTIES
      PASS_REGULAR_EXPRESSION "\\[PASS\\]|\\[skip\\]"
      FAIL_REGULAR_EXPRESSION "\\[FAIL\\]|\\[FAILED\\]"
      TIMEOUT 120)

  message(STATUS "kernel-set test: ${test_name}")
endforeach()
