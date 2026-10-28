cmake_minimum_required(VERSION 3.20)
project(Kinesis VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for language servers (clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -march=native bakes the build machine's ISA into the binary: it is right for a local
# benchmark and wrong for anything shipped, so it is a switch rather than a default.
option(KINESIS_NATIVE_ARCH "Compile for the host CPU's instruction set (-march=native)" ON)
option(KINESIS_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# Compiler Warning & Optimization Flags
if(MSVC)
    add_compile_options(
        /W4
        /permissive-
        /volatile:iso
        /Zc:inline
        /Zc:preprocessor
        /O2
        /arch:AVX2
    )
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -O3
    )
    if(KINESIS_NATIVE_ARCH)
        add_compile_options(-march=native)
    else()
        # Baseline that still enables the AVX2 + FMA kernels guarded by KINESIS_HAS_AVX2.
        # Without either of these, x86-64 compilers target SSE2 and the SIMD paths compile out.
        add_compile_options(-mavx2 -mfma)
    endif()
    if(KINESIS_SANITIZE)
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# Core Library Include Directories
set(KINESIS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Core Library Sources
set(KINESIS_SOURCES
    src/core/arena_allocator.cpp
    src/compute/gemm.cpp
)

# Kinesis Static Library
add_library(kinesis STATIC ${KINESIS_SOURCES})
target_include_directories(kinesis PUBLIC
    $<BUILD_INTERFACE:${KINESIS_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Multi-threading support (for blocked parallel GEMM)
find_package(Threads REQUIRED)
target_link_libraries(kinesis PUBLIC Threads::Threads)

# Test Suite Executable
set(TEST_SOURCES
    tests/test_main.cpp
    tests/test_arena_allocator.cpp
    tests/test_virtual_allocator.cpp
    tests/test_slab_allocator.cpp
    tests/test_memory_sandbox.cpp
    tests/test_hardware_info.cpp
    tests/test_tensor_view.cpp
    tests/test_gemm.cpp
    tests/test_dual.cpp
    tests/test_regressions.cpp
)

add_executable(kinesis_tests ${TEST_SOURCES})
target_include_directories(kinesis_tests PRIVATE ${KINESIS_INCLUDE_DIR} tests)
target_link_libraries(kinesis_tests PRIVATE kinesis)

# Latency Benchmark Executable
add_executable(kinesis_benchmarks benchmarks/bench_latency.cpp)
target_include_directories(kinesis_benchmarks PRIVATE ${KINESIS_INCLUDE_DIR})
target_link_libraries(kinesis_benchmarks PRIVATE kinesis)

# CTest configuration
enable_testing()
add_test(NAME AllUnitTests COMMAND kinesis_tests)
