#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace kinesis::core {

// Hardware-Adaptive Alignment Granularities
inline constexpr size_t kGpuSectorAlignment =
    32; // NVIDIA 32-byte sectors, AMD Wave32 / dual 32B lines
inline constexpr size_t kCpuCacheLineAlignment =
    64; // Standard x86_64 L1 cache lines & AVX-512
inline constexpr size_t kAppleSiliconAlignment =
    128; // Apple Silicon M-series performance cores & GPU bursts

// Target default alignment (64-byte baseline for CPU; mathematically satisfies
// 32-byte sub-sectors)
inline constexpr size_t kDefaultAlignment = kCpuCacheLineAlignment;

// Explicit precision types
using f32 = float;
using f64 = double;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using byte = std::byte;

// Tensor dimension & indexing types
using Dim = u32; // 32-bit dimension: shrinks tensor metadata to fit in single
                 // 64B cache line
using Stride = i64; // 64-bit stride: guarantees zero overflow for large
                    // multi-gigabyte buffers

// C++20 Concepts for mathematical operations
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template <typename T>
concept Integral = std::is_integral_v<T>;

// Memory Layout Orders
enum class MemoryOrder {
  RowMajor, // C-contiguous (default)
  ColMajor  // Fortran-contiguous
};

// Alignment helper functions
[[nodiscard]] constexpr size_t align_up(size_t address,
                                        size_t alignment) noexcept {
  return (address + alignment - 1) & ~(alignment - 1);
}

[[nodiscard]] inline uintptr_t align_pointer(uintptr_t ptr,
                                             size_t alignment) noexcept {
  return (ptr + alignment - 1) & ~(alignment - 1);
}

[[nodiscard]] inline bool is_aligned(const void *ptr,
                                     size_t alignment) noexcept {
  return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

} // namespace kinesis::core
