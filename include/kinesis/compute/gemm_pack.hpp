#pragma once

#include "kinesis/core/types.hpp"
#include "kinesis/core/tensor_view.hpp"
#include "kinesis/core/allocator.hpp"
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace kinesis::compute {

/**
 * @brief Dynamic Cache-Tile Sizing Formula (OS-Independent).
 * 
 * Computes optimal square tile dimension B based on target L1 cache capacity:
 *   Working Set = 3 * B^2 * sizeof(T) <= L1_Bytes
 *   B = sqrt(L1_Bytes / (3 * sizeof(T)))
 */
[[nodiscard]] constexpr int64_t compute_dynamic_gemm_tile(size_t l1_capacity_bytes = 49152, 
                                                          size_t element_size = sizeof(float)) noexcept {
    if (element_size == 0 || l1_capacity_bytes == 0) return 32;
    // B = sqrt(l1_capacity_bytes / (3 * element_size))
    // Integer square root via Newton-Raphson for constexpr compatibility
    const size_t target = l1_capacity_bytes / (3 * element_size);
    size_t x = target;
    size_t y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + target / x) / 2;
    }
    // Snap to multiple of 8 for vector register alignment
    int64_t tile = static_cast<int64_t>(x);
    tile = (tile / 8) * 8;
    return (tile < 16) ? 16 : tile;
}

/**
 * @brief GotoBLAS / BLIS Micro-Panel Packing for Matrix B.
 * 
 * Packs a (K x N) block of matrix B into contiguous vertical micro-panels of width Nr.
 * Layout in memory: B_packed[n_panels][k_rows][Nr]
 * Completely eliminates TLB page table walk misses and converts non-unit strides into 
 * sequential stride-1 burst reads for SIMD vectors.
 */
inline void pack_B_panel(const core::TensorView<const float>& b,
                         int64_t k_start, int64_t k_end,
                         int64_t n_start, int64_t n_end,
                         float* packed_b,
                         int64_t Nr = 8) noexcept {
    float* dest = packed_b;
    for (int64_t j = n_start; j < n_end; j += Nr) {
        const int64_t nr_actual = std::min(Nr, n_end - j);

        for (int64_t k = k_start; k < k_end; ++k) {
            int64_t r = 0;
            for (; r < nr_actual; ++r) {
                *dest++ = b(k, j + r);
            }
            // Zero-pad partial tail micro-panels to avoid register garbage
            for (; r < Nr; ++r) {
                *dest++ = 0.0f;
            }
        }
    }
}

/**
 * @brief GotoBLAS / BLIS Micro-Panel Packing for Matrix A.
 * 
 * Packs a (M x K) block of matrix A into contiguous horizontal micro-panels of height Mr.
 * Layout in memory: A_packed[m_panels][k_cols][Mr]
 */
inline void pack_A_panel(const core::TensorView<const float>& a,
                         int64_t m_start, int64_t m_end,
                         int64_t k_start, int64_t k_end,
                         float* packed_a,
                         int64_t Mr = 4) noexcept {
    float* dest = packed_a;
    for (int64_t i = m_start; i < m_end; i += Mr) {
        const int64_t mr_actual = std::min(Mr, m_end - i);

        for (int64_t k = k_start; k < k_end; ++k) {
            int64_t r = 0;
            for (; r < mr_actual; ++r) {
                *dest++ = a(i + r, k);
            }
            // Zero-pad partial tail micro-panels
            for (; r < Mr; ++r) {
                *dest++ = 0.0f;
            }
        }
    }
}

} // namespace kinesis::compute
