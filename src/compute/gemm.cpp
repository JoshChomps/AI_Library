#include "kinesis/compute/gemm.hpp"
#include "kinesis/compute/gemm_pack.hpp"
#include "kinesis/compute/simd.hpp"
#include "kinesis/compute/elementwise.hpp"
#include <algorithm>
#include <array>
#include <cassert>

namespace kinesis::compute {

void gemm_naive(const core::TensorView<const float>& a,
                const core::TensorView<const float>& b,
                core::TensorView<float> c,
                float alpha,
                float beta) {
    assert(a.rank() == 2 && b.rank() == 2 && c.rank() == 2);
    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(1);
    assert(b.size(0) == K && c.size(0) == M && c.size(1) == N);

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += a(i, k) * b(k, j);
            }
            if (beta == 0.0f) {
                c(i, j) = alpha * sum;
            } else {
                c(i, j) = alpha * sum + beta * c(i, j);
            }
        }
    }
}

void gemm_blocked(const core::TensorView<const float>& a,
                  const core::TensorView<const float>& b,
                  core::TensorView<float> c,
                  float alpha,
                  float beta) {
    assert(a.rank() == 2 && b.rank() == 2 && c.rank() == 2);
    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t N = b.size(1);
    assert(b.size(0) == K && c.size(0) == M && c.size(1) == N);

    // Step 1: Scale existing C by beta
    if (beta == 0.0f) {
        c.fill(0.0f);
    } else if (beta != 1.0f) {
        scale(c, beta, c);
    }

    // BLAS semantics: with alpha == 0 the product is not formed at all, so a NaN or infinity in
    // A or B cannot leak into C.
    if (alpha == 0.0f || K == 0) {
        return;
    }

    // Cache blocking tile sizes tuned for modern L1/L2 data caches
    constexpr int64_t Mc = 64;
    constexpr int64_t Kc = 64;
    constexpr int64_t Nc = 64;

    // Hoisted out of the inner loops: whether the SIMD path is usable depends only on the
    // layouts of B and C, never on the loop indices.
    const bool simd_rows = (b.stride(1) == 1 && c.stride(1) == 1);

    // Outer tile loops
    for (int64_t m_tile = 0; m_tile < M; m_tile += Mc) {
        const int64_t m_end = std::min(M, m_tile + Mc);

        for (int64_t k_tile = 0; k_tile < K; k_tile += Kc) {
            const int64_t k_end = std::min(K, k_tile + Kc);

            for (int64_t n_tile = 0; n_tile < N; n_tile += Nc) {
                const int64_t n_end = std::min(N, n_tile + Nc);
                const int64_t count = n_end - n_tile;

                // Inner compute kernel: i, k, j ordering for streaming contiguous access
                for (int64_t i = m_tile; i < m_end; ++i) {
                    for (int64_t k = k_tile; k < k_end; ++k) {
                        const float a_val = alpha * a(i, k);

                        if (simd_rows) {
                            const float* b_row = &b(k, n_tile);
                            float* c_row = &c(i, n_tile);
                            int64_t j = 0;

#if defined(KINESIS_HAS_AVX2)
                            __m256 va = _mm256_set1_ps(a_val);
                            for (; j + 8 <= count; j += 8) {
                                __m256 vb = _mm256_loadu_ps(b_row + j);
                                __m256 vc = _mm256_loadu_ps(c_row + j);
                                vc = _mm256_fmadd_ps(va, vb, vc);
                                _mm256_storeu_ps(c_row + j, vc);
                            }
#elif defined(KINESIS_HAS_NEON)
                            float32x4_t va = vdupq_n_f32(a_val);
                            for (; j + 4 <= count; j += 4) {
                                float32x4_t vb = vld1q_f32(b_row + j);
                                float32x4_t vc = vld1q_f32(c_row + j);
                                vc = vmlaq_f32(vc, va, vb);
                                vst1q_f32(c_row + j, vc);
                            }
#endif
                            for (; j < count; ++j) {
                                c_row[j] += a_val * b_row[j];
                            }
                        } else {
                            // Stride-aware fallback: reads and writes through the accessors, so
                            // transposed or otherwise strided views stay correct.
                            for (int64_t j = n_tile; j < n_end; ++j) {
                                c(i, j) += a_val * b(k, j);
                            }
                        }
                    }
                }
            }
        }
    }
}

void gemm_packed(const core::TensorView<const float>& a,
                 const core::TensorView<const float>& b,
                 core::TensorView<float> c,
                 float alpha,
                 float beta,
                 core::Allocator* scratch) {
    assert(a.rank() == 2 && b.rank() == 2 && c.rank() == 2);
    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t N = b.size(1);
    assert(b.size(0) == K && c.size(0) == M && c.size(1) == N);

    // The packed inner loop walks A and C with raw pointers, which is only valid when their
    // columns are unit-stride. Transposed or strided views (e.g. the A^T passed by gramian) go
    // through the blocked kernel, which indexes via the stride-aware accessors.
    // B may have any stride: pack_B_panel reads it through its accessor.
    if (a.stride(1) != 1 || c.stride(1) != 1) {
        gemm_blocked(a, b, c, alpha, beta);
        return;
    }

    // Step 1: Scale existing C by beta
    if (beta == 0.0f) {
        c.fill(0.0f);
    } else if (beta != 1.0f) {
        scale(c, beta, c);
    }

    // BLAS semantics: alpha == 0 means C = beta * C, with A and B untouched.
    if (alpha == 0.0f || K == 0) {
        return;
    }

    // Tile dimensions derived from the target L1 capacity. These are compile-time constants, so
    // the packing buffer below can be a fixed-size block rather than a per-call allocation.
    constexpr int64_t Nr = 8; // AVX2 256-bit register width
    constexpr int64_t Tile = compute_dynamic_gemm_tile(49152, sizeof(float)); // 64
    constexpr size_t kPackedFloats =
        static_cast<size_t>(Tile * ((Tile + Nr - 1) / Nr) * Nr);

    // No heap allocation on the hot path: without a caller-supplied scratch allocator the packed
    // panel lives in a thread-local block (16 KB for the default tile), which keeps the
    // "zero allocations in the hot loop" guarantee that the arena design is built around.
    alignas(64) static thread_local std::array<float, kPackedFloats> tls_packed_b;

    const size_t packed_b_bytes = kPackedFloats * sizeof(float);
    float* packed_b = tls_packed_b.data();
    if (scratch != nullptr) {
        packed_b = static_cast<float*>(scratch->allocate(packed_b_bytes, core::kDefaultAlignment));
    }

    // Outer tile loops: k, n (packed B panel is reused across all M rows)
    for (int64_t k_tile = 0; k_tile < K; k_tile += Tile) {
        const int64_t k_end = std::min(K, k_tile + Tile);
        const int64_t k_len = k_end - k_tile;

        for (int64_t n_tile = 0; n_tile < N; n_tile += Tile) {
            const int64_t n_end = std::min(N, n_tile + Tile);
            const int64_t n_len = n_end - n_tile;

            // Pack B sub-panel into contiguous micro-panels of width Nr
            pack_B_panel(b, k_tile, k_end, n_tile, n_end, packed_b, Nr);

            for (int64_t i = 0; i < M; ++i) {
                const float* a_row = &a(i, k_tile);
                float* c_row = &c(i, n_tile);

                // Stream across packed B micro-panels
                const float* b_p = packed_b;
                for (int64_t j = 0; j < n_len; j += Nr) {
                    const int64_t count = std::min(Nr, n_len - j);

                    if (count == Nr) {
                        // Full micro-panel: c_row[j .. j+Nr) is entirely inside the row.
#if defined(KINESIS_HAS_AVX2)
                        __m256 vc = _mm256_loadu_ps(c_row + j);
                        for (int64_t k = 0; k < k_len; ++k) {
                            __m256 va = _mm256_set1_ps(alpha * a_row[k]);
                            __m256 vb = _mm256_loadu_ps(b_p + k * Nr);
                            vc = _mm256_fmadd_ps(va, vb, vc);
                        }
                        _mm256_storeu_ps(c_row + j, vc);
#elif defined(KINESIS_HAS_NEON)
                        float32x4_t vc0 = vld1q_f32(c_row + j);
                        float32x4_t vc1 = vld1q_f32(c_row + j + 4);
                        for (int64_t k = 0; k < k_len; ++k) {
                            float32x4_t va = vdupq_n_f32(alpha * a_row[k]);
                            vc0 = vmlaq_f32(vc0, va, vld1q_f32(b_p + k * Nr));
                            vc1 = vmlaq_f32(vc1, va, vld1q_f32(b_p + k * Nr + 4));
                        }
                        vst1q_f32(c_row + j, vc0);
                        vst1q_f32(c_row + j + 4, vc1);
#else
                        for (int64_t k = 0; k < k_len; ++k) {
                            const float a_val = alpha * a_row[k];
                            const float* b_col = b_p + k * Nr;
                            for (int64_t r = 0; r < Nr; ++r) {
                                c_row[j + r] += a_val * b_col[r];
                            }
                        }
#endif
                    } else {
                        // Partial micro-panel (N not a multiple of Nr). Scalar so that neither
                        // the load nor the store touches memory past the end of the row: a
                        // full-width store here overran C on the last row, and a full-width load
                        // could fault when C ended on a page boundary.
                        for (int64_t k = 0; k < k_len; ++k) {
                            const float a_val = alpha * a_row[k];
                            const float* b_col = b_p + k * Nr;
                            for (int64_t r = 0; r < count; ++r) {
                                c_row[j + r] += a_val * b_col[r];
                            }
                        }
                    }

                    b_p += k_len * Nr;
                }
            }
        }
    }

    if (scratch != nullptr && packed_b != nullptr) {
        scratch->deallocate(packed_b, packed_b_bytes, core::kDefaultAlignment);
    }
}

void gemm(const core::TensorView<const float>& a,
          const core::TensorView<const float>& b,
          core::TensorView<float> c,
          float alpha,
          float beta) {
    if (a.size(0) >= 128 && b.size(1) >= 128) {
        gemm_packed(a, b, c, alpha, beta);
    } else {
        gemm_blocked(a, b, c, alpha, beta);
    }
}

} // namespace kinesis::compute
