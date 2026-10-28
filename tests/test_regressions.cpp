// Regression tests for the defects found in the 2026-09-17 design review.
// Each test names the defect it pins down; see docs/DESIGN_REVIEW_2026-09-17.md section 6.
#include "test_harness.hpp"
#include "kinesis/compute/gemm.hpp"
#include "kinesis/compute/elementwise.hpp"
#include "kinesis/core/arena_allocator.hpp"
#include "kinesis/core/slab_allocator.hpp"
#include "kinesis/core/tensor.hpp"
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using namespace kinesis::core;
using namespace kinesis::compute;

namespace {

void fill_random(Tensor<float>& t, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* data = t.data();
    const int64_t n = t.total_elements();
    for (int64_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

} // namespace

// Defect 1: gemm_packed indexed A and C assuming unit column stride, so a transposed operand
// silently produced wrong results. gramian() passes A^T, so every Gramian of a matrix with at
// least 128 columns was wrong (128 is the threshold that dispatches to the packed kernel).
KINESIS_TEST_CASE(Regression_GramianWithPackedDispatchMatchesReference) {
    ArenaAllocator arena(16 * 1024 * 1024);
    constexpr int64_t M = 200;
    constexpr int64_t N = 130; // >= 128 so gemm() takes the packed path

    Tensor<float> A(Shape{M, N}, arena);
    fill_random(A, 7);

    Tensor<float> G(Shape{N, N}, 0.0f, arena);
    Tensor<float> G_ref(Shape{N, N}, 0.0f, arena);

    gramian(A.view(), G.view());
    gemm_naive(A.t(), A.view(), G_ref.view());

    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            ASSERT_NEAR(G(i, j), G_ref(i, j), 2e-3f);
        }
    }
}

// Defect 1 (general form): any transposed or strided operand must go through a stride-aware path.
KINESIS_TEST_CASE(Regression_TransposedAndStridedOperandsMatchReference) {
    ArenaAllocator arena(32 * 1024 * 1024);
    constexpr int64_t M = 130;
    constexpr int64_t K = 140;
    constexpr int64_t N = 150;

    Tensor<float> At(Shape{K, M}, arena); // A stored transposed: A = At^T is (M, K)
    Tensor<float> B(Shape{K, N}, arena);
    fill_random(At, 11);
    fill_random(B, 13);

    Tensor<float> C(Shape{M, N}, 0.0f, arena);
    Tensor<float> C_ref(Shape{M, N}, 0.0f, arena);

    gemm(At.t(), B.view(), C.view());
    gemm_naive(At.t(), B.view(), C_ref.view());
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            ASSERT_NEAR(C(i, j), C_ref(i, j), 2e-3f);
        }
    }

    // Strided output: write into the transpose of an (N, M) tensor, so C has column stride M.
    Tensor<float> Ct(Shape{N, M}, 0.0f, arena);
    gemm(At.t(), B.view(), Ct.t());
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            ASSERT_NEAR(Ct(j, i), C_ref(i, j), 2e-3f);
        }
    }
}

// Defect 2: with N not a multiple of the 8-wide micro-panel, the tail used full-width SIMD
// loads and stores, reading (AVX2) or writing (NEON) past the end of the output row. The canary
// tensor is allocated immediately after C in the same arena: C's byte size is a multiple of the
// 64-byte allocation alignment, so any write past C lands in the canary.
KINESIS_TEST_CASE(Regression_PackedTailDoesNotOverrunOutput) {
    ArenaAllocator arena(16 * 1024 * 1024);
    constexpr int64_t M = 128;
    constexpr int64_t K = 70;
    constexpr int64_t N = 131; // 131 % 8 == 3 -> partial micro-panel on every row

    static_assert((M * N * sizeof(float)) % 64 == 0,
                  "C must end on the allocator's alignment boundary for the canary to be adjacent");

    Tensor<float> A(Shape{M, K}, arena);
    Tensor<float> B(Shape{K, N}, arena);
    fill_random(A, 17);
    fill_random(B, 19);

    Tensor<float> C(Shape{M, N}, 0.0f, arena);
    Tensor<float> canary(Shape{64}, 12345.0f, arena);
    ASSERT_EQ(reinterpret_cast<const char*>(canary.data()),
              reinterpret_cast<const char*>(C.data()) + M * N * sizeof(float));

    Tensor<float> C_ref(Shape{M, N}, 0.0f, arena);
    gemm(A.view(), B.view(), C.view());
    gemm_naive(A.view(), B.view(), C_ref.view());

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            ASSERT_NEAR(C(i, j), C_ref(i, j), 2e-3f);
        }
    }
    for (int64_t i = 0; i < canary.total_elements(); ++i) {
        ASSERT_EQ(canary(i), 12345.0f);
    }
}

// Defect 9: read-only views could not be passed to kernels, and Tensor::clone() did not compile.
KINESIS_TEST_CASE(Regression_ConstViewsReachKernelsAndCloneCompiles) {
    ArenaAllocator arena(1024 * 1024);

    Tensor<float> weights(Shape{2, 3}, 2.0f, arena);
    const Tensor<float>& read_only = weights; // e.g. permanent or memory-mapped weights

    Tensor<float> x(Shape{3, 2}, 1.0f, arena);
    Tensor<float> y(Shape{2, 2}, 0.0f, arena);

    gemm(read_only.view(), x.view(), y.view());
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 2; ++j) {
            ASSERT_NEAR(y(i, j), 6.0f, 1e-5f);
        }
    }

    // Elementwise kernels accept read-only inputs too.
    Tensor<float> sum(Shape{2, 3}, 0.0f, arena);
    add(read_only.view(), weights.view(), sum.view());
    ASSERT_NEAR(sum(1, 2), 4.0f, 1e-5f);

    // clone() previously failed to compile: it passed a TensorView<const T> to copy_from().
    Tensor<float> copy = weights.clone(arena);
    ASSERT_TRUE(copy.data() != weights.data());
    ASSERT_EQ(copy.shape(), weights.shape());
    for (int64_t i = 0; i < copy.total_elements(); ++i) {
        ASSERT_EQ(copy.data()[i], weights.data()[i]);
    }
}

// Defect 4: shape validation was assert-only, so release builds silently truncated dimensions
// above 2^32 and wrote past the dimension array for ranks above 6. Slicing and broadcasting had
// the same problem.
KINESIS_TEST_CASE(Regression_ShapeAndViewValidationSurviveRelease) {
    bool threw = false;
    try {
        Shape too_many{1, 2, 3, 4, 5, 6, 7};
        (void)too_many.rank();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    threw = false;
    try {
        Shape too_large{static_cast<int64_t>(5'000'000'000)};
        (void)too_large.rank();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    threw = false;
    try {
        Shape negative{-1};
        (void)negative.rank();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    std::vector<float> raw(24, 0.0f);
    TensorView<float> view(raw.data(), Shape{4, 6});

    threw = false;
    try {
        (void)view.slice(0, 0, 99);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    threw = false;
    try {
        (void)view.broadcast_to(Shape{6});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

// Defect 12: the singularity test used an absolute threshold, so a well-conditioned system
// expressed in small units was rejected as singular.
KINESIS_TEST_CASE(Regression_CramerThresholdIsScaleInvariant) {
    float x0 = 0.0f, x1 = 0.0f;

    // Same system as the 2x2 unit test, scaled by 1e-7. Solution stays (1, 3).
    const float s = 1e-7f;
    const bool ok = solve_cramer_2x2(2.0f * s, 1.0f * s, 1.0f * s, 3.0f * s,
                                     5.0f * s, 10.0f * s, x0, x1);
    ASSERT_TRUE(ok);
    ASSERT_NEAR(x0, 1.0f, 1e-3f);
    ASSERT_NEAR(x1, 3.0f, 1e-3f);

    // Exactly singular, and near-singular at large scale: both rejected.
    ASSERT_FALSE(solve_cramer_2x2(1e6f, 2e6f, 2e6f, 4e6f, 1.0f, 2.0f, x0, x1));
    ASSERT_FALSE(solve_cramer_2x2(1e6f, 2e6f, 2e6f, 4e6f + 1.0f, 1.0f, 2.0f, x0, x1));

    // 3x3, scaled down by the same factor; solution stays (1, 2, 3).
    float A3[9] = {1.0f * s, 2.0f * s, 3.0f * s,
                   0.0f * s, 1.0f * s, 4.0f * s,
                   5.0f * s, 6.0f * s, 0.0f * s};
    float b3[3] = {14.0f * s, 14.0f * s, 17.0f * s};
    float x3[3] = {0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(solve_cramer_3x3(A3, b3, x3));
    ASSERT_NEAR(x3[0], 1.0f, 1e-3f);
    ASSERT_NEAR(x3[1], 2.0f, 1e-3f);
    ASSERT_NEAR(x3[2], 3.0f, 1e-3f);
}

// BLAS semantics: alpha == 0 means C = beta * C without touching A or B, so a non-finite value
// in A must not leak into the result. Covers both the blocked and packed dispatch paths.
KINESIS_TEST_CASE(Regression_AlphaZeroLeavesOperandsUnread) {
    ArenaAllocator arena(16 * 1024 * 1024);
    const float nan_value = std::numeric_limits<float>::quiet_NaN();

    for (int64_t size : {8, 130}) {
        Tensor<float> A(Shape{size, size}, nan_value, arena);
        Tensor<float> B(Shape{size, size}, 1.0f, arena);
        Tensor<float> C(Shape{size, size}, 2.0f, arena);

        gemm(A.view(), B.view(), C.view(), 0.0f, 3.0f);

        for (int64_t i = 0; i < size; ++i) {
            for (int64_t j = 0; j < size; ++j) {
                ASSERT_TRUE(!std::isnan(C(i, j)));
                ASSERT_NEAR(C(i, j), 6.0f, 1e-5f);
            }
        }
    }
}

// ReLU's NaN behaviour must not depend on the instruction set: x86 maxps returns 0 for NaN, so
// the NEON path uses compare-and-select instead of vmaxq_f32, which would propagate NaN.
KINESIS_TEST_CASE(Regression_ReluNaNPolicyIsIsaIndependent) {
    ArenaAllocator arena(64 * 1024);
    Tensor<float> x(Shape{12}, arena);
    Tensor<float> y(Shape{12}, 0.0f, arena);

    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    for (int64_t i = 0; i < 12; ++i) {
        x(i) = (i % 3 == 0) ? nan_value : (i % 3 == 1 ? -1.5f : 2.5f);
    }

    relu(x.view(), y.view());
    for (int64_t i = 0; i < 12; ++i) {
        if (i % 3 == 2) {
            ASSERT_NEAR(y(i), 2.5f, 1e-6f);
        } else {
            ASSERT_EQ(y(i), 0.0f);
        }
    }
}

// Section 6.2: the arena's capacity check was `current_offset_ + padding + bytes > capacity_`,
// which wraps for a large request. allocate(SIZE_MAX) then passed the check and returned a
// pointer past the end of the arena.
KINESIS_TEST_CASE(Regression_ArenaRejectsOverflowingAndInvalidRequests) {
    ArenaAllocator arena(4096);

    bool threw = false;
    try {
        (void)arena.allocate(std::numeric_limits<size_t>::max(), 64);
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
    ASSERT_EQ(arena.allocated_bytes(), static_cast<size_t>(0)); // the failed request consumed nothing

    // A size just under the wrap point must also be refused rather than wrapping the sum.
    threw = false;
    try {
        (void)arena.allocate(std::numeric_limits<size_t>::max() - 32, 64);
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    // Alignment validation must survive a release build: 0 would make `alignment - 1` SIZE_MAX,
    // and a non-power-of-two mask would return a misaligned pointer.
    threw = false;
    try {
        (void)arena.allocate(16, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    threw = false;
    try {
        (void)arena.allocate(16, 48);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    // Still usable afterwards, and still handing back correctly aligned memory.
    void* p = arena.allocate(128, 64);
    ASSERT_TRUE(p != nullptr);
    ASSERT_TRUE(is_aligned(p, 64));
}

// Section 6.2: the slab validated its arguments with asserts only, so a release build would
// hand out an undersized block and link a foreign pointer into the intrusive free list.
KINESIS_TEST_CASE(Regression_SlabValidationSurvivesRelease) {
    SlabAllocator<256> slab(8);
    const size_t free_before = slab.free_blocks();

    bool threw = false;
    try {
        (void)slab.allocate(4096, 64); // larger than a block
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    threw = false;
    try {
        (void)slab.allocate(16, 4096); // stricter alignment than the slab provides
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    ASSERT_EQ(slab.free_blocks(), free_before);

    // A pointer the slab does not own must be refused, not pushed onto the free list.
    int foreign = 0;
    slab.deallocate(&foreign, 256, 64);
    ASSERT_EQ(slab.free_blocks(), free_before);

    // A real block still round-trips.
    void* p = slab.allocate(256, 64);
    ASSERT_TRUE(p != nullptr);
    ASSERT_EQ(slab.free_blocks(), free_before - 1);
    slab.deallocate(p);
    ASSERT_EQ(slab.free_blocks(), free_before);
}

// Defect 4, remaining structural operations: transpose, t(), make_identity and copy_from also
// validated with assert only. In a release build these corrupted memory rather than failing.
// Each case places a canary directly after the buffer so a pre-fix overrun is a deterministic
// assertion failure instead of silent corruption.
KINESIS_TEST_CASE(Regression_StructuralViewOpsValidateInRelease) {
    // make_identity on a non-square 4x2 view: the diagonal loop used to run to i = 3 and write
    // (3, 3) at offset 9 of an 8-element buffer.
    {
        std::vector<float> storage(8 + 8, 0.0f);
        for (size_t i = 8; i < storage.size(); ++i) storage[i] = 777.0f; // canary
        TensorView<float> tall(storage.data(), Shape{4, 2});

        bool threw = false;
        try {
            tall.make_identity();
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        ASSERT_TRUE(threw);
        for (size_t i = 8; i < storage.size(); ++i) {
            ASSERT_EQ(storage[i], 777.0f);
        }
    }

    // copy_from with a larger strided source: the strided path walked the *source's* 3x3
    // indices while writing into a 2x2 destination, landing past its end.
    {
        std::vector<float> src_storage(9, 5.0f);
        TensorView<float> src_rows(src_storage.data(), Shape{3, 3});
        TensorView<const float> src = src_rows.t(); // non-contiguous, forces the strided path

        std::vector<float> dst_storage(4 + 8, 0.0f);
        for (size_t i = 4; i < dst_storage.size(); ++i) dst_storage[i] = 777.0f; // canary
        TensorView<float> dst(dst_storage.data(), Shape{2, 2});

        bool threw = false;
        try {
            dst.copy_from(src);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        ASSERT_TRUE(threw);
        for (size_t i = 4; i < dst_storage.size(); ++i) {
            ASSERT_EQ(dst_storage[i], 777.0f);
        }
    }

    // transpose with a dimension index past the rank, and past the 6-slot dimension array.
    {
        std::vector<float> raw(24, 0.0f);
        TensorView<float> view(raw.data(), Shape{4, 6});

        bool threw = false;
        try {
            (void)view.transpose(0, 2); // rank 2: slot 2 is unused
        } catch (const std::out_of_range&) {
            threw = true;
        }
        ASSERT_TRUE(threw);

        threw = false;
        try {
            (void)view.transpose(0, 7); // past the end of the dimension array
        } catch (const std::out_of_range&) {
            threw = true;
        }
        ASSERT_TRUE(threw);

        TensorView<float> vec(raw.data(), Shape{24});
        threw = false;
        try {
            (void)vec.t(); // t() is only defined for matrices
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        ASSERT_TRUE(threw);
    }

    // The valid cases still work.
    {
        std::vector<float> sq(9, 3.0f);
        TensorView<float> m(sq.data(), Shape{3, 3});
        m.make_identity();
        ASSERT_EQ(m(0, 0), 1.0f);
        ASSERT_EQ(m(0, 1), 0.0f);
        ASSERT_EQ(m(2, 2), 1.0f);
        ASSERT_EQ(m.t().shape(), m.shape());
    }
}
