#include "test_harness.hpp"
#include "kinesis/compute/gemm.hpp"
#include "kinesis/core/arena_allocator.hpp"
#include "kinesis/core/tensor.hpp"
#include <random>

using namespace kinesis::core;
using namespace kinesis::compute;

KINESIS_TEST_CASE(GEMM_IdentityMultiplication) {
    ArenaAllocator arena(1024 * 1024);

    // 4x4 matrix A
    Tensor<float> A(Shape{4, 4}, arena);
    // 4x4 Identity matrix I
    Tensor<float> I(Shape{4, 4}, 0.0f, arena);
    // Output C
    Tensor<float> C(Shape{4, 4}, 0.0f, arena);

    for (int64_t i = 0; i < 4; ++i) {
        I(i, i) = 1.0f;
        for (int64_t j = 0; j < 4; ++j) {
            A(i, j) = static_cast<float>(i * 4 + j + 1);
        }
    }

    gemm(A.view(), I.view(), C.view());

    for (int64_t i = 0; i < 4; ++i) {
        for (int64_t j = 0; j < 4; ++j) {
            ASSERT_NEAR(C(i, j), A(i, j), 1e-5f);
        }
    }
}

KINESIS_TEST_CASE(GEMM_BlockedVsNaiveEquivalence) {
    ArenaAllocator arena(8 * 1024 * 1024);

    constexpr int64_t M = 128;
    constexpr int64_t K = 96;
    constexpr int64_t N = 64;

    Tensor<float> A(Shape{M, K}, arena);
    Tensor<float> B(Shape{K, N}, arena);
    Tensor<float> C_naive(Shape{M, N}, arena);
    Tensor<float> C_blocked(Shape{M, N}, arena);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            A(i, k) = dist(rng);
        }
    }
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t j = 0; j < N; ++j) {
            B(k, j) = dist(rng);
        }
    }

    // Compute naive reference
    gemm_naive(A.view(), B.view(), C_naive.view(), 1.5f, 0.0f);

    // Compute blocked SIMD kernel
    gemm_blocked(A.view(), B.view(), C_blocked.view(), 1.5f, 0.0f);

    // Verify numerical equivalence
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            ASSERT_NEAR(C_blocked(i, j), C_naive(i, j), 1e-4f);
        }
    }
}

KINESIS_TEST_CASE(GEMM_AlphaBetaAccumulation) {
    ArenaAllocator arena(1024 * 1024);

    Tensor<float> A(Shape{2, 2}, 1.0f, arena);
    Tensor<float> B(Shape{2, 2}, 2.0f, arena);
    Tensor<float> C(Shape{2, 2}, 5.0f, arena);

    // A @ B = [[4, 4], [4, 4]]
    // 2.0 * (A @ B) + 3.0 * C = 2.0 * 4 + 3.0 * 5 = 8 + 15 = 23
    gemm(A.view(), B.view(), C.view(), 2.0f, 3.0f);

    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 2; ++j) {
            ASSERT_NEAR(C(i, j), 23.0f, 1e-5f);
        }
    }
}

KINESIS_TEST_CASE(GEMM_PackedVsNaiveEquivalence) {
    ArenaAllocator arena(16 * 1024 * 1024);

    constexpr int64_t M = 160;
    constexpr int64_t K = 128;
    constexpr int64_t N = 144;

    Tensor<float> A(Shape{M, K}, arena);
    Tensor<float> B(Shape{K, N}, arena);
    Tensor<float> C_naive(Shape{M, N}, arena);
    Tensor<float> C_packed(Shape{M, N}, arena);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            A(i, k) = dist(rng);
        }
    }
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t j = 0; j < N; ++j) {
            B(k, j) = dist(rng);
        }
    }

    // Reference naive GEMM
    gemm_naive(A.view(), B.view(), C_naive.view(), 1.0f, 0.0f);

    // GotoBLAS Micro-Panel Packed GEMM
    gemm_packed(A.view(), B.view(), C_packed.view(), 1.0f, 0.0f, &arena);

    // Assert exact numerical agreement
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            ASSERT_NEAR(C_packed(i, j), C_naive(i, j), 1e-4f);
        }
    }
}

KINESIS_TEST_CASE(GEMM_RectangularAndGramian) {
    ArenaAllocator arena(1024 * 1024);

    // Rectangular non-square matrix A of shape (3, 2)
    Tensor<float> A(Shape{3, 2}, arena);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;
    A(2, 0) = 5.0f; A(2, 1) = 6.0f;

    // Gramian G = A^T * A of shape (2, 2)
    Tensor<float> G(Shape{2, 2}, arena);
    gramian(A.view(), G.view());

    // Expected G:
    // [1*1 + 3*3 + 5*5,  1*2 + 3*4 + 5*6] = [35, 44]
    // [2*1 + 4*3 + 6*5,  2*2 + 4*4 + 6*6] = [44, 56]
    ASSERT_NEAR(G(0, 0), 35.0f, 1e-5f);
    ASSERT_NEAR(G(0, 1), 44.0f, 1e-5f);
    ASSERT_NEAR(G(1, 0), 44.0f, 1e-5f);
    ASSERT_NEAR(G(1, 1), 56.0f, 1e-5f);
}

KINESIS_TEST_CASE(Compute_CramerRuleSolvers) {
    // 2x2 system:
    // 2*x0 + 1*x1 = 5
    // 1*x0 + 3*x1 = 10
    // det = 6 - 1 = 5; x0 = (15 - 10)/5 = 1; x1 = (20 - 5)/5 = 3
    float x0 = 0.0f, x1 = 0.0f;
    bool ok2 = solve_cramer_2x2(2.0f, 1.0f, 1.0f, 3.0f, 5.0f, 10.0f, x0, x1);
    ASSERT_TRUE(ok2);
    ASSERT_NEAR(x0, 1.0f, 1e-5f);
    ASSERT_NEAR(x1, 3.0f, 1e-5f);

    // 3x3 system:
    // 1*x0 + 2*x1 + 3*x2 = 14
    // 0*x0 + 1*x1 + 4*x2 = 14
    // 5*x0 + 6*x1 + 0*x2 = 17
    // Expected: x0 = 1, x1 = 2, x2 = 3
    float A3[9] = {
        1.0f, 2.0f, 3.0f,
        0.0f, 1.0f, 4.0f,
        5.0f, 6.0f, 0.0f
    };
    float b3[3] = {14.0f, 14.0f, 17.0f};
    float x3[3] = {0.0f, 0.0f, 0.0f};
    bool ok3 = solve_cramer_3x3(A3, b3, x3);
    ASSERT_TRUE(ok3);
    ASSERT_NEAR(x3[0], 1.0f, 1e-4f);
    ASSERT_NEAR(x3[1], 2.0f, 1e-4f);
    ASSERT_NEAR(x3[2], 3.0f, 1e-4f);
}


