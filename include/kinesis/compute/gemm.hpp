#pragma once

#include "kinesis/core/tensor_view.hpp"
#include <cmath>

namespace kinesis::core {
class Allocator;
}

namespace kinesis::compute {

/**
 * @brief General Matrix Multiplication (GEMM)
 * Computes: C = alpha * (A @ B) + beta * C
 *
 * Inputs are read-only views, so `const` tensors and memory-mapped weights can be passed
 * directly; a `TensorView<float>` converts implicitly.
 *
 * A, B and C may use any strides (transposed views included). C must not alias A or B.
 *
 * @param a Matrix A of shape (M, K)
 * @param b Matrix B of shape (K, N)
 * @param c Matrix C of shape (M, N) to store output
 * @param alpha Scalar multiplier for A @ B (default 1.0f)
 * @param beta Scalar multiplier for existing C (default 0.0f)
 */
void gemm(const core::TensorView<const float>& a,
          const core::TensorView<const float>& b,
          core::TensorView<float> c,
          float alpha = 1.0f,
          float beta = 0.0f);

/**
 * @brief Naive reference GEMM for mathematical verification and unit testing.
 */
void gemm_naive(const core::TensorView<const float>& a,
                const core::TensorView<const float>& b,
                core::TensorView<float> c,
                float alpha = 1.0f,
                float beta = 0.0f);

/**
 * @brief Cache-blocked single-threaded GEMM (SIMD inner loop, stride-aware fallback).
 */
void gemm_blocked(const core::TensorView<const float>& a,
                  const core::TensorView<const float>& b,
                  core::TensorView<float> c,
                  float alpha = 1.0f,
                  float beta = 0.0f);


/**
 * @brief Panel-packed GEMM: packs matrix B into contiguous micro-panels to keep the inner loop
 * on unit-stride memory.
 *
 * The packed inner loop requires unit column stride on A and C; other layouts are dispatched to
 * gemm_blocked, which reads through the strided accessors. When `scratch` is null the packing
 * buffer comes from a thread-local block rather than the heap, so the hot path does not allocate.
 */
void gemm_packed(const core::TensorView<const float>& a,
                 const core::TensorView<const float>& b,
                 core::TensorView<float> c,
                 float alpha = 1.0f,
                 float beta = 0.0f,
                 core::Allocator* scratch = nullptr);

/**
 * @brief Computes the Gramian matrix G = A^T * A.
 * Normalizes an M x N rectangular matrix into an N x N symmetric positive semi-definite matrix.
 * Essential for the Moore-Penrose Pseudoinverse (A^+ = (A^T A)^{-1} A^T) and least-squares regression.
 * 
 * @param a Input matrix A of shape (M, N)
 * @param g Output Gramian matrix G of shape (N, N)
 */
inline void gramian(const core::TensorView<const float>& a,
                    core::TensorView<float> g) {
    assert(a.rank() == 2 && g.rank() == 2 && "Gramian inputs must be 2D matrices");
    assert(g.size(0) == a.size(1) && g.size(1) == a.size(1) &&
           "Gramian G must be (N, N) for input A of shape (M, N)");
    gemm(a.t(), a, g, 1.0f, 0.0f);
}

/**
 * @brief Solves a 2x2 linear system A * x = b with the closed-form Cramer's rule.
 * Ideal for real-time 2D planar kinematics and ray-intersection tests.
 *
 * The singularity test is scale-invariant: the determinant is compared against the magnitude of
 * the products it was formed from, so a well-conditioned system expressed in small units
 * (A = 1e-7 * I) is solved rather than rejected, and a near-singular system in large units is
 * rejected rather than "solved" into garbage. Returns false when the system is singular to
 * within `relative_epsilon`.
 */
inline bool solve_cramer_2x2(float a00, float a01,
                             float a10, float a11,
                             float b0,  float b1,
                             float& x0, float& x1,
                             float relative_epsilon = 1e-6f) noexcept {
    float det = a00 * a11 - a01 * a10;
    const float scale = std::fabs(a00 * a11) + std::fabs(a01 * a10);
    if (det == 0.0f || std::fabs(det) <= relative_epsilon * scale) return false;
    float inv_det = 1.0f / det;
    x0 = (b0 * a11 - a01 * b1) * inv_det;
    x1 = (a00 * b1 - b0 * a10) * inv_det;
    return true;
}

/**
 * @brief Solves a 3x3 linear system A * x = b with the closed-form Cramer's rule.
 * Ideal for robotics SE(3) spatial rotations, camera calibration, and 3D kinematics.
 * Input A is given as 9 row-major floats, b as 3 floats, output x as 3 floats.
 *
 * Uses the same scale-invariant singularity test as solve_cramer_2x2. For systems larger than
 * 3x3, or ill-conditioned ones, prefer a QR or LU solve: Cramer's rule is not backward stable.
 */
inline bool solve_cramer_3x3(const float A[9], const float b[3], float x[3],
                             float relative_epsilon = 1e-6f) noexcept {
    float c0 = A[4] * A[8] - A[5] * A[7];
    float c1 = A[5] * A[6] - A[3] * A[8];
    float c2 = A[3] * A[7] - A[4] * A[6];
    float det = A[0] * c0 + A[1] * c1 + A[2] * c2;
    const float scale = std::fabs(A[0] * c0) + std::fabs(A[1] * c1) + std::fabs(A[2] * c2);
    if (det == 0.0f || std::fabs(det) <= relative_epsilon * scale) return false;
    float inv_det = 1.0f / det;

    x[0] = (b[0] * (A[4] * A[8] - A[5] * A[7]) +
            b[1] * (A[2] * A[7] - A[1] * A[8]) +
            b[2] * (A[1] * A[5] - A[2] * A[4])) * inv_det;

    x[1] = (A[0] * (b[1] * A[8] - b[2] * A[5]) +
            A[3] * (b[2] * A[2] - b[0] * A[8]) +
            A[6] * (b[0] * A[5] - b[1] * A[2])) * inv_det;

    x[2] = (A[0] * (A[4] * b[2] - A[7] * b[1]) +
            A[3] * (A[7] * b[0] - A[1] * b[2]) +
            A[6] * (A[1] * b[1] - A[4] * b[0])) * inv_det;
    return true;
}

} // namespace kinesis::compute

