#pragma once

#include "kinesis/core/types.hpp"
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
    // AVX2 *and* FMA must be enabled explicitly: MSVC needs /arch:AVX2 (which defines __AVX2__
    // and implies FMA), GCC/Clang need -mavx2 -mfma (or -march=native). Keying this off
    // _MSC_VER alone emitted AVX2/FMA instructions into builds targeting CPUs without them,
    // which faults with an illegal instruction at runtime; and __AVX2__ alone does not imply
    // FMA, so _mm256_fmadd_ps could fail to compile.
    #if defined(__AVX2__) && (defined(_MSC_VER) || defined(__FMA__))
        #include <immintrin.h>
        #define KINESIS_HAS_AVX2 1
    #endif
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define KINESIS_HAS_NEON 1
#endif

namespace kinesis::compute::simd {

// SIMD Vector width for single precision float
#if defined(KINESIS_HAS_AVX2)
inline constexpr size_t kSimdFloatWidth = 8; // 256-bit AVX2 holds 8 floats
#elif defined(KINESIS_HAS_NEON)
inline constexpr size_t kSimdFloatWidth = 4; // 128-bit NEON holds 4 floats
#else
inline constexpr size_t kSimdFloatWidth = 4; // Scalar fallback batch size
#endif

// Vectorized Add: dst[i] = a[i] + b[i]
inline void add_f32(const float* a, const float* b, float* dst, size_t count) noexcept {
    size_t i = 0;
#if defined(KINESIS_HAS_AVX2)
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(dst + i, vr);
    }
#elif defined(KINESIS_HAS_NEON)
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(dst + i, vr);
    }
#endif
    // Scalar remainder
    for (; i < count; ++i) {
        dst[i] = a[i] + b[i];
    }
}

// Vectorized Sub: dst[i] = a[i] - b[i]
inline void sub_f32(const float* a, const float* b, float* dst, size_t count) noexcept {
    size_t i = 0;
#if defined(KINESIS_HAS_AVX2)
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(dst + i, vr);
    }
#elif defined(KINESIS_HAS_NEON)
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vr = vsubq_f32(va, vb);
        vst1q_f32(dst + i, vr);
    }
#endif
    for (; i < count; ++i) {
        dst[i] = a[i] - b[i];
    }
}

// Vectorized Mul: dst[i] = a[i] * b[i]
inline void mul_f32(const float* a, const float* b, float* dst, size_t count) noexcept {
    size_t i = 0;
#if defined(KINESIS_HAS_AVX2)
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(dst + i, vr);
    }
#elif defined(KINESIS_HAS_NEON)
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(dst + i, vr);
    }
#endif
    for (; i < count; ++i) {
        dst[i] = a[i] * b[i];
    }
}

// Vectorized Scale: dst[i] = a[i] * scalar
inline void scale_f32(const float* a, float scalar, float* dst, size_t count) noexcept {
    size_t i = 0;
#if defined(KINESIS_HAS_AVX2)
    __m256 vs = _mm256_set1_ps(scalar);
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(dst + i, vr);
    }
#elif defined(KINESIS_HAS_NEON)
    float32x4_t vs = vdupq_n_f32(scalar);
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vr = vmulq_f32(va, vs);
        vst1q_f32(dst + i, vr);
    }
#endif
    for (; i < count; ++i) {
        dst[i] = a[i] * scalar;
    }
}

// Vectorized ReLU: dst[i] = a[i] > 0 ? a[i] : 0
//
// NaN policy: a NaN input yields 0 on every path. That is what x86 `maxps` does (it returns its
// second operand when either input is NaN) and what the scalar tail does; the NEON path uses an
// explicit compare-and-select so it agrees, instead of propagating NaN as `vmaxq_f32` would.
// Note this differs from PyTorch, where relu(NaN) is NaN.
inline void relu_f32(const float* a, float* dst, size_t count) noexcept {
    size_t i = 0;
#if defined(KINESIS_HAS_AVX2)
    __m256 vzero = _mm256_setzero_ps();
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_max_ps(va, vzero);
        _mm256_storeu_ps(dst + i, vr);
    }
#elif defined(KINESIS_HAS_NEON)
    float32x4_t vzero = vdupq_n_f32(0.0f);
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vr = vbslq_f32(vcgtq_f32(va, vzero), va, vzero);
        vst1q_f32(dst + i, vr);
    }
#endif
    for (; i < count; ++i) {
        dst[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}

} // namespace kinesis::compute::simd
