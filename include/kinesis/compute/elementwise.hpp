#pragma once

#include "kinesis/compute/simd.hpp"
#include "kinesis/core/tensor_view.hpp"
#include <cmath>

namespace kinesis::compute {

/**
 * @brief Element-wise addition: out = a + b
 */
inline void add(const core::TensorView<const float>& a,
                const core::TensorView<const float>& b,
                core::TensorView<float> out) {
    assert(a.shape() == b.shape() && a.shape() == out.shape());
    if (a.is_contiguous() && b.is_contiguous() && out.is_contiguous()) {
        simd::add_f32(a.data(), b.data(), out.data(), static_cast<size_t>(out.total_elements()));
        return;
    }
    // General non-contiguous iteration
    out.iterate_indices([&](const int64_t* idx) {
        out.at_indices(idx) = a.at_indices(idx) + b.at_indices(idx);
    });
}

/**
 * @brief Element-wise subtraction: out = a - b
 */
inline void sub(const core::TensorView<const float>& a,
                const core::TensorView<const float>& b,
                core::TensorView<float> out) {
    assert(a.shape() == b.shape() && a.shape() == out.shape());
    if (a.is_contiguous() && b.is_contiguous() && out.is_contiguous()) {
        simd::sub_f32(a.data(), b.data(), out.data(), static_cast<size_t>(out.total_elements()));
        return;
    }
    out.iterate_indices([&](const int64_t* idx) {
        out.at_indices(idx) = a.at_indices(idx) - b.at_indices(idx);
    });
}

/**
 * @brief Element-wise multiplication (Hadamard product): out = a * b
 */
inline void mul(const core::TensorView<const float>& a,
                const core::TensorView<const float>& b,
                core::TensorView<float> out) {
    assert(a.shape() == b.shape() && a.shape() == out.shape());
    if (a.is_contiguous() && b.is_contiguous() && out.is_contiguous()) {
        simd::mul_f32(a.data(), b.data(), out.data(), static_cast<size_t>(out.total_elements()));
        return;
    }
    out.iterate_indices([&](const int64_t* idx) {
        out.at_indices(idx) = a.at_indices(idx) * b.at_indices(idx);
    });
}

/**
 * @brief Scalar scaling: out = a * scalar
 */
inline void scale(const core::TensorView<const float>& a,
                  float scalar,
                  core::TensorView<float> out) {
    assert(a.shape() == out.shape());
    if (a.is_contiguous() && out.is_contiguous()) {
        simd::scale_f32(a.data(), scalar, out.data(), static_cast<size_t>(out.total_elements()));
        return;
    }
    out.iterate_indices([&](const int64_t* idx) {
        out.at_indices(idx) = a.at_indices(idx) * scalar;
    });
}

/**
 * @brief Rectified Linear Unit (ReLU): out = max(0, a)
 */
inline void relu(const core::TensorView<const float>& a,
                 core::TensorView<float> out) {
    assert(a.shape() == out.shape());
    if (a.is_contiguous() && out.is_contiguous()) {
        simd::relu_f32(a.data(), out.data(), static_cast<size_t>(out.total_elements()));
        return;
    }
    out.iterate_indices([&](const int64_t* idx) {
        float val = a.at_indices(idx);
        out.at_indices(idx) = val > 0.0f ? val : 0.0f;
    });
}

/**
 * @brief Sigmoid activation: out = 1 / (1 + exp(-a))
 */
inline void sigmoid(const core::TensorView<const float>& a,
                    core::TensorView<float> out) {
    assert(a.shape() == out.shape());
    if (a.is_contiguous() && out.is_contiguous()) {
        // Flat loop over contiguous memory. The exponential itself is a scalar libm call, but
        // this avoids the per-element recursive index walk used by the general path.
        const float* src = a.data();
        float* dst = out.data();
        const size_t n = static_cast<size_t>(out.total_elements());
        for (size_t i = 0; i < n; ++i) {
            dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
        }
        return;
    }
    out.iterate_indices([&](const int64_t* idx) {
        float val = a.at_indices(idx);
        out.at_indices(idx) = 1.0f / (1.0f + std::exp(-val));
    });
}

/**
 * @brief Gaussian Error Linear Unit (GELU), tanh approximation:
 * out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * This is the "gelu_new" form used by GPT-2 and Hugging Face Transformers; its maximum error
 * against the exact erf definition is ~4.7e-4 over [-6, 6]. Frameworks that default to the exact
 * form (PyTorch) will differ by that much, which matters when porting weights between stacks.
 */
inline void gelu(const core::TensorView<const float>& a,
                 core::TensorView<float> out) {
    assert(a.shape() == out.shape());
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kCoeff = 0.044715f;

    const auto gelu_scalar = [](float x) {
        const float x3 = x * x * x;
        const float inner = kSqrt2OverPi * (x + kCoeff * x3);
        return 0.5f * x * (1.0f + std::tanh(inner));
    };

    if (a.is_contiguous() && out.is_contiguous()) {
        const float* src = a.data();
        float* dst = out.data();
        const size_t n = static_cast<size_t>(out.total_elements());
        for (size_t i = 0; i < n; ++i) {
            dst[i] = gelu_scalar(src[i]);
        }
        return;
    }
    out.iterate_indices([&](const int64_t* idx) {
        out.at_indices(idx) = gelu_scalar(a.at_indices(idx));
    });
}

/**
 * @brief Exact GELU: out = 0.5 * x * (1 + erf(x / sqrt(2))).
 *
 * Matches PyTorch's default GELU. Slower than the tanh approximation above, but the right choice
 * when numerical agreement with another framework matters.
 */
inline void gelu_exact(const core::TensorView<const float>& a,
                       core::TensorView<float> out) {
    assert(a.shape() == out.shape());
    constexpr float kInvSqrt2 = 0.70710678118654752440f;

    const auto gelu_scalar = [](float x) {
        return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
    };

    if (a.is_contiguous() && out.is_contiguous()) {
        const float* src = a.data();
        float* dst = out.data();
        const size_t n = static_cast<size_t>(out.total_elements());
        for (size_t i = 0; i < n; ++i) {
            dst[i] = gelu_scalar(src[i]);
        }
        return;
    }
    out.iterate_indices([&](const int64_t* idx) {
        out.at_indices(idx) = gelu_scalar(a.at_indices(idx));
    });
}

} // namespace kinesis::compute
