#pragma once

#include "kinesis/core/types.hpp"
#include <cmath>
#include <array>
#include <concepts>
#include <iostream>

namespace kinesis::autograd {

using core::FloatingPoint;

/**
 * @brief A type usable as the element of a Dual or Jet number.
 *
 * Deliberately structural rather than "is a floating-point type": nesting is how forward-mode
 * AD reaches second derivatives, and Dual<Dual<float>> must therefore be a legal element type.
 * Evaluating f on a Dual<Dual<T>> yields f'' in .dot.dot, which is what the FoMoH-style
 * second-order methods in the design notes need.
 */
template <typename T>
concept DualScalar = requires(T a, T b) {
    { -a } -> std::convertible_to<T>;
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
    { a < b } -> std::convertible_to<bool>;
    T{0};
    static_cast<T>(1);
};

/**
 * @brief Forward-Mode Automatic Differentiation Dual Number (x + x_dot * epsilon).
 * 
 * Satisfies the algebraic ring where epsilon != 0, but epsilon^2 == 0.
 * Evaluating any smooth function f(x + epsilon) yields:
 *   f(x + epsilon) = f(x) + f'(x) * epsilon
 * where `val` contains f(x) and `dot` contains the exact derivative f'(x).
 */
template <DualScalar T = float>
struct Dual {
    T val{0}; // Primal function value f(x)
    T dot{0}; // Tangent derivative value f'(x)

    // Constructors
    constexpr Dual() noexcept = default;
    constexpr Dual(T v) noexcept : val(v), dot(0) {}
    constexpr Dual(T v, T d) noexcept : val(v), dot(d) {}

    // Factory helper to introduce an independent variable with unit derivative
    [[nodiscard]] static constexpr Dual variable(T v) noexcept {
        return Dual(v, static_cast<T>(1));
    }

    [[nodiscard]] static constexpr Dual constant(T v) noexcept {
        return Dual(v, static_cast<T>(0));
    }

    // Unary Operators
    constexpr Dual operator+() const noexcept { return *this; }
    constexpr Dual operator-() const noexcept { return Dual(-val, -dot); }

    // Compound Assignment Operators with Dual
    constexpr Dual& operator+=(const Dual& rhs) noexcept {
        val += rhs.val;
        dot += rhs.dot;
        return *this;
    }

    constexpr Dual& operator-=(const Dual& rhs) noexcept {
        val -= rhs.val;
        dot -= rhs.dot;
        return *this;
    }

    constexpr Dual& operator*=(const Dual& rhs) noexcept {
        dot = dot * rhs.val + val * rhs.dot; // Product Rule: (u*v)' = u'*v + u*v'
        val *= rhs.val;
        return *this;
    }

    constexpr Dual& operator/=(const Dual& rhs) noexcept {
        // Quotient Rule: (u/v)' = (u'*v - u*v') / (v*v)
        dot = (dot * rhs.val - val * rhs.dot) / (rhs.val * rhs.val);
        val /= rhs.val;
        return *this;
    }

    // Compound Assignment with Scalar
    constexpr Dual& operator+=(T rhs) noexcept { val += rhs; return *this; }
    constexpr Dual& operator-=(T rhs) noexcept { val -= rhs; return *this; }
    constexpr Dual& operator*=(T rhs) noexcept { val *= rhs; dot *= rhs; return *this; }
    constexpr Dual& operator/=(T rhs) noexcept { val /= rhs; dot /= rhs; return *this; }

    // Binary Operators (Dual op Dual)
    friend constexpr Dual operator+(Dual lhs, const Dual& rhs) noexcept { lhs += rhs; return lhs; }
    friend constexpr Dual operator-(Dual lhs, const Dual& rhs) noexcept { lhs -= rhs; return lhs; }
    friend constexpr Dual operator*(Dual lhs, const Dual& rhs) noexcept { lhs *= rhs; return lhs; }
    friend constexpr Dual operator/(Dual lhs, const Dual& rhs) noexcept { lhs /= rhs; return lhs; }

    // Binary Operators with Scalar (Dual op Scalar)
    friend constexpr Dual operator+(Dual lhs, T rhs) noexcept { lhs += rhs; return lhs; }
    friend constexpr Dual operator-(Dual lhs, T rhs) noexcept { lhs -= rhs; return lhs; }
    friend constexpr Dual operator*(Dual lhs, T rhs) noexcept { lhs *= rhs; return lhs; }
    friend constexpr Dual operator/(Dual lhs, T rhs) noexcept { lhs /= rhs; return lhs; }

    // Binary Operators with Scalar (Scalar op Dual)
    friend constexpr Dual operator+(T lhs, const Dual& rhs) noexcept {
        return Dual(lhs + rhs.val, rhs.dot);
    }
    friend constexpr Dual operator-(T lhs, const Dual& rhs) noexcept {
        return Dual(lhs - rhs.val, -rhs.dot);
    }
    friend constexpr Dual operator*(T lhs, const Dual& rhs) noexcept {
        return Dual(lhs * rhs.val, lhs * rhs.dot);
    }
    friend constexpr Dual operator/(T lhs, const Dual& rhs) noexcept {
        return Dual(lhs / rhs.val, (-lhs * rhs.dot) / (rhs.val * rhs.val));
    }

    // Comparison Operators (comparing primal values)
    friend constexpr bool operator==(const Dual& lhs, const Dual& rhs) noexcept { return lhs.val == rhs.val; }
    friend constexpr bool operator!=(const Dual& lhs, const Dual& rhs) noexcept { return lhs.val != rhs.val; }
    friend constexpr bool operator<(const Dual& lhs, const Dual& rhs) noexcept { return lhs.val < rhs.val; }
    friend constexpr bool operator<=(const Dual& lhs, const Dual& rhs) noexcept { return lhs.val <= rhs.val; }
    friend constexpr bool operator>(const Dual& lhs, const Dual& rhs) noexcept { return lhs.val > rhs.val; }
    friend constexpr bool operator>=(const Dual& lhs, const Dual& rhs) noexcept { return lhs.val >= rhs.val; }

    friend constexpr bool operator==(const Dual& lhs, T rhs) noexcept { return lhs.val == rhs; }
    friend constexpr bool operator!=(const Dual& lhs, T rhs) noexcept { return lhs.val != rhs; }
    friend constexpr bool operator<(const Dual& lhs, T rhs) noexcept { return lhs.val < rhs; }
    friend constexpr bool operator<=(const Dual& lhs, T rhs) noexcept { return lhs.val <= rhs; }
    friend constexpr bool operator>(const Dual& lhs, T rhs) noexcept { return lhs.val > rhs; }
    friend constexpr bool operator>=(const Dual& lhs, T rhs) noexcept { return lhs.val >= rhs; }

    // Stream Output
    friend std::ostream& operator<<(std::ostream& os, const Dual& d) {
        os << d.val << " + " << d.dot << "ε";
        return os;
    }
};

// Elementary Transcendental Math Functions for Dual<T>
template <DualScalar T>
[[nodiscard]] inline Dual<T> sin(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::cos; using std::sin;
    return Dual<T>(sin(x.val), cos(x.val) * x.dot);
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> cos(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::cos; using std::sin;
    return Dual<T>(cos(x.val), -sin(x.val) * x.dot);
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> tan(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::tan;
    T t = tan(x.val);
    return Dual<T>(t, (static_cast<T>(1) + t * t) * x.dot);
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> exp(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::exp;
    T e = exp(x.val);
    return Dual<T>(e, e * x.dot);
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> log(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::log;
    return Dual<T>(log(x.val), x.dot / x.val);
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> sqrt(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::sqrt;
    T s = sqrt(x.val);
    // d/dx sqrt(x) is unbounded at x == 0. Propagating the infinity is deliberate: returning a
    // finite 0 would hide the singularity from a caller stepping an optimiser through it.
    return Dual<T>(s, x.dot / (static_cast<T>(2) * s));
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> pow(const Dual<T>& x, T exponent) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::pow;
    T val_pow = pow(x.val, exponent);
    // x^0 is the constant 1, so its derivative is 0. Without this case the general formula
    // evaluates 0 * pow(0, -1) = 0 * inf = NaN at x == 0, poisoning the whole tangent.
    if (exponent == static_cast<T>(0)) {
        return Dual<T>(val_pow, static_cast<T>(0));
    }
    T deriv = exponent * pow(x.val, exponent - static_cast<T>(1)) * x.dot;
    return Dual<T>(val_pow, deriv);
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> abs(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::abs;
    T sgn = (x.val > static_cast<T>(0)) ? static_cast<T>(1) : (x.val < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(0));
    return Dual<T>(abs(x.val), sgn * x.dot);
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> sinh(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::cosh; using std::sinh;
    return Dual<T>(sinh(x.val), cosh(x.val) * x.dot);
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> cosh(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::cosh; using std::sinh;
    return Dual<T>(cosh(x.val), sinh(x.val) * x.dot);
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> tanh(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::tanh;
    T th = tanh(x.val);
    return Dual<T>(th, (static_cast<T>(1) - th * th) * x.dot);
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> atan(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::atan;
    return Dual<T>(atan(x.val), x.dot / (static_cast<T>(1) + x.val * x.val));
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> sigmoid(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::exp;
    T s = static_cast<T>(1) / (static_cast<T>(1) + exp(-x.val));
    return Dual<T>(s, s * (static_cast<T>(1) - s) * x.dot);
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> relu(const Dual<T>& x) noexcept {
    return (x.val > static_cast<T>(0)) ? x : Dual<T>(static_cast<T>(0), static_cast<T>(0));
}

/**
 * @brief Value plus N first-order tangents: one forward pass yields the gradient with respect to
 * N independent variables (a robot arm's kinematic Jacobian across N joints, for instance).
 *
 * This is the type Ceres Solver calls `Jet<T, N>`, and the name is chosen to match it. It is
 * *not* a hyper-dual number: a hyper-dual carries an eps1*eps2 component and so yields second
 * derivatives, whereas every component here is first order. For second derivatives, nest instead
 * -- Dual<Dual<T>> puts f'' in .dot.dot, and Jet<Dual<T>, N> gives a Hessian-vector product.
 */
template <DualScalar T = float, size_t N = 6>
struct Jet {
    T val{0};
    std::array<T, N> grad{};

    constexpr Jet() noexcept {
        grad.fill(T{0});
    }

    constexpr Jet(T v) noexcept : val(v) {
        grad.fill(T{0});
    }

    constexpr Jet(T v, const std::array<T, N>& g) noexcept : val(v), grad(g) {}

    // Factory to initialize the i-th coordinate direction (basis vector e_i)
    [[nodiscard]] static constexpr Jet coordinate(T v, size_t index) noexcept {
        Jet hd(v);
        if (index < N) {
            hd.grad[index] = static_cast<T>(1);
        }
        return hd;
    }

    constexpr Jet operator+() const noexcept { return *this; }
    constexpr Jet operator-() const noexcept {
        Jet res(-val);
        for (size_t i = 0; i < N; ++i) res.grad[i] = -grad[i];
        return res;
    }

    constexpr Jet& operator+=(const Jet& rhs) noexcept {
        val += rhs.val;
        for (size_t i = 0; i < N; ++i) grad[i] += rhs.grad[i];
        return *this;
    }

    constexpr Jet& operator-=(const Jet& rhs) noexcept {
        val -= rhs.val;
        for (size_t i = 0; i < N; ++i) grad[i] -= rhs.grad[i];
        return *this;
    }

    constexpr Jet& operator*=(const Jet& rhs) noexcept {
        for (size_t i = 0; i < N; ++i) {
            grad[i] = grad[i] * rhs.val + val * rhs.grad[i];
        }
        val *= rhs.val;
        return *this;
    }

    constexpr Jet& operator/=(const Jet& rhs) noexcept {
        T denom = rhs.val * rhs.val;
        for (size_t i = 0; i < N; ++i) {
            grad[i] = (grad[i] * rhs.val - val * rhs.grad[i]) / denom;
        }
        val /= rhs.val;
        return *this;
    }

    // Scalar Operations
    constexpr Jet& operator+=(T rhs) noexcept { val += rhs; return *this; }
    constexpr Jet& operator-=(T rhs) noexcept { val -= rhs; return *this; }
    constexpr Jet& operator*=(T rhs) noexcept {
        val *= rhs;
        for (size_t i = 0; i < N; ++i) grad[i] *= rhs;
        return *this;
    }
    constexpr Jet& operator/=(T rhs) noexcept {
        val /= rhs;
        for (size_t i = 0; i < N; ++i) grad[i] /= rhs;
        return *this;
    }

    friend constexpr Jet operator+(Jet lhs, const Jet& rhs) noexcept { lhs += rhs; return lhs; }
    friend constexpr Jet operator-(Jet lhs, const Jet& rhs) noexcept { lhs -= rhs; return lhs; }
    friend constexpr Jet operator*(Jet lhs, const Jet& rhs) noexcept { lhs *= rhs; return lhs; }
    friend constexpr Jet operator/(Jet lhs, const Jet& rhs) noexcept { lhs /= rhs; return lhs; }

    friend constexpr Jet operator+(Jet lhs, T rhs) noexcept { lhs += rhs; return lhs; }
    friend constexpr Jet operator-(Jet lhs, T rhs) noexcept { lhs -= rhs; return lhs; }
    friend constexpr Jet operator*(Jet lhs, T rhs) noexcept { lhs *= rhs; return lhs; }
    friend constexpr Jet operator/(Jet lhs, T rhs) noexcept { lhs /= rhs; return lhs; }

    friend constexpr Jet operator+(T lhs, const Jet& rhs) noexcept {
        Jet res(lhs + rhs.val);
        res.grad = rhs.grad;
        return res;
    }
    friend constexpr Jet operator-(T lhs, const Jet& rhs) noexcept {
        Jet res(lhs - rhs.val);
        for (size_t i = 0; i < N; ++i) res.grad[i] = -rhs.grad[i];
        return res;
    }
    friend constexpr Jet operator*(T lhs, const Jet& rhs) noexcept {
        Jet res(lhs * rhs.val);
        for (size_t i = 0; i < N; ++i) res.grad[i] = lhs * rhs.grad[i];
        return res;
    }
    friend constexpr Jet operator/(T lhs, const Jet& rhs) noexcept {
        T denom = rhs.val * rhs.val;
        Jet res(lhs / rhs.val);
        for (size_t i = 0; i < N; ++i) res.grad[i] = (-lhs * rhs.grad[i]) / denom;
        return res;
    }
};

template <DualScalar T, size_t N>
[[nodiscard]] inline Jet<T, N> sin(const Jet<T, N>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::cos; using std::sin;
    T s = sin(x.val);
    T c = cos(x.val);
    Jet<T, N> res(s);
    for (size_t i = 0; i < N; ++i) res.grad[i] = c * x.grad[i];
    return res;
}

template <DualScalar T, size_t N>
[[nodiscard]] inline Jet<T, N> cos(const Jet<T, N>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::cos; using std::sin;
    T c = cos(x.val);
    T neg_s = -sin(x.val);
    Jet<T, N> res(c);
    for (size_t i = 0; i < N; ++i) res.grad[i] = neg_s * x.grad[i];
    return res;
}

template <DualScalar T, size_t N>
[[nodiscard]] inline Jet<T, N> exp(const Jet<T, N>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::exp;
    T e = exp(x.val);
    Jet<T, N> res(e);
    for (size_t i = 0; i < N; ++i) res.grad[i] = e * x.grad[i];
    return res;
}

template <DualScalar T, size_t N>
[[nodiscard]] inline Jet<T, N> log(const Jet<T, N>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::log;
    Jet<T, N> res(log(x.val));
    for (size_t i = 0; i < N; ++i) res.grad[i] = x.grad[i] / x.val;
    return res;
}

template <DualScalar T, size_t N>
[[nodiscard]] inline Jet<T, N> sqrt(const Jet<T, N>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::sqrt;
    T s = sqrt(x.val);
    T factor = static_cast<T>(2) * s;
    Jet<T, N> res(s);
    // As in Dual::sqrt, the tangent at x == 0 is genuinely unbounded; do not mask it with 0.
    for (size_t i = 0; i < N; ++i) res.grad[i] = x.grad[i] / factor;
    return res;
}

template <DualScalar T, size_t N>
[[nodiscard]] inline Jet<T, N> tanh(const Jet<T, N>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::tanh;
    T th = tanh(x.val);
    T deriv = static_cast<T>(1) - th * th;
    Jet<T, N> res(th);
    for (size_t i = 0; i < N; ++i) res.grad[i] = deriv * x.grad[i];
    return res;
}

template <DualScalar T, size_t N>
[[nodiscard]] inline Jet<T, N> sigmoid(const Jet<T, N>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::exp;
    T s = static_cast<T>(1) / (static_cast<T>(1) + exp(-x.val));
    T deriv = s * (static_cast<T>(1) - s);
    Jet<T, N> res(s);
    for (size_t i = 0; i < N; ++i) res.grad[i] = deriv * x.grad[i];
    return res;
}

template <DualScalar T, size_t N>
[[nodiscard]] inline Jet<T, N> relu(const Jet<T, N>& x) noexcept {
    if (x.val > static_cast<T>(0)) {
        return x;
    }
    return Jet<T, N>(static_cast<T>(0));
}

// Inverse trigonometric functions, needed for SO(3)/SE(3) logarithm maps and joint-limit terms.
template <DualScalar T>
[[nodiscard]] inline Dual<T> asin(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::asin; using std::sqrt;
    return Dual<T>(asin(x.val), x.dot / sqrt(static_cast<T>(1) - x.val * x.val));
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> acos(const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::acos; using std::sqrt;
    return Dual<T>(acos(x.val), -x.dot / sqrt(static_cast<T>(1) - x.val * x.val));
}

/**
 * @brief Two-argument arctangent, quadrant-correct in the primal and exact in the tangent.
 *
 * d atan2(y, x) = (x dy - y dx) / (x^2 + y^2), which stays finite everywhere except the origin,
 * unlike differentiating atan(y/x) -- that form loses the quadrant and blows up as x -> 0.
 */
template <DualScalar T>
[[nodiscard]] inline Dual<T> atan2(const Dual<T>& y, const Dual<T>& x) noexcept {
    // ADL two-step: a plain float resolves to <cmath>, a nested Dual to the overload below.
    using std::atan2;
    const T denom = x.val * x.val + y.val * y.val;
    return Dual<T>(atan2(y.val, x.val), (x.val * y.dot - y.val * x.dot) / denom);
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> min(const Dual<T>& a, const Dual<T>& b) noexcept {
    return (b.val < a.val) ? b : a;
}

template <DualScalar T>
[[nodiscard]] inline Dual<T> max(const Dual<T>& a, const Dual<T>& b) noexcept {
    return (a.val < b.val) ? b : a;
}

/**
 * @brief Backwards-compatible spelling of Jet.
 * @deprecated The type is a first-order multi-tangent number, not a hyper-dual; prefer Jet.
 */
template <DualScalar T = float, size_t N = 6>
using HyperDual = Jet<T, N>;

} // namespace kinesis::autograd
