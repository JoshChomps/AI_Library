#pragma once

#include "kinesis/core/types.hpp"
#include <array>
#include <cassert>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <sstream>

namespace kinesis::core {

inline constexpr size_t kMaxDimensions = 6;

/// Largest extent a single dimension can hold (dimensions are stored as `Dim` = uint32_t).
inline constexpr int64_t kMaxDimensionExtent =
    static_cast<int64_t>(std::numeric_limits<uint32_t>::max());

namespace detail {

/// Validation that stays enabled in release builds: a silently truncated dimension or an
/// out-of-range rank corrupts every downstream size computation, so these are real checks
/// rather than asserts.
[[noreturn]] inline void throw_shape_error(const char* what) { throw std::invalid_argument(what); }

constexpr void check_rank(size_t rank) {
    if (rank > kMaxDimensions) {
        throw_shape_error("Shape/Strides rank exceeds kMaxDimensions (6)");
    }
}

constexpr void check_extent(int64_t dim) {
    if (dim < 0) {
        throw_shape_error("Shape dimension must be non-negative");
    }
    if (dim > kMaxDimensionExtent) {
        throw_shape_error("Shape dimension exceeds the 32-bit extent limit (see docs/DESIGN_REVIEW_2026-09-17.md T5)");
    }
}

} // namespace detail

/**
 * @brief Non-allocating, stack-resident dimension shape descriptor.
 * Completely eliminates heap allocation overhead for tensor metadata.
 */
class Shape {
public:
    constexpr Shape() noexcept : rank_(0), dims_{} {}

    constexpr explicit Shape(size_t rank) : rank_(static_cast<uint32_t>(rank)), dims_{} {
        detail::check_rank(rank);
    }

    constexpr Shape(std::initializer_list<int64_t> dims) : rank_(static_cast<uint32_t>(dims.size())) {
        detail::check_rank(dims.size());
        size_t i = 0;
        for (int64_t d : dims) {
            detail::check_extent(d);
            dims_[i++] = static_cast<uint32_t>(d);
        }
    }

    constexpr explicit Shape(std::span<const int64_t> dims) : rank_(static_cast<uint32_t>(dims.size())) {
        detail::check_rank(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            detail::check_extent(dims[i]);
            dims_[i] = static_cast<uint32_t>(dims[i]);
        }
    }

    constexpr explicit Shape(std::span<const uint32_t> dims) : rank_(static_cast<uint32_t>(dims.size())) {
        detail::check_rank(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            dims_[i] = dims[i];
        }
    }

    [[nodiscard]] constexpr size_t rank() const noexcept { return rank_; }
    [[nodiscard]] constexpr size_t ndim() const noexcept { return rank_; }

    [[nodiscard]] constexpr uint32_t operator[](size_t index) const noexcept {
        assert(index < rank_ && "Dimension index out of range");
        return dims_[index];
    }

    [[nodiscard]] constexpr uint32_t& operator[](size_t index) noexcept {
        assert(index < rank_ && "Dimension index out of range");
        return dims_[index];
    }

    /**
     * @brief Number of elements described by this shape.
     *
     * Six 32-bit extents can multiply past the range of int64_t, so the product is computed with
     * an overflow check and saturates at INT64_MAX rather than wrapping to a small (dangerously
     * plausible) value. Allocation paths must therefore treat INT64_MAX as "too large" — see
     * Tensor::allocate_storage.
     */
    [[nodiscard]] constexpr int64_t total_elements() const noexcept {
        if (rank_ == 0) return 0;
        int64_t total = 1;
        const size_t len = rank_;
        for (size_t i = 0; i < len; ++i) {
            const int64_t dim = static_cast<int64_t>(dims_[i]);
            if (dim == 0) return 0;
            if (total > std::numeric_limits<int64_t>::max() / dim) {
                return std::numeric_limits<int64_t>::max();
            }
            total *= dim;
        }
        return total;
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return total_elements() == 0;
    }

    [[nodiscard]] constexpr const uint32_t* data() const noexcept { return dims_.data(); }

    /**
     * @brief Whole-descriptor equality.
     *
     * `std::array::operator==` compares all six extents with no branch per dimension, which the
     * compiler lowers to the same handful of word comparisons as a hand-written bitwise version —
     * without reading a uint32_t array through a uint64_t pointer (strict-aliasing and alignment
     * undefined behaviour). Unused slots stay zero because every constructor value-initializes
     * `dims_` and the extent setters are bounds-checked.
     */
    constexpr bool operator==(const Shape& other) const noexcept {
        return rank_ == other.rank_ && dims_ == other.dims_;
    }

    constexpr bool operator!=(const Shape& other) const noexcept {
        return !(*this == other);
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "(";
        const size_t len = rank_;
        for (size_t i = 0; i < len; ++i) {
            oss << dims_[i] << (i + 1 < len ? ", " : "");
        }
        oss << ")";
        return oss.str();
    }

private:
    uint32_t rank_{0};
    std::array<uint32_t, kMaxDimensions> dims_{};
};

/**
 * @brief Non-allocating stack-resident stride descriptor.
 */
class Strides {
public:
    constexpr Strides() noexcept : rank_(0), strides_{} {}

    constexpr explicit Strides(size_t rank) : rank_(rank), strides_{} {
        detail::check_rank(rank);
    }

    constexpr Strides(std::initializer_list<int64_t> strides) : rank_(strides.size()) {
        detail::check_rank(strides.size());
        size_t i = 0;
        for (int64_t s : strides) {
            strides_[i++] = s;
        }
    }

    constexpr explicit Strides(std::span<const int64_t> strides) : rank_(strides.size()) {
        detail::check_rank(strides.size());
        for (size_t i = 0; i < strides.size(); ++i) {
            strides_[i] = strides[i];
        }
    }

    [[nodiscard]] constexpr size_t rank() const noexcept { return rank_; }

    [[nodiscard]] constexpr int64_t operator[](size_t index) const noexcept {
        assert(index < rank_);
        return strides_[index];
    }

    [[nodiscard]] constexpr int64_t& operator[](size_t index) noexcept {
        assert(index < rank_);
        return strides_[index];
    }

    [[nodiscard]] constexpr const int64_t* data() const noexcept { return strides_.data(); }

    /// Whole-descriptor equality; see the note on Shape::operator== for why this is not a
    /// hand-rolled word comparison.
    constexpr bool operator==(const Strides& other) const noexcept {
        return rank_ == other.rank_ && strides_ == other.strides_;
    }

    constexpr bool operator!=(const Strides& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief Compute canonical C-contiguous (Row-Major) strides for a given shape.
     */
    [[nodiscard]] static constexpr Strides from_shape_row_major(const Shape& shape) noexcept {
        Strides s;
        s.rank_ = shape.rank();
        if (s.rank_ == 0) return s;

        int64_t acc = 1;
        for (size_t i = s.rank_; i > 0; --i) {
            s.strides_[i - 1] = acc;
            acc *= shape[i - 1];
        }
        return s;
    }

    /**
     * @brief Compute canonical Fortran-contiguous (Col-Major) strides for a given shape.
     */
    [[nodiscard]] static constexpr Strides from_shape_col_major(const Shape& shape) noexcept {
        Strides s;
        s.rank_ = shape.rank();
        if (s.rank_ == 0) return s;

        int64_t acc = 1;
        for (size_t i = 0; i < s.rank_; ++i) {
            s.strides_[i] = acc;
            acc *= shape[i];
        }
        return s;
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "(";
        for (size_t i = 0; i < rank_; ++i) {
            oss << strides_[i] << (i + 1 < rank_ ? ", " : "");
        }
        oss << ")";
        return oss.str();
    }

private:
    size_t rank_{0};
    std::array<int64_t, kMaxDimensions> strides_{};
};

/**
 * @brief Check if a shape and strides describe a C-contiguous memory layout.
 */
[[nodiscard]] constexpr bool is_c_contiguous(const Shape& shape, const Strides& strides) noexcept {
    if (shape.rank() == 0) return true;
    int64_t expected_stride = 1;
    for (size_t i = shape.rank(); i > 0; --i) {
        if (shape[i - 1] > 1 && strides[i - 1] != expected_stride) {
            return false;
        }
        expected_stride *= shape[i - 1];
    }
    return true;
}

} // namespace kinesis::core
