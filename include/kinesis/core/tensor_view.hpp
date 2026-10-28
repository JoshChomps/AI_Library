#pragma once

#include "kinesis/core/shape.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <stdexcept>
#include <type_traits>

namespace kinesis::core {

/**
 * @brief Zero-copy, non-owning multi-dimensional Tensor View.
 *
 * Capable of wrapping raw camera frames from ROS 2 loaned buffers,
 * financial order book arrays, memory-mapped weights, or slices of memory
 * arenas with zero memory copies and complete control over strides and offsets.
 */
template <typename T> class TensorView {
public:
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;

  constexpr TensorView() noexcept
      : data_(nullptr), shape_{}, strides_{}, offset_(0) {}

  /**
   * @brief Construct a zero-copy contiguous view over an existing memory
   * buffer.
   */
  constexpr TensorView(pointer data, Shape shape) noexcept
      : data_(data), shape_(shape),
        strides_(Strides::from_shape_row_major(shape)), offset_(0) {}

  /**
   * @brief Construct a view with explicit custom strides and offset.
   */
  constexpr TensorView(pointer data, Shape shape, Strides strides,
                       int64_t offset = 0) noexcept
      : data_(data), shape_(shape), strides_(strides), offset_(offset) {}

  /**
   * @brief Implicit read-only conversion: TensorView<U> -> TensorView<const U>.
   *
   * Mirrors `std::span<T>` -> `std::span<const T>`. Without it, a `const Tensor`, a view over
   * memory-mapped weights, or anything else that is legitimately read-only could not be handed to
   * a kernel, and `Tensor::clone()` could not compile.
   */
  // Declared as a template so it is never treated as a copy constructor (which would suppress
  // the implicit one for non-const T).
  template <typename U>
    requires(std::is_const_v<T> && std::is_same_v<const U, T>)
  constexpr TensorView(const TensorView<U>& other) noexcept
      : data_(other.raw_base()), shape_(other.shape()), strides_(other.strides()),
        offset_(other.offset()) {}

  // Factory method for expressive borrowing
  [[nodiscard]] static constexpr TensorView
  from_borrowed(pointer data, Shape shape) noexcept {
    return TensorView(data, shape);
  }

  [[nodiscard]] static constexpr TensorView
  from_borrowed(pointer data, Shape shape, Strides strides,
                int64_t offset = 0) noexcept {
    return TensorView(data, shape, strides, offset);
  }

  // Accessors
  [[nodiscard]] constexpr pointer raw_base() noexcept { return data_; }
  [[nodiscard]] constexpr const_pointer raw_base() const noexcept {
    return data_;
  }

  [[nodiscard]] constexpr pointer data() noexcept { return data_ + offset_; }
  [[nodiscard]] constexpr const_pointer data() const noexcept {
    return data_ + offset_;
  }

  [[nodiscard]] constexpr const Shape &shape() const noexcept { return shape_; }
  [[nodiscard]] constexpr const Strides &strides() const noexcept {
    return strides_;
  }
  [[nodiscard]] constexpr int64_t offset() const noexcept { return offset_; }
  [[nodiscard]] constexpr size_t rank() const noexcept { return shape_.rank(); }
  [[nodiscard]] constexpr size_t ndim() const noexcept { return shape_.rank(); }
  [[nodiscard]] constexpr int64_t size(size_t dim) const noexcept {
    return shape_[dim];
  }
  [[nodiscard]] constexpr int64_t stride(size_t dim) const noexcept {
    return strides_[dim];
  }
  [[nodiscard]] constexpr int64_t total_elements() const noexcept {
    return shape_.total_elements();
  }
  [[nodiscard]] constexpr bool empty() const noexcept {
    return total_elements() == 0;
  }

  [[nodiscard]] constexpr bool is_contiguous() const noexcept {
    return is_c_contiguous(shape_, strides_);
  }

  // =========================================================================
  // Multi-Dimensional Element Accessors
  // =========================================================================

  // 1D Indexing
  [[nodiscard]] constexpr reference operator()(int64_t i) noexcept {
    assert(shape_.rank() == 1 && "TensorView must be 1D");
    assert(i >= 0 && i < shape_[0]);
    return data_[offset_ + i * strides_[0]];
  }

  [[nodiscard]] constexpr const_reference operator()(int64_t i) const noexcept {
    assert(shape_.rank() == 1 && "TensorView must be 1D");
    assert(i >= 0 && i < shape_[0]);
    return data_[offset_ + i * strides_[0]];
  }

  // 2D Indexing (Matrix)
  [[nodiscard]] constexpr reference operator()(int64_t i, int64_t j) noexcept {
    assert(shape_.rank() == 2 && "TensorView must be 2D");
    assert(i >= 0 && i < shape_[0] && j >= 0 && j < shape_[1]);
    return data_[offset_ + i * strides_[0] + j * strides_[1]];
  }

  [[nodiscard]] constexpr const_reference operator()(int64_t i,
                                                     int64_t j) const noexcept {
    assert(shape_.rank() == 2 && "TensorView must be 2D");
    assert(i >= 0 && i < shape_[0] && j >= 0 && j < shape_[1]);
    return data_[offset_ + i * strides_[0] + j * strides_[1]];
  }

  // 3D Indexing (Image: C, H, W or Batch, Seq, Feat)
  [[nodiscard]] constexpr reference operator()(int64_t i, int64_t j,
                                               int64_t k) noexcept {
    assert(shape_.rank() == 3 && "TensorView must be 3D");
    assert(i >= 0 && i < shape_[0] && j >= 0 && j < shape_[1] && k >= 0 &&
           k < shape_[2]);
    return data_[offset_ + i * strides_[0] + j * strides_[1] + k * strides_[2]];
  }

  [[nodiscard]] constexpr const_reference operator()(int64_t i, int64_t j,
                                                     int64_t k) const noexcept {
    assert(shape_.rank() == 3 && "TensorView must be 3D");
    assert(i >= 0 && i < shape_[0] && j >= 0 && j < shape_[1] && k >= 0 &&
           k < shape_[2]);
    return data_[offset_ + i * strides_[0] + j * strides_[1] + k * strides_[2]];
  }

  // 4D Indexing (Batch, Channel, Height, Width)
  [[nodiscard]] constexpr reference operator()(int64_t b, int64_t c, int64_t h,
                                               int64_t w) noexcept {
    assert(shape_.rank() == 4 && "TensorView must be 4D");
    return data_[offset_ + b * strides_[0] + c * strides_[1] + h * strides_[2] +
                 w * strides_[3]];
  }

  [[nodiscard]] constexpr const_reference
  operator()(int64_t b, int64_t c, int64_t h, int64_t w) const noexcept {
    assert(shape_.rank() == 4 && "TensorView must be 4D");
    return data_[offset_ + b * strides_[0] + c * strides_[1] + h * strides_[2] +
                 w * strides_[3]];
  }

  // =========================================================================
  // Zero-Copy Transformations
  // =========================================================================

  /**
   * @brief Zero-copy slice along specified dimension.
   * @param dim Dimension to slice along.
   * @param start Inclusive start index.
   * @param end Exclusive end index.
   */
  [[nodiscard]] TensorView slice(size_t dim, int64_t start, int64_t end) const {
    // Checked in every build: an out-of-range slice silently produces a view over memory the
    // tensor does not own, which is far more expensive to debug than this comparison.
    if (dim >= shape_.rank() || start < 0 || start > end ||
        end > static_cast<int64_t>(shape_[dim])) {
      throw std::out_of_range("TensorView::slice: dimension or bounds out of range");
    }

    Shape new_shape = shape_;
    new_shape[dim] = end - start;

    int64_t new_offset = offset_ + start * strides_[dim];
    return TensorView(data_, new_shape, strides_, new_offset);
  }

  /**
   * @brief Zero-copy transpose swapping two dimensions.
   */
  [[nodiscard]] TensorView transpose(size_t dim0, size_t dim1) const {
    // Checked in every build. Shape::operator[] is assert-only (it is a per-element accessor),
    // so an index of 6 or more would swap past the end of the dimension array on the stack, and
    // one between rank and 6 would silently swap a live dimension with an unused slot.
    if (dim0 >= shape_.rank() || dim1 >= shape_.rank()) {
      throw std::out_of_range("TensorView::transpose: dimension out of range");
    }

    Shape new_shape = shape_;
    Strides new_strides = strides_;

    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    return TensorView(data_, new_shape, new_strides, offset_);
  }

  /**
   * @brief Zero-copy 2D matrix transpose shortcut.
   */
  [[nodiscard]] TensorView t() const {
    if (shape_.rank() != 2) {
      throw std::invalid_argument("TensorView::t: only valid for 2D matrices");
    }
    return transpose(0, 1);
  }

  /**
   * @brief Conjugate/transpose operator shortcut: ~view
   */
  [[nodiscard]] TensorView operator~() const {
    return t();
  }

  /**
   * @brief Fills a square 2D matrix view as the Identity matrix (I).
   */
  void make_identity() {
    // Checked in every build, which is why this is no longer noexcept. On a non-square view the
    // diagonal loop below runs to shape_[0] and writes (i, i) past the last column: a 4x2 view
    // would write element (3, 3) at offset 9 of an 8-element buffer.
    if (shape_.rank() != 2 || shape_[0] != shape_[1]) {
      throw std::invalid_argument("TensorView::make_identity: requires a square 2D view");
    }
    fill(static_cast<T>(0));
    const size_t n = shape_[0];
    for (size_t i = 0; i < n; ++i) {
      (*this)(i, i) = static_cast<T>(1);
    }
  }

  /**
   * @brief Zero-copy broadcasting to a target shape.
   * Expands any dimension of size 1 by setting its stride to 0.
   * Uses loop fission to eliminate branch stalls in hot path.
   */
  [[nodiscard]] TensorView broadcast_to(const Shape &target_shape) const {
    // Checked in every build: if the target rank were lower, `rank_diff` would wrap around and
    // the loop below would write past the stride array.
    if (target_shape.rank() < shape_.rank()) {
      throw std::invalid_argument("TensorView::broadcast_to: target rank is lower than source rank");
    }

    const size_t target_rank = target_shape.rank();
    const size_t src_rank = shape_.rank();
    const size_t rank_diff = target_rank - src_rank;
    Strides new_strides(target_rank);

    // Loop 1 (Loop Fission): Leading broadcast dimensions (100% unconditional zero-fill)
    for (size_t i = 0; i < rank_diff; ++i) {
      new_strides[i] = 0;
    }

    // Loop 2 (Loop Fission): Trailing source dimensions (100% unconditional mapping)
    for (size_t src_idx = 0; src_idx < src_rank; ++src_idx) {
      const size_t dst_idx = src_idx + rank_diff;
      if (shape_[src_idx] == target_shape[dst_idx]) {
        new_strides[dst_idx] = strides_[src_idx];
      } else if (shape_[src_idx] == 1) {
        new_strides[dst_idx] = 0; // Broadcast stride
      } else {
        throw std::invalid_argument("Incompatible broadcast shapes");
      }
    }
    return TensorView(data_, target_shape, new_strides, offset_);
  }

  /**
   * @brief Reshape view into new dimensions without memory copying.
   * Only valid if the tensor is C-contiguous.
   */
  [[nodiscard]] TensorView reshape(const Shape &new_shape) const {
    if (!is_contiguous()) {
      throw std::runtime_error(
          "Cannot reshape non-contiguous TensorView without copying");
    }
    if (new_shape.total_elements() != total_elements()) {
      throw std::invalid_argument("Total element mismatch in reshape");
    }
    return TensorView(data_, new_shape,
                      Strides::from_shape_row_major(new_shape), offset_);
  }

  // =========================================================================
  // Value Modifiers & Data Movement
  // =========================================================================

  /**
   * @brief Fill tensor with a constant scalar value.
   */
  void fill(const T &value) {
    if (is_contiguous()) {
      std::fill_n(data(), total_elements(), value);
      return;
    }
    // Non-contiguous iteration fallback
    iterate_indices(
        [this, &value](const int64_t *idx) { at_indices(idx) = value; });
  }

  /**
   * @brief Copy elements from another tensor view into this one.
   */
  void copy_from(const TensorView<const T> &src) {
    // Checked in every build. With mismatched shapes the contiguous path below reads
    // total_elements() of *this* from src -- past the end of a smaller source -- and the strided
    // path walks *src's* indices while writing into *this*, past the end of a smaller
    // destination.
    if (!(shape_ == src.shape())) {
      throw std::invalid_argument("TensorView::copy_from: shape mismatch");
    }
    if (is_contiguous() && src.is_contiguous()) {
      std::copy_n(src.data(), total_elements(), data());
      return;
    }
    // General non-contiguous copy
    src.iterate_indices([this, &src](const int64_t *idx) {
      at_indices(idx) = src.at_indices(idx);
    });
  }

  // Internal helper for indexing with an arbitrary coordinate array
  [[nodiscard]] T &at_indices(const int64_t *idx) noexcept {
    int64_t flat = offset_;
    for (size_t d = 0; d < shape_.rank(); ++d) {
      flat += idx[d] * strides_[d];
    }
    return data_[flat];
  }

  [[nodiscard]] const T &at_indices(const int64_t *idx) const noexcept {
    int64_t flat = offset_;
    for (size_t d = 0; d < shape_.rank(); ++d) {
      flat += idx[d] * strides_[d];
    }
    return data_[flat];
  }

  template <typename Func> void iterate_indices(Func &&f) const {
    if (shape_.rank() == 0)
      return;
    std::array<int64_t, kMaxDimensions> current_idx{};
    iterate_recursive(0, current_idx.data(), f);
  }

private:
  template <typename Func>
  void iterate_recursive(size_t dim, int64_t *current_idx, Func &&f) const {
    if (dim == shape_.rank()) {
      f(current_idx);
      return;
    }
    for (int64_t i = 0; i < shape_[dim]; ++i) {
      current_idx[dim] = i;
      iterate_recursive(dim + 1, current_idx, f);
    }
  }

  pointer data_{nullptr};
  Shape shape_{};
  Strides strides_{};
  int64_t offset_{0};
};

/**
 * @brief Free function helper for tensor transpose.
 */
template <typename T>
[[nodiscard]] inline TensorView<T> transpose(const TensorView<T>& view, size_t dim0 = 0, size_t dim1 = 1) {
  return view.transpose(dim0, dim1);
}

} // namespace kinesis::core
