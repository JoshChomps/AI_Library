#pragma once

#include "kinesis/core/allocator.hpp"
#include "kinesis/core/tensor_view.hpp"
#include <limits>
#include <new>
#include <utility>

namespace kinesis::core {

/**
 * @brief Owning Tensor backed by an Allocator (Heap or Arena).
 * 
 * Owns its underlying buffer and follows strict RAII.
 * Easily yields non-owning zero-copy TensorViews.
 */
template <typename T>
class Tensor {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;

    Tensor() noexcept : allocator_(&HeapAllocator::instance()), data_(nullptr), shape_{}, strides_{}, total_bytes_(0) {}

    /**
     * @brief Allocate tensor with shape using specified allocator.
     * @param shape Dimensions of the tensor.
     * @param allocator Reference to memory allocator (defaults to HeapAllocator).
     */
    explicit Tensor(Shape shape, Allocator& allocator = HeapAllocator::instance())
        : allocator_(&allocator), shape_(shape), strides_(Strides::from_shape_row_major(shape)) {
        allocate_storage();
    }

    /**
     * @brief Construct tensor with shape and fill value.
     */
    Tensor(Shape shape, const T& initial_val, Allocator& allocator = HeapAllocator::instance())
        : allocator_(&allocator), shape_(shape), strides_(Strides::from_shape_row_major(shape)) {
        allocate_storage();
        view().fill(initial_val);
    }

    ~Tensor() {
        deallocate_storage();
    }

    // Move semantics
    Tensor(Tensor&& other) noexcept
        : allocator_(other.allocator_),
          data_(other.data_),
          shape_(other.shape_),
          strides_(other.strides_),
          total_bytes_(other.total_bytes_) {
        other.data_ = nullptr;
        other.total_bytes_ = 0;
        other.shape_ = Shape{};
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            deallocate_storage();
            allocator_ = other.allocator_;
            data_ = other.data_;
            shape_ = other.shape_;
            strides_ = other.strides_;
            total_bytes_ = other.total_bytes_;

            other.data_ = nullptr;
            other.total_bytes_ = 0;
            other.shape_ = Shape{};
        }
        return *this;
    }

    // Explicit cloning instead of accidental copy
    [[nodiscard]] Tensor clone(Allocator& target_allocator) const {
        Tensor copy(shape_, target_allocator);
        copy.view().copy_from(view());
        return copy;
    }

    [[nodiscard]] Tensor clone() const {
        return clone(*allocator_);
    }

    // Disable implicit copy construction and copy assignment
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // View accessors.
    // Ref-qualified so a view can never be taken from a temporary: `TensorView<float> v =
    // Tensor<float>(shape);` would otherwise dangle the moment the temporary died.
    [[nodiscard]] TensorView<T> view() & noexcept {
        return TensorView<T>(data_, shape_, strides_, 0);
    }

    [[nodiscard]] TensorView<const T> view() const & noexcept {
        return TensorView<const T>(data_, shape_, strides_, 0);
    }

    TensorView<T> view() && = delete;

    // Implicit conversion to TensorView (lvalues only, for the same reason)
    operator TensorView<T>() & noexcept { return view(); }
    operator TensorView<const T>() const & noexcept { return view(); }

    // Transpose shortcuts
    [[nodiscard]] TensorView<T> t() & noexcept { return view().t(); }
    [[nodiscard]] TensorView<const T> t() const & noexcept { return view().t(); }
    [[nodiscard]] TensorView<T> operator~() & noexcept { return t(); }
    [[nodiscard]] TensorView<const T> operator~() const & noexcept { return t(); }

    /**
     * @brief Construct an Identity Matrix of size n x n.
     */
    [[nodiscard]] static Tensor eye(uint32_t n, Allocator& allocator = HeapAllocator::instance()) {
        Tensor result(Shape{n, n}, allocator);
        result.view().fill(static_cast<T>(0));
        for (uint32_t i = 0; i < n; ++i) {
            result(i, i) = static_cast<T>(1);
        }
        return result;
    }

    // Direct accessors
    [[nodiscard]] pointer data() noexcept { return data_; }
    [[nodiscard]] const_pointer data() const noexcept { return data_; }
    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] const Strides& strides() const noexcept { return strides_; }
    [[nodiscard]] size_t rank() const noexcept { return shape_.rank(); }
    [[nodiscard]] int64_t total_elements() const noexcept { return shape_.total_elements(); }
    [[nodiscard]] size_t total_bytes() const noexcept { return total_bytes_; }
    [[nodiscard]] Allocator& allocator() const noexcept { return *allocator_; }

    // Multi-dim element access
    template <typename... Args>
    [[nodiscard]] decltype(auto) operator()(Args... args) noexcept {
        return view()(args...);
    }

    template <typename... Args>
    [[nodiscard]] decltype(auto) operator()(Args... args) const noexcept {
        return view()(args...);
    }

private:
    void allocate_storage() {
        const int64_t count = shape_.total_elements();
        if (count <= 0) {
            data_ = nullptr;
            total_bytes_ = 0;
            return;
        }
        // Shape::total_elements() saturates at INT64_MAX when six 32-bit extents overflow, so
        // reject any element count whose byte size cannot be represented instead of wrapping to a
        // small allocation that later gets indexed out of bounds.
        if (static_cast<uint64_t>(count) > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        total_bytes_ = static_cast<size_t>(count) * sizeof(T);
        data_ = static_cast<pointer>(allocator_->allocate(total_bytes_, kDefaultAlignment));
        if (data_ == nullptr) {
            total_bytes_ = 0;
            throw std::bad_alloc();
        }
    }

    void deallocate_storage() noexcept {
        if (data_ && allocator_) {
            allocator_->deallocate(data_, total_bytes_, kDefaultAlignment);
            data_ = nullptr;
            total_bytes_ = 0;
        }
    }

    Allocator* allocator_{nullptr};
    pointer data_{nullptr};
    Shape shape_{};
    Strides strides_{};
    size_t total_bytes_{0};
};

} // namespace kinesis::core
