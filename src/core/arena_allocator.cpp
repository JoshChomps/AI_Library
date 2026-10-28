#include "kinesis/core/arena_allocator.hpp"
#include <stdexcept>
#include <utility>

namespace kinesis::core {

ArenaAllocator::ArenaAllocator(size_t capacityBytes)
    : capacity_(capacityBytes), owns_buffer_(true) {
    if (capacityBytes > 0) {
        // An arena is one large, long-lived region, which is exactly the case where going
        // straight to OS pages pays off: no CRT bookkeeping, page-aligned, and the
        // page-rounding waste is amortised over the whole arena rather than per tensor.
        buffer_ = static_cast<std::byte*>(VirtualMemoryAllocator::instance().allocate(capacityBytes, kDefaultAlignment));
    }
}

ArenaAllocator::ArenaAllocator(std::span<std::byte> externalBuffer) noexcept
    : buffer_(externalBuffer.data()),
      capacity_(externalBuffer.size()),
      current_offset_(0),
      peak_offset_(0),
      owns_buffer_(false) {
}

ArenaAllocator::~ArenaAllocator() {
    if (owns_buffer_ && buffer_) {
        VirtualMemoryAllocator::instance().deallocate(buffer_, capacity_, kDefaultAlignment);
        buffer_ = nullptr;
    }
}

ArenaAllocator::ArenaAllocator(ArenaAllocator&& other) noexcept
    : buffer_(other.buffer_),
      capacity_(other.capacity_),
      current_offset_(other.current_offset_),
      peak_offset_(other.peak_offset_),
      owns_buffer_(other.owns_buffer_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.current_offset_ = 0;
    other.peak_offset_ = 0;
    other.owns_buffer_ = false;
}

ArenaAllocator& ArenaAllocator::operator=(ArenaAllocator&& other) noexcept {
    if (this != &other) {
        if (owns_buffer_ && buffer_) {
            VirtualMemoryAllocator::instance().deallocate(buffer_, capacity_, kDefaultAlignment);
        }
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        current_offset_ = other.current_offset_;
        peak_offset_ = other.peak_offset_;
        owns_buffer_ = other.owns_buffer_;

        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.current_offset_ = 0;
        other.peak_offset_ = 0;
        other.owns_buffer_ = false;
    }
    return *this;
}

void* ArenaAllocator::allocate(size_t bytes, size_t alignment) {
    if (bytes == 0) {
        return nullptr;
    }

    // Checked in every build, not just debug: alignment 0 would make `alignment - 1` SIZE_MAX
    // below, and a non-power-of-two mask would hand back a misaligned pointer that the SIMD
    // kernels then use.
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("ArenaAllocator::allocate: alignment must be a power of two");
    }
    if (buffer_ == nullptr) {
        throw std::bad_alloc();
    }

    const uintptr_t current_addr = reinterpret_cast<uintptr_t>(buffer_) + current_offset_;
    const size_t padding = (alignment - (current_addr & (alignment - 1))) & (alignment - 1);

    // Capacity is checked by subtraction rather than by computing `current_offset_ + padding +
    // bytes`. That sum wraps for a large `bytes` -- allocate(SIZE_MAX) used to wrap to a small
    // value, pass the bounds check and return a pointer past the end of the arena. The class
    // invariant current_offset_ <= capacity_ makes each subtraction below safe.
    if (padding > capacity_ - current_offset_) {
        throw std::bad_alloc();
    }
    const size_t offset_after_padding = current_offset_ + padding;
    if (bytes > capacity_ - offset_after_padding) {
        throw std::bad_alloc();
    }

    current_offset_ = offset_after_padding + bytes;
    if (current_offset_ > peak_offset_) {
        peak_offset_ = current_offset_;
    }

    return reinterpret_cast<void*>(current_addr + padding);
}

void ArenaAllocator::deallocate([[maybe_unused]] void* ptr,
                               [[maybe_unused]] size_t bytes,
                               [[maybe_unused]] size_t alignment) noexcept {
    // In a monotonic arena, individual deallocation is a no-op.
}

bool ArenaAllocator::owns(const void* ptr) const noexcept {
    const auto* p = static_cast<const std::byte*>(ptr);
    return (p >= buffer_) && (p < (buffer_ + capacity_));
}

void ArenaAllocator::reset() noexcept {
    current_offset_ = 0;
}

} // namespace kinesis::core
