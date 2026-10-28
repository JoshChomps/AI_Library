#pragma once

#include "kinesis/core/allocator.hpp"
#include <cassert>
#include <span>
#include <string_view>

namespace kinesis::core {

/**
 * @brief High-performance Monotonic Bump Allocator (Arena).
 * 
 * Guarantees zero runtime heap allocation and deterministic O(1) latency.
 * Designed specifically for high-frequency trading tick processing,
 * camera frame inference loops, and robotics control cycles.
 */
class ArenaAllocator final : public Allocator {
public:
    /**
     * @brief Construct an arena with internally allocated, aligned buffer.
     * @param capacityBytes Maximum capacity in bytes.
     */
    explicit ArenaAllocator(size_t capacityBytes);

    /**
     * @brief Construct an arena backed by an existing, external buffer.
     * @param externalBuffer Span of raw bytes (e.g. stack array, shared memory).
     */
    explicit ArenaAllocator(std::span<std::byte> externalBuffer) noexcept;

    ~ArenaAllocator() override;

    // Non-copyable (unique memory ownership)
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    // Movable
    ArenaAllocator(ArenaAllocator&& other) noexcept;
    ArenaAllocator& operator=(ArenaAllocator&& other) noexcept;

    /**
     * @brief Allocate contiguous aligned memory by bumping the offset.
     * @param bytes Number of bytes to allocate.
     * @param alignment Alignment boundary (must be power of two, defaults to 64).
     * @return Aligned pointer to allocated memory.
     */
    [[nodiscard]] void* allocate(size_t bytes, size_t alignment = kDefaultAlignment) override;

    /**
     * @brief Deallocate individual pointer. In a monotonic arena, this is an intentional no-op.
     * Memory is reclaimed in bulk via reset() or ScopedMarker.
     */
    void deallocate(void* ptr, size_t bytes, size_t alignment) noexcept override;

    /**
     * @brief Check if a pointer resides within this arena's memory bounds.
     */
    [[nodiscard]] bool owns(const void* ptr) const noexcept override;

    /**
     * @brief Reset the entire arena to zero allocated bytes in O(1) time.
     */
    void reset() noexcept override;

    /**
     * @brief Get current allocated byte count.
     */
    [[nodiscard]] size_t allocated_bytes() const noexcept { return current_offset_; }

    /**
     * @brief Get total capacity in bytes.
     */
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }

    /**
     * @brief Get remaining free bytes.
     */
    [[nodiscard]] size_t available_bytes() const noexcept {
        return (current_offset_ < capacity_) ? (capacity_ - current_offset_) : 0;
    }

    /**
     * @brief Get peak memory usage recorded since creation or last clear.
     */
    [[nodiscard]] size_t peak_bytes() const noexcept { return peak_offset_; }

    /**
     * @brief Base pointer to the underlying buffer.
     */
    [[nodiscard]] const std::byte* data() const noexcept { return buffer_; }
    [[nodiscard]] std::byte* data() noexcept { return buffer_; }

    // =========================================================================
    // RAII Scoped Marker (Automatic rewind for temporary activation memory)
    // =========================================================================
    class [[nodiscard]] ScopedMarker {
    public:
        explicit ScopedMarker(ArenaAllocator& arena) noexcept
            : arena_(arena), saved_offset_(arena.current_offset_) {}

        ~ScopedMarker() noexcept {
            arena_.current_offset_ = saved_offset_;
        }

        ScopedMarker(const ScopedMarker&) = delete;
        ScopedMarker& operator=(const ScopedMarker&) = delete;
        ScopedMarker(ScopedMarker&&) = delete;
        ScopedMarker& operator=(ScopedMarker&&) = delete;

        [[nodiscard]] size_t saved_offset() const noexcept { return saved_offset_; }

    private:
        ArenaAllocator& arena_;
        size_t saved_offset_;
    };

    /**
     * @brief Create an RAII marker to rewind memory when exiting scope.
     */
    [[nodiscard]] ScopedMarker scope() noexcept {
        return ScopedMarker(*this);
    }

private:
    std::byte* buffer_{nullptr};
    size_t capacity_{0};
    size_t current_offset_{0};
    size_t peak_offset_{0};
    bool owns_buffer_{false};
};

} // namespace kinesis::core
