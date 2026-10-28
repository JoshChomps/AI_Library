#pragma once

#include "kinesis/core/allocator.hpp"
#include "kinesis/core/virtual_memory.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

namespace kinesis::core {

/**
 * @brief Fixed-size Segregated Block Allocator (Slab / BlockPool).
 * 
 * Provides deterministic O(1) allocation and O(1) deallocation for fixed-size objects
 * (e.g. quaternions, 64-byte state vectors, 4x4 affine matrices, and small tensors).
 * Uses an intrusive singly-linked free list stored directly within inactive blocks.
 * Zero external heap fragmentation; backed by direct OS virtual memory pages.
 */
template <size_t BlockSize, size_t Alignment = kDefaultAlignment>
class SlabAllocator final : public Allocator {
    static_assert(BlockSize >= sizeof(void*), "BlockSize must be at least pointer-sized for intrusive free-list");
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");

public:
    static constexpr size_t kActualBlockSize = align_up(BlockSize, Alignment);

    /**
     * @brief Construct a SlabAllocator with a given block capacity.
     * @param blockCount Maximum number of blocks in this slab.
     */
    explicit SlabAllocator(size_t blockCount)
        : block_count_(blockCount),
          free_count_(blockCount),
          slab_bytes_(align_up(blockCount * kActualBlockSize, VirtualMemory::page_size())) {
        
        if (slab_bytes_ > 0) {
            slab_memory_ = static_cast<std::byte*>(VirtualMemory::allocate(slab_bytes_));
            if (!slab_memory_) {
                throw std::bad_alloc();
            }
            initialize_free_list();
        }
    }

    ~SlabAllocator() override {
        if (slab_memory_) {
            VirtualMemory::free(slab_memory_, slab_bytes_);
            slab_memory_ = nullptr;
        }
    }

    // Non-copyable
    SlabAllocator(const SlabAllocator&) = delete;
    SlabAllocator& operator=(const SlabAllocator&) = delete;

    // Movable
    SlabAllocator(SlabAllocator&& other) noexcept
        : slab_memory_(other.slab_memory_),
          free_list_(other.free_list_),
          block_count_(other.block_count_),
          free_count_(other.free_count_),
          slab_bytes_(other.slab_bytes_) {
        other.slab_memory_ = nullptr;
        other.free_list_ = nullptr;
        other.block_count_ = 0;
        other.free_count_ = 0;
        other.slab_bytes_ = 0;
    }

    SlabAllocator& operator=(SlabAllocator&& other) noexcept {
        if (this != &other) {
            if (slab_memory_) {
                VirtualMemory::free(slab_memory_, slab_bytes_);
            }
            slab_memory_ = other.slab_memory_;
            free_list_ = other.free_list_;
            block_count_ = other.block_count_;
            free_count_ = other.free_count_;
            slab_bytes_ = other.slab_bytes_;

            other.slab_memory_ = nullptr;
            other.free_list_ = nullptr;
            other.block_count_ = 0;
            other.free_count_ = 0;
            other.slab_bytes_ = 0;
        }
        return *this;
    }

    /**
     * @brief Allocate a single block from the free list in O(1) time.
     * @param bytes Requested byte size (must be <= kActualBlockSize).
     * @param alignment Requested alignment (must be <= Alignment).
     */
    [[nodiscard]] void* allocate(size_t bytes, size_t alignment = Alignment) override {
        // Checked in release too: handing back a block that is smaller or less aligned than the
        // caller asked for is a silent buffer overflow, and asserts are compiled out under
        // NDEBUG -- which is exactly the build that runs in production.
        if (bytes > kActualBlockSize) {
            throw std::bad_alloc();
        }
        if (alignment > Alignment) {
            throw std::bad_alloc();
        }

        if (!free_list_) {
            throw std::bad_alloc(); // Slab pool exhausted
        }

        // Pop head block from intrusive free-list in ~3 instructions
        Node* node = free_list_;
        free_list_ = node->next;
        --free_count_;

        return static_cast<void*>(node);
    }

    /**
     * @brief Return a block to the free list in O(1) time.
     */
    void deallocate(void* ptr, [[maybe_unused]] size_t bytes, [[maybe_unused]] size_t alignment = Alignment) noexcept override {
        if (!ptr) return;

        // Rejecting a foreign or interior pointer in every build, not just under an assert.
        // Linking one into the intrusive free list would corrupt it silently: the next
        // allocate() would hand out memory this slab does not own, and the fault would surface
        // somewhere unrelated. This function is noexcept, so the only safe action is to refuse.
        // No assert here: refusing is the defined behaviour, and tests/test_regressions.cpp
        // checks it in the debug build.
        if (!owns(ptr)) {
            return;
        }
        const size_t offset = static_cast<size_t>(static_cast<std::byte*>(ptr) - slab_memory_);
        if (offset % kActualBlockSize != 0) {
            return;
        }

        // Push block back to head of intrusive free-list in ~3 instructions
        auto* node = static_cast<Node*>(ptr);
        node->next = free_list_;
        free_list_ = node;
        ++free_count_;
    }

    /**
     * @brief Convenience O(1) deallocate for fixed-size slab blocks.
     */
    void deallocate(void* ptr) noexcept {
        deallocate(ptr, kActualBlockSize, Alignment);
    }


    /**
     * @brief Check if a pointer resides within this slab's address range.
     */
    [[nodiscard]] bool owns(const void* ptr) const noexcept override {
        if (!slab_memory_ || !ptr) return false;
        const auto* byte_ptr = static_cast<const std::byte*>(ptr);
        return (byte_ptr >= slab_memory_) && (byte_ptr < slab_memory_ + (block_count_ * kActualBlockSize));
    }

    /**
     * @brief Reset the entire slab, restoring all blocks to the free list in O(N) time.
     */
    void reset() noexcept override {
        free_count_ = block_count_;
        initialize_free_list();
    }

    [[nodiscard]] size_t block_size() const noexcept { return kActualBlockSize; }
    [[nodiscard]] size_t total_blocks() const noexcept { return block_count_; }
    [[nodiscard]] size_t free_blocks() const noexcept { return free_count_; }
    [[nodiscard]] size_t allocated_blocks() const noexcept { return block_count_ - free_count_; }

private:
    struct Node {
        Node* next;
    };

    void initialize_free_list() noexcept {
        if (!slab_memory_ || block_count_ == 0) {
            free_list_ = nullptr;
            return;
        }

        free_list_ = reinterpret_cast<Node*>(slab_memory_);
        Node* current = free_list_;

        for (size_t i = 0; i < block_count_ - 1; ++i) {
            auto* next = reinterpret_cast<Node*>(reinterpret_cast<std::byte*>(current) + kActualBlockSize);
            current->next = next;
            current = next;
        }
        current->next = nullptr;
    }

    std::byte* slab_memory_{nullptr};
    Node*      free_list_{nullptr};
    size_t     block_count_{0};
    size_t     free_count_{0};
    size_t     slab_bytes_{0};
};

} // namespace kinesis::core
