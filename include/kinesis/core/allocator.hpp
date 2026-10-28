#pragma once

#include "kinesis/core/virtual_memory.hpp"
#include <cstddef>
#include <new>

namespace kinesis::core {

/**
 * @brief Abstract Base Memory Resource interface.
 * Decouples all tensor operations from specific memory backends (Arena, Pool, Virtual Memory).
 */
class Allocator {
public:
    virtual ~Allocator() = default;

    /**
     * @brief Allocate a contiguous block of memory with guaranteed alignment.
     * @param bytes Number of bytes to allocate.
     * @param alignment Alignment boundary (defaults to kDefaultAlignment). Must be a power of two.
     * @return Aligned pointer to allocated memory.
     */
    [[nodiscard]] virtual void* allocate(size_t bytes, size_t alignment = kDefaultAlignment) = 0;

    /**
     * @brief Deallocate previously allocated memory.
     * For Arenas, this can be an intentional no-op (bulk reclaimed at reset).
     */
    virtual void deallocate(void* ptr, size_t bytes, size_t alignment = kDefaultAlignment) noexcept = 0;

    /**
     * @brief Does this allocator manage the storage @p ptr points into?
     *
     * @return true only when the allocator can *prove* the pointer lies inside a region it
     *         currently manages, which requires it to track region bounds (ArenaAllocator and
     *         SlabAllocator do). false therefore means "not mine, or not determinable" -- it is
     *         not evidence that some other allocator owns the pointer.
     *
     * @note A truthful "I cannot tell" is the only safe answer for allocators that keep no
     *       registry of their outstanding blocks. An unconditional `return true` would make
     *       every caller's ownership branch take the wrong path for stack, static and
     *       foreign-allocator pointers alike.
     */
    [[nodiscard]] virtual bool owns(const void* ptr) const noexcept = 0;

    /**
     * @brief Reset all allocations in O(1) time if supported.
     */
    virtual void reset() noexcept {}
};

/**
 * @brief Direct Kernel Virtual Memory Allocator (Bypasses standard C runtime malloc).
 * Allocates directly from OS virtual memory pages (VirtualAlloc on Windows, mmap on POSIX).
 * Naturally satisfies 32B, 64B, and 128B alignment requirements (page-aligned to 4096 bytes).
 *
 * @warning One kernel call per allocation, and every block is rounded up to a page (64 KB of
 * address space per reservation on Windows). This is the right backing store for a handful of
 * large, long-lived regions -- an arena, a slab, a weight buffer -- and the wrong one for
 * per-tensor allocation: a 12-byte tensor would consume a whole page, and freeing it costs a
 * kernel call plus, on Linux, TLB-shootdown IPIs to every core running the process. Use
 * HeapAllocator for small or short-lived blocks.
 */
class VirtualMemoryAllocator final : public Allocator {
public:
    [[nodiscard]] void* allocate(size_t bytes, [[maybe_unused]] size_t alignment = kDefaultAlignment) override {
        if (bytes == 0) return nullptr;
        void* ptr = VirtualMemory::allocate(bytes);
        if (!ptr) throw std::bad_alloc();
        return ptr;
    }

    void deallocate(void* ptr, size_t bytes, [[maybe_unused]] size_t alignment) noexcept override {
        if (!ptr) return;
        VirtualMemory::free(ptr, bytes);
    }

    /// Always false: each allocation is an independent OS mapping and no registry of them is
    /// kept, so this allocator cannot prove ownership of any pointer. See Allocator::owns.
    [[nodiscard]] bool owns([[maybe_unused]] const void* ptr) const noexcept override {
        return false;
    }

    static VirtualMemoryAllocator& instance() noexcept {
        static VirtualMemoryAllocator s_instance;
        return s_instance;
    }
};

/**
 * @brief General-purpose aligned heap allocator; the default backing store for Tensor.
 *
 * Uses aligned operator new, so small allocations are served from the C++ runtime's own
 * size-class free lists without entering the kernel, while still honouring the 64-byte
 * alignment the SIMD kernels assume. This is the fallback for allocations that have no arena:
 * hot paths are expected to pass an ArenaAllocator instead.
 */
class HeapAllocator final : public Allocator {
public:
    [[nodiscard]] void* allocate(size_t bytes, size_t alignment = kDefaultAlignment) override {
        if (bytes == 0) return nullptr;
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::bad_alloc(); // aligned operator new requires a power-of-two alignment
        }
        return ::operator new(bytes, std::align_val_t{alignment});
    }

    void deallocate(void* ptr, [[maybe_unused]] size_t bytes, size_t alignment) noexcept override {
        if (!ptr) return;
        // The unsized aligned form: the sized overload is not guaranteed to be declared as a
        // replaceable global, and passing a size that disagrees with the allocation is UB.
        ::operator delete(ptr, std::align_val_t{alignment});
    }

    /// Always false: the runtime heap keeps no queryable bounds. See Allocator::owns.
    [[nodiscard]] bool owns([[maybe_unused]] const void* ptr) const noexcept override {
        return false;
    }

    static HeapAllocator& instance() noexcept {
        static HeapAllocator s_instance;
        return s_instance;
    }
};

} // namespace kinesis::core
