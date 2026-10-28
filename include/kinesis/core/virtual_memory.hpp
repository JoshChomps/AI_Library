#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "kinesis/core/types.hpp"

namespace kinesis::core {

/**
 * @brief Direct OS Virtual Memory Management (bypasses standard C runtime malloc).
 * 
 * Provides deterministic, fragmentation-free memory allocation directly from 
 * the operating system kernel via VirtualAlloc on Windows and mmap on POSIX.
 */
class VirtualMemory {
public:
    /// Returns the system virtual memory page size (typically 4096 bytes).
    [[nodiscard]] static size_t page_size() noexcept {
#if defined(_WIN32) || defined(_WIN64)
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return static_cast<size_t>(si.dwPageSize);
#else
        return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
    }

    /// Returns the system allocation granularity (64KB on Windows, page size on POSIX).
    [[nodiscard]] static size_t allocation_granularity() noexcept {
#if defined(_WIN32) || defined(_WIN64)
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return static_cast<size_t>(si.dwAllocationGranularity);
#else
        return page_size();
#endif
    }

    /**
     * @brief Allocates and commits contiguous read-write physical pages.
     * @param bytes Number of bytes to allocate (will be aligned up to page boundary).
     * @return Aligned pointer to committed memory, or nullptr on failure.
     */
    [[nodiscard]] static void* allocate(size_t bytes) noexcept {
        if (bytes == 0) return nullptr;
        const size_t aligned_size = align_up(bytes, page_size());

#if defined(_WIN32) || defined(_WIN64)
        void* ptr = VirtualAlloc(nullptr, aligned_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        return ptr;
#else
        void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, 
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return (ptr == MAP_FAILED) ? nullptr : ptr;
#endif
    }

    /**
     * @brief Reserves virtual address space without committing physical RAM.
     * @param bytes Number of bytes to reserve (aligned up to allocation granularity).
     * @return Base address of reserved region, or nullptr on failure.
     */
    [[nodiscard]] static void* reserve(size_t bytes) noexcept {
        if (bytes == 0) return nullptr;
        const size_t aligned_size = align_up(bytes, allocation_granularity());

#if defined(_WIN32) || defined(_WIN64)
        return VirtualAlloc(nullptr, aligned_size, MEM_RESERVE, PAGE_READWRITE);
#else
        void* ptr = mmap(nullptr, aligned_size, PROT_NONE, 
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return (ptr == MAP_FAILED) ? nullptr : ptr;
#endif
    }

    /**
     * @brief Commits physical pages within a previously reserved address space.
     * @param ptr Starting address (must be within a reserved region).
     * @param bytes Number of bytes to commit.
     * @return True if commit succeeded, false otherwise.
     */
    static bool commit(void* ptr, size_t bytes) noexcept {
        if (!ptr || bytes == 0) return false;
        const size_t aligned_size = align_up(bytes, page_size());

#if defined(_WIN32) || defined(_WIN64)
        void* res = VirtualAlloc(ptr, aligned_size, MEM_COMMIT, PAGE_READWRITE);
        return res != nullptr;
#else
        int res = mprotect(ptr, aligned_size, PROT_READ | PROT_WRITE);
        return res == 0;
#endif
    }

    /**
     * @brief Decommits physical pages, returning RAM to OS while preserving virtual address reservation.
     */
    static bool decommit(void* ptr, size_t bytes) noexcept {
        if (!ptr || bytes == 0) return false;
        const size_t aligned_size = align_up(bytes, page_size());

#if defined(_WIN32) || defined(_WIN64)
        return VirtualFree(ptr, aligned_size, MEM_DECOMMIT) != 0;
#else
        int res = madvise(ptr, aligned_size, MADV_DONTNEED);
        return res == 0;
#endif
    }

    /**
     * @brief Releases entire allocated or reserved virtual memory region back to the OS.
     * @param ptr Pointer returned by allocate() or reserve().
     * @param bytes Number of bytes originally allocated (required for POSIX munmap).
     */
    static void free(void* ptr, size_t bytes) noexcept {
        if (!ptr) return;

#if defined(_WIN32) || defined(_WIN64)
        (void)bytes;
        // On Windows, MEM_RELEASE requires dwSize to be 0
        VirtualFree(ptr, 0, MEM_RELEASE);
#else
        const size_t aligned_size = align_up(bytes, page_size());
        munmap(ptr, aligned_size);
#endif
    }
};

} // namespace kinesis::core
