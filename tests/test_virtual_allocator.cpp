#include "test_harness.hpp"
#include "kinesis/core/virtual_memory.hpp"
#include "kinesis/core/allocator.hpp"

using namespace kinesis::core;

TEST_CASE(VirtualMemory_PageAllocationAndFree) {
    size_t page_sz = VirtualMemory::page_size();
    EXPECT_TRUE(page_sz >= 4096);

    // Allocate 1 page
    void* p = VirtualMemory::allocate(page_sz);
    EXPECT_NOT_NULL(p);

    // Verify pointer is aligned to at least 64 bytes (it should be page-aligned to 4096 bytes)
    EXPECT_TRUE(is_aligned(p, 64));
    EXPECT_TRUE(is_aligned(p, 128));

    // Touch memory (write & read)
    auto* bytes = static_cast<uint8_t*>(p);
    bytes[0] = 42;
    bytes[page_sz - 1] = 99;
    EXPECT_EQ(bytes[0], 42);
    EXPECT_EQ(bytes[page_sz - 1], 99);

    VirtualMemory::free(p, page_sz);
}

TEST_CASE(VirtualMemoryAllocator_PolymorphicInterface) {
    auto& alloc = VirtualMemoryAllocator::instance();

    void* p = alloc.allocate(1024, kDefaultAlignment);
    EXPECT_NOT_NULL(p);
    EXPECT_TRUE(is_aligned(p, kDefaultAlignment));

    // owns() must answer "I cannot prove it" rather than claiming everything. This allocator
    // keeps no registry of its mappings, so it reports false even for its own pointer -- and
    // must certainly not claim a stack address.
    int on_the_stack = 0;
    EXPECT_FALSE(alloc.owns(p));
    EXPECT_FALSE(alloc.owns(&on_the_stack));

    alloc.deallocate(p, 1024, kDefaultAlignment);
}

TEST_CASE(HeapAllocator_AlignedAndDistinctFromPageAllocator) {
    auto& alloc = HeapAllocator::instance();

    // Small allocations must still meet the alignment the SIMD kernels assume, without
    // rounding each one up to an OS page the way the page allocator would.
    void* small = alloc.allocate(12, kDefaultAlignment);
    EXPECT_NOT_NULL(small);
    EXPECT_TRUE(is_aligned(small, kDefaultAlignment));

    void* wide = alloc.allocate(4096, 128);
    EXPECT_NOT_NULL(wide);
    EXPECT_TRUE(is_aligned(wide, 128));

    EXPECT_TRUE(alloc.allocate(0, kDefaultAlignment) == nullptr);

    int on_the_stack = 0;
    EXPECT_FALSE(alloc.owns(small));
    EXPECT_FALSE(alloc.owns(&on_the_stack));

    alloc.deallocate(wide, 4096, 128);
    alloc.deallocate(small, 12, kDefaultAlignment);
    alloc.deallocate(nullptr, 0, kDefaultAlignment); // must be a safe no-op
}
