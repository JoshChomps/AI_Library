#include "test_harness.hpp"
#include "kinesis/core/slab_allocator.hpp"

using namespace kinesis::core;

TEST_CASE(SlabAllocator_FixedBlockAllocationAndReuse) {
    constexpr size_t kBlockSize = 64;
    constexpr size_t kBlockCount = 16;

    SlabAllocator<kBlockSize> slab(kBlockCount);
    EXPECT_EQ(slab.total_blocks(), kBlockCount);
    EXPECT_EQ(slab.free_blocks(), kBlockCount);

    // Allocate first block
    void* b1 = slab.allocate(kBlockSize);
    EXPECT_NOT_NULL(b1);
    EXPECT_TRUE(slab.owns(b1));
    EXPECT_TRUE(is_aligned(b1, kDefaultAlignment));
    EXPECT_EQ(slab.allocated_blocks(), 1);

    // Allocate second block
    void* b2 = slab.allocate(kBlockSize);
    EXPECT_NOT_NULL(b2);
    EXPECT_TRUE(slab.owns(b2));
    EXPECT_TRUE(b1 != b2);
    EXPECT_EQ(slab.allocated_blocks(), 2);

    // Deallocate b1, then allocate again -> should return b1 due to LIFO intrusive list
    slab.deallocate(b1, kBlockSize);
    EXPECT_EQ(slab.allocated_blocks(), 1);

    void* b3 = slab.allocate(kBlockSize);
    EXPECT_EQ(b3, b1); // Proof of instant O(1) block recycling!

    slab.deallocate(b2, kBlockSize);
    slab.deallocate(b3, kBlockSize);
    EXPECT_EQ(slab.free_blocks(), kBlockCount);
}

TEST_CASE(SlabAllocator_ResetReclaimsAllBlocks) {
    constexpr size_t kBlockSize = 128;
    constexpr size_t kBlockCount = 8;

    SlabAllocator<kBlockSize> slab(kBlockCount);

    void* ptrs[kBlockCount];
    for (size_t i = 0; i < kBlockCount; ++i) {
        ptrs[i] = slab.allocate(kBlockSize);
        EXPECT_NOT_NULL(ptrs[i]);
    }
    EXPECT_EQ(slab.free_blocks(), 0);

    // Reset restores all blocks in one call
    slab.reset();
    EXPECT_EQ(slab.free_blocks(), kBlockCount);
    EXPECT_EQ(slab.allocated_blocks(), 0);
}
