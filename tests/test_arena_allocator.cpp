#include "test_harness.hpp"
#include "kinesis/core/arena_allocator.hpp"
#include <array>

using namespace kinesis::core;

KINESIS_TEST_CASE(ArenaAllocator_AlignmentAndBumping) {
    // 1 MB Arena
    ArenaAllocator arena(1024 * 1024);
    ASSERT_EQ(arena.capacity(), 1024 * 1024);
    ASSERT_EQ(arena.allocated_bytes(), 0);

    // Allocate 10 floats (40 bytes), default alignment 64
    void* p1 = arena.allocate(40, 64);
    ASSERT_TRUE(p1 != nullptr);
    ASSERT_TRUE(is_aligned(p1, 64));
    ASSERT_EQ(arena.allocated_bytes(), 40); // 40 bytes allocated without trailing padding until next allocation

    // Allocate 100 doubles (800 bytes), alignment 64
    void* p2 = arena.allocate(800, 64);
    ASSERT_TRUE(p2 != nullptr);
    ASSERT_TRUE(is_aligned(p2, 64));
    ASSERT_TRUE(p2 > p1);
    ASSERT_TRUE(arena.owns(p1));
    ASSERT_TRUE(arena.owns(p2));

    // Reset
    arena.reset();
    ASSERT_EQ(arena.allocated_bytes(), 0);

    // Reallocate starts at beginning
    void* p3 = arena.allocate(40, 64);
    ASSERT_EQ(p1, p3);
}

KINESIS_TEST_CASE(ArenaAllocator_OutOfMemory) {
    // 256 bytes arena
    ArenaAllocator arena(256);
    void* p1 = arena.allocate(128, 64);
    ASSERT_TRUE(p1 != nullptr);

    // Try allocating 256 bytes more (exceeds capacity)
    bool threw = false;
    try {
        (void)arena.allocate(256, 64);
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

KINESIS_TEST_CASE(ArenaAllocator_ScopedMarker) {
    ArenaAllocator arena(1024 * 1024);
    void* p1 = arena.allocate(128, 64);
    (void)p1;
    size_t offset_before = arena.allocated_bytes();

    {
        auto scope = arena.scope();
        void* p2 = arena.allocate(256, 64);
        ASSERT_TRUE(p2 != nullptr);
        ASSERT_TRUE(arena.allocated_bytes() > offset_before);
    } // scope exits, memory rewound

    ASSERT_EQ(arena.allocated_bytes(), offset_before);

    // Next allocation reuses the exact same address as p2
    void* p3 = arena.allocate(256, 64);
    ASSERT_TRUE(p3 != nullptr);
    ASSERT_TRUE(is_aligned(p3, 64));
}

KINESIS_TEST_CASE(ArenaAllocator_ExternalBuffer) {
    // Zero-heap arena bound to stack buffer
    alignas(64) std::array<std::byte, 1024> stack_buffer{};
    ArenaAllocator arena(std::span<std::byte>(stack_buffer.data(), stack_buffer.size()));

    ASSERT_EQ(arena.capacity(), 1024);
    void* ptr = arena.allocate(64, 64);
    ASSERT_TRUE(ptr != nullptr);
    ASSERT_TRUE(is_aligned(ptr, 64));
    ASSERT_TRUE(arena.owns(ptr));
}
