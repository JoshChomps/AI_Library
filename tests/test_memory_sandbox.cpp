#include "test_harness.hpp"
#include "kinesis/core/memory_sandbox.hpp"
#include <sstream>

using namespace kinesis::core;

TEST_CASE(MemorySandbox_SequentialAccessHighHitRate) {
    // 32 KB L1 with 64-byte lines (8-way associative)
    MemorySandbox sandbox({32768, 64, 8});

    // Sequential streaming across 1024 floats (4096 bytes, well within 32 KB L1)
    uintptr_t base_addr = 0x10000;
    for (size_t i = 0; i < 1024; ++i) {
        sandbox.access_read(base_addr + i * sizeof(float), sizeof(float));
    }

    // 1024 floats of 4 bytes = 4096 bytes = 64 cache lines of 64 bytes
    // First read to each line is a cold miss (64 misses), next 15 reads to that line are hits!
    // Total accesses = 1024, Misses = 64, Hits = 960
    // Hit rate = 960 / 1024 = 93.75%
    EXPECT_EQ(sandbox.l1_accesses(), 1024);
    EXPECT_EQ(sandbox.l1_misses(), 64);
    EXPECT_EQ(sandbox.l1_hits(), 960);
    EXPECT_TRUE(sandbox.l1_hit_rate() > 0.90);
    EXPECT_EQ(sandbox.cache_line_splits(), 0);
}

TEST_CASE(MemorySandbox_CacheLineSplitDetection) {
    MemorySandbox sandbox({32768, 64, 8});

    // Access aligned to boundary: address 60, size 4 (spans 60..63 within line 0) -> no split
    sandbox.access_read(60, 4);
    EXPECT_EQ(sandbox.cache_line_splits(), 0);

    // Unaligned access crossing 64-byte boundary: address 62, size 4 (spans 62..65 across lines 0 & 1)
    sandbox.access_read(62, 4);
    EXPECT_EQ(sandbox.cache_line_splits(), 1);
}

TEST_CASE(MemorySandbox_WarpBankConflictDetection) {
    MemorySandbox sandbox;

    // Case 1: 32 threads accessing 32 distinct 4-byte banks (stride 4 bytes)
    std::array<uintptr_t, 32> bank_free_addrs;
    for (size_t t = 0; t < 32; ++t) {
        bank_free_addrs[t] = t * 4; // bank 0, 1, 2, ..., 31
    }
    size_t conflict = sandbox.record_warp_bank_access(bank_free_addrs);
    EXPECT_EQ(conflict, 1); // 1 = zero bank conflict (1 thread per bank)

    // Case 2: 32 threads all accessing stride 128 bytes (all land on bank 0!)
    std::array<uintptr_t, 32> conflicting_addrs;
    for (size_t t = 0; t < 32; ++t) {
        conflicting_addrs[t] = t * 128; // (128 / 4) % 32 = 32 % 32 = 0 (Bank 0!)
    }
    size_t conflict32 = sandbox.record_warp_bank_access(conflicting_addrs);
    EXPECT_EQ(conflict32, 32); // Catastrophic 32-way bank conflict!
}

TEST_CASE(MemorySandbox_VisualDashboardRendering) {
    MemorySandbox sandbox({32768, 64, 8});
    for (size_t i = 0; i < 256; ++i) {
        sandbox.access_read(0x40000 + i * sizeof(float), sizeof(float));
    }
    std::ostringstream oss;
    sandbox.print_dashboard(oss);
    EXPECT_TRUE(!oss.str().empty());
    EXPECT_TRUE(oss.str().find("KINESIS MEMORY SANDBOX DASHBOARD") != std::string::npos);
}


// Section 6.2: a split access touches two cache lines, but the model looked up only the line
// containing the start address, so the second line's miss was never counted.
TEST_CASE(MemorySandbox_SplitAccessQueriesEveryLineItTouches) {
    MemorySandbox sandbox;

    // 4 bytes straddling the 64-byte boundary: two lines, so two lookups and two compulsory
    // misses. Looking up only the start address would have recorded one access and one miss.
    sandbox.access_read(62, 4);
    EXPECT_EQ(sandbox.cache_line_splits(), 1);
    EXPECT_EQ(sandbox.l1_accesses(), 2);
    EXPECT_EQ(sandbox.l1_misses(), 2);
    EXPECT_EQ(sandbox.page_splits(), 0); // lines 0 and 1 are both inside page 0

    // Both lines are resident now, so the same access is two hits.
    sandbox.access_read(62, 4);
    EXPECT_EQ(sandbox.cache_line_splits(), 2);
    EXPECT_EQ(sandbox.l1_accesses(), 4);
    EXPECT_EQ(sandbox.l1_hits(), 2);

    // A non-split access is still a single lookup.
    sandbox.access_read(128, 4);
    EXPECT_EQ(sandbox.l1_accesses(), 5);
    EXPECT_EQ(sandbox.cache_line_splits(), 2);
}

TEST_CASE(MemorySandbox_PageSplitsAreCountedSeparately) {
    MemorySandbox sandbox;

    // Inside page 0: a line split (lines 0 and 1) but no page split.
    sandbox.access_read(62, 4);
    EXPECT_EQ(sandbox.cache_line_splits(), 1);
    EXPECT_EQ(sandbox.page_splits(), 0);

    // Straddling the 4 KB boundary at 4096: lines 63 and 64, pages 0 and 1, so both.
    sandbox.access_read(4094, 8);
    EXPECT_EQ(sandbox.cache_line_splits(), 2);
    EXPECT_EQ(sandbox.page_splits(), 1);
}
