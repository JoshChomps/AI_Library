#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include "kinesis/core/types.hpp"

namespace kinesis::core {

/**
 * @brief Deterministic Software Hardware-Cache Simulator.
 * 
 * Simulates multi-level CPU and GPU memory hierarchies (L1 Data Cache, L2 Cache,
 * cache-line split penalties, and 32-bank shared memory conflicts) in software.
 * Eliminates CPU turboboost jitter and prefetcher noise during algorithmic verification.
 */
class MemorySandbox {
public:
    struct CacheConfig {
        size_t capacity_bytes{32768}; // 32 KB default L1
        size_t line_size_bytes{64};   // 64 bytes cache line
        size_t associativity{8};      // 8-way set-associative
    };

    explicit MemorySandbox(CacheConfig l1_cfg = {32768, 64, 8}, 
                           CacheConfig l2_cfg = {524288, 64, 8})
        : l1_config_(l1_cfg), l2_config_(l2_cfg) {
        init_cache(l1_cache_, l1_config_);
        init_cache(l2_cache_, l2_config_);
    }

    void reset_stats() noexcept {
        l1_accesses_ = 0;
        l1_hits_ = 0;
        l1_misses_ = 0;
        l2_accesses_ = 0;
        l2_hits_ = 0;
        l2_misses_ = 0;
        cache_line_splits_ = 0;
        page_splits_ = 0;
        bank_conflicts_ = 0;
        timestamp_ = 0;
        clear_cache(l1_cache_);
        clear_cache(l2_cache_);
    }

    /**
     * @brief Record a memory read of `size` bytes at physical address `addr`.
     */
    void access_read(uintptr_t addr, size_t size = 4) noexcept {
        access_internal(addr, size, false);
    }

    /**
     * @brief Record a memory write of `size` bytes at physical address `addr`.
     */
    void access_write(uintptr_t addr, size_t size = 4) noexcept {
        access_internal(addr, size, true);
    }

    /**
     * @brief Record a warp (32 threads) accessing shared memory addresses to detect bank conflicts.
     * @param addrs Array of 32 byte-addresses accessed by warp threads 0..31.
     * @return Maximum bank conflict degree (1 = zero conflicts, 32 = full 32-way serialization).
     */
    size_t record_warp_bank_access(const std::array<uintptr_t, 32>& addrs) noexcept {
        std::array<size_t, 32> bank_histogram{};
        for (uintptr_t addr : addrs) {
            size_t bank_id = (addr / 4) % 32;
            bank_histogram[bank_id]++;
        }

        size_t max_conflict = 0;
        for (size_t count : bank_histogram) {
            if (count > max_conflict) {
                max_conflict = count;
            }
        }

        if (max_conflict > 1) {
            bank_conflicts_ += (max_conflict - 1);
        }

        return max_conflict;
    }

    // Metrics
    [[nodiscard]] size_t l1_accesses() const noexcept { return l1_accesses_; }
    [[nodiscard]] size_t l1_hits() const noexcept { return l1_hits_; }
    [[nodiscard]] size_t l1_misses() const noexcept { return l1_misses_; }
    [[nodiscard]] double l1_hit_rate() const noexcept {
        return l1_accesses_ == 0 ? 1.0 : static_cast<double>(l1_hits_) / static_cast<double>(l1_accesses_);
    }

    [[nodiscard]] size_t l2_accesses() const noexcept { return l2_accesses_; }
    [[nodiscard]] size_t l2_hits() const noexcept { return l2_hits_; }
    [[nodiscard]] size_t l2_misses() const noexcept { return l2_misses_; }
    [[nodiscard]] double l2_hit_rate() const noexcept {
        return l2_accesses_ == 0 ? 1.0 : static_cast<double>(l2_hits_) / static_cast<double>(l2_accesses_);
    }

    [[nodiscard]] size_t cache_line_splits() const noexcept { return cache_line_splits_; }

    /// Accesses that straddled a 4 KB page boundary: a second TLB translation, and
    /// potentially a second page fault, on top of the extra cache-line lookup.
    [[nodiscard]] size_t page_splits() const noexcept { return page_splits_; }
    [[nodiscard]] size_t bank_conflicts() const noexcept { return bank_conflicts_; }

    /**
     * @brief Render an ASCII/Unicode visual dashboard of cache and bank metrics.
     */
    void print_dashboard(std::ostream& os = std::cout) const {
        os << "\n======================================================\n";
        os << "           KINESIS MEMORY SANDBOX DASHBOARD           \n";
        os << "======================================================\n";

        auto render_bar = [](double rate, size_t bar_width = 20) -> std::string {
            size_t filled = static_cast<size_t>(rate * bar_width);
            if (filled > bar_width) filled = bar_width;
            std::string bar = "[";
            for (size_t i = 0; i < filled; ++i) bar += "=";
            for (size_t i = filled; i < bar_width; ++i) bar += " ";
            bar += "]";
            return bar;
        };

        double l1_rate = l1_hit_rate() * 100.0;
        double l2_rate = l2_hit_rate() * 100.0;

        os << "L1 Data Cache: " << render_bar(l1_hit_rate()) << " " 
           << std::fixed << std::setprecision(1) << l1_rate << "%\n";
        os << "  - Accesses: " << l1_accesses_ << " | Hits: " << l1_hits_ 
           << " | Misses: " << l1_misses_ << "\n\n";

        os << "L2 Data Cache: " << render_bar(l2_hit_rate()) << " " 
           << std::fixed << std::setprecision(1) << l2_rate << "%\n";
        os << "  - Accesses: " << l2_accesses_ << " | Hits: " << l2_hits_ 
           << " | Misses: " << l2_misses_ << "\n\n";

        os << "Coherence & Latency Hazards:\n";
        os << "  - Cache-Line Splits: " << cache_line_splits_
           << (cache_line_splits_ == 0 ? " (Optimal: 0 splits)" : " (each costs one extra line lookup)") << "\n";
        os << "  - Page Splits (4K):  " << page_splits_
           << (page_splits_ == 0 ? " (Optimal: 0 splits)" : " (WARNING: second TLB translation)") << "\n";
        os << "  - Bank Conflicts:    " << bank_conflicts_ 
           << (bank_conflicts_ == 0 ? " (Optimal Coalescing: 0 conflicts)" : " (WARNING: Warp Serialization!)") << "\n";
        os << "======================================================\n";
    }

    /**
     * @brief Render a visual bar histogram of warp thread distribution across the 32 memory banks.
     */
    static void print_bank_histogram(const std::array<size_t, 32>& histogram, std::ostream& os = std::cout) {
        os << "\n--- 32-Bank Shared Memory Warp Distribution ---\n";
        for (size_t b = 0; b < 32; ++b) {
            os << "Bank " << std::setw(2) << b << ": ";
            for (size_t i = 0; i < histogram[b]; ++i) os << "#";
            if (histogram[b] > 1) os << " (" << histogram[b] << "-way conflict!)";
            os << "\n";
        }
        os << "-----------------------------------------------\n";
    }

private:
    struct CacheLine {
        bool valid{false};
        uintptr_t tag{0};
        uint64_t last_used{0};
    };

    struct Set {
        std::vector<CacheLine> ways;
    };

    void init_cache(std::vector<Set>& cache, const CacheConfig& cfg) {
        size_t num_sets = cfg.capacity_bytes / (cfg.line_size_bytes * cfg.associativity);
        if (num_sets == 0) num_sets = 1;
        cache.resize(num_sets);
        for (auto& set : cache) {
            set.ways.resize(cfg.associativity);
        }
    }

    void clear_cache(std::vector<Set>& cache) {
        for (auto& set : cache) {
            for (auto& line : set.ways) {
                line.valid = false;
                line.tag = 0;
                line.last_used = 0;
            }
        }
    }

    bool query_and_update(std::vector<Set>& cache, const CacheConfig& cfg, uintptr_t addr) noexcept {
        uintptr_t line_addr = addr / cfg.line_size_bytes;
        size_t num_sets = cache.size();
        size_t set_idx = line_addr % num_sets;
        uintptr_t tag = line_addr / num_sets;

        auto& set = cache[set_idx];
        ++timestamp_;

        // Check for hit
        for (auto& line : set.ways) {
            if (line.valid && line.tag == tag) {
                line.last_used = timestamp_;
                return true; // Hit
            }
        }

        // Miss: Find LRU or invalid slot
        size_t lru_idx = 0;
        uint64_t oldest = UINT64_MAX;
        for (size_t i = 0; i < set.ways.size(); ++i) {
            if (!set.ways[i].valid) {
                lru_idx = i;
                break;
            }
            if (set.ways[i].last_used < oldest) {
                oldest = set.ways[i].last_used;
                lru_idx = i;
            }
        }

        set.ways[lru_idx].valid = true;
        set.ways[lru_idx].tag = tag;
        set.ways[lru_idx].last_used = timestamp_;

        return false; // Miss
    }

    void access_internal(uintptr_t addr, size_t size, [[maybe_unused]] bool is_write) noexcept {
        if (size == 0) {
            return;
        }

        const size_t line_bytes = l1_config_.line_size_bytes;
        const uintptr_t line_start = addr / line_bytes;
        const uintptr_t line_end = (addr + size - 1) / line_bytes;

        if (line_start != line_end) {
            ++cache_line_splits_;
        }

        // A 4 KB page split is the expensive case: it needs a second TLB translation and can
        // take a second page fault, where a plain line split within a page only costs the extra
        // line lookup below. Counted separately so the dashboard does not conflate them.
        constexpr size_t kPageBytes = 4096;
        if (addr / kPageBytes != (addr + size - 1) / kPageBytes) {
            ++page_splits_;
        }

        // Every line the access touches is looked up, not just the line the access starts in.
        // A split access that begins on a resident line and ends on an evicted one misses, and
        // querying only `addr` recorded that as a clean hit -- which made split accesses look
        // free in exactly the case the sandbox exists to illustrate.
        for (uintptr_t line = line_start; line <= line_end; ++line) {
            const uintptr_t line_addr = line * line_bytes;

            ++l1_accesses_;
            if (query_and_update(l1_cache_, l1_config_, line_addr)) {
                ++l1_hits_;
                continue;
            }
            ++l1_misses_;

            // Query L2 on L1 Miss
            ++l2_accesses_;
            if (query_and_update(l2_cache_, l2_config_, line_addr)) {
                ++l2_hits_;
            } else {
                ++l2_misses_;
            }
        }
    }

    CacheConfig l1_config_;
    CacheConfig l2_config_;
    std::vector<Set> l1_cache_;
    std::vector<Set> l2_cache_;

    size_t l1_accesses_{0};
    size_t l1_hits_{0};
    size_t l1_misses_{0};
    size_t l2_accesses_{0};
    size_t l2_hits_{0};
    size_t l2_misses_{0};
    size_t cache_line_splits_{0};
    size_t page_splits_{0};
    size_t bank_conflicts_{0};
    uint64_t timestamp_{0};
};

} // namespace kinesis::core
