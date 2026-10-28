// Kinesis latency benchmarks.
//
// Reporting rules this file follows, because the numbers are quoted in the docs:
//  - Every timed region is repeated, and the distribution is reported (min / median / p95),
//    not a single mean. A mean over one run hides the page faults and scheduler noise that
//    dominate the first iterations.
//  - Every timed loop consumes its result through a compiler barrier, so nothing is folded
//    away or hoisted out. A loop whose body is loop-invariant is measured as what it is.
//  - Buffers are initialised before they are read. Reading an uninitialised float is undefined
//    behaviour, and the optimiser is entitled to assume it does not happen.
//  - First-touch page faults are separated from steady-state cost by pre-faulting.
//  - Every Kinesis number is printed next to a standard-library baseline, so the comparison is
//    visible rather than implied.
//  - The build configuration is printed with the results; the same source compiled without
//    -march=native or with a different -ffp-contract setting does not produce the same numbers.
#include "kinesis/kinesis.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory_resource>
#include <string>
#include <vector>

using namespace kinesis::core;
using namespace kinesis::compute;

namespace {

using Clock = std::chrono::steady_clock; // monotonic; high_resolution_clock need not be

// Keep a value alive across the optimiser: the asm block is opaque, so the compiler must
// materialise `value` and must assume memory was read.
template <typename T>
inline void do_not_optimize(const T& value) {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    volatile char sink = *reinterpret_cast<const volatile char*>(&value);
    (void)sink;
#endif
}

inline void clobber_memory() {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" : : : "memory");
#endif
}

struct Stats {
    double min_ns = 0.0;
    double median_ns = 0.0;
    double p95_ns = 0.0;
    size_t samples = 0;
};

// Run `body` `reps` times, each measuring `inner_ops` operations, and summarise per-operation
// cost. `warmup` reps are run first and discarded.
template <typename Body>
Stats measure(size_t reps, size_t inner_ops, Body&& body, size_t warmup = 3) {
    for (size_t i = 0; i < warmup; ++i) {
        body();
    }
    std::vector<double> samples;
    samples.reserve(reps);
    for (size_t i = 0; i < reps; ++i) {
        const auto t0 = Clock::now();
        body();
        const auto t1 = Clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() /
                          static_cast<double>(inner_ops));
    }
    std::sort(samples.begin(), samples.end());
    Stats s;
    s.samples = samples.size();
    s.min_ns = samples.front();
    s.median_ns = samples[samples.size() / 2];
    s.p95_ns = samples[static_cast<size_t>(static_cast<double>(samples.size() - 1) * 0.95)];
    return s;
}

void print_row(const std::string& label, const Stats& s, const char* unit = "ns/op") {
    std::cout << "  " << std::left << std::setw(38) << label << std::right << std::fixed
              << std::setprecision(2) << std::setw(9) << s.min_ns << "  " << std::setw(9)
              << s.median_ns << "  " << std::setw(9) << s.p95_ns << "   " << unit << "\n";
}

void print_header(const char* title) {
    std::cout << "\n======================================================\n";
    std::cout << title << "\n";
    std::cout << "======================================================\n";
    std::cout << "  " << std::left << std::setw(38) << "" << std::right << std::setw(9) << "min"
              << "  " << std::setw(9) << "median" << "  " << std::setw(9) << "p95" << "\n";
}

void print_build_configuration() {
    std::cout << "\n======================================================\n";
    std::cout << "                 BUILD CONFIGURATION                  \n";
    std::cout << "======================================================\n";
#if defined(__clang__)
    std::cout << "Compiler:          clang " << __clang_major__ << "." << __clang_minor__ << "\n";
#elif defined(_MSC_VER)
    std::cout << "Compiler:          MSVC " << _MSC_VER << "\n";
#elif defined(__GNUC__)
    std::cout << "Compiler:          gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
#else
    std::cout << "Compiler:          unknown\n";
#endif
    std::cout << "Build type:        "
#if defined(NDEBUG)
              << "NDEBUG (asserts off)\n";
#else
              << "debug (asserts on -- these numbers are not release numbers)\n";
#endif
    std::cout << "Compiled SIMD:     "
#if defined(KINESIS_HAS_AVX2)
              << "AVX2"
#elif defined(KINESIS_HAS_NEON)
              << "NEON"
#else
              << "scalar"
#endif
#if defined(__FMA__)
              << " + FMA"
#endif
              << "\n";
    // FP contraction decides whether a*b+c becomes a single FMA, which changes both the timing
    // and the last bits of the result. Record it rather than leaving it to the reader.
    std::cout << "FP contraction:    "
#if defined(__FP_FAST_FMAF)
              << "fast FMA available for float\n";
#else
              << "no fast float FMA reported by the compiler\n";
#endif
    const auto feats = HardwareInfo::query_cpu_features();
    std::cout << "Runtime CPU:       " << HardwareInfo::cpu_architecture_name() << "\n";
    std::cout << "Cache (L1D/L2/L3): " << (feats.l1_data_cache_size / 1024) << " KB / "
              << (feats.l2_cache_size / 1024) << " KB / " << (feats.l3_cache_size / (1024 * 1024))
              << " MB" << (feats.cache_sizes_measured ? " (measured)" : " (assumed)") << "\n";
    std::cout << "\nNOTE: no CPU affinity is pinned and no frequency governor is fixed, so the\n";
    std::cout << "      p95 column includes scheduler and turbo effects. Compare medians.\n";
}

} // namespace

void benchmark_allocators() {
    constexpr size_t kOpsPerRep = 20'000;
    constexpr size_t kAllocSize = 1024; // 1 KB per allocation
    constexpr size_t kReps = 15;

    print_header("          MEMORY ALLOCATION LATENCY BENCHMARK         ");

    // Baseline 1: the C heap. Allocating and freeing one size repeatedly is the friendliest
    // possible case for a size-binned allocator, so this is a lower bound on malloc, not a
    // typical figure.
    {
        auto stats = measure(kReps, kOpsPerRep, [&] {
            for (size_t i = 0; i < kOpsPerRep; ++i) {
                void* p = std::malloc(kAllocSize);
                if (p == nullptr) {
                    std::cerr << "malloc failed\n";
                    std::abort();
                }
                *static_cast<volatile char*>(p) = 1;
                do_not_optimize(p);
                std::free(p);
            }
        });
        print_row("std::malloc + free (baseline)", stats);
    }

    // Baseline 2: the standard library's own monotonic arena, which is what Kinesis'
    // ArenaAllocator is a re-implementation of. Without this row the arena's number means
    // nothing.
    {
        std::vector<std::byte> backing(64 * 1024 * 1024);
        auto stats = measure(kReps, kOpsPerRep, [&] {
            std::pmr::monotonic_buffer_resource pool(backing.data(), backing.size());
            for (size_t i = 0; i < kOpsPerRep; ++i) {
                void* p = pool.allocate(kAllocSize, 64);
                *static_cast<volatile char*>(p) = 1;
                do_not_optimize(p);
                if ((i & 0x3FF) == 0x3FF) {
                    pool.release();
                }
            }
        });
        print_row("std::pmr::monotonic_buffer (baseline)", stats);
    }

    // Kinesis arena. The arena is constructed and pre-faulted outside the timed region: a fresh
    // 64 MB VirtualAlloc costs one minor fault per 4 KB page on first touch, which is a
    // property of the OS, not of the bump pointer being measured.
    {
        ArenaAllocator arena(64 * 1024 * 1024);
        {
            void* whole = arena.allocate(63 * 1024 * 1024, 64);
            std::memset(whole, 0, 63 * 1024 * 1024);
            clobber_memory();
            arena.reset();
        }
        auto stats = measure(kReps, kOpsPerRep, [&] {
            for (size_t i = 0; i < kOpsPerRep; ++i) {
                void* p = arena.allocate(kAllocSize, 64);
                *static_cast<volatile char*>(p) = 1;
                do_not_optimize(p);
                // Reset periodically, like a 1 kHz robotics or trading loop.
                if ((i & 0x3FF) == 0x3FF) {
                    arena.reset();
                }
            }
            arena.reset();
        });
        print_row("Kinesis ArenaAllocator (pre-faulted)", stats);
    }

    // The same arena without pre-faulting, to show what the first pass actually costs.
    {
        auto stats = measure(kReps, kOpsPerRep, [&] {
            ArenaAllocator arena(64 * 1024 * 1024);
            for (size_t i = 0; i < kOpsPerRep; ++i) {
                void* p = arena.allocate(kAllocSize, 64);
                *static_cast<volatile char*>(p) = 1;
                do_not_optimize(p);
                if ((i & 0x3FF) == 0x3FF) {
                    arena.reset();
                }
            }
        }, 1);
        print_row("  ... including first-touch faults", stats);
    }

    {
        SlabAllocator<kAllocSize> slab(1024);
        auto stats = measure(kReps, kOpsPerRep, [&] {
            for (size_t i = 0; i < kOpsPerRep; ++i) {
                void* p = slab.allocate(kAllocSize);
                *static_cast<volatile char*>(p) = 1;
                do_not_optimize(p);
                slab.deallocate(p);
            }
        });
        print_row("Kinesis SlabAllocator (alloc+free)", stats);
    }
}

void benchmark_tensor_views() {
    constexpr size_t kOpsPerRep = 200'000;
    constexpr size_t kReps = 15;

    print_header("       ZERO-COPY TENSORVIEW SLICING BENCHMARK         ");

    // Initialised: reading an indeterminate float would be UB, and the optimiser assumes it
    // never happens.
    alignas(64) static float raw_buffer[1024];
    for (size_t i = 0; i < 1024; ++i) {
        raw_buffer[i] = static_cast<float>(i) * 0.5f;
    }

    TensorView<float> base_view(raw_buffer, Shape{32, 32});

    // The slice bounds depend on the loop counter, so the slicing cannot be hoisted out of the
    // loop as a loop-invariant expression -- which is what a fixed [4,20)x[8,24) slice was.
    auto stats = measure(kReps, kOpsPerRep, [&] {
        float sum = 0.0f;
        for (size_t i = 0; i < kOpsPerRep; ++i) {
            const int64_t r0 = static_cast<int64_t>(i & 0xF);
            const int64_t c0 = static_cast<int64_t>((i >> 4) & 0xF);
            auto sliced = base_view.slice(0, r0, r0 + 16).slice(1, c0, c0 + 16);
            sum += sliced(0, 0);
            do_not_optimize(sliced);
        }
        do_not_optimize(sum);
    });
    print_row("slice(0,..).slice(1,..) + one read", stats);
    std::cout << "\n  A subview is a descriptor copy (no allocation, no element movement), so\n";
    std::cout << "  this measures descriptor construction plus one strided load.\n";
}

void run_gemm_benchmark_size(int64_t N, size_t reps) {
    ArenaAllocator arena(256 * 1024 * 1024);
    Tensor<float> A(Shape{N, N}, arena);
    Tensor<float> B(Shape{N, N}, arena);
    Tensor<float> C(Shape{N, N}, 0.0f, arena);

    // Varied data: a matrix of all 1.0 and 2.0 gives every FMA the same operands, which is not
    // representative of cache or denormal behaviour.
    for (int64_t i = 0; i < N * N; ++i) {
        A.data()[i] = static_cast<float>((i % 17) - 8) * 0.125f;
        B.data()[i] = static_cast<float>((i % 23) - 11) * 0.0625f;
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);

    auto stats = measure(reps, 1, [&] {
        gemm(A.view(), B.view(), C.view());
        do_not_optimize(C.data()[0]);
        clobber_memory();
    });

    // Checksum, printed so a change in the kernel that alters results cannot pass unnoticed.
    double checksum = 0.0;
    for (int64_t i = 0; i < N * N; ++i) {
        checksum += static_cast<double>(C.data()[i]);
    }

    const double best_gflops = ops / stats.min_ns;      // ns -> GFLOP/s directly
    const double median_gflops = ops / stats.median_ns;

    std::cout << "  " << std::left << std::setw(14) << (std::to_string(N) + "x" + std::to_string(N))
              << std::right << std::fixed << std::setprecision(3) << std::setw(10)
              << stats.min_ns / 1e6 << " ms" << std::setw(10) << stats.median_ns / 1e6 << " ms"
              << std::setprecision(2) << std::setw(10) << best_gflops << std::setw(10)
              << median_gflops << "   checksum " << std::setprecision(4) << checksum << "\n";
}

void benchmark_gemm() {
    print_header("             GEMM PERFORMANCE BENCHMARK               ");
    std::cout << "  " << std::left << std::setw(14) << "size" << std::right << std::setw(13)
              << "best" << std::setw(13) << "median" << std::setw(10) << "GFLOP/s" << std::setw(10)
              << "median" << "\n";
    run_gemm_benchmark_size(256, 20);
    run_gemm_benchmark_size(512, 10);
    run_gemm_benchmark_size(1024, 5);
    std::cout << "\n  NOTE: no baseline library is linked. Until these run next to OpenBLAS,\n";
    std::cout << "  BLIS or Accelerate on the same machine, the GFLOP/s column says how fast\n";
    std::cout << "  this kernel is, not how fast it is compared to anything.\n";
}

void benchmark_memory_sandbox() {
    std::cout << "\n======================================================\n";
    std::cout << "       HARDWARE MEMORY SANDBOX & CACHE EMULATION       \n";
    std::cout << "======================================================\n";
    std::cout << "This is a model, not a measurement: it counts hits and misses under an\n";
    std::cout << "assumed cache geometry. It cannot validate real cache behaviour -- that\n";
    std::cout << "needs hardware counters (perf, VTune, uProf) or a trace simulator.\n\n";

    MemorySandbox sandbox;

    // 1. Sequential stream (GotoBLAS micro-panel style)
    for (size_t i = 0; i < 1024; ++i) {
        sandbox.access_read(0x00040000 + i * sizeof(float), sizeof(float));
    }

    // 2. Strided access across rows
    for (size_t i = 0; i < 128; ++i) {
        sandbox.access_read(0x00080000 + i * 256, sizeof(float));
    }

    sandbox.print_dashboard(std::cout);

    // Shared-memory bank conflicts, swept rather than asserted. For a warp of 32 lanes reading
    // at a constant stride s (in 4-byte words) across 32 banks, the number of lanes landing on
    // the busiest bank is gcd(s, 32): any odd stride is conflict-free, and only a stride that
    // is a multiple of 32 produces the full 32-way serialisation.
    std::cout << "\nGPU shared-memory bank conflicts by word stride (32 banks, 32 lanes):\n";
    std::cout << "  stride(words)   measured degree   gcd(stride,32)\n";
    for (size_t stride_words : {1u, 2u, 3u, 4u, 8u, 16u, 32u, 33u}) {
        std::array<uintptr_t, 32> warp{};
        for (size_t t = 0; t < 32; ++t) {
            warp[t] = 0x1000 + t * stride_words * 4;
        }
        const size_t degree = sandbox.record_warp_bank_access(warp);
        size_t a = stride_words, b = 32;
        while (b != 0) { const size_t tmp = a % b; a = b; b = tmp; }
        std::cout << "  " << std::setw(8) << stride_words << std::setw(16) << degree << "x"
                  << std::setw(17) << a << (degree == a ? "" : "   <-- model disagrees") << "\n";
    }

    std::array<uintptr_t, 32> conflicted_warp{};
    std::array<size_t, 32> hist{};
    for (size_t t = 0; t < 32; ++t) {
        conflicted_warp[t] = 0x1000 + t * 128; // stride 32 words -> full 32-way conflict
        hist[(conflicted_warp[t] / 4) % 32]++;
    }
    std::cout << "\nHistogram for the 32-word stride (every lane on bank 0):\n";
    MemorySandbox::print_bank_histogram(hist, std::cout);

    std::cout << "\n[TIP] For real-time interactive visualization, open:\n";
    std::cout << "      docs/memory_sandbox_visualizer.html in your browser!\n";
    std::cout << "======================================================\n";
}

int main() {
    HardwareInfo::print_system_report(std::cout);
    print_build_configuration();
    benchmark_allocators();
    benchmark_tensor_views();
    benchmark_gemm();
    benchmark_memory_sandbox();
    return 0;
}
