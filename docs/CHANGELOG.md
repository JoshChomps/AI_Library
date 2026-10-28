# Kinesis Development Changelog

All notable changes and architectural milestones for **Kinesis** are documented in this file.

---

## [Unreleased: working tree] - 2026-09-18

### Fixed (all 15 review findings; see [Design Review §6.0](DESIGN_REVIEW_2026-09-17.md#s6-0))
* **Correctness**
  * `gemm_packed` no longer assumes unit column stride, transposed and strided operands fall back to the stride-aware kernel, so `gramian()` is correct again for matrices with ≥ 128 columns.
  * SIMD tail micro-panels use a scalar loop bounded by the real column count. The previous full-width tail **wrote** past the end of the output row on NEON and **read** past it on AVX2; a page-boundary reproduction segfaults against the old code and completes against the new one.
  * `Shape::operator==` no longer type-puns a `uint32_t` array through `const uint64_t*`.
  * Shape, Strides, slice and broadcast validation now throws instead of asserting, so it survives `NDEBUG`. `total_elements()` saturates rather than overflowing.
  * `TensorView<T>` converts to `TensorView<const T>` through a constrained template constructor, so read-only weights reach the kernels and `Tensor::clone()` compiles.
  * The Cramer solvers use a relative singularity threshold, so a well-conditioned system expressed in small units is no longer rejected.
  * BLAS semantics for α = 0: A and B are left unread, so a NaN in A cannot leak into C.
  * `relu(NaN)` returns 0 on NEON as it already did on AVX2.
  * `ArenaAllocator::allocate` cannot be made to wrap its capacity check; `SlabAllocator` validates size, alignment and pointer ownership in release builds.
  * `TensorView::transpose`, `t`, `make_identity` and `copy_from` validate in every build. Found in a follow-up audit: with asserts compiled out, `make_identity` on a non-square view and `copy_from` with a mismatched shape wrote past the end of their buffers, demonstrated with canaries against the old code. `make_identity` is no longer `noexcept`.
* **Honesty of reported hardware**
  * `detect_gpu()` reports `Unknown` with `detected == false` instead of inventing an 8 GB NVIDIA GPU on every x86-64 host.
  * CPU feature detection performs the max-leaf, OSXSAVE and XCR0 checks Intel's procedure requires, and reads cache geometry from CPUID leaf 4 / 0x8000001D. On the development machine this corrected L1D/L2/L3 from 32 KB / 512 KB / 16 MB to **48 KB / 1280 KB / 24 MB**.
  * The system report prints `[NOT IMPLEMENTED]` for the safety policy, which no code reads, instead of `[ACTIVE]`.
* **Performance**
  * `gemm_packed` uses a thread-local packing buffer, removing a heap allocation and a 16 KB zero-fill from every call.
  * `HeapAllocator` is a real aligned-heap allocator instead of an alias for the page allocator, so a small tensor no longer costs a system call and a whole page.
  * `sigmoid` and `gelu` take a contiguous fast path; loop unswitching and a no-op tile loop removed from `gemm_blocked`.

### Added
* `tests/test_regressions.cpp`: one test per finding, each verified to fail against the pre-fix code where that could be demonstrated.
* Second derivatives: `HyperDual` renamed to **`Jet`** (its Ceres name; a deprecated alias remains), and a structural `DualScalar` concept makes `Dual<Dual<T>>` legal. Added `atan2`, `asin`, `acos`, `min`, `max` and `gelu_exact`.
* `MemorySandbox` counts 4 KB page splits separately from cache-line splits, and looks up every line a split access touches.
* CMake options `KINESIS_NATIVE_ARCH` (default ON) and `KINESIS_SANITIZE`.
* **53 test cases, all passing** in both a debug build and `-O3 -DNDEBUG` (up from 35).

### Changed
* `benchmarks/bench_latency.cpp` rewritten: min/median/p95 over repeated runs, initialised buffers, compiler barriers against hoisting, first-touch faults separated out, build configuration printed, and `std::malloc` / `std::pmr::monotonic_buffer_resource` baselines alongside the Kinesis numbers. First result: the arena is **1.15×** faster than `std::pmr::monotonic_buffer_resource` (6.34 vs 7.29 ns), while first-touch faults cost **27.84 ns/alloc**.
* `docs/memory_sandbox_visualizer.html`: the XOR swizzle simulation accounts for the stride and reproduces gcd(s, 32); it now shows the swizzle helping at s = 32 and *hurting* at s = 1 and 33, where the old code claimed a 1.0× win for every stride.
* All three build entry points (`CMakeLists.txt`, `build.ps1`, `build.bat`) list the same ten test files and pass explicit ISA flags.

---

## [Unreleased: documentation review] - 2026-09-17

### Added (documentation)
* **`docs/DESIGN_REVIEW_2026-09-17.md`**: strategy, logic and architecture review of the whole project.
  * Recomputed every checkable formula in the documents (§2.1) and compared each design decision against papers, vendor documentation, leading repositories and Hugging Face resources (§2.2, §2.5).
  * Architecture coherence analysis T1-T17 (§3): where decisions conflict, what the drawbacks are, and what state-of-the-art systems do instead.
  * A CPU/GPU optimization map (§4), an implementation-status matrix (§5), 15 code defects (§6), documentation corrections (§7) and low-level conventions (§8).
* **Review notes** added inline to `README.md`, `ARCHITECTURE.md`, `BENCHMARKS.md`, `LEARNING_GUIDE.md`, `PRODUCTION_SPEC_AND_CRITIQUE_ANALYSIS.md` and `GGML_ANALYSIS_AND_IMPROVEMENTS.md`, plus **Invalidation 19** in `LEARNING_JOURNAL.md`. Original text was left intact.

### Changed (documentation)
* `README.md`: directory tree corrected (`hardware_info.hpp`, `autograd/dual.hpp`, `test_virtual_allocator.cpp`, `test_hardware_info.cpp`, `test_dual.cpp`, the newer docs), test count corrected to **35 test cases**, and compiler detection order corrected to match `build.ps1` (`clang++`, `g++`, `zig c++`, `cl.exe`).
* `LEARNING_GUIDE.md`: removed a duplicated copy of the "Silicon & Hardware Experiments" section so Exercises 1-9 and Experiments 1-4 read in order.
* Machine-specific `file:///c:/Users/...` links replaced with repository-relative links.

### Code already in the working tree (not part of the review; recorded here for traceability)
* **32-bit tensor descriptors**: `Dim = uint32_t` with `Shape` storing `uint32_t` rank + 6 dims (28 bytes); `Strides` keeps `int64_t` strides. Measured `TensorView` size is 104 bytes (see review §2.1 L7).
* **Word-compare equality**: `Shape::operator==` / `Strides::operator==` compare packed 64-bit words at runtime (review flags the `Shape` version as undefined behaviour, §6, defect 3).
* **`Strides(size_t rank)` constructor**, which fixes `broadcast_to` producing rank-0 strides for a rank-N view.
* **Loop-fission `broadcast_to`**, `TensorView::t()`, `operator~`, `make_identity()`, `Tensor::eye()`.
* **`compute::gramian()`** and branchless-style `solve_cramer_2x2` / `solve_cramer_3x3` in `gemm.hpp`.
* **Tests**: 5 new cases (descriptor sizing, operators/identity, loop-fission broadcast, rectangular Gramian, Cramer solvers); `tests/test_dual.cpp` wired into `CMakeLists.txt` and `build.ps1`. **35 test cases total**; pass status was not re-verified by the review itself. The 2026-09-18 remediation pass built and ran them: 35/35 passing at that baseline, 53/53 after the fixes.

### Known defects (as recorded on 2026-09-17; all fixed on 2026-09-18, see above)
* `gemm_packed` assumes unit column stride, so `gramian` is wrong for matrices with ≥ 128 columns; SIMD tails read/write past the end of the output row; `gemm()` heap-allocates per call. See [Design Review §6](DESIGN_REVIEW_2026-09-17.md#s6-1).

---

## [Phase 1: Foundation] - 2026-09-15

### Added
* **Memory Architecture**:
  * `Allocator` abstract base interface (`allocate`, `deallocate`, `owns`, `reset`).
  * `HeapAllocator`: 64-byte aligned heap allocator (`_aligned_malloc` on Windows, `posix_memalign` on POSIX).
  * `ArenaAllocator`: 64-byte cache-aligned monotonic bump allocator with $O(1)$ allocation, $O(1)$ bulk resets, and RAII `ScopedMarker` unwinding for temporary activations.
* **Tensor Stride & View Mathematics**:
  * `Shape`: Non-allocating stack-resident multi-dimensional shape descriptor (up to 6D).
  * `Strides`: Row-major and Col-major stride computation with contiguous layout checks.
  * `TensorView<T>`: Non-owning zero-copy multi-dimensional view with $O(1)$ slicing, $O(1)$ dimension transposition, zero-copy broadcasting (stride = 0), and multi-dimensional indexing `operator()(i, j, ...)`.
  * `Tensor<T>`: Owning tensor backed by an `Allocator` policy with RAII storage management.
* **Compute Kernels**:
  * `simd`: Vectorized intrinsics abstraction supporting AVX2, ARM NEON, and portable scalar fallbacks.
  * `elementwise`: Vectorized `add`, `sub`, `mul`, `scale`, `relu`, `sigmoid`, and `gelu`.
  * `gemm`: Cache-blocked matrix multiplication ($C = \alpha AB + \beta C$) with $L1/L2$ cache tiling and contiguous inner register streams.
  * `gemm_naive`: Reference implementation for numerical correctness validation.
* **Testing & Build Systems**:
  * `test_harness.hpp`: Zero-dependency, single-header C++20 unit test runner.
  * 13 unit tests covering memory alignment, bump allocation, out-of-memory handling, scoped unwinding, stride math, zero-copy slicing, matrix transpositions, broadcasting, elementwise activations, and GEMM equivalence (100% pass rate).
  * `bench_latency.cpp`: Micro-benchmarking allocator latency, view slicing latency, and GEMM GFLOPS.
  * `build.bat` and `build.ps1`: Automated build and test scripts discovering available C++20 compilers (Clang, GCC, Zig, or MSVC).
  * `CMakeLists.txt`: Standard CMake build configuration.
* **Documentation**:
  * Comprehensive `README.md` with high-frequency trading and ROS 2 camera usage examples.
  * `docs/ARCHITECTURE.md`: Deep mathematical and layout specification.
  * `docs/BENCHMARKS.md`: Empirical performance benchmarks.

---

## [Phase 1.5: Silicon Memory & Compute Upgrade] - 2026-09-15

### Added
* **Direct Virtual Memory Subsystem (`virtual_memory.hpp`)**:
  * Direct OS kernel virtual memory manager bypassing C-runtime `malloc`/`free` (`VirtualAlloc`/`VirtualFree` on Windows, `mmap`/`munmap` on POSIX).
  * System page size detection (`4096` bytes baseline, `16384` bytes on Apple Silicon).
  * Page-aligned commit-on-demand and full virtual range reservation.
* **Segregated Slab Allocator / BlockPool (`slab_allocator.hpp`)**:
  * Fixed-size segregated slab allocator with intrusive LIFO singly-linked free-list (`FreeNode` embedded directly within unused block payload, achieving 0 bytes metadata overhead per block).
  * Benchmarked at **$0.64\text{ ns}$ allocation latency** ($40\times$ faster than OS heap `malloc`).
* **VirtualMemoryAllocator Polymorphic Engine (`allocator.hpp`)**:
  * Modern C++20 allocator implementing the `Allocator` interface, completely replacing `_aligned_malloc` as the default backend for owning `Tensor<T>`.
* **Hardware Cache Emulation Sandbox (`memory_sandbox.hpp`)**:
  * 32 KB L1 Data Cache (64-byte line size, 8-way set-associative LRU eviction model).
  * 512 KB L2 Data Cache (8-way set-associative LRU model).
  * Physical cache-line split detection for unaligned memory addresses crossing 64-byte boundaries.
  * GPU 32-Bank shared memory conflict detection with automated Warp Serialization Multiplier calculation ($1\times$ to $32\times$).
  * Visual ANSI/Unicode terminal dashboard rendering (`print_dashboard()`, `print_bank_histogram()`).
* **GotoBLAS Micro-Panel Packing & Dynamic Cache Tiling (`gemm_pack.hpp`, `gemm.cpp`)**:
  * GotoBLAS/BLIS micro-panel memory re-sorting (`pack_B_panel`, $N_r = 8$) for Translation Lookaside Buffer (TLB) thrashing elimination.
  * Hardware cache tile auto-tuning: $B = \sqrt{\frac{\text{L1\_Capacity}}{3 \times \text{sizeof}(T)}}$.
  * Benchmarked at **$29.27\text{ GFLOPS}$ ($256 \times 256$)** and **$26.27\text{ GFLOPS}$ ($512 \times 512$)**.
* **Interactive Web Visualizer (`docs/memory_sandbox_visualizer.html`)**:
  * Dark-mode HTML5/JS hardware visualizer with live address prober, 64-set cache matrix viewer, warp stride slider, and XOR swizzle conflict resolver.
* **Hardware Topology & Silicon Protection Engine (`hardware_info.hpp`)**:
  * Direct CPU architecture & cache topology detection via x86_64 `cpuid` (AVX2, AVX-512, FMA, AMX) and ARM64 NEON.
  * Platform profiling across Desktop/Server, Edge SBC (Raspberry Pi), and Embedded MCU (ESP32).
  * Silicon safety safeguards: transient voltage droop ($V_{droop} = L \frac{di}{dt}$) staged warmup dampening and thermal governor thresholding (85°C warning / 92°C critical throttling).
  * Embedded MCU zero-virtual profile (`KINESIS_EMBEDDED`) routing allocations to static internal SRAM.
* **Extended Test Suite**:
  * 12 new unit tests (25 total, 100% pass rate) validating virtual memory allocation, slab block recycling, cache simulator hit rates, cache-line split triggers, 32-bank warp conflict detection, GotoBLAS packed GEMM numerical equivalence, and HardwareInfo CPU/GPU detection.

---

## [Phase 2: Autodiff & Dual Numbers] - 2026-09-15

### Added
* **Dual Numbers Forward-Mode Automatic Differentiation (`dual.hpp`)**:
  * `Dual<T>`: Pure stack-allocated forward-mode autodiff scalar value-derivative pair ($x + \dot{x}\epsilon$, $\epsilon^2 = 0$) with 0 bytes heap allocation.
  * Complete operator overloads (`+`, `-`, `*`, `/`, compound assignments, scalar interactions, comparisons) strictly adhering to the ring of dual numbers.
  * Transcendental and neural activations: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `pow`, `abs`, `sinh`, `cosh`, `tanh`, `sigmoid`, `relu`, `atan`.
  * `HyperDual<T, N>`: Multi-tangent dual number for evaluating full gradient vectors and Jacobian rows with respect to $N$ independent coordinate variables simultaneously in a single forward pass without building tape graphs.
  * Integration into master header [kinesis.hpp](../include/kinesis/kinesis.hpp).
* **Testing & Verification**:
  * Added `tests/test_dual.cpp` covering basic arithmetic, product rule, quotient rule, chain rule, transcendental functions, multi-variable gradient evaluation, and robotic manipulator forward kinematics Jacobian in 0 microseconds.
  * Total unit tests expanded to **30 passed out of 30 tests (100% pass rate)**.

> **Review note (2026-09-17):** The working tree now contains **35 test cases** (4 arena, 2 virtual memory, 2 slab, 4 sandbox, 3 hardware, 9 tensor view, 6 GEMM, 5 dual). The review did not build or run them, so the pass status should be re-verified, and note that the GEMM suite does not cover partial SIMD panels or strided inputs, which is why two packed-kernel defects went unnoticed. See [Design Review §6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defect 13).


---

## [Phase 2.5: Neural Architecture & Peer-Review Blueprint] - In Planning
* **Forensic Analysis & Fault Rectification**:
  * Completed SOTA literature review of 6 critical algorithmic faults across backprop alternatives, higher-order tensor optimizers (SOAP, Tensor-Muon), and GGML inference debt.
  * Rigorous empirical evaluation of Fault 2: Demonstrated why Direct Feedback Alignment (DFA) / GrAPE does not exceed backpropagation's computational value on dense silicon.
* **Microarchitectural Codebase Audit**:
  * 32-bit descriptor compression: Target 64-byte single-cache-line `TensorView`.
  * $O(1)$ SIMD shape equality comparison (`_mm256_cmpeq_epi32`).
  * Loop fission in broadcasting to eliminate pipeline branch stalls.
  * Prefix `++i` vs. Postfix `i++` and C++20 concept zero-overhead type predicate clarifications.
* **Documentation & Peer-Review Deliverables**:
  * Authored `docs/PRODUCTION_SPEC_AND_CRITIQUE_ANALYSIS.md` for external agent/engineer review.
  * Authored `docs/GGML_ANALYSIS_AND_IMPROVEMENTS.md` dissecting GGML flaws and Kinesis solutions.
  * Appended Invalidation 18 to `docs/LEARNING_JOURNAL.md`.

> **Review note (2026-09-17):** Status after the 2026-09-17 review: the microarchitectural audit items are partly implemented (32-bit descriptors, word-compare equality, loop fission, Gramian, Cramer solvers are in the working tree), but the 64-byte `TensorView` target was not reached (104 bytes), the equality trick is undefined behaviour, and the SOTA literature summary contains citation errors, GrAPE is ICLR 2026 with materially different numbers. See [Design Review §2.1](DESIGN_REVIEW_2026-09-17.md#s2-1) and [§2.4](DESIGN_REVIEW_2026-09-17.md#s2-4).



