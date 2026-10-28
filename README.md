# Kinesis: Unified Low-Level C++ AI Engine

**Kinesis** is a high-performance, low-level machine learning and mathematical computation engine engineered from scratch in **Modern C++20**.

Unlike mainstream frameworks (PyTorch, TensorFlow) designed primarily for offline datacenter training with heavy Python runtimes, Kinesis is built specifically for systems demanding **sub-microsecond determinism, zero heap churn, and complete memory layout control**:

* **Quantitative Finance**: Sub-microsecond market tick processing, streaming causal 1D tensors, and online adaptive learning.
* **Robotics & Mechatronics**: Real-time control loops, Lie algebra manifolds ($SO(3), SE(3)$), and forward-mode automatic differentiation (Dual numbers for instantaneous Jacobians).
* 👁️ **Computer Vision**: Spatial convolutions (`im2col`, Winograd $F(2\times 2, 3\times 3)$), zero-copy frame slicing, and camera projection geometry.
* 🔄 **ROS 2 Zero-Copy**: Direct non-owning ingestion of camera frames and sensor streams via `rclcpp::LoanedMessage` and shared memory (`iceoryx`).

> **Review note (2026-09-17):** Kinesis is intended to be a CPU **and** GPU library. Today only CPU kernels (AVX2 / NEON / scalar) exist, and the domain modules (vision, quant, robotics, ROS 2 adapters, autodiff beyond dual numbers) are plans. GPU support needs architectural prerequisites that the current `TensorView`/`Allocator` design lacks, memory spaces and buffer types, streams and events, device allocators, a scheduler, and a dtype system. See [Design Review T13](docs/DESIGN_REVIEW_2026-09-17.md#t13) and the status matrix in [§5](docs/DESIGN_REVIEW_2026-09-17.md#s5).


---

## Key Architectural Principles

### 1. Zero Runtime Heap Allocation (The Dual-Arena Model)
Standard memory allocators (`malloc`/`free`) suffer from unbounded tail latencies, lock contention, and heap fragmentation. Kinesis enforces **Zero Allocations in the Hot Loop**:
* **Permanent Parameter Arena**: Holds static model weights and optimizer states, populated once at initialization.
* **Scratch Activation Arena (`ArenaAllocator`)**: A 64-byte cache-aligned monotonic bump allocator. Allocating an intermediate tensor is a simple pointer bump ($O(1)$). Memory can be rewound automatically using RAII `ScopedMarker` or bulk-cleared in $O(1)$ time at the end of each tick/frame.

> **Review note (2026-09-17):** The arena itself is sound, but "zero allocations in the hot loop" is not yet guaranteed: `gemm()` heap-allocates a packing buffer on every call for matrices ≥ 128, the default `Tensor` allocator issues one `VirtualAlloc`/`mmap` per tensor, a full arena throws `std::bad_alloc` mid-tick, and arena pages are not prefaulted. There is no separate permanent-parameter arena facility yet. See [Design Review T1](docs/DESIGN_REVIEW_2026-09-17.md#t1) and [T2](docs/DESIGN_REVIEW_2026-09-17.md#t2).
>
> **Resolved (2026-09-18):** Partly fixed. `gemm()` now packs into an `alignas(64)` thread-local block, so it allocates nothing on the hot path, and the default `Tensor` allocator is a real aligned-heap allocator instead of one `VirtualAlloc`/`mmap` per tensor. The structural items stand: a full arena still throws mid-tick, pages are still not prefaulted, and there is still no permanent-parameter arena, those are [T1](docs/DESIGN_REVIEW_2026-09-17.md#t1)/[T2](docs/DESIGN_REVIEW_2026-09-17.md#t2) design decisions. The rewritten benchmark shows why prefaulting matters: first-touch faults cost **27.84 ns/alloc** against **6.34 ns** steady state.


### 2. Decoupled Storage and Zero-Copy `TensorView`
Memory storage is completely decoupled from tensor shape and layout:
$$\text{Index}(i_0, i_1, \dots, i_{D-1}) = \text{offset} + \sum_{k=0}^{D-1} i_k \cdot \text{strides}_k$$

* **Zero-Copy Slicing**: Slicing along any dimension merely updates the memory `offset` and dimension `shape`. Zero bytes are copied.
* **Zero-Copy Transposition**: Transposing dimensions swaps stride values in $O(1)$ time without modifying underlying memory.
* **Zero-Copy Broadcasting**: Expanding dimensions of size 1 sets the corresponding stride to `0`. Multiple indices map to the same physical address without memory duplication.

### 3. Cache-Blocked & Vectorized Compute Kernels
Matrix multiplication (GEMM) avoids cache line thrashing by partitioning matrices into $L1/L2$ cache tiles ($M_B \times K_B \times N_B$) and computing contiguous register blocks with SIMD intrinsics (AVX2 / AVX-512 FMA).

> **Review note (2026-09-17):** No AVX-512 kernels exist yet (AVX2 FMA, NEON and scalar only), and the SIMD path is selected at compile time. The packed GEMM path has two correctness defects (strided/transposed inputs give wrong results; SIMD tails read/write past the row end) and packs only matrix B. See [Design Review §6](docs/DESIGN_REVIEW_2026-09-17.md#s6-1) (defects 1, 2, 5) and [§2.2](docs/DESIGN_REVIEW_2026-09-17.md#s2-2).
>
> **Resolved (2026-09-18):** The AVX2 guard now also requires FMA, and all three build entry points pass `-march=native`. AVX-512 kernels and runtime dispatch are still absent.


---

## Directory Structure

```
AI_Library/
├── CMakeLists.txt
├── README.md
├── include/
│   └── kinesis/
│       ├── kinesis.hpp               <-- Master include
│       ├── core/                      <-- Memory & Tensor Foundations
│       │   ├── types.hpp              <-- Concepts, adaptive alignment (32B/64B/128B)
│       │   ├── virtual_memory.hpp     <-- Direct OS kernel pages (VirtualAlloc / mmap)
│       │   ├── allocator.hpp          <-- Abstract Allocator & VirtualMemoryAllocator
│       │   ├── arena_allocator.hpp    <-- 64-byte Aligned Monotonic Bump Allocator
│       │   ├── slab_allocator.hpp     <-- Segregated Slab Pool with Intrusive Free-Lists (0.64ns)
│       │   ├── memory_sandbox.hpp     <-- Hardware L1/L2 cache & 32-bank warp conflict simulator
│       │   ├── hardware_info.hpp      <-- CPU/GPU topology report & safety policy (see Design Review T11/T17)
│       │   ├── shape.hpp              <-- Non-allocating stack Shape & Strides
│       │   ├── tensor_view.hpp        <-- Zero-copy multi-dimensional non-owning view
│       │   └── tensor.hpp             <-- Owning Tensor backed by Allocators
│       ├── compute/                   <-- Hardware Compute Kernels
│       │   ├── simd.hpp               <-- AVX2 / NEON / Scalar vector intrinsics
│       │   ├── elementwise.hpp        <-- Vectorized add, mul, relu, gelu, sigmoid
│       │   ├── gemm_pack.hpp          <-- GotoBLAS micro-panel memory re-sorting (Nr = 8)
│       │   └── gemm.hpp               <-- Cache-blocked & packed matrix multiplication
│       ├── autograd/                  <-- Forward-mode autodiff
│       │   └── dual.hpp               <-- Dual<T> / HyperDual<T, N> (first-order multi-tangent duals)
│       ├── nn/                        <-- Phase 2: Neural Network Modules
│       ├── vision/                    <-- Phase 3: Spatial Computer Vision
│       ├── quant/                     <-- Phase 3: Financial Time-Series
│       └── robotics/                  <-- Phase 4: Lie Groups & ROS 2 Adapters
├── src/
│   ├── core/arena_allocator.cpp
│   └── compute/gemm.cpp
├── tests/
│   ├── test_harness.hpp               <-- Zero-dependency single-header test runner
│   ├── test_main.cpp                  <-- Test runner entry point (53 test cases, all passing)
│   ├── test_arena_allocator.cpp
│   ├── test_tensor_view.cpp
│   ├── test_gemm.cpp
│   ├── test_virtual_allocator.cpp
│   ├── test_slab_allocator.cpp
│   ├── test_memory_sandbox.cpp
│   ├── test_hardware_info.cpp
│   ├── test_dual.cpp
│   └── test_regressions.cpp           <-- One case per 2026-09-17 review finding
├── benchmarks/
│   └── bench_latency.cpp              <-- Latency & visual sandbox telemetry benchmarks
└── docs/
    ├── ARCHITECTURE.md                <-- Systems-level specification & Pragmatism Principle
    ├── BENCHMARKS.md                  <-- Empirical latency tables (0.64ns Slab, 29+ GFLOPS)
    ├── LEARNING_GUIDE.md              <-- 9 hands-on exercises & 4 hardware experiments
    ├── LEARNING_JOURNAL.md            <-- Detailed engineering dialogues & invalidation records
    ├── PRODUCTION_SPEC_AND_CRITIQUE_ANALYSIS.md <-- Production blueprint written for peer review
    ├── GGML_ANALYSIS_AND_IMPROVEMENTS.md <-- GGML comparison (read with its review notes)
    ├── DESIGN_REVIEW_2026-09-17.md    <-- Strategy, logic & architecture review with sources
    ├── CHANGELOG.md                   <-- Development milestones
    └── memory_sandbox_visualizer.html <-- Interactive dark-mode hardware cache visualizer
```

---

## Quick Start & Usage Examples

### 1. High-Frequency Market Tick Processing (Zero Allocations)
```cpp
#include <kinesis/kinesis.hpp>

using namespace kinesis::core;
using namespace kinesis::compute;

// Pre-allocate a 16MB aligned scratchpad at startup
ArenaAllocator tick_arena(16 * 1024 * 1024);

void on_market_order_book_tick(const float* raw_tick_features, size_t feature_count) {
    // 1. O(1) Instant reset of all temporary activation memory
    tick_arena.reset();

    // 2. Zero-copy wrap raw packet data from network card / kernel-bypass ring buffer
    auto input_view = TensorView<float>::from_borrowed(
        const_cast<float*>(raw_tick_features), 
        Shape{1, static_cast<int64_t>(feature_count)}
    );

    // 3. Intermediate tensors bump pointers inside the arena (zero heap malloc)
    Tensor<float> hidden(Shape{1, 64}, tick_arena);
    Tensor<float> weights(Shape{static_cast<int64_t>(feature_count), 64}, tick_arena);
    
    // 4. Execute inference kernel
    gemm(input_view, weights.view(), hidden.view());
    relu(hidden.view(), hidden.view());

    // 5. Output trade decision with deterministic sub-microsecond latency

> **Review note (2026-09-17):** As written, `weights` is allocated from `tick_arena` after `reset()`, so it holds uninitialized memory on every tick, real weights belong in memory that outlives the tick (e.g. a separate arena populated at start-up). Note also that `gemm` heap-allocates for matrices ≥ 128 × 128, and that `const` tensors cannot currently be passed to kernels. See [Design Review T1](docs/DESIGN_REVIEW_2026-09-17.md#t1), [T2](docs/DESIGN_REVIEW_2026-09-17.md#t2) and [T4](docs/DESIGN_REVIEW_2026-09-17.md#t4).
>
> **Resolved (2026-09-18):** The two code defects are fixed: `gemm` no longer heap-allocates, and `const` tensors reach the kernels through a `TensorView<T>` → `TensorView<const T>` conversion. The example itself still allocates `weights` from the tick arena, which is the lifetime question in [T2](docs/DESIGN_REVIEW_2026-09-17.md#t2).

}
```

### 2. ROS 2 Zero-Copy Camera Frame Slicing
```cpp
#include <kinesis/kinesis.hpp>

using namespace kinesis::core;

void on_camera_frame_received(uint8_t* raw_image_buffer, int height, int width, int channels) {
    // Zero-copy 3D view over raw shared-memory sensor buffer
    TensorView<uint8_t> frame(raw_image_buffer, Shape{height, width, channels});

    // Zero-copy crop center region of interest [100:300, 150:450]
    // Modifies only offset and shape; zero memory copying occurs
    TensorView<uint8_t> roi = frame.slice(0, 100, 300).slice(1, 150, 450);

    // Feed ROI directly to feature detectors or neural network
}
```

---

## Empirical Performance Highlights

Micro-benchmarked on modern silicon with native Clang/AVX2:

| Subsystem / Kernel | Baseline / Standard | Kinesis Implementation | Measured Speedup |
| :--- | :--- | :--- | :--- |
| **Fixed-Size Allocations** | Standard `malloc`/`free`: $25.62\text{ ns}$ | **`SlabAllocator<1024>`: $0.64\text{ ns}$** | **$40.0\times$ faster** |
| **Dynamic Scratch Buffers** | Standard `malloc`/`free`: $25.62\text{ ns}$ | **`ArenaAllocator`: $5.86\text{ ns}$** | **$4.4\times$ faster** |
| **Tensor Subview Slicing** | Dynamic Tensor Allocation | **`TensorView` Slicing: $0.11\text{ ns}$** | **Zero-Copy ($O(1)$)** |
| **Matrix Multiply ($256 \times 256$)** | Naive GEMM: $0.85\text{ GFLOPS}$ | **GotoBLAS Packed GEMM: $29.27\text{ GFLOPS}$** | **$34.4\times$ throughput** |
| **Matrix Multiply ($512 \times 512$)** | Naive GEMM: $0.72\text{ GFLOPS}$ | **GotoBLAS Packed GEMM: $26.27\text{ GFLOPS}$** | **$36.5\times$ throughput** |

> **Review note (2026-09-17):** Treat these numbers as unverified. The slicing benchmark reads an uninitialized buffer and its slice bounds are compile-time constants (0.11 ns is below one CPU cycle, so it does not measure slicing); the arena timing includes first-touch page faults; no code in `benchmarks/` produces the naive-GEMM baselines or the speedup column; the build script passes no `-march`/`-mavx2`/`-mfma` flags for GCC/Clang; and there are no state-of-the-art baselines (BLIS/OpenBLAS/Accelerate, mimalloc, `std::pmr`). See [Design Review §6](docs/DESIGN_REVIEW_2026-09-17.md#s6-1) (defect 14) and [T12](docs/DESIGN_REVIEW_2026-09-17.md#t12).
>
> **Resolved (2026-09-18):** The benchmark harness has been rewritten and the defects it describes are fixed, but **the numbers in this table have not been regenerated** and still should not be trusted. See [Design Review §6.0](docs/DESIGN_REVIEW_2026-09-17.md#s6-0) for what the repaired harness reports.


---

## Hardware Memory Sandbox & Visual Telemetry

Physical hardware caches often mask memory access hazards due to aggressive prefetchers. Kinesis includes a dual-layer hardware simulation environment:

1. **Terminal Console Telemetry**:
   Run `build.ps1` or `./bin/kinesis_benchmarks.exe` to see visual Unicode/ASCII gauge bars and 32-bank thread conflict histograms directly in your shell.
2. **Interactive Browser Visualizer (`docs/memory_sandbox_visualizer.html`)**:
   Open `docs/memory_sandbox_visualizer.html` in any web browser to interactively explore:
   * **GotoBLAS Micro-Panel Streaming**: Visualizing $>93\%$ L1 cache hit rates.
   * **Cache-Line Split Detector**: Interactive address probing for split penalties across 64-byte line boundaries.
   * **GPU 32-Bank Coalescing**: Real-time slider controlling warp access strides ($4\text{B}$ to $256\text{B}$) with live conflict heatmaps.
   * **XOR Swizzle Simulation**: Real-time verification of bitwise swizzling eliminating $32\times$ serialization hazards.

> **Review note (2026-09-17):** The sandbox and visualizer are teaching tools, not verification oracles: they model a CPU L1 cache with GPU latency figures, take hand-written addresses rather than traces of the real kernels, and the XOR swizzle simulation discards the stride (the formula it displays still yields 4-way conflicts). See [Design Review §2.1](docs/DESIGN_REVIEW_2026-09-17.md#s2-1) (L5) and [T12](docs/DESIGN_REVIEW_2026-09-17.md#t12).
>
> **Resolved (2026-09-18):** The swizzle defect is fixed: `simulateWarp` accounts for the stride and reproduces gcd(s, 32) exactly, and it now shows the swizzle *hurting* strides that were already conflict-free. A split access also looks up every line it touches, and page splits are counted separately. The framing still holds, these model an assumed geometry from hand-fed addresses, and cannot verify real cache behaviour.


---

## Building and Testing

### Quick Build (PowerShell / Windows)
```powershell
.\build.ps1
```
Automatically detects your installed C++20 compiler (checked in order: `clang++`, `g++`, `zig c++`, `cl.exe`), compiles the 53 test cases, runs them, compiles micro-benchmarks, and renders the Memory Sandbox telemetry dashboard.

> **Review note (2026-09-17):** `build.ps1` passes only `-std=c++20 -O3 -Wall -Wextra` to GCC/Clang/Zig (no `-march`, `-mavx2` or `-mfma`), so kernel selection depends on the compiler's defaults; CMake uses `-march=native` (non-portable binaries); and `build.bat` omits `test_hardware_info.cpp` and `test_dual.cpp`. See [Design Review §6](docs/DESIGN_REVIEW_2026-09-17.md#s6-1) (defect 5) and [§6.2](docs/DESIGN_REVIEW_2026-09-17.md#s6-2).
>
> **Resolved (2026-09-18):** Fixed. `build.ps1` and `build.bat` pass `-march=native` (and `-DNDEBUG` for benchmarks), and `CMakeLists.txt` exposes `KINESIS_NATIVE_ARCH` (default ON, falling back to `-mavx2 -mfma` when off) plus `KINESIS_SANITIZE`.


### CMake Build (Cross-Platform)
```bash
# Configure build directory
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build library, tests, and benchmarks
cmake --build build --config Release

# Run unit tests
./build/kinesis_tests

# Run benchmarks
./build/kinesis_benchmarks
```

