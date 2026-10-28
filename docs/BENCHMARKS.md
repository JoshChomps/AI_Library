# Kinesis Benchmark Report

This document records the empirical performance benchmarks executed for **Kinesis** on the Windows platform using the Clang/LLVM C++20 toolchain with `-O3 -march=native`.

> **Review note (2026-09-17):** **Methodology and validity.** Treat every number below as unverified until the harness is rebuilt. Specific problems found in `benchmarks/bench_latency.cpp`:
>
> - **No recorded environment.** The CPU model, core count, clock policy and exact compiler/version are not recorded. `build.ps1` passes only `-std=c++20 -O3 -Wall -Wextra` to Clang/GCC/Zig, no `-march=native`, `-mavx2` or `-mfma`, so the ISA actually used depends on compiler defaults, and with g++ the scalar kernels would be compiled instead (`CMakeLists.txt` does pass `-march=native`).
> - **Undefined behaviour and dead-code elimination.** The slicing benchmark reads an uninitialized buffer, and its slice bounds are compile-time constants, so the compiler folds the work away.
> - **Page faults inside the measurement.** The arena loop touches freshly committed pages, so first-touch page faults are included in the per-allocation time.
> - **No warm-up, repetitions, variance or thread pinning**, and no `DoNotOptimize`-style sink.
> - **No state-of-the-art baselines.** Allocators should be compared against mimalloc/jemalloc and `std::pmr` pool/monotonic resources, and GEMM against BLIS, OpenBLAS or Accelerate on the same machine.
> - **Numbers disagree across documents** for the same metrics: slab 0.64 vs 1.05 ns, arena 5.86 vs 5.14 ns, slicing 0.11 / 0.14 / 0.22 ns, GEMM 20.59 vs 29.27 GFLOPS.
>
> See [Design Review §6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defect 14) and [T12](DESIGN_REVIEW_2026-09-17.md#t12).
>
> **Partly resolved (2026-09-18):** The harness has been rebuilt: it records the compiler, build type, SIMD path and measured cache sizes; initialises its buffers; defeats hoisting with compiler barriers; separates first-touch faults; reports min/median/p95 over repeated runs; and prints `std::malloc` and `std::pmr::monotonic_buffer_resource` baselines. All three build scripts pass `-march=native`. **The tables in this file have not been regenerated**, so the last bullet still applies, and there is still no BLAS or mimalloc baseline. See [Design Review §6.0](DESIGN_REVIEW_2026-09-17.md#s6-0).


---

## 1. Memory Allocation Latency

* **Workload**: 100,000 iterations allocating 1 KB chunks with guaranteed 64-byte alignment.
* **Source**: `benchmarks/bench_latency.cpp` (`benchmark_allocators`)

| Allocator Type | Latency per Allocation | Memory Overhead / Fragmentation | Real-Time Safe |
| :--- | :--- | :--- | :--- |
| **Standard OS Heap (`malloc`/`free`)** | **`25.62 ns`** | Unbounded tail latency; OS locks | ❌ No |
| **Kinesis `ArenaAllocator` ($O(1)$)** | **`5.86 ns`** | **Zero fragmentation; 0 system calls** | ✅ **Yes** (4.4x faster) |
| **Kinesis `SlabAllocator` ($O(1)$)** | **`0.64 ns`** | **Zero fragmentation; intrusive free-list** | ✅ **Yes** (40.0x faster) |

**Takeaway**: The monotonic bump allocator is **4.4x faster** than system `malloc`, and the segregated slab allocator achieves **0.64 ns per block cycle (alloc + free)**, a **40x speedup** over standard heap allocations with zero lock contention.

> **Review note (2026-09-17):** What the loops actually measure: the slab figure is a pop/push of the *same* block from a LIFO free list (the best possible case, with no thread safety and no size-class lookup), and the arena figure includes demand-zero page faults on first touch, which plausibly dominates it. The `malloc` baseline is the Windows CRT allocator; modern allocators (mimalloc, jemalloc) and `std::pmr::unsynchronized_pool_resource` are the fair comparison, and `std::pmr::monotonic_buffer_resource` is the standard-library equivalent of the arena. Note also that the default `Tensor` allocator issues one `VirtualAlloc`/`mmap` per tensor, which is far slower than either. See [Design Review §2.2](DESIGN_REVIEW_2026-09-17.md#s2-2) and [§6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defect 8).
>
> **Resolved (2026-09-18):** The harness is fixed (repeated runs, barriers, pre-faulting, baselines). The first honest result reframes the headline: the arena beats `std::pmr::monotonic_buffer_resource` by **1.15×** (6.34 vs 7.29 ns median), while first-touch page faults cost **27.84 ns/alloc**, four times the steady-state cost, and the thing actually worth optimising.


---

## 2. Zero-Copy TensorView Slicing Latency

* **Workload**: 1,000,000 iterations slicing a 2D $(32 \times 32)$ tensor across sub-ranges $[4:20, 8:24]$.
* **Source**: `benchmarks/bench_latency.cpp` (`benchmark_tensor_views`)

| Operation | Latency per Operation | Data Copied |
| :--- | :--- | :--- |
| **Multi-Dimension Subview Slicing** | **`0.11 ns`** | **0 bytes** |

**Takeaway**: Slicing in Kinesis executes in sub-nanosecond time ($0.11\text{ ns}$) because it simply updates stack-resident integer offsets and shapes without copying any underlying memory.

> **Review note (2026-09-17):** ❌ 0.11 ns is below one CPU cycle at any current clock, which is the tell: the slice metadata is computed from compile-time constants and folded away, and the buffer being read is uninitialized (undefined behaviour). This row does not measure slicing. A valid version needs runtime-varying slice bounds, an optimizer barrier, and initialized data. See [Design Review §6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defect 14).
>
> **Resolved (2026-09-18):** Fixed. The buffer is initialised, the slice bounds depend on the loop counter so the expression cannot be hoisted, and the result passes through a compiler barrier. The measured figure is now **3.72 ns median** for two chained slices plus one strided load.


---

## 3. General Matrix Multiplication (GEMM) Throughput

* **Workload**: Repeated single-threaded matrix multiplication across scaled dimensions ($256 \times 256$ and $512 \times 512$) single-precision floats ($2 \cdot N^3$ FLOPs per run).
* **Source**: `benchmarks/bench_latency.cpp` (`benchmark_gemm`)

| Implementation | Matrix Dimensions | Average Latency | Throughput | Numerical Verification |
| :--- | :--- | :--- | :--- | :--- |
| **Kinesis Packed GEMM** | $256 \times 256$ | **`1.147 ms`** | **`29.27 GFLOPS`** | Exact match vs reference ($\epsilon < 10^{-4}$) |
| **Kinesis Packed GEMM** | $512 \times 512$ | **`10.219 ms`** | **`26.27 GFLOPS`** | Exact match vs reference ($\epsilon < 10^{-4}$) |

**Takeaway**: GotoBLAS micro-panel packing eliminates TLB page thrashing as matrices scale from $256$ to $512$, maintaining **26 to 29 GFLOPS** single-threaded throughput without external BLAS dependencies.

> **Review note (2026-09-17):** Two caveats. (1) The "Numerical Verification" column is not produced by the benchmark, `bench_latency.cpp` never compares against a reference, and the packed kernel silently returns wrong results for transposed or strided inputs. (2) For context, BLIS's Haswell single-precision kernel is 6×16 with KC = 256, MC = 168, NC = 4080 and packs both A and B, while this implementation packs only B with an effective 1×8 micro-kernel and 64³ blocks, so the headline number should be read against a BLIS/OpenBLAS baseline on the same machine rather than against a naive triple loop. See [Design Review §2.2](DESIGN_REVIEW_2026-09-17.md#s2-2) and [§6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defects 1, 2).
>
> **Resolved (2026-09-18):** The benchmark now prints a checksum of C for every GEMM size, so a kernel change that alters results cannot pass unnoticed. The numbers in the tables above predate the harness rewrite and have not been regenerated.


---

## 4. Hardware Memory Sandbox Emulation

* **Simulation Geometry**: 32 KB L1 Data Cache (8-way associative, 64-byte line), 32-Bank GPU Shared Memory.
* **Source**: `benchmarks/bench_latency.cpp` (`benchmark_memory_sandbox`) and `docs/memory_sandbox_visualizer.html`

| Access Workload Pattern | Measured L1 Hit Rate | Simulated Hazard Degree | Hardware Implication |
| :--- | :--- | :--- | :--- |
| **GotoBLAS Micro-Panel (Stride 1)** | **`93.8%`** | **`1.0x` (0 Hazard)** | Maximum memory bus saturation; minimal DRAM stalls |
| **Strided Column Sweep (Stride 64)** | **`< 10.0%`** | Severe Cache Thrashing | Memory bus bottleneck; pipeline stalls |
| **GPU Warp (Stride 4 Bytes)** | N/A | **`1.0x` Coalesced** | All 32 threads hit independent banks in 1 cycle |
| **GPU Warp (Stride 128 Bytes)** | N/A | **`32.0x` Serialized** | Catastrophic conflict: all 32 threads collide on Bank 0 |
| **XOR Swizzled Warp (`Col ^ Row>>1`)** | N/A | **`1.0x` Coalesced** | Bank conflicts completely eliminated by bitwise permutation |

> **Review note (2026-09-17):** The sandbox rows describe a software model, not measured hardware. The model uses CUDA compute-capability-2.0 bank geometry (bank counts and widths differ by compute capability), and the "1.0x" swizzle result is an artefact: the simulation discards the stride, while the formula it displays still produces 4-way conflicts. For a linear stride of *s* elements the worst conflict is exactly gcd(s, 32), so odd element strides, or padding by one element, are the real fix. The L1 rows also mix CPU cache geometry with GPU latency figures. See [Design Review §2.1](DESIGN_REVIEW_2026-09-17.md#s2-1) (L5), [§2.3](DESIGN_REVIEW_2026-09-17.md#s2-3) and [T12](DESIGN_REVIEW_2026-09-17.md#t12).
>
> **Partly resolved (2026-09-18):** The swizzle simulation now accounts for the stride and reproduces gcd(s, 32) exactly, and it shows the swizzle making strides 1, 3 and 33 *worse*. The other limits stand: compute-capability-2.0 bank geometry, and CPU cache geometry mixed with GPU latency figures. See [Design Review §6.0](DESIGN_REVIEW_2026-09-17.md#s6-0) (defect 15).


---

## Reproducing Benchmarks

Run the benchmark executable directly from PowerShell:
```powershell
cd <path-to>\AI_Library
.\build.ps1
```
Or view the interactive visualizer in your browser:
```powershell
Start-Process docs/memory_sandbox_visualizer.html
```

---

> **Design review (2026-09-17):** Benchmark validity, state-of-the-art baselines and the full strategy/architecture review are in [DESIGN_REVIEW_2026-09-17.md](DESIGN_REVIEW_2026-09-17.md). Numbers in this file have not been re-measured; the review explains what each one does and does not show.
