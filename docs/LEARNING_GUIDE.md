# Kinesis Mastery & Learning Guide

Welcome to the hands-on engineering track for **Kinesis**. This guide is structured so you can **type out, experiment with, break, and re-implement** every core subsystem to develop deep muscle memory for systems programming, SIMD vectorization, and low-level AI math.

---

## 1. Practice Modules: What to Re-Type from Scratch

Don't just read the code, re-type these core sections in a scratch file (e.g. `tests/scratch_practice.cpp`), build them with `build.ps1`, and inspect the assembly.

> **Review note (2026-09-17):** Some snippets below have drifted from the current sources (Exercises 3 and 6 in particular), and some observations they ask you to reproduce come from benchmarks with known validity problems. Where that is the case there is a note. A full strategy, logic and architecture review of the project, with sources, is in [DESIGN_REVIEW_2026-09-17.md](DESIGN_REVIEW_2026-09-17.md).


### Exercise 1: The Fast Alignment Bitmask
* **File to study**: `include/kinesis/core/types.hpp` and `src/core/arena_allocator.cpp`
* **The Code**:
  ```cpp
  constexpr size_t align_up(size_t address, size_t alignment) noexcept {
      return (address + alignment - 1) & ~(alignment - 1);
  }
  ```
* **What to practice**:
  1. Write out the binary representation of `alignment = 64` (`0b01000000`), `alignment - 1` (`0b00111111`), and `~(alignment - 1)` (`0b11000000`).
  2. Prove mathematically why `& ~(alignment - 1)` zeros out the lower 6 bits (which represent values $0 \dots 63$).
  3. **Experiment to break it**: Pass an alignment that is NOT a power of 2 (e.g. `alignment = 60`). Observe how the bitwise formula fails, and explain why `assert((alignment & (alignment - 1)) == 0)` is mandatory.

---

### Exercise 2: Multi-Dimensional Stride Arithmetic
* **File to study**: `include/kinesis/core/shape.hpp` and `include/kinesis/core/tensor_view.hpp`
* **The Code**:
  ```cpp
  int64_t flat_index = offset;
  for (size_t d = 0; d < rank; ++d) {
      flat_index += indices[d] * strides[d];
  }
  ```
* **What to practice**:
  1. Manually calculate the row-major strides for shape $(4, 3, 2)$:
     $$\text{stride}_2 = 1, \quad \text{stride}_1 = 1 \times 2 = 2, \quad \text{stride}_0 = 2 \times 3 = 6 \implies (6, 2, 1)$$
  2. **Implement Slicing on Paper**:
     Take shape $(4, 3, 2)$ with strides $(6, 2, 1)$ and offset $0$.
     Slice dimension $0$ from $[1:3]$ (size 2) and dimension $1$ from $[2:3]$ (size 1).
     * New offset: $\text{offset} = 0 + 1 \times 6 + 2 \times 2 = 10$.
     * New shape: $(2, 1, 2)$.
     * Strides remain $(6, 2, 1)$.
  3. **Implement Broadcasting**:
     Set $\text{stride}_d = 0$ for a dimension with size $1$. Verify why `index[d] * 0` always yields the exact same memory address.

---

### Exercise 3: Cache-Blocked Matrix Multiplication (GEMM)
* **File to study**: `src/compute/gemm.cpp`
* **The Code to Type**:
  ```cpp
  for (int64_t m = m_tile; m < m_end; ++m) {
      for (int64_t k = k_tile; k < k_end; ++k) {
          float a_val = a(m, k);
          for (int64_t n = n_tile; n < n_end; ++n) {
              c(m, n) += a_val * b(k, n);
          }
      }
  }
  ```
* **Key Experiment**:
  1. Change the loop order from `m, k, n` to the naive `m, n, k`:
     ```cpp
     // NAIVE CACHE-THRASHING LOOP:
     for (int64_t m = 0; m < M; ++m) {
         for (int64_t n = 0; n < N; ++n) {
             for (int64_t k = 0; k < K; ++k) {
                 c(m, n) += a(m, k) * b(k, n); // b(k, n) jumps by N on every step!
             }
         }
     }
     ```
  2. Run `bench_latency.cpp` comparing both. Observe how `m, k, n` is **5x to 15x faster** purely because $B(k, n)$ and $C(m, n)$ are accessed contiguously along row memory!

> **Review note (2026-09-17):** The snippet above is a simplified sketch: the current `gemm_blocked` also has AVX2/NEON inner loops and a contiguity check, and `gemm()` dispatches to `gemm_packed` once M and N reach 128. The loop-order lesson is real, but the bigger lever in a production kernel is **register tiling**, keeping an MR×NR block of C in registers (ATLAS/BLIS; Yotov et al., SPAA 2007). BLIS's Haswell single-precision kernel is 6×16; the Kinesis packed kernel effectively accumulates 1×8, so this is the main performance gap to study. See [Design Review §2.2](DESIGN_REVIEW_2026-09-17.md#s2-2).


---

### Exercise 4: Direct Virtual Memory (`VirtualAlloc` / `mmap`) vs. `malloc`
* **File to study**: `include/kinesis/core/virtual_memory.hpp` and `include/kinesis/core/allocator.hpp`
* **The Code to Type**:
  ```cpp
  // Windows: VirtualAlloc kernel bypass
  void* ptr = ::VirtualAlloc(nullptr, aligned_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  // POSIX: mmap kernel bypass
  void* ptr = ::mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ```
* **What to practice**:
  1. Explain why bypassing the C runtime `malloc` eliminates thread-local bucket lock contention in multi-threaded trading engines.
  2. Prove how allocating in multiples of the operating system page size (4 KB on Windows/Linux, 16 KB on Apple Silicon) guarantees that every allocation begins at a cache-line boundary without padding waste.

> **Review note (2026-09-17):** Careful, the opposite lesson also matters. Page-granular allocation means *every* allocation costs at least one page: a 100-byte request wastes about 97% of a 4 KB page, and on Windows each `VirtualAlloc` reservation consumes a 64 KB granule of address space. That is why general-purpose allocators (and `std::pmr` pools) reserve large regions once and sub-allocate. Kinesis currently routes every `Tensor` through one `VirtualAlloc`/`mmap`, which is the costly path. See [Design Review §2.2](DESIGN_REVIEW_2026-09-17.md#s2-2) and [§6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defect 8).
>
> **Resolved (2026-09-18):** Fixed. `HeapAllocator` is now a real aligned-heap allocator rather than an alias for the page allocator, so a small tensor no longer costs a system call and a whole page; `VirtualMemoryAllocator` is named explicitly where page granularity is actually wanted, such as the arena's backing region.


---

### Exercise 5: Segregated Slab Allocator with Intrusive LIFO Free-Lists
* **File to study**: `include/kinesis/core/slab_allocator.hpp`
* **The Code to Type**:
  ```cpp
  struct FreeNode {
      FreeNode* next;
  };
  
  void* allocate() {
      if (!free_list_) return nullptr;
      FreeNode* node = free_list_;
      free_list_ = node->next; // O(1) Pop
      return static_cast<void*>(node);
  }
  
  void deallocate(void* ptr) {
      FreeNode* node = static_cast<FreeNode*>(ptr);
      node->next = free_list_; // O(1) Push
      free_list_ = node;
  }
  ```
* **What to practice**:
  1. Understand the magic of an **intrusive linked list**: when a block is free, its memory stores the pointer to the next free block (`sizeof(FreeNode*) = 8` bytes). When the user allocates the block, their data overwrites that pointer! **Zero memory overhead per block**.
  2. Run `bench_latency.cpp` and observe why the `SlabAllocator` achieves **$0.64\text{ ns}$ allocation latency** ($40\times$ faster than standard heap `malloc`).

> **Review note (2026-09-17):** What that number measures: popping and pushing the *same* block of a LIFO free list, single-threaded, with no size-class lookup, the best possible case. The arena figure in the same benchmark includes first-touch page faults. Compare against mimalloc/jemalloc and `std::pmr::unsynchronized_pool_resource` before drawing conclusions. See [Design Review §6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defect 14).
>
> **Resolved (2026-09-18):** The benchmark now reports min/median/p95 over repeated runs against `std::malloc` and `std::pmr::monotonic_buffer_resource` baselines. The caveat itself still holds: this is the friendliest possible access pattern for a LIFO free list.


---

### Exercise 6: GotoBLAS Micro-Panel Packing ($B$ Matrix $N_r = 8$)
* **File to study**: `include/kinesis/compute/gemm_pack.hpp` and `src/compute/gemm.cpp`
* **The Code to Type**:
  ```cpp
  template <typename T, size_t Nr = 8>
  void pack_B_panel(const TensorView<T>& B, int64_t k_start, int64_t k_end, 
                    int64_t n_start, int64_t n_end, T* packed_b) {
      size_t idx = 0;
      for (int64_t n = n_start; n < n_end; n += Nr) {
          for (int64_t k = k_start; k < k_end; ++k) {
              for (size_t nr = 0; nr < Nr; ++nr) {
                  packed_b[idx++] = (n + nr < n_end) ? B(k, n + nr) : T(0);
              }
          }
      }
  }
  ```
* **What to practice**:
  1. Draw a $4 \times 8$ sub-matrix on paper in row-major order.
  2. Trace how `pack_B_panel` re-orders elements so that an entire column of 8 floats is stored consecutively in memory.
  3. Explain why this layout allows SIMD AVX2 `_mm256_loadu_ps` to stream 8 floats in a single clock cycle without non-unit striding.

> **Review note (2026-09-17):** The real `pack_B_panel` differs from the snippet: it is float-only, takes `Nr` as a runtime parameter, and uses `std::min` plus a zero-padding loop for the tail. `pack_A_panel` exists too but is never called, only B is packed. Two known defects live in this path: the packed kernel assumes unit column stride (so transposed inputs, including `gramian`, give wrong results), and the SIMD tail reads/writes past the end of the output row. See [Design Review §6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defects 1, 2).
>
> **Partly resolved (2026-09-18):** Both defects in this path are fixed: strided and transposed operands go to the stride-aware kernel, and partial micro-panels no longer read or write past the row. `pack_A_panel` is still never called. See [Design Review §6.0](DESIGN_REVIEW_2026-09-17.md#s6-0).


---

### Exercise 7: The Pragmatism Principle on Physical Silicon
* **The Concept**: Why an $O(M \cdot K + K \cdot N)$ preparation step beats naive $O(1)$ indexing.
* **The Mathematical Proof**:
  * For matrix dimension $N = 1024$:
    * Naive $O(1)$ inner loop suffers $1,024$ strided memory hops per row, causing continuous TLB misses ($>100\text{ cycles}$ per miss).
    * GotoBLAS packing performs $2 \times 1024^2 \approx 2 \times 10^6$ memory writes once.
    * The compute loop executes $2 \times 1024^3 \approx 2.14 \times 10^9$ FLOPs.
    * Compute work is $> 1,000\times$ larger than packing work! Packing cost is amortized to $< 2\%$, while compute throughput increases by **$300\%\text{ to }500\%$**.
* **Rule**: Slower than $O(1)$ is welcomed whenever the amortized physical hardware benefit is mathematically demonstrated.

---

### Exercise 8: Hardware Cache Emulation with `MemorySandbox`
* **File to study**: `include/kinesis/core/memory_sandbox.hpp`
* **What to practice**:
  1. Instantiate `MemorySandbox sandbox`.
  2. Feed 512 sequential addresses: `sandbox.access_read(0x40000 + i * 4, 4)`. Observe `sandbox.l1_hit_rate()` reach $>93\%$.
  3. Feed 128 strided addresses: `sandbox.access_read(0x80000 + i * 256, 4)`. Observe cache miss spikes.
---

### Exercise 9: Hardware Topology Detection & Silicon Safety Policies
* **File to study**: `include/kinesis/core/hardware_info.hpp`
* **The Code to Type**:
  ```cpp
  auto cpu_feats = HardwareInfo::query_cpu_features();
  auto gpu = HardwareInfo::detect_gpu();
  HardwareSafetyPolicy safety{};
  
  std::cout << "Detected CPU: " << HardwareInfo::cpu_architecture_name() << "\n";
  std::cout << "L1 Data Cache: " << (cpu_feats.l1_data_cache_size / 1024) << " KB\n";
  std::cout << "Transient Slew-Rate Warmup: " << (safety.enable_staged_warmup ? "ACTIVE" : "OFF") << "\n";
  ```
* **What to practice**:
  1. Explain why staged micro-stepping warmups protect motherboard VRMs against transient voltage drops ($V_{droop} = L \frac{di}{dt}$) during heavy AVX-512 / Tensor Core matrix operations.
  2. Contrast how Kinesis configures itself differently on a desktop (dynamic page allocator) vs. an ESP32 microcontroller (`KINESIS_EMBEDDED` static SRAM pool).

> **Review note (2026-09-17):** None of this is wired up yet: `HardwareSafetyPolicy` is never read by any code (though the report prints the protections as ACTIVE), `detect_gpu()` returns hard-coded values, cache sizes are constants, and `KINESIS_EMBEDDED` only selects an enum. On the physics: the CPU already stages its own wide-SIMD voltage and frequency transitions (measured at ≈ 8-20 µs at a quarter dispatch rate, with ≈ 680 µs before relaxing), so treat "staged warmup" as a latency technique rather than hardware protection. See [Design Review T11](DESIGN_REVIEW_2026-09-17.md#t11), [T17](DESIGN_REVIEW_2026-09-17.md#t17) and [§6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defects 6, 10).
>
> **Resolved (2026-09-18):** Partly fixed. The report no longer prints the protections as ACTIVE, it prints `[NOT IMPLEMENTED]`, and `HardwareSafetyPolicy` carries a warning that constructing one enforces nothing. `detect_gpu()` and the cache sizes are fixed too. `KINESIS_EMBEDDED` still only selects an enum.


---

## 2. Silicon & Hardware Experiments

Run these empirical experiments by modifying `benchmarks/bench_latency.cpp`:

### Experiment 1: The Cost of Cache-Line Splits
* **Test**: Allocate an array of floats at address `0x...00` (64-byte aligned) vs `0x...04` (misaligned by 4 bytes).
* **Workload**: Stream 10,000,000 SIMD reads across both.
* **Observe**: Measure the clock cycle penalty when vector registers cross two physical 64-byte cache lines.

### Experiment 2: Cache-Tile Scaling ($N \times 16\text{ KB}$)
* **Test**: In `gemm.cpp`, change tile sizes:
  * Small: $B = 32$ (Working set $\approx 12\text{ KB}$)
  * Sweet spot: $B = 64$ (Working set $\approx 48\text{ KB}$)
  * Thrashing: $B = 256$ (Working set $\approx 768\text{ KB}$ - spills into L2/L3)
* **Graph**: Plot GFLOPS vs. Tile Size to find your CPU's exact optimal inflection point.

### Experiment 3: 32-Way GPU Bank Conflict vs. XOR Swizzling
* **Test**: In `MemorySandbox`, simulate a 32-thread warp accessing shared memory with stride 128 bytes:
  $$\text{Bank} = \left(\frac{t \times 128}{4}\right) \pmod{32} = (32 \times t) \pmod{32} = 0$$
* **Observe**: All 32 threads collide on Bank 0, causing a catastrophic **$32\times$ serialization penalty**.
* **Fix (corrected 2026-09-17)**: swizzle the *column within each row* rather than the bank index, which is the form CUDA shared-memory tiles use:
  $$\text{col}' = \text{col} \oplus (t \bmod 32), \qquad \text{Bank} = (t \cdot s + \text{col}') \bmod 32$$
  At $s = 32$ words this collapses to $\text{Bank} = \text{col} \oplus t$, a bijection over the 32 lanes, so the penalty really does drop to $1.0\times$. Note what the sweep in `bench_latency` prints: the same swizzle *introduces* a conflict at $s = 1$ or $s = 33$, which were already conflict-free. A swizzle is chosen for one layout, not applied blindly.
* **Cheaper fix**: make the element stride odd. Padding a 32-word row to 33 words gives $\gcd(33, 32) = 1$ with no address arithmetic at all.

> **Review note (2026-09-17, resolved):** ❌ The formula above was wrong as first written; it and the visualizer have both since been corrected, and what follows is the original diagnosis. Work this one out on paper before trusting it: with stride 128 B the first term is 0 for every thread, so the bank reduces to ⌊t/4⌋ mod 32, eight distinct banks with four threads each, i.e. a **4-way** conflict, not 1.0×. The browser visualizer showed 1.0× only because its code ignored the stride entirely; `simulateWarp` now models the column swizzle and reproduces gcd(s, 32) exactly. The correct closed form for a linear stride of *s* elements is worst-conflict = gcd(s, 32), so the real fix is an **odd element stride** (or padding a row by one element): stride 33 words is conflict-free while stride 32 words collides 32 ways. See [Design Review §2.1](DESIGN_REVIEW_2026-09-17.md#s2-1) (L5) and [§2.3](DESIGN_REVIEW_2026-09-17.md#s2-3).


### Experiment 4: Exploring the Memory Sandbox Web Visualizer
* Open `docs/memory_sandbox_visualizer.html` in your browser.
* Test the interactive address prober with unaligned addresses like `0x0004003E` to visually see cache-line splits trigger.
* Slide the Warp Stride slider and toggle XOR swizzling in real-time to watch bank conflicts resolve.

---

## 3. Mathematical Foundations: Advanced Concepts Ahead

### A. Forward-Mode Autodiff: Dual Numbers ($x + \epsilon \dot{x}$)
* **The Rule**: $\epsilon \ne 0$, but $\epsilon^2 = 0$.
* **Taylor Expansion**:
  $$f(x + \epsilon) = f(x) + f'(x)\epsilon + \frac{f''(x)}{2!}\epsilon^2 + \dots = f(x) + f'(x)\epsilon$$
* **Multiplication**:
  $$(u + \dot{u}\epsilon)(v + \dot{v}\epsilon) = uv + (u\dot{v} + \dot{u}v)\epsilon + \dot{u}\dot{v}\epsilon^2 = uv + (u\dot{v} + \dot{u}v)\epsilon$$
  *(Notice the product rule naturally emerges without any graph tape!)*

### B. Spatial Dynamics: Featherstone + Dual Numbers
* In Phase 4, we will combine **Dual Numbers** with **Featherstone’s Articulated Body Algorithm (ABA)**:
  * Rigid body transformations on $SE(3)$ represented as 6D spatial twists:
    $$\mathbf{v} = \begin{bmatrix} \omega \\ v \end{bmatrix} \in \mathbb{R}^6$$
  * By evaluating Featherstone's recursive equations using `Dual<float>`, the forward dynamics and exact state Jacobians ($\frac{\partial \ddot{q}}{\partial q}$) are computed simultaneously in **linear $O(N)$ time**.

> **Review note (2026-09-17):** The technique is sound and worth learning, but the complexity claim is not: ∂q̈/∂q has n² entries, so it cannot be produced in O(N), and forward mode with N tangents multiplies every operation's cost by N. It is also established practice rather than new, Drake runs dual numbers (`AutoDiffXd`) through its multibody dynamics, and Pinocchio provides *analytical* derivatives of RNEA/ABA that outperform automatic differentiation (Carpentier & Mansard, RSS 2018: about 40× faster than finite differences). See [Design Review T7](DESIGN_REVIEW_2026-09-17.md#t7) and [§2.5](DESIGN_REVIEW_2026-09-17.md#s2-5).

---

> **Design review (2026-09-17):** Where these exercises drift from the current code, or rest on benchmark numbers that do not hold up, the notes above say so. Full details: [DESIGN_REVIEW_2026-09-17.md](DESIGN_REVIEW_2026-09-17.md).
