# Kinesis Architecture Specification

This document provides a technical explanation of the architectural choices, memory layouts, and mathematical foundations implemented in **Kinesis**.

---

## 1. Zero-Allocation Memory Engine

In high-frequency quantitative finance and hard real-time robotics (e.g., 500-1000 Hz ROS 2 control loops), standard heap allocation (`malloc`/`free` or C++ `new`/`delete`) introduces two critical points of failure:
1. **Unbounded Tail Latency**: Thread lock contention and OS memory compaction can cause latency spikes into tens of microseconds.
2. **Heap Fragmentation**: Repeated variable-sized tensor allocations degrade cache locality over time.

### The Arena Allocator Design (`ArenaAllocator`)
Kinesis uses a **64-byte cache-aligned monotonic bump allocator**:

```
Memory Arena Layout:
[Base Pointer] ──> | Aligned Block 1 | Aligned Block 2 | ... | Bump Pointer ──> Free Space | [Capacity]
```

* **Alignment Formula**:
  $$\text{addr}_{\text{aligned}} = (\text{addr} + \text{alignment} - 1) \ \& \ \sim(\text{alignment} - 1)$$
  Every allocation is aligned to `kDefaultAlignment = 64` bytes, ensuring compatibility with 512-bit vector registers (AVX-512) and preventing cross-cache-line performance penalties.
* **$O(1)$ Allocation**: Allocation advances `current_offset_` by the requested size plus alignment padding.
* **$O(1)$ Reset**: At the end of an execution step (e.g., each camera frame or market tick), `reset()` sets `current_offset_ = 0` without issuing a single system call or freeing physical pages.
* **RAII Scoped Markers (`ArenaAllocator::ScopedMarker`)**:
  Temporary intermediate activation tensors in a sub-graph automatically rewind memory upon exiting C++ block scope:
  ```cpp
  {
      auto scope = arena.scope();
      Tensor<float> temp_activation(Shape{64, 128}, arena);
      // Memory rewinds to 'scope' entry point upon leaving block
  }
  ```


> **Review note (2026-09-17):** The arena primitive is sound (it matches `std::pmr::monotonic_buffer_resource` and TensorFlow Lite Micro's arena), but the zero-allocation guarantee is not enforced end-to-end: `gemm()` heap-allocates a packing buffer per call, the default `Tensor` allocator issues one `VirtualAlloc`/`mmap` per tensor, a full arena throws mid-tick, and pages are not prefaulted (first-touch page faults land in the hot loop). Rewinding a scope also silently invalidates any `TensorView` into that memory, and the error-handling policy mixes exceptions with asserts that vanish in release builds, which suits neither the HFT nor the embedded target. See [T1](DESIGN_REVIEW_2026-09-17.md#t1), [T2](DESIGN_REVIEW_2026-09-17.md#t2), [T14](DESIGN_REVIEW_2026-09-17.md#t14) and [§2.2](DESIGN_REVIEW_2026-09-17.md#s2-2).
>
> **Partly resolved (2026-09-18):** `gemm()` no longer allocates (its packing buffer is thread-local), the default `Tensor` allocator is an aligned heap allocator rather than one `VirtualAlloc`/`mmap` per tensor, and the shape, view, arena and slab checks now survive release builds. Still open: a full arena throws mid-tick, pages are not prefaulted, the rewritten benchmark measures first-touch faults at **27.84 ns/alloc against 6.34 ns** steady state, and rewinding a scope still invalidates views silently. See [§6.0](DESIGN_REVIEW_2026-09-17.md#s6-0).

---

## 2. Zero-Copy TensorView & Stride Mathematics

Kinesis completely decouples **physical memory storage** from the **logical tensor shape**.

### Generalized Stride Multi-Index Formula
For an $N$-dimensional tensor with coordinate $(i_0, i_1, \dots, i_{D-1})$, the 1D physical buffer offset is computed as:

$$\text{Index}(i_0, \dots, i_{D-1}) = \text{offset} + \sum_{k=0}^{D-1} i_k \cdot \text{strides}_k$$

### Stride Computation
For canonical C-contiguous (row-major) order:
$$\text{stride}_{D-1} = 1$$
$$\text{stride}_k = \text{stride}_{k+1} \times \text{shape}_{k+1} \quad \text{for } k = D-2, \dots, 0$$

### Zero-Copy Operations
* **Slicing**: Slicing dimension $d$ on range $[a, b)$:
  $$\text{offset}' = \text{offset} + a \cdot \text{stride}_d, \quad \text{shape}'_d = b - a$$
  *Memory cost*: 0 bytes copied.
* **Transposition**: Swapping dimensions $j$ and $k$ simply swaps $\text{shape}_j \leftrightarrow \text{shape}_k$ and $\text{strides}_j \leftrightarrow \text{strides}_k$.
  *Memory cost*: 0 bytes copied.
* **Broadcasting**: Expanding a dimension of size $1$ to size $M$ sets its stride to $0$:
  $$\text{stride}'_d = 0$$
  Advancing index $i_d$ yields $i_d \cdot 0 = 0$, mapping all slice elements to the same physical address without memory duplication.


> **Review note (2026-09-17):** Strided zero-copy views are the standard design (NumPy/PyTorch, `std::mdspan` with `layout_stride`). Two gaps: kernels do not handle non-unit strides consistently, `gemm_packed` silently mis-computes transposed inputs, which breaks `gramian` for matrices with ≥ 128 columns, and a `TensorView<const T>` cannot be passed to any kernel, which blocks read-only or memory-mapped weights. See [T3](DESIGN_REVIEW_2026-09-17.md#t3), [T4](DESIGN_REVIEW_2026-09-17.md#t4) and [§6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defects 1 and 9).
>
> **Resolved (2026-09-18):** Both gaps are closed. `gemm_packed` routes transposed and strided operands to the stride-aware kernel, so `gramian` is correct for any width, and `TensorView<T>` converts to `TensorView<const T>`, so read-only weights reach the kernels. See [§6.0](DESIGN_REVIEW_2026-09-17.md#s6-0) (defects 1 and 9).

---

## 3. Cache-Blocked GEMM (Matrix Multiplication)

Naive triple-loop matrix multiplication exhibits $O(N^3)$ memory traffic because inner-loop iterations across column $j$ in matrix $B$ repeatedly evict L1 and L2 cache lines.

### Tiling & Loop Order Optimization
Kinesis implements **Cache-Blocked GEMM**:
1. **L1/L2 Cache Tiling**: Partitions matrices into blocks ($M_c = 64, K_c = 64, N_c = 64$) that fit within the L1/L2 data cache.
2. **$i, k, j$ Traversal**: The innermost loop runs across column index $j$. In row-major layout, both $B(k, j)$ and $C(i, j)$ are contiguous in memory.
3. **SIMD Vectorization**: The inner $j$ loop streams contiguous elements into 256-bit vector registers using AVX2 Fused Multiply-Add (FMA) instructions:
   $$c_{\text{vec}} = \text{fma}(a_{\text{scalar}}, b_{\text{vec}}, c_{\text{vec}})$$


> **Review note (2026-09-17):** Actual implementation: `gemm_blocked` (64³ tiles, i-k-j order, AVX2 FMA) and, when M and N ≥ 128, `gemm_packed`, which packs only B and accumulates one row of C at a time (effectively a 1×8 micro-kernel). BLIS's Haswell configuration packs both A and B and uses a 6×16 single-precision kernel with KC = 256, MC = 168, NC = 4080. The review also documents two correctness defects (strided inputs; SIMD tails that read/write past the row) and a heap allocation on every `gemm()` call. See [§2.2](DESIGN_REVIEW_2026-09-17.md#s2-2) and [§6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defects 1, 2, 7).
>
> **Resolved (2026-09-18):** Both defects are fixed. `gemm_packed` now returns to the stride-aware kernel unless A and C have unit column stride, and partial micro-panels use a scalar loop bounded by the real column count. A page-boundary reproduction of the old tail segfaults; the new one completes. The packing buffer is thread-local, so the hot path allocates nothing. Packing A and register tiling the micro-kernel remain undone, those belong with [T3](DESIGN_REVIEW_2026-09-17.md#t3).

---

## 4. Integration Blueprint for Target Domains

### Robotics & ROS 2
* Wraps incoming image pointers from `rclcpp::LoanedMessage` directly into `TensorView<uint8_t>` without memory copies.
* Enables real-time feature extraction and neural inference inside pre-allocated arenas.

### Quantitative Finance
* Zero heap allocations during market trading hours.
* Zero-copy subviews over market order books and trade tick ring buffers.

---

## 5. Hardware Memory Granularity & Vendor Discrepancies

Cache geometries and memory transaction widths diverge significantly across processor vendors:

| Architecture | Granularity / Line Size | Transaction Model |
| :--- | :--- | :--- |
| **Intel / AMD x86_64** | 64 bytes | 64-byte burst lines across DDR4/DDR5 |
| **Apple Silicon M-Series** | 128 bytes (Performance cores) | High-bandwidth 128-byte streaming cache lines |
| **NVIDIA GPUs (Ampere/Hopper)** | 32-byte sectors | 128-byte transaction broken into 4x 32-byte sub-sectors |
| **AMD GPUs (RDNA / CDNA)** | 32 / 64 bytes (Dual 32B lines) | Wavefront 32 (RDNA) vs Wavefront 64 (CDNA) |

* **Hardware-Adaptive Alignment**:
  While CPU memory lines are 64 bytes, NVIDIA GPUs process memory in **32-byte sectors** (32 threads per warp reading 4-byte values equals 128 bytes total).
  Kinesis aligns allocations to `64` bytes by default on CPU (and `128` bytes on GPU/Apple Silicon), which naturally satisfies multiples of 32 without cross-sector transaction penalties.
* **The GPU L3 / Infinity Cache Difference**:
  * NVIDIA relies on massive, high-bandwidth **L2 caches** (e.g. 50-96 MB on Ada Lovelace/Hopper) and omits L3.
  * AMD incorporates **Infinity Cache (MALL)**, an on-die L3 cache that reduces memory traffic to VRAM.
  * Kinesis benchmarks take vendor cache hierarchies into account when sizing working sets.

> **Review note (2026-09-17):** Sourcing check. ✅ GPU on-chip L1 per streaming multiprocessor and a larger L2 appear in NVIDIA's Ada (128 KB L1; 98,304 KB L2 on AD102, 16× GA102) and Ampere (192/128 KB L1; 40 MB L2 on A100) tuning guides and in FlashAttention §2.1; ✅ Apple's 128-byte cache line is confirmed (`hw.cachelinesize`). ❓ "32-byte sectors", "128-byte transactions", Wave32/Wave64, "CDNA" and "MALL" were not found in any cited source. NVIDIA's alignment guidance concerns *tensor dimensions* (multiples of 8/16/64 elements for Tensor Cores), not the byte alignment of allocations, and `kDefaultAlignment` is always 64 in code. The benchmarks do not account for vendor cache hierarchies. See [§2.3](DESIGN_REVIEW_2026-09-17.md#s2-3) and [§5](DESIGN_REVIEW_2026-09-17.md#s5).


---

## 6. Pluggable Allocator Architecture (Rust-Inspired)

Rather than hardcoding allocation behavior, Kinesis adopts an interchangeable, composable allocator model:

1. **`ArenaAllocator` (Monotonic Bump)**:
   - $O(1)$ allocation, $O(1)$ reset.
   - Ideal for transient activations within a single tick or camera frame.
2. **`BlockPool` / `SlabAllocator` (Malloc Grouping)**:
   - Pre-allocates segregated memory pools for fixed tensor sizes (e.g., 64B quaternions, 1KB state vectors).
   - $O(1)$ allocation and free via internal free-lists without dynamic heap fragmentation.
3. **`CudaStreamPool` (GPU Backend)**:
   - Wraps stream-ordered virtual memory pools (`cudaMemPool_t`) for zero-synchronization asynchronous GPU buffer reuse.


> **Review note (2026-09-17):** The `Allocator` interface duplicates the standard `std::pmr::memory_resource` model (`ArenaAllocator` ≈ `monotonic_buffer_resource`, `SlabAllocator` ≈ `unsynchronized_pool_resource`). `CudaStreamPool` names a real, current CUDA API, the Stream-Ordered Memory Allocator (`cudaMallocAsync`, `cudaMemPoolCreate`), which PyTorch exposes as `backend:cudaMallocAsync`, but it is not implemented, and the host-only `Allocator` interface has no notion of devices or streams. See [T13](DESIGN_REVIEW_2026-09-17.md#t13) and [§2.3](DESIGN_REVIEW_2026-09-17.md#s2-3).

---

## 7. Dynamic Cache-Tile Sizing ($N \times 16\text{ KB}$)

Instead of fixed $64 \times 64$ tiles, Kinesis supports parameterized dynamic tiling based on runtime CPU cache detection:

$$\text{Working Set} = 3 \cdot B^2 \times \text{sizeof}(T)$$

* For single-precision `float` (4 bytes), target working set is bounded by:
  $$3 \cdot B^2 \times 4 \le N \times 16\text{ KB}$$
  * $N = 1$ ($16\text{ KB}$ L1): $B \approx 36$
  * $N = 2$ ($32\text{ KB}$ L1): $B \approx 52$
  * $N = 3$ ($48\text{ KB}$ L1 - Modern Intel/AMD): $B \approx 64$
  * $N = 4$ ($64\text{ KB}$ L1): $B \approx 73$


> **Review note (2026-09-17):** The arithmetic is right, but (a) this is a loop-tiling bound, not the GotoBLAS/BLIS model, which sizes kc·nr to L1, mc·kc to L2 and kc·nc to L3 (Low et al., TOMS 2016); and (b) nothing is detected at runtime, `gemm_packed` hard-codes a 49,152-byte L1 and `HardwareInfo` returns constant cache sizes. See [§2.2](DESIGN_REVIEW_2026-09-17.md#s2-2) and [T11](DESIGN_REVIEW_2026-09-17.md#t11).

---

## 8. Robotics Dynamics Math: Classical vs. Spatial vs. Featherstone

When computing robotic kinematics and dynamics, standard Euclidean Jacobians exhibit fundamental limitations:

1. **Classical Euclidean Jacobian**:
   - Computes $\dot{x} = J(q)\dot{q}$ using Euler angles or translation vectors.
   - Suffers from representation singularities (gimbal lock) and non-invariant coordinate dependencies.
2. **Spatial Jacobian on Lie Group $SE(3)$**:
   - Represents rigid body motion as elements of the Lie group $SE(3)$ with Lie algebra $\mathfrak{se}(3)$ 6D twists:
     $$\mathbf{v} = \begin{bmatrix} \omega \\ v \end{bmatrix} \in \mathbb{R}^6$$
   - Coordinate-free, singularity-free, and geometrically rigorous.
3. **Featherstone’s Spatial Vector Algorithms (ABA / RNEA)**:
   - **Recursive Newton-Euler Algorithm (RNEA)** computes inverse dynamics in **$O(N)$ linear time**.
   - **Articulated Body Algorithm (ABA)** computes forward dynamics in **$O(N)$ linear time**, replacing $O(N^3)$ matrix inversions of the mass matrix $M(q)$.

> **Review note (2026-09-17):** Lie-group parameterizations remove *representation* singularities (gimbal lock), not *kinematic* singularities, so "singularity-free" needs that qualifier. For dynamics derivatives, state-of-the-art libraries such as Pinocchio provide analytical derivatives of RNEA/ABA that are faster than automatic differentiation (Carpentier & Mansard, RSS 2018). See [T7](DESIGN_REVIEW_2026-09-17.md#t7) and [§2.5](DESIGN_REVIEW_2026-09-17.md#s2-5).

---

## 9. Memory Re-Sorting, Packing, and Layout Transformations

Raw matrix multiplication and convolution speed is fundamentally bounded by memory layout. In large-scale ML, robotics vision, and high-frequency trading, **re-sorting and packing memory ahead of computation yields 2x to 10x performance gains**:

### A. The GotoBLAS / BLIS Matrix Packing Philosophy
* Rather than streaming directly across arbitrary matrix strides, large matrices are packed into continuous micro-panels ($\tilde{A}$ of size $M_c \times K_c$ and $\tilde{B}$ of size $K_c \times N_c$).
* **Why Packing Wins**:
  1. **TLB Thrashing Elimination**: Packing guarantees that active compute panels reside in contiguous physical memory pages, eliminating Translation Lookaside Buffer page faults.
  2. **100% Sequential Burst Reads**: Eliminates stride jumps across non-contiguous rows, allowing the hardware memory prefetcher to run at theoretical peak memory bandwidth.
  3. **Amortized Cost**: Packing takes $O(M \cdot K + K \cdot N)$ memory writes, which is dwarfed by the $O(M \cdot K \cdot N)$ compute FLOPs as matrix dimensions scale.

> **Review note (2026-09-17):** The work ratio is correct (≈ 1 : 1,024 at N = 1,024), but the "2x to 10x" gains were never measured, and Kinesis currently packs only B. Packing gives *virtual* contiguity, not "contiguous physical memory pages", and it reduces TLB misses rather than "page faults"; huge pages are the complementary fix. See [§2.1](DESIGN_REVIEW_2026-09-17.md#s2-1) (L11) and [§2.2](DESIGN_REVIEW_2026-09-17.md#s2-2).


### B. Offline Weight Pre-Packing (Inference Superpower)
* In robotics control loops (1 kHz) and quantitative market tick handlers, model weights $W$ do not change during deployment ($y = Wx + b$).
* **Kinesis Optimization**: Model weights are pre-sorted and pre-packed into optimal SIMD register micro-panels once at startup. In the hot loop, the kernel streams the input vector $x$ directly across pre-sorted, cache-pinned weights with **zero runtime packing overhead**.

> **Review note (2026-09-17):** Common state-of-the-art practice (XNNPACK weight pre-packing; Arm KleidiAI packing + micro-kernels), but **not implemented**, packing happens on every call. Also, for batch-1 inference `y = Wx` is a matrix-vector product, where GEMM-style panels add little; quantized or SIMD row kernels matter more. See [§2.2](DESIGN_REVIEW_2026-09-17.md#s2-2).


### C. Vision Layout: NHWC (Channels-Last) & `im2col`
* **NHWC vs. NCHW**: In NHWC, color channels ($R, G, B$) and feature maps are contiguous along the innermost dimension. A single SIMD vector instruction loads an entire pixel's channels simultaneously.
* **`im2col` Transform**: Re-sorts overlapping 2D/3D receptive fields into contiguous columns, allowing complex convolutions to be evaluated directly by our high-performance GEMM engine.

> **Review note (2026-09-17):** "Implicit GEMM" is a cuDNN (GPU) algorithm. On CPUs, the Indirect Convolution Algorithm (Dukhan, 2019) avoids im2col's data copy with an indirection buffer and runs up to 62% faster, note that it relies on indirection, which the affine-indexing rule in §11 forbids. See [§2.2](DESIGN_REVIEW_2026-09-17.md#s2-2) and [T9](DESIGN_REVIEW_2026-09-17.md#t9).


### D. GPU Shared Memory Swizzling & Bank Conflicts
* On GPUs, shared memory is divided into 32 banks (4 bytes wide each). Linear strided access causes multiple threads in a warp to hit the same bank, serializing memory accesses (up to 32x bandwidth loss).
* **XOR Swizzling**: Memory indices are re-sorted via bitwise XOR ($\text{Bank} = \text{Col} \oplus (\text{Row} \gg \text{shift})$), mathematically guaranteeing that all 32 threads land on independent memory banks without bank conflicts.

> **Review note (2026-09-17):** ❌ The guarantee does not hold. The cited NVIDIA post (Harris, 2013) ties bank geometry to compute capability (16 banks at 1.x, 32 at 2.0, configurable 4/8-byte banks at 3.x) and proposes no mitigation; "swizzling" in Fabian Giesen's post means permuting *texture* address bits. For a linear stride of *s* elements the worst conflict is exactly gcd(s, 32), so any odd element stride is already conflict-free, while the documented XOR formulas still leave 4-way conflicts. See [§2.1](DESIGN_REVIEW_2026-09-17.md#s2-1) (L5) and [§2.3](DESIGN_REVIEW_2026-09-17.md#s2-3).


### E. Tensor Core Register Fragments & `ldmatrix` In-Flight Re-Sorting
* NVIDIA Tensor Cores (Ampere, Hopper, Blackwell) do not execute operations on flat 1D row arrays. Instead, they require matrices to be partitioned into distributed matrix fragments across the 32 threads of a warp (e.g. `mma.sync.aligned.m16n8k16`).
* **In-Flight Re-Sorting**: Rather than manually shuffling registers in software, dedicated hardware instructions (`ldmatrix.x4`) re-sort and permute 2D shared memory blocks directly into Tensor Core register fragment distributions in a single clock cycle.

> **Review note (2026-09-17):** ❓ `ldmatrix.x4`, `mma.sync.aligned.m16n8k16` and "register fragment distributions" were not found in NVIDIA's Ada/Ampere tuning guides or any other cited source. State-of-the-art GPU matrix multiplication relies on cuBLAS/cuBLASLt with Tensor-Core-friendly dimensions (NVIDIA Matrix Multiplication guide). See [§2.3](DESIGN_REVIEW_2026-09-17.md#s2-3).


### F. Cache-Oblivious Re-Sorting: Morton Z-Order Curves
* In addition to block-tiling, 2D/3D tensor memory can be indexed via **Morton (Z-order) space-filling curves** by interleaving bit representations of row and column coordinates:
  $$\text{Morton}(r, c) = \text{interleave\_bits}(r, c)$$
* **Locality Guarantee**: Morton ordering guarantees spatial locality recursively at every level of the memory hierarchy (L1, L2, L3, and DRAM) without requiring hardcoded tile dimensions.


> **Review note (2026-09-17):** ❌ As a performance strategy for dense linear algebra: Yotov et al. (SPAA 2007) found that "even highly optimized cache-oblivious programs perform significantly worse than corresponding cache-conscious programs" with register-tuned micro-kernels. Morton order remains useful for spatial indexing. See [§2.2](DESIGN_REVIEW_2026-09-17.md#s2-2).

---

## 10. GPU Binary Analysis, Local Memory Spills (`STL`), & Constant Caches

Hardware profiling via NVIDIA binary disassemblers (`nvdisasm` on driver shader caches `$HOME/.nv/GLCache/`) reveals critical silicon-level failure modes that inform Kinesis's low-level layout:

### A. The Local Memory Spill Trap (`STL`)
* In GPU shaders and CUDA kernels, declaring unbounded stack arrays or passing complex structs by value exhausts the physical register file.
* **The `STL` Penalty**: When registers spill, the compiler generates `STL` (Store to Local Memory) instructions. Despite the name, "Local Memory" is physically backed by off-chip DRAM / L1 cache, with latencies exceeding 200-800 cycles.
* **Kinesis Defense**:
  1. Tensor metadata is strictly bounded (the `Shape` and `Strides` stack descriptors are capped at 6 dimensions, fitting in registers without spilling).
  2. Discrete boolean masks and states are explicitly bit-packed (e.g. compressing `int[220]` arrays into `uint[7]` bitmasks), preventing compiler-induced `STL` emissions.


> **Review note (2026-09-17):** The source (Danil's article) is about OpenGL/Vulkan shader compilation on NVIDIA drivers; it does not say that "Local Memory" is backed by off-chip memory with 200-800-cycle latency. On CPUs, tensor metadata is not kept "in registers": a 104-byte `TensorView` passed by value goes by pointer (Windows x64) or in memory (SysV). See [§2.3](DESIGN_REVIEW_2026-09-17.md#s2-3) and [T5](DESIGN_REVIEW_2026-09-17.md#t5).

### B. The 1-2 KB Constant Memory (`CONST`) Hardware Knee
* Modern NVIDIA GPUs feature dedicated, low-latency constant memory caches.
* **The Cache Cliff**: While total constant memory can be 64 KB, the ultra-fast L1 constant cache on many consumer and embedded architectures is only **1 KB to 2 KB** (~256 unique 32-bit floats).
* When neural network weights or constant pools exceed 2 KB (e.g. 7 KB of weight matrices), constant cache misses cause execution throughput to collapse by **15x to 30x** (dropping frame rates from 60 FPS to 2 FPS).
* **Quantization & Deduplication**: Kinesis inference kernels enforce weight quantization (FP8/INT8) and scalar clustering to ensure hot neural weights fit strictly within the sub-2 KB constant cache boundary.


> **Review note (2026-09-17):** The 1-2 KB "knee" was the article author's observation on his own low-end GPU for specific shaders, not a general NVIDIA property; in compute kernels, weights live in device buffers rather than shader constant pools. Quantization (FP8/INT8/INT4) is a core state-of-the-art optimization for throughput and memory (FP8 Formats, Micikevicius et al.; Hugging Face quantization overview), but Kinesis has no quantized kernels yet. See [§2.3](DESIGN_REVIEW_2026-09-17.md#s2-3).

---

## 11. GPU Cache Architecture Rules & Memory Coherence

Silicon latency benchmarks and architectural tuning guides (Grassi, Chips & Cheese, NVIDIA Architecture Guides) establish six immutable laws of GPU memory coherence implemented in Kinesis:

1. **The $16.7\times$ Latency Waterfall**:
   * Cache hits take $\approx 30\text{ cycles}$; VRAM DRAM fetches take $\approx 500\text{ cycles}$. A 1% drop in cache hit rate adds millions of stalled cycles.
   * *Kinesis Rule*: Slices in hot compute loops must be strictly contiguous (stride 1) or packed into contiguous arena memory prior to execution.
2. **Channel Packing (1 Vector Load Beats 4 Scalar Loads)**:
   * Loading 4 separate scalar buffers issues 4 independent bus transactions with 4x arbitration overhead. Packing into vectors (`RGBA` / `f32x4`) issues a single 16B/32B transaction.
   * *Kinesis Rule*: Channels-Last (NHWC) and SIMD vector alignment across all vision and feature tensors.
3. **Elimination of Dependent Memory Reads**:
   * Dependent reads ($A[\text{index}[i]]$) serialize memory access ($500\text{ cycles} + 500\text{ cycles} = 1000\text{ cycles}$).
   * *Kinesis Rule*: All tensor index calculations must be pure affine functions ($\text{offset} + i \cdot \text{stride}$), enabling parallel address generation.
4. **Hardware Latency Hiding via Occupancy (No Software Prefetching)**:
   * Manual prefetch instructions waste registers and stall instruction issue slots. Latency hiding is achieved via warp occupancy.
   * *Kinesis Rule*: Maximize active warps and unroll loop bodies to allow the hardware warp scheduler to hide memory latency.
5. **Scale-Space Locality (Mipmaps & Feature Pyramids)**:
   * Sample strides explode when receptive fields expand across high-resolution spatial maps.
   * *Kinesis Rule*: Computer vision modules implement multi-scale feature pyramids to ensure memory access remains localized within L1/L2 cache boundaries.
6. **Hit-Rate Verification (Target: L1 $>80\%$, L2 $>70\%$)**:
   * *Kinesis Rule*: Algorithmic efficiency is measured via simulated hardware hit rates in the `MemorySandbox` rather than noisy wall-clock timers.


> **Review note (2026-09-17):** These rules come from a GPU **shader/texture** article (Grassi, 2025): its ≈ 30-cycle L1 hit vs ≈ 500-cycle VRAM fetch are GPU figures (a CPU L1 hit is ≈ 4 cycles, L2 ≈ 12). Rule 3 (affine indexing only) conflicts with paged KV caches, embeddings, MoE routing, gather/scatter and indirect convolution; rule 6's hit-rate targets are unsourced, and the sandbox cannot observe real kernels. See [T9](DESIGN_REVIEW_2026-09-17.md#t9), [T12](DESIGN_REVIEW_2026-09-17.md#t12), [§2.2](DESIGN_REVIEW_2026-09-17.md#s2-2) and [§2.3](DESIGN_REVIEW_2026-09-17.md#s2-3).

---

## 12. The Pragmatism Principle: When $O(N)$ Beats Dogmatic $O(1)$ on Silicon

In computer systems engineering, a dogmatic pursuit of theoretical $O(1)$ asymptotic complexity can become a local maximum trap. An $O(1)$ operation that causes cache-line pollution, memory fragmentation, or security vulnerabilities is far worse on physical hardware than an intentional $O(N)$ preparation step that yields orders-of-magnitude faster computation:

### A. The Three Silicon Inversions
1. **GotoBLAS Micro-Panel Packing ($O(M \cdot K + K \cdot N)$ vs. Naive $O(1)$)**:
   * Naive GEMM uses $O(1)$ direct index computation ($A_{i, k} \cdot B_{k, j}$) without pre-processing, but suffers severe non-unit strides and Translation Lookaside Buffer (TLB) thrashing.
   * GotoBLAS packing requires an explicit $O(M \cdot K + K \cdot N)$ memory copy into contiguous panels.
   * **The Silicon Reality**: Compute work scales as $2 M K N$ FLOPs. For $M = K = N = 1024$, the ratio of compute to memory writes is $> 1,000 : 1$. The packing overhead is $< 2\%$ of total runtime, but delivers a **$3\times$ to $5\times$ net compute acceleration** because contiguous memory eliminates TLB misses and maximizes SIMD burst throughput.
2. **Deterministic Slab Allocation & Free-List Validation ($O(N)$ Bulk Reset vs. Unchecked $O(1)$ Pointer Bumps)**:
   * Raw pointer bumping without boundary checks is $O(1)$, but in 24/7 high-frequency quantitative execution and safety-critical robotics, unchecked bumping risks silent memory corruption.
   * Our segregated `SlabAllocator` maintains intrusive LIFO free-lists with explicit page commitment. Resetting a slab traverses block chains in $O(N)$ relative to block count, but completely eliminates dynamic memory fragmentation, providing deterministic $0.64\text{ ns}$ allocation cycles.
3. **Lie Manifold Spatial Dynamics ($O(N)$ Featherstone ABA vs. $O(1)$ Hardcoded Analytic Kinematics)**:
   * Fixed analytic kinematics for a specific 6-DOF robotic arm can be hardcoded into $O(1)$ trigonometric polynomial expressions.
   * However, Featherstone's $O(N)$ Articulated Body Algorithm scales universally across arbitrary branched, floating-base kinematic trees (quadrupeds, humanoids) without coordinate singularities or gimbal lock.
   * Evaluated with Dual Numbers ($x + \epsilon \dot{x}$), Featherstone ABA computes forward accelerations and exact state Jacobians simultaneously in a single $O(N)$ linear pass.

### B. The Kinesis Pragmatism Rule
$$\text{Adopt } O(N) \iff \frac{\text{Amortized Hardware Speedup}}{\text{Preparation Overhead}} \gg 1 \quad \lor \quad \text{Deterministic Safety Guarantee}$$


> **Review note (2026-09-17):** The principle is reasonable, but its evidence standard should be measurement rather than "mathematically evident" arguments: example 1's speedups were never measured; example 2's "free-list validation" does not exist (slab checks are `assert`-only); and example 3's "exact state Jacobians simultaneously in a single O(N) linear pass" is incorrect (a dense n×n Jacobian cannot be written in O(n)) and not novel (Drake `AutoDiffXd`; Pinocchio's analytical derivatives). See [T7](DESIGN_REVIEW_2026-09-17.md#t7), [T12](DESIGN_REVIEW_2026-09-17.md#t12) and [§2.5](DESIGN_REVIEW_2026-09-17.md#s2-5).

---

## 13. Hardware Emulation & Visualization Interface Architecture

Physical CPU and GPU execution can mask memory layout bugs due to aggressive out-of-order execution buffers and hardware stream prefetchers. Kinesis incorporates a dedicated software hardware simulator and visual telemetry interface:

### A. The `MemorySandbox` Core Engine (`include/kinesis/core/memory_sandbox.hpp`)
* **L1 Data Cache Model**:
  * 32 KB capacity, 64-byte line size, 8-way set-associative LRU eviction policy (64 sets total).
  * Tracks hits, misses, cold line fills, and simulated cycle penalties ($30\text{ cycles}$ hit vs $500\text{ cycles}$ cold VRAM DRAM miss).
* **Cache-Line Split Detector**:
  * Evaluates whether a multi-byte memory access (e.g. 4-byte `float` or 32-byte AVX vector) crosses a 64-byte physical line boundary ($\lfloor \frac{\text{addr}}{64} \rfloor \neq \lfloor \frac{\text{addr} + \text{size} - 1}{64} \rfloor$).
* **GPU 32-Bank Warp Conflict Detector**:
  * Emulates a 32-thread SIMT warp addressing 32 independent 4-byte shared memory banks.
  * Generates a conflict histogram and computes the **Warp Serialization Multiplier** ($1.0\times$ to $32.0\times$), warning when access patterns collapse into serialized clock cycles.

### B. Visual Telemetry Interfaces
1. **Terminal Console Dashboard**:
   * Integrated directly into CLI test and benchmark binaries via `sandbox.print_dashboard()` and `MemorySandbox::print_bank_histogram()`.
   * Renders visual Unicode/ASCII gauge bars and thread distribution histograms during CI/CD runs.
2. **Interactive Dark-Mode Web Visualizer (`docs/memory_sandbox_visualizer.html`)**:
   * A zero-dependency HTML5/JS dashboard featuring:
     * **Workload Presets**: Instant simulation of GotoBLAS panels, strided thrashing, coalesced warps, 32-way conflicts, and XOR swizzling.
     * **Live Address Prober**: Interactive hex/decimal address entry with physical tag decoding, set index lookup, and cache-line split detection.
     * **Interactive 64-Set Cache Matrix**: Visual inspection of all 64 L1 cache sets and their 8 ways.
     * **Real-time Warp Stride Slider & XOR Swizzle Toggle**: Interactive slider demonstrating how non-linear XOR indexing ($\text{Bank} = \text{Col} \oplus (\text{Row} \gg 1)$) eliminates warp serialization in real-time.


> **Review note (2026-09-17):** Useful as a teaching tool, not as a verification oracle: addresses are hand-fed (not traced from real kernels), the CPU L1 model is annotated with GPU latencies, there is no TLB, prefetcher or cycle accounting (despite §13.A), the bank model uses CUDA compute-capability-2.0 geometry, and the visualizer's XOR swizzle discards the stride. On x86 the expensive split hazard is the split *lock* (thousands of cycles), not ordinary line-split loads. See [T12](DESIGN_REVIEW_2026-09-17.md#t12), [§2.1](DESIGN_REVIEW_2026-09-17.md#s2-1) (L5) and [§6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defect 15).
>
> **Resolved (2026-09-18):** Partly fixed. A split access now looks up every cache line it touches instead of only the first, and 4 KB page splits are counted separately from line splits. The deeper limitation stands: hand-fed addresses and an assumed geometry cannot validate real cache behaviour, which needs hardware counters or a trace simulator.

---

## 14. Hardware Topology Detection, Silicon Protection, and Embedded Profiles

### A. The Hardware & Topology Detection Subsystem (`hardware_info.hpp`)
To ensure Kinesis maximizes silicon potential without relying on hardcoded platform assumptions, the engine queries runtime hardware topology directly:
1. **CPU Architecture & Cache Geometry**:
   * **x86_64**: Queries processor extended feature flags via `cpuid` (`EAX=1`, `EAX=7`) to detect AVX2, AVX-512F, FMA, and AMX. Cache line granularity is verified at 64 bytes.
   * **ARM64**: Identifies ARMv8-A/ARMv9-A NEON vector engines. Sets cache lines to 64 bytes (Raspberry Pi 4/5) or 128 bytes (Apple Silicon).
   * **Embedded / Microcontrollers**: Detects Xtensa (ESP32 LX6/LX7) and RISC-V targets.
2. **GPU Vendor & Acceleration Discovery**:
   * Enumerates device capabilities across NVIDIA (CUDA/Tensor Cores), AMD (CDNA/RDNA), Intel (Arc/Xe), Apple Silicon (Metal/ANE), and Broadcom VideoCore (Raspberry Pi Vulkan Compute).
   * Automatically configures FP16 half-precision and Tensor Core warp shuffle pathways based on hardware vendor signatures.


> **Review note (2026-09-17):** ❌ Not implemented as described: `detect_gpu()` returns hard-coded results (an 8 GB NVIDIA GPU with Tensor Cores on any non-Apple x86-64 host; VideoCore VII on any Linux aarch64 host), the AVX/AVX2/AVX-512 flags skip Intel's required OSXSAVE/XGETBV checks, cache sizes are constants, and nothing consumes the results. State-of-the-art runtime dispatch: ggml CPU backend variants scored at runtime, OpenBLAS `DYNAMIC_ARCH`, Google Highway. See [T11](DESIGN_REVIEW_2026-09-17.md#t11) and [§6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defects 6, 10).
>
> **Resolved (2026-09-18):** Fixed. `detect_gpu()` reports `GpuVendor::Unknown` with `detected == false` rather than inventing a device; feature detection performs the max-leaf, OSXSAVE and XCR0 checks; and cache geometry is read from CPUID leaf 4 / 0x8000001D. On the development machine that corrected L1D/L2/L3 from 32 KB / 512 KB / 16 MB to the true **48 KB / 1280 KB / 24 MB**, every compiled-in constant had been wrong. Detection results still feed nothing ([T11](DESIGN_REVIEW_2026-09-17.md#t11) is a design decision, not a defect).

### B. Silicon Protection Engineering (Preventing Hardware Damage)
Modern CPUs and GPUs feature built-in hardware limiters (PROCHOT#, THERMTRIP#, PL1/PL2 power caps). However, unconstrained high-performance software can still cause hardware degradation, power supply trips, or thermal fatigue. Kinesis implements four hardware design safeguards:

1. **Transient Power Shock Dampening ($V_{droop} = L \frac{di}{dt}$)**:
   * **The Hazard**: When thousands of SIMD/Tensor Core execution units transition from idle to full load in a single clock cycle, the instantaneous current ramp rate ($\frac{di}{dt}$) causes an inductive voltage drop across motherboard Voltage Regulator Modules (VRMs). This $V_{droop}$ can dip core voltage below minimum silicon stability ($V_{min}$), causing instant system crashes or tripping power supply Over-Current Protection (OCP).
   * **Kinesis Defense**: Implements **Staged Workload Warmup** (`HardwareSafetyPolicy::enable_staged_warmup`), micro-stepping initial matrix iterations across smaller tile slices to ramp current smoothly over several microseconds.
2. **Thermal Fatigue & Electromigration Mitigation**:
   * **The Hazard**: Constantly cycling silicon temperature between 35°C and 85°C produces mechanical shear stress on BGA solder joints due to thermal expansion mismatch ($\Delta \alpha \cdot \Delta T$). Also, running at prolonged high temperatures accelerates electromigration in nano-scale copper interconnects.
   * **Kinesis Defense**: Continuous temperature monitoring with proactive software throttling at $85^\circ\text{C}$ (warning) and $92^\circ\text{C}$ (critical yield), pacing batch iterations before the hardware triggers its catastrophic thermal clock cliff (e.g. throttling down to 400 MHz).
3. **VRM & VRAM Hotspot Relief via Cache-Pinned Micro-Panels**:
   * **The Hazard**: High-frequency GDDR6X memory chips and motherboard VRM MOSFETs run significantly hotter than the GPU core itself. Streaming continuous unblocked matrix sweeps forces constant 100% duty cycle on external memory buses, overheating memory chips.
   * **Kinesis Defense**: GotoBLAS micro-panel packing pins active compute working sets inside on-die L1/L2 caches, slashing off-chip VRAM bus transactions by up to $80\%$.
4. **Memory Quota Fencing**:
   * Prevents runaway allocations from invoking the OS Out-Of-Memory (OOM) killer or triggering disk paging swap thrashing, which degrades NVMe SSD endurance and causes 10,000x latency spikes.


> **Review note (2026-09-17):** None of these safeguards is implemented (`HardwareSafetyPolicy` is never used, although the system report prints them as ACTIVE). Physically, the CPU already stages wide-SIMD transitions itself, Travis Downs measured ≈ 8-20 µs voltage transitions at ¼ dispatch rate and ≈ 680 µs before relaxing, so "staged warmup" is not a software safety mechanism, although avoiding sporadic wide-SIMD bursts is a valid *latency* technique. CPU-side packing cannot reduce GPU VRAM traffic (item 3). Treat these as hypotheses until implemented and measured. See [T17](DESIGN_REVIEW_2026-09-17.md#t17) and [§2.2](DESIGN_REVIEW_2026-09-17.md#s2-2).
>
> **Partly resolved (2026-09-18):** The system report no longer claims these protections are active, it prints `[NOT IMPLEMENTED]`, and `HardwareSafetyPolicy` now carries a warning that constructing one enforces nothing. None of the safeguards exists yet. See [§6.0](DESIGN_REVIEW_2026-09-17.md#s6-0) (defect 6).

### C. Architecture Decision: User-Space Library vs. Kernel Driver
* **Why Kinesis Is NOT a Kernel Driver**:
  * An OS kernel driver (Ring 0 / `.sys` / `.ko`) introduces severe liability: any algorithmic flaw causes an instant operating system Blue Screen of Death (BSOD) or kernel panic. Also, drivers require root/administrator privileges, WHQL code-signing certificates, and cannot easily link standard high-level math libraries.
* **Achieving Driver-Grade Speed in User-Space (Ring 3)**:
  * Modern high-performance AI engines (TensorRT, PyTorch, llama.cpp, CUDA runtime) operate in user-space but bypass OS bottlenecks through:
    1. **Direct Virtual Page Allocation & Locking**: `VirtualAlloc(MEM_COMMIT)` / `mmap(MAP_HUGETLB)` combined with `VirtualLock` / `mlock` to keep physical pages locked in RAM.
    2. **CPU Core Affinity & NUMA Node Pinning**: Pinning worker threads to dedicated physical cores (`SetThreadAffinityMask` / `pthread_setaffinity_np`) to prevent OS scheduler migrations and preserve L1/L2 cache warmth.
    3. **Direct Hardware Queue Doorbell Submissions**: Submitting command buffers directly to GPU hardware doorbell registers via Vulkan / DX12 / CUDA without kernel context-switch overhead.


> **Review note (2026-09-17):** Page locking, core affinity / NUMA pinning and user-mode GPU submission are valid techniques, but none is implemented; the current default allocator issues one `VirtualAlloc`/`mmap` per tensor with no prefaulting or locking. User-mode submission details are vendor- and driver-specific (❓ unsourced here). See [§2.2](DESIGN_REVIEW_2026-09-17.md#s2-2) and [§5](DESIGN_REVIEW_2026-09-17.md#s5).
>
> **Partly resolved (2026-09-18):** The default allocator is no longer one `VirtualAlloc`/`mmap` per tensor; it is an aligned heap allocator, and page allocation is reserved for large regions such as an arena's backing store. Page locking, prefaulting and pinning are still not implemented.

### D. Embedded Robustness: Raspberry Pi vs. ESP32

| Feature / Metric | Raspberry Pi (Pi 4 / Pi 5) | ESP32 (LX6 / LX7 / RISC-V) |
| :--- | :--- | :--- |
| **Compute Profile** | `PlatformProfile::EdgeSBC` | `PlatformProfile::EmbeddedMCU` |
| **Operating System** | 64-bit Linux (Debian / Ubuntu) with MMU | Bare-metal / FreeRTOS (No MMU) |
| **Memory Capacity** | 2 GB, 16 GB LPDDR4X | **520 KB Internal SRAM** (Optional 4-8MB PSRAM) |
| **Virtual Memory (`mmap`)** | ✅ Fully Supported (`virtual_memory.hpp`) | ❌ **No Virtual Memory / No OS Pages** |
| **SIMD Vector Engine** | ARM NEON (128-bit vector registers) | Custom DSP MAC instructions (`esp-dsp`) |
| **GPU Acceleration** | Broadcom VideoCore VI/VII (Vulkan Compute) | ❌ None (Pure Microcontroller CPU) |
| **Kinesis Allocation Model** | Dynamic Page Arena + Slab Allocator | **Static SRAM Arena (`uint8_t g_ram[64*1024]`)** |

* **The Embedded Zero-Virtual Profile (`KINESIS_EMBEDDED`)**:
  * On microcontrollers like the ESP32, Kinesis compiles with `KINESIS_EMBEDDED`, completely disabling the OS virtual memory layer.
  * Tensors and forward-mode automatic differentiation operate directly over statically allocated SRAM buffers (`ArenaAllocator(g_sram_arena, 64 * 1024)`).
  * This guarantees **zero heap fragmentation, zero system calls, and sub-millisecond execution times** for real-time motor control, state estimation, and sensor fusion directly on microcontrollers.


> **Review note (2026-09-17):** `KINESIS_EMBEDDED` only selects an enum value; `virtual_memory.hpp` still includes `<windows.h>`/`<sys/mman.h>` unconditionally and core headers pull in `<iostream>`. The state-of-the-art route on ESP32 is TensorFlow Lite Micro with Espressif's ESP-NN kernels (MobileNetV3-Small drops from 26 s to 1.4 s on ESP32-S3), which also sets realistic budgets for the "1000 Hz" claims. VideoCore Vulkan compute is plausible on a Raspberry Pi 5 but was reported slower than the CPU on a Pi 4 (ncnn issue #2435). See [§2.2](DESIGN_REVIEW_2026-09-17.md#s2-2) and [§2.3](DESIGN_REVIEW_2026-09-17.md#s2-3).

---

## 15. The 64-Byte Cache-Line TensorView & Microarchitectural Optimizations

In high-performance systems engineering, data structure layout must mirror the physical hardware cache hierarchy:

### A. 32-Bit Descriptor Compression (`Dim = u32`)
* In prior iterations, `Shape` and `Strides` stored dimensions using 64-bit integers (`int64_t`). With 6 dimensions, `Shape` was 56 bytes and `Strides` was 48 bytes, resulting in a **120-byte `TensorView`**.
* Passing a 120-byte struct by value crosses **two distinct 64-byte CPU cache lines**, causing cache-line splits and forcing the compiler to spill metadata to stack memory.
* In neural networks and numerical modules, individual dimensions never exceed $2^{31} - 1 \approx 2.14 \times 10^9$ elements. Compressing dimensions to `uint32_t` (`Dim = u32`):
  * `Shape`: 6 dimensions $\times 4\text{B} = 24\text{B} + 4\text{B rank} = \mathbf{28\text{ bytes}}$.
  * `Strides`: 6 strides $\times 8\text{B} = \mathbf{48\text{ bytes}}$ (retaining 64-bit strides to safely address multi-gigabyte virtual buffers).
  * Data pointer (8B) + offset (8B) = 16 bytes.
  * Total `TensorView` fits tightly within the hardware cache budget, mapping cleanly into vector registers.
* **Capability Invariant**: `total_elements()` returns `int64_t`, ensuring total volume never overflows on multi-billion element tensors.


> **Review note (2026-09-17):** ❌ `TensorView` is **104 bytes** on x86-64 (`Strides` also stores a `size_t` rank, plus padding and a 64-bit offset), and neither 104 nor 64 bytes would be passed in vector registers (Windows x64 passes such aggregates by pointer). `uint32_t` dims silently truncate in release builds and conflict with GGUF's `uint64` dimensions; "individual dimensions never exceed 2³¹ − 1" does not hold for flattened views of large buffers. See [T5](DESIGN_REVIEW_2026-09-17.md#t5), [§2.1](DESIGN_REVIEW_2026-09-17.md#s2-1) (L7, L12) and [§6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defect 4).
>
> **Partly resolved (2026-09-18):** Out-of-range dimensions now throw in every build instead of truncating. The rest stands: `TensorView` is still 104 bytes, and the 32-bit extent cap is a design decision ([T5](DESIGN_REVIEW_2026-09-17.md#t5)), not a defect. See [§6.0](DESIGN_REVIEW_2026-09-17.md#s6-0) (defect 4).

### B. $O(1)$ Branchless Word-Level Equality (`Shape::operator==`)
* Rather than looping $O(N)$ across dimensions with early-exit branches, `Shape::operator==` evaluates in **1 single clock cycle** using bitwise XOR across packed 64-bit words:
  ```cpp
  const uint64_t* p1 = reinterpret_cast<const uint64_t*>(dims_.data());
  const uint64_t* p2 = reinterpret_cast<const uint64_t*>(other.dims_.data());
  return ((p1[0] ^ p2[0]) | (p1[1] ^ p2[1]) | (p1[2] ^ p2[2])) == 0;
  ```
  Since unused dimensions are zero-initialized, all 24 bytes are compared simultaneously with zero branch mispredictions.


> **Review note (2026-09-17):** ❌ Reading a `uint32_t` array through `const uint64_t*` is undefined behaviour (strict aliasing, plus under-aligned 8-byte loads), and the zero-slot invariant breaks after any out-of-rank write in a release build. It is also not "1 clock cycle". `bool operator==(const Shape&) const = default;` (or comparing `rank_` and `dims_` directly) compiles to equivalent branch-free code without UB. See [§6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defect 3).
>
> **Resolved (2026-09-18):** Fixed. `Shape::operator==` compares `rank_` and `dims_` directly.

### C. Loop Fission in Broadcasting (`broadcast_to`)
* Slicing and broadcasting previously checked `if (i < rank_diff)` inside every iteration of a single loop.
* Applying **Loop Fission**:
  1. Loop 1 zero-fills the leading broadcast dimensions unconditionally (`new_strides[i] = 0`), vectorizing into a single SIMD store.
  2. Loop 2 maps the trailing source dimensions directly (`new_strides[dst] = strides_[src]`).
* Eliminates all conditional branching from the hot broadcasting kernel.


> **Review note (2026-09-17):** Loop 2 still contains two data-dependent branches (and a `throw`), so branching is not eliminated. GCC already enables `-fsplit-loops` / `-funswitch-loops` at `-O3`, and with rank ≤ 6 the change is a readability improvement rather than a performance one. See [§8](DESIGN_REVIEW_2026-09-17.md#s8).

### D. Rectangular Matrix Gramians & Closed-Form Cramer Solvers
1. **Gramian Normalization ($G = A^T A$)**:
   * Rectangular non-square matrices ($M \neq N$, $M > N$) have no direct inverse.
   * `compute::gramian(A, G)` computes $G = A^T A \in \mathbb{R}^{N \times N}$ via cache-blocked GEMM, enabling the **Moore-Penrose Pseudoinverse** ($A^+ = (A^T A)^{-1} A^T$) for least-squares regression.
2. **Branchless Closed-Form Cramer Solvers**:
   * `compute::solve_cramer_2x2` and `compute::solve_cramer_3x3` evaluate linear systems $A x = b$ using direct determinant expansion in CPU registers with 0 heap allocations, optimal for real-time 3D robotics and camera calibrations.


> **Review note (2026-09-17):** Forming AᵀA squares the condition number (κ(AᵀA) = κ(A)²), so least-squares solvers should prefer QR or SVD; A⁺ = (AᵀA)⁻¹Aᵀ also requires full column rank, and a symmetric rank-k update (SYRK) computes AᵀA more cheaply than general GEMM. `gramian` currently returns wrong results for matrices with ≥ 128 columns (packed-kernel stride defect). The Cramer solvers do branch and divide, and their absolute `|det| < 1e-12f` threshold is not scale-invariant. See [§6](DESIGN_REVIEW_2026-09-17.md#s6-1) (defects 1, 12) and [§7](DESIGN_REVIEW_2026-09-17.md#s7).
>
> **Partly resolved (2026-09-18):** `gramian` is correct at every width, and the Cramer solvers use a relative threshold scaled by the magnitude of the terms that form the determinant, so they are scale invariant. They still branch and divide, and the numerical advice above (QR or SVD over the normal equations; SYRK for AᵀA) is unaffected. See [§6.0](DESIGN_REVIEW_2026-09-17.md#s6-0) (defects 1, 12).

---

## 16. Dual Numbers & Multi-Tangent Forward-Mode Automatic Differentiation

### A. The Dual Number Algebra ($\epsilon^2 = 0$)
Kinesis implements forward-mode autodiff via stack-allocated `Dual<T>` numbers ($x + \dot{x}\epsilon$, where $\epsilon^2 = 0$):
* Taylor series expansion shows exact derivatives emerge in the dual coefficient:
  $$f(x + \epsilon) = f(x) + f'(x)\epsilon$$
* Eliminates graph construction and tape allocation overhead for functions with few inputs and many outputs.

### B. Multi-Tangent `HyperDual<T, N>`
* Propagates derivatives with respect to $N$ independent parameters simultaneously in a single forward pass.
* Evaluates robotics forward dynamics and state Jacobians in single-pass $O(N)$ time with zero tape memory.

> **Review note (2026-09-17):** `HyperDual<T, N>` is a first-order N-tangent dual number (Ceres calls this a `Jet`); hyper-dual numbers in the literature (Fike & Alonso) carry an ε₁ε₂ term for exact second derivatives, which this type lacks, and the `FloatingPoint` constraint prevents nesting `Dual<Dual<T>>`. "State Jacobians in single-pass O(N)" is incorrect: an n×n Jacobian has n² entries, and forward mode with N tangents multiplies each operation's cost by N. Storing dual numbers as tensor elements also conflicts with SIMD layout and float-only kernels. See [T6](DESIGN_REVIEW_2026-09-17.md#t6), [T7](DESIGN_REVIEW_2026-09-17.md#t7), [§2.4](DESIGN_REVIEW_2026-09-17.md#s2-4) and [§2.5](DESIGN_REVIEW_2026-09-17.md#s2-5).
>
> **Partly resolved (2026-09-18):** Renamed `Jet<T, N>` to match Ceres, with `HyperDual` kept as a deprecated alias. A structural `DualScalar` concept replaces `FloatingPoint`, so `Dual<Dual<T>>` is legal and yields exact second derivatives in `.dot.dot`. The complexity and SIMD-layout points above are unaffected. See [§6.0](DESIGN_REVIEW_2026-09-17.md#s6-0) (defect 11).

---

> **Design review (2026-09-17):** A full strategy, logic and architecture review of this document, with recomputed numbers, SOTA comparisons (papers, vendor documentation, repositories, Hugging Face) and implementation status, is in [DESIGN_REVIEW_2026-09-17.md](DESIGN_REVIEW_2026-09-17.md). The notes above mark where a claim is qualified or contradicted; the original text is kept for history.
