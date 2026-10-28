# Kinesis Design Review: Strategy, Logic & Architecture

**Date:** 2026-09-17
**Scope:** commit `fe62642` plus the uncommitted working tree
**Type:** review (2026-09-17), followed by a remediation pass (2026-09-18).
**Status:** all 15 findings in [§6.1](#s6-1) are resolved, and every item in [§6.2](#s6-2) is either fixed or explicitly deferred with a reason. The suite went from 35 tests to **53, all passing**, in both a debug build and `-O3 -DNDEBUG`.

> The review itself was carried out without building or running anything: every **CONFIRMED** verdict below was reached by recomputing the math, reading the code, or checking a primary source. The fixes came afterwards, and each one is pinned by a test that fails against the pre-fix code, see [§6.0](#s6-0).

> This review challenges the project's own documents instead of taking them at face value. Each strategy was checked three ways:
> 1. The math and logic in the docs were recomputed.
> 2. The implementation behind each claim was read.
> 3. Each claim was compared against papers, vendor documentation, state-of-the-art repositories and Hugging Face resources (§10).
>
> Inline **Review note (2026-09-17)** callouts in the other documents link back to this file.

**Evidence legend**

| Mark | Meaning |
|---|---|
| ✅ | Validated: the code or source agrees |
| ⚠️ | Partially true: context-dependent, or the right idea with a wrong detail |
| ❌ | Contradicted by math, code or source |
| ❓ | Not found in the cited sources, or never measured |

## Contents
- [§0 Scope & method](#s0)
- [§1 Executive summary](#s1)
- [§2 Strategy & logic validation](#s2)
  - [§2.1 Logic & numeric checks](#s2-1)
  - [§2.2 CPU strategies](#s2-2)
  - [§2.3 GPU strategies](#s2-3)
  - [§2.4 Learning & optimizer strategies](#s2-4)
  - [§2.5 Robotics, autodiff & quant strategies](#s2-5)
- [§3 Architecture coherence: do the decisions work together? (T1-T17)](#s3)
- [§4 SOTA optimization map (CPU & GPU)](#s4)
- [§5 Implementation-status matrix](#s5)
- [§6 Code defects from `/code-review max`](#s6)
- [§7 Documentation accuracy corrections](#s7)
- [§8 Conventions & low-level notes](#s8)
- [§9 Suggested decision order](#s9)
- [§10 Sources](#s10)

---

<a id="s0"></a>
## §0 Scope & method

- **Code reviewed:**
  - all of `include/`, `src/`, `tests/` and `benchmarks/`
  - `CMakeLists.txt`, `build.ps1`, `build.bat`
  - `docs/memory_sandbox_visualizer.html`
- **Documents reviewed:**
  - `README.md`
  - `docs/ARCHITECTURE.md`, `docs/BENCHMARKS.md`, `docs/CHANGELOG.md`, `docs/LEARNING_GUIDE.md`
  - `docs/LEARNING_JOURNAL.md` (Invalidations 1-18)
  - `docs/PRODUCTION_SPEC_AND_CRITIQUE_ANALYSIS.md`, `docs/GGML_ANALYSIS_AND_IMPROVEMENTS.md`
- **Method:**
  1. Recomputed every checkable number and formula in the docs using pure arithmetic or Monte Carlo (results in §2.1).
  2. Read the implementation behind each claim. File:line references point to the working tree as of 2026-09-17.
  3. Compared each strategy with primary sources (§10). Nothing is attributed to a source that wasn't read. Where only a secondary summary was available, this is stated.
- **Not done:** no build, test run or benchmark run. Defects marked "CONFIRMED" were confirmed by code and type analysis or math, not by execution. The pass status of the 35 test cases should be re-verified.
- **Intent assumed:** Kinesis is meant to be a **CPU and GPU** library with optimizations across the whole stack, per the project owner. GPU strategies are therefore evaluated as in-scope plans.

<a id="s1"></a>
## §1 Executive summary

*As written on 2026-09-17. Items that the 2026-09-18 remediation pass fixed are marked ✅ in place; everything unmarked is still open, and most of it is a design question rather than a defect.*

1. **The core primitives are sound and conventional.** The bump arena with scoped rewind, intrusive free-list slab, strided non-owning views, cache-blocked GEMM and stack-allocated dual numbers all match established designs (`std::pmr`, `std::mdspan`, BLAS, Ceres `Jet`, Stan Math). The project's value lies in combining them well, not in novelty.
2. **Several headline strategies point in the right direction according to top sources.**
   - GPU on-chip fast memory per streaming multiprocessor, with work tiled to fit it: NVIDIA Ada/Ampere tuning guides; FlashAttention §2.1.
   - Stream-ordered GPU memory pools: CUDA runtime.
   - Allocation-free, caller-controlled kernels: Arm KleidiAI.
   - Reversible networks for activation memory: Rev-ViT.
   - Paged KV caches: vLLM.
   - Muon and Shampoo-class optimizers: Moonshot's "Muon is Scalable", AlgoPerf 2024.
3. **The reasoning behind many claims contains checkable errors** (§2.1):
   - a cache-stall estimate off by 1000×;
   - a TCN receptive-field claim off by ~64×;
   - a Newton-Schulz divergence bound that belongs to a different iteration;
   - the wrong forward-gradient variance constant;
   - an equal-FLOPs backprop/DFA comparison that ignores output width;
   - XOR "swizzle" formulas that still leave 4-way conflicts.

   Several citations are also wrong: the GrAPE venue and details, the matrix-multiplication exponent described as a lower bound, and the Ada L2 size.
4. **Several decisions conflict with each other, and the docs never discuss it** (§3):
   - zero-copy strided views vs "contiguous-only hot loops";
   - Dual-element tensors vs SIMD/SoA layout;
   - "affine indexing only" vs paged KV, embeddings and ring buffers;
   - two arena lifetimes vs training;
   - const views vs read-only weights, ✅ *fixed 2026-09-18: `TensorView<T>` converts to `TensorView<const T>`*;
   - bit-exact determinism vs ISA-specific kernels.
5. **The GPU ambition has no architectural foothold yet** (T13). `TensorView` holds a raw host `T*`, `Allocator` is host-only, and there are no memory spaces, buffer types, streams, events or scheduler. State-of-the-art CPU+GPU systems (ggml backends, MLX, Candle, PyTorch's caching allocator + CUDA Graphs) make these first-class *before* writing device kernels.
6. **GEMM is only partially "GotoBLAS", and it has two correctness defects.**
   - Design: it packs only B, uses an effective 1×8 micro-kernel and fixed 64³ blocks. BLIS Haswell uses a 6×16 single-precision kernel with KC=256/MC=168/NC=4080 and packs both A and B.
   - Defects: strided inputs silently give wrong results, which breaks `gramian` for N ≥ 128; SIMD tails read and write past the row end, ✅ *fixed 2026-09-18, both demonstrated against the pre-fix code (§6.0)*.
7. **The descriptor "wins" aren't real.**
   - `TensorView` is 104 bytes, not 64, and the ABI passes it by pointer anyway.
   - `Shape::operator==` has undefined behaviour, ✅ *fixed 2026-09-18*.
   - `uint32_t` dimensions truncate silently in release builds, while GGUF itself uses `uint64` dimensions, ✅ *fixed 2026-09-18: out-of-range extents now throw in every build. The 32-bit cap itself is a T5 decision and remains*.
8. **`HyperDual<T,N>` is a first-order multi-tangent dual** (Ceres calls this a `Jet`), not a hyper-dual number. The second-order FoMoH/DRFC plans have no primitive to build on, ✅ *fixed 2026-09-18: renamed `Jet`, and a structural `DualScalar` concept makes `Dual<Dual<T>>` legal, which yields f″*.
9. **`HardwareInfo` doesn't detect what it reports.**
   - The GPU result is hard-coded, ✅ *fixed 2026-09-18: it now reports `Unknown`*.
   - AVX support skips the OS-state (XGETBV) check, ✅ *fixed 2026-09-18*.
   - Cache sizes are constants: ✅ *fixed 2026-09-18: read from CPUID, which showed every constant was wrong for the development machine*.
   - The "silicon protection" policy is never used, and CPUs already manage wide-SIMD voltage/frequency transitions in hardware. *(Still unused; the report no longer calls it ACTIVE.)*
10. **The evidence base isn't yet trustworthy.**
    - The slicing benchmark has undefined behaviour and is constant-folded, ✅ *fixed 2026-09-18*.
    - Allocator timings include page faults, ✅ *fixed 2026-09-18: faults are now measured separately, and turn out to cost 4× the steady-state allocation*.
    - No code produces the naive-GEMM baselines. *(Still true: the tables have not been regenerated.)*
    - The same metrics carry different numbers in different documents.

---

<a id="s2"></a>
## §2 Strategy & logic validation

<a id="s2-1"></a>
### §2.1 Logic & numeric checks (recomputed during this review)

| # | Claim (where) | Check performed | Result |
|---|---|---|---|
| L1 | "In a tight loop executing 1 billion iterations, a 1% drop in cache hit rate adds 4.7 million stalled cycles" (LEARNING_JOURNAL Inv. 10, takeaway 1) | 1e9 × 0.01 × (500 − 30) | **4.7 billion** stalled cycles. ❌ Off by 1000× |
| L2 | "With 6 dilated layers, a compact model can observe an entire historical window of 4,096 order book ticks" (LEARNING_JOURNAL Inv. 14, Decision 4) | Receptive field of stacked dilated causal convolutions, R = 1 + (k−1)·Σ dₗ with d = 1, 2, …, 32 | k = 2 → **64**; k = 3 → **127**. Reaching ≥ 4,096 needs **12 layers** (k = 2 gives exactly 4,096). ❌ |
| L3 | "Newton-Schulz iterations diverge catastrophically if the maximum singular value σ_max(X₀) ≥ √3" (PRODUCTION_SPEC Fault 4) | Iterated the scalar maps | √3 is the bound for the **cubic** iteration 1.5x − 0.5x³ (converges at 1.73, diverges at 1.74). Muon's **quintic** map with (3.4445, −4.7750, 2.0315) diverges from σ₀ ≈ **1.26** and never converges to 1; iterates settle in ≈ [0.68, 1.12], consistent with Keller Jordan's description. ⚠️ Bound attributed to the wrong iteration, though Frobenius normalization (σ ≤ 1) keeps the quintic safe ✅ |
| L4 | "E‖g_est − ∇f‖² = (D − 1)‖∇f‖²" with v ~ N(0, I) (PRODUCTION_SPEC Fault 1) | Monte Carlo, D = 50, 40,000 samples | Gaussian v: **50.9 ≈ D + 1**; Rademacher v: **49.0 ≈ D − 1**. ⚠️ (D − 1) is the constant for Rademacher perturbations. The conclusion that variance grows with D still stands |
| L5 | "XOR swizzling … mathematically guaranteeing that all 32 threads land on independent memory banks" (ARCHITECTURE §9.D; LEARNING_GUIDE Experiment 3; visualizer) | Enumerated bank assignments for 32 threads | For a linear stride of *s* elements, the worst conflict is exactly **gcd(s, 32)** (8 B → 2×, 16 B → 4×, 128 B → 32×, 132 B → 1×), so **any odd element stride is conflict-free**. The guide's formula (t·128/4) ⊕ (t/4) and the visualizer label's (t·stride/4) ⊕ (t≫2) both still produce **4-way conflicts** (8 banks). The visualizer only shows 1.0× because its code discards the stride. ❌ |
| L6 | Backprop and DFA both cost "2 · M · K · N" backward FLOPs per layer (PRODUCTION_SPEC Fault 2 table) | Per-layer cost at hidden width d = 4,096 | Backprop (Wᵀδ): ≈ 2·d² = 3.4e7. DFA (B·e): ≈ 2·d·n_out, so n_out = 10 gives 8.2e4 (**~400× cheaper**) and n_out = 50,257 (GPT-2 vocabulary) gives 4.1e8 (**~12× more expensive**). ❌ DFA's cost depends on output width |
| L7 | "Total `TensorView` fits tightly within the hardware cache budget" / "exactly 64 bytes" (ARCHITECTURE §15.A; Inv. 17/18) | x86-64 layout: pointer 8 + `Shape` 28 + padding 4 + `Strides` 56 (`size_t` rank + 6 × `int64_t`) + offset 8 | **104 bytes**. The previous layout was 128 bytes, not 120. ❌ |
| L8 | "16 concurrent users with 8k context requires over 32 GB of RAM strictly for the KV cache" (GGML doc Flaw 1) | 7B multi-head attention in FP16: 2 × 32 layers × 4,096 × 2 bytes per token | 0.5 MiB/token → **64 GiB**. ✅ Grouped-query-attention models need about 4× less |
| L9 | GELU approximation constant 0.044715 (`include/kinesis/compute/elementwise.hpp:110`) | Maximum \|error\| against the exact erf GELU on [−6, 6] | **4.7e-4** (0.147 would give 4.1e-2). ✅ Good choice. Document it as the GPT-2 "gelu_new" tanh approximation, since PyTorch's default GELU is the exact erf form |
| L10 | Tile size B = 64 for a 48 KB L1 (ARCHITECTURE §7) | 3 × 64² × 4 = 49,152 | ✅ Arithmetic is correct. ⚠️ But this is a loop-tiling model, not the GotoBLAS/BLIS model (§2.2) |
| L11 | Packing is ~1,000× cheaper than compute at N = 1,024 (ARCHITECTURE §12) | 2 × 1024² ≈ 2.1e6 writes vs 2 × 1024³ ≈ 2.15e9 FLOPs | ✅ Ratio ≈ 1 : 1,024. ❓ The "3× to 5× net compute acceleration" was never measured |
| L12 | "Individual dimensions never exceed 2³¹ − 1" (ARCHITECTURE §15.A; PRODUCTION_SPEC Point 1) | Counter-examples | GGUF stores tensor dimensions and offsets as **uint64**, and flattened views of multi-GB buffers exceed 2³² elements. ❌ |
| L13 | "The theoretical lower bound for dense matrix multiplication and inversion … is ∼O(N^2.371) (Alman-Williams 2024)" (PRODUCTION_SPEC Point 4) | Literature | ω < 2.371339 (*More Asymmetry Yields Faster Matrix Multiplication*, 2024) and ω < 2.371177 (arXiv 2608.16884) are **upper** bounds. The known lower bound is Ω(n²). ❌ |

<a id="s2-2"></a>
### §2.2 CPU strategies vs evidence

| Strategy (where) | Evidence | Verdict & implication |
|---|---|---|
| Bump arena + RAII scoped rewind (ARCH §1) | `std::pmr::monotonic_buffer_resource` (C++17) releases memory only when destroyed, can start from an initial buffer, and grows geometrically from an upstream resource. TensorFlow Lite Micro uses one arena split into a non-persistent head and a persistent tail, laid out by `GreedyMemoryPlanner` from tensor lifetimes, with scratch buffers requested at prepare time. ggml's graph allocator reserves for a worst-case graph and allocates every tensor before compute. Stan Math allocates reverse-mode memory from an arena that stays valid until `recover_memory()` | ✅ as a primitive. ⚠️ What those SOTA systems add, and Kinesis lacks, is a **planning pass** (sizes and offsets known before the hot loop) and **explicit lifetime classes** (T1, T2) |
| Replace `malloc` with direct OS pages (`HeapAllocator = VirtualMemoryAllocator`) (Inv. 4; ARCH §6) | Allocating each object with `mmap`/`VirtualAlloc` costs a system call and rounds up to whole pages (a 100-byte request wastes ~97% of a 4 KB page). Each `munmap` in a multithreaded process triggers TLB-shootdown IPIs, which is why user-space allocators request large chunks and sub-allocate. Latency-sensitive systems prefault and lock memory (`MAP_POPULATE`, `mlock`/`mlockall`) so the hot path never takes a page fault (Rigtorp; Red Hat Enterprise Linux for Real Time) | ❌ as the *default per-tensor* allocator. ✅ for large reservations with commit-on-demand (the CPU analogue of PyTorch's `expandable_segments`), combined with sub-allocation and prefaulting |
| Segregated slab with intrusive free list (ARCH §6, §12.A.2) | Standard fixed-size pool design, e.g. `std::pmr::unsynchronized_pool_resource` | ✅ as a primitive. ❌ The "free-list validation" cited as a Pragmatism Principle example doesn't exist: size and ownership checks are `assert`-only |
| GotoBLAS/BLIS packing & micro-kernel (ARCH §9.A; Inv. 8) | BLIS Haswell registers `bli_sgemm_haswell_asm_6x16` (MR = 6, NR = 16) with KC = 256, MC = 168, NC = 4080 for single precision, and packs **both** A and B. The salykova tutorial derives MR/NR from the number of YMM registers and sizes kc·nr to fill L1, mc·kc to fill L2 and kc·nc to fill L3. Yotov et al. (SPAA 2007): high-performance codes "must at least be processor-conscious", with micro-kernels "carefully optimized for the processor pipeline, registers and L1 I-cache"; the ATLAS micro-kernel keeps a block of C in registers | ⚠️ Kinesis packs only B and accumulates one row of C at a time, which is effectively a 1×8 kernel using one register. It also ties every block size to L1 (64³) and repacks B for every 64×64 block. "Nr matches SIMD register width" (Inv. 8) is a misunderstanding: MR × NR is chosen to fill the *register file* |
| "3·B²·sizeof(T) ≤ L1" tile model (ARCH §7; Inv. 5) | The BLIS analytical model (Low et al., TOMS 2016) derives KC/MC/NC for each cache level from cache geometry | ⚠️ The formula is a classic loop-tiling bound, not the model GotoBLAS/BLIS use. The "dynamic" sizing also isn't wired up: `gemm.cpp:136` hard-codes 49,152 |
| Morton / cache-oblivious re-sorting that "guarantees spatial locality recursively at every level" (ARCH §9.F) | Yotov et al. 2007: "even highly optimized cache-oblivious programs perform significantly worse than corresponding cache-conscious programs"; "use of recursive microkernels is not recommended" | ❌ as a performance strategy for dense linear algebra. Morton order is still useful for spatial indexing |
| Offline weight pre-packing (ARCH §9.B) | XNNPACK supplies low-level operators to TFLite, ExecuTorch, ONNX Runtime and MediaPipe; KleidiAI supplies packing routines plus micro-kernels | ✅ common SOTA practice, but ❌ not implemented. For B = 1 inference (`y = Wx`) the operation is a matrix-*vector* product, where GEMM-style B̃ panels bring little; quantized or SIMD row kernels matter more |
| NHWC + `im2col` / "Implicit GEMM" for CPU vision (ARCH §9.C; Inv. 8) | "Implicit GEMM" is a cuDNN (GPU) algorithm. On CPUs, the Indirect Convolution Algorithm replaces im2col's data copy with an indirection buffer: memory drops in proportion to input channels, and it runs up to 62% faster than GEMM-based convolution (Dukhan 2019) | ⚠️ The goal is right but the terminology is GPU-specific. The CPU SOTA relies on *indirection*, a dependent read that ARCH §11's "affine only" rule forbids (T9) |
| Compile-time ISA selection (`include/kinesis/compute/simd.hpp:6-14`) | ggml builds several CPU backend variants and picks one at runtime via `ggml_backend_score()` (PR #10256); OpenBLAS offers `DYNAMIC_ARCH`; Google Highway dispatches at runtime | ❌ for anything that runs anywhere other than the build machine (T11) |
| ARM targets: hand-written NEON kernels for Raspberry Pi / Apple Silicon (ARCH §5, §14.D) | **Arm KleidiAI** provides micro-kernels for NEON, SVE, SME/SME2, DotProd and I8MM. Its design has "No dynamic memory allocation", "No memory management" and no internal threading: the framework decides what to process and on which thread. On Apple Silicon, reverse-engineering and academic work report that Accelerate's BLAS routines (e.g. `sgemm`) run on the undocumented AMX matrix coprocessor | ✅ KleidiAI is the closest SOTA match to Kinesis' zero-allocation kernel philosophy and a good model for its API. ⚠️ On Apple Silicon, benchmark any hand-written NEON GEMM against Accelerate |
| ESP32 "zero-virtual" profile running dynamics + inference at 1 kHz (ARCH §14.D; Inv. 12) | Espressif **ESP-NN** provides optimized kernels for ESP32/ESP32-S3/P4 and is used by TensorFlow Lite Micro. On ESP32-S3, convolution runs 14.24× faster than ANSI C, and MobileNetV3-Small inference drops from 26 s to 1.4 s | ⚠️ The SOTA route on MCUs is TFLite Micro + vendor kernels + an arena planner, which also sets realistic budgets for "1 kHz" claims. ❌ `KINESIS_EMBEDDED` isn't implemented: `virtual_memory.hpp` includes `<windows.h>`/`<sys/mman.h>` unconditionally |
| Silicon protection: "Staged Workload Warmup" to dampen VRM di/dt (ARCH §14.B; Inv. 12) | Travis Downs measured wide-SIMD transitions on Intel. The processor's own power management runs a voltage transition of ~8-20 µs at ¼ the normal instruction dispatch rate, adds a ~10 µs halted frequency transition for heavy AVX-512, and waits ~680 µs of disuse before reverting | ❌ as software hardware protection, since the CPU already stages the transition. ✅ as a **latency** concern: avoid sporadic wide-SIMD bursts in latency-critical loops, or keep the core in its wide-vector state |
| Cache latencies used by the sandbox and visualizer: 30 vs 500 cycles (ARCH §11, §13) | Chips and Cheese report 4-cycle latency for Skylake's 32 KB L1D and ~12 cycles for L2 (Skylake 256 KB, Zen 2 512 KB). The 30/500 figures are Charles Grassi's **GPU** numbers | ❌ for a CPU L1 model: the sandbox mixes CPU cache geometry with GPU latencies |
| "Cache-line split" hazard detection (ARCH §13; visualizer) | Split **locks** (LOCK-prefixed operations spanning two cache lines) cost thousands of cycles and trigger kernel bus-lock detection and throttling (Rigtorp; Linux kernel docs). Ordinary loads that cross a line boundary are a much smaller effect than loads that cross a page boundary | ⚠️ Focus detection on split locks and page splits |
| Zero-copy memory-mapped model files (GGML doc §5) | GGUF is designed for `mmap`: tensor data is padded to `general.alignment` (default **32**, must be a multiple of 8), and dimensions and offsets are `uint64`. safetensors advertises zero-copy loading | ✅ as a strategy. ❌ Not representable today because of `uint32_t` dims, float-only kernels, and no dtype or quantized-block model (T5) |
| Strided non-owning views (ARCH §2) | `std::mdspan` (C++23) offers the `layout_right`, `layout_left` and `layout_stride` layout policies, plus an `AccessorPolicy` whose `data_handle_type` can refer to non-host memory | ✅ The design matches the standard. mdspan's accessor model is also the cleanest path to device memory (T4, T13) |

<a id="s2-3"></a>
### §2.3 GPU strategies vs evidence (GPU is in scope)

| Strategy (where) | Evidence | Verdict & implication |
|---|---|---|
| GPUs have fast on-chip memory per streaming multiprocessor; tile work to fit it (ARCH §5, §11; Inv. 3, 10) | NVIDIA's **Ada** tuning guide: combined L1 cache capacity of 128 KB, and 98,304 KB of L2 on AD102 (16× GA102). NVIDIA's **Ampere** guide: unified L1 of 192 KB (compute capability 8.0) or 128 KB (8.6), and 40 MB of L2 on the A100. **FlashAttention** §2.1: the A100 has "192KB of on-chip SRAM per each of 108 streaming multiprocessors" at ~19 TB/s, versus 40-80 GB of HBM at 1.5-2.0 TB/s, and "operations are increasingly bottlenecked by memory (HBM) accesses" | ✅ The premise is correct. The SOTA delivers it through **IO-aware fused kernels**, not manual address re-sorting: FlashAttention combines tiling, kernel fusion and recomputation, and Triton compiles tile-based kernels. FlashAttention also classifies large-inner-dimension matmul as *compute-bound*; the memory-bound operations (elementwise, softmax, normalization) are where fusion pays off |
| 32/64/128-byte "sector" alignment for GPU work (ARCH §5; `include/kinesis/core/types.hpp:12-21`) | NVIDIA's *Matrix Multiplication Background User's Guide*: before cuBLAS 11, FP16 GEMM needed dimensions that were multiples of 8 to use Tensor Cores. Since cuBLAS 11, "Tensor Cores may be used regardless, but efficiency is better when matrix dimensions are multiples of 16 bytes". On the A100, multiples of up to 128 bytes (64 FP16 elements) can help further | ⚠️ Reinterpret the rule. What matters is that **tensor dimension sizes** (hidden size, vocabulary, batch × sequence) are multiples of 8/16/64 elements, not the byte alignment of allocations. ❓ "32-byte sectors", "128-byte transactions", "Wave32/Wave64", "CDNA" and "MALL" weren't found in any cited source |
| Shared-memory bank conflicts and XOR "swizzling" (ARCH §9.D; Inv. 8; `include/kinesis/core/memory_sandbox.hpp:70-89`; visualizer) | Mark Harris (NVIDIA, 2013): bank geometry depends on compute capability, with 16 banks at 1.x, 32 banks at 2.0, and configurable 4- or 8-byte banks at 3.x. Colliding requests are serialized; the post gives no mitigation and never mentions XOR. Fabian Giesen's "swizzling" (2011) permutes *texture* address bits, e.g. into Morton order. Charles Grassi (2025): "Never manually 'optimize' texture data layout, the driver handles this" | ⚠️ The bank model is valid for CUDA shared-memory kernels at compute capability 2.0. ❌ The XOR guarantee (see L5). ✅ The simple mitigation follows from the math: use odd element strides, or pad by one element |
| Tensor Core "warp shuffle" pathways, `ldmatrix.x4` in-flight re-sorting, `cp.async` (ARCH §9.E; Inv. 1, 8, 10) | None of these appear in the fetched NVIDIA Ada or Ampere tuning guides. The Ampere guide describes asynchronous copies through the `pipeline` API | ❓ Unsourced. SOTA GPU matmul relies on cuBLAS/cuBLASLt with Tensor-Core-friendly shapes |
| 1-2 KB constant-cache knee → quantize/deduplicate weights to fit (ARCH §10.B; Inv. 9) | Danil's article reports this for *his own* low-end NVIDIA GPU and specific OpenGL shaders ("around One or Two Kbyte of CONST is limit for 'good performance'"). In compute kernels, weights live in device buffers, not shader constant pools. The SOTA motivations for low precision are throughput and memory: FP8 (E4M3/E5M2) matches 16-bit training quality with unchanged hyperparameters up to 175B parameters, and Hugging Face's quantization overview lists per-hardware methods (bitsandbytes, GPTQ, AWQ, GGUF, torchao, FP8, …) | ❌ as a general rule for compute kernels. ✅ Quantization itself is a core SOTA optimization |
| "No software prefetching: warp occupancy hides latency"; "mipmaps as cache optimization" (ARCH §11.4-5) | Grassi's article is about **shader** programming and texture sampling. For CUDA compute, the documented levers are asynchronous streams, pinned host memory with `non_blocking` copies, caching allocators and CUDA Graphs (PyTorch CUDA semantics) | ⚠️ True in its original shader context, but not a design rule for compute kernels |
| `CudaStreamPool` wrapping `cudaMemPool_t` (ARCH §6) | The CUDA runtime's **Stream-Ordered Memory Allocator** provides `cudaMallocAsync`, `cudaMemPoolCreate` and `cudaMallocFromPoolAsync`. PyTorch can use it via `backend:cudaMallocAsync` (CUDA 11.4+). PyTorch's native caching allocator frees memory "without device synchronizations", and `expandable_segments` grows one segment per stream instead of creating one per allocation | ✅ A valid, current SOTA choice, but ❌ not implemented |
| Zero allocations in GPU hot paths | PyTorch pairs its caching allocator with **CUDA Graphs**, which replay a whole captured workload with a single `cudaGraphLaunch`, using static addresses from a private memory pool. Hugging Face's `StaticCache` pre-allocates the KV cache so decoding can be compiled, at the cost of running attention over masked padding | ✅ Achievable, but it requires static shapes and explicit trade-offs |
| Bit-exact determinism "regardless of thread scheduling" and "across single-threaded, multi-threaded, and heterogeneous runs" (GGML doc Flaw 4; PRODUCTION_SPEC §2) | PyTorch: "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms", nor between CPU and GPU; `cudnn.benchmark` can pick different algorithms from run to run. Thinking Machines: "the primary reason nearly all LLM inference endpoints are nondeterministic is that the load (and thus batch-size) nondeterministically varies". Their batch-invariant kernels took 42 s vs 26 s for default vLLM on 1,000 sequences (55 s before attention-kernel improvements). SGLang ships a deterministic mode. Intel oneMKL's Conditional Numerical Reproducibility needs a fixed code path (`MKL_CBWR`) and a fixed thread count | ❌ as stated. ⚠️ Achievable inside an explicitly scoped configuration, at a measurable cost (T8) |
| Paged KV cache ("Paged-Slab Attention") (GGML doc Flaw 1) | vLLM's PagedAttention wastes memory only in each sequence's last block (<4%), versus 60-80% measured in *earlier* serving systems. Hugging Face Transformers offers Dynamic, Static, Offloaded and Quantized caches, each with documented trade-offs, plus continuous batching over a paged attention backend. llama.cpp already uses a unified KV cache with `seq_cp` prefix sharing | ✅ A SOTA strategy, but ❌ not implemented. It also contradicts ARCH §11.3's "no dependent reads" rule (T9) |
| CPU+GPU runtime architecture (implied by the roadmap) | **ggml-backend**: a device registry, per-device buffer types (alignment, max size), and a scheduler that "handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends", plus async graph compute with synchronization and events. **MLX**: arrays live in unified memory, the device is chosen per operation, and the scheduler inserts cross-stream dependencies. **Candle** (Hugging Face): CPU (MKL/Accelerate), CUDA with NCCL, WASM, and llama.cpp quantized types. Hugging Face **`kernels`**: compute kernels loaded from the Hub, built for each PyTorch/CUDA configuration. Transformers' optimization overview: compilation, attention backends, kernels, quantization, caching, parallelism, continuous batching | ✅ The pattern is well established. ❌ Kinesis has none of these abstractions yet (T13) |
| Raspberry Pi VideoCore Vulkan compute for vision (ARCH §14.D; Inv. 12) | Community benchmarks on the Pi 5 (VideoCore VII) report useful ncnn Vulkan acceleration. On the Pi 4 (VideoCore VI), ncnn users reported Vulkan running slower than the CPU (ncnn issue #2435) | ⚠️ Plausible on the Pi 5; don't assume it on the Pi 4 |

<a id="s2-4"></a>
### §2.4 Learning & optimizer strategies vs evidence

| Strategy (where) | Evidence | Verdict & implication |
|---|---|---|
| "Why AdamW is Outdated"; Lion "outperforms AdamW" (Inv. 15 §5) | Zhao et al. (2024): SGD aside, Adam, Adafactor, Lion and Sophia "all perform comparably" for language models, both at their optimum and across hyperparameters. Dettmers et al.: 8-bit optimizers "maintain 32-bit performance with a small fraction of the memory footprint" | ⚠️ Memory is a stronger argument than accuracy, and 8-bit optimizer states address it without switching algorithms |
| Muon for 2D weights; "Tensor-Muon" for N-D weights (Inv. 15/16; Fault 4) | Keller Jordan: a quintic Newton-Schulz iteration with (3.4445, −4.7750, 2.0315) after Frobenius normalization gives *approximate* orthogonalization. It targets 2D hidden-layer parameters, with AdamW still used for embeddings and the classifier head. Moonshot, "Muon is Scalable for LLM Training": ~2× compute efficiency vs AdamW at compute-optimal settings, after adding weight decay and adjusting per-parameter update scale (Moonlight, 3B/16B MoE trained on 5.7T tokens) | ✅ A strong strategy. ⚠️ Muon complements AdamW rather than replacing it, and scaling it needs the two published adjustments |
| Shampoo/SOAP suffer an "O(d³) Eigendecomposition Latency Cliff [that] stalls GPU streaming" (Fault 4) | PyTorch's **Distributed Shampoo** keeps per-step wall-clock overhead to at most 10% vs diagonal methods, and **won the AlgoPerf 2024 external-tuning track** (28% faster training than the baseline). **SOAP** runs Adam in Shampoo's eigenbasis, refreshing the eigenbasis at a set frequency, and reports >40% fewer iterations and >35% less wall-clock time than AdamW on 360M/660M language models | ❌ as a blocker: amortized, infrequent preconditioner updates are the SOTA fix. Muon's quintic can't replace the inverse-root computations Shampoo needs, because that's a different iteration |
| Reversible networks remove the activation-memory wall; "IEEE 754 Non-Associative Drift destroys bit-exact reversibility" → bit-exact fixed-point coupling (Fault 3; Inv. 15/16) | **Rev-ViT** (CVPR 2022): up to 15.5× lower memory at identical complexity and accuracy, and up to 3.9× higher throughput on deeper models. **RevNet** (NeurIPS 2017): float32 reconstruction error is noticeable but doesn't significantly affect final performance; the authors already propose fixed-point activations if needed. **Behrmann et al. (2021)**: inverses can explode and yield incorrect gradients in memory-saving backprop, mitigated by a regularizer or bi-Lipschitz building blocks. **Chen et al. (2016)**, checkpointing: O(√n) memory for one extra forward pass (48 GB → 7 GB at +30% time on a 1,000-layer network) | ✅ Reversibility is a validated memory strategy. ⚠️ But "exponential drift → divergence" depends on Lipschitz constants and isn't universal. Fixed-point coupling is prior art with several requirements: straight-through gradients through rounding, bit-identical recomputation (T8), and no wrap-around (modular arithmetic is invertible, but an overflowed activation is numerically meaningless). Checkpointing remains the default baseline to beat |
| DFA/GrAPE "does NOT exceed the computational value of standard backpropagation" (Fault 2; Inv. 18) | **GrAPE** is *Scaling Direct Feedback Learning with Jacobian Alignment Guarantees* (Caillon, Fagnou, Delattre, Allauzen), published at ICLR **2026**, not 2024. Per its abstract, it "performs infrequent BP anchor steps on a single mini-batch", not a 9:1 schedule. A secondary paper-notes summary reports calibration of one mini-batch every T epochs (~0.5% of BP's backward passes) and results well below "98.5-99.5% of BP": VGG-16/CIFAR-100 56.93% vs 70.33%, ResNet-20 64.82% vs 68.72%, ResNet-56 66.92% vs 71.42%, Transformer/WikiText-103 33.1 vs 29.8 perplexity. Verify these against the paper PDF. **Launay et al. (2020)**: DFA does train Transformers, with a gap to BP | ⚠️ The verdict (don't replace backprop for dense training) is reasonable. The supporting numbers aren't supported by these sources: "98.5-99.5%", "1.35-1.55× more FLOPs", "<65% Tensor Core utilization", "ImageNet ≈18%" and "perplexity diverges". Backward cost also depends on output width (L6) |
| Forward gradients; activation perturbation (Fault 1; Inv. 15) | Baydin et al. (2022) introduced forward gradients. Fournier et al. (ICML 2023) show random guesses scale poorly in high dimension, and that guesses derived from local losses help | ✅ The variance concern is valid. ⚠️ Fix the constant (L4). "MeZO-style low-rank subspace projection" mislabels MeZO, which uses full-dimensional seeded perturbations |
| FoMoH-style second-order steps with `HyperDual<float, 2>` (Inv. 16 §2) | Cobb, Baydin, Pearlmutter & Jha (2024) use **hyper-dual numbers** to evaluate directional derivatives and their second-order terms together | ✅ as a method. ❌ Kinesis' `HyperDual` has no ε₁ε₂ component, and the `FloatingPoint` concept blocks `Dual<Dual<T>>` nesting. Also, h_v can be negative in non-convex regions, so the damping λ must exceed \|min h_v\| |

<a id="s2-5"></a>
### §2.5 Robotics, autodiff & quant strategies vs evidence

| Strategy (where) | Evidence | Verdict & implication |
|---|---|---|
| "The ‘New Math’ Fusion": propagate dual numbers through Featherstone's ABA to get "exact state Jacobians simultaneously in a single O(N) linear pass" / "in 0 microseconds" (ARCH §12.A.3, §16; Inv. 6; PRODUCTION_SPEC Fault 5) | **Carpentier & Mansard (RSS 2018)**: analytical derivatives of rigid-body dynamics run "about 40 times faster" than finite differences on a humanoid. On HyQ they take 5 µs vs 7 µs (inverse dynamics) and 8 µs vs 20 µs (direct dynamics), against the fastest code-generated implementation. **Pinocchio** ships analytical derivatives of RNEA/ABA and supports CppAD, CasADi and CppADCodeGen. **Drake** runs `AutoDiffXd` (dual numbers) through MultibodyPlant; its benchmarking issues report roughly 100× the cost of `double`. arXiv 2105.05102 improves analytical derivatives by up to 1.4× over Pinocchio, and arXiv 2302.06001 computes second-order derivatives ~3× faster than AD | ❌ Not novel, and the complexity claim is wrong. ∂q̈/∂q has n² entries, so it can't even be *written* in O(n), and forward mode with 2n tangents costs O(n) per scalar operation on top of ABA's O(n). "0 µs" is timer resolution. ✅ Dual-number AD over Featherstone is still a valid, established technique; benchmark it against analytical derivatives and code generation, especially for the ESP32 target |
| Reverse-mode "Monotonic Linear Arena Tape" (Inv. 14 Decision 2) | **Stan Math**: C++ reverse-mode AD whose memory comes from an arena that stays valid until `recover_memory()`. **Enzyme** (NeurIPS 2020): AD on optimized LLVM IR, with a 4.5× geometric-mean speedup over AD applied before optimization | ✅ Arena tapes are the established design. Enzyme is a SOTA alternative that can differentiate Kinesis' own optimized C++ kernels |
| Lie-group state (SO(3)/SE(3)) and spatial Jacobians (ARCH §8) | Solà, Deray & Atchuthan, "A micro Lie theory for state estimation in robotics": Jacobian formulas plus the `manif` C++ library | ✅ A good reference implementation. Lie-group parameterizations remove *representation* singularities (gimbal lock), not *kinematic* singularities, so "singularity-free" needs that qualifier |
| Dilated causal TCN for tick streams (Inv. 14 Decision 4) | Bai, Kolter & Koltun (2018) establish TCNs as a strong generic sequence model | ✅ as a technique. ❌ The receptive-field arithmetic is wrong (L2) |
| Zero-copy circular tick buffer, where "the convolution streams directly across the circular view" (Inv. 14 Decision 4) | **LMAX Disruptor**: a pre-allocated ring buffer with cache-line padding against false sharing, reaching ~25M messages/s at <50 ns, indexed by sequence numbers masked to a power-of-two size. The **magic ring buffer** (fgiesen 2012) maps the same physical memory twice so wrap-around becomes contiguous; practitioners note it is "mostly not worth it" for small buffers | ⚠️ A wrap-around window can't be expressed as `offset + i·stride`. It needs masking (non-affine, T9), two-segment views, or a double mapping of shared memory, which `VirtualMemory` can't create because it only supports anonymous private mappings (T10) |
| Online RLS "evaluates in < 200 ns per tick" (Inv. 14 Decision 5) | No source | ❓ Cost is O(d²) per update with d unspecified, and it was never measured. Covariance-form RLS is also numerically fragile over long runs (square-root and QR variants exist) |

---

<a id="s3"></a>
## §3 Architecture coherence: do the decisions work together?

Each item names the decisions involved, why they conflict, the drawback, what state-of-the-art systems do, and a recommendation. **None of these have been changed in code.**

<a id="t1"></a>
### T1: "Zero allocation in the hot loop" is a convention, not a mechanism
- **Decisions involved:**
  - zero-allocation hot loop (README §1, ARCH §1);
  - replacing `malloc` with OS pages (Inv. 4);
  - GotoBLAS packing, which needs workspace;
  - the planned reverse-mode tape;
  - the "Option C+" allocated path (Inv. 16).
- **Why they conflict:** nothing enforces the guarantee.
  - Kernels have no workspace parameter, so `gemm()` heap-allocates its packing buffer on every call (`src/compute/gemm.cpp:144-151, 239`), ✅ *fixed 2026-09-18: the buffer is now thread-local, so `gemm()` no longer allocates. Kernels still take no workspace parameter, so the recommendation below stands*.
  - The default allocator makes a system call per tensor (`include/kinesis/core/allocator.hpp:73`), ✅ *fixed 2026-09-18: the default is now an aligned heap allocator*.
  - A full arena throws mid-tick (`src/core/arena_allocator.cpp:75-77`).
  - Nothing computes memory needs ahead of time.
- **Drawback:** the guarantee holds only if every caller avoids every allocating path by convention. One missed path reintroduces allocator latency, or an exception, into a 1 kHz loop.
- **SOTA:**
  - ggml's graph allocator reserves for a worst-case graph, then allocates all tensors before compute;
  - TensorFlow Lite Micro's arena planner;
  - oneDNN's user-managed scratchpad;
  - KleidiAI micro-kernels, which never allocate;
  - on GPU, PyTorch's caching allocator plus CUDA Graphs with private pools.
- **Recommendation:** split *plan* from *execute*.
  - A prepare step computes the worst-case workspace for each operation.
  - Kernels take an explicit workspace or scratch argument.
  - Arenas are sized and checked at initialization, so steady state can't allocate or throw.
- **Irony:** the GGML document criticizes ggml's "static buffer with manual two-pass offset scanning", but that mechanism is exactly the SOTA way to reach this goal.

<a id="t2"></a>
### T2: Two arenas aren't enough lifetimes, and views have no lifetime safety
- **Decisions involved:**
  - the dual-arena model (permanent + scratch);
  - non-owning `TensorView` everywhere;
  - `ScopedMarker` rewind;
  - reverse-mode training plans.
- **Why they conflict:** training needs at least three lifetimes, and the view model can't detect when memory disappears.
  - The three lifetimes are **persistent** (weights, optimizer state), **step** (activations saved for backward, the tape) and **transient** (per-kernel workspace).
  - Activations needed by the backward pass can't live in a per-tick scratch arena that gets rewound during the forward pass.
  - Rewinding or resetting silently invalidates every view into that memory.
  - `Tensor`'s implicit conversions aren't ref-qualified, so `TensorView<float> v = Tensor<float>(Shape{4});` dangles (`include/kinesis/core/tensor.hpp:101-102`).
- **Drawback:** "Rust-inspired" allocation without Rust's borrow checker loses the property that makes arenas safe. Errors show up as silent corruption.
- **SOTA:**
  - TensorFlow Lite Micro separates persistent (tail) from non-persistent (head) regions;
  - Stan Math documents that its arena stays valid until `recover_memory()`;
  - AddressSanitizer can poison memory on rewind;
  - debug builds can store generation counters in views.
- **Recommendation:**
  - name the lifetime classes in the architecture;
  - add debug-mode epochs or poisoning;
  - ref-qualify the conversions (lvalue-only, deleted for rvalues).

<a id="t3"></a>
### T3: Zero-copy strided views vs "contiguous-only hot loops" vs kernel stride handling
- **Decisions involved:**
  - zero-copy transpose and broadcast via strides (ARCH §2);
  - "Slices in hot compute loops must be strictly contiguous (stride 1) or packed into contiguous arena memory prior to execution" (ARCH §11.1);
  - the packed GEMM kernel.
- **Why they conflict:** the core feature produces exactly the views the rule forbids, and the kernel doesn't enforce the rule.
  - `gemm_packed` silently assumes unit column stride for A and C (`src/compute/gemm.cpp:169-170`).
  - As a result, `gramian(A)`, which passes `A.t()`, returns wrong results whenever A has ≥ 128 columns.
  - Copying to contiguous memory needs workspace (T1), so the zero-copy benefit disappears at the compute boundary.
- **SOTA:**
  - BLAS keeps transposes zero-copy through operation flags and leading dimensions, and GotoBLAS packing reads through any stride;
  - PyTorch makes the copy explicit with `.contiguous()`;
  - `std::mdspan` offers `layout_stride`.
- **Recommendation:** define the kernel-boundary contract.
  - Normalize transposes and strides at entry.
  - Let packing read through strides.
  - Dispatch unsupported layouts to a stride-aware path.
  - Document the preconditions.

<a id="t4"></a>
### T4: The const model blocks "permanent, read-only weights"
- **Decisions involved:**
  - deep-const views plus a separate `TensorView<const T>` type;
  - kernels that take `const TensorView<float>&`;
  - permanent or memory-mapped weights (ARCH §9.B; GGML doc §5).
- **Why they conflict:** there's no conversion between `TensorView<T>` and `TensorView<const T>`.
  - A `const Tensor<float>` or read-only mapped view can't be passed to any kernel.
  - For the same reason, `Tensor::clone() const` is ill-formed and fails to compile the first time it's used (`include/kinesis/core/tensor.hpp:77-80`).
  - Outputs are passed by value (104-byte copies) because a const view can't write.
- **SOTA:** `std::span`/`std::mdspan` use shallow constness with an implicit `span<T>` → `span<const T>` conversion.
- **Recommendation:**
  - make views shallow-const;
  - add a converting constructor;
  - have kernels take `TensorView<const float>` for inputs.
- ✅ **Resolved 2026-09-18** along the last two lines: a constrained template converting constructor, and `const TensorView<const float>&` inputs throughout. `clone()` compiles. Views remain deep-const and outputs are still passed by value, the first recommendation is still open.

<a id="t5"></a>
### T5: Descriptor compaction vs actual layout, ABI, extents and dtype
- **Decisions involved:**
  - `uint32_t` dims for a "64-byte `TensorView`" (ARCH §15.A);
  - 64-bit strides;
  - rank ≤ 6;
  - a planned compile-time `Shape<MaxRank>`;
  - zero-copy GGUF weights.
- **Why they conflict:**
  - **Size.** The view is **104 bytes** (L7): `Strides` repeats the rank as a `size_t`, plus padding and a 64-bit offset. The duplicated rank is also how the committed `broadcast_to` produced a rank-0 `Strides` for a rank-N view; the working tree fixes that.
  - **Why the size barely matters.** Most kernels take views by `const&`, and ABIs pass such aggregates in memory anyway: Windows x64 passes any aggregate that isn't 1, 2, 4 or 8 bytes by pointer to a caller-made copy.
  - **Extents.** `uint32_t` dims cap any single extent at 4.29 × 10⁹ and truncate silently in release builds, while GGUF uses `uint64` dimensions and offsets, ✅ *fixed 2026-09-18: they now throw instead of truncating; the cap is unchanged pending this decision*. The PRODUCTION_SPEC plan to also make strides and the offset 32-bit would overflow once a trailing product exceeds 2³¹.
  - **Dtype.** There's no runtime dtype and no quantized block types, so GGUF tensors can't be viewed as stored.
  - **Rank.** "6D robotics Lie groups" (GGML doc Flaw 5) conflates a 6-element twist, which is a rank-1 tensor, with a rank-6 tensor. The planned `Shape<MaxRank>` would also split every signature into rank-specific types.
- **SOTA:**
  - PyTorch's `SizesAndStrides` stores five `int64` sizes and strides inline, with out-of-line overflow (dynamic rank, no heap allocation in the common case);
  - ggml uses `int64` element counts and byte strides, plus `ggml_type` block sizes.
- **Recommendation:**
  - favour correctness: 64-bit extents, or release-mode checks;
  - keep a single rank field;
  - add a dtype/block model before quantized or GPU work;
  - measure before compacting further.

<a id="t6"></a>
### T6: Dual numbers as tensor elements vs SIMD layout and float-only kernels
- **Decisions involved:**
  - "any tensor can instantly hold dual numbers (`TensorView<Dual<float>>`)" (Inv. 16);
  - the SIMD and cache-layout principles;
  - float-only kernels.
- **Why they conflict:**
  - Every kernel takes `TensorView<float>`, and the `Numeric` concept excludes `Dual`, so dual tensors can be stored but not computed.
  - Array-of-structs duals interleave value and tangent. That halves the effective SIMD width, blocks reuse of the float kernels, and costs (N+1)× memory for `HyperDual<T,N>`.
  - A tensor JVP through GEMM is simply two GEMMs (Ċ = ȦB + AḂ) over separate tangent tensors.
- **SOTA:**
  - JAX `jvp` and PyTorch forward-mode AD propagate tangents as separate tensors (struct-of-arrays);
  - array-of-structs scalars (Ceres `Jet`, Drake `AutoDiffXd`) are used for small scalar programs.
- **Recommendation:**
  - keep array-of-structs duals for "Regime B" scalar problems (≤ 128 variables);
  - for tensor code, specify struct-of-arrays tangent tensors with per-kernel JVP rules.

<a id="t7"></a>
### T7: Differentiable Featherstone claims vs complexity, prior art and the embedded budget
See §2.5. Three follow-ups:
- Replace the "new math, O(N), 0 µs" framing with "dual-number AD over Featherstone recursions", an established technique.
- Benchmark it against Pinocchio's analytical derivatives and against code generation.
- Budget it against ESP32 reality (§2.2).

<a id="t8"></a>
### T8: Determinism promises vs ISA-specific kernels, FP contraction and NaN semantics
- **Decisions involved:**
  - "Deterministic Fixed-Tile Binary Tree Reduction: Bit-exact identical logits regardless of thread scheduling" (GGML doc; PRODUCTION_SPEC);
  - FMA for "infinite intermediate precision";
  - per-ISA SIMD kernels;
  - bit-exact fixed-point reversible coupling.
- **Why they conflict:** different targets already compute different bits.
  - **FMA.** The AVX2 path uses fused `_mm256_fmadd_ps`, while the NEON path uses unfused `vmlaq_f32`.
  - **Contraction.** Scalar tails depend on compiler defaults. Clang defaults to `-ffp-contract=on`, which fuses within a statement. GCC historically defaulted to `fast` for C++ (fusing across statements), and recent GCC changes move toward `off`/`on` depending on dialect, so it's version-dependent. MSVC `/fp:precise` doesn't contract unless `/fp:contract` is given.
  - **NaN.** `relu(NaN)` returns 0 on AVX2 (`maxps` returns its second operand) but NaN with NEON's `vmaxq_f32`, ✅ *fixed 2026-09-18: NEON uses compare-and-select, so both return 0. FMA and contraction differences remain*.
  - **Dependency.** Fixed-point reversible coupling needs bit-identical recomputation, so it inherits all of the above.
- **SOTA:**
  - PyTorch documents no reproducibility guarantee across releases, platforms, or CPU vs GPU;
  - oneMKL's Conditional Numerical Reproducibility needs a fixed code path and thread count;
  - batch-invariant kernels cost throughput (Thinking Machines);
  - Simon Boehm shows that inlining alone can change FMA fusion.
- **Recommendation:** do these before building on bit-exact assumptions:
  - define the determinism scope (same binary + ISA + thread count);
  - set an explicit contraction policy;
  - decide the NaN semantics;
  - accept the performance cost.

<a id="t9"></a>
### T9: "Affine indexing only / no dependent reads" vs the workloads on the roadmap
- **The rule:** "All tensor index calculations must be pure affine functions ($\text{offset} + i \cdot \text{stride}$)" (ARCH §11.3). It derives from GPU *texture* advice about dependent texture reads in shaders.
- **Why it conflicts:** the roadmap's workloads need indirection or non-affine indexing:
  - embeddings (`W[token]`);
  - paged KV block tables (the planned Paged-Slab attention);
  - MoE routing;
  - gather/scatter;
  - indirect convolution (the CPU SOTA, §2.2);
  - reverse-tape closures;
  - power-of-two ring buffers.
- **Recommendation:** downgrade the rule to a guideline: hoist indirection out of inner loops, batch gathers, and keep the innermost loop affine where possible.

<a id="t10"></a>
### T10: "Zero-copy circular view" vs the view model and the virtual-memory API
See §2.5. Choose and document one representation:
- masked indices;
- two-segment views;
- a mirrored mapping, which requires shared-memory mappings on Windows/POSIX;
- copy-on-wrap for small windows.

<a id="t11"></a>
### T11: Detection → configuration → kernels isn't connected
- **Decisions involved:**
  - runtime hardware detection (ARCH §14.A);
  - adaptive alignment (Inv. 3);
  - dynamic tile sizing from detected caches (Inv. 5);
  - compile-time SIMD selection.
- **Why they conflict:** nothing carries detection results to the kernels.
  - Detection results are never consumed, and the cache sizes in `HardwareInfo` are constants. *(2026-09-18: the sizes are now measured, 48 KB / 1280 KB / 24 MB on the development machine, against constants of 32 KB / 512 KB / 16 MB, but still consumed by nothing, which is the point of this section.)*
  - `gemm_packed` hard-codes a 49,152-byte L1.
  - Alignment is always 64.
  - The ISA choice is baked in at compile time: `-march=native` in CMake, and nothing at all in `build.ps1`. *(2026-09-18: all three build entry points now pass it explicitly; it is still compile-time.)*
  - Tests assert AVX2 on any x86-64, ✅ *fixed 2026-09-18*.
- **SOTA:**
  - ggml's CPU backend variants, scored at runtime;
  - OpenBLAS `DYNAMIC_ARCH`;
  - Highway;
  - BLIS's analytical block sizes, derived from cache parameters.
- **Recommendation:** build one `KernelContext` (ISA level, block sizes, thread pool, workspace) from detection once, and pass it to kernels.

<a id="t12"></a>
### T12: The verification strategy can't validate the decisions it justifies
- **Decisions involved:**
  - "Algorithmic efficiency is measured via simulated hardware hit rates in the `MemorySandbox` rather than noisy wall-clock timers" (ARCH §11.6);
  - the Pragmatism Principle ("mathematically evident" benefit);
  - the published benchmark tables.
- **Why they conflict:**
  - **The sandbox models the wrong hardware.** It's fed hand-written addresses, not traces of the real kernels. It pairs CPU L1 geometry with GPU latency constants, has no TLB, prefetcher or cycle accounting, models banks with compute-capability-2.0 geometry, and its swizzle demo is tautological (L5), ✅ *fixed 2026-09-18: it now reproduces gcd(s, 32)*.
  - **The benchmarks don't measure what they claim.** They contain undefined behaviour and constant folding, include page faults, record no compiler or CPU, and have no SOTA baselines (§6, defect 14), ✅ *fixed 2026-09-18: the harness is rewritten with a `std::pmr` baseline. There is still no BLAS or mimalloc baseline, and the published tables have not been regenerated*.
- **SOTA:**
  - hardware counters: Linux `perf`, Intel VTune, AMD uProf (cache misses, dTLB misses);
  - trace-driven simulators running the real binary: Cachegrind, DynamoRIO drcachesim;
  - split-lock detection;
  - benchmarks with pinning, repetitions and variance, against BLIS/OpenBLAS/Accelerate, mimalloc and `std::pmr` baselines.
- **Recommendation:** keep the sandbox and visualizer as teaching tools, and make decisions from measured counters and reproducible benchmarks.

<a id="t13"></a>
### T13: The CPU+GPU ambition vs the current abstractions
- **Validated premise:** GPU fast-memory hierarchies and IO-aware tiling are real and SOTA (§2.3).
- **Missing prerequisites.** Each of these exists in SOTA systems:
  1. **Memory spaces & buffer types:** device pointers with per-buffer-type alignment and maximum size (ggml buffer types; mdspan accessors). Today `TensorView` holds a raw host `T*` and `Allocator` is host-only.
  2. **Streams, async execution and events:** ggml-backend, MLX streams, CUDA streams.
  3. **Device allocators:** caching or stream-ordered pools, plus static pools that work with CUDA Graphs (PyTorch; CUDA runtime).
  4. **A scheduler/dispatcher** that assigns operations to devices and copies tensors between them (ggml's `ggml_backend_sched`; the PyTorch dispatcher).
  5. **A kernel-sourcing strategy:**
     - vendor libraries (cuBLAS/cuBLASLt/cuDNN);
     - generated kernels (Triton);
     - distributed kernels (Hugging Face `kernels`);
     - Vulkan/Metal compute for Raspberry Pi and Apple.
  6. **A dtype system:** FP16/BF16/FP8 and quantized blocks, with shape padding for Tensor Cores (NVIDIA Matrix Multiplication guide).
  7. **A host-transfer policy:** pinned memory, `non_blocking` copies, and offloading (as in Hugging Face's offloaded KV caches). Unified-memory platforms (Apple, Raspberry Pi) change this model (MLX).
- **Recommendation:** decide these before more CPU-only kernels harden the public API. Otherwise most current signatures, such as `gemm(const TensorView<float>&, …)`, will need to change later.

<a id="t14"></a>
### T14: Error-handling policy vs HFT/embedded targets
- **Current state:** four different styles coexist.
  - exceptions (`broadcast_to`, `reshape`, allocators);
  - `assert`-only checks that vanish in release builds (`slice`, `Shape` bounds, slab size);
  - `bool` returns (the Cramer solvers);
  - `noexcept` functions that call helpers which aren't.
- **Why it conflicts:** exceptions are often disabled in HFT and ESP32 builds, while `assert`-only validation lets release builds corrupt memory (§6, defect 4).
- **SOTA:** configurable hardening levels (e.g. libc++ hardening modes), `std::expected` (C++23), status codes.
- **Recommendation:** one documented policy:
  - cheap boundary checks in all builds;
  - asserts for internal invariants;
  - an exception-free build option.

<a id="t15"></a>
### T15: Modularity, build structure and concurrency model
- **Why it conflicts:**
  - **Modularity.** The docs say "Core Kinesis … should remain a pure, general tensor math engine", yet `kinesis.hpp` pulls `<windows.h>`, `<iostream>`, the memory sandbox and hardware printing into every translation unit. Teaching and simulation tools also live in `core/`.
  - **Build.** There's one static library and three build entry points that have drifted: `build.bat` lacks two test files, and `build.ps1` passes no ISA flags.
  - **Concurrency.** There's no thread model: allocators aren't thread-safe, and `Threads` is linked but unused. The NUMA, paged-KV and multi-request plans all assume one.
  - **False sharing.** Concerns like Apple's 128-byte cache lines are conflated with allocation alignment.
- **SOTA:**
  - ggml split into `ggml-base` plus backend libraries (PR #10256);
  - per-thread allocator caches;
  - `std::hardware_destructive_interference_size` padding for shared mutable state.

<a id="t16"></a>
### T16: Learning-stack dependencies
The "two-regime" plan (PRODUCTION_SPEC Fault 5) is sound. But Regime A (exact reverse mode + bit-exact reversible coupling + Muon) carries dependencies the docs don't state:
- invertible-architecture constraints;
- straight-through gradients through rounding;
- bit-identical recomputation (T8);
- mixed-precision scaling;
- AdamW for non-2D parameters;
- different Newton-Schulz iterations for orthogonalization (Muon) and for inverse roots (Shampoo).

Separately, the cubic Newton-Schulz formula in Inv. 15/16 differs from the quintic in Fault 4. See §2.4.

<a id="t17"></a>
### T17: Hardware-protection claims
- **Declared but unused:** `HardwareSafetyPolicy` is never consumed, yet `print_system_report` prints "Slew-Rate Vdroop Protection: [ACTIVE]" (`include/kinesis/core/hardware_info.hpp:256`), ✅ *fixed 2026-09-18: it now prints `[NOT IMPLEMENTED]`*.
- **Already handled in hardware:** the CPU itself stages voltage and frequency transitions for wide SIMD (§2.2), and hardware and firmware enforce thermal and power limits.
- **Mixed CPU/GPU claim:** "GotoBLAS micro-panel packing … slashing off-chip VRAM bus transactions by up to 80%" (ARCH §14.B.3) mixes CPU-side packing with GPU memory.
- **Recommendation:** label these as hypotheses until they're implemented and measured.

---

<a id="s4"></a>
## §4 SOTA optimization map (CPU & GPU)

| Area | Optimization | Kinesis today | SOTA references |
|---|---|---|---|
| CPU | Packed GEMM with a register-tiled micro-kernel and per-level blocking | ⚠️ Partial | BLIS; salykova; Yotov 2007 |
| CPU | Runtime ISA dispatch | ❌ | ggml CPU variants; OpenBLAS `DYNAMIC_ARCH`; Highway |
| CPU | Multithreading with caller-controlled partitioning | ❌ | BLIS; KleidiAI (no internal threading) |
| CPU | Quantized INT8/INT4 micro-kernels and weight pre-packing | ❌ | KleidiAI; XNNPACK; llama.cpp |
| CPU | Vendor BLAS/accelerator fallback (MKL, OpenBLAS, Accelerate/AMX) | ❌ | Candle; llama.cpp; Accelerate |
| CPU | Memory planning (offsets known before execution) | ❌ | ggml `gallocr`; TensorFlow Lite Micro |
| CPU | Prefaulting, page locking, huge pages | ❌ | Rigtorp; Red Hat Enterprise Linux for Real Time |
| CPU | NUMA placement / weight mirroring | ❌ | llama.cpp `--numa`, `GGML_NUMA_MIRROR` |
| MCU | Vendor NN kernels + arena planner | ❌ | ESP-NN; TensorFlow Lite Micro |
| GPU | IO-aware fused kernels (attention, elementwise, norms) | ❌ | FlashAttention; Triton; Hugging Face `kernels` |
| GPU | Caching / stream-ordered allocators; CUDA Graphs | ❌ | PyTorch CUDA semantics; CUDA Stream-Ordered Memory Allocator |
| GPU | Mixed precision with Tensor-Core-friendly shapes; FP8 | ❌ | NVIDIA Matrix Multiplication guide; FP8 Formats |
| GPU | Paged KV cache + continuous batching | ❌ | vLLM; Hugging Face Transformers |
| GPU | Backend scheduler, buffer types, async events | ❌ | ggml-backend; MLX |
| All | Deterministic mode with explicit scope | ❌ | PyTorch reproducibility docs; oneMKL CNR; Thinking Machines |

---

<a id="s5"></a>
## §5 Implementation-status matrix (claims in the docs vs the code)

*Updated 2026-09-18 to the code after the remediation pass. Where a status changed, the old one is kept in italics.*

| Claim | Status | Evidence |
|---|---|---|
| Zero heap allocation in hot loops (README §1, ARCH §1) | ⚠️ | Kernels no longer allocate: `gemm()` packs into a thread-local block. The guarantee is still a convention, a `Tensor` built without an arena uses the heap, a full arena throws, and pages are not prefaulted (T1). *Was: an allocation per `gemm()` call and an mmap per tensor.* |
| Permanent parameter arena | ❌ | No such facility; the README quick-start allocates weights from the per-tick arena |
| `ArenaAllocator` O(1) bump + `ScopedMarker` | ✅ | `include/kinesis/core/arena_allocator.hpp`, `src/core/arena_allocator.cpp` |
| `SlabAllocator` O(1) | ✅ | `include/kinesis/core/slab_allocator.hpp` |
| Slab "free-list validation" (ARCH §12.A.2) | ⚠️ | Size, alignment, ownership and block boundary are checked in every build; double-free is still undetected. *Was ❌: assert-only.* |
| Chained commit-on-demand virtual arena (Inv. 4) | ❌ | `VirtualMemory::reserve/commit` exist, but no allocator uses them |
| Page locking / prefaulting / huge pages (ARCH §14.C) | ❌ | Not implemented; `lock_pages_in_physical_ram` is unused |
| NUMA / core pinning (GGML doc Flaw 3) | ❌ | Not implemented |
| `KINESIS_EMBEDDED` zero-virtual profile (ARCH §14.D) | ❌ | Only selects an enum value (`include/kinesis/core/hardware_info.hpp:89`) |
| Adaptive 32/64/128-byte alignment (Inv. 3) | ❌ | Constants only; always 64 (`include/kinesis/core/types.hpp:12-21`) |
| Zero-copy slice / transpose / broadcast | ✅ | `include/kinesis/core/tensor_view.hpp`. Structural operations (`slice`, `transpose`, `t`, `broadcast_to`, `copy_from`, `make_identity`) validate in every build; per-element `operator()` stays assert-only by design, like `std::vector::operator[]`. *Was: all assert-only.* |
| 64-byte `TensorView` (ARCH §15.A) | ❌ | 104 bytes (L7) |
| Branch-free O(1) `Shape::operator==` (ARCH §15.B) | ✅ | O(1) over the fixed six-slot array, no undefined behaviour. `&&` may compile to a branch, which is immaterial at this size. *Was ⚠️: implemented via UB.* |
| Compile-time `Shape<Rank>` / `TensorView<T, Rank>` (GGML doc) | ❌ | Runtime rank ≤ 6 |
| "Multi-threaded" blocked GEMM (`gemm.hpp` comment) | ❌ | Single-threaded |
| GotoBLAS packing (ARCH §9.A) | ⚠️ | B only, 1×8 kernel, 64³ blocks; `pack_A_panel` unused. Defects 1-2 fixed. |
| Dynamic tile sizing from detected caches (ARCH §7) | ❌ | Still a compile-time `compute_dynamic_gemm_tile(49152, …)` (`src/compute/gemm.cpp:161`). The caches are now measured, but nothing consumes them (T11). |
| Offline weight pre-packing (ARCH §9.B) | ❌ | Packing happens on every call |
| AVX-512 kernels (README §3) | ❌ | AVX2 / NEON / scalar only |
| Runtime ISA dispatch | ❌ | Compile-time `#if` (`include/kinesis/compute/simd.hpp:6-18`) |
| CPU feature detection (ARCH §14.A) | ✅ | Max-leaf, OSXSAVE and XCR0 checks; cache geometry from CPUID leaf 4 / 0x8000001D (`hardware_info.hpp`). The results feed nothing yet (T11). *Was ⚠️: no OS-state check, caches hard-coded.* |
| GPU vendor/capability detection (ARCH §14.A) | ❌ | Not implemented, and now says so: `Unknown` with `detected == false` (`hardware_info.hpp:252`). *Was: fabricated results.* |
| Silicon protection: warmup, thermal governor, quota fences (ARCH §14.B) | ❌ | `HardwareSafetyPolicy` is never consumed; the report now prints `[NOT IMPLEMENTED]` rather than `[ACTIVE]` |
| Sandbox cycle accounting & coalescing metrics (ARCH §13; Inv. 7) | ❌ | Only hit/miss, line-split, page-split and bank counters exist |
| XOR swizzle demo (visualizer) | ✅ | Models the column swizzle against the stride and reproduces gcd(s, 32) (`docs/memory_sandbox_visualizer.html`, `bankOfLane`). *Was ❌: ignored the stride.* |
| `Dual<T>` forward-mode AD | ✅ | `include/kinesis/autograd/dual.hpp`. The `sqrt` and `pow` edge cases are fixed; a few functions are still missing (§6.2). |
| `HyperDual<T,N>` second-order capability | ⚠️ | Renamed `Jet`, first-order by design. Second derivatives come from nesting, `Dual<Dual<T>>`. *Was ❌: no route to second order.* |
| `gramian`, Cramer 2×2/3×3 | ✅ | Defects 1 and 12 fixed. The docs still call the Cramer solvers branch-free and division-free, which they are not (§7). |
| Deterministic reduction; reverse tape; layers; optimizers; convolutions; quant ring buffer; robotics; ROS 2 adapters; Paged-Slab KV; GGUF parser; CudaStreamPool | ❌ planned | No code yet, though several documents describe them as existing |

---

<a id="s6"></a>
## §6 Code defects from `/code-review max`

The verdict column records the state at review time:
- **CONFIRMED:** shown by code or type analysis, or by math.
- **PLAUSIBLE:** the mechanism is real, but whether it triggers depends on the toolchain or hardware.

<a id="s6-0"></a>
### §6.0 Remediation status (2026-09-18)

All fifteen are fixed. Three were demonstrated against the pre-fix code rather than argued:

| Demonstration | Pre-fix | Post-fix |
|---|---|---|
| Defect 1: strip the stride guard from `gemm_packed`, run the suite | `Regression_GramianWithPackedDispatchMatchesReference` fails, G(0,0) = 1.75 against a reference 70.35 | passes |
| Defect 2: place C so its last element ends on the final byte of a committed page range, then run the original full-width tail load | **segmentation fault** (exit 139) | completes, `c[last]` correct |
| Defect 4 follow-up: restore the assert-only checks, build with `-DNDEBUG`, surround the buffers with canaries | `make_identity` on a 4×2 view overwrites offset 9 of an 8-element buffer; `copy_from` of a 3×3 strided source writes three floats past a 2×2 destination | both throw; canaries intact |

AddressSanitizer was the intended instrument for defect 2, but this toolchain (`zig c++` 0.16.0 /
clang 21.1 on Windows) ships no ASan runtime, linking `-fsanitize=address` fails with undefined
`__asan_init`: so the page-boundary construction above was used instead. `CMakeLists.txt` now
exposes `-DKINESIS_SANITIZE=ON` for toolchains that do have it.

| # | Fix | Pinned by |
|---|---|---|
| 1 | `gemm_packed` returns to the stride-aware `gemm_blocked` unless A and C have unit column stride | `Regression_GramianWithPackedDispatchMatchesReference`, `Regression_TransposedAndStridedOperandsMatchReference` |
| 2 | Partial micro-panels use a scalar loop over `count`, so neither the load nor the store leaves the row | `Regression_PackedTailDoesNotOverrunOutput` (M=128, K=70, N=131, adjacent canary tensor) |
| 3 | `operator==` compares `rank_` and `dims_` directly; the `uint64_t` type-pun is gone | compiles clean; covered by the existing shape tests |
| 4 | Shape and Strides validate rank, extent and overflow by throwing `std::invalid_argument`; `total_elements()` saturates. **Follow-up found during a later audit:** the same assert-only pattern survived in `transpose`, `t`, `make_identity` and `copy_from`, where it was a release-build memory-safety bug (see the demonstration above); all four now validate in every build | `Regression_ShapeAndViewValidationSurviveRelease`, `Regression_StructuralViewOpsValidateInRelease` (both also run under `-DNDEBUG`) |
| 5 | The AVX2 guard requires FMA (`__AVX2__ && (_MSC_VER \|\| __FMA__)`); all three build entry points pass `-march=native` and list the same ten test files | `HardwareInfo_CpuFeaturesAndCaches` asserts that `KINESIS_HAS_AVX2` implies runtime AVX2 + FMA |
| 6 | `detect_gpu()` reports `GpuVendor::Unknown` with `detected == false` instead of inventing a device; the report prints `[NOT IMPLEMENTED]` for the unenforced safety policy | `HardwareInfo_GpuDetectionIsHonestAboutWhatItKnows`, `HardwareInfo_SafetyPolicyIsADeclarationNotAMechanism` |
| 7 | The packed B panel lives in an `alignas(64)` thread-local block, so the hot path allocates nothing | existing GEMM equivalence tests |
| 8 | `HeapAllocator` is a real aligned-heap allocator (`operator new(std::align_val_t)`); `owns()` answers false rather than true when it cannot prove ownership; the arena names `VirtualMemoryAllocator` explicitly for its backing region | `HeapAllocator_AlignedAndDistinctFromPageAllocator`, `VirtualMemoryAllocator_PolymorphicInterface` |
| 9 | A constrained *template* converting constructor gives `TensorView<T>` → `TensorView<const T>` without suppressing the copy constructor; kernels take `const TensorView<const float>&` | `Regression_ConstViewsReachKernelsAndCloneCompiles` |
| 10 | Feature detection checks the max CPUID leaf, OSXSAVE, and the XCR0 state bits for YMM, ZMM and tile state; cache geometry is read from CPUID leaf 4 / 0x8000001D instead of being assumed | `HardwareInfo_CpuFeaturesAndCaches`. On the review machine this changed the reported L1D/L2/L3 from 32 KB / 512 KB / 16 MB to the true 48 KB / 1280 KB / 24 MB |
| 11 | `HyperDual` is renamed `Jet` (the Ceres name for exactly this type) with a deprecated alias; a structural `DualScalar` concept replaces `FloatingPoint`, so `Dual<Dual<T>>` gives second derivatives; `<cmath>` calls use the ADL two-step | `Dual_NestingYieldsSecondDerivatives` (f'' of sin(x)e^x matches to 1e-5) |
| 12 | The Cramer solvers compare `\|det\|` against a *relative* threshold scaled by the magnitude of the terms that formed it | `Regression_CramerThresholdIsScaleInvariant` |
| 13 | The wrong `= 11` comment is corrected to `= 14`; tail, transposed and strided coverage is added | `tests/test_regressions.cpp` |
| 14 | The benchmark reports min/median/p95 over repeated runs, initialises its buffers, defeats hoisting with compiler barriers and loop-dependent slice bounds, separates first-touch faults, prints the build configuration, and runs next to `std::malloc` and `std::pmr::monotonic_buffer_resource` baselines | it now runs; see the note below |
| 15 | `simulateWarp` models the column swizzle against the stride and reproduces gcd(s, 32); the guide's Experiment 3 formula is corrected | the `bench_latency` sweep prints measured degree against gcd(s, 32) for eight strides, with no disagreement |

**What the repaired benchmark actually says** (i7-12650H, `-O3 -DNDEBUG -march=native`, medians):
the arena is **6.34 ns/alloc against 7.29 ns for `std::pmr::monotonic_buffer_resource`**, a
1.15× edge, not an order of magnitude, while first-touch page faults cost **27.84 ns/alloc**,
over four times the steady-state figure. GEMM reaches **12-17 GFLOP/s**, and no baseline library
is linked to compare it against. These are the first numbers in this project that were produced
by a harness that measures what it claims to measure; the figures in `BENCHMARKS.md` were not.

<a id="s6-1"></a>
### §6.1 Findings (most severe first)

| # | Location | Defect | Trigger / consequence | Verdict |
|---|---|---|---|---|
| 1 | `src/compute/gemm.cpp:169-170` | `gemm_packed` indexes `a_row[k]` and `c_row + j` assuming unit column stride | Any transposed or strided A or C silently gives wrong products. `gramian()` (`include/kinesis/compute/gemm.hpp:70`) passes `a.t()`, so G ≠ AᵀA whenever A has ≥ 128 columns | CONFIRMED |
| 2 | `src/compute/gemm.cpp:178` (AVX2) and `:194-205` (NEON) | Tail micro-panels (count < Nr) use full-width loads and stores | NEON writes up to 3 floats past the end of C on the last row. AVX2 reads up to 7 floats past it, and faults if C ends on a page boundary (e.g. a VirtualAlloc-backed M = 1024, N = 129). `inf · 0` padding can write NaN into neighbouring elements | CONFIRMED (UB); crash PLAUSIBLE |
| 3 | `include/kinesis/core/shape.hpp:91-93` | `Shape::operator==` reads a `uint32_t` array through `const uint64_t*` | Strict-aliasing violation plus misaligned 8-byte loads. It relies on unused slots being zero, which out-of-rank writes in release builds break. `bool operator==(const Shape&) const = default;` gives equivalent branch-free code without UB | CONFIRMED (UB) |
| 4 | `include/kinesis/core/shape.hpp:28-43` | In release builds, dims ≥ 2³² truncate, rank > 6 overflows `dims_`, and `total_elements()` can overflow `int64_t` | `Shape{5'000'000'000}` stores 705,032,704; `Shape{1,2,3,4,5,6,7}` writes past the array | CONFIRMED |
| 5 | `include/kinesis/compute/simd.hpp:7`; `build.ps1:77, 91` | `defined(__AVX2__) \|\| defined(_MSC_VER)` enables AVX2 + FMA on any MSVC/clang-cl x64 build; `__AVX2__` doesn't imply FMA; `build.ps1` passes no ISA flags | Possible outcomes: illegal-instruction faults on non-AVX2 CPUs (MSVC); target-feature compile errors (clang-cl, or GCC with `-mavx2` but no `-mfma`); or silently scalar kernels with g++. The benchmarks can't be reproduced from the documented build | PLAUSIBLE |
| 6 | `include/kinesis/core/hardware_info.hpp:187-227, 256` | `detect_gpu()` returns hard-coded hardware, and the report prints protections as active | Every non-Apple x86-64 host "has" an 8 GB NVIDIA GPU with Tensor Cores; every Linux aarch64 host "has" a VideoCore VII | CONFIRMED |
| 7 | `src/compute/gemm.cpp:144-151, 239` | `gemm()` never passes scratch memory to `gemm_packed` | A heap allocation and a 16 KB zero-fill on every call with M, N ≥ 128 | CONFIRMED |
| 8 | `include/kinesis/core/allocator.hpp:46-73` | `HeapAllocator` aliases `VirtualMemoryAllocator`, and `owns()` returns true for any pointer | Every tensor costs a system call and at least one page, takes demand-zero page faults on first touch, and triggers TLB shootdowns on free in multithreaded processes | CONFIRMED |
| 9 | `include/kinesis/core/tensor.hpp:77-80, 96`; kernel signatures | No `TensorView<T>` ↔ `TensorView<const T>` conversion; kernels take `const TensorView<float>&` | `const Tensor<float> W; gemm(W.view(), …)` doesn't compile, and `Tensor::clone()` fails to compile on first use | CONFIRMED |
| 10 | `include/kinesis/core/hardware_info.hpp:139-148` | AVX/AVX2/AVX-512/FMA flags are read without OSXSAVE + XGETBV (XCR0) or max-leaf checks | Reports AVX2 as usable when the OS doesn't save YMM state (VMs, XSAVE disabled), contrary to Intel's detection procedure | CONFIRMED |
| 11 | `include/kinesis/autograd/dual.hpp:206-207` | `HyperDual<T,N>` is first-order only, and the `FloatingPoint` concept blocks `Dual<Dual<T>>` | No second derivatives are available for the documented FoMoH/DRFC plans | CONFIRMED (design) |
| 12 | `include/kinesis/compute/gemm.hpp:82, 99` | The Cramer solvers use an absolute `\|det\| < 1e-12f` threshold | They reject A = 1e-7·I (det = 1e-14) but accept near-singular large-scale systems. The docs also call them branch-free and division-free | CONFIRMED |
| 13 | `tests/test_gemm.cpp:99` (and comment at `:169`) | The packed test uses N = 144 (tail tile 16, a multiple of 8) with contiguous inputs only; the Gramian test is 3×2 | Defects 1-2 pass the suite. Also, the Cramer test comment says "= 11" while `b3[1] = 14` | CONFIRMED |
| 14 | `benchmarks/bench_latency.cpp:68, 78-82, 33-49` | The slicing benchmark reads an uninitialized buffer with constant slice bounds; the arena loop includes first-touch page faults; there are no baselines | "0.11 ns per subview" (under one CPU cycle) doesn't measure slicing, and no code produces the README's naive-GEMM baselines and speedups | CONFIRMED |
| 15 | `docs/memory_sandbox_visualizer.html:1190-1195, 907`; `docs/LEARNING_GUIDE.md` Experiment 3 | The swizzle simulation discards the stride, and the displayed formulas differ from the code | It always shows 1.0×, while the displayed formulas actually give 4× (L5) | CONFIRMED |

<a id="s6-2"></a>
### §6.2 Lower-severity notes

*Status added 2026-09-18. ✅ fixed · ➖ deliberately left, with the reason given.*
- **`dual.hpp` edge cases:**
  - ✅ `sqrt` at 0 gives inf/NaN for `Dual` (`:152-155`) but silently 0 for `HyperDual` (`:346-353`), which is inconsistent. **Fixed:** both now propagate the unbounded tangent; masking a singularity with 0 is the more dangerous default for an optimiser stepping through it.
  - ✅ `pow(x, 0)` at x = 0 yields a NaN derivative (`:158-162`). **Fixed:** x⁰ is constant, so the tangent is 0. Pinned by `Dual_PowWithZeroExponentDoesNotProduceNaN`.
- **`dual.hpp` API gaps:**
  - ✅ **Partly fixed:** `atan2`, `asin`, `acos`, `min` and `max` added (`atan2` matters most, it is the quadrant-correct, everywhere-finite form needed by SO(3)/SE(3) log maps). `gelu_exact` now uses `std::erf` on the float path. ➖ `pow(T, Dual)`, `pow(Dual, Dual)`, `log1p`/`expm1` and `hypot` are still absent; no caller needs them yet.
  - ➖ `Jet` (formerly `HyperDual`) still lacks `tan`, `abs`, `pow`, `atan`, `sinh` and `cosh`. Left until a caller needs them; the ADL two-step means adding them is mechanical.
  - ➖ The scalar constructor is still implicit. Making it `explicit` would break the `Dual + 1.0f` mixed arithmetic the tests rely on; revisit alongside the tensor-element design (T6).
- **Elementwise kernels:**
  - ✅ `sigmoid` and `gelu` always use recursive index iteration (`elementwise.hpp:93-118`). **Fixed:** both take a contiguous fast path.
  - ✅ `relu(NaN)` differs between AVX2 and NEON (`simd.hpp:127` vs `:134`; see T8). **Fixed:** the NEON path uses compare-and-select so both return 0, pinned by `Regression_ReluNaNPolicyIsIsaIndependent`.
- **Allocators:**
  - ✅ `slab_allocator.hpp:97-125` has no release-mode size, ownership or double-free checks. **Fixed:** size and alignment are checked by throwing, and `deallocate` refuses foreign or interior pointers instead of corrupting the free list. Pinned by `Regression_SlabValidationSurvivesRelease`. ➖ Double-free is still undetected, catching it needs a per-block bitmap the intrusive list has nowhere to put.
  - ✅ The capacity check in `src/core/arena_allocator.cpp:73-75` can wrap for huge `bytes`. **Fixed:** the check is done by subtraction against the class invariant, and alignment is validated in every build. Pinned by `Regression_ArenaRejectsOverflowingAndInvalidRequests`.
- **Tensor and view types:**
  - ✅ Implicit view conversions from temporaries dangle (`tensor.hpp:101-102`; T2). **Fixed:** `view()`, `t()` and the conversion operator are ref-qualified, and the rvalue overload is `= delete`.
  - ➖ `Strides` still duplicates the rank (`shape.hpp:218`; T5). Removing it changes the descriptor layout, which belongs with the T5 decision rather than ahead of it.
- **GEMM:**
  - ➖ `pack_A_panel` is still unused. It is the half of Goto's algorithm the kernel has yet to use; deleting it would remove the scaffolding for the micro-kernel work in T3.
  - ✅ The `m_tile` loop at `gemm.cpp:165` performs no tiling. **Fixed:** removed.
  - ✅ With α = 0, `gemm_packed` still computes and propagates NaN from A/B, unlike BLAS. **Fixed:** both kernels return after the β scaling, pinned by `Regression_AlphaZeroLeavesOperandsUnread`. ➖ Aliasing between C and A/B is still undocumented.
- **Memory sandbox:** ✅ only the first line of a split access is simulated. **Fixed:** every line the access touches is looked up, and 4 KB page splits are counted separately from line splits, since a page split costs a second TLB translation while a line split costs one extra lookup. Pinned by `MemorySandbox_SplitAccessQueriesEveryLineItTouches` and `MemorySandbox_PageSplitsAreCountedSeparately`. ➖ L2 is still not refreshed on an L1 hit, which models a strictly exclusive hierarchy; real x86 L2s are non-inclusive-non-exclusive, and fixing it properly means choosing a policy to model.
- **Tests and build:**
  - ✅ `tests/test_hardware_info.cpp:32-35` requires AVX2/FMA on every x86-64 host. **Fixed:** the test now asserts internal consistency (AVX2 ⇒ AVX, NEON never on x86) and only requires AVX2 when `KINESIS_HAS_AVX2` was compiled in.
  - ✅ `build.bat` is missing `test_hardware_info.cpp` and `test_dual.cpp`. **Fixed:** all three entry points now list the same ten test files.
  - ✅ `-march=native` in `CMakeLists.txt:31` produces non-portable binaries. **Fixed:** it is now the `KINESIS_NATIVE_ARCH` option (default ON), and turning it off falls back to `-mavx2 -mfma` so the SIMD kernels do not silently compile out. `KINESIS_SANITIZE` was added alongside it.

---

<a id="s7"></a>
## §7 Documentation accuracy corrections (by document)

Each item below also has an inline **Review note (2026-09-17)** callout in its document.

- **README.md:**
  - The zero-allocation and AVX-512 claims (T1; §5).
  - The quick-start allocates weights from the per-tick arena, so they're uninitialized every tick, and `gemm` allocates for sizes ≥ 128.
  - Evidence for the performance table (§6, defect 14).
  - GPU support is a plan with prerequisites (T13).
  - The directory tree and test count were stale (now fixed).
- **docs/ARCHITECTURE.md:** per-section notes referencing §2 and §3:
  - vendor-table sourcing;
  - alignment vs shape multiples;
  - bank geometry and the gcd rule;
  - cache-oblivious evidence;
  - CPU vs GPU latencies;
  - detection and safety;
  - the 104-byte view;
  - UB equality;
  - compiler loop splitting;
  - normal equations;
  - dual-number naming.
- **docs/BENCHMARKS.md:** methodology caveats, and numbers that are inconsistent across documents.
- **docs/PRODUCTION_SPEC_AND_CRITIQUE_ANALYSIS.md:**
  - **Fault 1:** the variance constant; the MeZO description.
  - **Fault 2:** GrAPE venue, schedule and results; Launay 2020; dependence on output width.
  - **Fault 3:** RevNet, Rev-ViT, Behrmann and checkpointing evidence; prerequisites for fixed-point coupling.
  - **Fault 4:** Shampoo/SOAP evidence; Newton-Schulz bounds and roles.
  - **Fault 5:** see T7.
  - **Fault 6:** see the GGML items below.
  - **Point 1:** 104 bytes; 32-bit stride overflow; ABI.
  - **Point 2:** UB.
  - **Point 3:** vector dimension vs tensor rank.
  - **Point 4:** ω is an upper bound.
  - **Point 5:** normal equations square the condition number (κ(AᵀA) = κ(A)²), QR/SVD are preferred, and A⁺ = (AᵀA)⁻¹Aᵀ requires full column rank.
  - **Point 6:** GCC `-O3` already splits loops.
  - **Point 8:** char-type aliasing.
  - **Point 10:** contraction and determinism.
  - **§4 invariants:** implementation status.
- **docs/GGML_ANALYSIS_AND_IMPROVEMENTS.md:**
  - The 60-80% waste figure comes from the vLLM work and describes earlier serving systems.
  - llama.cpp already has a unified KV cache with `seq_cp` prefix sharing, plus an open paged-KV design discussion (#21961).
  - tinyBLAS (C++ templates, not assembly blobs) was merged in 2024 (PR #6414).
  - `--numa distribute|isolate|numactl` and `GGML_NUMA_MIRROR` (PR #27986) exist.
  - Nondeterminism stems from batch invariance, not thread count; row-parallel matmul keeps the per-element summation order.
  - ggml was split into `ggml-base` plus backend libraries (PR #10256).
  - The Kinesis column describes plans, and its numbers conflict with other documents.
  - ggml-backend is the pattern Kinesis needs for GPU (T13).
- **docs/LEARNING_JOURNAL.md:** Invalidation 19 added, with markers at:
  - **Inv. 10:** the stall math. Ada's L2 is 96 MB (16× GA102), with 72 MB enabled on the RTX 4090. "40% bus traffic" is unsourced. The cycle figures come from Grassi, not Chips and Cheese, which reports nanoseconds.
  - **Inv. 14:** the TCN receptive field.
  - **Inv. 15:** HyperDual; GrAPE.
  - **Inv. 16:** the second-order h_v.
  - **Inv. 17-18:** 64 bytes; GrAPE numbers.
- **docs/LEARNING_GUIDE.md:**
  - Exercises 3 and 6 show code that differs from the current sources.
  - Exercise 4: page-size waste.
  - Exercise 5: benchmark caveat.
  - Experiment 3: the formula still gives 4× conflicts (gcd rule).
  - Exercise 9: the safety policy isn't implemented.
  - The duplicated "Silicon & Hardware Experiments" block was removed.
- **docs/CHANGELOG.md:** working-tree changes recorded, test count updated, links made relative.

---

<a id="s8"></a>
## §8 Conventions & low-level notes

These are secondary to §2, §3, but worth fixing the next time the surrounding code is touched.

- **Loop fission (`tensor_view.hpp:234-249`):**
  - Fine for readability, but GCC already enables `-fsplit-loops` and `-funswitch-loops` at `-O3`, and with rank ≤ 6 there's no measurable gain.
  - Loop 2 still contains two branches, so the "100% unconditional / branch-free" comments are inaccurate.
- **Loop unswitching opportunities:**
  - the contiguity check inside the (i, k) loops of `gemm_blocked` (`gemm.cpp:77`);
  - the `beta == 0` check inside the (i, j) loops of `gemm_naive` (`gemm.cpp:29`);
  - the per-panel `count == 8` branch in `gemm_packed` (`gemm.cpp:184`). Splitting full panels from a single tail step is also the natural place to fix defect 2 with masked loads and stores.
- **Prefix vs postfix increment:**
  - Loops consistently use `++i`.
  - Postfix appears only on built-in integers and pointers (`*dest++`, `dims_[i++]`, counters), where it costs nothing.
  - PRODUCTION_SPEC Point 7 is correct. No change is needed, though a clang-tidy rule could enforce the style.
- **Hoisting bounds and metadata into locals:**
  - `const size_t len = rank_` is used in `shape.hpp`, but not in `at_indices`, `iterate_recursive` or `is_c_contiguous`.
  - It matters most for `TensorView<uint8_t>`, `char` or `std::byte` (e.g. ROS image frames). Character types may alias any object, so unless rank, strides and offset are hoisted into locals, the compiler must reload them after every write.
- **Iteration:** `iterate_recursive` recurses per dimension with a lambda call per element. An iterative "odometer", or a stride-walking pointer with a coalesced contiguous inner dimension (NumPy's nditer approach), is faster and simpler.
- **Duplicated kernels:** `add`/`sub`/`mul`/`scale`/`relu` each repeat the same three ISA branches. A single template with an operation functor, or compiler auto-vectorization for trivial maps, removes the duplication.
- **Formatting:**
  - There's no `.clang-format` or `.editorconfig`.
  - `tensor_view.hpp` (2-space) and `types.hpp` were reformatted in the same change as the semantic edits, adding ~570 lines of diff noise. Commit formatting separately.
- **Include hygiene:**
  - `<iostream>` is included in `dual.hpp`, `hardware_info.hpp` and `memory_sandbox.hpp`.
  - `<windows.h>` is reachable from `tensor.hpp` via `allocator.hpp`.
  - Unused includes: `<thread>` (`gemm.cpp`), `<string_view>` (`types.hpp`, `arena_allocator.hpp`), `<numeric>` (`shape.hpp`), `<memory>` (`tensor.hpp`).
  - `gemm.hpp` uses `std::abs` without including `<cmath>`.
- **Naming:**
  - camelCase parameters (`capacityBytes`, `externalBuffer`, `blockCount`) sit among snake_case code.
  - `HeapAllocator` names an allocator that doesn't use the heap, ✅ *fixed 2026-09-18: it is now a real aligned-heap allocator*.
  - `Shape(4)` (rank 4) vs `Shape{4}` (one dimension of size 4) is a footgun.
- **Magic numbers:** 49,152 (L1), 128 (dispatch threshold), 64 (tiles), `1e-12f` (determinant) and `Nr = 8` should be named, or moved into a configuration/context object (T11). *(2026-09-18: `1e-12f` is gone, replaced by a relative `relative_epsilon` parameter; the others remain.)*
- **Performance claims in comments:** "~3 instructions", "1 clock cycle", "zero branch mispredictions" and "branchless" (on code that branches) belong in benchmarks, where they can be measured.
- **API hygiene:**
  - `at_indices`/`iterate_indices` are public, though commented as internal.
  - `operator~` for transpose is surprising; Eigen uses `.transpose()` and NumPy uses `.T`.
  - The free `transpose(view)` silently swaps dims 0 and 1 for any rank.
  - Default arguments on virtual functions differ between `Allocator` (64) and `SlabAllocator` (`Alignment`), and they're bound statically.
  - Some `noexcept` functions call helpers that aren't `noexcept`.
- **Index types:** `int64_t` indices, `size_t` rank and `uint32_t` dims mix signedness, which triggers the `-Wconversion` warnings CMake enables. Pick one index type.
- **Tests:**
  - The harness turns assertion failures into exceptions, but `assert` aborts the whole run in debug builds.
  - There's no Debug/Release/ASan/UBSan matrix.
  - Hardware tests assume the build host.
- **Documentation hygiene:**
  - Absolute `file:///c:/Users/...` links have now been replaced.
  - The same metric has different values in different documents: 0.64 vs 1.05 ns; 5.86 vs 5.14 ns; 0.11/0.14/0.22 ns; 20.59 vs 29.27 GFLOPS.
  - Documents give test counts of 13/22/25/30, but **35** test cases now exist: 4 arena, 2 virtual memory, 2 slab, 4 sandbox, 3 hardware, 9 tensor view, 6 GEMM and 5 dual.

---

<a id="s9"></a>
## §9 Suggested decision order (recommendations only)

1. Fix correctness defects 1-4 and 9. Add tests for partial panels and strided inputs, and add Debug/Release + ASan/UBSan runs.
2. Decide the **kernel contract**: an explicit workspace (the KleidiAI/oneDNN model), stride/transpose handling, const views, and a dtype (T1, T3-T5).
3. Decide the **CPU+GPU abstractions** (memory spaces, buffer types, streams, scheduler) before adding more kernels (T13).
4. Decide the **lifetime model** and the **error-handling policy** before building layers and the reverse-mode tape (T2, T14).
5. Decide the **determinism scope** before building bit-exact reversible coupling (T8, T16).
6. Wire detection into a `KernelContext` with runtime dispatch, and rebuild the evidence base with reproducible benchmarks against SOTA baselines (T11, T12).
7. Correct the logic errors in the documents (§2.1), and relabel the hypotheses: hardware protection, the constant-cache knee, and XOR swizzle.
8. Record future decisions as ADRs with *Status* (Proposed / Accepted / Implemented / Superseded), *Evidence*, *Consequences* and *Supersedes*.

---

<a id="s10"></a>
## §10 Sources

### Papers
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://pldi19.sigplan.org/details/mapl-2019-papers/1/Triton-An-Intermediate-Language-and-Compiler-for-Tiled-Neural-Network-Computations) (Tillet, Kung, Cox; MAPL 2019)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) (Micikevicius et al., 2022)
- [Analytical Modeling Is Enough for High-Performance BLIS](https://dl.acm.org/doi/10.1145/2925987) (Low, Igual, Smith, Quintana-Ortí; ACM TOMS 2016)
- [An Experimental Comparison of Cache-oblivious and Cache-conscious Programs](https://dl.acm.org/doi/10.1145/1248377.1248394) (Yotov, Roeder, Pingali, Gunnels, Gustavson; SPAA 2007)
- [The Indirect Convolution Algorithm](https://arxiv.org/abs/1907.02129) (Dukhan, 2019)
- [Analytical Derivatives of Rigid Body Dynamics Algorithms](http://www.roboticsproceedings.org/rss14/p38.pdf) (Carpentier & Mansard; RSS 2018)
- [arXiv 2105.05102: Efficient analytical derivatives of rigid-body dynamics using spatial vector algebra](https://arxiv.org/abs/2105.05102)
- [arXiv 2302.06001: On Second-Order Derivatives of Rigid-Body Dynamics: Theory & Implementation](https://arxiv.org/abs/2302.06001)
- [arXiv 1709.03799: Automatic Differentiation of Rigid Body Dynamics for Optimal Control and Estimation](https://arxiv.org/abs/1709.03799)
- [Enzyme: Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients](https://arxiv.org/abs/2010.01709) (Moses & Churavy; NeurIPS 2020)
- [A micro Lie theory for state estimation in robotics](https://arxiv.org/abs/1812.01537) (Solà, Deray, Atchuthan)
- [The Stan Math Library: Reverse-Mode Automatic Differentiation in C++](https://arxiv.org/abs/1509.07164)
- [The Development of Hyper-Dual Numbers for Exact Second-Derivative Calculations](https://www.semanticscholar.org/paper/The-Development-of-Hyper-Dual-Numbers-for-Exact-Fike-Alonso/eee195a1b3dd324b95fbd5bf148d6fba49f70850) (Fike & Alonso, 2011)
- [Second-Order Forward-Mode Automatic Differentiation for Optimization (FoMoH)](https://arxiv.org/abs/2408.10419) (Cobb, Baydin, Pearlmutter, Jha; 2024)
- [Perturbation confusion in forward automatic differentiation of higher-order functions](https://engineering.purdue.edu/~qobi/papers/jfp2019.pdf) (Manzyuk et al.; JFP 2019)
- [Scaling Direct Feedback Learning with Jacobian Alignment Guarantees (GrAPE)](https://mlanthology.org/iclr/2026/caillon2026iclr-scaling/) (Caillon, Fagnou, Delattre, Allauzen; ICLR 2026), plus the [paper-notes summary (secondary)](https://en.papernotes.org/ICLR2026/others/scaling_direct_feedback_learning_with_jacobian_alignment_guarantees/)
- [Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures](https://arxiv.org/abs/2006.12878) (Launay et al.; NeurIPS 2020)
- [Can Forward Gradient Match Backpropagation?](https://arxiv.org/abs/2306.06968) (Fournier et al.; ICML 2023)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) (Moonshot AI, 2025)
- [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321)
- [A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer](https://arxiv.org/abs/2309.06497)
- [Deconstructing What Makes a Good Optimizer for Language Models](https://arxiv.org/abs/2407.07972)
- [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861) (Dettmers et al.)
- [Reversible Vision Transformers](https://arxiv.org/abs/2302.04869) (Mangalam et al.; CVPR 2022)
- [The Reversible Residual Network: Backpropagation Without Storing Activations](https://arxiv.org/abs/1707.04585) (Gomez et al.; NeurIPS 2017)
- [Understanding and Mitigating Exploding Inverses in Invertible Neural Networks](https://arxiv.org/abs/2006.09347) (Behrmann et al.)
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (Chen et al., 2016)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) (Narayanan et al., 2021)
- [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) (Bai, Kolter, Koltun, 2018)
- [More Asymmetry Yields Faster Matrix Multiplication](https://arxiv.org/abs/2404.16349) and [arXiv 2608.16884 (improved exponent)](https://arxiv.org/abs/2608.16884)
- [Disruptor: High performance alternative to bounded queues for exchanging data between concurrent threads](https://lmax-exchange.github.io/disruptor/disruptor.html) (Thompson et al., LMAX, 2011)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/2023-06-20-vllm)

### Vendor & company documentation
- **NVIDIA:**
  - [Ada GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
  - [Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
  - [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
  - [Stream-Ordered Memory Allocator (CUDA runtime API)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html)
  - [CUDA Graph best practices](https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/introduction.html)
  - [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc) (Harris, 2013)
- **Intel:**
  - [How to detect New Instruction support (OSXSAVE/XGETBV)](https://www.intel.com/content/dam/develop/external/us/en/documents/how-to-detect-new-instruction-support-in-the-4th-generation-intel-core-processor-family.pdf)
  - [oneMKL Conditional Numerical Reproducibility](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2023-2/get-started-with-conditional-num-reproducibility.html)
- **Microsoft:** [x64 calling convention](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention)
- **PyTorch:**
  - [CUDA semantics (caching allocator, expandable segments, CUDA Graphs, pinned memory)](https://docs.pytorch.org/docs/stable/notes/cuda.html)
  - [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
  - [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- **Apple MLX:** [Unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- **Deterministic inference:**
  - Thinking Machines: [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
  - LMSYS: [Deterministic inference in SGLang](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- **Embedded and mobile kernels:**
  - Arm: [KleidiAI](https://github.com/ARM-software/kleidiai)
  - Espressif: [ESP-NN](https://github.com/espressif/esp-nn)
  - Google: [XNNPACK](https://github.com/google/XNNPACK)
  - [TensorFlow Lite Micro memory management](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md)
- **UXL Foundation:** [oneDNN user-managed scratchpad](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_scratchpad.html)
- **Red Hat:** [Using mlock() on RHEL for Real Time](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_for_real_time/8/html/optimizing_rhel_8_for_real_time_for_low_latency_operation/assembly_using-mlock-system-calls-on-rhel-for-real-time_optimizing-rhel8-for-real-time-for-low-latency-operation)
- **Linux kernel:** [Bus lock detection and handling](https://www.kernel.org/doc/html/v5.18/x86/buslock.html)
- **GCC:** [Optimize options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
- **cppreference:**
  - [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan)
  - [`std::pmr::monotonic_buffer_resource`](https://en.cppreference.com/w/cpp/memory/monotonic_buffer_resource)
  - [`reinterpret_cast` (type aliasing)](https://en.cppreference.com/w/cpp/language/reinterpret_cast)

### Repositories
- **BLIS:** [flame/blis Haswell configuration](https://github.com/flame/blis/blob/master/config/haswell/bli_cntx_init_haswell.c)
- **ggml-org/llama.cpp:**
  - [ggml-backend.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h)
  - [ggml-alloc.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-alloc.h)
  - [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
  - [PR #6414 (tinyBLAS)](https://github.com/ggml-org/llama.cpp/pull/6414)
  - [PR #10256 (backends as libraries)](https://github.com/ggml-org/llama.cpp/pull/10256)
  - [PR #27986 (NUMA mirror)](https://github.com/ggml-org/llama.cpp/pull/27986)
  - [Discussion #21961 (paged KV design)](https://github.com/ggml-org/llama.cpp/discussions/21961)
- **PyTorch:** [pytorch/pytorch `SizesAndStrides`](https://github.com/pytorch/pytorch/blob/main/c10/core/impl/SizesAndStrides.h)
- **Serving and kernels:** [vllm-project/vllm](https://github.com/vllm-project/vllm), [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention), [triton-lang/triton](https://github.com/triton-lang/triton)
- **Robotics and autodiff:**
  - [stack-of-tasks/pinocchio](https://github.com/stack-of-tasks/pinocchio)
  - [ceres-solver jet.h](https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/jet.h)
  - RobotLocomotion/drake autodiff benchmarking: [#8482](https://github.com/RobotLocomotion/drake/issues/8482), [#10322](https://github.com/RobotLocomotion/drake/issues/10322)
  - [EnzymeAD/Enzyme](https://github.com/EnzymeAD/Enzyme)
- **SIMD dispatch:** [google/highway](https://github.com/google/highway)
- **Optimizers:**
  - [facebookresearch/optimizers: Distributed Shampoo](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md)
  - [Keller Jordan: Muon](https://kellerjordan.github.io/posts/muon/)
  - [MLCommons: AlgoPerf 2024 results](https://mlcommons.org/2024/08/mlc-algoperf-benchmark-competition/)
- **Raspberry Pi GPU:** [Tencent/ncnn issue #2435 (Vulkan on Raspberry Pi 4)](https://github.com/Tencent/ncnn/issues/2435)

### Hugging Face
- **Transformers:**
  - [KV cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache)
  - [Optimization overview](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one)
  - [Quantization overview](https://huggingface.co/docs/transformers/main/en/quantization/overview)
- [safetensors](https://huggingface.co/docs/safetensors/index)
- [`kernels` library](https://github.com/huggingface/kernels) and [kernels-community](https://huggingface.co/kernels-community)
- [Candle](https://github.com/huggingface/candle)
- Community blog: [FlashAttention: Making Attention I/O-Aware](https://huggingface.co/blog/garg-aayush/flash-attention)

### Articles & measurements
- **Charles Grassi:** [GPU cache hierarchy](https://charlesgrassi.dev/blog/gpu-cache-hierarchy/) (2025)
- **Chips and Cheese:**
  - [Measuring GPU memory latency](https://chipsandcheese.com/p/measuring-gpu-memory-latency) (2021)
  - [Microbenchmarking NVIDIA's RTX 4090](https://chipsandcheese.com/p/microbenchmarking-nvidias-rtx-4090) (2022)
  - [Going armchair quarterback on Golden Cove's caches](https://chipsandcheese.com/p/going-armchair-quarterback-on-golden-coves-caches)
  - [Analyzing Zen 2's Cinebench R15 lead](https://chipsandcheese.com/p/analyzing-zen-2s-cinebench-r15-lead)
- **Danil:** [Decompiling NVIDIA shaders, and optimizing](https://danilw.github.io/blog/decompiling_and_optimizing_nvidia_shaders/)
- **Fabian Giesen:**
  - [Texture tiling and swizzling](https://fgiesen.wordpress.com/2011/01/17/texture-tiling-and-swizzling/) (2011)
  - [The magic ring buffer](https://fgiesen.wordpress.com/2012/07/21/the-magic-ring-buffer/) (2012)
  - Critique: [The Magic Ring Buffer: A Beautiful Trick That's Mostly Not Worth It](https://hftuniversity.com/post/the-magic-ring-buffer-a-beautiful-trick-that-s-mostly-not-worth-it)
- **Travis Downs:** [Gathering Intel on Intel AVX-512 transitions](https://travisdowns.github.io/blog/2020/01/17/avxfreq1.html) (2020)
- **Erik Rigtorp:**
  - [Latency implications of virtual memory](https://rigtorp.se/virtual-memory/)
  - [Performance impact of split locks](https://rigtorp.se/split-locks/)
- **Simon Boehm:** [Can function inlining affect floating-point outputs?](https://siboehm.com/articles/23/Inlining-FMA-FP-consistency)
- **Ethan Epperly:** [Don't solve the normal equations](https://www.ethanepperly.com/index.php/2022/07/26/dont-solve-the-normal-equations/)
- **salykova:** [Advanced matrix multiplication optimization on modern multi-core processors](https://salykova.github.io/matmul-cpu)
- **cpufun:** [More M1 fun: hardware information (128-byte cache line)](https://cpufun.substack.com/p/more-m1-fun-hardware-information)
- **Apple AMX research:** [PQC-AMX (IACR ePrint 2024/195)](https://eprint.iacr.org/2024/195)
