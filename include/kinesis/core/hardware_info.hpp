#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <array>
#include <iostream>
#include <iomanip>

#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
    #include <cpuid.h>
#endif

namespace kinesis::core {

// Platform Execution Profile
enum class PlatformProfile {
    DesktopServer,  // High-performance x86_64 or Server ARM64 (Full Virtual Memory, multi-core, large L3)
    EdgeSBC,        // Single-Board Computers (Raspberry Pi 4/5, NVIDIA Jetson, Linux ARM64 with NEON)
    EmbeddedMCU     // Bare-metal microcontrollers (ESP32, STM32, RISC-V with no MMU, limited SRAM <= 520KB)
};

// Target CPU Architecture
enum class CpuArchitecture {
    x86_64,
    ARM64,
    ARM32,
    RISCV,
    Xtensa,  // ESP32 LX6/LX7
    Unknown
};

// Target GPU Hardware Vendor
enum class GpuVendor {
    NVIDIA,             // Hopper, Blackwell, Ada Lovelace, Ampere
    AMD,                // CDNA3, RDNA3/4
    Intel,              // Ponte Vecchio, Arc, Battlemage
    Apple,              // M-series Unified Memory GPU
    Broadcom_VideoCore, // Raspberry Pi 4/5 (VideoCore VI/VII)
    ARM_Mali,           // Embedded / Mobile Mali
    Qualcomm_Adreno,    // Snapdragon / Robotics Adreno
    None,               // Confirmed absent: headless / bare-metal execution
    Unknown             // Not determined: no GPU runtime is linked in to ask
};

/**
 * @brief Declared limits for thermal, transient-power and memory behaviour.
 *
 * @warning This is a declaration, not a mechanism. No code in Kinesis reads these fields,
 * so constructing one enforces nothing. Two of the assumptions behind it also need
 * qualifying before anything is built on top:
 *  - Wide-SIMD voltage and frequency transitions are managed by the CPU itself (measured at
 *    roughly 8-20 us at a quarter dispatch rate, with about 680 us before relaxing), so staged
 *    warmup is a *latency* technique for avoiding sporadic wide-SIMD bursts on a hot path,
 *    not protection for the silicon.
 *  - Reading die temperature needs a privileged MSR or an OS/vendor API; none is used here.
 * See docs/DESIGN_REVIEW_2026-09-17.md T17.
 */
struct HardwareSafetyPolicy {
    // Thermal Guards (Celsius)
    float max_operating_temp_c = 85.0f;     // Warning threshold
    float critical_throttle_temp_c = 92.0f; // Force sleep/yield before hardware THERMTRIP# (100-105C)
    
    // Transient Power Shock (di/dt) Mitigation
    bool enable_staged_warmup = true;       // Ramp up large matrix kernels over micro-iterations to protect VRMs
    size_t warmup_micro_steps = 4;          // Slices to stage full AVX-512 / Tensor Core activation
    
    // Memory Quota Fences
    size_t max_ram_quota_bytes = 0;         // 0 = unbounded; otherwise bounds arena to prevent OS paging stalls
    bool lock_pages_in_physical_ram = true; // Lock committed pages to prevent paging file swapping
};

/**
 * @brief Comprehensive Hardware and System Topology Detector.
 */
class HardwareInfo {
public:
    // Detection methods
    [[nodiscard]] static constexpr CpuArchitecture detect_cpu_architecture() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        return CpuArchitecture::x86_64;
#elif defined(__aarch64__) || defined(_M_ARM64)
        return CpuArchitecture::ARM64;
#elif defined(__arm__) || defined(_M_ARM)
        return CpuArchitecture::ARM32;
#elif defined(__riscv)
        return CpuArchitecture::RISCV;
#elif defined(__XTENSA__) || defined(ESP_PLATFORM)
        return CpuArchitecture::Xtensa;
#else
        return CpuArchitecture::Unknown;
#endif
    }

    [[nodiscard]] static constexpr PlatformProfile detect_platform_profile() noexcept {
#if defined(ESP_PLATFORM) || defined(__XTENSA__) || defined(KINESIS_EMBEDDED)
        return PlatformProfile::EmbeddedMCU;
#elif defined(__aarch64__) && (defined(__linux__) || defined(__unix__))
        // Detect typical Linux SBC targets (Raspberry Pi 4/5, Jetson)
        return PlatformProfile::EdgeSBC;
#else
        return PlatformProfile::DesktopServer;
#endif
    }

    [[nodiscard]] static std::string_view cpu_architecture_name() noexcept {
        switch (detect_cpu_architecture()) {
            case CpuArchitecture::x86_64: return "x86_64 (Intel / AMD)";
            case CpuArchitecture::ARM64:  return "ARM64 (Apple Silicon / Raspberry Pi 5 / Neoverse)";
            case CpuArchitecture::ARM32:  return "ARM32 (Raspberry Pi 2/3 32-bit / Cortex-A7)";
            case CpuArchitecture::RISCV:  return "RISC-V (ESP32-C3/C6 / SiFive)";
            case CpuArchitecture::Xtensa: return "Xtensa LX6/LX7 (ESP32 / ESP32-S3)";
            default:                      return "Unknown Architecture";
        }
    }

    [[nodiscard]] static std::string_view platform_profile_name() noexcept {
        switch (detect_platform_profile()) {
            case PlatformProfile::DesktopServer: return "Desktop / Server (Virtual Memory Active)";
            case PlatformProfile::EdgeSBC:        return "Edge SBC (Linux ARM64 / Embedded NEON)";
            case PlatformProfile::EmbeddedMCU:    return "Embedded MCU (Bare-Metal / Static SRAM)";
        }
        return "Unknown Profile";
    }

    // Hardware Feature Queries
    struct CpuFeatures {
        bool has_avx = false;
        bool has_avx2 = false;
        bool has_avx512f = false;
        bool has_fma = false;
        bool has_neon = false;
        bool has_amx = false;
        size_t cache_line_size = 64;
        size_t l1_data_cache_size = 32 * 1024;
        size_t l2_cache_size = 512 * 1024;
        size_t l3_cache_size = 16 * 1024 * 1024;

        // False when the cache fields below are the compiled-in fallbacks rather than values
        // read back from the hardware. Anything that sizes a tile from them should check this.
        bool cache_sizes_measured = false;
    };

    [[nodiscard]] static CpuFeatures query_cpu_features() noexcept {
        CpuFeatures features{};

#if defined(__x86_64__) || defined(_M_X64)
        uint32_t regs[4] = {0};

        // A CPUID leaf above the reported maximum returns the highest supported leaf's data
        // instead of failing, so every leaf below is gated on leaf 0 first.
        cpuid_query(0, 0, regs);
        const uint32_t max_basic_leaf = regs[0];
        if (max_basic_leaf < 1) {
            return features;
        }

        // EAX=1: Processor Info and Feature Bits
        cpuid_query(1, 0, regs);
        const bool cpu_has_avx = (regs[2] & (1u << 28)) != 0;
        const bool cpu_has_fma = (regs[2] & (1u << 12)) != 0;
        const bool has_xsave = (regs[2] & (1u << 26)) != 0;
        const bool has_osxsave = (regs[2] & (1u << 27)) != 0; // OS has executed XSETBV

        // A CPU feature bit only says the silicon can decode the instruction. Using YMM or ZMM
        // registers also requires the OS to have enabled saving them across context switches,
        // which is what OSXSAVE plus the XCR0 state-component bits report. Without this check,
        // AVX code faults with #UD on an OS that has not enabled the state (Intel, "How to
        // detect New Instruction support in the 4th generation Intel Core processor family").
        uint64_t xcr0 = 0;
        if (has_xsave && has_osxsave) {
            xcr0 = read_xcr0();
        }
        constexpr uint64_t kXcr0AvxMask = 0x6;         // XMM (bit 1) | YMM (bit 2)
        constexpr uint64_t kXcr0Avx512Mask = 0xE6;     // AVX | opmask | ZMM_Hi256 | Hi16_ZMM
        constexpr uint64_t kXcr0AmxMask = 0x60000;     // XTILECFG (bit 17) | XTILEDATA (bit 18)
        const bool os_saves_ymm = (xcr0 & kXcr0AvxMask) == kXcr0AvxMask;
        const bool os_saves_zmm = (xcr0 & kXcr0Avx512Mask) == kXcr0Avx512Mask;
        const bool os_saves_tiles = (xcr0 & kXcr0AmxMask) == kXcr0AmxMask;

        features.has_avx = cpu_has_avx && os_saves_ymm;
        features.has_fma = cpu_has_fma && os_saves_ymm; // FMA operates on YMM state

        // EAX=7, ECX=0: Structured Extended Feature Flags
        if (max_basic_leaf >= 7) {
            cpuid_query(7, 0, regs);
            features.has_avx2 = (regs[1] & (1u << 5)) != 0 && os_saves_ymm;
            features.has_avx512f = (regs[1] & (1u << 16)) != 0 && os_saves_zmm;
            features.has_amx = (regs[3] & (1u << 22)) != 0 && os_saves_tiles; // AMX-BF16
        }

        // Cache geometry, read from the hardware rather than assumed. Leaf 4 is Intel's
        // deterministic cache parameters; AMD exposes the same layout at leaf 0x8000001D.
        features.cache_sizes_measured = query_x86_cache_topology(max_basic_leaf, features);
        if (!features.cache_sizes_measured) {
            // Fallbacks, flagged as such by cache_sizes_measured == false.
            features.cache_line_size = 64;
            features.l1_data_cache_size = 32 * 1024;
            features.l2_cache_size = 512 * 1024;
            features.l3_cache_size = 16 * 1024 * 1024;
        }

#elif defined(__aarch64__) || defined(_M_ARM64)
        features.has_neon = true; // NEON is mandatory in ARMv8-A
        features.cache_line_size = 64; // Raspberry Pi 4/5 is 64 bytes (Apple M-series is 128B)
#if defined(__APPLE__)
        features.cache_line_size = 128; // Apple Silicon M-series performance cores
#endif
        features.l1_data_cache_size = 64 * 1024;
        features.l2_cache_size = 512 * 1024;
        features.l3_cache_size = 2 * 1024 * 1024;

#elif defined(__XTENSA__) || defined(ESP_PLATFORM)
        // ESP32 Microcontroller Profile
        features.cache_line_size = 32; // 32-byte cache line on ESP32 flash cache
        features.l1_data_cache_size = 32 * 1024;
        features.l2_cache_size = 0;
        features.l3_cache_size = 0;
#endif

        return features;
    }

    // GPU Discovery
    struct GpuInfo {
        GpuVendor vendor = GpuVendor::Unknown;
        std::string device_name = "Not detected";
        size_t vram_bytes = 0;
        size_t compute_units = 0;
        bool has_tensor_cores = false;
        bool supports_fp16 = false;

        // True only when the fields above came from a device query. While no GPU runtime is
        // linked, this is false on every host that could have a GPU, and callers must not
        // treat vendor, VRAM or compute_units as facts about the machine.
        bool detected = false;
    };

    /**
     * @brief Report what is known about an attached GPU.
     *
     * @warning Kinesis links no GPU runtime, so on any host that could have one this reports
     * GpuVendor::Unknown with detected == false. Real enumeration needs a driver API --
     * cudaGetDeviceProperties, hipGetDeviceProperties, Metal's MTLCopyAllDevices or
     * vkEnumeratePhysicalDevices -- behind the backend abstraction described in
     * docs/DESIGN_REVIEW_2026-09-17.md T13. Returning a plausible-looking device instead would
     * make every consumer of this function silently wrong.
     */
    [[nodiscard]] static GpuInfo detect_gpu() noexcept {
        GpuInfo gpu{};
        [[maybe_unused]] auto arch = detect_cpu_architecture();
        auto profile = detect_platform_profile();

        if (profile == PlatformProfile::EmbeddedMCU) {
            // The only case that is knowable without a driver: the embedded profile is selected
            // for bare-metal parts that have no GPU and no compute runtime to query.
            gpu.vendor = GpuVendor::None;
            gpu.device_name = "None (bare-metal microcontroller)";
            gpu.detected = true;
            return gpu;
        }

        gpu.vendor = GpuVendor::Unknown;
        gpu.device_name = "Not detected (no GPU runtime linked)";
        gpu.detected = false;
        return gpu;
    }

    /**
     * @brief Print an ASCII hardware topology summary to console.
     */
    static void print_system_report(std::ostream& os = std::cout) {
        auto cpu_feats = query_cpu_features();
        auto gpu = detect_gpu();

        os << "\n======================================================\n";
        os << "         KINESIS HARDWARE & TOPOLOGY DETECTOR         \n";
        os << "======================================================\n";
        os << "CPU Architecture:  " << cpu_architecture_name() << "\n";
        os << "Platform Profile:  " << platform_profile_name() << "\n";
        os << "Hardware Caches" << (cpu_feats.cache_sizes_measured ? ":\n" : " (compiled-in defaults, not measured):\n");
        os << "  - Cache Line:    " << cpu_feats.cache_line_size << " Bytes\n";
        os << "  - L1 Data Cache: " << (cpu_feats.l1_data_cache_size / 1024) << " KB\n";
        os << "  - L2 Cache:      " << (cpu_feats.l2_cache_size / 1024) << " KB\n";
        os << "  - L3 Cache:      " << (cpu_feats.l3_cache_size / (1024 * 1024)) << " MB\n";
        os << "SIMD Acceleration:\n";
        os << "  - AVX2:          " << (cpu_feats.has_avx2 ? "[YES]" : "[NO]") << "\n";
        os << "  - AVX-512F:      " << (cpu_feats.has_avx512f ? "[YES]" : "[NO]") << "\n";
        os << "  - FMA:           " << (cpu_feats.has_fma ? "[YES]" : "[NO]") << "\n";
        os << "  - ARM NEON:      " << (cpu_feats.has_neon ? "[YES]" : "[NO]") << "\n";
        os << "GPU Accelerator:\n";
        os << "  - Device:        " << gpu.device_name << "\n";
        if (gpu.detected) {
            os << "  - Tensor Cores:  " << (gpu.has_tensor_cores ? "[YES]" : "[NO]") << "\n";
            os << "  - FP16 Compute:  " << (gpu.supports_fp16 ? "[YES]" : "[NO]") << "\n";
        } else {
            os << "  - Tensor Cores:  [UNKNOWN]\n";
            os << "  - FP16 Compute:  [UNKNOWN]\n";
        }
        const HardwareSafetyPolicy policy{};
        os << "Hardware Safety (declared policy; no enforcement implemented):\n";
        os << "  - Slew-Rate Vdroop Protection: [NOT IMPLEMENTED]\n";
        os << "  - Thermal Guard Ceiling:       " << std::fixed << std::setprecision(1)
           << policy.max_operating_temp_c << " C (Warning) / "
           << policy.critical_throttle_temp_c << " C (Throttle)  [NOT ENFORCED: no temperature source]\n";
        os << "======================================================\n";
    }

private:
    static void cpuid_query([[maybe_unused]] uint32_t leaf,
                            [[maybe_unused]] uint32_t subleaf,
                            [[maybe_unused]] uint32_t regs[4]) noexcept {
#if defined(_MSC_VER)
        __cpuidex(reinterpret_cast<int*>(regs), static_cast<int>(leaf), static_cast<int>(subleaf));
#elif defined(__x86_64__) || defined(__i386__)
        __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#else
        regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
    }

    // Read XCR0, the extended control register holding the OS-enabled state components.
    // Only safe to execute once CPUID.1:ECX.OSXSAVE is set; the caller checks that.
    [[nodiscard]] static uint64_t read_xcr0() noexcept {
#if defined(_MSC_VER)
        return _xgetbv(0);
#elif defined(__x86_64__) || defined(__i386__)
        uint32_t eax = 0, edx = 0;
        __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
        return (static_cast<uint64_t>(edx) << 32) | eax;
#else
        return 0;
#endif
    }

    // Walk the deterministic cache-parameter subleaves and fill in line size and L1D/L2/L3
    // capacity. Returns false when the leaf is unavailable, leaving the caller's fallbacks in
    // place. Layout per subleaf: EAX[4:0] type (1 = data, 2 = instruction, 3 = unified),
    // EAX[7:5] level, EBX[11:0] line size - 1, EBX[21:12] partitions - 1, EBX[31:22] ways - 1,
    // ECX sets - 1; capacity = ways * partitions * line_size * sets.
    [[nodiscard]] static bool query_x86_cache_topology([[maybe_unused]] uint32_t max_basic_leaf,
                                                       [[maybe_unused]] CpuFeatures& out) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        uint32_t regs[4] = {0};

        uint32_t leaf = 0;
        if (max_basic_leaf >= 4) {
            leaf = 4; // Intel deterministic cache parameters
        } else {
            cpuid_query(0x80000000u, 0, regs);
            if (regs[0] >= 0x8000001Du) {
                leaf = 0x8000001Du; // AMD, identical encoding
            }
        }
        if (leaf == 0) {
            return false;
        }

        bool found_any = false;
        for (uint32_t subleaf = 0; subleaf < 16; ++subleaf) {
            cpuid_query(leaf, subleaf, regs);
            const uint32_t type = regs[0] & 0x1Fu;
            if (type == 0) {
                break; // no more cache levels
            }
            const uint32_t level = (regs[0] >> 5) & 0x7u;
            const size_t line_size = static_cast<size_t>((regs[1] & 0xFFFu) + 1);
            const size_t partitions = static_cast<size_t>(((regs[1] >> 12) & 0x3FFu) + 1);
            const size_t ways = static_cast<size_t>(((regs[1] >> 22) & 0x3FFu) + 1);
            const size_t sets = static_cast<size_t>(regs[2]) + 1;
            const size_t size = ways * partitions * line_size * sets;

            if (level == 1 && type == 1) { // L1 data
                out.l1_data_cache_size = size;
                out.cache_line_size = line_size;
                found_any = true;
            } else if (level == 2) {
                out.l2_cache_size = size;
                found_any = true;
            } else if (level == 3) {
                out.l3_cache_size = size;
                found_any = true;
            }
        }
        return found_any;
#else
        return false;
#endif
    }
};

} // namespace kinesis::core
