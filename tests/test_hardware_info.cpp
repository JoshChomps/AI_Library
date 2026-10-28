#include "test_harness.hpp"
#include "kinesis/core/hardware_info.hpp"
#include <sstream>

using namespace kinesis::core;

TEST_CASE(HardwareInfo_CpuArchitectureAndProfile) {
    auto arch = HardwareInfo::detect_cpu_architecture();
    auto profile = HardwareInfo::detect_platform_profile();

    EXPECT_TRUE(arch != CpuArchitecture::Unknown);
    EXPECT_FALSE(HardwareInfo::cpu_architecture_name().empty());
    EXPECT_FALSE(HardwareInfo::platform_profile_name().empty());

#if defined(__x86_64__) || defined(_M_X64)
    EXPECT_EQ(arch, CpuArchitecture::x86_64);
    EXPECT_EQ(profile, PlatformProfile::DesktopServer);
#endif
}

TEST_CASE(HardwareInfo_CpuFeaturesAndCaches) {
    auto features = HardwareInfo::query_cpu_features();

    // Cache line size must be a valid power-of-two (32, 64, or 128)
    EXPECT_TRUE(features.cache_line_size == 32 || features.cache_line_size == 64 || features.cache_line_size == 128);
    EXPECT_TRUE((features.cache_line_size & (features.cache_line_size - 1)) == 0);

    // L1 data cache must be non-zero
    EXPECT_TRUE(features.l1_data_cache_size >= 16 * 1024);

    // Caches get larger as they get further from the core. L3 may legitimately be absent.
    EXPECT_TRUE(features.l2_cache_size > features.l1_data_cache_size);
    if (features.l3_cache_size > 0) {
        EXPECT_TRUE(features.l3_cache_size > features.l2_cache_size);
    }

    // The ISA flags must be internally consistent rather than whatever this particular host
    // happens to support: AVX2 and FMA both operate on YMM state, so neither can be reported
    // without AVX, and AVX-512F cannot be reported without AVX2's prerequisites either.
    if (features.has_avx2) {
        EXPECT_TRUE(features.has_avx);
    }
    if (features.has_fma) {
        EXPECT_TRUE(features.has_avx);
    }
    if (features.has_avx512f) {
        EXPECT_TRUE(features.has_avx);
    }

#if defined(__x86_64__) || defined(_M_X64)
    // NEON is an ARM feature and must never be reported on x86.
    EXPECT_FALSE(features.has_neon);
#elif defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_TRUE(features.has_neon);   // mandatory in ARMv8-A
    EXPECT_FALSE(features.has_avx2);
#endif

    // Whatever the compiler was told to target must actually be available at runtime, otherwise
    // the kernels compiled behind KINESIS_HAS_AVX2 would fault.
#if defined(KINESIS_HAS_AVX2)
    EXPECT_TRUE(features.has_avx2);
    EXPECT_TRUE(features.has_fma);
#endif
}

TEST_CASE(HardwareInfo_GpuDetectionIsHonestAboutWhatItKnows) {
    auto gpu = HardwareInfo::detect_gpu();
    EXPECT_FALSE(gpu.device_name.empty());

    // No GPU runtime is linked, so on any host that could have a GPU the result must say
    // "unknown" rather than describing a device nobody asked the driver about.
    if (!gpu.detected) {
        EXPECT_EQ(gpu.vendor, GpuVendor::Unknown);
        EXPECT_EQ(gpu.vram_bytes, static_cast<size_t>(0));
        EXPECT_EQ(gpu.compute_units, static_cast<size_t>(0));
        EXPECT_FALSE(gpu.has_tensor_cores);
        EXPECT_FALSE(gpu.supports_fp16);
    } else {
        // The one knowable case is the bare-metal profile, which has no GPU at all.
        EXPECT_EQ(gpu.vendor, GpuVendor::None);
    }
}

TEST_CASE(HardwareInfo_SafetyPolicyIsADeclarationNotAMechanism) {
    HardwareSafetyPolicy policy{};
    EXPECT_TRUE(policy.max_operating_temp_c < policy.critical_throttle_temp_c);
    EXPECT_EQ(policy.max_operating_temp_c, 85.0f);
    EXPECT_EQ(policy.critical_throttle_temp_c, 92.0f);
    EXPECT_TRUE(policy.warmup_micro_steps > 0);

    std::stringstream ss;
    HardwareInfo::print_system_report(ss);
    const std::string report = ss.str();
    EXPECT_TRUE(report.find("KINESIS HARDWARE & TOPOLOGY DETECTOR") != std::string::npos);

    // Nothing reads HardwareSafetyPolicy, so the report must not claim the protections are on.
    EXPECT_TRUE(report.find("[ACTIVE]") == std::string::npos);
    EXPECT_TRUE(report.find("[NOT IMPLEMENTED]") != std::string::npos);
}
