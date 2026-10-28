#pragma once

#include <cmath>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace kinesis::test {

struct TestCase {
    std::string name;
    std::function<void()> func;
};

class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry s_instance;
        return s_instance;
    }

    void add_test(std::string name, std::function<void()> func) {
        tests_.push_back({std::move(name), std::move(func)});
    }

    int run_all() {
        int passed = 0;
        int failed = 0;
        std::cout << "\n======================================================\n";
        std::cout << "        KINESIS AI ENGINE TEST RUNNER (C++20)         \n";
        std::cout << "======================================================\n\n";

        for (const auto& test : tests_) {
            std::cout << "[ RUN      ] " << test.name << " ... ";
            auto start = std::chrono::high_resolution_clock::now();
            try {
                test.func();
                auto end = std::chrono::high_resolution_clock::now();
                auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                std::cout << "\033[32mPASSED\033[0m (" << duration_us << " us)\n";
                passed++;
            } catch (const std::exception& e) {
                std::cout << "\033[31mFAILED\033[0m: " << e.what() << "\n";
                failed++;
            } catch (...) {
                std::cout << "\033[31mFAILED\033[0m: Unknown exception\n";
                failed++;
            }
        }

        std::cout << "\n------------------------------------------------------\n";
        std::cout << "Summary: " << passed << " passed, " << failed << " failed out of " 
                  << tests_.size() << " tests.\n";
        std::cout << "======================================================\n";

        return (failed == 0) ? 0 : 1;
    }

private:
    std::vector<TestCase> tests_;
};

struct TestRegistrar {
    TestRegistrar(std::string name, std::function<void()> func) {
        TestRegistry::instance().add_test(std::move(name), std::move(func));
    }
};

} // namespace kinesis::test

#define KINESIS_TEST_CASE(name) \
    static void test_func_##name(); \
    static ::kinesis::test::TestRegistrar registrar_##name(#name, test_func_##name); \
    static void test_func_##name()

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error(std::string("Assertion failed: ") + #cond + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            throw std::runtime_error(std::string("Assertion failed: ") + #a + " == " + #b + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

#define ASSERT_NEAR(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            throw std::runtime_error(std::string("Assertion failed: |") + #a + " - " + #b + "| <= " + #eps + \
                                     " (got " + std::to_string(a) + " vs " + std::to_string(b) + ") at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

// Aliases for standard testing ergonomics
#define TEST_CASE(name) KINESIS_TEST_CASE(name)
#define EXPECT_TRUE(cond) ASSERT_TRUE(cond)
#define EXPECT_FALSE(cond) ASSERT_FALSE(cond)
#define EXPECT_EQ(a, b) ASSERT_EQ(a, b)
#define EXPECT_NOT_NULL(ptr) ASSERT_TRUE((ptr) != nullptr)
#define EXPECT_NEAR(a, b, eps) ASSERT_NEAR(a, b, eps)

