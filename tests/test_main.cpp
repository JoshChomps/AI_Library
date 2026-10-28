#include "test_harness.hpp"

int main() {
    return kinesis::test::TestRegistry::instance().run_all();
}
