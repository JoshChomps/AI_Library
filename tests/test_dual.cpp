#include "test_harness.hpp"
#include "kinesis/autograd/dual.hpp"
#include <cmath>
#include <type_traits>

using namespace kinesis::autograd;

TEST_CASE(DualArithmeticBasic) {
    Dual<float> a(3.0f, 1.0f);
    Dual<float> b(2.0f, 0.0f);

    // Addition
    auto c = a + b;
    EXPECT_NEAR(c.val, 5.0f, 1e-5f);
    EXPECT_NEAR(c.dot, 1.0f, 1e-5f);

    // Subtraction
    auto d = a - b;
    EXPECT_NEAR(d.val, 1.0f, 1e-5f);
    EXPECT_NEAR(d.dot, 1.0f, 1e-5f);

    // Multiplication
    auto m = a * b;
    EXPECT_NEAR(m.val, 6.0f, 1e-5f);
    EXPECT_NEAR(m.dot, 2.0f, 1e-5f); // (a*b)' = a'*b + a*b' = 1*2 + 3*0 = 2

    // Division
    auto div = a / b;
    EXPECT_NEAR(div.val, 1.5f, 1e-5f);
    EXPECT_NEAR(div.dot, 0.5f, 1e-5f); // (1*2 - 3*0) / 4 = 2/4 = 0.5

    // Scalar operations
    auto s_add = a + 4.0f;
    EXPECT_NEAR(s_add.val, 7.0f, 1e-5f);
    EXPECT_NEAR(s_add.dot, 1.0f, 1e-5f);

    auto s_left_add = 4.0f + a;
    EXPECT_NEAR(s_left_add.val, 7.0f, 1e-5f);
    EXPECT_NEAR(s_left_add.dot, 1.0f, 1e-5f);

    auto s_mul = a * 2.0f;
    EXPECT_NEAR(s_mul.val, 6.0f, 1e-5f);
    EXPECT_NEAR(s_mul.dot, 2.0f, 1e-5f);

    auto s_div = a / 2.0f;
    EXPECT_NEAR(s_div.val, 1.5f, 1e-5f);
    EXPECT_NEAR(s_div.dot, 0.5f, 1e-5f);

    auto s_inv = 6.0f / a;
    EXPECT_NEAR(s_inv.val, 2.0f, 1e-5f);
    // (6/a)' = -6*a' / a^2 = -6*1 / 9 = -2/3
    EXPECT_NEAR(s_inv.dot, -2.0f / 3.0f, 1e-5f);

    // Unary minus
    auto neg = -a;
    EXPECT_NEAR(neg.val, -3.0f, 1e-5f);
    EXPECT_NEAR(neg.dot, -1.0f, 1e-5f);
}

TEST_CASE(DualAlgebraicRules) {
    // Product Rule: f(x) = x * (2x + 1) = 2x^2 + x -> f'(x) = 4x + 1
    // At x = 3: f(3) = 2*9 + 3 = 21, f'(3) = 4*3 + 1 = 13
    auto x = Dual<float>::variable(3.0f);
    auto f_prod = x * (2.0f * x + 1.0f);
    EXPECT_NEAR(f_prod.val, 21.0f, 1e-5f);
    EXPECT_NEAR(f_prod.dot, 13.0f, 1e-5f);

    // Quotient Rule: f(x) = x / (x + 2) -> f'(x) = 2 / (x + 2)^2
    // At x = 2: f(2) = 2/4 = 0.5, f'(2) = 2/16 = 0.125
    auto x2 = Dual<float>::variable(2.0f);
    auto f_quot = x2 / (x2 + 2.0f);
    EXPECT_NEAR(f_quot.val, 0.5f, 1e-5f);
    EXPECT_NEAR(f_quot.dot, 0.125f, 1e-5f);

    // Polynomial: f(x) = 3x^3 - 5x^2 + 7x - 4 -> f'(x) = 9x^2 - 10x + 7
    // At x = 2: f(2) = 24 - 20 + 14 - 4 = 14, f'(2) = 36 - 20 + 7 = 23
    auto poly = 3.0f * (x2 * x2 * x2) - 5.0f * (x2 * x2) + 7.0f * x2 - 4.0f;
    EXPECT_NEAR(poly.val, 14.0f, 1e-5f);
    EXPECT_NEAR(poly.dot, 23.0f, 1e-5f);
}

TEST_CASE(DualTranscendentalFunctions) {
    auto x = Dual<float>::variable(0.5f);

    // sin(x) -> cos(x)
    auto s = sin(x);
    EXPECT_NEAR(s.val, std::sin(0.5f), 1e-5f);
    EXPECT_NEAR(s.dot, std::cos(0.5f), 1e-5f);

    // cos(x) -> -sin(x)
    auto c = cos(x);
    EXPECT_NEAR(c.val, std::cos(0.5f), 1e-5f);
    EXPECT_NEAR(c.dot, -std::sin(0.5f), 1e-5f);

    // tan(x) -> 1 + tan^2(x)
    auto t = tan(x);
    float t_val = std::tan(0.5f);
    EXPECT_NEAR(t.val, t_val, 1e-5f);
    EXPECT_NEAR(t.dot, 1.0f + t_val * t_val, 1e-5f);

    // exp(x) -> exp(x)
    auto e = exp(x);
    EXPECT_NEAR(e.val, std::exp(0.5f), 1e-5f);
    EXPECT_NEAR(e.dot, std::exp(0.5f), 1e-5f);

    // log(x) -> 1/x
    auto l = log(x);
    EXPECT_NEAR(l.val, std::log(0.5f), 1e-5f);
    EXPECT_NEAR(l.dot, 1.0f / 0.5f, 1e-5f);

    // sqrt(x) -> 1 / (2*sqrt(x))
    auto sq = sqrt(x);
    EXPECT_NEAR(sq.val, std::sqrt(0.5f), 1e-5f);
    EXPECT_NEAR(sq.dot, 1.0f / (2.0f * std::sqrt(0.5f)), 1e-5f);

    // Composite Chain Rule: f(x) = sin(x^2) -> f'(x) = 2x * cos(x^2)
    auto chain_x = Dual<float>::variable(1.2f);
    auto chain_res = sin(chain_x * chain_x);
    EXPECT_NEAR(chain_res.val, std::sin(1.2f * 1.2f), 1e-5f);
    EXPECT_NEAR(chain_res.dot, 2.0f * 1.2f * std::cos(1.2f * 1.2f), 1e-5f);

    // Activation Functions: tanh, sigmoid, relu
    auto act_x = Dual<float>::variable(0.7f);
    auto th = tanh(act_x);
    float th_expected = std::tanh(0.7f);
    EXPECT_NEAR(th.val, th_expected, 1e-5f);
    EXPECT_NEAR(th.dot, 1.0f - th_expected * th_expected, 1e-5f);

    auto sig = sigmoid(act_x);
    float s_exp = 1.0f / (1.0f + std::exp(-0.7f));
    EXPECT_NEAR(sig.val, s_exp, 1e-5f);
    EXPECT_NEAR(sig.dot, s_exp * (1.0f - s_exp), 1e-5f);

    auto r_pos = relu(act_x);
    EXPECT_NEAR(r_pos.val, 0.7f, 1e-5f);
    EXPECT_NEAR(r_pos.dot, 1.0f, 1e-5f);

    auto r_neg = relu(Dual<float>::variable(-0.5f));
    EXPECT_NEAR(r_neg.val, 0.0f, 1e-5f);
    EXPECT_NEAR(r_neg.dot, 0.0f, 1e-5f);
}

TEST_CASE(HyperDualMultiTangentGradients) {
    // Test function: f(x, y, z) = x^2 * y + y * sin(z)
    // Partial derivatives:
    // df/dx = 2 * x * y
    // df/dy = x^2 + sin(z)
    // df/dz = y * cos(z)
    // Evaluated at x=2.0, y=3.0, z=0.0:
    // f = 4 * 3 + 3 * 0 = 12.0
    // df/dx = 2 * 2 * 3 = 12.0
    // df/dy = 4 + 0 = 4.0
    // df/dz = 3 * cos(0) = 3.0

    auto x = HyperDual<float, 3>::coordinate(2.0f, 0);
    auto y = HyperDual<float, 3>::coordinate(3.0f, 1);
    auto z = HyperDual<float, 3>::coordinate(0.0f, 2);

    auto f = (x * x) * y + y * sin(z);

    EXPECT_NEAR(f.val, 12.0f, 1e-5f);
    EXPECT_NEAR(f.grad[0], 12.0f, 1e-5f); // df/dx
    EXPECT_NEAR(f.grad[1], 4.0f, 1e-5f);  // df/dy
    EXPECT_NEAR(f.grad[2], 3.0f, 1e-5f);  // df/dz
}

TEST_CASE(HyperDualRoboticKinematicsJacobian) {
    // Forward Kinematics of a 2-link planar robotic manipulator:
    // Lengths: L1 = 1.0, L2 = 1.0
    // End effector coordinates:
    //   x_ee = L1 * cos(th1) + L2 * cos(th1 + th2)
    //   y_ee = L1 * sin(th1) + L2 * sin(th1 + th2)
    //
    // Joint angles: th1 = 0, th2 = pi/2
    // x_ee = cos(0) + cos(pi/2) = 1.0 + 0.0 = 1.0
    // y_ee = sin(0) + sin(pi/2) = 0.0 + 1.0 = 1.0
    //
    // Jacobian rows:
    // J_11 = dx/dth1 = -L1*sin(th1) - L2*sin(th1+th2) = 0 - 1 = -1.0
    // J_12 = dx/dth2 = -L2*sin(th1+th2) = -1.0
    // J_21 = dy/dth1 = L1*cos(th1) + L2*cos(th1+th2) = 1.0 + 0.0 = 1.0
    // J_22 = dy/dth2 = L2*cos(th1+th2) = 0.0

    const float pi = 3.14159265358979323846f;
    auto th1 = HyperDual<float, 2>::coordinate(0.0f, 0);
    auto th2 = HyperDual<float, 2>::coordinate(pi / 2.0f, 1);

    auto x_ee = cos(th1) + cos(th1 + th2);
    auto y_ee = sin(th1) + sin(th1 + th2);

    EXPECT_NEAR(x_ee.val, 1.0f, 1e-4f);
    EXPECT_NEAR(x_ee.grad[0], -1.0f, 1e-4f); // J_11
    EXPECT_NEAR(x_ee.grad[1], -1.0f, 1e-4f); // J_12

    EXPECT_NEAR(y_ee.val, 1.0f, 1e-4f);
    EXPECT_NEAR(y_ee.grad[0], 1.0f, 1e-4f);  // J_21
    EXPECT_NEAR(y_ee.grad[1], 0.0f, 1e-4f);  // J_22
}

// Defect 11: the type formerly called HyperDual is a first-order multi-tangent number (Ceres'
// Jet), so second derivatives have to come from nesting. These pin both the rename and the
// nesting, which the old FloatingPoint constraint rejected outright.
TEST_CASE(Dual_NestingYieldsSecondDerivatives) {
    static_assert(std::is_same_v<HyperDual<float, 3>, Jet<float, 3>>,
                  "HyperDual must remain a spelling of Jet");

    // f(x) = sin(x) * exp(x);  f' = (cos x + sin x) e^x;  f'' = 2 cos(x) e^x
    using Nested = Dual<Dual<float>>;
    const float a = 1.3f;

    Nested x;
    x.val = Dual<float>(a, 1.0f); // inner tangent seeds d/dx
    x.dot = Dual<float>(1.0f, 0.0f); // outer tangent seeds d/dx

    Nested f = sin(x) * exp(x);

    EXPECT_NEAR(f.val.val, std::sin(a) * std::exp(a), 1e-5f);
    EXPECT_NEAR(f.val.dot, (std::cos(a) + std::sin(a)) * std::exp(a), 1e-5f);
    EXPECT_NEAR(f.dot.dot, 2.0f * std::cos(a) * std::exp(a), 1e-5f);
}

TEST_CASE(Dual_Atan2IsQuadrantCorrectAndFiniteAtVerticalAxis) {
    const float pi = 3.14159265358979323846f;

    // atan(y/x) is undefined at x == 0; atan2's tangent (x*dy - y*dx)/(x^2 + y^2) is not.
    auto up = atan2(Dual<float>::constant(1.0f), Dual<float>::variable(0.0f));
    EXPECT_NEAR(up.val, pi / 2.0f, 1e-5f);
    EXPECT_NEAR(up.dot, -1.0f, 1e-5f);

    // Second quadrant: atan(y/x) would report -pi/4 here.
    auto q2 = atan2(Dual<float>::constant(1.0f), Dual<float>::constant(-1.0f));
    EXPECT_NEAR(q2.val, 3.0f * pi / 4.0f, 1e-5f);

    // d/dx asin(x) = 1/sqrt(1 - x^2), d/dx acos(x) = -1/sqrt(1 - x^2)
    auto as = asin(Dual<float>::variable(0.5f));
    EXPECT_NEAR(as.val, std::asin(0.5f), 1e-5f);
    EXPECT_NEAR(as.dot, 1.0f / std::sqrt(1.0f - 0.25f), 1e-5f);
    auto ac = acos(Dual<float>::variable(0.5f));
    EXPECT_NEAR(ac.dot, -1.0f / std::sqrt(1.0f - 0.25f), 1e-5f);
}

TEST_CASE(Dual_PowWithZeroExponentDoesNotProduceNaN) {
    // x^0 is constant 1, but the general power rule evaluates 0 * pow(0, -1) = 0 * inf at x = 0.
    auto p = pow(Dual<float>::variable(0.0f), 0.0f);
    EXPECT_NEAR(p.val, 1.0f, 1e-6f);
    EXPECT_FALSE(std::isnan(p.dot));
    EXPECT_NEAR(p.dot, 0.0f, 1e-6f);

    auto q = pow(Dual<float>::variable(2.0f), 3.0f);
    EXPECT_NEAR(q.val, 8.0f, 1e-5f);
    EXPECT_NEAR(q.dot, 12.0f, 1e-5f); // 3 * 2^2
}
