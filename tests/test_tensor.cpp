#include <cassert>
#include <cmath>
#include <iostream>
#include "mist/experimental/tensor.hpp"

using namespace mist::tensor;

int main()
{
    // Coordinate symbols
    constexpr symb<0> t;
    constexpr symb<1> r;
    constexpr symb<2> theta;
    constexpr symb<3> phi;

    // Test basic evaluation
    auto f = r * r;
    assert(f(0.0, 3.0, 0.0, 0.0) == 9.0);

    // Test string output
    assert(f("t", "r", "θ", "φ") == "(r * r)");

    // Test trig functions
    auto g = sin(theta) * sin(theta);
    assert(std::abs(g(0.0, 1.0, M_PI / 2, 0.0) - 1.0) < 1e-10);

    // Test canonical ordering: x * y == y * x (same type)
    auto ab = t * r;
    auto ba = r * t;
    static_assert(std::same_as<decltype(ab), decltype(ba)>);

    // Test tensor product symmetry
    static_assert(std::same_as<decltype(tens(dx<0>{}, dx<1>{})),
                               decltype(tens(dx<1>{}, dx<0>{}))>);

    // Build Schwarzschild metric
    constexpr param<"M"> M;
    auto schwarz_f = one - two * M / r;

    auto metric = neg_one * schwarz_f * tens(d(t), d(t))
                + (one / schwarz_f) * tens(d(r), d(r))
                + r * r * tens(d(theta), d(theta))
                + r * r * sin(theta) * sin(theta) * tens(d(phi), d(phi));

    // Verify diagonal
    static_assert(Diagonal4x4<decltype(metric)>);

    // Verify symmetric
    static_assert(Symmetric4x4<decltype(metric)>);

    // Extract components
    auto g_00 = metric.get<0, 0>();
    auto g_11 = metric.get<1, 1>();
    auto g_22 = metric.get<2, 2>();
    auto g_01 = metric.get<0, 1>();

    // Off-diagonal is zero type
    static_assert(is_zero_v<decltype(g_01)>);

    // Symmetry: get<0,1> == get<1,0>
    static_assert(std::same_as<decltype(metric.get<0,1>()),
                               decltype(metric.get<1,0>())>);

    // Test inverse metric
    auto g_inv = inv(metric);
    auto g_inv_00 = g_inv.get<0, 0>();
    auto g_inv_11 = g_inv.get<1, 1>();
    static_assert(is_zero_v<decltype(g_inv.get<0, 1>())>);

    // Test partial derivatives
    auto dr_r = partial<1>(r);
    static_assert(std::same_as<decltype(dr_r), lit<1>>);

    auto dt_r = partial<0>(r);
    static_assert(std::same_as<decltype(dt_r), lit<0>>);

    // Print metric symbolically
    std::cout << "Schwarzschild metric:\n";
    std::cout << "g_00 = " << g_00("t", "r", "θ", "φ") << "\n";
    std::cout << "g_11 = " << g_11("t", "r", "θ", "φ") << "\n";
    std::cout << "g_22 = " << g_22("t", "r", "θ", "φ") << "\n";
    std::cout << "g_33 = " << metric.get<3,3>()("t", "r", "θ", "φ") << "\n";

    std::cout << "\nAll tests passed.\n";
    return 0;
}
