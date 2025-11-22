#include <cassert>
#include <cmath>
#include <iostream>
#include "mist/ndarray.hpp"

using namespace mist;

// =============================================================================
// Helper functions
// =============================================================================

bool approx_equal(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

template<typename T, std::size_t N>
bool vec_equal(const vec_t<T, N>& a, const vec_t<T, N>& b, double tol = 1e-10) {
    for (std::size_t i = 0; i < N; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (!approx_equal(a[i], b[i], tol)) return false;
        } else {
            if (a[i] != b[i]) return false;
        }
    }
    return true;
}

// =============================================================================
// lazy_t tests
// =============================================================================

void test_lazy_basic() {
    std::cout << "Testing lazy_t basic... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    auto arr = lazy(space, [](ivec_t<2> idx) { return idx[0] + idx[1]; });

    assert(arr(ivec(0, 0)) == 0);
    assert(arr(ivec(3, 4)) == 7);
    assert(arr(ivec(9, 9)) == 18);

    assert(size(arr) == 100);
    assert(shape(arr)[0] == 10);
    assert(shape(arr)[1] == 10);

    std::cout << "PASSED\n";
}

void test_lazy_with_captures() {
    std::cout << "Testing lazy_t with captures... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    double dx = 0.1;
    double dy = 0.2;

    auto arr = lazy(space, [dx, dy](ivec_t<2> idx) {
        return idx[0] * dx + idx[1] * dy;
    });

    assert(approx_equal(arr(ivec(0, 0)), 0.0));
    assert(approx_equal(arr(ivec(5, 5)), 0.5 + 1.0));
    assert(approx_equal(arr(ivec(9, 9)), 0.9 + 1.8));

    std::cout << "PASSED\n";
}

// =============================================================================
// cached_t tests
// =============================================================================

void test_cached_basic() {
    std::cout << "Testing cached_t basic... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    cached_t<double, 2> arr(space, memory::host);

    // Write
    arr(ivec(0, 0)) = 1.0;
    arr(ivec(5, 5)) = 2.0;
    arr(ivec(9, 9)) = 3.0;

    // Read
    assert(approx_equal(arr(ivec(0, 0)), 1.0));
    assert(approx_equal(arr(ivec(5, 5)), 2.0));
    assert(approx_equal(arr(ivec(9, 9)), 3.0));

    // data() access
    assert(data(arr) != nullptr);
    assert(location(arr) == memory::host);

    std::cout << "PASSED\n";
}

void test_cached_move() {
    std::cout << "Testing cached_t move... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    cached_t<double, 2> arr1(space, memory::host);
    arr1(ivec(5, 5)) = 42.0;

    // Move construct
    cached_t<double, 2> arr2(std::move(arr1));
    assert(approx_equal(arr2(ivec(5, 5)), 42.0));

    // Move assign
    cached_t<double, 2> arr3(space, memory::host);
    arr3 = std::move(arr2);
    assert(approx_equal(arr3(ivec(5, 5)), 42.0));

    std::cout << "PASSED\n";
}

void test_cached_with_offset() {
    std::cout << "Testing cached_t with non-zero start... ";

    auto space = index_space(ivec(5, 10), uvec(10, 10));
    cached_t<double, 2> arr(space, memory::host);

    arr(ivec(5, 10)) = 1.0;
    arr(ivec(14, 19)) = 2.0;

    assert(approx_equal(arr(ivec(5, 10)), 1.0));
    assert(approx_equal(arr(ivec(14, 19)), 2.0));

    std::cout << "PASSED\n";
}

// =============================================================================
// cached_view_t tests
// =============================================================================

void test_cached_view() {
    std::cout << "Testing cached_view_t... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    double buffer[100] = {0};

    cached_view_t<double, 2> view(space, buffer);

    view(ivec(3, 4)) = 42.0;
    assert(approx_equal(view(ivec(3, 4)), 42.0));
    assert(approx_equal(buffer[3 * 10 + 4], 42.0));

    std::cout << "PASSED\n";
}

// =============================================================================
// cached_vec_t tests
// =============================================================================

void test_cached_vec_aos() {
    std::cout << "Testing cached_vec_t AoS... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    cached_vec_t<double, 3, 2, layout::aos> arr(space, memory::host);

    vec_t<double, 3> v = {1.0, 2.0, 3.0};
    arr(ivec(5, 5)) = v;

    vec_t<double, 3> loaded = arr(ivec(5, 5));
    assert(vec_equal(v, loaded));

    std::cout << "PASSED\n";
}

void test_cached_vec_soa() {
    std::cout << "Testing cached_vec_t SoA... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    cached_vec_t<double, 3, 2, layout::soa> arr(space, memory::host);

    vec_t<double, 3> v = {1.0, 2.0, 3.0};
    arr(ivec(5, 5)) = v;

    vec_t<double, 3> loaded = arr(ivec(5, 5));
    assert(vec_equal(v, loaded));

    // Verify SoA memory layout
    // Component 0 at offset 0*100 + 55 = 55
    // Component 1 at offset 1*100 + 55 = 155
    // Component 2 at offset 2*100 + 55 = 255
    assert(approx_equal(data(arr)[55], 1.0));
    assert(approx_equal(data(arr)[155], 2.0));
    assert(approx_equal(data(arr)[255], 3.0));

    std::cout << "PASSED\n";
}

void test_cached_vec_generic() {
    std::cout << "Testing cached_vec_t generic access... ";

    auto sp = index_space(ivec(0, 0), uvec(10, 10));

    // This function works with both layouts
    auto fill_array = [](auto& arr) {
        for (auto idx : space(arr)) {
            arr(idx) = dvec(1.0, 2.0, 3.0);
        }
    };

    cached_vec_t<double, 3, 2, layout::aos> aos(sp, memory::host);
    cached_vec_t<double, 3, 2, layout::soa> soa(sp, memory::host);

    fill_array(aos);
    fill_array(soa);

    // Both should have same values
    for (auto idx : sp) {
        vec_t<double, 3> aos_val = aos(idx);
        vec_t<double, 3> soa_val = soa(idx);
        assert(vec_equal(aos_val, soa_val));
    }

    std::cout << "PASSED\n";
}

// =============================================================================
// map tests
// =============================================================================

void test_map_basic() {
    std::cout << "Testing map basic... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    cached_t<double, 2> arr(space, memory::host);

    for (auto idx : space) {
        arr(idx) = idx[0] + idx[1];
    }

    auto doubled = map(arr, [](double x) { return 2.0 * x; });

    assert(approx_equal(doubled(ivec(0, 0)), 0.0));
    assert(approx_equal(doubled(ivec(3, 4)), 14.0));
    assert(approx_equal(doubled(ivec(9, 9)), 36.0));

    std::cout << "PASSED\n";
}

void test_map_rvalue() {
    std::cout << "Testing map rvalue... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));

    // map from a temporary lazy array
    auto result = map(
        lazy(space, [](ivec_t<2> idx) { return static_cast<double>(idx[0] + idx[1]); }),
        [](double x) { return x * x; }
    );

    assert(approx_equal(result(ivec(0, 0)), 0.0));
    assert(approx_equal(result(ivec(3, 4)), 49.0));

    std::cout << "PASSED\n";
}

void test_map_chained() {
    std::cout << "Testing map chained... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    auto arr = lazy(space, [](ivec_t<2> idx) { return static_cast<double>(idx[0] + idx[1]); });

    auto result = map(map(arr, [](double x) { return x + 1; }), [](double x) { return x * 2; });

    // (0 + 1) * 2 = 2
    assert(approx_equal(result(ivec(0, 0)), 2.0));
    // (7 + 1) * 2 = 16
    assert(approx_equal(result(ivec(3, 4)), 16.0));

    std::cout << "PASSED\n";
}

// =============================================================================
// cache tests
// =============================================================================

void test_cache_lazy() {
    std::cout << "Testing cache from lazy... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    auto lazy_arr = lazy(space, [](ivec_t<2> idx) { return static_cast<double>(idx[0] * idx[1]); });

    auto cached = cache(lazy_arr, memory::host, exec::cpu);

    assert(approx_equal(cached(ivec(0, 0)), 0.0));
    assert(approx_equal(cached(ivec(3, 4)), 12.0));
    assert(approx_equal(cached(ivec(9, 9)), 81.0));

    std::cout << "PASSED\n";
}

void test_cache_cached() {
    std::cout << "Testing cache from cached (copy)... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    cached_t<double, 2> original(space, memory::host);

    for (auto idx : space) {
        original(idx) = idx[0] + idx[1];
    }

    auto copy = cache(original, memory::host, exec::cpu);

    // Verify copy
    for (auto idx : space) {
        assert(approx_equal(copy(idx), original(idx)));
    }

    // Modify original, copy should be unchanged
    original(ivec(5, 5)) = 999.0;
    assert(approx_equal(copy(ivec(5, 5)), 10.0));

    std::cout << "PASSED\n";
}

// =============================================================================
// extract tests
// =============================================================================

void test_extract_basic() {
    std::cout << "Testing extract basic... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    auto arr = lazy(space, [](ivec_t<2> idx) { return idx[0] * 10 + idx[1]; });

    auto subspace = index_space(ivec(2, 3), uvec(4, 5));
    auto sub = extract(arr, subspace);

    assert(size(sub) == 20);
    assert(sub(ivec(2, 3)) == 23);
    assert(sub(ivec(5, 7)) == 57);

    std::cout << "PASSED\n";
}

// =============================================================================
// insert tests
// =============================================================================

void test_insert_basic() {
    std::cout << "Testing insert basic... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    auto arr = lazy(space, [](ivec_t<2>) { return 0; });

    auto subspace = index_space(ivec(2, 2), uvec(6, 6));
    auto patch = lazy(subspace, [](ivec_t<2>) { return 1; });

    auto result = insert(arr, patch);

    // Outside subspace: 0
    assert(result(ivec(0, 0)) == 0);
    assert(result(ivec(1, 1)) == 0);
    assert(result(ivec(9, 9)) == 0);

    // Inside subspace: 1
    assert(result(ivec(2, 2)) == 1);
    assert(result(ivec(5, 5)) == 1);
    assert(result(ivec(7, 7)) == 1);

    std::cout << "PASSED\n";
}

void test_insert_extract_pattern() {
    std::cout << "Testing insert/extract pattern... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    cached_t<double, 2> arr(space, memory::host);

    for (auto idx : space) {
        arr(idx) = idx[0] + idx[1];
    }

    // Extract interior, double it, insert back
    auto inner = index_space(ivec(1, 1), uvec(8, 8));
    auto result = cache(
        insert(arr, map(extract(arr, inner), [](double x) { return 2.0 * x; })),
        memory::host, exec::cpu
    );

    // Boundary unchanged
    assert(approx_equal(result(ivec(0, 0)), 0.0));
    assert(approx_equal(result(ivec(0, 5)), 5.0));
    assert(approx_equal(result(ivec(9, 9)), 18.0));

    // Interior doubled
    assert(approx_equal(result(ivec(1, 1)), 4.0));   // (1+1)*2 = 4
    assert(approx_equal(result(ivec(5, 5)), 20.0));  // (5+5)*2 = 20
    assert(approx_equal(result(ivec(8, 8)), 32.0));  // (8+8)*2 = 32

    std::cout << "PASSED\n";
}

// =============================================================================
// Constructor tests
// =============================================================================

void test_zeros() {
    std::cout << "Testing zeros... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    auto arr = zeros<double>(space);

    for (auto idx : space) {
        assert(approx_equal(arr(idx), 0.0));
    }

    std::cout << "PASSED\n";
}

void test_ones() {
    std::cout << "Testing ones... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    auto arr = ones<double>(space);

    for (auto idx : space) {
        assert(approx_equal(arr(idx), 1.0));
    }

    std::cout << "PASSED\n";
}

void test_fill() {
    std::cout << "Testing fill... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    auto arr = fill<double>(space, 3.14);

    for (auto idx : space) {
        assert(approx_equal(arr(idx), 3.14));
    }

    std::cout << "PASSED\n";
}

void test_indices() {
    std::cout << "Testing indices... ";

    auto space = index_space(ivec(5, 10), uvec(10, 10));
    auto arr = indices(space);

    assert(arr(ivec(5, 10)) == ivec(5, 10));
    assert(arr(ivec(10, 15)) == ivec(10, 15));

    std::cout << "PASSED\n";
}

void test_offsets() {
    std::cout << "Testing offsets... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    auto arr = offsets(space);

    assert(arr(ivec(0, 0)) == 0);
    assert(arr(ivec(0, 1)) == 1);
    assert(arr(ivec(1, 0)) == 10);
    assert(arr(ivec(9, 9)) == 99);

    std::cout << "PASSED\n";
}

void test_range() {
    std::cout << "Testing range... ";

    auto arr = range(10);

    assert(size(arr) == 10);
    assert(arr(ivec(0)) == 0);
    assert(arr(ivec(5)) == 5);
    assert(arr(ivec(9)) == 9);

    std::cout << "PASSED\n";
}

void test_linspace() {
    std::cout << "Testing linspace... ";

    // Endpoint inclusive (default)
    auto arr1 = linspace(0.0, 1.0, 11);
    assert(size(arr1) == 11);
    assert(approx_equal(arr1(ivec(0)), 0.0));
    assert(approx_equal(arr1(ivec(5)), 0.5));
    assert(approx_equal(arr1(ivec(10)), 1.0));

    // Endpoint exclusive
    auto arr2 = linspace(0.0, 1.0, 10, false);
    assert(size(arr2) == 10);
    assert(approx_equal(arr2(ivec(0)), 0.0));
    assert(approx_equal(arr2(ivec(5)), 0.5));

    std::cout << "PASSED\n";
}

void test_coords() {
    std::cout << "Testing coords... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    auto origin = dvec(0.0, 0.0);
    auto delta = dvec(0.1, 0.2);

    auto arr = coords(space, origin, delta);

    auto c00 = arr(ivec(0, 0));
    assert(approx_equal(c00[0], 0.0));
    assert(approx_equal(c00[1], 0.0));

    auto c55 = arr(ivec(5, 5));
    assert(approx_equal(c55[0], 0.5));
    assert(approx_equal(c55[1], 1.0));

    auto c99 = arr(ivec(9, 9));
    assert(approx_equal(c99[0], 0.9));
    assert(approx_equal(c99[1], 1.8));

    std::cout << "PASSED\n";
}

// =============================================================================
// safe_at tests
// =============================================================================

void test_safe_at_host() {
    std::cout << "Testing safe_at on host... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));
    cached_t<double, 2> arr(space, memory::host);

    safe_at(arr, ivec(5, 5)) = 42.0;
    double val = safe_at(arr, ivec(5, 5));

    assert(approx_equal(val, 42.0));

    std::cout << "PASSED\n";
}

// =============================================================================
// 3D tests
// =============================================================================

void test_3d_array() {
    std::cout << "Testing 3D array... ";

    auto space = index_space(ivec(0, 0, 0), uvec(10, 10, 10));
    auto arr = lazy(space, [](ivec_t<3> idx) {
        return idx[0] * 100 + idx[1] * 10 + idx[2];
    });

    assert(arr(ivec(0, 0, 0)) == 0);
    assert(arr(ivec(1, 2, 3)) == 123);
    assert(arr(ivec(9, 9, 9)) == 999);

    auto cached = cache(arr, memory::host, exec::cpu);
    assert(cached(ivec(5, 5, 5)) == 555);

    std::cout << "PASSED\n";
}

// =============================================================================
// NdArray concept tests
// =============================================================================

void test_concept_satisfaction() {
    std::cout << "Testing NdArray concept satisfaction... ";

    auto space = index_space(ivec(0, 0), uvec(10, 10));

    // lazy_t satisfies NdArray
    auto lazy_arr = lazy(space, [](ivec_t<2>) { return 0.0; });
    static_assert(NdArray<decltype(lazy_arr)>);

    // cached_t satisfies NdArray and CachedNdArray
    cached_t<double, 2> cached_arr(space, memory::host);
    static_assert(NdArray<decltype(cached_arr)>);
    static_assert(CachedNdArray<decltype(cached_arr)>);
    static_assert(WritableNdArray<decltype(cached_arr)>);

    // cached_view_t satisfies NdArray and CachedNdArray
    double buffer[100];
    cached_view_t<double, 2> view(space, buffer);
    static_assert(NdArray<decltype(view)>);
    static_assert(CachedNdArray<decltype(view)>);

    // cached_vec_t satisfies NdArray
    cached_vec_t<double, 3, 2> vec_arr(space, memory::host);
    static_assert(NdArray<decltype(vec_arr)>);

    std::cout << "PASSED\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== NdArray Tests ===\n\n";

    std::cout << "--- lazy_t ---\n";
    test_lazy_basic();
    test_lazy_with_captures();

    std::cout << "\n--- cached_t ---\n";
    test_cached_basic();
    test_cached_move();
    test_cached_with_offset();

    std::cout << "\n--- cached_view_t ---\n";
    test_cached_view();

    std::cout << "\n--- cached_vec_t ---\n";
    test_cached_vec_aos();
    test_cached_vec_soa();
    test_cached_vec_generic();

    std::cout << "\n--- map ---\n";
    test_map_basic();
    test_map_rvalue();
    test_map_chained();

    std::cout << "\n--- cache ---\n";
    test_cache_lazy();
    test_cache_cached();

    std::cout << "\n--- extract ---\n";
    test_extract_basic();

    std::cout << "\n--- insert ---\n";
    test_insert_basic();
    test_insert_extract_pattern();

    std::cout << "\n--- constructors ---\n";
    test_zeros();
    test_ones();
    test_fill();
    test_indices();
    test_offsets();
    test_range();
    test_linspace();
    test_coords();

    std::cout << "\n--- safe_at ---\n";
    test_safe_at_host();

    std::cout << "\n--- 3D ---\n";
    test_3d_array();

    std::cout << "\n--- concepts ---\n";
    test_concept_satisfaction();

    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
