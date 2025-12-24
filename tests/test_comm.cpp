#include <cassert>
#include <iostream>
#include "mist/comm.hpp"

using namespace mist;

void test_build_plan_local_overlap() {
    // Create two overlapping 1D array views
    auto space1 = index_space(ivec(0), uvec(10));   // [0, 10)
    auto space2 = index_space(ivec(5), uvec(10));   // [5, 15)

    auto arr1 = array_t<double, 1>(space1);
    auto arr2 = array_t<double, 1>(space2);

    // Fill arr1 with values
    for (auto idx : space1) {
        arr1(idx) = static_cast<double>(idx[0]);
    }

    // Publications are const views, requests are mutable views
    auto pub_view = view(static_cast<const array_t<double, 1>&>(arr1));
    auto req_view = view(arr2);

    // Build plan: arr1 publishes, arr2 requests
    auto comm = comm_t{};
    auto pubs = std::vector{pub_view};
    auto reqs = std::vector{req_view};

    auto plan = comm.build_plan<
        array_view_t<const double, 1>,
        array_view_t<double, 1>
    >(pubs, reqs);

    assert(plan.local_copies.size() == 1);
    assert(size(plan.local_copies[0].overlap) == 5);  // [5, 10) overlap

    // Execute the exchange
    comm.exchange(plan);

    // Verify the overlapping region was copied
    for (int i = 5; i < 10; ++i) {
        assert(arr2(ivec(i)) == static_cast<double>(i));
    }

    std::cout << "test_build_plan_local_overlap: PASSED\n";
}

void test_combine_local() {
    auto comm = comm_t{};

    auto result = comm.combine(42, [](int a, int b) { return a + b; });
    assert(result == 42);  // No other ranks, returns local value

    std::cout << "test_combine_local: PASSED\n";
}

void test_rank_size() {
    auto comm = comm_t{};

    assert(comm.rank() == 0);
    assert(comm.size() == 1);

    std::cout << "test_rank_size: PASSED\n";
}

void test_intersect() {
    auto a = index_space(ivec(0), uvec(10));   // [0, 10)
    auto b = index_space(ivec(5), uvec(10));   // [5, 15)
    auto c = intersect(a, b);

    assert(start(c)[0] == 5);
    assert(shape(c)[0] == 5);  // [5, 10)

    // Non-overlapping
    auto d = index_space(ivec(20), uvec(5));   // [20, 25)
    auto e = intersect(a, d);
    assert(size(e) == 0);

    std::cout << "test_intersect: PASSED\n";
}

int main() {
    test_intersect();
    test_build_plan_local_overlap();
    test_combine_local();
    test_rank_size();

    std::cout << "All comm tests passed!\n";
    return 0;
}
