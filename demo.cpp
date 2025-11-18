#include <iostream>
#include <cmath>
#include <vector>
#include "mist.hpp"

using namespace mist;

// Helper function to print vectors
template<typename T, std::size_t S>
void print_vec(const vec_t<T, S>& v, const char* name) {
    std::cout << name << " = [";
    for (std::size_t i = 0; i < S; ++i) {
        std::cout << v[i];
        if (i < S - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "=== Mist Library Demo ===\n\n";
    
    // === Basic Construction ===
    std::cout << "1. Vector Construction:\n";
    auto v1 = vec(1.0, 2.0, 3.0);
    auto v2 = dvec(4.0, 5.0, 6.0);
    auto v3 = ivec(1, 2, 3);
    auto v4 = uvec(10, 20, 30);
    
    print_vec(v1, "v1 (auto)");
    print_vec(v2, "v2 (dvec)");
    print_vec(v3, "v3 (ivec)");
    print_vec(v4, "v4 (uvec)");
    std::cout << "\n";
    
    // === Range Construction ===
    std::cout << "2. Range Construction:\n";
    auto r5 = range<5>();
    print_vec(r5, "range<5>");
    std::cout << "\n";
    
    // === Vector Arithmetic ===
    std::cout << "3. Vector Arithmetic:\n";
    auto sum_v = v1 + v2;
    auto diff = v2 - v1;
    auto scaled = v1 * 2.0;
    auto scaled2 = 3.0 * v1;
    auto divided = v1 / 2.0;
    
    print_vec(sum_v, "v1 + v2");
    print_vec(diff, "v2 - v1");
    print_vec(scaled, "v1 * 2.0");
    print_vec(scaled2, "3.0 * v1");
    print_vec(divided, "v1 / 2.0");
    std::cout << "\n";
    
    // === Dot Product ===
    std::cout << "4. Dot Product:\n";
    auto dp = dot(v1, v2);
    std::cout << "dot(v1, v2) = " << dp << "\n\n";
    
    // === Map Function ===
    std::cout << "5. Map Function:\n";
    auto squared = map(v1, [](double x) { return x * x; });
    auto sqrt_vec = map(v1, [](double x) { return std::sqrt(x); });
    
    print_vec(squared, "v1 squared");
    print_vec(sqrt_vec, "sqrt(v1)");
    std::cout << "\n";
    
    // === Reduction Functions ===
    std::cout << "6. Reduction Functions:\n";
    auto v_sum = dvec(1.0, 2.0, 3.0, 4.0);
    auto v_prod = dvec(2.0, 3.0, 4.0);
    auto v_bool1 = vec(true, false, true);
    auto v_bool2 = vec(true, true, true);
    
    std::cout << "sum([1, 2, 3, 4]) = " << sum(v_sum) << "\n";
    std::cout << "product([2, 3, 4]) = " << product(v_prod) << "\n";
    std::cout << "any([true, false, true]) = " << any(v_bool1) << "\n";
    std::cout << "all([true, false, true]) = " << all(v_bool1) << "\n";
    std::cout << "all([true, true, true]) = " << all(v_bool2) << "\n\n";
    
    // === Mixed Type Operations ===
    std::cout << "7. Mixed Type Operations:\n";
    auto mixed = v3 + ivec(10, 20, 30);
    auto float_result = dvec(1.5, 2.5, 3.5) * 2;
    
    print_vec(mixed, "ivec + ivec");
    print_vec(float_result, "dvec * int");
    std::cout << "\n";
    
    // === Free Functions ===
    std::cout << "8. Free Functions:\n";
    std::cout << "size(v1) = " << size(v1) << "\n";
    std::cout << "at(v1, 1) = " << at(v1, 1) << "\n";
    std::cout << "data(v1) = " << data(v1) << "\n\n";
    
    // === Range-based For Loop ===
    std::cout << "9. Range-based For Loop:\n";
    std::cout << "Iterating over v1: ";
    for (const auto& x : v1) {
        std::cout << x << " ";
    }
    std::cout << "\n\n";
    
    // === Data Pointer Access ===
    std::cout << "10. Data Pointer Access:\n";
    auto ptr = data(v1);
    std::cout << "data(v1) = " << ptr << "\n";
    std::cout << "data(v1)[0] = " << ptr[0] << "\n";
    std::cout << "data(v1)[1] = " << ptr[1] << "\n\n";
    
    // === Comparison Operators ===
    std::cout << "11. Comparison Operators:\n";
    auto va = ivec(1, 2, 3);
    auto vb = ivec(1, 2, 3);
    auto vc = ivec(1, 2, 4);
    
    std::cout << "va == vb: " << (va == vb) << "\n";
    std::cout << "va == vc: " << (va == vc) << "\n";
    std::cout << "va < vc: " << (va < vc) << "\n\n";
    
    // === Index Space ===
    std::cout << "12. Index Space:\n";
    auto space = index_space(ivec(0, 0), uvec(10, 20));
    
    std::cout << "Index space:\n";
    print_vec(start(space), "  start");
    print_vec(shape(space), "  shape");
    std::cout << "  size = " << size(space) << "\n\n";
    
    // === Contains Function ===
    std::cout << "13. Contains Function:\n";
    auto offset_space = index_space(ivec(2, 4), uvec(10, 20));
    
    auto idx_valid1 = ivec(2, 4);     // Min valid (start)
    auto idx_valid2 = ivec(6, 15);    // Within range
    auto idx_valid3 = ivec(11, 23);   // Max valid (start + shape - 1)
    auto idx_invalid1 = ivec(0, 0);   // Below start
    auto idx_invalid2 = ivec(2, 3);   // Column below start
    auto idx_invalid3 = ivec(1, 10);  // Row below start
    auto idx_invalid4 = ivec(12, 10); // Row beyond end
    auto idx_invalid5 = ivec(5, 24);  // Column beyond end
    
    std::cout << "Space: start = [2, 4], shape = [10, 20]\n";
    std::cout << "(Valid absolute range is [2, 4] to [11, 23])\n";
    std::cout << "contains(space, [2, 4]) = " << contains(offset_space, idx_valid1) << " (min)\n";
    std::cout << "contains(space, [6, 15]) = " << contains(offset_space, idx_valid2) << " (middle)\n";
    std::cout << "contains(space, [11, 23]) = " << contains(offset_space, idx_valid3) << " (max)\n";
    std::cout << "contains(space, [0, 0]) = " << contains(offset_space, idx_invalid1) << " (below start)\n";
    std::cout << "contains(space, [2, 3]) = " << contains(offset_space, idx_invalid2) << " (col < start)\n";
    std::cout << "contains(space, [1, 10]) = " << contains(offset_space, idx_invalid3) << " (row < start)\n";
    std::cout << "contains(space, [12, 10]) = " << contains(offset_space, idx_invalid4) << " (row >= end)\n";
    std::cout << "contains(space, [5, 24]) = " << contains(offset_space, idx_invalid5) << " (col >= end)\n\n";
    
    // === Multi-dimensional Indexing ===
    std::cout << "14. Multi-dimensional Indexing:\n";
    std::cout << "Using space with start = [0, 0], shape = [10, 20]\n";
    auto idx1 = ivec(2, 3);
    auto idx2 = ivec(0, 0);
    auto idx3 = ivec(9, 19);
    
    std::cout << "ndoffset(space, [2, 3]) = " << ndoffset(space, idx1) << " (expected: 2*20 + 3 = 43)\n";
    std::cout << "ndoffset(space, [0, 0]) = " << ndoffset(space, idx2) << "\n";
    std::cout << "ndoffset(space, [9, 19]) = " << ndoffset(space, idx3) << " (expected: 9*20 + 19 = 199)\n";
    
    auto recovered = ndindex(space, 43);
    print_vec(recovered, "ndindex(space, 43)");
    
    std::cout << "\nWith offset space: start = [2, 4], shape = [10, 20]\n";
    auto off_idx1 = ivec(2, 4);   // First element
    auto off_idx2 = ivec(7, 13);  // Offset (5, 9) -> 5*20 + 9 = 109
    
    std::cout << "ndoffset(offset_space, [2, 4]) = " << ndoffset(offset_space, off_idx1) << " (expected: 0)\n";
    std::cout << "ndoffset(offset_space, [7, 13]) = " << ndoffset(offset_space, off_idx2) << " (expected: (7-2)*20 + (13-4) = 109)\n";
    
    auto recovered2 = ndindex(offset_space, 109);
    print_vec(recovered2, "ndindex(offset_space, 109)");
    std::cout << "\n";
    
    // === Buffer Read/Write ===
    std::cout << "15. Buffer Read/Write (Scalars):\n";
    std::vector<double> buffer(size(space), 0.0);
    
    // Write some values
    ndwrite(buffer.data(), space, ivec(0, 0), 1.5);
    ndwrite(buffer.data(), space, ivec(2, 3), 42.0);
    ndwrite(buffer.data(), space, ivec(9, 19), 99.9);
    
    // Read them back
    std::cout << "buffer[0, 0] = " << ndread(buffer.data(), space, ivec(0, 0)) << "\n";
    std::cout << "buffer[2, 3] = " << ndread(buffer.data(), space, ivec(2, 3)) << "\n";
    std::cout << "buffer[9, 19] = " << ndread(buffer.data(), space, ivec(9, 19)) << "\n\n";
    
    // === SoA Buffer Read/Write ===
    std::cout << "16. Struct-of-Arrays (SoA) Read/Write:\n";
    constexpr std::size_t vec_size = 3;
    std::vector<double> soa_buffer(size(space) * vec_size, 0.0);
    
    // Write some 3-vectors
    ndwrite_soa<double, vec_size>(soa_buffer.data(), space, ivec(0, 0), dvec(1.0, 2.0, 3.0));
    ndwrite_soa<double, vec_size>(soa_buffer.data(), space, ivec(1, 1), dvec(4.0, 5.0, 6.0));
    ndwrite_soa<double, vec_size>(soa_buffer.data(), space, ivec(2, 2), dvec(7.0, 8.0, 9.0));
    
    // Read them back
    auto vec_00 = ndread_soa<double, vec_size>(soa_buffer.data(), space, ivec(0, 0));
    auto vec_11 = ndread_soa<double, vec_size>(soa_buffer.data(), space, ivec(1, 1));
    auto vec_22 = ndread_soa<double, vec_size>(soa_buffer.data(), space, ivec(2, 2));
    
    print_vec(vec_00, "soa_buffer[0, 0]");
    print_vec(vec_11, "soa_buffer[1, 1]");
    print_vec(vec_22, "soa_buffer[2, 2]");
    std::cout << "\n";
    
    // === Index Space Iteration ===
    std::cout << "17. Index Space Iteration:\n";
    auto small_space = index_space(ivec(0, 0), uvec(3, 4));
    std::cout << "Iterating over 3x4 space:\n";
    int count = 0;
    for (auto index : small_space) {
        std::cout << "  [" << index[0] << ", " << index[1] << "]";
        if (++count % 4 == 0) std::cout << "\n";
    }
    std::cout << "\n";
    
    // === Parallel Traversal with for_each ===
    std::cout << "18. Parallel Traversal (for_each):\n";
    std::cout << "Using for_each on 3x4 space with different execution policies:\n";
    
    // CPU execution (serial) - default
    std::vector<int> results_cpu(size(small_space), 0);
    for_each(small_space, [&results_cpu](ivec_t<2> index) {
        int linear_idx = index[0] * 4 + index[1];
        results_cpu[linear_idx] = index[0] * 10 + index[1];
    });
    
    // CPU execution (explicit)
    std::vector<int> results_cpu_explicit(size(small_space), 0);
    for_each(small_space, [&results_cpu_explicit](ivec_t<2> index) {
        int linear_idx = index[0] * 4 + index[1];
        results_cpu_explicit[linear_idx] = index[0] * 10 + index[1];
    }, exec::cpu);
    
    std::cout << "CPU (default) results:\n";
    for (std::size_t i = 0; i < 12; ++i) {
        std::cout << "  " << results_cpu[i];
        if ((i + 1) % 4 == 0) std::cout << "\n";
    }
    
    // OpenMP execution (if available)
    std::cout << "OpenMP execution: ";
    try {
        std::vector<int> results_omp(size(small_space), 0);
        for_each(small_space, [&results_omp](ivec_t<2> index) {
            int linear_idx = index[0] * 4 + index[1];
            results_omp[linear_idx] = index[0] * 10 + index[1];
        }, exec::omp);
        
        std::cout << "SUCCESS\n";
        for (std::size_t i = 0; i < 12; ++i) {
            std::cout << "  " << results_omp[i];
            if ((i + 1) % 4 == 0) std::cout << "\n";
        }
    } catch (const std::runtime_error& e) {
        std::cout << "NOT AVAILABLE (" << e.what() << ")\n";
    }
    std::cout << "\n";
    
    // === Map-Reduce ===
    std::cout << "19. Map-Reduce:\n";
    std::cout << "Using map_reduce on 3x4 space:\n";

    // Sum of all linear indices
    auto sum_idx = map_reduce(small_space, 0,
        [](ivec_t<2> idx) { return idx[0] * 4 + idx[1]; },  // map
        [](int a, int b) { return a + b; }                  // reduce
    );
    std::cout << "Sum of linear indices: " << sum_idx << " (expected: 0+1+...+11 = 66)\n";

    // Product of (index + 1) - using std::multiplies
    auto prod = map_reduce(small_space, 1,
        [](ivec_t<2> idx) { return idx[0] + idx[1] + 1; },
        std::multiplies<int>{}
    );
    std::cout << "Product of (row+col+1): " << prod << "\n";

    // Maximum linear index
    auto max_idx = map_reduce(small_space, 0,
        [](ivec_t<2> idx) { return idx[0] * 4 + idx[1]; },
        [](int a, int b) { return std::max(a, b); }
    );
    std::cout << "Maximum linear index: " << max_idx << "\n";

    // OpenMP map-reduce
    std::cout << "OpenMP map-reduce: ";
    try {
        auto sum_omp = map_reduce(small_space, 0,
            [](ivec_t<2> idx) { return idx[0] * 4 + idx[1]; },
            std::plus<int>{},
            exec::omp
        );
        std::cout << "SUCCESS (sum = " << sum_omp << ")\n";
    } catch (const std::runtime_error& e) {
        std::cout << "NOT AVAILABLE (" << e.what() << ")\n";
    }
    std::cout << "\n";

    // === Compile-time Evaluation ===
    std::cout << "20. Compile-time Evaluation:\n";
    constexpr auto cv1 = dvec(1.0, 2.0, 3.0);
    constexpr auto cv2 = dvec(4.0, 5.0, 6.0);
    constexpr auto csum = cv1 + cv2;
    constexpr auto cdot = dot(cv1, cv2);
    constexpr auto cvsum = sum(cv1);
    constexpr auto cvprod = product(dvec(2.0, 3.0, 4.0));
    
    print_vec(csum, "constexpr sum");
    std::cout << "constexpr dot = " << cdot << "\n";
    std::cout << "constexpr sum(cv1) = " << cvsum << "\n";
    std::cout << "constexpr product([2, 3, 4]) = " << cvprod << "\n\n";
    
    // === Higher Dimensions ===
    std::cout << "21. Higher Dimensions:\n";
    auto v5d = dvec(1.0, 2.0, 3.0, 4.0, 5.0);
    auto scaled5d = v5d * 10.0;
    
    print_vec(v5d, "5D vector");
    print_vec(scaled5d, "5D vector * 10");
    std::cout << "\n";
    
    std::cout << "=== Demo Complete ===\n";
    
    return 0;
}
