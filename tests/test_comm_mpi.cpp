#include <cassert>
#include <iostream>
#include <mpi.h>
#include "mist/comm.hpp"

using namespace mist;

void test_comm_from_mpi() {
    auto comm = comm_t::from_mpi(MPI_COMM_WORLD);

    int expected_rank, expected_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &expected_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &expected_size);

    assert(comm.rank() == expected_rank);
    assert(comm.size() == expected_size);

    if (comm.rank() == 0) {
        std::cout << "test_comm_from_mpi: PASSED (size=" << comm.size() << ")\n";
    }
}

void test_combine_sum() {
    auto comm = comm_t::from_mpi(MPI_COMM_WORLD);

    // Each rank contributes its rank number
    int local_value = comm.rank();
    int result = comm.combine(local_value, [](int a, int b) { return a + b; });

    // Sum of 0 + 1 + ... + (n-1) = n*(n-1)/2
    int expected = comm.size() * (comm.size() - 1) / 2;
    assert(result == expected);

    if (comm.rank() == 0) {
        std::cout << "test_combine_sum: PASSED (sum=" << result << ")\n";
    }
}

void test_combine_min() {
    auto comm = comm_t::from_mpi(MPI_COMM_WORLD);

    double local_value = 100.0 - comm.rank();
    double result = comm.combine(local_value, [](double a, double b) { return a < b ? a : b; });

    // Min should be 100 - (size - 1)
    double expected = 100.0 - (comm.size() - 1);
    assert(result == expected);

    if (comm.rank() == 0) {
        std::cout << "test_combine_min: PASSED (min=" << result << ")\n";
    }
}

void test_exchange_1d() {
    auto comm = comm_t::from_mpi(MPI_COMM_WORLD);
    int rank = comm.rank();
    int size = comm.size();

    // Each rank owns a contiguous chunk of a global [0, 100) domain
    int chunk_size = 100 / size;
    int my_start = rank * chunk_size;
    int my_end = (rank == size - 1) ? 100 : (rank + 1) * chunk_size;
    int my_size = my_end - my_start;

    // Create arrays with 2 ghost cells on each side
    int num_ghosts = 2;
    auto interior = index_space(ivec(my_start), uvec(my_size));
    auto with_ghosts = expand(interior, num_ghosts);

    auto arr = array_t<double, 1>(with_ghosts);

    // Fill interior with rank-based values
    for (auto idx : interior) {
        arr(idx) = static_cast<double>(idx[0]);
    }

    // Create views
    auto pub_view = view(static_cast<const array_t<double, 1>&>(arr), interior);

    // Publications and requests
    auto pubs = std::vector{pub_view};
    auto reqs = std::vector<array_view_t<double, 1>>{};

    // Request left ghost region (if not leftmost rank)
    if (rank > 0) {
        auto left_ghost = index_space(ivec(my_start - num_ghosts), uvec(num_ghosts));
        reqs.push_back(view(arr, left_ghost));
    }

    // Request right ghost region (if not rightmost rank)
    if (rank < size - 1) {
        auto right_ghost = index_space(ivec(my_end), uvec(num_ghosts));
        reqs.push_back(view(arr, right_ghost));
    }

    // Build and execute exchange
    auto plan = comm.build_plan<
        array_view_t<const double, 1>,
        array_view_t<double, 1>
    >(pubs, reqs);
    comm.exchange(plan);

    // Verify ghost cells have correct values
    bool success = true;
    if (rank > 0) {
        for (int i = my_start - num_ghosts; i < my_start; ++i) {
            if (arr(ivec(i)) != static_cast<double>(i)) {
                success = false;
                std::cerr << "Rank " << rank << ": left ghost[" << i << "] = "
                          << arr(ivec(i)) << ", expected " << i << "\n";
            }
        }
    }
    if (rank < size - 1) {
        for (int i = my_end; i < my_end + num_ghosts; ++i) {
            if (arr(ivec(i)) != static_cast<double>(i)) {
                success = false;
                std::cerr << "Rank " << rank << ": right ghost[" << i << "] = "
                          << arr(ivec(i)) << ", expected " << i << "\n";
            }
        }
    }

    // Reduce success across all ranks
    int local_success = success ? 1 : 0;
    int global_success = comm.combine(local_success, [](int a, int b) { return a * b; });

    if (comm.rank() == 0) {
        if (global_success) {
            std::cout << "test_exchange_1d: PASSED\n";
        } else {
            std::cout << "test_exchange_1d: FAILED\n";
        }
    }
    assert(global_success);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test_comm_from_mpi();
    test_combine_sum();
    test_combine_min();

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size >= 2) {
        test_exchange_1d();
    } else if (rank == 0) {
        std::cout << "test_exchange_1d: SKIPPED (need >= 2 ranks)\n";
    }

    if (rank == 0) {
        std::cout << "All MPI comm tests passed!\n";
    }

    MPI_Finalize();
    return 0;
}
