#pragma once

#ifdef MIST_HAS_MPI

#include <mpi.h>
#include "communicator.hpp"

namespace mist::driver {

// =============================================================================
// mpi_communicator_t - MPI-based distributed communicator
// =============================================================================

class mpi_communicator_t : public communicator_t {
public:
    // Initialize MPI (calls MPI_Init if not already initialized)
    mpi_communicator_t(int* argc, char*** argv);

    // Finalize MPI (calls MPI_Finalize)
    ~mpi_communicator_t();

    // Non-copyable, non-movable
    mpi_communicator_t(const mpi_communicator_t&) = delete;
    mpi_communicator_t& operator=(const mpi_communicator_t&) = delete;
    mpi_communicator_t(mpi_communicator_t&&) = delete;
    mpi_communicator_t& operator=(mpi_communicator_t&&) = delete;

    auto rank() const -> int override;
    auto size() const -> int override;

    void sendrecv(int peer,
                  std::span<const std::byte> send,
                  std::span<std::byte> recv) override;

    auto allreduce_min(double local) -> double override;
    auto allreduce_max(double local) -> double override;
    auto allreduce_sum(double local) -> double override;

    void broadcast_string(std::string& str) override;

    void barrier() override;

private:
    int rank_;
    int size_;
    bool owns_mpi_;  // true if we called MPI_Init
};

// Factory function declared in communicator.hpp, defined in mpi_communicator.cpp

} // namespace mist::driver

#endif // MIST_HAS_MPI
