// mpi_communicator.cpp - MPI-based communicator implementation

#ifdef MIST_HAS_MPI

#include "mist/driver/mpi_communicator.hpp"
#include <stdexcept>

namespace mist::driver {

mpi_communicator_t::mpi_communicator_t(int* argc, char*** argv)
    : rank_(0)
    , size_(1)
    , owns_mpi_(false)
{
    int initialized = 0;
    MPI_Initialized(&initialized);

    if (!initialized) {
        MPI_Init(argc, argv);
        owns_mpi_ = true;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

mpi_communicator_t::~mpi_communicator_t() {
    if (owns_mpi_) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    }
}

auto mpi_communicator_t::rank() const -> int {
    return rank_;
}

auto mpi_communicator_t::size() const -> int {
    return size_;
}

void mpi_communicator_t::sendrecv(int peer,
                                   std::span<const std::byte> send,
                                   std::span<std::byte> recv) {
    MPI_Sendrecv(
        send.data(), static_cast<int>(send.size()), MPI_BYTE, peer, 0,
        recv.data(), static_cast<int>(recv.size()), MPI_BYTE, peer, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
}

auto mpi_communicator_t::allreduce_min(double local) -> double {
    double result = 0.0;
    MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return result;
}

auto mpi_communicator_t::allreduce_max(double local) -> double {
    double result = 0.0;
    MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
}

auto mpi_communicator_t::allreduce_sum(double local) -> double {
    double result = 0.0;
    MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return result;
}

void mpi_communicator_t::broadcast_string(std::string& str) {
    // First broadcast the length
    int len = static_cast<int>(str.size());
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize on non-root ranks
    if (rank_ != 0) {
        str.resize(len);
    }

    // Broadcast the string data
    MPI_Bcast(str.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
}

void mpi_communicator_t::barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

auto make_mpi_communicator(int* argc, char*** argv) -> std::unique_ptr<communicator_t> {
    return std::make_unique<mpi_communicator_t>(argc, argv);
}

} // namespace mist::driver

#endif // MIST_HAS_MPI
