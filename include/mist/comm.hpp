#pragma once

#include <span>
#include <vector>
#include "ndarray.hpp"

#ifdef MIST_WITH_MPI
#include <mpi.h>
#endif

namespace mist {

// =============================================================================
// Exchange plan: pre-computed routing for data transfers
// =============================================================================

template<typename T, std::size_t S>
struct exchange_plan_t {
    using src_view_t = array_view_t<const T, S>;
    using dest_view_t = array_view_t<T, S>;

    struct local_copy_t {
        src_view_t src;
        dest_view_t dest;
        index_space_t<S> overlap;
    };

    struct send_t {
        int dest_rank;
        src_view_t src;
        index_space_t<S> overlap;
    };

    struct recv_t {
        int src_rank;
        dest_view_t dest;
        index_space_t<S> overlap;
    };

    std::vector<local_copy_t> local_copies;
    std::vector<send_t> sends;
    std::vector<recv_t> recvs;
};

// =============================================================================
// Communicator
// =============================================================================

struct comm_t {
    int rank_ = 0;
    int size_ = 1;
#ifdef MIST_WITH_MPI
    MPI_Comm mpi_comm_ = MPI_COMM_NULL;
#endif

    // --- Factory ---

#ifdef MIST_WITH_MPI
    static auto from_mpi(MPI_Comm comm) -> comm_t;
#endif

    // --- Topology ---

    auto rank() const -> int { return rank_; }
    auto size() const -> int { return size_; }

    // --- Exchange ---

    template<typename T, std::size_t S>
    auto build_plan(
        std::span<array_view_t<const T, S>> publications,
        std::span<array_view_t<T, S>> requests
    ) -> exchange_plan_t<T, S>;

    template<typename T, std::size_t S>
    void exchange(const exchange_plan_t<T, S>& plan);

    // --- Reduce ---

    template<typename T, typename BinaryOp>
    auto combine(T local_value, BinaryOp op) -> T;
};

// =============================================================================
// MPI helpers
// =============================================================================

#ifdef MIST_WITH_MPI

namespace detail {

// Metadata for a publication: the index space it provides
template<std::size_t S>
struct publication_meta_t {
    int rank;
    ivec_t<S> start;
    uvec_t<S> shape;
    ivec_t<S> parent_start;
    uvec_t<S> parent_shape;
};

// Type trait to extract MPI type and element count
template<typename T>
struct mpi_type_traits {
    static constexpr bool is_supported = false;
};

template<>
struct mpi_type_traits<double> {
    static constexpr bool is_supported = true;
    static constexpr std::size_t count = 1;
    static auto type() -> MPI_Datatype { return MPI_DOUBLE; }
};

template<>
struct mpi_type_traits<float> {
    static constexpr bool is_supported = true;
    static constexpr std::size_t count = 1;
    static auto type() -> MPI_Datatype { return MPI_FLOAT; }
};

template<>
struct mpi_type_traits<int> {
    static constexpr bool is_supported = true;
    static constexpr std::size_t count = 1;
    static auto type() -> MPI_Datatype { return MPI_INT; }
};

// Specialization for vec_t (treat as array of underlying type)
template<typename T, std::size_t N>
struct mpi_type_traits<vec_t<T, N>> {
    static constexpr bool is_supported = mpi_type_traits<T>::is_supported;
    static constexpr std::size_t count = N;
    static auto type() -> MPI_Datatype { return mpi_type_traits<T>::type(); }
};

// Create MPI subarray datatype for a view
template<typename T, std::size_t S>
auto make_mpi_subarray(const array_view_t<T, S>& view, const index_space_t<S>& overlap) -> MPI_Datatype {
    using value_type = std::remove_const_t<T>;

    static_assert(mpi_type_traits<value_type>::is_supported, "Unsupported MPI element type");

    // Get MPI type for element
    MPI_Datatype base_type = mpi_type_traits<value_type>::type();
    constexpr std::size_t elem_count = mpi_type_traits<value_type>::count;

    // Create contiguous type for the element (handles vec_t)
    MPI_Datatype element_type;
    if constexpr (elem_count > 1) {
        MPI_Type_contiguous(static_cast<int>(elem_count), base_type, &element_type);
        MPI_Type_commit(&element_type);
    } else {
        element_type = base_type;
    }

    // Convert to int arrays for MPI
    int sizes[S];
    int subsizes[S];
    int starts[S];

    auto parent_space = parent(view);
    for (std::size_t i = 0; i < S; ++i) {
        sizes[i] = static_cast<int>(shape(parent_space).data[i]);
        subsizes[i] = static_cast<int>(shape(overlap).data[i]);
        starts[i] = static_cast<int>(start(overlap).data[i] - start(parent_space).data[i]);
    }

    MPI_Datatype subarray_type;
    MPI_Type_create_subarray(
        static_cast<int>(S),
        sizes,
        subsizes,
        starts,
        MPI_ORDER_C,
        element_type,
        &subarray_type
    );
    MPI_Type_commit(&subarray_type);

    // Free the intermediate contiguous type if we created one
    if constexpr (elem_count > 1) {
        MPI_Type_free(&element_type);
    }

    return subarray_type;
}

// Generate a unique tag from overlap region (for message matching)
template<std::size_t S>
auto make_tag(const index_space_t<S>& overlap) -> int {
    // Simple hash of start and shape
    std::size_t h = 0;
    for (std::size_t i = 0; i < S; ++i) {
        h ^= std::hash<int>{}(start(overlap).data[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<unsigned int>{}(shape(overlap).data[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    // MPI tags must be non-negative and fit in int
    return static_cast<int>(h & 0x7FFFFFFF);
}

} // namespace detail

inline auto comm_t::from_mpi(MPI_Comm comm) -> comm_t {
    auto result = comm_t{};
    MPI_Comm_rank(comm, &result.rank_);
    MPI_Comm_size(comm, &result.size_);
    result.mpi_comm_ = comm;
    return result;
}

#endif // MIST_WITH_MPI

// =============================================================================
// Implementation
// =============================================================================

template<typename T, std::size_t S>
auto comm_t::build_plan(
    std::span<array_view_t<const T, S>> publications,
    std::span<array_view_t<T, S>> requests
) -> exchange_plan_t<T, S> {
    auto plan = exchange_plan_t<T, S>{};

#ifdef MIST_WITH_MPI
    if (mpi_comm_ != MPI_COMM_NULL && size_ > 1) {
        // Gather publication count from all ranks
        int local_pub_count = static_cast<int>(publications.size());
        auto pub_counts = std::vector<int>(size_);
        MPI_Allgather(&local_pub_count, 1, MPI_INT, pub_counts.data(), 1, MPI_INT, mpi_comm_);

        // Compute displacements
        auto displacements = std::vector<int>(size_);
        int total_pubs = 0;
        for (int r = 0; r < size_; ++r) {
            displacements[r] = total_pubs;
            total_pubs += pub_counts[r];
        }

        // Serialize local publication metadata
        using meta_t = detail::publication_meta_t<S>;
        auto local_meta = std::vector<meta_t>{};
        local_meta.reserve(local_pub_count);
        for (const auto& pub : publications) {
            local_meta.push_back({
                .rank = rank_,
                .start = start(space(pub)),
                .shape = shape(space(pub)),
                .parent_start = start(parent(pub)),
                .parent_shape = shape(parent(pub))
            });
        }

        // Allgather publication metadata
        // Create MPI type for publication_meta_t
        MPI_Datatype meta_type;
        MPI_Type_contiguous(sizeof(meta_t), MPI_BYTE, &meta_type);
        MPI_Type_commit(&meta_type);

        // Scale counts and displacements for the meta type
        auto recv_counts = std::vector<int>(pub_counts);
        auto recv_displs = std::vector<int>(displacements);

        auto all_meta = std::vector<meta_t>(total_pubs);
        MPI_Allgatherv(
            local_meta.data(), local_pub_count, meta_type,
            all_meta.data(), recv_counts.data(), recv_displs.data(), meta_type,
            mpi_comm_
        );
        MPI_Type_free(&meta_type);

        // Match requests with publications
        for (const auto& req : requests) {
            auto req_space = space(req);

            // Check local publications first
            for (const auto& pub : publications) {
                auto overlap = intersect(req_space, space(pub));
                if (mist::size(overlap) > 0) {
                    plan.local_copies.push_back({
                        .src = pub,
                        .dest = req,
                        .overlap = overlap
                    });
                }
            }

            // Check remote publications
            for (const auto& meta : all_meta) {
                if (meta.rank == rank_) continue;  // Skip local

                auto pub_space = index_space(meta.start, meta.shape);
                auto overlap = intersect(req_space, pub_space);

                if (mist::size(overlap) > 0) {
                    plan.recvs.push_back({
                        .src_rank = meta.rank,
                        .dest = req,
                        .overlap = overlap
                    });
                }
            }
        }

        // For each local publication, check if any remote rank needs it
        for (const auto& pub : publications) {
            auto pub_space = space(pub);

            // We need to know what other ranks are requesting
            // This requires another round of communication: gather request metadata

            // For simplicity, we'll use a symmetric approach:
            // Each rank sends its publications to ranks that might need them
            // based on the recvs we've computed locally
        }

        // Compute sends by gathering recv information
        // Each rank broadcasts what it's receiving, so senders know what to send
        int local_recv_count = static_cast<int>(plan.recvs.size());
        auto recv_counts_per_rank = std::vector<int>(size_);
        MPI_Allgather(&local_recv_count, 1, MPI_INT, recv_counts_per_rank.data(), 1, MPI_INT, mpi_comm_);

        // Serialize recv metadata (what this rank needs from others)
        struct recv_meta_t {
            int requesting_rank;
            int providing_rank;
            ivec_t<S> overlap_start;
            uvec_t<S> overlap_shape;
        };

        auto local_recv_meta = std::vector<recv_meta_t>(plan.recvs.size());
        for (std::size_t i = 0; i < plan.recvs.size(); ++i) {
            local_recv_meta[i].requesting_rank = rank_;
            local_recv_meta[i].providing_rank = plan.recvs[i].src_rank;
            local_recv_meta[i].overlap_start = start(plan.recvs[i].overlap);
            local_recv_meta[i].overlap_shape = shape(plan.recvs[i].overlap);
        }

        // Gather all recv metadata
        int total_recvs = 0;
        auto recv_meta_displs = std::vector<int>(size_);
        for (int r = 0; r < size_; ++r) {
            recv_meta_displs[r] = total_recvs;
            total_recvs += recv_counts_per_rank[r];
        }

        MPI_Datatype recv_meta_type;
        MPI_Type_contiguous(sizeof(recv_meta_t), MPI_BYTE, &recv_meta_type);
        MPI_Type_commit(&recv_meta_type);

        auto all_recv_meta = std::vector<recv_meta_t>(total_recvs);
        MPI_Allgatherv(
            local_recv_meta.data(), local_recv_count, recv_meta_type,
            all_recv_meta.data(), recv_counts_per_rank.data(), recv_meta_displs.data(), recv_meta_type,
            mpi_comm_
        );
        MPI_Type_free(&recv_meta_type);

        // Build sends: for each recv where we are the provider
        for (const auto& rmeta : all_recv_meta) {
            if (rmeta.providing_rank == rank_) {
                auto overlap = index_space(rmeta.overlap_start, rmeta.overlap_shape);

                // Find the local publication that provides this
                for (const auto& pub : publications) {
                    if (contains(space(pub), overlap)) {
                        plan.sends.push_back({
                            .dest_rank = rmeta.requesting_rank,
                            .src = pub,
                            .overlap = overlap
                        });
                        break;
                    }
                }
            }
        }

        return plan;
    }
#endif // MIST_WITH_MPI

    // Local-only fallback
    for (const auto& req : requests) {
        for (const auto& pub : publications) {
            auto overlap = intersect(space(req), space(pub));

            if (mist::size(overlap) > 0) {
                plan.local_copies.push_back({
                    .src = pub,
                    .dest = req,
                    .overlap = overlap
                });
            }
        }
    }

    return plan;
}

template<typename T, std::size_t S>
void comm_t::exchange(const exchange_plan_t<T, S>& plan) {
    // Execute local copies
    for (const auto& copy : plan.local_copies) {
        for (auto idx : copy.overlap) {
            copy.dest(idx) = copy.src(idx);
        }
    }

#ifdef MIST_WITH_MPI
    if (mpi_comm_ != MPI_COMM_NULL && size_ > 1) {
        auto requests = std::vector<MPI_Request>{};
        auto datatypes = std::vector<MPI_Datatype>{};

        // Post receives
        for (const auto& recv : plan.recvs) {
            auto dtype = detail::make_mpi_subarray(recv.dest, recv.overlap);
            datatypes.push_back(dtype);

            MPI_Request req;
            int tag = detail::make_tag(recv.overlap);
            MPI_Irecv(
                data(recv.dest),
                1,
                dtype,
                recv.src_rank,
                tag,
                mpi_comm_,
                &req
            );
            requests.push_back(req);
        }

        // Post sends
        for (const auto& send : plan.sends) {
            auto dtype = detail::make_mpi_subarray(send.src, send.overlap);
            datatypes.push_back(dtype);

            MPI_Request req;
            int tag = detail::make_tag(send.overlap);
            MPI_Isend(
                const_cast<T*>(data(send.src)),
                1,
                dtype,
                send.dest_rank,
                tag,
                mpi_comm_,
                &req
            );
            requests.push_back(req);
        }

        // Wait for all
        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        }

        // Free datatypes
        for (auto& dtype : datatypes) {
            MPI_Type_free(&dtype);
        }
    }
#endif // MIST_WITH_MPI
}

template<typename T, typename BinaryOp>
auto comm_t::combine(T local_value, BinaryOp op) -> T {
#ifdef MIST_WITH_MPI
    if (mpi_comm_ != MPI_COMM_NULL && size_ > 1) {
        // Gather all values to all ranks
        auto all_values = std::vector<T>(size_);
        MPI_Allgather(&local_value, sizeof(T), MPI_BYTE,
                      all_values.data(), sizeof(T), MPI_BYTE, mpi_comm_);

        // Reduce locally using the provided binary op
        T result = all_values[0];
        for (int i = 1; i < size_; ++i) {
            result = op(result, all_values[i]);
        }
        return result;
    }
#else
    (void)op;
#endif
    return local_value;
}

} // namespace mist
