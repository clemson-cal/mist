#pragma once

#include <functional>
#include <span>
#include <vector>
#include "ndarray.hpp"

namespace mist {

// =============================================================================
// Exchange plan: pre-computed routing for data transfers
// =============================================================================

template<typename View>
struct exchange_plan_t {
    struct local_copy_t {
        View src;
        View dest;
        index_space_t<View::rank> overlap;
    };

    struct remote_transfer_t {
        int remote_rank;
        View local_view;
        index_space_t<View::rank> overlap;
    };

    std::vector<local_copy_t> local_copies;
    std::vector<remote_transfer_t> sends;
    std::vector<remote_transfer_t> recvs;
};

// =============================================================================
// Communicator
// =============================================================================

struct comm_t {
    int rank_ = 0;
    int size_ = 1;

    // --- Topology ---

    auto rank() const -> int { return rank_; }
    auto size() const -> int { return size_; }

    // --- Exchange ---

    template<typename View>
    auto build_plan(
        std::span<const View> publications,
        std::span<const View> requests
    ) -> exchange_plan_t<View>;

    template<typename View>
    void exchange(const exchange_plan_t<View>& plan);

    // --- Reduce ---

    template<typename T, typename BinaryOp>
    auto combine(T local_value, BinaryOp op) -> T;
};

// =============================================================================
// Implementation
// =============================================================================

template<typename View>
auto comm_t::build_plan(
    std::span<const View> publications,
    std::span<const View> requests
) -> exchange_plan_t<View> {
    auto plan = exchange_plan_t<View>{};

    // For each request, find overlapping publications
    for (const auto& req : requests) {
        for (const auto& pub : publications) {
            auto overlap = intersect(space(req), space(pub));

            if (mist::size(overlap) > 0) {
                // Local copy (same rank in single-process mode)
                plan.local_copies.push_back({
                    .src = pub,
                    .dest = req,
                    .overlap = overlap
                });
            }
        }
    }

    // Remote transfers would be computed here when MPI is enabled
    // For now, all transfers are local

    return plan;
}

template<typename View>
void comm_t::exchange(const exchange_plan_t<View>& plan) {
    // Execute local copies
    for (const auto& copy : plan.local_copies) {
        for (auto idx : copy.overlap) {
            copy.dest(idx) = copy.src(idx);
        }
    }

    // Remote transfers would be executed here when MPI is enabled
    // sends: post MPI_Isend for each
    // recvs: post MPI_Irecv for each
    // then MPI_Waitall
}

template<typename T, typename BinaryOp>
auto comm_t::combine(T local_value, BinaryOp /*op*/) -> T {
    // In local-only mode, no other ranks to combine with
    // When MPI is enabled: MPI_Allreduce with custom op
    return local_value;
}

} // namespace mist
