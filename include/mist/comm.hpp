#pragma once

#include <functional>
#include <span>
#include <vector>
#include "ndarray.hpp"

namespace mist {

// =============================================================================
// Exchange plan: pre-computed routing for data transfers
// =============================================================================

template<typename SrcView, typename DestView>
struct exchange_plan_t {
    static_assert(SrcView::rank == DestView::rank, "View ranks must match");
    static constexpr std::size_t rank = SrcView::rank;

    struct local_copy_t {
        SrcView src;
        DestView dest;
        index_space_t<rank> overlap;
    };

    struct send_t {
        int dest_rank;
        SrcView src;
        index_space_t<rank> overlap;
    };

    struct recv_t {
        int src_rank;
        DestView dest;
        index_space_t<rank> overlap;
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

    // --- Topology ---

    auto rank() const -> int { return rank_; }
    auto size() const -> int { return size_; }

    // --- Exchange ---

    template<typename SrcView, typename DestView>
    auto build_plan(
        std::span<SrcView> publications,
        std::span<DestView> requests
    ) -> exchange_plan_t<SrcView, DestView>;

    template<typename SrcView, typename DestView>
    void exchange(const exchange_plan_t<SrcView, DestView>& plan);

    // --- Reduce ---

    template<typename T, typename BinaryOp>
    auto combine(T local_value, BinaryOp op) -> T;
};

// =============================================================================
// Implementation
// =============================================================================

template<typename SrcView, typename DestView>
auto comm_t::build_plan(
    std::span<SrcView> publications,
    std::span<DestView> requests
) -> exchange_plan_t<SrcView, DestView> {
    auto plan = exchange_plan_t<SrcView, DestView>{};

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

template<typename SrcView, typename DestView>
void comm_t::exchange(const exchange_plan_t<SrcView, DestView>& plan) {
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
