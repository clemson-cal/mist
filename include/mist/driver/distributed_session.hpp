#pragma once

#include <iostream>
#include "communicator.hpp"
#include "engine.hpp"
#include "repl_session.hpp"

namespace mist::driver {

// =============================================================================
// distributed_session_t - distributed REPL session
//
// Root rank reads commands, broadcasts to all ranks, all execute, root outputs.
// =============================================================================

class distributed_session_t {
public:
    distributed_session_t(engine_t& engine, communicator_t& comm,
                          std::ostream& out = std::cout,
                          std::ostream& err = std::cerr);

    int run();

private:
    engine_t& engine_;
    communicator_t& comm_;
    std::ostream& out_;
    std::ostream& err_;
    color::scheme_t colors_;
    color::scheme_t err_colors_;
    bool should_stop_ = false;
    bool had_error_ = false;

    void format_response(const response_t& r);
};

} // namespace mist::driver
