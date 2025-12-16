#pragma once

#include <iostream>
#include "engine.hpp"
#include "../socket.hpp"

namespace mist::driver {

// =============================================================================
// socket_session_t - machine interface via sockets
// =============================================================================

class socket_session_t {
public:
    socket_session_t(engine_t& engine, std::ostream& out = std::cout);

    void run();

    auto command_port() const -> int { return command_socket_.port(); }
    auto response_port() const -> int { return response_socket_.port(); }

private:
    engine_t& engine_;
    std::ostream& out_;
    socket_t command_socket_;
    socket_t response_socket_;

    void print_ports();
    auto read_command(socket_t& client) -> std::optional<command_t>;
    void write_response(socket_t& client, const response_t& r);
};

} // namespace mist::driver

// =============================================================================
// Include implementations for header-only mode
// =============================================================================

#ifndef MIST_DRIVER_SEPARATE_COMPILATION
#include "socket_session.ipp"
#endif
