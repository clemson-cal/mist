// socket_session.ipp - implementation file for socket_session_t

#ifdef MIST_DRIVER_SEPARATE_COMPILATION
#define MIST_INLINE
#else
#define MIST_INLINE inline
#endif

namespace mist::driver {

// =============================================================================
// socket_session_t implementation
// =============================================================================

MIST_INLINE socket_session_t::socket_session_t(engine_t& engine, std::ostream& out)
    : engine_(engine)
    , out_(out)
{
    command_socket_.listen(0);
    response_socket_.listen(0);
}

MIST_INLINE void socket_session_t::print_ports() {
    out_ << "command_port=" << command_socket_.port() << "\n";
    out_ << "response_port=" << response_socket_.port() << "\n";
    out_ << "data_port=" << engine_.data_socket_port() << "\n";
    out_.flush();
}

MIST_INLINE void socket_session_t::run() {
    print_ports();

    // Accept connections on command and response sockets
    auto cmd_client = command_socket_.accept();
    auto resp_client = response_socket_.accept();

    // Main loop: read commands, execute, send responses
    while (true) {
        auto cmd_opt = read_command(cmd_client);
        if (!cmd_opt) {
            break; // Connection closed or error
        }

        auto should_stop = false;
        engine_.execute(*cmd_opt, [this, &resp_client, &should_stop](const response_t& r) {
            write_response(resp_client, r);
            if (std::holds_alternative<resp::stopped>(r)) {
                should_stop = true;
            }
        });

        if (should_stop) {
            break;
        }
    }
}

MIST_INLINE auto socket_session_t::read_command(socket_t& client) -> std::optional<command_t> {
    try {
        // Read size-prefixed binary data
        auto size = client.recv<uint64_t>();
        auto buffer = std::vector<char>(size);
        client.recv(buffer.data(), size);

        // Deserialize command
        auto iss = std::istringstream{std::string{buffer.begin(), buffer.end()}};
        auto reader = binary_reader{iss};
        auto cmd = command_t{};
        if (deserialize(reader, cmd)) {
            return cmd;
        }
        return std::nullopt;
    } catch (...) {
        return std::nullopt;
    }
}

MIST_INLINE void socket_session_t::write_response(socket_t& client, const response_t& r) {
    // Serialize response to binary
    auto oss = std::ostringstream{};
    auto writer = binary_writer{oss};
    serialize(writer, r);
    auto data = oss.str();

    // Send size-prefixed
    client.send_with_size(data.data(), data.size());
}

} // namespace mist::driver

#undef MIST_INLINE
