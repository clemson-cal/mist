#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

namespace mist {

// =============================================================================
// Socket: TCP socket for binary data transmission
// =============================================================================

class socket_t {
public:
    socket_t() = default;

    explicit socket_t(int fd) : fd_(fd) {}

    ~socket_t() {
        close();
    }

    // Non-copyable
    socket_t(const socket_t&) = delete;
    auto operator=(const socket_t&) -> socket_t& = delete;

    // Movable
    socket_t(socket_t&& other) noexcept : fd_(other.fd_) {
        other.fd_ = -1;
    }

    auto operator=(socket_t&& other) noexcept -> socket_t& {
        if (this != &other) {
            close();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    // =========================================================================
    // Server operations
    // =========================================================================

    void listen(uint16_t port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) {
            throw std::runtime_error("socket: failed to create socket");
        }

        int opt = 1;
        setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port);

        if (::bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
            close();
            throw std::runtime_error("socket: failed to bind to port " + std::to_string(port));
        }

        if (::listen(fd_, 1) < 0) {
            close();
            throw std::runtime_error("socket: failed to listen");
        }
    }

    auto accept() -> socket_t {
        sockaddr_in client_addr{};
        socklen_t addrlen = sizeof(client_addr);
        int client_fd = ::accept(fd_, reinterpret_cast<sockaddr*>(&client_addr), &addrlen);
        if (client_fd < 0) {
            throw std::runtime_error("socket: failed to accept connection");
        }
        return socket_t(client_fd);
    }

    /// Accept with periodic interrupt check. Returns nullopt if interrupted.
    template<typename InterruptCheck>
    auto accept_interruptible(InterruptCheck is_interrupted) -> std::optional<socket_t> {
        while (true) {
            // Use select with 100ms timeout to allow interrupt checking
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(fd_, &readfds);

            timeval timeout{};
            timeout.tv_sec = 0;
            timeout.tv_usec = 100000;  // 100ms

            int result = ::select(fd_ + 1, &readfds, nullptr, nullptr, &timeout);
            if (result < 0) {
                if (errno == EINTR) continue;  // Interrupted by signal, retry
                throw std::runtime_error("socket: select failed");
            }
            if (result == 0) {
                // Timeout - check for interrupt
                if (is_interrupted()) {
                    return std::nullopt;
                }
                continue;
            }
            // Connection ready
            sockaddr_in client_addr{};
            socklen_t addrlen = sizeof(client_addr);
            int client_fd = ::accept(fd_, reinterpret_cast<sockaddr*>(&client_addr), &addrlen);
            if (client_fd < 0) {
                throw std::runtime_error("socket: failed to accept connection");
            }
            return socket_t(client_fd);
        }
    }

    auto port() const -> uint16_t {
        sockaddr_in address{};
        socklen_t addrlen = sizeof(address);
        if (getsockname(fd_, reinterpret_cast<sockaddr*>(&address), &addrlen) < 0) {
            throw std::runtime_error("socket: failed to get socket name");
        }
        return ntohs(address.sin_port);
    }

    // =========================================================================
    // Client operations
    // =========================================================================

    void connect(const std::string& host, uint16_t port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) {
            throw std::runtime_error("socket: failed to create socket");
        }

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(port);

        // For simplicity, only support localhost (127.0.0.1)
        if (host == "localhost" || host == "127.0.0.1") {
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        } else {
            close();
            throw std::runtime_error("socket: only localhost connections supported");
        }

        if (::connect(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
            close();
            throw std::runtime_error("socket: failed to connect to " + host + ":" + std::to_string(port));
        }
    }

    // =========================================================================
    // Data transmission
    // =========================================================================

    void send(const void* data, std::size_t size) {
        auto bytes = static_cast<const char*>(data);
        std::size_t sent = 0;
        while (sent < size) {
            auto n = ::send(fd_, bytes + sent, size - sent, 0);
            if (n < 0) {
                throw std::runtime_error("socket: send failed");
            }
            sent += static_cast<std::size_t>(n);
        }
    }

    void recv(void* data, std::size_t size) {
        auto bytes = static_cast<char*>(data);
        std::size_t received = 0;
        while (received < size) {
            auto n = ::recv(fd_, bytes + received, size - received, 0);
            if (n <= 0) {
                throw std::runtime_error("socket: recv failed or connection closed");
            }
            received += static_cast<std::size_t>(n);
        }
    }

    template<typename T>
        requires std::is_trivially_copyable_v<T>
    void send(const T& value) {
        send(&value, sizeof(T));
    }

    template<typename T>
        requires std::is_trivially_copyable_v<T>
    auto recv() -> T {
        T value;
        recv(&value, sizeof(T));
        return value;
    }

    void send_with_size(const void* data, std::size_t size) {
        auto count = static_cast<uint64_t>(size);
        send(count);
        send(data, size);
    }

    auto recv_with_size(void* buffer, std::size_t buffer_size) -> std::size_t {
        auto count = recv<uint64_t>();
        if (count > buffer_size) {
            throw std::runtime_error("socket: buffer too small for incoming data");
        }
        recv(buffer, count);
        return static_cast<std::size_t>(count);
    }

    // =========================================================================
    // State
    // =========================================================================

    auto is_open() const -> bool {
        return fd_ >= 0;
    }

    void close() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    auto fd() const -> int {
        return fd_;
    }

private:
    int fd_ = -1;
};

} // namespace mist
