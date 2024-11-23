//===-- Socket.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_SOCKET_H
#define LLDB_TOOLS_LLDB_DAP_SOCKET_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <vector>

#ifdef _WIN32
#include "lldb/Host/windows/windows.h"
#include <winsock2.h>
#include <ws2tcpip.h>
typedef ADDRESS_FAMILY sa_family_t;
typedef SOCKET NativeSocket;
#else
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/un.h>
typedef int NativeSocket;
#endif

namespace lldb_dap {

struct Socket;

/// Manages socket addresses for socket based communication.
struct SocketAddress {
  SocketAddress();
  explicit SocketAddress(const struct addrinfo *);
  explicit SocketAddress(const struct sockaddr &);
  explicit SocketAddress(const struct sockaddr_in &);
  explicit SocketAddress(const struct sockaddr_in6 &);
  explicit SocketAddress(const struct sockaddr_un &);

  static SocketAddress addressOf(const Socket &s, std::error_code &EC);

  sa_family_t getFamily() const { return m_address.sa.sa_family; }
  uint16_t getPort() const {
    switch (getFamily()) {
    case AF_INET:
      return ntohs(m_address.sa_ipv4.sin_port);
    case AF_INET6:
      return ntohs(m_address.sa_ipv6.sin6_port);
    }
    return 0;
  }
  std::string getName() const;
  /// Returns the path for AF_UNIX sockets, otherwise "".
  std::string getPath() const;
  /// Get the length for the current socket address family
  socklen_t getLength() const { return m_address.sa.sa_len; };

  bool isLocalhost() const;
  bool isAnyAddr() const;

  void setFamily(sa_family_t family);
  bool setToAnyAddress(sa_family_t family, uint16_t port);
  bool setPort(uint16_t port) {
    switch (getFamily()) {
    case AF_INET:
      m_address.sa_ipv4.sin_port = htons(port);
      return true;
    case AF_INET6:
      m_address.sa_ipv6.sin6_port = htons(port);
      return true;
    }
    return false;
  }

  void clear() { bzero(&m_address, sizeof(sockaddr_t)); }

  struct sockaddr &sockaddr() { return m_address.sa; }
  const struct sockaddr &sockaddr() const { return m_address.sa; }

private:
  typedef union sockaddr_tag {
    struct sockaddr sa;
    struct sockaddr_in sa_ipv4;
    struct sockaddr_in6 sa_ipv6;
    struct sockaddr_un su;
    struct sockaddr_storage ss;
  } sockaddr_t;
  sockaddr_t m_address;
};

/// Manages the lifetime of a socket.
struct Socket {
  NativeSocket fd;

  /// Close the socket.
  void close();

  explicit Socket(NativeSocket fd);
  ~Socket();
  Socket(Socket &&S);
  Socket(const Socket &S) = delete;
  Socket &operator=(const Socket &S) = delete;
};

#ifdef _WIN32
/// Ensures proper initialization and cleanup of winsock resources
///
/// Make sure that calls to WSAStartup and WSACleanup are balanced.
class WSABalancer {
public:
  WSABalancer();
  ~WSABalancer();
};
#endif // _WIN32

/// Manages listening for socket connections.
///
/// SocketListener handles listening for both unix and tcp based connections.
struct SocketListener {
  SocketListener(const SocketListener &S) = delete;
  SocketListener &operator=(const SocketListener &S) = delete;
  SocketListener(SocketListener &&SL);
  ~SocketListener();

  /// Creates a listening socket bound to the specified name.
  ///
  /// Handles the socket creation, binding, and immediately starts listening for
  /// incoming connections.
  ///
  /// \param[in] name
  ///     Socket names formatted like `protocol:name`.
  ///
  ///     Supported protocols include tcp and unix sockets.
  ///
  ///     Names must follow the following formats:
  ///
  ///     * tcp://host:port
  ///     * tcp:host:port
  ///     * tcp:port (host will be assumed 0.0.0.0)
  ///     * :port (implicit tcp)
  ///     * unix:///path
  ///     * /path (implicit unix)
  static llvm::Expected<SocketListener> createListener(llvm::StringRef name);

  /// Returns an array of active listening sockets.
  llvm::Expected<std::vector<std::string>> addresses() const;

  /// Shutdown the socket listener.
  void shutdown();

  /// Returns true if listening, otherwise false.
  bool isListening() const { return m_listening; }

  /// Accept returns the next client connection, blocking until a connection is
  /// made or Shutdown is called.
  llvm::Expected<Socket> accept();

private:
  SocketListener(std::vector<Socket> sockets, int fds[2]);

  std::atomic<bool> m_listening;
  std::vector<Socket> m_sockets;

  /// If a separate thread calls shutdown, the listening file descriptor
  /// could be closed while ::poll is waiting for it to be ready to perform a
  /// I/O operations. ::poll will continue to block even after FD is closed so
  /// use a self-pipe mechanism to get ::poll to return
  int m_pipe[2] = {};

#ifdef _WIN32
  WSABalancer _;
#endif // _WIN32
};

} // namespace lldb_dap

#endif
