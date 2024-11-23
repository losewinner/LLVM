//===-- Socket.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Socket.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Threading.h"
#include <cstdint>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/_types/_socklen_t.h>
#include <sys/poll.h>
#include <system_error>

#ifndef _WIN32
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#else
#include "llvm/Support/Windows/WindowsSupport.h"
// winsock2.h must be included before afunix.h. Briefly turn off clang-format to
// avoid error.
// clang-format off
#include <winsock2.h>
#include <afunix.h>
// clang-format on
#include <io.h>
#endif // _WIN32

#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif

namespace {

enum SocketProtocol { ProtocolTcp, ProtocolUnixDomain };

std::error_code getLastSocketErrorCode() {
#ifdef _WIN32
  return std::error_code(::WSAGetLastError(), std::system_category());
#else
  return llvm::errnoAsErrorCode();
#endif
}

llvm::Expected<std::vector<lldb_dap::SocketAddress>>
getAddressInfo(llvm::StringRef host) {
  std::vector<lldb_dap::SocketAddress> add_list;

  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags = 0;

  struct addrinfo *service_info_list = nullptr;

  int err =
      getaddrinfo(host.str().c_str(), nullptr, &hints, &service_info_list);
  if (err != 0)
    return llvm::createStringError("getaddrinfo failed: %s", gai_strerror(err));

  for (struct addrinfo *service_ptr = service_info_list; service_ptr != nullptr;
       service_ptr = service_ptr->ai_next) {
    add_list.emplace_back(lldb_dap::SocketAddress(service_ptr));
  }

  if (service_info_list)
    ::freeaddrinfo(service_info_list);

  return add_list;
}

int CloseSocket(NativeSocket sockfd) {
#ifdef _WIN32
  return ::closesocket(sockfd);
#else
  return ::close(sockfd);
#endif
}

llvm::Expected<std::pair<llvm::StringRef, int>>
detectHostAndPort(llvm::StringRef name) {
  llvm::StringRef host;
  llvm::StringRef port;
  std::tie(host, port) = name.split(":");

  if (host == "" || host == "*") {
    host = "0.0.0.0";
  }

  if (port == "") {
    port = "0";
  }

  int portnu = 0;
  if (!llvm::to_integer(port, portnu, 10)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid host:port specification: '%s'",
                                   name.str().c_str());
  }

  return std::make_pair(host, portnu);
}

} // namespace

namespace lldb_dap {

#ifdef _WIN32
WSABalancer::WSABalancer() {
  WSADATA WsaData;
  ::memset(&WsaData, 0, sizeof(WsaData));
  if (WSAStartup(MAKEWORD(2, 2), &WsaData) != 0) {
    llvm::report_fatal_error("WSAStartup failed");
  }
}

WSABalancer::~WSABalancer() { WSACleanup(); }
#endif // _WIN32

SocketAddress::SocketAddress() { clear(); }

SocketAddress::SocketAddress(const struct addrinfo *info) {
  clear();
  if (info && info->ai_addrlen > 0 &&
      size_t(info->ai_addrlen) <= sizeof m_address) {
    memcpy(&m_address.sa, info->ai_addr, static_cast<size_t>(info->ai_addrlen));
    m_address.sa.sa_len = info->ai_addrlen;
  }
}

SocketAddress::SocketAddress(const struct sockaddr &sa) { m_address.sa = sa; }

SocketAddress::SocketAddress(const struct sockaddr_in &sa) {
  m_address.sa_ipv4 = sa;
}

SocketAddress::SocketAddress(const struct sockaddr_in6 &sa) {
  m_address.sa_ipv6 = sa;
}

SocketAddress::SocketAddress(const struct sockaddr_un &su) {
  m_address.su = su;
}

std::string SocketAddress::getPath() const {
  if (getFamily() == AF_UNIX)
    return std::string(m_address.su.sun_path);
  return "";
}

std::string SocketAddress::getName() const {
  if (getFamily() == AF_UNIX)
    return "unix://" + getPath();

  uint64_t port = getPort();
  std::string hostname;
  if (isLocalhost())
    hostname = "localhost";
  else if (isAnyAddr())
    hostname = getFamily() == AF_INET ? "0.0.0.0" : "[::]";
  else {
    char hbuf[NI_MAXHOST];
    if (getnameinfo(&m_address.sa, getLength(), hbuf, sizeof(hbuf), nullptr, 0,
                    NI_NUMERICHOST) == 0) {
      hostname = std::string(hbuf);
    }
  }

  return "tcp://" + hostname + ":" + std::to_string(port);
}

bool SocketAddress::isLocalhost() const {
  switch (getFamily()) {
  case AF_INET:
    return m_address.sa_ipv4.sin_addr.s_addr == htonl(INADDR_LOOPBACK);
  case AF_INET6:
    return 0 == memcmp(&m_address.sa_ipv6.sin6_addr, &in6addr_loopback, 16);
  }
  return false;
}

bool SocketAddress::isAnyAddr() const {
  switch (getFamily()) {
  case AF_INET:
    return m_address.sa_ipv4.sin_addr.s_addr == htonl(INADDR_ANY);
  case AF_INET6:
    return 0 == memcmp(&m_address.sa_ipv6.sin6_addr, &in6addr_any, 16);
  }
  return false;
}

void SocketAddress::setFamily(sa_family_t family) {
  m_address.sa.sa_family = family;
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) ||       \
    defined(__OpenBSD__)
  switch (family) {
  case AF_INET:
    m_address.sa.sa_len = sizeof(struct sockaddr_in);
    break;

  case AF_INET6:
    m_address.sa.sa_len = sizeof(struct sockaddr_in6);
    break;

  case AF_UNIX:
    m_address.sa.sa_len = SUN_LEN(&m_address.su);
    break;

  default:
    assert(0 && "unsupported socket family");
  }
#endif
}

SocketAddress SocketAddress::addressOf(const Socket &s, std::error_code &EC) {
  SocketAddress sa;
  socklen_t len = sizeof(sockaddr_t);

  if (::getsockname(s.fd, &sa.sockaddr(), &len) == -1) {
    EC = getLastSocketErrorCode();
    return SocketAddress{};
  }

  // If the socket is a struct sockaddr_un then the length is returned by the
  // len parameter not the sa_len field. Update the field to match the
  // returned address.
  sa.m_address.sa.sa_len = len;
  return sa;
}

bool SocketAddress::setToAnyAddress(sa_family_t family, uint16_t port) {
  switch (family) {
  case AF_INET:
    setFamily(family);
    if (setPort(port)) {
      m_address.sa_ipv4.sin_addr.s_addr = htonl(INADDR_ANY);
      return true;
    }
    break;

  case AF_INET6:
    setFamily(family);
    if (setPort(port)) {
      m_address.sa_ipv6.sin6_addr = in6addr_any;
      return true;
    }
    break;
  }

  clear();
  return false;
}

Socket::Socket(NativeSocket fd) : fd(fd) {}
Socket::Socket(Socket &&S) : fd(S.fd) { S.fd = -1; }

Socket::~Socket() {
  close();
}

void Socket::close() {
  if (fd == -1) {
    return;
  }

  CloseSocket(fd);
  fd = -1;
}

SocketListener::SocketListener(std::vector<Socket> sockets, int fds[2])
    : m_listening(true), m_sockets(std::move(sockets)), m_pipe{fds[0], fds[1]} {
}

SocketListener::SocketListener(SocketListener &&SL)
    : m_listening(SL.m_listening.load()), m_sockets(std::move(SL.m_sockets)),
      m_pipe{SL.m_pipe[0], SL.m_pipe[1]} {
  SL.m_listening = false;
  SL.m_sockets.clear();
  SL.m_pipe[0] = -1;
  SL.m_pipe[1] = -1;
}

llvm::Expected<SocketListener>
SocketListener::createListener(llvm::StringRef name) {
  SocketProtocol protocol;

  if (name.consume_front("tcp://") || name.consume_front("tcp:") ||
      name.starts_with(":")) {
    protocol = ProtocolTcp;
  } else if (name.consume_front("unix://") || name.consume_front("unix:") ||
             name.starts_with("/")) {
    protocol = ProtocolUnixDomain;
  } else if (name.contains(":")) {
    protocol = ProtocolTcp;
  } else {
    return llvm::createStringError(
        "invalid address, expected '[tcp://][host]:port' or "
        "'[unix://]/path' but got %s.",
        name.str().c_str());
  }

  std::vector<SocketAddress> addresses;
  if (protocol == ProtocolTcp) {
    auto maybeHostAndPort = detectHostAndPort(name);
    if (auto Err = maybeHostAndPort.takeError())
      return Err;

    llvm::StringRef host;
    int port;
    std::tie(host, port) = *maybeHostAndPort;

    llvm::Expected<std::vector<SocketAddress>> maybeAddresses =
        getAddressInfo(host);
    if (auto Err = maybeAddresses.takeError()) {
      return Err;
    }

    for (auto &address : *maybeAddresses) {
      SocketAddress listen_address = address;
      if (!listen_address.isLocalhost())
        listen_address.setToAnyAddress(address.getFamily(), port);
      else
        listen_address.setPort(port);

      addresses.push_back(std::move(listen_address));
    }
  } else {
    llvm::StringRef path = name;
    if (llvm::sys::fs::exists(path)) {
      return llvm::make_error<llvm::StringError>(
          std::make_error_code(std::errc::file_exists), "file exists at path");
    }

    struct sockaddr_un addr;
    bzero(&addr, sizeof(struct sockaddr_un));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path.str().c_str(), sizeof(addr.sun_path) - 1);
    addr.sun_len = SUN_LEN(&addr);

    addresses.push_back(SocketAddress(addr));
  }

  std::vector<Socket> sockets;
  for (auto &address : addresses) {
    bool isTCP =
        address.getFamily() == AF_INET || address.getFamily() == AF_INET6;

    NativeSocket fd =
        ::socket(address.getFamily(), SOCK_STREAM, isTCP ? IPPROTO_TCP : 0);
    if (fd == -1) {
      return llvm::createStringError(getLastSocketErrorCode(),
                                     "socket() failed");
    }

    int val = 1;
    if (::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(int)) == -1) {
      return llvm::make_error<llvm::StringError>(getLastSocketErrorCode(),
                                                 "setsockopt() failed");
    }

    if (::bind(fd, &address.sockaddr(), address.getLength()) == -1) {
      return llvm::make_error<llvm::StringError>(getLastSocketErrorCode(),
                                                 "bind() failed");
    }

    if (::listen(fd, llvm::hardware_concurrency().compute_thread_count()) ==
        -1) {
      return llvm::make_error<llvm::StringError>(getLastSocketErrorCode(),
                                                 "listen() failed");
    }

    sockets.emplace_back(fd);
  }

  int pipeFD[2];
#ifdef _WIN32
  // Reserve 1 byte for the pipe and use default textmode
  if (::_pipe(pipeFD, 1, 0) == -1)
#else
  if (::pipe(pipeFD) == -1)
#endif // _WIN32
    return llvm::make_error<llvm::StringError>(getLastSocketErrorCode(),
                                               "pipe failed");

  return SocketListener{std::move(sockets), pipeFD};
}

SocketListener::~SocketListener() {
  shutdown();
  if (m_pipe[0] != -1)
    close(m_pipe[0]);
  if (m_pipe[1] != -1)
    close(m_pipe[1]);
}

llvm::Expected<std::vector<std::string>> SocketListener::addresses() const {
  std::vector<std::string> addrs;
  std::error_code EC;
  for (auto &s : m_sockets) {
    auto addr = SocketAddress::addressOf(s, EC);
    if (EC)
      return llvm::make_error<llvm::StringError>(EC);

    addrs.push_back(addr.getName());
  }
  return addrs;
}

void SocketListener::shutdown() {
  bool listening = m_listening;

  if (!listening)
    return;

  if (!m_listening.compare_exchange_strong(listening, false))
    return;

  for (auto &s : m_sockets) {
    // If the socket has a path (e.g. a unix:// socket), remove it after
    // closing.
    std::error_code _; // ignoring failures during shutdown.
    std::string path = SocketAddress::addressOf(s, _).getPath();
    s.close();
    if (!path.empty())
      unlink(path.c_str());
  }

  // Write to the pipe to indiciate that accept() should exit immediately.
  char byte = 'A';
  ssize_t written = ::write(m_pipe[1], &byte, 1);

  // Ignore any write() error
  (void)written;
}

llvm::Expected<Socket> SocketListener::accept() {
  if (m_sockets.empty()) {
    return llvm::createStringError(
        std::make_error_code(std::errc::bad_file_descriptor), "no open socket");
  }

  std::vector<struct pollfd> pollfds;

  for (auto &s : m_sockets)
    pollfds.emplace_back(pollfd{s.fd, POLLIN, 0});

  pollfds.emplace_back(pollfd{m_pipe[0], POLLIN, 0});

  while (m_listening.load()) {
    int status;
#ifdef _WIN32
    status = WSAPoll(pollfds.data(), pollfds.size(), -1);
#else
    status = ::poll(pollfds.data(), pollfds.size(), -1);
#endif

    if (status == -1) {
      return llvm::make_error<llvm::StringError>(getLastSocketErrorCode(),
                                                 "poll() failed");
    }

    for (auto &pfd : pollfds) {
      if (pfd.revents & POLLIN) {
        // Check if shutdown was requested.
        if (pfd.fd == m_pipe[0]) {
          break;
        }

        struct sockaddr client;
        socklen_t len;
        int fd =
            llvm::sys::RetryAfterSignal(-1, ::accept, pfd.fd, &client, &len);
        if (fd == -1) {
          return llvm::make_error<llvm::StringError>(getLastSocketErrorCode(),
                                                     "accept() failed");
        }

        return Socket{fd};
      }
    }
  }

  return llvm::make_error<llvm::StringError>(
      std::make_error_code(std::errc::connection_aborted), "socket shutdown");
}

} // namespace lldb_dap
