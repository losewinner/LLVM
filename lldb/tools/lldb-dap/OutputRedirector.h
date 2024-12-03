//===-- OutputRedirector.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#ifndef LLDB_TOOLS_LLDB_DAP_OUTPUT_REDIRECTOR_H
#define LLDB_TOOLS_LLDB_DAP_OUTPUT_REDIRECTOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <utility>
#include <vector>

namespace lldb_dap {

/// Manages the lifetime of file descriptor redirects.
struct OutputRedirector {
  OutputRedirector() = default;

  OutputRedirector(const OutputRedirector &) = delete;
  OutputRedirector &operator=(const OutputRedirector &) = delete;

  ~OutputRedirector();

  /// Redirects the output of a given file descriptor to a callback.
  ///
  /// \param[in] fd
  ///     Either -1 or the fd duplicate into the new handle.
  ///
  /// \param[in] callback
  ///     A callback invoked each time the file is written.
  ///
  /// \return
  ///     A new file handle for the output.
  llvm::Expected<int> RedirectFd(int fd,
                                 std::function<void(llvm::StringRef)> callback);

private:
  std::vector<int> m_fds; // owned fds, closed on dealloc.
  std::vector<std::pair<int, int>>
      m_redirects; // pairs (new, old) of redirected fds.
};

} // namespace lldb_dap

#endif // LLDB_TOOLS_LLDB_DAP_OUTPUT_REDIRECTOR_H
