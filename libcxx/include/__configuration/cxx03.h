//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONFIGURATION_CXX03_H
#define _LIBCPP___CONFIGURATION_CXX03_H

#ifndef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  pragma GCC system_header
#endif

// NOLINTNEXTLINE(libcpp-cpp-version-check)
#  if __cplusplus < 201103L
#    define _LIBCPP_CXX03_LANG
#  endif

#endif // _LIBCPP___CONFIGURATION_CXX03_H
