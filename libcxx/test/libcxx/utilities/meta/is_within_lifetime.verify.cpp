//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <type_traits>

// LWG4138 <https://cplusplus.github.io/LWG/issue4138>
// std::is_within_lifetime shouldn't work when a function type is
// explicitly specified, even if it isn't evaluated

#include <type_traits>
#include <cassert>

#include "test_macros.h"

void fn();

int main(int, char**) {
#ifdef __cpp_lib_is_within_lifetime
  constexpr bool _ = true ? false : std::is_within_lifetime<void()>(&fn);
  // expected-error@*:* {{static assertion failed due to requirement '!is_function_v<void ()>': std::is_within_lifetime<T> cannot explicitly specify T as a function type}}
#else
  // expected-no-diagnostics
#endif
  return 0;
}
