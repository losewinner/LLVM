//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include "test_macros.h"

TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wprivate-header")
#include <__utility/pointer_int_pair.h>
TEST_DIAGNOSTIC_POP

#include <cassert>

int main(int, char**) {
  {
    std::__pointer_int_pair<int*, 1> pair = nullptr;
    assert(pair.__get_value() == 0);
    assert(pair.__get_ptr() == nullptr);
  }
  {
    std::__pointer_int_pair<int*, 1> pair(nullptr, 1);
    assert(pair.__get_value() == 1);
    assert(pair.__get_ptr() == nullptr);
  }
  {
    static_assert(alignof(int) <= 4);
    std::__pointer_int_pair<int*, 1> pair(reinterpret_cast<int*>(4), 0);
    assert(pair.__get_value() == 0);
    assert(pair.__get_ptr() == reinterpret_cast<int*>(4));
  }
}
