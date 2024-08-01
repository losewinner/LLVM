//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Check that user-specializations are diagnosed
// See [cmp.result]/1

#include <compare>

struct S {};

template <>
class std::compare_three_way_result<S>; // expected-error {{cannot be specialized}}
