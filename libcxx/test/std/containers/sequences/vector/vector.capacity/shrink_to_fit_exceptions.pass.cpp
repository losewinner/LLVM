//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// Check that std::vector::shrink_to_fit provides strong exception guarantees

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../common.h"
#include "count_new.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"

template <typename T = int, typename Alloc = std::allocator<T> >
void test_allocation_exception(std::vector<T, Alloc>& v) {
  (void)v;
}

template <typename T = int, typename Alloc = std::allocator<throwing_data<T> > >
void test_construction_exception(std::vector<throwing_data<T>, Alloc>& v, const std::vector<T>& in) {
  (void)v;
  (void)in;
}

void test_allocation_exceptions() {}

void test_construction_exceptions() {}

int main(int, char**) {
  test_allocation_exceptions();
  test_construction_exceptions();
}
