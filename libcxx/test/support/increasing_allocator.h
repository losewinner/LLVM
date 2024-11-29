//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_INCREASING_ALLOCATOR_H
#define TEST_SUPPORT_INCREASING_ALLOCATOR_H

#include <cstddef>
#include <memory>

#include "test_macros.h"

#if TEST_STD_VER >= 23

// increasing_allocator is a custom allocator that maintains an incrementing minimum allocation size,
// requiring that each call to allocate_at_least allocate at least this minimum size, which may exceed
// the requested amount. This unique design makes increasing_allocator particularly useful for testing
// the shrink_to_fit functionality in std::vector, vector<bool>, and std::basic_string, ensuring that
// shrink_to_fit does not increase the capacity of the allocated memory.

template <typename T>
struct increasing_allocator {
  using value_type         = T;
  std::size_t min_elements = 1000;
  increasing_allocator()   = default;

  template <typename U>
  constexpr increasing_allocator(const increasing_allocator<U>& other) noexcept : min_elements(other.min_elements) {}

  constexpr std::allocation_result<T*> allocate_at_least(std::size_t n) {
    if (n < min_elements)
      n = min_elements;
    min_elements += 1000;
    return std::allocator<T>{}.allocate_at_least(n);
  }
  constexpr T* allocate(std::size_t n) { return std::allocator<T>{}.allocate(n); }
  constexpr void deallocate(T* p, std::size_t n) noexcept { std::allocator<T>{}.deallocate(p, n); }
};

template <typename T, typename U>
bool operator==(increasing_allocator<T>, increasing_allocator<U>) {
  return true;
}
#endif // TEST_STD_VER >= 23

#endif // TEST_SUPPORT_INCREASING_ALLOCATOR_H
