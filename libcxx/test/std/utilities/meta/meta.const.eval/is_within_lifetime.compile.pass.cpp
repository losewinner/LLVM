//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <type_traits>

// template <class T>
//   consteval bool is_within_lifetime(const T*) noexcept; // C++26

#include <type_traits>
#include <cassert>

#include "test_macros.h"

#ifndef __cpp_lib_is_within_lifetime

// Check that it doesn't exist if the feature test macro isn't defined (via ADL)
template <class T>
constexpr decltype(static_cast<void>(is_within_lifetime(std::declval<T>())), bool{}) is_within_lifetime_exists(int) {
  return true;
}
template <class T>
constexpr bool is_within_lifetime_exists(long) {
  return false;
}

static_assert(!is_within_lifetime_exists<const std::integral_constant<bool, false>*>(0), "");

#elif TEST_STD_VER < 26
#  error __cpp_lib_is_within_lifetime defined before C++26
#else // defined(__cpp_lib_is_within_lifetime) && TEST_STD_VER >= 26

ASSERT_SAME_TYPE(decltype(std::is_within_lifetime(std::declval<int*>())), bool);
ASSERT_SAME_TYPE(decltype(std::is_within_lifetime(std::declval<const int*>())), bool);
ASSERT_SAME_TYPE(decltype(std::is_within_lifetime(std::declval<void*>())), bool);
ASSERT_SAME_TYPE(decltype(std::is_within_lifetime(std::declval<const void*>())), bool);

ASSERT_NOEXCEPT(std::is_within_lifetime(std::declval<int*>()));
ASSERT_NOEXCEPT(std::is_within_lifetime(std::declval<const int*>()));
ASSERT_NOEXCEPT(std::is_within_lifetime(std::declval<void*>()));
ASSERT_NOEXCEPT(std::is_within_lifetime(std::declval<const void*>()));

template <class T>
concept is_within_lifetime_exists = requires(T t) { std::is_within_lifetime(t); };

struct S {};

static_assert(is_within_lifetime_exists<int*>);
static_assert(is_within_lifetime_exists<const int*>);
static_assert(is_within_lifetime_exists<void*>);
static_assert(is_within_lifetime_exists<const void*>);
static_assert(!is_within_lifetime_exists<int>);               // Not a pointer
static_assert(!is_within_lifetime_exists<decltype(nullptr)>); // Not a pointer
static_assert(!is_within_lifetime_exists<void() const>);      // Not a pointer
static_assert(!is_within_lifetime_exists<int S::*>);          // Doesn't accept pointer-to-member
static_assert(!is_within_lifetime_exists<void (S::*)()>);
static_assert(!is_within_lifetime_exists<void (*)()>); // Doesn't match `const T*`

constexpr int i = 0;
static_assert(std::is_within_lifetime(&i));
static_assert(std::is_within_lifetime(const_cast<int*>(&i)));
static_assert(std::is_within_lifetime(static_cast<const void*>(&i)));
static_assert(std::is_within_lifetime(static_cast<void*>(const_cast<int*>(&i))));
static_assert(std::is_within_lifetime<const int>(&i));
static_assert(std::is_within_lifetime<int>(const_cast<int*>(&i)));
static_assert(std::is_within_lifetime<const void>(static_cast<const void*>(&i)));
static_assert(std::is_within_lifetime<void>(static_cast<void*>(const_cast<int*>(&i))));

constexpr union {
  int active;
  int inactive;
} u{.active = 1};
static_assert(std::is_within_lifetime(&u.active) && !std::is_within_lifetime(&u.inactive));

consteval bool f() {
  union {
    int active;
    int inactive;
  };
  if (std::is_within_lifetime(&active) || std::is_within_lifetime(&inactive))
    return false;
  active = 1;
  if (!std::is_within_lifetime(&active) || std::is_within_lifetime(&inactive))
    return false;
  inactive = 1;
  if (std::is_within_lifetime(&active) || !std::is_within_lifetime(&inactive))
    return false;
  int j;
  S s;
  return std::is_within_lifetime(&j) && std::is_within_lifetime(&s);
}
static_assert(f());

#  ifndef TEST_COMPILER_MSVC
// MSVC doesn't support consteval propagation :(
template <typename T>
constexpr void does_escalate(T p) {
  std::is_within_lifetime(p);
}

template <typename T, void(T) = does_escalate<T>>
constexpr bool check_escalated(int) {
  return false;
}
template <typename T>
constexpr bool check_escalated(long) {
  return true;
}
static_assert(check_escalated<int*>(0), "");
static_assert(check_escalated<void*>(0), "");
#  endif // defined(TEST_COMPILER_MSVC)

#endif // defined(__cpp_lib_is_within_lifetime) && TEST_STD_VER >= 26
