//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FUNCTIONAL_MOVE_ONLY_FUNCTION_H
#define _LIBCPP___FUNCTIONAL_MOVE_ONLY_FUNCTION_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 23 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

// move_only_function design:
//
// move_only_function has a small buffer with a size of `3 * sizeof(void*)` bytes. This buffer can only be used when the
// object that should be stored is trivially relocatable (currently only when it is trivially move constructible and
// trivially destructible). There is also a bool in the lower bits of the vptr stored which is set when the contained
// object is not trivially destructible.
//
// trivially relocatable: It would also be possible to store nothrow_move_constructible types, but that would mean
// that move_only_function itself would not be trivially relocatable anymore. The decision to keep move_only_function
// trivially relocatable was made because we expect move_only_function to be mostly used to store a functor. To only
// forward functors there is C++26's std::function_ref.
//
// buffer size: We did a survey of six implementations from various vendors. Three of them had a buffer size of 24 bytes
// on 64 bit systems. This also allows storing a std::string or std::vector inside the small buffer (once the compiler
// has full support of trivially_relocatable annotations).
//
// trivially-destructible bit: This allows us to keep the overall binary size smaller because we don't have to store
// a pointer to a noop function inside the vtable. It also avoids loading the vtable during destruction, potentially
// resulting in fewer cache misses. The downside is that calling the function now also requires setting the lower bits
// of the pointer to zero, but this is a very fast operation on modern CPUs.
//
// interaction with copyable_function: When converting a copyable_function into a move_only_function we want to avoid
// wrapping the copyable_function inside the move_only_function to avoid a double indirection. Instead, we copy the
// small buffer and use copyable_function's vtable.

// NOLINTBEGIN(readability-duplicate-include)
#  define _LIBCPP_IN_MOVE_ONLY_FUNCTION_H

#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF &
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF &&
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_CV const
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_CV const
#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF &
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_CV const
#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF &&
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT true
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT true
#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF &
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT true
#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF &&
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT true
#  define _LIBCPP_MOVE_ONLY_FUNCTION_CV const
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT true
#  define _LIBCPP_MOVE_ONLY_FUNCTION_CV const
#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF &
#  include <__functional/move_only_function_impl.h>

#  define _LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT true
#  define _LIBCPP_MOVE_ONLY_FUNCTION_CV const
#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF &&
#  include <__functional/move_only_function_impl.h>

#  undef _LIBCPP_IN_MOVE_ONLY_FUNCTION_H
// NOLINTEND(readability-duplicate-include)

#endif // _LIBCPP_STD_VER >= 23 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

#endif // _LIBCPP___FUNCTIONAL_MOVE_ONLY_FUNCTION_H
