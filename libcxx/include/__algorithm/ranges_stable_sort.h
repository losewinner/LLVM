//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_STABLE_SORT_H
#define _LIBCPP___ALGORITHM_RANGES_STABLE_SORT_H

#include <__algorithm/iterator_operations.h>
#include <__algorithm/make_projected.h>
#include <__algorithm/stable_sort.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/invoke.h>
#include <__functional/ranges_operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__iterator/projected.h>
#include <__iterator/sortable.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/dangling.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_same.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
struct __stable_sort {
  template <class _Iter, class _Sent, class _Comp, class _Proj>
  _LIBCPP_HIDE_FROM_ABI static _Iter __stable_sort_fn_impl(_Iter __first, _Sent __last, _Comp& __comp, _Proj& __proj) {
    auto __last_iter = ranges::next(__first, __last);

    auto&& __projected_comp = std::__make_projected(__comp, __proj);
    constexpr auto __default_comp        = is_same_v<_Comp, ranges::less>;
    constexpr auto __default_proj        = __is_identity<_Proj>::value;
    constexpr auto __integral_value      = is_integral_v<iter_value_t<_Iter>>;
    constexpr auto __integral_projection = __default_proj && __integral_value;
    // constexpr auto __integral_projection = is_integral_v<remove_reference_t<invoke_result_t<_Proj&,
    // iter_value_t<_Iter>>>>;
    // TODO: Support projection in stable_sort
    std::__stable_sort_impl<_RangeAlgPolicy>(
        std::move(__first),
        __last_iter,
        __projected_comp,
        _BoolConstant < __default_comp && __integral_projection > {});

    return __last_iter;
  }

  template <random_access_iterator _Iter, sentinel_for<_Iter> _Sent, class _Comp = ranges::less, class _Proj = identity>
    requires sortable<_Iter, _Comp, _Proj>
  _LIBCPP_HIDE_FROM_ABI _Iter operator()(_Iter __first, _Sent __last, _Comp __comp = {}, _Proj __proj = {}) const {
    return __stable_sort_fn_impl(std::move(__first), std::move(__last), __comp, __proj);
  }

  template <random_access_range _Range, class _Comp = ranges::less, class _Proj = identity>
    requires sortable<iterator_t<_Range>, _Comp, _Proj>
  _LIBCPP_HIDE_FROM_ABI borrowed_iterator_t<_Range>
  operator()(_Range&& __r, _Comp __comp = {}, _Proj __proj = {}) const {
    return __stable_sort_fn_impl(ranges::begin(__r), ranges::end(__r), __comp, __proj);
  }
};

inline namespace __cpo {
inline constexpr auto stable_sort = __stable_sort{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_RANGES_STABLE_SORT_H
