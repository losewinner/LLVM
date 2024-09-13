// RUN: %clang_cc1 -std=c++14 -x c++-header %s -emit-pch -o %t.pch
// RUN: %clang_cc1 -std=c++14 -x c++ /dev/null -include-pch %t.pch

// RUN: %clang_cc1 -std=c++14 -x c++-header %s -emit-pch -fpch-instantiate-templates -o %t.pch
// RUN: %clang_cc1 -std=c++14 -x c++ /dev/null -include-pch %t.pch

template <template <class...> class Templ, class...Types>
using TypePackDedup = __builtin_type_pack_dedup<Templ, Types...>;

template <class ...Ts>
struct TypeList {};

template <int i>
struct X {};

void fn1() {
  TypeList<int, double> l1 = TypePackDedup<TypeList, int, double, int>{};
  TypeList<> l2 = TypePackDedup<TypeList>{};
  TypeList<X<0>, X<1>> x1 = TypePackDedup<TypeList, X<0>, X<1>, X<0>, X<1>>{};
}

using Int = int;
using CInt = const Int;
using IntArray = Int[10];
using CIntArray = Int[10];
using IntPtr = int*;
using CIntPtr = const int*;
static_assert(
  __is_same(
    __builtin_type_pack_dedup<TypeList,
      Int, int,
      const int, const Int, CInt, const CInt,
      IntArray, Int[10], int[10],
      const IntArray, const int[10], CIntArray, const CIntArray,
      IntPtr, int*,
      const IntPtr, int* const,
      CIntPtr, const int*,
      const IntPtr*, int*const*,
      CIntPtr*, const int**,
      const CIntPtr*, const int* const*
    >,
    TypeList<int, const int, int[10], const int [10], int*, int* const, const int*, int*const *, const int**, const int*const*>),
  "");
