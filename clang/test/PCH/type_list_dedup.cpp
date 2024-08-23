// RUN: %clang_cc1 -std=c++14 -x c++-header %s -emit-pch -o %t.pch
// RUN: %clang_cc1 -std=c++14 -x c++ /dev/null -include-pch %t.pch

// RUN: %clang_cc1 -std=c++14 -x c++-header %s -emit-pch -fpch-instantiate-templates -o %t.pch
// RUN: %clang_cc1 -std=c++14 -x c++ /dev/null -include-pch %t.pch

template <template <class...> class Templ, class...Types>
using TypeListDedup = __type_list_dedup<Templ, Types...>;

template <class ...Ts>
struct TypeList {};

template <int i>
struct X {};

void fn1() {
  TypeList<int, double> l1 = TypeListDedup<TypeList, int, double, int>{};
  TypeList<> l2 = TypeListDedup<TypeList>{};
  TypeList<X<0>, X<1>> x1 = TypeListDedup<TypeList, X<0>, X<1>, X<0>, X<1>>{};
}
