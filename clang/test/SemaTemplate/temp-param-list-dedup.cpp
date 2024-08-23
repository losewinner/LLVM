// RUN: %clang_cc1 %s -verify
// expected-no-diagnostics

template <typename...> struct TypeList;

static_assert(__is_same(
  __type_list_dedup<TypeList, int, int*, int, double, float>,
  TypeList<int, int*, double, float>));

template <template<typename ...> typename Templ, typename ...Types>
struct Dependent {
  using empty_list = __type_list_dedup<Templ>;
  using same = __type_list_dedup<Templ, Types...>;
  using twice = __type_list_dedup<Templ, Types..., Types...>;
  using dep_only_types = __type_list_dedup<TypeList, Types...>;
  using dep_only_template = __type_list_dedup<Templ, int, double, int>;
}; 

static_assert(__is_same(Dependent<TypeList>::empty_list, TypeList<>));
static_assert(__is_same(Dependent<TypeList>::same, TypeList<>));
static_assert(__is_same(Dependent<TypeList>::twice, TypeList<>));
static_assert(__is_same(Dependent<TypeList>::dep_only_types, TypeList<>));
static_assert(__is_same(Dependent<TypeList>::dep_only_template, TypeList<int, double>));

static_assert(__is_same(Dependent<TypeList, int*, double*, int*>::empty_list, TypeList<>));
static_assert(__is_same(Dependent<TypeList, int*, double*, int*>::same, TypeList<int*, double*>));
static_assert(__is_same(Dependent<TypeList, int*, double*, int*>::twice, TypeList<int*, double*>));
static_assert(__is_same(Dependent<TypeList, int*, double*, int*>::dep_only_types, TypeList<int*, double*>));
static_assert(__is_same(Dependent<TypeList, int*, double*, int*>::dep_only_template, TypeList<int, double>));

// FIXME: tests for using template type alias as a template
// FIXME: tests for errors
// FIXME: tests for locations
