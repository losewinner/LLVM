// RUN: %clang_cc1 %s -verify

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


template <class ...T>
using Twice = TypeList<T..., T...>;

static_assert(__is_same(__type_list_dedup<Twice, int, double, int>, TypeList<int, double, int, double>));


template <int...> struct IntList;
// Wrong kinds of template arguments.
__type_list_dedup<IntList>* wrong_template; // expected-error {{template template argument has different template parameters than its corresponding template template parameter}}
                                            // expected-note@-3 {{template parameter has a different kind in template argument}}
__type_list_dedup<TypeList, 1, 2, 3>* wrong_template_args; // expected-error  {{template argument for template type parameter must be a type}}
                                                           // expected-note@* {{previous template template parameter is here}}
                                                           // expected-note@* {{template parameter from hidden source}}
__type_list_dedup<> not_enough_args; // expected-error {{too few template arguments for template '__type_list_dedup'}}
                                     // expected-note@* {{template declaration from hidden source}}
__type_list_dedup missing_template_args; // expected-error {{use of template '__type_list_dedup' requires template arguments}}
                                         // expected-note@* {{template declaration from hidden source}}

// Direct recursive use will fail because the signature of template parameters does not match.
// The intention for this test is to anticipate a failure mode where compiler may start running the deduplication recursively, going into
// an infinite loop without diagnosing an error.
// Currently, the type checking prevents us from it, but if the builtin becomes more generic, we need to be aware of it.
__type_list_dedup<__type_list_dedup, int, int, double>; // expected-error {{template template argument has different template parameters than its corresponding template template parameter}}
                                                        // expected-note@* {{template parameter has a different kind in template argument}}
                                                        // expected-note@* {{previous template template parameter is here}}

// FIXME: tests for locations of template arguments, ideally we should point into the original locations of the template arguments.
