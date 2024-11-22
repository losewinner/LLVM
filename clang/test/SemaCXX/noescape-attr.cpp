// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-no-diagnostics

template<typename T>
void test1(T __attribute__((noescape)) arr, int size);

void test2(int __attribute__((noescape)) arr, int size);

#if !__has_feature(attribute_noescape_nonpointer)
  #error "attribute_noescape_nonpointer should be supported"
#endif
