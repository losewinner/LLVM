// RUN: %check_clang_tidy -std=c++20 %s readability-use-span-first-last %t

namespace std {
template <typename T>
class span {
  T* ptr;
  __SIZE_TYPE__ len;

public:
  span(T* p, __SIZE_TYPE__ l) : ptr(p), len(l) {}
  
  span<T> subspan(__SIZE_TYPE__ offset) const {
    return span(ptr + offset, len - offset);
  }
  
  span<T> subspan(__SIZE_TYPE__ offset, __SIZE_TYPE__ count) const {
    return span(ptr + offset, count);
  }

  span<T> first(__SIZE_TYPE__ count) const {
    return span(ptr, count);
  }

  span<T> last(__SIZE_TYPE__ count) const {
    return span(ptr + (len - count), count);
  }

  __SIZE_TYPE__ size() const { return len; }
};
} // namespace std

void test() {
  int arr[] = {1, 2, 3, 4, 5};
  std::span<int> s(arr, 5);

  auto sub1 = s.subspan(0, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::first()' over 'subspan()'
  // CHECK-FIXES: auto sub1 = s.first(3);

  auto sub2 = s.subspan(s.size() - 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto sub2 = s.last(2);

  __SIZE_TYPE__ n = 2;
  auto sub3 = s.subspan(0, n);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::first()' over 'subspan()'
  // CHECK-FIXES: auto sub3 = s.first(n);

  auto sub4 = s.subspan(1, 2);  // No warning
  auto sub5 = s.subspan(2);     // No warning


#define ZERO 0
#define TWO 2
#define SIZE_MINUS(s, n) s.size() - n
#define MAKE_SUBSPAN(obj, n) obj.subspan(0, n)
#define MAKE_LAST_N(obj, n) obj.subspan(obj.size() - n)

  auto sub6 = s.subspan(SIZE_MINUS(s, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto sub6 = s.last(2);

  auto sub7 = MAKE_SUBSPAN(s, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: prefer 'span::first()' over 'subspan()'
  // CHECK-FIXES: auto sub7 = s.first(3);

  auto sub8 = MAKE_LAST_N(s, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto sub8 = s.last(2);

}

template <typename T>
void testTemplate() {
  T arr[] = {1, 2, 3, 4, 5};
  std::span<T> s(arr, 5);

  auto sub1 = s.subspan(0, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::first()' over 'subspan()'
  // CHECK-FIXES: auto sub1 = s.first(3);

  auto sub2 = s.subspan(s.size() - 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto sub2 = s.last(2);

  __SIZE_TYPE__ n = 2;
  auto sub3 = s.subspan(0, n);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::first()' over 'subspan()'
  // CHECK-FIXES: auto sub3 = s.first(n);

  auto sub4 = s.subspan(1, 2);  // No warning
  auto sub5 = s.subspan(2);     // No warning
}

// Test instantiation
void testInt() {
  testTemplate<int>();
}