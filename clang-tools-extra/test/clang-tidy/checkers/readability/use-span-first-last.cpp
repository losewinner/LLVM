// RUN: %check_clang_tidy %s readability-use-span-first-last %t

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
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer span::first() over subspan()
  // CHECK-FIXES: auto sub1 = s.first(3);

  auto sub2 = s.subspan(2);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer span::last() over subspan()
  // CHECK-FIXES: auto sub2 = s.last(s.size() - 2);

  __SIZE_TYPE__ n = 2;
  auto sub3 = s.subspan(0, n);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer span::first() over subspan()
  // CHECK-FIXES: auto sub3 = s.first(n);

  auto sub4 = s.subspan(1, 2);  // No warning
}