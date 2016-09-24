// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include <iostream>
#include <tuple>
#include <memory>
//#include <experimental/tuple>

template<class T>
constexpr T &lvalue(T &&v) {
  return v;
}

template<typename F, typename Tuple, size_t... I>
auto apply_impl(F &&f, Tuple &&t, std::index_sequence<I...>) {
  return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}

template<typename F, typename Tuple>
auto apply(F &&f, Tuple &&t) {
  using Indices = std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
  return apply_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices{});
}

constexpr
auto multiply(int a, int b, int c) {
  return a * b * c;
}

namespace detail {
template<char...  args>
struct str {
  static
  constexpr const char chars[sizeof...(args) + 1] = {args..., '\0'};
};

template<char...  args>
constexpr const char str<args...>::chars[sizeof...(args) + 1];

template<char...  S0, char...  S1>
constexpr str<S0..., S1...> operator+(str<S0...>, str<S1...>) {
  return {};
}
}

struct StrConst {

  const char *const str_ptr;
  const unsigned int str_size;

  template<std::size_t N>
  constexpr StrConst(const char(&a)[N])
      : str_ptr(a),
        str_size(N - 1) {
  }

  constexpr char operator[](const std::size_t n) const {
//        if (!(n<str_size)){
//            throw std::out_of_range(""); //g++ 5.2 cannot compile this
//        }
    return str_ptr[n];
  }

  constexpr bool operator==(StrConst other) const {
    if (str_size != other.str_size)
      return false;
    for (auto n = 0; n < str_size; n++)
      if (str_ptr[n] != other.str_ptr[n])
        return false;
    return true;
  }

  constexpr std::size_t size() const {
    return str_size;
  }
};

int main() {

  constexpr auto chars = StrConst("Hello, World!");
  using Indices = std::make_index_sequence<chars.size()>;

  constexpr auto str1 = StrConst(detail::str<'h', 'e', 'l', 'l', 'o'>().chars);
  static_assert(str1 == "hello", "");

  constexpr auto str2 = StrConst(
      (
          detail::str<'h', 'e'>() +
              detail::str<'l', 'l', 'o'>()
      ).chars
  );
  static_assert(str2 == "hello", "");

  constexpr StrConst str3 = detail::str<'h'>::chars;
  static_assert(str3 == "h", "");

  constexpr auto s1 = StrConst("hello");
  constexpr auto s2 = StrConst(detail::str<'h', 'e', 'l', 'l', 'o'>().chars);
  static_assert(s1 == s2, "");
  static_assert(s2 == s1, "");

  constexpr auto my_string = StrConst("Hello World!");
  static_assert(my_string.size() == 12, "");
  static_assert(my_string[4] == 'o', "");
  constexpr auto my_other_string = StrConst(my_string);
  static_assert(my_string == my_other_string, "");

  int a, b, c, d, e, f;

  auto add = [](int a, int b, int c) {
    return a + b + c;
  };

  auto tup1 = std::make_tuple(4, 5, 6);
  lvalue(std::tie(a, b, c)) = tup1;

  std::cout << a << std::endl
            << b << std::endl
            << c << std::endl;

  std::cout << apply(add, tup1) << std::endl;
  std::cout << apply(multiply, tup1) << std::endl;

  //std::cout << std::experimental::apply(add, tup1) << std::endl;
  //std::cout << std::experimental::apply(multiply, tup1) << std::endl;

  auto tup2 = std::make_tuple(1, 2, 3, std::make_tuple(4, 5, 6));
  lvalue(std::tie(a, b, c, lvalue(std::tie(d, e, f)))) = tup2;
  std::cout << a << std::endl
            << b << std::endl
            << c << std::endl
            << d << std::endl
            << e << std::endl
            << f << std::endl;

  auto ptr = std::make_shared<std::function<int(int, int)>>(
      [](int x, int y) -> int {
        return x + y;
      });

  auto ptr1 = ptr;
  decltype(ptr) ptr2(ptr);
  auto ptr3 = new decltype(ptr)(ptr);

  std::cout << (*ptr)(2, 3) << std::endl;
  std::cout << "Use count:" << " "
            << ptr.use_count() << " "
            << ptr1.use_count() << " "
            << ptr2.use_count() << " "
            << ptr3->use_count() << std::endl;

}
