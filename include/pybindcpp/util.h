// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef UTIL_H
#define UTIL_H

#include <typeindex>
#include <string>
#include <sstream>

namespace pybindcpp {

template<typename... Args>
std::string stringer(Args const &... args) {
  std::ostringstream stream;
  using List= int[];
  (void) List{0, (stream << args, 0)...};

  return stream.str();
}

template<typename F, typename Tuple, size_t... I>
decltype(auto) apply_impl(F &&f, Tuple &&t, std::index_sequence<I...>) {
  return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}

template<typename F, typename Tuple>
decltype(auto) apply(F &&f, Tuple &&t) {
  using Indices = std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
  return apply_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices{});
}

template<class T>
struct function_traits
    : public function_traits<decltype(&T::operator())> {
};

template<class T, class Ret, class... Args>
struct function_traits<Ret(T::*)(Args...) const>
    : public function_traits<Ret(Args...)> {
};

template<class Ret, class... Args>
struct function_traits<Ret(*)(Args...)>
    : public function_traits<Ret(Args...)> {
};

template<class Ret, class... Args>
struct function_traits<Ret(Args...)> {
  static constexpr std::size_t arity = sizeof...(Args);

  using return_type = Ret;

  template<size_t i>
  struct arg {
    static_assert(i < arity);
    using type = typename std::tuple_element<i, std::tuple<Args...>>::type;
  };
};

}

#endif // UTIL_H
