#ifndef PYBINDCPP_APPLY_H
#define PYBINDCPP_APPLY_H

#include <tuple>
#include <utility>

namespace pybindcpp {

template <typename F, typename Tuple, size_t... I>
decltype(auto)
apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>)
{
  return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}

template <typename F, typename Tuple>
decltype(auto)
apply(F&& f, Tuple&& t)
{
  using Indices =
    std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
  return apply_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices{});
}
}

#endif // PYBINDCPP_APPLY_H
