#ifndef PYBINDCPP_CALLABLE_H
#define PYBINDCPP_CALLABLE_H

#include "pyfunction.h"

namespace pybindcpp {

struct Callable {
  std::string type;
  void *func;
  void *data;
};

template <class F> struct callable_trait;

template <class Ret, class... Args> struct callable_trait<Ret (*)(Args...)> {
  static PyObject *get(Ret (*func)(Args...)) {
    using F = func_trait<decltype(func)>;
    auto func_type = F::str();
    auto func_ptr = static_cast<void *>(&func);

    auto rslt = api->func_c(func_ptr, func_type.c_str());
    return rslt;
  }
};

template <class F>
struct callable_trait : public callable_trait<decltype(&F::operator())> {};

template <class F, class Ret, class... Args>
Ret function_call(void *ptr, Args... args) {
  auto f = static_cast<F *>(ptr);
  return (*f)(args...);
};

template <class F, class Ret, class... Args>
struct callable_trait<Ret (F::*)(Args...) const> {
  static PyObject *get(F func) {
    auto func_new = new F(func);
    auto func_ptr = static_cast<void *>(func_new);

    auto c = &function_call<F, Ret, Args...>;
    auto c_ptr = static_cast<void *>(&c);

    auto callable = callable_trait<decltype(c)>::get(c);
    auto rslt = api->func_std(callable, func_ptr);

    Py_DecRef(callable);

    return rslt;
  }
};

} // namespace pybindcpp

#endif // PYBINDCPP_CALLABLE_H
