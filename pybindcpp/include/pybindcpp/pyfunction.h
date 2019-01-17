#ifndef PYBINDCPP_CAPSULE_H
#define PYBINDCPP_CAPSULE_H

#include "pybindcpp/func_trait.h"

namespace pybindcpp {

template <class F> F c_function(const char *module, const char *attr) {

  auto p = api->get_cfunction(module, attr);
  F *f = static_cast<F *>(p);
  return *f;
}

template <class F> struct py_function;

template <class Ret, class... Args> struct py_function<Ret(Args...)> {

  Ret (*f_ptr)(Args...);

  py_function(const char *module, const char *attr) {

    using F = decltype(f_ptr);

    auto cfunctype = func_trait<F>::str();

    auto ptr = api->get_pyfunction(module, attr, cfunctype.c_str());

    F *f = static_cast<F *>(ptr);
    f_ptr = *f;
  }

  Ret operator()(Args... args) { return f_ptr(args...); }

  py_function &operator=(const py_function &other) = delete;
  py_function &operator=(py_function &&other) = delete;
};
} // namespace pybindcpp

#endif // PYBINDCPP_CAPSULE_H
