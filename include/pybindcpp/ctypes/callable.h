#ifndef PYBINDCPP_CALLABLE_H
#define PYBINDCPP_CALLABLE_H

#include "pybindcpp/ctypes/pyfunction.h"

namespace pybindcpp {

template<class F>
struct callable_trait;

template<class F>
struct callable_trait
    : public callable_trait<decltype(&F::operator())> {
};

template<class F, class Ret, class... Args>
struct callable_trait<Ret(F::*)(Args...) const> {
  static
  PyObject *
  get(F f) {
    Py_RETURN_NONE;
  }
};

template<class Ret, class... Args>
struct callable_trait<Ret(*)(Args...)> {
  static
  PyObject *
  get(Ret(*func)(Args...)) {
    using F = func_trait<decltype(func)>;
    auto func_type = F::pyctype();

//  auto reg = capsule<PyObject *(*)(void *, PyObject *)>("pybindcpp.bind", "register_cap");
//  auto reg = c_function<PyObject *(*)(void *, PyObject *)>("pybindcpp.bind", "c_register");
    auto reg = py_function<PyObject *(void *, PyObject *)>("pybindcpp.bind", "register");

    auto o = reg((void *) (func), func_type);
    Py_DecRef(func_type);
    return o;
  }
};

};

#endif //PYBINDCPP_CALLABLE_H
