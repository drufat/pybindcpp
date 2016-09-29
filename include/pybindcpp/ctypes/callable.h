#ifndef PYBINDCPP_CALLABLE_H
#define PYBINDCPP_CALLABLE_H

#include <Python.h>

#include "pybindcpp/ctypes/types.h"

namespace pybindcpp {

template<class F>
struct callable_traits;

template<class F>
struct callable_traits
    : public callable_traits<decltype(&F::operator())> {
};

template<class F, class Ret, class... Args>
struct callable_traits<Ret(F::*)(Args...) const> {
  static
  PyObject *
  get(F f) {
    Py_RETURN_NONE;
  }
};

template<class Ret, class... Args>
struct callable_traits<Ret(*)(Args...)> {
  static
  PyObject *
  get(REGFUNCTYPE reg, Ret(*func)(Args...)) {
    const int func_signature[] = {
        ctype_map.at(typeid(Ret)),
        ctype_map.at(typeid(Args))...
    };
    constexpr auto func_signature_size = 1 + sizeof...(Args);
    auto o = reg(reinterpret_cast<void *>(func), func_signature, func_signature_size);
    return o;
  }
};

};

#endif //PYBINDCPP_CALLABLE_H
