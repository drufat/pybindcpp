#ifndef PYBINDCPP_CALLABLE_H
#define PYBINDCPP_CALLABLE_H

#include <Python.h>
#include <string>
#include <sstream>

#include "pybindcpp/ctypes/types.h"

namespace pybindcpp {

template<class F>
struct func_trait;

template<class Ret, class... Args>
struct func_trait<Ret(*)(Args...)> {

  static constexpr size_t size = 1 + sizeof...(Args);

  static auto value() {
    const std::array<std::type_index, size> a = {typeid(Ret), typeid(Args)...};
    return a;
  }

  static auto str() {
    std::stringstream ss;
    auto sign = value();
    for (size_t i = 0; i < size; i++) {
      if (i) { ss << ","; }
      ss << ctype_map.at(sign[i]);
    }
    return ss.str();
  }

  static auto pystr() {
    auto s = str();
    return PyBytes_FromString(s.c_str());
  }
  };

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
  get(REGFUNCTYPE reg, Ret(*func)(Args...)) {
    auto sign = func_trait<decltype(func)>::str();
    auto o = reg((void *) (func), sign.c_str());
    return o;
  }
};

};

#endif //PYBINDCPP_CALLABLE_H
