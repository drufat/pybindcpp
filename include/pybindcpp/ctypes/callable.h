#ifndef PYBINDCPP_CALLABLE_H
#define PYBINDCPP_CALLABLE_H

#include "pyfunction.h"

namespace pybindcpp {

template<class F, class Ret, class... Args>
Ret apply(PyObject *capsule, Args... args) {
  auto f = capsule_get<F>(capsule);
  return (*f)(args...);
};

template<class F>
struct callable_trait;

template<class Ret, class... Args>
struct callable_trait<Ret(*)(Args...)> {
  static
  PyObject *
  get(const API &api, Ret(*func)(Args...)) {

//    auto reg = capsule<PyObject *(*)(void *, PyObject *)>("pybindcpp.bind", "register_cap");
//    auto reg = c_function<PyObject *(*)(void *, PyObject *)>("pybindcpp.bind", "c_register");
//    auto reg = py_function<PyObject *(void *, PyObject *)>("pybindcpp.bind", "register");
    auto reg = api.register_;

    using F = func_trait<decltype(func)>;
    auto func_type = F::pyctype(api);
    auto o = reg(static_cast<void *>(&func), func_type);
    Py_DecRef(func_type);
    return o;
  }
};

template<class F>
struct callable_trait
    : public callable_trait<decltype(&F::operator())> {
};

template<class F, class Ret, class... Args>
struct callable_trait<Ret(F::*)(Args...) const> {
  static
  PyObject *
  get(const API &api, F func) {
    auto capsule = capsule_new(std::make_shared<F>(func));

    auto a = &apply<F, Ret, Args...>;
    auto callable = callable_trait<decltype(a)>::get(api, a);

    auto rslt = api.apply(callable, capsule);

    Py_DecRef(callable);
    Py_DecRef(capsule);

    return rslt;
  }
};

template<class F>
PyObject *varargs(const API &api, F func) {
  auto v = callable_trait<F>::get(api, func);
  return api.vararg(v);
}

}

#endif //PYBINDCPP_CALLABLE_H
