#ifndef PYBINDCPP_CALLABLE_H
#define PYBINDCPP_CALLABLE_H

#include "capsule.h"
#include "pyfunction.h"

namespace pybindcpp {

template <class F>
struct callable_trait;

template <class Ret, class... Args>
struct callable_trait<Ret (*)(Args...)>
{
  static PyObject* get(Ret (*func)(Args...))
  {
    using F = func_trait<decltype(func)>;

    auto func_type = F::pyctype();
    auto rslt = api->register_(static_cast<void*>(&func), func_type);

    Py_DecRef(func_type);
    return rslt;
  }
};

template <class F>
struct callable_trait : public callable_trait<decltype(&F::operator())>
{
};

template <class F, class Ret, class... Args>
struct callable_trait<Ret (F::*)(Args...) const>
{
  static PyObject* get(F func)
  {
    auto capsule = capsule_new(std::make_shared<F>(func));

    auto c = &capsule_call<F, Ret, Args...>;
    auto callable = callable_trait<decltype(c)>::get(c);

    auto rslt = api->apply(callable, capsule);

    Py_DecRef(callable);
    Py_DecRef(capsule);

    return rslt;
  }
};

template <class F>
PyObject*
varargs(F func)
{
  PyObject* v = callable_trait<F>::get(func);
  return api->vararg(v);
}
}

#endif // PYBINDCPP_CALLABLE_H
