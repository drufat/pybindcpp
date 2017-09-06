// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef MODULE_CPP_IMP_H
#define MODULE_CPP_IMP_H

#include <memory>
#include <type_traits>

#include "apply.h"
#include "capsule.h"
#include "cpython_types.h"

namespace pybindcpp {

PyObject*
varargs(std::shared_ptr<VarArg> func)
{
  auto gil = PyGILState_Ensure();

  auto caps = capsule_new(func);
  auto dmod = PyImport_ImportModule("pybindcpp.function");
  auto dfun = PyObject_GetAttrString(dmod, "function");
  auto obj = PyObject_CallFunctionObjArgs(dfun, caps, nullptr);
  Py_DecRef(dfun);
  Py_DecRef(dmod);
  Py_DecRef(caps);

  PyGILState_Release(gil);
  return obj;
}

PyObject*
varargs(VarArg func)
{
  return varargs(std::make_shared<VarArg>(func));
}

template <class... Args>
PyObject*
fun_ptr(std::shared_ptr<std::function<void(Args...)>> f)
{
  return varargs(std::make_shared<VarArg>(

    [f](PyObject* self, PyObject* args) -> PyObject* {

      auto parse = [args](Args&... a) { return arg_parse_tuple(args, a...); };

      std::tuple<Args...> tup;
      if (!apply(parse, tup))
        return NULL;

      try {
        apply(*f, tup);
        Py_RETURN_NONE;
      } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
      }

    }));
}

template <class Ret, class... Args>
PyObject*
fun_ptr(std::shared_ptr<std::function<Ret(Args...)>> f)
{

  return varargs(std::make_shared<VarArg>(

    [f](PyObject* self, PyObject* args) -> PyObject* {

      auto parse = [args](Args&... a) { return arg_parse_tuple(args, a...); };

      std::tuple<Args...> tup;
      if (!apply(parse, tup))
        return NULL;

      try {
        return build_value(apply(*f, tup));
      } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
      }

    }));
}

template <class F>
struct fun_trait;

template <class F>
struct fun_trait : public fun_trait<decltype(&F::operator())>
{
};

template <class F, class Ret, class... Args>
struct fun_trait<Ret (F::*)(Args...) const>
{
  static auto obj(F f)
  {
    static_assert(std::is_class<F>::value, "Requires class.");
    return fun_ptr(std::make_shared<std::function<Ret(Args...)>>(f));
  }
};

template <class Ret, class... Args>
struct fun_trait<Ret (*)(Args...)>
{
  static auto obj(Ret (*f)(Args...))
  {
    static_assert(std::is_pointer<decltype(f)>::value,
                  "Requires function pointer.");
    return fun_ptr(std::make_shared<std::function<Ret(Args...)>>(f));
  }
};

template <class Ret, class Class, class... Args>
PyObject*
method(Ret (Class::*f)(Args...))
{
  Py_RETURN_NONE;
}

template <class T, class Class>
PyObject*
method(T Class::*m)
{
  Py_RETURN_NONE;
}

template <typename Class, typename... Params>
PyObject*
build_constructor_(Class (*)(Params...))
{
  Py_RETURN_NONE;
}

template <class T>
PyObject*
constructor()
{
  T* f = nullptr;
  return build_constructor_(f);
}
}

#endif // MODULE_CPP_IMP_H
