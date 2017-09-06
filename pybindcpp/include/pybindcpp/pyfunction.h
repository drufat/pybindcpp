#ifndef PYBINDCPP_CAPSULE_H
#define PYBINDCPP_CAPSULE_H

#include "pybindcpp/func_trait.h"

namespace pybindcpp {

template <class F>
F
capsule(const char* module, const char* attr)
{

  auto p = api->get_capsule(module, attr);
  F* f = static_cast<F*>(p);
  return *f;
}

template <class F>
F
c_function(const char* module, const char* attr)
{

  auto p = api->get_cfunction(module, attr);
  F* f = static_cast<F*>(p);
  return *f;
}

template <class F>
struct py_function;

template <class Ret, class... Args>
struct py_function<Ret(Args...)>
{

  PyObject* m_ptr;
  Ret (*f_ptr)(Args...);

  py_function(const py_function& other)
  {
    f_ptr = other.f_ptr;
    m_ptr = other.m_ptr;
    Py_IncRef(m_ptr);
  }

  py_function& operator=(const py_function& other) = delete;
  py_function& operator=(py_function&& other) = delete;

  py_function(const char* module, const char* attr)
  {
    using F = decltype(f_ptr);

    auto cfunctype = func_trait<F>::pyctype();

    auto cfunc = api->get_pyfunction(module, attr, cfunctype);

    Py_DecRef(cfunctype);

    auto ptr = api->get_addr(cfunc);

    F* f = static_cast<F*>(ptr);
    f_ptr = *f;
    m_ptr = cfunc;
  }

  ~py_function() { Py_DecRef(m_ptr); }

  Ret operator()(Args... args) { return f_ptr(args...); }
};
}

#endif // PYBINDCPP_CAPSULE_H
