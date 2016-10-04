#ifndef PYBINDCPP_CAPSULE_H
#define PYBINDCPP_CAPSULE_H

#include "pybindcpp/ctypes/types.h"
#include "pybindcpp/ctypes/func_trait.h"

namespace pybindcpp {

template<class F>
F
capsule(const char *module, const char *attr) {

  auto mod = PyImport_ImportModule(module);
  if (!mod) throw std::runtime_error("Cannot load module.");

  auto cap = PyObject_GetAttrString(mod, attr);
  if (!cap) throw std::runtime_error("Cannot load attribute.");

  auto pnt = PyCapsule_GetPointer(cap, nullptr);
  if (!pnt) throw std::runtime_error("Cannot read capsule pointer.");

  F *f = static_cast<F *>(pnt);

  Py_DecRef(cap);
  Py_DecRef(mod);

  return *f;
}

template<class F>
F
c_function(const char *module, const char *attr) {

  auto mod = PyImport_ImportModule(module);
  if (!mod) throw std::runtime_error("Cannot load module.");

  auto cfunc = PyObject_GetAttrString(mod, attr);
  if (!cfunc) throw std::runtime_error("Cannot load attribute.");

  auto ctypes = PyImport_ImportModule("ctypes");
  if (!ctypes) throw;

  auto addressof = PyObject_GetAttrString(ctypes, "addressof");
  if (!addressof) throw;

  auto addr = PyObject_CallFunctionObjArgs(addressof, cfunc, nullptr);
  if (!addr) throw;

  auto ptr = PyLong_AsVoidPtr(addr);

  F *f = static_cast<F *>(ptr);

  Py_DecRef(addr);
  Py_DecRef(addressof);
  Py_DecRef(ctypes);
  Py_DecRef(cfunc);
  Py_DecRef(mod);

  return *f;
}

template<class F>
struct py_function;

template<class Ret, class ...Args>
struct py_function<Ret(Args...)> {

  PyObject *m_ptr;
  Ret (*f_ptr)(Args...);

  py_function(const py_function& other) {
    f_ptr = other.f_ptr;
    m_ptr = other.m_ptr;
    Py_IncRef(m_ptr);
  };

  py_function& operator=(const py_function& other) = delete;
  py_function& operator=(py_function&& other) = delete;

  py_function(const char *module, const char *attr) {
    using F = decltype(f_ptr);
    using trait = func_trait<F>;

    auto mod = PyImport_ImportModule(module);
    if (!mod) throw std::runtime_error("Cannot load module.");

    auto func = PyObject_GetAttrString(mod, attr);
    if (!func) throw std::runtime_error("Cannot load attribute.");

    auto cfunctype = trait::pyctype();

    auto cfunc = PyObject_CallFunctionObjArgs(cfunctype, func, nullptr);
    if (!cfunc) throw;

    auto ctypes = PyImport_ImportModule("ctypes");
    if (!ctypes) throw;

    auto addressof = PyObject_GetAttrString(ctypes, "addressof");
    if (!addressof) throw;

    auto addr = PyObject_CallFunctionObjArgs(addressof, cfunc, nullptr);
    if (!addr) throw;

    auto ptr = PyLong_AsVoidPtr(addr);

    F *f = static_cast<F *>(ptr);

    Py_DecRef(addr);
    Py_DecRef(addressof);
    Py_DecRef(ctypes);
    Py_DecRef(cfunctype);
    Py_DecRef(func);
    Py_DecRef(mod);

    f_ptr = *f;
    m_ptr = cfunc;
  }

  ~py_function() {
    Py_DecRef(m_ptr);
  }

  Ret operator()(Args... args) {
    return f_ptr(args...);
  }
};

}

#endif //PYBINDCPP_CAPSULE_H
