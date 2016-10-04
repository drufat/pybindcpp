#ifndef PYBINDCPP_CAPSULE_H
#define PYBINDCPP_CAPSULE_H

#include "pybindcpp/ctypes/types.h"

namespace pybindcpp {

template<class T>
T
capsule(const char *module, const char *attr) {

  auto mod = PyImport_ImportModule(module);
  if (!mod) throw std::runtime_error("Cannot load module.");

  auto cap = PyObject_GetAttrString(mod, attr);
  if (!cap) throw std::runtime_error("Cannot load attribute.");

  auto pnt = PyCapsule_GetPointer(cap, nullptr);
  if (!pnt) throw std::runtime_error("Cannot read capsule pointer.");

  T reg = reinterpret_cast<T>(pnt);

  Py_DecRef(cap);
  Py_DecRef(mod);

  return reg;
}

template<class F>
struct py_function;

template<class Ret, class ...Args>
struct py_function<Ret(Args...)> {

  PyObject *func_type;
  PyObject *m_ptr;
  Ret (*f_ptr)(Args...);

  py_function(const char *module, const char *name) {
    using F = func_trait<decltype(f_ptr)>;

    PyObject *func_type = F::pyctype();

    auto mod = PyImport_ImportModule(module);
    if (!mod) throw;

    auto func = PyObject_GetAttrString(mod, name);
    if (!func) throw;

    auto cfunc = PyObject_CallFunctionObjArgs(func_type, func, nullptr);
    if (!cfunc) throw;

    m_ptr = cfunc;

    auto tovoid = capsule<void *(*)(PyObject *)>("pybindcpp.bind", "c_tovoid_cap");
    f_ptr = (decltype(f_ptr)) tovoid(cfunc);

  }

  ~py_function() {
//    Py_DecRef(m_ptr);
  }

  Ret operator()(Args... args) {
    return f_ptr(args...);
  }
};

}

#endif //PYBINDCPP_CAPSULE_H
