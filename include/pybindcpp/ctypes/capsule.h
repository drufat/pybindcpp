#ifndef PYBINDCPP_CAPSULE_H
#define PYBINDCPP_CAPSULE_H

#include "pybindcpp/ctypes/types.h"

namespace pybindcpp {

template<class T>
T
capsule(const char *module, const char *attr) {

  auto mod = PyImport_ImportModule(module);
  if (!mod) throw;

  auto cap = PyObject_GetAttrString(mod, attr);
  if (!cap) throw;

  auto pnt = PyCapsule_GetPointer(cap, nullptr);
  if (!pnt) throw;

  T reg = reinterpret_cast<T>(pnt);

  Py_DecRef(cap);
  Py_DecRef(mod);

  return reg;
}

template<class Ret, class ...Args>
struct py_function {

  PyObject *m_ptr;
  Ret (*f_ptr)(Args...);

  py_function(const char *module, const char *name) {

    using F = func_trait<decltype(f_ptr)>;
    auto sign = F::value();
    auto size = F::size;

//    auto cfuncify = capsule<PyObject *(*)(const char *, const char *, int *, size_t)>("pybindcpp.register", "c_cfuncify_cap");
//    m_ptr = cfuncify(module, name, _signature, size);

  }

  Ret operator()(Args... args) {

  }
};

}

#endif //PYBINDCPP_CAPSULE_H
