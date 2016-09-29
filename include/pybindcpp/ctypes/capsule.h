#ifndef PYBINDCPP_CAPSULE_H
#define PYBINDCPP_CAPSULE_H

#include "pybindcpp/ctypes/types.h"

namespace pybindcpp {

template<class T>
T
cap(const char *module, const char *attr) {

  auto mod = PyImport_ImportModule(module);
  if (!mod) return NULL;

  auto cap = PyObject_GetAttrString(mod, attr);
  if (!cap) return NULL;

//  const auto name = std::string(module) + "." + std::string(attr);
  auto pnt = PyCapsule_GetPointer(cap, nullptr);
  if (!pnt) return NULL;

  T reg = reinterpret_cast<T>(pnt);

  Py_DecRef(cap);
  Py_DecRef(mod);

  return reg;
}

}

#endif //PYBINDCPP_CAPSULE_H
