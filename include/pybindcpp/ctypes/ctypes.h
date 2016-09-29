// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef PYBINDCPP_CTYPES_H
#define PYBINDCPP_CTYPES_H

#include <Python.h>

#include <string>
#include <map>
#include <typeindex>

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

template<class Ret, class... Args>
PyObject *
fun(REGFUNCTYPE reg, Ret(*func)(Args...)) {
  const int func_signature[] = {
      ctype_map.at(typeid(Ret)),
      ctype_map.at(typeid(Args))...
  };
  constexpr auto func_signature_size = 1 + sizeof...(Args);
  auto obj = reg(reinterpret_cast<void *>(func), func_signature, func_signature_size);
  return obj;
}

}

#endif //PYBINDCPP_CTYPES_H
