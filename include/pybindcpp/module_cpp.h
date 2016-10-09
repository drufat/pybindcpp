// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef MODULE_CPP_H
#define MODULE_CPP_H

#include <Python.h>
#include <functional>

#include "module_cpp_imp.h"

namespace pybindcpp {

template<class T>
PyObject *var(T t) {
  return build_value<T>(t);
}

template<class T>
PyObject *fun(T t) {
  return fun_trait<T>::obj(t);
}

struct ExtModule {
  PyObject *self;

  ExtModule(PyObject *obj)
      :
      self(obj) {
  }

  template<class T>
  PyObject *var2obj(T t) const {
    return build_value<T>(t);
  }

  template<class T>
  PyObject *fun2obj(T t) const {
    return fun_trait<T>::obj(t);
  }

  void add(const char *name, PyObject *obj) {
    PyModule_AddObject(self, name, obj);
  }

  template<class T>
  void var(const char *name, T t) {
    add(name, var2obj<T>(t));
  }

  template<class T>
  void fun(const char *name, T t) {
    add(name, fun2obj<T>(t));
  }

  template<class T>
  void varargs(const char *name, T t) {
    add(name, pybindcpp::varargs(t));
  }

};

namespace {

namespace __hidden__ {

inline
void
print(PyObject *obj) {
  PyObject_Print(obj, stdout, Py_PRINT_RAW);
}

struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "name",
    NULL,
    -1,
    NULL
};

} // end __hidden__ namespace

PyObject *
module_init(const char *name, std::function<void(ExtModule &)> exec) {
  using namespace __hidden__;

  moduledef.m_name = name;

  auto self = PyModule_Create(&moduledef);
  if (self == NULL) return NULL;

  ExtModule m(self);

  exec(m);

  return self;
}

} // end anonymous namespace

} // end python namespace

#define PYMODULE_INIT(name, exec)             \
PyMODINIT_FUNC                                \
PyInit_##name() {                             \
  return pybindcpp::module_init(#name, exec); \
}                                             \

#endif // MODULE_CPP_H
