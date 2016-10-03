// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef MODULE_H
#define MODULE_H

#include <Python.h>

#include <vector>
#include <list>
#include <map>
#include <tuple>
#include <functional>
#include <typeinfo>
#include <typeindex>
#include <string>
#include <sstream>
#include <iostream>
#include <memory>

#include "pybindcpp/types.h"
#include "pybindcpp/capsule.h"
#include "pybindcpp/storage.h"

namespace pybindcpp {

struct ExtModule {
  PyObject *self;

  ExtModule(PyObject *obj)
      :
      self(obj) {
  }

  void add(std::string name, PyObject *obj) {
    auto name_ = store(name);
    PyModule_AddObject(self, name_->c_str(), obj);
  }

  template<class T>
  void var(std::string name, T &&t) {
    add(name, pybindcpp::var<T>(std::forward<T>(t)));
  }

  template<class T>
  void fun(std::string name, T &&t) {
    add(name, pybindcpp::fun<T>(std::forward<T>(t)));
  }

  template<class T>
  void varargs(std::string name, T &&t) {
    add(name, pybindcpp::varargs(std::forward<T>(t)));
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
module_init(std::string name, std::function<void(ExtModule &)> exec) {
  using namespace __hidden__;

  moduledef.m_name = store(name)->c_str();

  auto self = PyModule_Create(&moduledef);
  if (self == NULL) return NULL;

  ExtModule m(self);

  exec(m);

  return self;
}

} // end anonymous namespace

} // end python namespace

#endif // MODULE_H
