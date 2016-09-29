// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef MODULE_H
#define MODULE_H

#include <Python.h>

#include <functional>

#include "pybindcpp/ctypes/ctypes.h"

namespace pybindcpp {

struct Module {
  const char *name;
  PyObject *self;
  REGFUNCTYPE reg;

  Module(const char *name_) :
      name(name_) {

    auto moduledef = new PyModuleDef(
        {
            PyModuleDef_HEAD_INIT,
            name,     // m_name
            nullptr,  // m_doc
            0,        // m_size
            nullptr,  // m_methods
            nullptr,  // m_slots
            nullptr,  // m_traverse
            nullptr,  // m_clear
            nullptr,  // m_free
        }
    );

    self = PyModule_Create(moduledef);
    if (!self) throw;

    reg = cap<REGFUNCTYPE>("pybindcpp.register", "c_register_cap");
    if (!reg) throw;

  }

  void add(const char *name, PyObject *obj) {
    PyModule_AddObject(self, name, obj);
  }

  template<class F>
  void fun(const char *name, F f) {
    add(name, pybindcpp::fun(reg, f));
  }

};

PyObject *
module_init(const char *name, std::function<void(Module &)> exec) {
  try {
    Module m(name);
    exec(m);
    return m.self;
  } catch (...) {
    return nullptr;
  };
}


}

#endif // MODULE_H
