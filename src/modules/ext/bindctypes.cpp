// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include <Python.h>

#include "pybindcpp/ctypes/types.h"
#include "pybindcpp/ctypes/ctypes.h"

using namespace pybindcpp;

extern "C"
int set_string(char c, int size, char *buffer) {
  for (int i = 0; i < size; i++) {
    buffer[i] = c;
  }
  return 0;
}

int add(int x, int y) {
  return x + y;
}

double add_d(double x, double y) {
  return x + y;
}

int minus(int x, int y) {
  return x - y;
};

struct Funcs {
  REGFUNCTYPE reg;
};

extern "C" {
struct Funcs funcs;
}

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

extern "C"
PyObject *
PyInit_bindctypes(void) {

  try {
    auto m = Module("bindctypes");
    m.fun("add", add);
    m.fun("minus", minus);
    m.fun("add_d", add_d);
    m.fun("set_string", set_string);
    return m.self;
  } catch (...) {
    return nullptr;
  }
}
