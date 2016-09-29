// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include <Python.h>

#include "pybindcpp/ctypes/ctypes.h"

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

static struct PyModuleDef moduledef =
    {
        PyModuleDef_HEAD_INIT,
        "bindctypes", // m_name
        nullptr,      // m_doc
        0,            // m_size
        nullptr,      // m_methods
        nullptr,      // m_slots
        nullptr,      // m_traverse
        nullptr,      // m_clear
        nullptr,      // m_free
    };

extern "C"
PyObject *
PyInit_bindctypes(void) {
  auto m = PyModule_Create(&moduledef);

  auto reg = cap<REGFUNCTYPE>("pybindcpp.register", "c_register_cap");
  if (!reg) return NULL;

  PyModule_AddObject(m, "add", fun(reg, add));
  PyModule_AddObject(m, "minus", fun(reg, minus));
  PyModule_AddObject(m, "add_d", fun(reg, add_d));
  PyModule_AddObject(m, "set_string", fun(reg, set_string));

  return m;
}
