// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include <Python.h>

#include "pybindcpp/ctypes.h"

extern "C"
int create_string(char c, int size, char *buffer) {
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
};;

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

  REGFUNCTYPE reg;
  {
    auto mod = PyImport_ImportModule("pybindcpp.register");
    if (!mod) return NULL;
    auto cap = PyObject_GetAttrString(mod, "c_register");
    if (!cap) return NULL;
    auto pnt = PyCapsule_GetPointer(cap, "pybindcpp.register.c_register");
    if (!pnt) return NULL;
    reg = reinterpret_cast<REGFUNCTYPE>(pnt);
    Py_DecRef(cap);
    Py_DecRef(mod);
  }

  fun(m, reg, "add", add);
  fun(m, reg, "minus", minus);
  fun(m, reg, "add_d", add_d);
  fun(m, reg, "create_string", create_string);

  return m;
}
