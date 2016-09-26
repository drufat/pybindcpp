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

void bindctypes_init(PyObject *m, REGFUNCTYPE reg) {
  {
    constexpr int func_signature[] = {c_int, c_int, c_int};
    constexpr size_t func_signature_size = sizeof(func_signature) / sizeof(int);
    PyModule_AddObject(
        m, "add",
        reg(reinterpret_cast<void *>(add), func_signature, func_signature_size)
    );
    PyModule_AddObject(
        m, "minus",
        reg(reinterpret_cast<void *>(minus), func_signature, func_signature_size)
    );
  }

  {
    constexpr int func_signature[] = {c_double, c_double, c_double};
    constexpr size_t func_signature_size = sizeof(func_signature) / sizeof(int);
    PyModule_AddObject(
        m, "add_d",
        reg(reinterpret_cast<void *>(add_d), func_signature, func_signature_size)
    );
  }
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

  auto pnt = PyCapsule_Import("pybindcpp.register.c_register", 0);
  if (!pnt) return NULL;
  auto reg = reinterpret_cast<REGFUNCTYPE>(pnt);

  bindctypes_init(m, reg);

  return m;
}
