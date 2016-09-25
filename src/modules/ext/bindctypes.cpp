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

extern "C"
void bind_init(REGFUNCTYPE reg) {
  {
    constexpr int func_signature[] = {c_int, c_int, c_int};
    constexpr size_t func_signature_size = sizeof(func_signature) / sizeof(int);
    reg("add", reinterpret_cast<void *>(add), func_signature, func_signature_size);
    reg("minus", reinterpret_cast<void *>(minus), func_signature, func_signature_size);
  }

  {
    constexpr int func_signature[] = {c_double, c_double, c_double};
    constexpr size_t func_signature_size = sizeof(func_signature) / sizeof(int);
    reg("add_d", reinterpret_cast<void *>(add_d), func_signature, func_signature_size);
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

PyMODINIT_FUNC
PyInit_bindctypes(void) {
  auto m = PyModule_Create(&moduledef);

  auto cap = PyCapsule_New(reinterpret_cast<void *>(bind_init), typeid(void *).name(), nullptr);
  Py_DECREF(cap);

  {
    auto mod = PyImport_ImportModule("pybindcpp.register");
    auto fun = PyObject_GetAttrString(mod, "add");
    auto args = Py_BuildValue("(ii)", 1, 2);
    auto obj = PyObject_Call(fun, args, NULL);
    Py_DECREF(args);
    Py_DECREF(fun);
    Py_DECREF(mod);
    PyModule_AddObject(m, "nothing", obj);
  }

  return m;
}
