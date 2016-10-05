// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include <Python.h>

#include "pybindcpp/ctypes/module.h"

using namespace pybindcpp;

extern "C"
int set_string(char c, int size, char *buffer) {
  for (int i = 0; i < size; i++) {
    buffer[i] = c;
  }
  return 0;
}

auto id(PyObject *o) {
  return o;
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

void
exec(ExtModule &m) {

//  api.error();
//  throw "Error!";
//  if (PyErr_Occurred()) {
//    PyErr_Clear();
//    throw "Error!";
//  }

  py_function<int(int)>(*m.api, "pybindcpp.sample", "id")(3);

  m.add("id_type", func_trait<decltype(&id)>::pyctype(*m.api));
  m.add("add_type", func_trait<decltype(&add)>::pyctype(*m.api));
  m.add("minus_type", func_trait<decltype(&minus)>::pyctype(*m.api));
  m.add("set_string_type", func_trait<decltype(&set_string)>::pyctype(*m.api));

  m.fun("id", id);
  m.fun("add", add);
  m.fun("minus", minus);
  m.fun("add_d", add_d);
  m.fun("set_string", set_string);
}

PYMODULE_INIT(bindctypes, exec)