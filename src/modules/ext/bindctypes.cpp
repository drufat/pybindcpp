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
  m.add("add_trait", func_trait<decltype(&add)>::pyctype());
  m.fun("add", add);
  m.fun("minus", minus);
  m.fun("add_d", add_d);
  m.fun("set_string", set_string);
}

PYMODULE_INIT(bindctypes, exec)