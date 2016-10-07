// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "pybindcpp/ctypes/module.h"

using namespace pybindcpp;

int
g(int x, int y) {
  return x + y;
}

void
simple(ExtModule &m) {
  m.fun("g", g);
}

PyMODINIT_FUNC
PyInit_simple(void) {
  return module_init("simple", simple);
}
