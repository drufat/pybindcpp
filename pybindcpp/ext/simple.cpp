// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "pybindcpp/module.h"

using namespace pybindcpp;

int
g(int x, int y)
{
  return x + y;
}

int
f(double x, double y)
{
  return x * y;
}

void
simple(ExtModule& m)
{
  m.fun("g", g);
  m.fun("f", f);
}

PyMODINIT_FUNC
PyInit_simple(void)
{
  return module_init("simple", simple);
}
