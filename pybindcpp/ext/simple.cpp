// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "pybindcpp/module.h"

using namespace pybindcpp;

int g(int x, int y) { return x + y; }

double f(double x, double y) { return x * y; }

void simple(ExtModule &m) {
  m.fun("g", g);
  m.fun("f", f);
}

PYBINDCPP_INIT(simple, simple);
