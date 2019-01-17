// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifdef PYBINDCPP_CAPI
#include <capi/module.h>
#else
#include <ctyp/module.h>
#endif

using namespace pybindcpp;

double f(double x, double y) { return x * y; }

int g(int x, int y) { return x + y; }

void module(ExtModule &m) {
  m.fun("f", f);
  m.fun("g", g);
}

#ifdef PYBINDCPP_CAPI
PYBINDCPP_INIT(simple_capi, module)
#else
PYBINDCPP_INIT(simple_ctyp, module)
#endif
