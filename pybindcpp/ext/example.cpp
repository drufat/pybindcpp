// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

#include <cmath>

#ifdef PYBINDCPP_CAPI
#include <capi/module.h>
#else
#include <ctyp/module.h>
#endif

using namespace pybindcpp;

constexpr double half = 0.5;
constexpr double pi = M_PI;

int f(int N, int n, int x) { return N + n + x; }

double mycos(double x) { return cos(x); }

void module(ExtModule &m) {
  m.var("half", half);
  m.var("pi", pi);
  m.var("one", static_cast<int>(1));
  m.var("two", static_cast<unsigned long>(2));
  m.var("true", true);
  m.var("false", false);
  m.var("name", "pybindcpp");

  m.fun("f", f);
  m.fun("mycos", mycos);
  m.fun("cos", [](double x) -> double { return cos(x); });
  m.fun("sin", static_cast<double (*)(double)>(sin));
}

#ifdef PYBINDCPP_CAPI
PYBINDCPP_INIT(example_capi, module)
#else
PYBINDCPP_INIT(example_ctyp, module)
#endif
