// Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.

#include <cmath>
#include <pybindcpp/module.h>

constexpr double half = 0.5;
constexpr double pi = M_PI;

int f(int N, int n, int x) { return N + n + x; }
double mycos(double x) { return cos(x); }

using namespace pybindcpp;

void init(module m) {
  // numbers
  m.add("half", half);
  m.add("pi", pi);
  m.add("one", static_cast<int>(1));
  m.add("two", static_cast<unsigned long>(2));

  // boolean
  m.add("true", true);
  m.add("false", false);

  // string
  m.add("name", "pybindcpp");

  // functions
  m.add("f", f);
  m.add("mycos", mycos);
  m.add("cos", [](double x) -> double { return cos(x); });
  m.add("sin", static_cast<double (*)(double)>(sin));
}

PYBINDCPP_INIT(example, init)
