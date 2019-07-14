// Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.

#include <cmath>
#include <functional>
#include <iostream>
#include <pybindcpp/ufunc.h>

using namespace pybindcpp;

double fn(long N, double x) { return N * x; }

long add_one(long x) { return x + 1; }

void import(module m) {
  add_ufunc<double, double>(m, "cos", cos);
  add_ufunc<float, float>(m, "sin", sin);
  add_ufunc<double, long, double>(m, "fn", fn);
  add_ufunc<long, long>(m, "add_one", add_one);
}

PYBINDCPP_INIT(ufunc, import)
