// Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.

#include <cmath>
#include <functional>
#include <iostream>
#include <pybindcpp/ufunc.h>

using namespace pybindcpp;

double my_cos(double x) { return cos(x); }
float my_sin(float x) { return sin(x); }
double fn(long N, double x) { return N * x; }
long add_one(long x) { return x + 1; }

void init(module m)
{
  add_ufunc<double, double>(m, "cos", my_cos);
  add_ufunc<float, float>(m, "sin", my_sin);
  add_ufunc<double, long, double>(m, "fn", fn);
  add_ufunc<long, long>(m, "add_one", add_one);
}

PYBINDCPP_INIT(ufunc, init)
