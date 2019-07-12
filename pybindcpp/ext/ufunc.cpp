// Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.

#include <cmath>
#include <functional>
#include <iostream>
#include <pb/ufunc.h>

using namespace pybindcpp;

double fn(long N, double x) { return N * x; }

long add_one(long x) { return x + 1; }

void import(module m) {
  //  m.add("cos", fn);
  //  m.add("loop1d_cos", loop1d);

  //  {
  //    int ntypes = 1;
  //    int nin = 1;
  //    int nout = 1;
  //    auto func = new PyUFuncGenericFunction[1]();
  //    func[0] = loop1d_cos::func;
  //    auto data = new void *[1]();
  //    data[0] = nullptr;
  //    auto types = new char[2]();
  //    types[0] = NPY_DOUBLE;
  //    types[1] = NPY_DOUBLE;
  //    auto bcos = PyUFunc_FromFuncAndData(func, data, types, ntypes, nin,
  //    nout,
  //                                        PyUFunc_None, "bcos", nullptr, 0);
  //    m.add("bcos", bcos);
  //  }

  add_ufunc<double, double>(m, "cos", cos);
  add_ufunc<float, float>(m, "sin", sin);
  add_ufunc<double, long, double>(m, "fn", fn);
  add_ufunc<long, long>(m, "add_one", add_one);
}

PyMODINIT_FUNC PyInit_ufunc() { return init_module("ufunc", import); }
