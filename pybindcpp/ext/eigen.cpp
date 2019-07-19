//Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.

#include <Eigen/Core>
#include <cmath>
#include <complex>
#include <numpy/arrayobject.h>
#include <pybindcpp/module.h>

using namespace pybindcpp;
using namespace Eigen;

template <class T>
void compute(const T &X, T &Y)
{
  Y = X.matrix() * X.matrix();
  Y(0, 0) = 0.0;
}

void init(module m)
{
  m.add("square", [](double *px, size_t x_dim0, size_t x_dim1,
                     double *py, size_t y_dim0, size_t y_dim1) {
    Map<ArrayXXd> X(px, x_dim0, x_dim1);
    Map<ArrayXXd> Y(py, y_dim0, y_dim1);
    compute(X, Y);
  });
}

PyMODINIT_FUNC PyInit_eigen()
{
  import_array();
  return pybindcpp::init_module("eigen", init);
}
