// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "numpy_pybind.h"

#include <Eigen/Core>
#include <cmath>
#include <complex>

using namespace pybindcpp;
using namespace Eigen;

template <class T> void computation(const T &X, T &Y) {
  Y = X.matrix() * X.matrix();
  Y(0, 0) = 0.0;
}

void eigen(ExtModule &m) {

  m.fun("square", [](PyObject *o) -> PyObject * {
    const auto x = PyArray_ContiguousFromAny(o, NPY_DOUBLE, 2, 2);
    if (!x)
      return NULL;

    auto y = PyArray_EMPTY(PyArray_NDIM(x), PyArray_DIMS(x), NPY_DOUBLE, 0);
    if (!y)
      return NULL;

    Map<ArrayXXd> X((double *)PyArray_DATA(x), PyArray_DIMS(x)[0],
                    PyArray_DIMS(x)[1]);
    Map<ArrayXXd> Y((double *)PyArray_DATA(y), PyArray_DIMS(y)[0],
                    PyArray_DIMS(y)[1]);

    computation(X, Y);

    Py_DECREF(x);
    return y;
  });
}

PYBINDCPP_INIT_NUMPY(eigen, eigen);
