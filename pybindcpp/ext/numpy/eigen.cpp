// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

#include <Eigen/Core>
#include <capi/module.h>
#include <cmath>
#include <complex>
#include <numpy/arrayobject.h>

using namespace pybindcpp;
using namespace Eigen;

template <class T> void computation(const T &X, T &Y) {
  Y = X.matrix() * X.matrix();
  Y(0, 0) = 0.0;
}

void eigen(ExtModule &m) {

  m.fun("square", [](PyObject *o) -> PyObject * {
    const auto x =
        (PyArrayObject *)PyArray_ContiguousFromAny(o, NPY_DOUBLE, 2, 2);
    if (!x)
      return nullptr;

    auto y = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(x), PyArray_DIMS(x),
                                            NPY_DOUBLE, 0);
    if (!y)
      return nullptr;

    Map<ArrayXXd> X(static_cast<double *>(PyArray_DATA(x)), PyArray_DIMS(x)[0],
                    PyArray_DIMS(x)[1]);
    Map<ArrayXXd> Y(static_cast<double *>(PyArray_DATA(y)), PyArray_DIMS(y)[0],
                    PyArray_DIMS(y)[1]);

    computation(X, Y);

    Py_DECREF(x);
    return (PyObject *)y;
  });
}

PyMODINIT_FUNC PyInit_eigen() {
  import_array();
  return pybindcpp::module_init("eigen", eigen);
}
