// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

#include <Eigen/Core>
#include <cmath>
#include <complex>
#include <numpy/arrayobject.h>
#include <pybindcpp/module.h>

using namespace pybindcpp;
using namespace Eigen;

template <class T> void computation(const T &X, T &Y) {
  Y = X.matrix() * X.matrix();
  Y(0, 0) = 0.0;
}

void import(module m) {

  m.add("square", [](PyObject *o) -> PyObject * {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    auto x = (PyArrayObject *)PyArray_ContiguousFromAny(o, NPY_DOUBLE, 2, 2);
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
    auto rslt = (PyObject *)y;

    PyGILState_Release(gstate);
    return rslt;
  });
}

PyMODINIT_FUNC PyInit_eigen() {
  import_array();
  return pybindcpp::init_module("eigen", import);
}
