// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

#include <fftw3.h>
#include <numpy/arrayobject.h>
#include <pybindcpp/module.h>

using namespace pybindcpp;

void import(module m) {

  m.add("fft", [](PyObject *o) -> PyObject * {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    const auto x =
        (PyArrayObject *)PyArray_ContiguousFromAny(o, NPY_CDOUBLE, 1, 1);
    if (!x)
      return nullptr;

    auto y = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(x), PyArray_DIMS(x),
                                            NPY_CDOUBLE, 0);
    if (!y)
      return nullptr;

    auto N = PyArray_DIMS(x)[0];
    auto in = static_cast<fftw_complex *>(PyArray_DATA(x));
    auto out = static_cast<fftw_complex *>(PyArray_DATA(y));

    auto plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    Py_DECREF(x);
    auto rslt = (PyObject *)y;

    PyGILState_Release(gstate);
    return rslt;
  });

  m.add("fft2", [](PyObject *o) -> PyObject * {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    const auto x =
        (PyArrayObject *)PyArray_ContiguousFromAny(o, NPY_CDOUBLE, 1, 2);
    if (!x)
      return nullptr;

    auto y = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(x), PyArray_DIMS(x),
                                            NPY_CDOUBLE, 0);
    if (!y)
      return nullptr;

    npy_intp N, M;
    if (PyArray_NDIM(x) == 2) {
      N = PyArray_DIMS(x)[1];
      M = PyArray_DIMS(x)[0];
    } else {
      N = PyArray_DIMS(x)[0];
      M = 1;
    }

    auto p = fftw_plan_dft_1d(N, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_complex *in;
    fftw_complex *out;
    for (int i = 0; i < M; i++) {
      in = (fftw_complex *)PyArray_GETPTR1(x, i);
      out = (fftw_complex *)PyArray_GETPTR1(y, i);
      fftw_execute_dft(p, in, out);
    }
    fftw_destroy_plan(p);

    Py_DECREF(x);
    auto rslt = (PyObject *)y;

    PyGILState_Release(gstate);
    return rslt;
  });
}

PyMODINIT_FUNC PyInit_fftw() {
  import_array();
  return pybindcpp::init_module("fftw", import);
}
