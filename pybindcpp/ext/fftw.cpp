// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "pybindcpp/module.h"
#include "pybindcpp/numpy.h"

#include <fftw3.h>

using namespace pybindcpp;

void
fftw(ExtModule& m)
{

  m.fun("fft", [](PyObject* o) -> PyObject* {

    const auto x = (PyArrayObject*)PyArray_ContiguousFromAny( //
      o, NPY_CDOUBLE, 1, 1                                    //
      );
    if (!x)
      return NULL;

    auto y = (PyArrayObject*)PyArray_EMPTY(            //
      PyArray_NDIM(x), PyArray_DIMS(x), NPY_CDOUBLE, 0 //
      );
    if (!y)
      return NULL;

    auto N = PyArray_DIMS(x)[0];
    auto in = (fftw_complex*)PyArray_DATA(x);
    auto out = (fftw_complex*)PyArray_DATA(y);

    auto plan = fftw_plan_dft_1d(                              //
      N, in, out, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED //
      );

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    Py_DECREF(x);
    return (PyObject*)y;
  });

  m.fun("fft2", [](PyObject* o) -> PyObject* {
    const auto x = (PyArrayObject*)PyArray_ContiguousFromAny( //
      o, NPY_CDOUBLE, 1, 2                                    //
      );
    if (!x)
      return NULL;

    auto y = (PyArrayObject*)PyArray_EMPTY(            //
      PyArray_NDIM(x), PyArray_DIMS(x), NPY_CDOUBLE, 0 //
      );
    if (!y)
      return NULL;

    int N;
    int M;
    if (PyArray_NDIM(x) == 2) {
      N = PyArray_DIMS(x)[1];
      M = PyArray_DIMS(x)[0];
    } else {
      N = PyArray_DIMS(x)[0];
      M = 1;
    }
    auto p = fftw_plan_dft_1d(                   //
      N, NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE //| FFTW_UNALIGNED
      );                                         //

    fftw_complex* in;
    fftw_complex* out;
    for (int i = 0; i < M; i++) {
      in = (fftw_complex*)PyArray_GETPTR1(x, i);
      out = (fftw_complex*)PyArray_GETPTR1(y, i);
      fftw_execute_dft(p, in, out);
    }
    fftw_destroy_plan(p);

    Py_DECREF(x);
    return (PyObject*)y;
  });
}

PyMODINIT_FUNC
PyInit_fftw(void)
{
  import_array();
  return module_init("fftw", fftw);
}
