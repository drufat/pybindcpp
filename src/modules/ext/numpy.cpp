// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "pybindcpp/module.h"
#include "pybindcpp/numpy.h"
#include "numpy_cpp.h"

using namespace pybindcpp;

double
fn(int N, double x) {
  return N * x;
}

template<class I0, class I1, class O>
void
fn_loop(char **args,
        npy_intp *dimensions,
        npy_intp *steps,
        void *data) {
  const auto len = dimensions[0];
  for (auto i = 0; i < len; i++) {
    auto &n = *(I0 *) (args[0] + i * steps[0]);
    auto &x = *(I1 *) (args[1] + i * steps[1]);
    auto &y = *(O *) (args[2] + i * steps[2]);
    y = fn(n, x);
  }
}

template<typename F, typename I0, typename I1, typename O>
decltype(auto)
loop1d_ii_o(F func) {
  return [func](char **args,
                npy_intp *dimensions,
                npy_intp *steps,
                void *data) {

    const auto len = dimensions[0];

    for (auto i = 0; i < len; i++) {
      auto &n = *(I0 * )(args[0] + i * steps[0]);
      auto &x = *(I1 * )(args[1] + i * steps[1]);
      auto &y = *(O * )(args[2] + i * steps[2]);
      y = func(n, x);
    }
  };
}

void
numpymodule(ExtModule &m) {

  ufunc_raw(m, "fn_ufunc1", {
      loop1d_ii_o<decltype(fn), int, double, double>(fn),
      loop1d_ii_o<decltype(fn), long, double, double>(fn),
  }, {
                NPY_INT, NPY_DOUBLE, NPY_DOUBLE,
                NPY_LONG, NPY_DOUBLE, NPY_DOUBLE,
            }, 2, 1);

  ufunc_raw(m, "fn_ufunc2", {
      fn_loop<int, double, double>,
      fn_loop<long, double, double>,
  }, {
                NPY_INT, NPY_DOUBLE, NPY_DOUBLE,
                NPY_LONG, NPY_DOUBLE, NPY_DOUBLE,
            }, 2, 1);

  ufunc_raw(m, "fn_ufunc3", {
      loop1d<double, decltype(fn), int, double>(fn),
      loop1d<double, decltype(fn), long, double>(fn),
  }, {
                NPY_INT, NPY_DOUBLE, NPY_DOUBLE,
                NPY_LONG, NPY_DOUBLE, NPY_DOUBLE,
            }, 2, 1);

  ufunc<double, decltype(fn), long, double>(m, "fn_ufunc", fn);

  m.fun("fn", [](int N, double x) {
    auto out = fn(N, x);
    return pybindcpp::build_value(out);
  });

  m.fun("fn_array", [](int N, PyObject *o) -> PyObject * {

    const auto x = (PyArrayObject *) PyArray_ContiguousFromAny(o,
                                                               NPY_DOUBLE,
                                                               1, 1);
    if (!x) return nullptr;

    auto y = (PyArrayObject *) PyArray_EMPTY(PyArray_NDIM(x),
                                             PyArray_DIMS(x),
                                             NPY_DOUBLE,
                                             0);
    if (!y) return nullptr;

    const auto len = PyArray_DIMS(x)[0];

    for (npy_intp i = 0; i < len; i++) {
      auto &xp = *(double *) PyArray_GETPTR1(x, i);
      auto &yp = *(double *) PyArray_GETPTR1(y, i);
      yp = fn(N, xp);
    }

    Py_DECREF(x);
    return (PyObject *) y;

  });

  m.fun("fn_array1", [](int N, PyObject *x) {

    auto xa = numpy::array_view<double, 1>(x);
    auto len = xa.dim(0);
    npy_intp shape[] = {len};
    auto out = numpy::array_view<double, 1>(shape);

    for (npy_intp i = 0; i < len; i++) {
      out[i] = fn(N, xa[i]);
    };

    return out.pyobj();

  });

  m.varargs("fn_array2", [](PyObject *self, PyObject *args) -> PyObject * {

    {
      int N;
      double x;
      if (pybindcpp::arg_parse_tuple(args, N, x)) {
        auto out = fn(N, x);
        return pybindcpp::build_value(out);
      }
    }
    PyErr_Clear();
    {
      int N;
      PyObject *x;
      if (pybindcpp::arg_parse_tuple(args, N, x)) {

        auto xa = numpy::array_view<double, 1>(x);
        auto len = xa.dim(0);
        npy_intp shape[] = {len};
        auto out = numpy::array_view<double, 1>(shape);

        for (npy_intp i = 0; i < len; i++) {
          out[i] = fn(N, xa[i]);
        };

        return out.pyobj();
      }
    }
    return NULL;
  });
}

PyMODINIT_FUNC
PyInit_numpy(void) {
  import_array();
  import_ufunc();
  return module_init("numpy", numpymodule);

}
