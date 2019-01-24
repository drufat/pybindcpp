// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

#include <capi/numpy.h>

using namespace pybindcpp;

double fn(long N, double x) { return N * x; }

long add_one(long x) { return x + 1; }

template <class I0, class I1, class O>
void fn_loop(char **args, npy_intp *dimensions, npy_intp *steps, void *data) {
  const auto N = dimensions[0];
  for (auto i = 0; i < N; i++) {
    auto &n = *reinterpret_cast<I0 *>(args[0] + i * steps[0]);
    auto &x = *reinterpret_cast<I1 *>(args[1] + i * steps[1]);
    auto &y = *reinterpret_cast<O *>(args[2] + i * steps[2]);
    y = fn(n, x);
  }
}

template <class F, class I0, class I1, class O> auto loop1d_ii_o(F func) {
  return
      [func](char **args, npy_intp *dimensions, npy_intp *steps, void *data) {
        const auto N = dimensions[0];
        for (auto i = 0; i < N; i++) {
          auto &n = *reinterpret_cast<I0 *>(args[0] + i * steps[0]);
          auto &x = *reinterpret_cast<I1 *>(args[1] + i * steps[1]);
          auto &y = *reinterpret_cast<O *>(args[2] + i * steps[2]);
          y = func(n, x);
        }
      };
}

static void ufunc_raw(ExtModule &m, const char *name,
                      std::vector<generic> funcs, std::vector<char> types,
                      size_t nin, size_t nout) {
  m.var(name, make_ufunc_imp(name, funcs, types, nin, nout));
}

void numpymodule(ExtModule &m) {

  ufunc_raw(m, "fn_ufunc1",
            {
                loop1d_ii_o<decltype(fn), int, double, double>(fn),
                loop1d_ii_o<decltype(fn), long, double, double>(fn),
            },
            {NPY_INT, NPY_DOUBLE, NPY_DOUBLE, NPY_LONG, NPY_DOUBLE, NPY_DOUBLE},
            2, 1);

  ufunc_raw(m, "fn_ufunc2",
            {
                fn_loop<int, double, double>,
                fn_loop<long, double, double>,
            },
            {NPY_INT, NPY_DOUBLE, NPY_DOUBLE, NPY_LONG, NPY_DOUBLE, NPY_DOUBLE},
            2, 1);

  ufunc_raw(
      m, "fn_ufunc3",
      {
          loop1d<std::function<double(int, int)>, double, int, double>(fn),
          loop1d<std::function<double(long, int)>, double, long, double>(fn),
      },
      {NPY_INT, NPY_DOUBLE, NPY_DOUBLE, NPY_LONG, NPY_DOUBLE, NPY_DOUBLE}, 2,
      1);

  ufunc(m, "fn_ufunc", fn);
  ufunc(m, "add_one", add_one);

  m.fun("fn", [](int N, double x) {
    auto out = fn(N, x);
    return pybindcpp::build_value(out);
  });

  m.fun("fn_array", [](int N, PyObject *o) -> PyObject * {
    auto x = (PyArrayObject *)PyArray_ContiguousFromAny(o, NPY_DOUBLE, 1, 1);
    if (!x)
      return nullptr;

    auto y = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(x), PyArray_DIMS(x),
                                            NPY_DOUBLE, 0);
    if (!y)
      return nullptr;

    const auto len = PyArray_DIMS(x)[0];

    for (npy_intp i = 0; i < len; i++) {
      auto &xp = *(double *)(PyArray_GETPTR1(x, i));
      auto &yp = *(double *)(PyArray_GETPTR1(y, i));
      yp = fn(N, xp);
    }

    Py_DECREF(x);
    return (PyObject *)y;
  });

  m.fun("fn_array1", [](int N, PyObject *o) -> PyObject * {
    if (PyFloat_Check(o)) {

      double x = PyFloat_AsDouble(o);
      auto out = fn(N, x);
      return PyFloat_FromDouble(out);

    } else {

      auto x = (PyArrayObject *)PyArray_ContiguousFromAny(o, NPY_DOUBLE, 1, 1);
      if (!x)
        return nullptr;

      auto y = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(x), PyArray_DIMS(x),
                                              NPY_DOUBLE, 0);
      if (!y)
        return nullptr;

      const auto len = PyArray_DIMS(x)[0];

      for (npy_intp i = 0; i < len; i++) {
        auto &xp = *(double *)(PyArray_GETPTR1(x, i));
        auto &yp = *(double *)(PyArray_GETPTR1(y, i));
        yp = fn(N, xp);
      }

      Py_DECREF(x);
      return (PyObject *)y;
    }
  });
}

PyMODINIT_FUNC PyInit_numpy() {
  import_array();
  import_ufunc();
  return pybindcpp::module_init("numpy", numpymodule);
}
