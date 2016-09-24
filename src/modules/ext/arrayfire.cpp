// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "python/module.h"
#include "python/numpy.h"

#include <vector>
#include <arrayfire.h>

using namespace pybindcpp;
using namespace af;

void
testBackend() {
  info();
  af_print(af::randu(5, 4));
}

void
test() {
  try {
    printf("=====================\n");
    printf("Trying CPU Backend\n");
    printf("=====================\n");
    setBackend(AF_BACKEND_CPU);
    testBackend();
  }
  catch (af::exception &e) {
    printf("Caught exception when trying CPU backend\n");
    fprintf(stderr, "%s\n", e.what());
  }
  try {
    printf("=====================\n");
    printf("Trying CUDA Backend\n");
    printf("=====================\n");
    setBackend(AF_BACKEND_CUDA);
    testBackend();
  }
  catch (af::exception &e) {
    printf("Caught exception when trying CUDA backend\n");
    fprintf(stderr, "%s\n", e.what());
  }
  try {
    printf("=====================\n");
    printf("Trying OpenCL Backend\n");
    printf("=====================\n");
    setBackend(AF_BACKEND_OPENCL);
    testBackend();
  }
  catch (af::exception &e) {
    printf("Caught exception when trying OpenCL backend\n");
    fprintf(stderr, "%s\n", e.what());
  }
}

void
set_backend(const char *backd) {
  auto b = std::string(backd);
  if (b == "cpu") {
    setBackend(AF_BACKEND_CPU);
  }
  if (b == "cuda") {
    setBackend(AF_BACKEND_CUDA);
  }
  if (b == "opencl") {
    setBackend(AF_BACKEND_OPENCL);
  }
}

array
addone(const array &X) {
  return X + 1;
}

auto
arrayfire(Module &m) {
  m.fun("test", test);
  m.fun("set_backend", set_backend);

  auto _ = [&](std::string name, array(*fn)(const array &)) {

    return m.varargs(name, [fn](PyObject *self, PyObject *args) -> PyObject * {

      PyObject *o;
      if (!arg_parse_tuple(args, o))
        return NULL;

      const auto x = (PyArrayObject *) PyArray_ContiguousFromAny(
          o,
          NPY_DOUBLE,
          1, 1
      );
      if (!x) return NULL;

      auto y = (PyArrayObject *) PyArray_EMPTY(
          PyArray_NDIM(x),
          PyArray_DIMS(x),
          NPY_DOUBLE,
          0
      );
      if (!y) return NULL;

      auto X = array(PyArray_DIMS(x)[0], (double *) PyArray_DATA(x));
      Py_DECREF(x);

      auto Y = fn(X);

      Y.host(PyArray_GETPTR1(y, 0));
      return (PyObject *) y;

    });
  };

  _("addone", addone);
  _("erf", erf);

  m.varargs("fft", [](PyObject *self, PyObject *args) -> PyObject * {

    PyObject *o;
    if (!arg_parse_tuple(args, o))
      return NULL;

    auto x = (PyArrayObject *) PyArray_ContiguousFromAny(
        o,
        NPY_CDOUBLE,
        1, 1);
    if (!x) return NULL;

    auto y = (PyArrayObject *) PyArray_EMPTY
        (PyArray_NDIM(x),
         PyArray_DIMS(x),
         NPY_CDOUBLE,
         0);
    if (!y) return NULL;

    auto X = array(PyArray_DIMS(x)[0], (cdouble *) PyArray_DATA(x));
    Py_DECREF(x);

    auto Y = fft(X);

    Y.host(PyArray_GETPTR1(y, 0));
    return (PyObject *) y;

  });

}

PyMODINIT_FUNC
PyInit_arrayfire(void) {
  import_array();
  return module_init("arrayfire", arrayfire);
}
