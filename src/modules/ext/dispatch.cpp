// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include<Python.h>

#include "pybindcpp/types.h"
#include "pybindcpp/capsule.h"

namespace {

PyObject *
dispatch(PyObject *self, PyObject *args) {
  PyObject *ofunc;
  PyObject *rargs;

  auto s = PyArg_ParseTuple(args, "OO", &ofunc, &rargs);
  if (!s) {
    PyErr_SetString(PyExc_TypeError, "Unable to parse args.");
    return NULL;
  }

  auto func = pybindcpp::capsule_get<pybindcpp::VarArg>(ofunc);
  if (!func) {
    PyErr_SetString(PyExc_TypeError, "Unable to unwrap std::function.");
    return NULL;
  }

  return (*func)(self, rargs);
}

PyMethodDef methods[] = {
    {"dispatch", dispatch, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "dispatch",
    NULL,
    -1,
    methods
};
}

PyMODINIT_FUNC
PyInit_dispatch(void) {
  return PyModule_Create(&moduledef);
}
