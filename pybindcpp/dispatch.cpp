// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include <Python.h>

#include <pybindcpp/module_cpp_imp.h>

namespace {

PyObject*
dispatch(PyObject* self, PyObject* args)
{
  PyObject* ofunc;
  PyObject* rargs;

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

static PyMethodDef methods[] = { { "dispatch", dispatch, METH_VARARGS, NULL },
                                 { NULL, NULL, 0, NULL } };

static PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "dispatch", // m_name
  nullptr,    // m_doc
  -1,         // m_size
  methods,    // m_methods
  nullptr,    // m_slots
  nullptr,    // m_traverse
  nullptr,    // m_clear
  nullptr,    // m_free
};
}

PyMODINIT_FUNC
PyInit_dispatch(void)
{
  return PyModule_Create(&moduledef);
}
