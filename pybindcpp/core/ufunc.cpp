// Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.

#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include <pybindcpp/module.h>

using namespace pybindcpp;

const static size_t numpy_types_size = 14;
const static size_t numpy_types[numpy_types_size] = {
    NPY_BOOL,      NPY_BYTE,  NPY_UBYTE,  NPY_SHORT,      NPY_USHORT,
    NPY_INT,       NPY_UINT,  NPY_LONG,   NPY_ULONG,      NPY_LONGLONG,
    NPY_ULONGLONG, NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
};
static PyUFuncGenericFunction generic_function;

static auto pyufunc(PyUFuncGenericFunction *func, void **data, char *types,
                    int ntypes, int nin, int nout, char *name) {
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  auto rslt = PyUFunc_FromFuncAndData(func, data, types, ntypes, nin, nout,
                                      PyUFunc_None, name, nullptr, 0);

  PyGILState_Release(gstate);
  return rslt;
}

void import(module m) {
  m.add("numpy_types_size", numpy_types_size);
  m.add("numpy_types", numpy_types);
  m.add("generic_function", generic_function);
  m.add("pyufunc", pyufunc);
}

PyMODINIT_FUNC PyInit_ufunc() {
  import_array();
  import_ufunc();
  return init_module("ufunc", import);
}
