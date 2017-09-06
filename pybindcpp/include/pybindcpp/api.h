#ifndef PYBINDCPP_API_H
#define PYBINDCPP_API_H

#include <Python.h>

namespace pybindcpp {

struct API
{
  PyObject* (*get_type)(const char*);

  void* (*get_capsule)(const char*, const char*);

  void* (*get_cfunction)(const char*, const char*);

  PyObject* (*get_pyfunction)(const char*, const char*, PyObject*);

  void* (*get_addr)(PyObject*);

  PyObject* (*register_)(void*, PyObject*);

  PyObject* (*apply)(PyObject*, PyObject*);

  PyObject* (*vararg)(PyObject*);

  void (*error)();
};

static API* api;

static PyObject*
import_pybindcpp()
{

  auto mod = PyImport_ImportModule("pybindcpp.api");
  if (!mod)
    throw "Cannot import";

  auto init_addr = PyObject_GetAttrString(mod, "init_addr");
  if (!init_addr)
    throw "Cannot access attribute.";

  void* ptr = PyLong_AsVoidPtr(init_addr);
  auto init = *static_cast<PyObject* (**)(API**)>(ptr);

  auto pyapi = init(&api);

  Py_DecRef(init_addr);
  Py_DecRef(mod);

  return pyapi;
}
}
#endif // PYBINDCPP_API_H
