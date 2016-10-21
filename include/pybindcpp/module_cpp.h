// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef MODULE_CPP_H
#define MODULE_CPP_H

#include <Python.h>
#include <functional>

#include "module_cpp_imp.h"

namespace pybindcpp {

template <class T>
PyObject*
var2obj(T t)
{
  return build_value<T>(t);
}

template <class T>
PyObject*
fun2obj(T t)
{
  return fun_trait<T>::obj(t);
}

struct ExtModule
{
  PyObject* self;

  ExtModule(PyObject* m)
    : self(m)
  {
  }

  void add(const char* name, PyObject* obj)
  {
    PyModule_AddObject(self, name, obj);
  }

  template <class T>
  void var(const char* name, T t)
  {
    add(name, var2obj<T>(t));
  }

  template <class T>
  void fun(const char* name, T t)
  {
    add(name, fun2obj<T>(t));
  }

  template <class T>
  void varargs(const char* name, T t)
  {
    add(name, pybindcpp::varargs(t));
  }
};

static void
print(PyObject* obj)
{
  PyObject_Print(obj, stdout, Py_PRINT_RAW);
}

static PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  nullptr, // m_name
  nullptr, // m_doc
  -1,      // m_size
  nullptr, // m_methods
  nullptr, // m_slots
  nullptr, // m_traverse
  nullptr, // m_clear
  nullptr, // m_free
};

static PyObject*
module_init(const char* name, std::function<void(ExtModule&)> exec)
{
  moduledef.m_name = name;

  auto module = PyModule_Create(&moduledef);
  if (!module)
    return nullptr;

  try {

    ExtModule m(module);
    exec(m);

  } catch (const char*& ex) {

    PyErr_SetString(PyExc_RuntimeError, ex);
    return nullptr;

  } catch (const std::exception& ex) {

    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return nullptr;

  } catch (...) {

    PyErr_SetString(PyExc_RuntimeError, "Unknown internal error.");
    return nullptr;
  };

  return module;
}

} // end pybindcpp namespace

#define PYMODULE_INIT(name, exec)                                              \
  PyMODINIT_FUNC PyInit_##name() { return pybindcpp::module_init(#name, exec); }

#endif // MODULE_CPP_H
