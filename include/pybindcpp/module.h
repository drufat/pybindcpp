// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef MODULE_H
#define MODULE_H

#include <Python.h>
#include <functional>
#include <memory>
#include <vector>

#include "api.h"
#include "callable_trait.h"
#include "capsule.h"
#include "pyfunction.h"

namespace pybindcpp {

template <class T>
PyObject*
var2obj(T t)
{
  return py_function<PyObject*(T)>("pybindcpp.bind", "id")(t);
}

template <class T>
PyObject*
fun2obj(T t)
{
  return callable_trait<T>::get(t);
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
  void fun_type(const char* name, T t)
  {
    add(name, func_trait<T>::pyctype());
  }

  template <class T>
  void varargs(const char* name, T t)
  {
    add(name, pybindcpp::varargs(t));
  }
};

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

  try {

    // keep a reference so that the API struct is not garbage collected
    import_pybindcpp();

    moduledef.m_name = name;

    auto m = PyModule_Create(&moduledef);
    if (!m)
      return nullptr;

    ExtModule module(m);

    exec(module);

    return m;

  } catch (const char* ex) {

    PyErr_SetString(PyExc_RuntimeError, ex);
    return nullptr;

  } catch (const std::exception& ex) {

    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return nullptr;

  } catch (...) {

    PyErr_SetString(PyExc_RuntimeError, "Unknown internal error.");
    return nullptr;
  };
}

} // end pybindcpp namespace

#define PYMODULE_INIT(name, exec)                                              \
  PyMODINIT_FUNC PyInit_##name() { return pybindcpp::module_init(#name, exec); }

#endif // MODULE_H
