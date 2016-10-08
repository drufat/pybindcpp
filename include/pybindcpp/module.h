// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef MODULE_H
#define MODULE_H

#include <Python.h>

#include <functional>
#include <vector>
#include <memory>

#include "capsule.h"
#include "api.h"
#include "callable_trait.h"
#include "pyfunction.h"

namespace pybindcpp {

struct ExtModule {
  PyObject *self;
  API *api;

  ExtModule(PyObject *m) : self(m) {
    auto __api__ = import_pybindcpp(&api);
    // keep a reference so that the API struct is not garbage collected
    add("__pybindcpp_api__", __api__);
  }

  void add(const char *name, PyObject *obj) {
    PyModule_AddObject(self, name, obj);
  }

  template<class T>
  void var(const char *name, T t) {
    add(name,
        py_function<PyObject *(T)>(*api, "pybindcpp.bind", "id")(t));
  }

  template<class F>
  void fun(const char *name, F f) {
    add(name,
        callable_trait<F>::get(*api, f));
  }

  template<class F>
  void fun_type(const char *name, F f) {
    add(name,
        func_trait<F>::pyctype(*api));
  }

  template<class F>
  void varargs(const char *name, F f) {
    add(name,
        pybindcpp::varargs(*api, f));
  }

};

static
std::function<void(ExtModule &)> __exec;

static
int
__module_exec(PyObject *module) {
  try {

    ExtModule m(module);
    __exec(m);
    return 0;

  } catch (const char *&ex) {

    PyErr_SetString(PyExc_RuntimeError, ex);
    return -1;

  } catch (const std::exception &ex) {

    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;

  } catch (...) {

    PyErr_SetString(PyExc_RuntimeError, "Unknown internal error.");
    return -1;

  };
}

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    nullptr,  // m_name
    nullptr,  // m_doc
    0,        // m_size
    nullptr,  // m_methods
    nullptr,  // m_slots
    nullptr,  // m_traverse
    nullptr,  // m_clear
    nullptr,  // m_free
};

#ifdef Py_mod_exec
static PyModuleDef_Slot __module_slots[] = {
    {Py_mod_exec, (void *) __module_exec},
    {0, NULL}
};
#endif

static
PyObject *
module_init(const char *name, std::function<void(ExtModule &)> exec) {
  moduledef.m_name = name;
  __exec = exec;
#ifdef Py_mod_exec
  moduledef.m_slots = __module_slots;
  return PyModuleDef_Init(&moduledef);
#else
  auto module = PyModule_Create(&moduledef);
  if (!module) return nullptr;

  if (__module_exec(module) != 0) {
    Py_DECREF(module);
    return nullptr;
  }
  return module;
#endif

}

}

#define PYMODULE_INIT(name, exec)             \
PyMODINIT_FUNC                                \
PyInit_##name() {                             \
  return pybindcpp::module_init(#name, exec); \
}                                             \


#endif // MODULE_H
