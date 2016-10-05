// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef MODULE_H
#define MODULE_H

#include <Python.h>

#include <functional>
#include <vector>
#include <memory>

#include "pybindcpp/capsule.h"
#include "pybindcpp/ctypes/api.h"
#include "pybindcpp/ctypes/callable.h"
#include "pybindcpp/ctypes/pyfunction.h"

namespace pybindcpp {

struct ExtModule {
  PyObject *__dict__;
  std::shared_ptr<API> api;

  ExtModule()  {

    api = std::make_shared<API>();
    import_pybindcpp(*api);

    __dict__ = PyDict_New();
    if (!__dict__) throw "Cannot create dictionary.";

    add("__pybindcpp_api__", capsule_new(api));

  }

  ~ExtModule() {
    Py_DecRef(__dict__);
  }

  void add(const char *name, PyObject *obj) {
    PyDict_SetItemString(__dict__, name, obj);
    Py_DecRef(obj);
  }

  template<class F>
  void fun(const char *name, F f) {
    add(name, callable_trait<F>::get(*api, f));
  }

};

static
std::function<void(ExtModule &)> __exec;

static
int
__module_exec(PyObject *module) {
  try {
    ExtModule m;
    __exec(m);
    auto d = PyModule_GetDict(module);
    PyDict_Update(d, m.__dict__);
    return 0;

  } catch (const char *&ex) {

    PyErr_SetString(PyExc_RuntimeError, ex);
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
__module_init(const char *name) {

  moduledef.m_name = name;
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

#define PYMODULE_INIT(name, exec)   \
extern "C" PyObject *               \
PyInit_##name() {                   \
  __exec = exec;                    \
  return __module_init(#name);      \
}                                   \


#endif // MODULE_H
