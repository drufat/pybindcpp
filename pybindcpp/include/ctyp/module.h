// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef MODULE_H
#define MODULE_H

#include <Python.h>
#include <functional>
#include <memory>
#include <vector>

#include "api.h"
#include "callable_trait.h"
#include "pyfunction.h"

namespace pybindcpp {

template <class T> PyObject *var2obj(T t) {
  using IdFuncType = py_function<PyObject *(T)>;
  return IdFuncType("pybindcpp.bind", "id")(t);
}

template <class T> PyObject *fun2obj(T t) { return callable_trait<T>::get(t); }

struct ExtModule {
  PyObject *self;

  ExtModule(PyObject *m) : self(m) {
    auto mod_pybind = PyImport_ImportModule("pybindcpp.api");
    if (!mod_pybind)
      throw "Cannot import pybindcpp.api.";

    auto api_addr = PyObject_GetAttrString(mod_pybind, "api_addr");
    if (!api_addr)
      throw "Cannot access pybindcpp.api.api_addr.";

    void *ptr = PyLong_AsVoidPtr(api_addr);
    api = static_cast<API *>(ptr);

    Py_DecRef(api_addr);
    Py_DecRef(mod_pybind);
  }

  void add(const char *name, PyObject *obj) {
    PyModule_AddObject(self, name, obj);
  }

  template <class T> void var(const char *name, T t) {
    add(name, var2obj<T>(t));
  }

  template <class T> void fun_type(const char *name, T t) {
    auto s = func_trait<T>::str();
    var(name, s.c_str());
  }

  template <class T> void fun(const char *name, T t) {
    add(name, fun2obj<T>(t));
  }

  template <class T> void varargs(const char *name, T t) {
    add(name, api->vararg(fun2obj<T>(t)));
  }
}; // namespace pybindcpp

using ExecType = std::function<void(ExtModule &)>;

static PyObject *module_init(PyModuleDef &moduledef, ExecType exec) {
  try {
    auto m = PyModule_Create(&moduledef);
    if (!m)
      return nullptr;
    ExtModule module(m);

    exec(module);

    return m;

  } catch (const char *ex) {

    PyErr_SetString(PyExc_RuntimeError, ex);
    return nullptr;

  } catch (const std::exception &ex) {

    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return nullptr;

  } catch (...) {

    PyErr_SetString(PyExc_RuntimeError, "Unknown internal error.");
    return nullptr;
  };
}

} // namespace pybindcpp

#define PYBINDCPP_INIT(name, exec)                                             \
  PyMODINIT_FUNC PyInit_##name() {                                             \
    static PyModuleDef moduledef = {                                           \
        PyModuleDef_HEAD_INIT,                                                 \
        #name,                                                                 \
        nullptr,                                                               \
        -1,                                                                    \
        nullptr,                                                               \
        nullptr,                                                               \
        nullptr,                                                               \
        nullptr,                                                               \
        nullptr,                                                               \
    };                                                                         \
    return pybindcpp::module_init(moduledef, exec);                            \
  }

#define PYBINDCPP_INIT_NUMPY(name, exec)                                       \
  PyMODINIT_FUNC PyInit_##name() {                                             \
    static PyModuleDef moduledef = {                                           \
        PyModuleDef_HEAD_INIT,                                                 \
        #name,                                                                 \
        nullptr,                                                               \
        -1,                                                                    \
        nullptr,                                                               \
        nullptr,                                                               \
        nullptr,                                                               \
        nullptr,                                                               \
        nullptr,                                                               \
    };                                                                         \
    import_array();                                                            \
    import_ufunc();                                                            \
    return pybindcpp::module_init(moduledef, exec);                            \
  }

#endif // MODULE_H
