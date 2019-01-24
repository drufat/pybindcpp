// Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.
#ifndef MODULE_H
#define MODULE_H

#include <Python.h>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "ctypes.h"

namespace pybindcpp {

struct ExtModule {
  API *api;

  using InitType = std::function<void(ExtModule &)>;

  ExtModule(PyObject *m, InitType init) {
    {
      std::stringstream ss;
      ss << "import pybindcpp.module as m\n";
      ss << "m.module_init(" << &m << ", " << &api << ")\n";
      auto cmd = ss.str();
      auto err = PyRun_SimpleString(cmd.c_str());
      if (err)
        throw "Cannot import pybindcpp.api.";
    }
    {
      api->pre_init();
      init(*this);
      api->commit_add();
      api->post_init();
    }
  }

  template <class T> void add(const char *name, T t) {
    using ct = ctype_trait<T>;
    ct::add(api);
    api->add(name, ct::box(t));
  }
};

using InitType = std::function<void(ExtModule &)>;

static PyObject *module_init(PyObject *m, ExtModule::InitType init) {
  try {

    ExtModule mod(m, init);
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

#define PYBINDCPP_INIT(NAME, INITFUNC)                                         \
                                                                               \
  static PyModuleDef moduledef = {                                             \
      PyModuleDef_HEAD_INIT,                                                   \
      #NAME,                                                                   \
      nullptr,                                                                 \
      -1,                                                                      \
      nullptr,                                                                 \
      nullptr,                                                                 \
      nullptr,                                                                 \
      nullptr,                                                                 \
      nullptr,                                                                 \
  };                                                                           \
                                                                               \
  PyMODINIT_FUNC PyInit_##NAME() {                                             \
    auto m = PyModule_Create(&moduledef);                                      \
    if (!m)                                                                    \
      return nullptr;                                                          \
    module_init(m, INITFUNC);                                                  \
    return m;                                                                  \
  }

#endif // MODULE_H
