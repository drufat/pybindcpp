// Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.
#ifndef MODULE_H
#define MODULE_H

#include "ctypes.h"
#include <sstream>

namespace pybindcpp {

struct module {
  PyObject *obj;
  module(PyObject *o) : obj(o) {}
  template <class T> auto add(const char *name, T t) {
    using ct = ctype_trait<T>;
    ct::add();
    return PyModule_AddObject(obj, name, ts->unbox(ct::box(t)));
  }
};

static auto import_typesytem() {
  //  auto m = PyImport_ImportModule("pybindcpp.module");
  //  auto init = PyObject_GetAttrString(m, "typesystem_init");
  //  auto ptr = PyLong_FromVoidPtr(&ts);
  //  auto ret = PyObject_CallFunctionObjArgs(init, ptr, nullptr);
  //  Py_DecRef(ptr);
  //  Py_DecRef(init);
  //  Py_DecRef(m);
  //  return ret;
  std::stringstream ss;
  ss << "import pybindcpp.module as m\n";
  ss << "m.typesystem_init(" << &ts << ")\n";
  return PyRun_SimpleString(ss.str().c_str());
}

static PyObject *init_module(const char *name,
                             std::function<void(module &)> init) {

  if (import_typesytem()) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot import typesystem.");
    return nullptr;
  }

  static PyModuleDef moduledef({
      PyModuleDef_HEAD_INIT,
      name,    // m_name
      nullptr, // m_doc
      -1,      // m_size
      nullptr, // m_methods
      nullptr, // m_slots
      nullptr, // m_traverse
      nullptr, // m_clear
      nullptr, // m_free
  });

  auto o = PyModule_Create(&moduledef);
  if (!o)
    return nullptr;

  auto m = module(o);

  ts->pre_init();
  init(m);
  ts->post_init();

  return o;
}

} // namespace pybindcpp

#define PYBINDCPP_INIT(NAME, INITFUNC)                                         \
  PyMODINIT_FUNC PyInit_##NAME() {                                             \
    return pybindcpp::init_module(#NAME, INITFUNC);                            \
  }

#endif // MODULE_H
