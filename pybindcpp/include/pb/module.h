// Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.
#ifndef MODULE_H
#define MODULE_H

#include "ctypes.h"
#include <iostream>
#include <sstream>

namespace pybindcpp {

/*
 *  Import typesystem needs to be called before anything
 *  else can be done.
 */
static auto import_typesytem() {
  std::stringstream ss;
  ss << "import pybindcpp.module as m\n";
  ss << "m.typesystem_init(" << &ts << ")\n";
  return PyRun_SimpleString(ss.str().c_str());
}

static auto import_typesytem_capi() {
  auto m = PyImport_ImportModule("pybindcpp.module");
  auto init = PyObject_GetAttrString(m, "typesystem_init");
  auto ptr = PyLong_FromVoidPtr(&ts);
  auto ret = PyObject_CallFunctionObjArgs(init, ptr, NULL);
  Py_DecRef(ptr);
  Py_DecRef(init);
  Py_DecRef(m);
  Py_DecRef(ret);
  return 0;
}

/*
 *  Never call this function during static init of C
 *  extension as then it outlives the python module
 *  that it depends on for memory management. It is
 *  ok to call it from one level of indirection.
 */
template <class Ret, class... Args>
auto import_func(const char *module, const char *name) {
  using F = std::function<Ret(Args...)>;
  auto tid = ctype_trait<F>::add();
  Box box;
  ts->import_func(module, name, tid, &box);
  return F(Func<Ret, Args...>(box));
}

/*
 * Represents a Python module.
 */
struct module {
  using add_box_t = std::function<void(const char *, Box)>;
  add_box_t add_box;
  module(add_box_t add_box_) : add_box(add_box_) {}
  template <class T> auto add(const char *name, T t) {
    using ct = ctype_trait<T>;
    ct::add();
    add_box(name, ct::box(t));
  }
};

static PyObject *init_module(const char *name,
                             std::function<void(module)> init) {

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

  auto obj = PyModule_Create(&moduledef);
  if (!obj)
    return nullptr;

  auto err = import_typesytem();
  if (err) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot import typesystem.");
    return nullptr;
  }

  ts->pre_init();
  module m([=](const char *name, Box box) { ts->add_box(obj, name, box); });
  init(m);
  ts->post_init();

  return obj;
}

} // namespace pybindcpp

#define PYBINDCPP_INIT(NAME, INITFUNC)                                         \
  PyMODINIT_FUNC PyInit_##NAME() {                                             \
    return pybindcpp::init_module(#NAME, INITFUNC);                            \
  }

#endif // MODULE_H
