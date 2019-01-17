#ifndef PYBINDCPP_API_H
#define PYBINDCPP_API_H

#include <Python.h>

namespace pybindcpp {

struct API {
  void *(*get_cfunction)(const char *, const char *);
  void *(*get_pyfunction)(const char *, const char *, const char *);
  PyObject *(*func_c)(void *, const char *);
  PyObject *(*func_std)(PyObject *, void *);
  PyObject *(*vararg)(PyObject *);
  void (*error)();
};

static API *api;

} // namespace pybindcpp
#endif // PYBINDCPP_API_H
