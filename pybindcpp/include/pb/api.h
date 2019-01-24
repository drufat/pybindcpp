#ifndef PYBINDCPP_CTYP_API_H
#define PYBINDCPP_CTYP_API_H

#include <Python.h>

namespace pybindcpp {

enum Native { CPP, PY };

struct Box {
  size_t tid;
  void *ptr;
  void (*deleter)(void *);
};

struct API {
  void (*print_)(const char *);
  void (*error)();

  void (*pre_init)();
  void (*post_init)();

  size_t (*add_type)(size_t, const size_t *, size_t, const char *);
  void (*add)(const char *, Box);
  void (*add_caller)(size_t, Box);
  void (*add_callback)(size_t, size_t);

  void (*commit_add)();

  void *(*get_cfunction)(const char *, const char *);
  void *(*get_pyfunction)(const char *, const char *, const char *);
  PyObject *(*func_c)(void *, const char *);
  PyObject *(*func_std)(PyObject *, void *);
  PyObject *(*vararg)(PyObject *);
};

} // namespace pybindcpp
#endif // PYBINDCPP_CTYP_API_H
