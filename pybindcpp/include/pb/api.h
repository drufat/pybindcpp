#ifndef PYBINDCPP_CTYP_API_H
#define PYBINDCPP_CTYP_API_H

#include <Python.h>

namespace pybindcpp {

struct Box {
  size_t tid;
  void *ptr;
  void (*deleter)(void *);
};

struct TypeSystem {
  size_t type_counter;

  size_t (*add_type)(size_t, const size_t *, size_t, const char *);
  void (*add_caller)(size_t, Box);
  void (*add_callback)(size_t, size_t);

  PyObject *(*unbox)(Box);

  void (*import_func)(const char *, const char *, size_t, Box *);

  void (*pre_init)();
  void (*post_init)();
};

} // namespace pybindcpp
#endif // PYBINDCPP_CTYP_API_H
