#ifndef PYBINDCPP_CAPSULE_H
#define PYBINDCPP_CAPSULE_H

#include "pybindcpp/ctypes/types.h"

namespace pybindcpp {

template<class T>
T
capsule(const char *module, const char *attr) {

  auto mod = PyImport_ImportModule(module);
  if (!mod) throw;

  auto cap = PyObject_GetAttrString(mod, attr);
  if (!cap) throw;

  auto pnt = PyCapsule_GetPointer(cap, nullptr);
  if (!pnt) throw;

  T reg = reinterpret_cast<T>(pnt);

  Py_DecRef(cap);
  Py_DecRef(mod);

  return reg;
}

template<class F>
struct py_function;

template<class Ret, class ...Args>
struct py_function<Ret(Args...)> {

  PyObject *m_ptr;
  Ret (*f_ptr)(Args...);

  py_function(const char *module, const char *name) {
    using F = func_trait<decltype(f_ptr)>;

    auto func_type = F::pyctype();

    using FUNCIFY = void (*)(
        const char *, const char *, PyObject *,
        PyObject **, void **
    );
    auto funcify = capsule<FUNCIFY>("pybindcpp.register", "c_cfuncify_cap");
    void *v_ptr;
    funcify(module, name, func_type, &m_ptr, &v_ptr);
    f_ptr = (decltype(f_ptr)) v_ptr;
  }

  Ret operator()(Args... args) {
    return f_ptr(args...);
  }
};

}

#endif //PYBINDCPP_CAPSULE_H
