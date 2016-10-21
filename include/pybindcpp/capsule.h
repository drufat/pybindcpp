// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef CAPSULE_H
#define CAPSULE_H

#include <Python.h>
#include <memory>

namespace pybindcpp {

template <class T>
void
capsule_destructor(PyObject* o)
{
  auto p = PyCapsule_GetPointer(o, typeid(T).name());
  // decrement reference
  delete static_cast<std::shared_ptr<T>*>(p);
}

template <class T>
PyObject*
capsule_new(std::shared_ptr<T> t)
{
  // increment reference
  auto p = new std::shared_ptr<T>(t);
  return PyCapsule_New(p, typeid(T).name(), &capsule_destructor<T>);
}

template <class T>
std::shared_ptr<T>
capsule_get(PyObject* o)
{
  auto p = PyCapsule_GetPointer(o, typeid(T).name());
  if (!p) {
    PyErr_SetString(PyExc_TypeError, "Capsule is of incorrect type.");
    return NULL;
  }
  return *static_cast<std::shared_ptr<T>*>(p);
}

template <class F, class Ret, class... Args>
Ret
capsule_call(PyObject* o, Args... args)
{
  auto f = capsule_get<F>(o);
  return (*f)(args...);
};
}

#endif // CAPSULE_H
