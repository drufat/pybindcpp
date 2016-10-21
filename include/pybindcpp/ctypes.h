// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef PYBINDCPP_TYPES_H
#define PYBINDCPP_TYPES_H

#include <Python.h>
#include <stdbool.h>

#include <map>
#include <string>
#include <typeindex>

namespace pybindcpp {

const std::map<std::type_index, const char*> ctype_map = {

  { typeid(bool), "c_bool" },
  { typeid(wchar_t), "c_wchar" },
  { typeid(char), "c_char" },
  { typeid(unsigned char), "c_ubyte" },
  { typeid(short), "c_short" },
  { typeid(unsigned short), "c_ushort" },
  { typeid(int), "c_int" },
  { typeid(unsigned int), "c_uint" },
  { typeid(long), "c_long" },
  { typeid(unsigned long), "c_ulong" },
  { typeid(long long), "c_longlong" },
  { typeid(unsigned long long), "c_ulonglong" },
  { typeid(size_t), "c_size_t" },
  { typeid(ssize_t), "c_ssize_t" },
  { typeid(float), "c_float" },
  { typeid(double), "c_double" },
  { typeid(long double), "c_longdouble" },

  { typeid(char*), "c_char_p" },
  { typeid(wchar_t*), "c_wchar_p" },
  { typeid(void*), "c_void_p" },
  { typeid(PyObject*), "py_object" },
  { typeid(double*), "POINTER(c_double)" },

  { typeid(const char*), "c_char_p" },
  { typeid(const wchar_t*), "c_wchar_p" },
  { typeid(const void*), "c_void_p" },
  { typeid(const PyObject*), "py_object" },
  { typeid(const double*), "POINTER(c_double)" },

  { typeid(void), "None" },

};

using VOIDFUNCTYPE = std::function<void()>;
using REGFUNCTYPE = std::function<PyObject*(void*, PyObject*)>;
using INIFUNCTYPE = std::function<PyObject*()>;
}

#endif // PYBINDCPP_TYPES_H
