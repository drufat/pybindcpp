// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef PYBINDCPP_CTYPES_H
#define PYBINDCPP_CTYPES_H

#include <map>
#include <typeindex>

enum CTYPE {
  c_wchar,
  c_byte,
  c_ubyte,
  c_short,
  c_ushort,
  c_int,
  c_uint,
  c_long,
  c_ulong,
  c_longlong,
  c_ulonglong,
  c_size_t,
  c_ssize_t,
  c_float,
  c_double,
  c_longdouble,
  c_char_p,
  c_wchar_p,
  c_void_p,
};

const std::map<std::type_index, CTYPE> ctype_map = {

    {typeid(wchar_t), c_wchar},
    {typeid(char), c_byte},
    {typeid(unsigned char), c_ubyte},
    {typeid(short), c_short},
    {typeid(unsigned short), c_ushort},
    {typeid(int), c_int},
    {typeid(unsigned int), c_uint},
    {typeid(long), c_long},
    {typeid(unsigned long), c_ulong},
    {typeid(long long), c_longlong},
    {typeid(unsigned long long), c_ulonglong},
    {typeid(size_t), c_size_t},
    {typeid(ssize_t), c_ssize_t},
    {typeid(float), c_float},
    {typeid(double), c_double},
    {typeid(long double), c_longdouble},
    {typeid(char *), c_char_p},
    {typeid(wchar_t *), c_wchar_p},
    {typeid(void *), c_void_p},
};

using VOIDFUNCTYPE = void (*)();
using REGFUNCTYPE = _object *(*)(const char *, void *, const int *, size_t);

#endif //PYBINDCPP_CTYPES_H
