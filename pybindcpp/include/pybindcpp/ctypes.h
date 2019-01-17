// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef PYBINDCPP_TYPES_H
#define PYBINDCPP_TYPES_H

#include <Python.h>
#include <stdbool.h>

#include <functional>
#include <map>
#include <string>
#include <typeindex>

namespace pybindcpp {

static const std::map<size_t, std::string> ctype_map = {

    {typeid(bool).hash_code(), "c_bool"},
    {typeid(wchar_t).hash_code(), "c_wchar"},
    {typeid(char).hash_code(), "c_char"},
    {typeid(unsigned char).hash_code(), "c_ubyte"},
    {typeid(short).hash_code(), "c_short"},
    {typeid(unsigned short).hash_code(), "c_ushort"},
    {typeid(int).hash_code(), "c_int"},
    {typeid(unsigned int).hash_code(), "c_uint"},
    {typeid(long).hash_code(), "c_long"},
    {typeid(unsigned long).hash_code(), "c_ulong"},
    {typeid(long long).hash_code(), "c_longlong"},
    {typeid(unsigned long long).hash_code(), "c_ulonglong"},
    {typeid(size_t).hash_code(), "c_size_t"},
    {typeid(ssize_t).hash_code(), "c_ssize_t"},
    {typeid(float).hash_code(), "c_float"},
    {typeid(double).hash_code(), "c_double"},
    {typeid(long double).hash_code(), "c_longdouble"},

    {typeid(char *).hash_code(), "c_char_p"},
    {typeid(wchar_t *).hash_code(), "c_wchar_p"},
    {typeid(void *).hash_code(), "c_void_p"},
    {typeid(PyObject *).hash_code(), "py_object"},
    {typeid(int *).hash_code(), "POINTER(c_int)"},
    {typeid(double *).hash_code(), "POINTER(c_double)"},

    {typeid(const char *).hash_code(), "c_char_p"},
    {typeid(const wchar_t *).hash_code(), "c_wchar_p"},
    {typeid(const void *).hash_code(), "c_void_p"},
    {typeid(const PyObject *).hash_code(), "py_object"},
    {typeid(const int *).hash_code(), "POINTER(c_int)"},
    {typeid(const double *).hash_code(), "POINTER(c_double)"},

    {typeid(void).hash_code(), "None"},

};

} // namespace pybindcpp

#endif // PYBINDCPP_TYPES_H
