// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include<Python.h>

#include <map>
#include <typeindex>

//const std::map<std::type_index, std::string> ctype_map = {
//
//    {typeid(wchar_t), "c_wchar"},
//    {typeid(char), "c_byte"},
//    {typeid(unsigned char), "c_ubyte"},
//    {typeid(short), "c_short"},
//    {typeid(unsigned short), "c_ushort"},
//    {typeid(int), "c_int"},
//    {typeid(unsigned int), "c_uint"},
//    {typeid(long), "c_long"},
//    {typeid(unsigned long), "c_ulong"},
//    {typeid(long long), "c_longlong"},
//    {typeid(unsigned long long), "c_ulonglong"},
//
//};

extern "C"
int create_string(char c, int size, char *buffer) {
  for (int i = 0; i < size; i++) {
    buffer[i] = c;
  }
  return 0;
}

int add(int x, int y) {
  return x + y;
}

int minus(int x, int y) {
  return x - y;
}

using VOIDFUNCTYPE = void (*)();
using REGFUNCTYPE = PyObject *(*)(const char *, void *, const char *);

struct Funcs {
  REGFUNCTYPE reg;
};

extern "C" {
struct Funcs funcs;
}

extern "C"
void bind_init(REGFUNCTYPE reg) {
  reg("add", reinterpret_cast<void *>(add), "c_int c_int c_int");
  reg("minus", reinterpret_cast<void *>(minus), "c_int c_int c_int");

  auto a = reinterpret_cast<void *>(add);
  auto b = reinterpret_cast<int (*)(int, int)>(a);
  b(10, 10);
}

static struct PyModuleDef moduledef =
    {
        PyModuleDef_HEAD_INIT,
        "bybindctypes",  // m_name
        nullptr,         // m_doc
        0,               // m_size
        nullptr,         // m_methods
        nullptr,         // m_slots
        nullptr,         // m_traverse
        nullptr,         // m_clear
        nullptr,         // m_free
    };

PyMODINIT_FUNC
PyInit_bindctypes(void) {
  auto m = PyModule_Create(&moduledef);

  auto cap = PyCapsule_New(reinterpret_cast<void*>(bind_init), "capsule", nullptr);
  Py_DECREF(cap);

  {
    auto dmod = PyImport_ImportModule("pybindcpp.helper");
    auto dfun = PyObject_GetAttrString(dmod, "eq");
    auto args = Py_BuildValue("(ii)", 1, 2);
    auto obj = PyObject_Call(dfun, args, NULL);
    Py_DECREF(args);
    Py_DECREF(dfun);
    Py_DECREF(dmod);
    PyModule_AddObject(m, "nothing", obj);
  }

  return m;
}
