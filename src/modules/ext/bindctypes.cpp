// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include<Python.h>

#include <map>
#include <typeindex>

const std::map<std::type_index, std::string> ctype_map = {

    {typeid(wchar_t),           "c_wchar"},
    {typeid(char),              "c_byte"},
    {typeid(unsigned char),     "c_ubyte"},
    {typeid(short),             "c_short"},
    {typeid(unsigned short),    "c_ushort"},
    {typeid(int),               "c_int"},
    {typeid(unsigned int),      "c_uint"},
    {typeid(long),              "c_long"},
    {typeid(unsigned long),     "c_ulong"},
    {typeid(long long),         "c_longlong"},
    {typeid(unsigned long long),"c_ulonglong"},

};

extern "C"
int create_string(char c, int size, char* buffer)
{
    for (int i = 0; i<size; i++)
    {
        buffer[i] = c;
    }
    return 0;
}

extern "C"
int add(int x, int y)
{
    return x + y;
}

extern "C"
int minus(int x, int y)
{
    return x - y;
}

extern "C"
PyObject*
register_function(void(*func)() , const char* signature)
{
    //    typeid(char*);
    //    printf("%s\n", ctype_map.at(typeid(char)).c_str());
    Py_RETURN_NONE;
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
PyInit_bindctypes(void)
{
    auto m = PyModule_Create(&moduledef);

    auto f1 = register_function(reinterpret_cast<void(*)()>(&add), "c_int, c_int, c_int");
    auto f2 = register_function(reinterpret_cast<void(*)()>(&minus), "c_int, c_int, c_int");

    {
        auto dmod = PyImport_ImportModule("pybindcpp.helper");
        auto dfun = PyObject_GetAttrString(dmod, "nothing");
        auto args = Py_BuildValue("()");
        auto obj = PyObject_Call(dfun, args, NULL);
        Py_DECREF(args);
        Py_DECREF(dfun);
        Py_DECREF(dmod);
        PyModule_AddObject(m, "nothing", obj);
    }

    Py_INCREF(Py_None);
    PyModule_AddObject(m, "none", Py_None);

    return m;
}
