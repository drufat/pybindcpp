// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef TYPES_H
#define TYPES_H

#include <map>
#include <memory>

#include "python/util.h"
#include "python/capsule.h"

namespace python {


typedef unsigned int uint;
typedef unsigned long ulong;

const std::map<std::type_index, std::string> BuildValueTypes = {

    {typeid(char*),            "s"},
    {typeid(const char*),      "s"},
    {typeid(char),             "b"},
    {typeid(double),           "d"},
    {typeid(float),            "f"},
    {typeid(int),              "i"},
    {typeid(uint),             "I"},
    {typeid(long),             "l"},
    {typeid(ulong),            "k"},
    {typeid(PyObject*),        "N"},

};


const std::map<std::type_index, std::string> ArgParseTypes = {

    {typeid(const char*),      "s"},
    {typeid(unsigned char),    "b"},
    {typeid(double),           "d"},
    {typeid(float),            "f"},
    {typeid(int),              "i"},
    {typeid(uint),             "I"},
    {typeid(long),             "l"},
    {typeid(ulong),            "k"},
    {typeid(PyObject*),        "O"},

};



template<class T>
auto convert(const T& t)
{
    return t;
}


template<>
auto convert<bool>(const bool& value)
{
    auto o = value ? Py_True : Py_False;
    Py_INCREF(o);
    return o;
}


template<>
auto convert<std::string>(const std::string& value)
{
    return PyUnicode_FromString(value.c_str());
}


template<>
auto convert<double>(const double& value)
{
    return PyFloat_FromDouble(value);
}


template<class... Args>
auto build_value_imp(Args... args)
{
    auto format = stringer(BuildValueTypes.at(typeid(Args))...);
    return Py_BuildValue(format.c_str(), args...);
}


template<class... Args>
auto build_value(Args... args)
{
    return build_value_imp(convert(args)...);
}


template<class... Args>
auto arg_parse_tuple(PyObject* obj, Args&... args)
{
    auto format = stringer(ArgParseTypes.at(typeid(Args))...);
    return PyArg_ParseTuple(obj, format.c_str(), &args...);
}


template<class... Args>
auto packtuple(Args... args)
// returns new reference to tuple without incrementing references to members
{
    const Py_ssize_t n = sizeof...(Args);
    auto rslt = PyTuple_Pack(n, args...);
    using List = int[];
    (void) List { Py_DecRef(args)... };
    return rslt;
}

template<class... Args>
auto unpacktuple(PyObject* tuple, Args... args)
// returns borrowed references to tuple members
{
    const Py_ssize_t n = sizeof...(Args);
    return PyArg_UnpackTuple(tuple, "pythontypes", n, n, args...);

}

using VarArg = std::function<PyObject*(PyObject*, PyObject*)>;

template<class T>
PyObject*
var(T &&t)
{
    return build_value<T>(std::forward<T>(t));
}

PyObject*
varargs(std::shared_ptr<VarArg> func)
{
    auto gil = PyGILState_Ensure();

    auto caps = capsule_new(func);
    auto dmod = PyImport_ImportModule("pybinding.function");
    auto dfun = PyObject_GetAttrString(dmod, "function");
    auto args = Py_BuildValue("(O)", caps);
    auto obj = PyObject_Call(dfun, args, NULL);
    Py_DECREF(args);
    Py_DECREF(dfun);
    Py_DECREF(dmod);
    Py_DECREF(caps);

    PyGILState_Release(gil);
    return obj;
}

PyObject*
varargs(VarArg func)
{
    return varargs(std::make_shared<decltype(func)>(func));
}

template<class... Args>
PyObject*
fun(std::shared_ptr<std::function<void(Args...)>> f)
{
    return varargs(std::make_shared<VarArg>(

     [f](PyObject* self, PyObject* args) -> PyObject* {

        auto parse = [args](Args&... a) {
            return arg_parse_tuple(args, a...);
        };

        std::tuple<Args...> tup;
        if(!apply(parse, tup)) return NULL;

        try {
            apply(*f, tup);
            Py_RETURN_NONE;
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return NULL;
        }

    }));
}

template<class Ret, class... Args>
PyObject*
fun(std::shared_ptr<std::function<Ret(Args...)>> f)
{
    return varargs(std::make_shared<VarArg>(

    [f](PyObject* self, PyObject* args) -> PyObject* {

        auto parse = [args](Args&... a) {
            return arg_parse_tuple(args, a...);
        };

        std::tuple<Args...> tup;
        if(!apply(parse, tup)) return NULL;

        try {
            return build_value(apply(*f, tup));
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return NULL;
        }

    }));
}

template<class Ret, class... Args>
PyObject*
fun(std::function<Ret(Args...)> f)
{
    auto ptr = std::make_shared<decltype(f)>(f);
    return fun(ptr);
}

template<class Ret, class... Args>
PyObject*
fun(Ret(*f)(Args...))
{
    auto ptr = std::make_shared<std::function<Ret(Args...)>>(f);
    return fun(ptr);
}

template<class Ret, class Class, class... Args>
PyObject*
fun(Ret(Class::*f)(Args...))
{
    Py_RETURN_NONE;
}

template<class T, class Class>
PyObject*
fun(T Class::*m)
{
    Py_RETURN_NONE;
}

template<typename Class, typename ... Params  >
PyObject*
build_constructor_(Class (*)(Params...))
{
    Py_RETURN_NONE;
}

template<class T>
PyObject*
constructor()
{
    T *f = nullptr;
    return build_constructor_(f);
}

}

#endif // TYPES_H
