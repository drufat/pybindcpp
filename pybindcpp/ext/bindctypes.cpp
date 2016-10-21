// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include <Python.h>

#include <pybindcpp/module.h>

using namespace pybindcpp;

extern "C" int
set_string(char c, int size, char* buffer)
{
  for (int i = 0; i < size; i++) {
    buffer[i] = c;
  }
  return 0;
}

auto
id(PyObject* o)
{
  return o;
}

int
add(int x, int y)
{
  return x + y;
}

double
add_d(double x, double y)
{
  return x + y;
}

int
minus(int x, int y)
{
  return x - y;
};

PyObject*
error()
{
  PyErr_SetString(PyExc_RuntimeError, "Called error().");
  return NULL;
}

void
exec(ExtModule& m)
{

  py_function<int(int)>("pybindcpp.bind", "id")(3);

  m.var("one", 1);
  m.var("two", 2.0);
  m.var("greet", "Hello, World!");

  m.fun_type("id_type", id);
  m.fun_type("add_type", add);
  m.fun_type("minus_type", minus);
  m.fun_type("set_string_type", set_string);

  m.fun("id", id);
  m.fun("add", add);
  m.fun("minus", minus);
  m.fun("add_d", add_d);
  m.fun("set_string", set_string);
  m.fun("mul", [](int x, int y) { return x * y; });

  m.fun("error", error);
  m.varargs("func", [](PyObject* data, PyObject* args) -> PyObject* {
    PyErr_SetString(PyExc_RuntimeError, "Another error.");
    return NULL;
  });
}

PYMODULE_INIT(bindctypes, exec)
