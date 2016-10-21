// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

#include <cmath>

#ifdef NATIVE_CPP
#include <pybindcpp/module_cpp.h>
#else
#include <pybindcpp/cpython_types.h>
#include <pybindcpp/module.h>
#endif

using namespace pybindcpp;

int
f(int N, int n, int x)
{
  return N + n + x;
}

PyObject*
py_parsing(PyObject* self, PyObject* args)
{
  int i, j;
  double d;
  long l;
  arg_parse_tuple(args, i, j, d, l);
  std::string str = "string";
  return build_value(i, j, d, l, str, str.c_str(), true, false);
}

PyObject*
py_func(PyObject* self, PyObject* args)
{
  int N, n, x;
  arg_parse_tuple(args, N, n, x);
  auto out = f(N, n, x);
  return build_value(out);
}

int
g(int x, int y)
{
  return x + y;
}

void
add_one(int N, double* x)
{
  for (int i = 0; i < N; i++) {
    x[i] += 1;
  }
}

extern "C" PyObject*
py_g(PyObject* self, PyObject* args)
{
  int x, y;
  arg_parse_tuple(args, x, y);
  auto out = g(x, y);
  return build_value(out);
}

constexpr double pi = M_PI;
constexpr double half = 0.5;

void
module(ExtModule& m)
{

  m.var("half", half);
  m.var("pi", pi);

  m.var("one", (long)1);
  m.var("two", (unsigned long)2);
  m.var("true", true);
  m.var("false", false);
  m.var("name", "native");
  m.var("name1", "native");

  static int N, n, x;

  m.fun("f", [&](int N_, int n_, int x_) {
    N = N_;
    n = n_;
    x = x_;
    return f(N, n, x);
  });

  m.fun("closure", [&]() {

    return build_value(N, n, x);

  });

  m.fun("add_one", add_one);

  m.varargs("h", py_g);

  m.varargs("parsing", py_parsing);
  m.varargs("func", py_func);

  m.varargs("manytypes", [](PyObject* self, PyObject* args) -> PyObject* {
    {
      unsigned int N;
      double i;
      if (arg_parse_tuple(args, N, i)) {
        auto out = N + (int)i;
        return build_value(out);
      }
    }
    PyErr_Clear();
    {
      unsigned int N;
      PyObject* i;
      if (arg_parse_tuple(args, N, i)) {
        auto out = i;
        return build_value(out);
      }
    }
    return NULL;
  });

  m.fun("S", [](PyObject* o) { return build_value(o); });

  m.fun("g_cfun", g);
  m.fun("g_fun", std::function<int(int, int)>(g));
  m.fun<std::function<int(int, int)>>(
    "g_afun", [](int x, int y) -> int { return g(x, y); });
  m.add("g_ofun", fun2obj(&g));

  auto f_one = std::function<int()>([]() { return 1; });
  m.fun("f_one", f_one);

  auto f_func = std::function<PyObject*()>([=]() { return fun2obj(f_one); });
  m.fun("f_func", f_func);

  m.add("caps_int", capsule_new(std::make_shared<int>(3)));
  m.add("caps_double", capsule_new(std::make_shared<double>(3.0)));
  m.add("caps_string", capsule_new(std::make_shared<std::string>("Hello!")));
  m.fun("PyCapsule_GetName", PyCapsule_GetName);
}

#ifdef NATIVE_CPP
PYMODULE_INIT(native_cpp, module)
#else
PYMODULE_INIT(native, module)
#endif
