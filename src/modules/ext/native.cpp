// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "pybindcpp/module.h"

using namespace pybindcpp;

struct MyClass {

  int memberdata;

  MyClass(int a) {
    memberdata = a;
  }

  void method() {
    memberdata++;
  }

  void method1(int a) {
    memberdata = a;
  }
};

int
f(int N, int n, int x) {
  return N + n + x;
}

PyObject *
py_parsing(PyObject *self, PyObject *args) {
  int i, j;
  double d;
  long l;
  arg_parse_tuple(args, i, j, d, l);
  std::string str = "string";
  return build_value(i, j, d, l, str, str.c_str(), true, false);
}

PyObject *
py_func(PyObject *self, PyObject *args) {
  int N, n, x;
  arg_parse_tuple(args, N, n, x);
  auto out = f(N, n, x);
  return build_value(out);
}

int
g(int x, int y) {
  return x + y;
}

extern "C"
PyObject *
py_g(PyObject *self, PyObject *args) {
  int x, y;
  arg_parse_tuple(args, x, y);
  auto out = g(x, y);
  return build_value(out);
}

void
native(ExtModule &m) {
  m.var("one", (long) 1);
  m.var("two", (ulong) 2);
  m.var("true", true);
  m.var("false", false);
  m.var("name", "native");
  m.var("name1", std::string("native"));

  m.varargs("h", py_g);

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

  m.varargs("manytypes", [](PyObject *self, PyObject *args) -> PyObject * {
    {
      uint N;
      double i;
      if (arg_parse_tuple(args, N, i)) {
        auto out = N + (int) i;
        return build_value(out);
      }
    }

    PyErr_Clear();

    {
      uint N;
      PyObject *i;
      if (arg_parse_tuple(args, N, i)) {
        auto out = i;
        return build_value(out);
      }
    }
    return NULL;
  });

  m.fun("S", [](PyObject *o) {
    return build_value(o);
  });

  m.fun("g_cfun", g);
  m.fun("g_fun", std::function<int(int, int)>(g));
  m.fun<std::function<int(int, int) >>(
      "g_afun",
      [](int x, int y) -> int {
        return g(x, y);
      }
  );
  m.add("g_ofun", fun(g));

  auto f_one = std::function<int()>(
      []() {
        return 1;
      }
  );
  m.fun("f_one", f_one);

  auto f_func = std::function<PyObject *()>(
      [=]() {
        return fun(f_one);
      }
  );
  m.fun("f_func", f_func);

  m.varargs("parsing", py_parsing);
  m.varargs("func", py_func);

  m.add("MyClass", constructor<MyClass(int)>());
  m.add("memberdata", method(&MyClass::memberdata));
  m.add("method", method(&MyClass::method));

  m.add("caps_int", capsule_new(std::make_shared<int>(3)));
  m.add("caps_double", capsule_new(std::make_shared<double>(3.0)));
  m.add("caps_string", capsule_new(std::make_shared<std::string>("Hello!")));
  m.fun("PyCapsule_GetName", PyCapsule_GetName);

}

PyMODINIT_FUNC
PyInit_native(void) {
  return module_init("native", native);
}

