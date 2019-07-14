// Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.

#include <cmath>

#include <pybindcpp/module.h>

using namespace pybindcpp;

static double half = 0.5;
constexpr double pi = M_PI;

constexpr char hello[] = "Hello, World!";
static void *ptr = nullptr;

int f(int N, int n, int x) { return N + n + x; }

double mycos(double x) { return cos(x); }

int sum(const int *x, size_t N) {
  int s = 0;
  for (size_t i = 0; i < N; i++) {
    s += x[i];
  }
  return s;
}

template <class T> T add(T x, T y) { return x + y; }

void import(module m) {
  m.add("half", half);
  m.add("pi", pi);
  m.add("one", static_cast<int>(1));
  m.add("two", static_cast<unsigned long>(2));
  m.add("true", true);
  m.add("false", false);
  m.add("name", "name");
  m.add("ptr_double", &half);

  m.add("ptr_char", hello);
  m.add("ptr_void", ptr);

  m.add("f", f);
  m.add("mycos", mycos);
  m.add("sin", static_cast<double (*)(double)>(sin));
  m.add("sum", sum);

  m.add("add_d", add<double>);
  m.add("add_i", add<int>);

  m.add("cos", [](double x) -> double { return cos(x); });
  m.add("mycos_", std::function<double(double)>(mycos));

  static int x = 0;
  m.add("get_x", [&]() -> int { return x; });
  m.add("set_x", [&](int y) { x = y; });

  m.add("apply", [](std::function<int(int)> f, int arg) { return f(arg); });
  m.add("get", [](int x) { return [x]() { return x; }; });

  m.add("fapp", [](std::function<int(std::function<int(int)>, int)> a, int x) {
    auto f = [](int y) { return y; };
    return a(f, x);
  });
  m.add("farg", [](std::function<int(std::function<int(int)>)> g,
                   std::function<int(int)> h) { return g(h); });
  m.add("fret",
        [](int x) { return [x]() { return [x]() { return x + 1; }; }; });

  m.add("fidentity", [](std::function<int(int)> f) { return f; });

  m.add("import_func", [](const char *name) {
    return import_func<double, double>("math", name);
  });

  m.add("import_sin", import_func<double, double>("math", "sin"));
  m.add("import_log", import_func<double, double>("math", "log"));

  m.add("py_double", [](PyObject *o) { return PyNumber_Add(o, o); });
  m.add("py_square", [](PyObject *o) { return PyNumber_Multiply(o, o); });
}

PYBINDCPP_INIT(example, import)
