// Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.
#ifndef UFUNC_H
#define UFUNC_H

#include "module.h"
#include <functional>

namespace pybindcpp {

template <class F, class... X> struct loop1d_imp;

template <class F, class A0, class A1> struct loop1d_imp<F, A0, A1> {
  static auto imp(F func) {
    return [func](char **args, intptr_t *dims, intptr_t *steps, void *data) {
      const auto N = dims[0];
      for (auto i = 0; i < N; i++) {
        auto &a0 = *reinterpret_cast<A0 *>(args[0] + i * steps[0]);
        auto &a1 = *reinterpret_cast<A1 *>(args[1] + i * steps[1]);
        a1 = func(a0);
      }
    };
  }
};

template <class F, class A0, class A1, class A2>
struct loop1d_imp<F, A0, A1, A2> {
  static auto imp(F func) {
    return [func](char **args, intptr_t *dims, intptr_t *steps, void *data) {
      const auto N = dims[0];
      for (auto i = 0; i < N; i++) {
        auto &a0 = *reinterpret_cast<A0 *>(args[0] + i * steps[0]);
        auto &a1 = *reinterpret_cast<A1 *>(args[1] + i * steps[1]);
        auto &a2 = *reinterpret_cast<A2 *>(args[2] + i * steps[2]);
        a2 = func(a0, a1);
      }
    };
  }
};

template <class F, class A0, class A1, class A2, class A3>
struct loop1d_imp<F, A0, A1, A2, A3> {
  static auto imp(F func) {
    return [func](char **args, intptr_t *dims, intptr_t *steps, void *data) {
      const auto N = dims[0];
      for (auto i = 0; i < N; i++) {
        auto &a0 = *reinterpret_cast<A0 *>(args[0] + i * steps[0]);
        auto &a1 = *reinterpret_cast<A1 *>(args[1] + i * steps[1]);
        auto &a2 = *reinterpret_cast<A2 *>(args[2] + i * steps[2]);
        auto &a3 = *reinterpret_cast<A3 *>(args[3] + i * steps[3]);
        a3 = func(a0, a1, a2);
      }
    };
  }
};

template <class F, class A0, class A1, class A2, class A3, class A4>
struct loop1d_imp<F, A0, A1, A2, A3, A4> {
  static auto imp(F func) {
    return [func](char **args, intptr_t *dims, intptr_t *steps, void *data) {
      const auto N = dims[0];
      for (auto i = 0; i < N; i++) {
        auto &a0 = *reinterpret_cast<A0 *>(args[0] + i * steps[0]);
        auto &a1 = *reinterpret_cast<A1 *>(args[1] + i * steps[1]);
        auto &a2 = *reinterpret_cast<A2 *>(args[2] + i * steps[2]);
        auto &a3 = *reinterpret_cast<A3 *>(args[3] + i * steps[3]);
        auto &a4 = *reinterpret_cast<A4 *>(args[4] + i * steps[4]);
        a4 = func(a0, a1, a2, a3);
      }
    };
  }
};

template <class F, class A0, class A1, class A2, class A3, class A4, class A5>
struct loop1d_imp<F, A0, A1, A2, A3, A4, A5> {
  static auto imp(F func) {
    return [func](char **args, intptr_t *dims, intptr_t *steps, void *data) {
      const auto N = dims[0];
      for (auto i = 0; i < N; i++) {
        auto &a0 = *reinterpret_cast<A0 *>(args[0] + i * steps[0]);
        auto &a1 = *reinterpret_cast<A1 *>(args[1] + i * steps[1]);
        auto &a2 = *reinterpret_cast<A2 *>(args[2] + i * steps[2]);
        auto &a3 = *reinterpret_cast<A3 *>(args[3] + i * steps[3]);
        auto &a4 = *reinterpret_cast<A4 *>(args[4] + i * steps[4]);
        auto &a5 = *reinterpret_cast<A5 *>(args[5] + i * steps[5]);
        a5 = func(a0, a1, a2, a3, a4);
      }
    };
  }
};

template <class Ret, class... Args>
void add_ufunc(module &m, const char *name, Ret (*fn)(Args...)) {
  using F = Ret (*)(Args...);
  using L = std::function<void(char **, intptr_t *, intptr_t *, void *)>;
  auto args_ufunc =
      import_func<PyObject *, char *, F, L>("pybindcpp.ufunc", "args_ufunc");
  auto make_ufunc =
      import_func<PyObject *, PyObject *>("pybindcpp.ufunc", "make_ufunc");
  L loop1d = loop1d_imp<F, Args..., Ret>::imp(fn);

  PyObject *args = args_ufunc(const_cast<char *>(name), fn, loop1d);
  //  std::cout << "args  " << args->ob_refcnt << std::endl;
  PyObject *ufunc = make_ufunc(args);
  //  std::cout << "ufunc " << ufunc->ob_refcnt << std::endl;
  m.add(name, ufunc);
  // delete reference to ufunc but not to args
  Py_DecRef(ufunc);
  //  std::cout << "ufunc " << ufunc->ob_refcnt << std::endl;
}

// using LOOP1D = std::function<void(char **args, intptr_t *dims, intptr_t
// *steps,
//                                  void *data)>;

// template <class F, class Ret, class... Args> LOOP1D loop1d(F func) {
//  using CF = Ret (*)(Args...);
//  auto target = func.template target<CF>();
//  if (target) {
//    return loop1d_imp<decltype(*target), Args..., Ret>::imp(*target);
//  } else {
//    // less efficient
//    return loop1d_imp<F, Args..., Ret>::imp(func);
//  }
//}

} // namespace pybindcpp

#endif // UFUNC_H
