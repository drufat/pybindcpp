// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef NUMPY_H
#define NUMPY_H

#include <capi/module.h>
#include <functional>
#include <iostream>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace pybindcpp {

static const std::unordered_map<std::type_index, char> NumpyTypes = {

    {typeid(npy_bool), NPY_BOOL},       {typeid(npy_byte), NPY_BYTE},

    {typeid(npy_double), NPY_DOUBLE},   {typeid(npy_float), NPY_FLOAT},
    {typeid(npy_cdouble), NPY_CDOUBLE}, {typeid(npy_cfloat), NPY_CFLOAT},

    {typeid(npy_short), NPY_SHORT},     {typeid(npy_int), NPY_INT},
    {typeid(npy_long), NPY_LONG},       {typeid(npy_ushort), NPY_USHORT},
    {typeid(npy_uint), NPY_UINT},       {typeid(npy_ulong), NPY_ULONG},

    {typeid(PyObject *), NPY_OBJECT},

};

template <class F, class... X> struct loop1d_imp;

template <class F, class A0, class A1> struct loop1d_imp<F, A0, A1> {
  static auto imp(F func) {
    return [func](char **args, npy_intp *dims, npy_intp *steps, void *data) {
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
    return [func](char **args, npy_intp *dims, npy_intp *steps, void *data) {
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
    return [func](char **args, npy_intp *dims, npy_intp *steps, void *data) {
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
    return [func](char **args, npy_intp *dims, npy_intp *steps, void *data) {
      const auto N = dims[0];
      for (auto i = 0; i < N; i++) {
        auto &a0 = *reinterpret_cast<A0 *>(args[0] + i * steps[0]);
        auto &a1 = *reinterpret_cast<A1 *>(args[1] + i * steps[1]);
        auto &a2 = *reinterpret_cast<A2 *>(args[2] + i * steps[2]);
        auto &a3 = *reinterpret_cast<A3 *>(args[3] + i * steps[3]);
        auto &a4 = *reinterpret_cast<A3 *>(args[4] + i * steps[4]);
        a3 = func(a0, a1, a2, a4);
      }
    };
  }
};

using generic = std::function<void(char **args, npy_intp *dims, npy_intp *steps,
                                   void *data)>;

template <class F, class Ret, class... Args> generic loop1d(F func) {
  auto target = func.template target<Ret (*)(Args...)>();
  if (target) {
    std::cout << "- C" << std::endl;
    return loop1d_imp<decltype(*target), Args..., Ret>::imp(*target);
  } else {
    // less efficient
    std::cout << "- C++" << std::endl;
    return loop1d_imp<F, Args..., Ret>::imp(func);
  }
}

static void generic_target(char **args, npy_intp *dims, npy_intp *steps,
                           void *data) {
  auto f = *static_cast<generic *>(data);
  return f(args, dims, steps, nullptr);
}

struct UFuncObjects {
  std::vector<generic> funcs;
  std::vector<char> types;
  std::vector<PyUFuncGenericFunction> cfuncs;
  std::vector<void *> cdata;
};

static PyObject *make_ufunc_imp(const char *name, std::vector<generic> funcs,
                                std::vector<char> types, size_t nin,
                                size_t nout) {
  auto ntypes = funcs.size();
  assert(types.size() == ntypes * (nin + nout));

  // to eventually store as a capsule
  auto objs = std::make_shared<UFuncObjects>();
  objs->funcs = funcs;
  objs->types = types;

  for (auto &f : objs->funcs) {

    auto t = f.target<PyUFuncGenericFunction>();
    if (t) {
      std::cout << "= C" << std::endl;
      objs->cfuncs.push_back(*t);
      objs->cdata.push_back(nullptr);
    } else {
      std::cout << "= C++" << std::endl;
      objs->cfuncs.push_back(generic_target);
      objs->cdata.push_back(static_cast<void *>(&f));
    }
  }
  assert(objs->cdata.size() == ntypes);
  assert(objs->cfuncs.size() == ntypes);

  auto __func = PyUFunc_FromFuncAndData(objs->cfuncs.data(), objs->cdata.data(),
                                        objs->types.data(), ntypes, nin, nout,
                                        PyUFunc_None, name, nullptr, 0);
  // store the __objs for as long as __func lives
  auto __objs = capsule_new(objs);

  auto mod = PyImport_ImportModule("pybindcpp.function");
  auto ufunc = PyObject_GetAttrString(mod, "ufunc");
  auto o = PyObject_CallFunctionObjArgs(ufunc, __func, __objs, nullptr);
  Py_DecRef(__objs);
  Py_DecRef(__func);
  Py_DecRef(ufunc);
  Py_DecRef(mod);

  return o;
}

template <class F> struct ufunc_trait;

template <class Ret, class... Args> struct ufunc_trait<Ret (*)(Args...)> {
  static PyObject *get(const char *name, Ret (*func)(Args...)) {
    using T = ufunc_trait<std::function<Ret(Args...)>>;
    return T::get(name, func);
  }
};

template <class F>
struct ufunc_trait : public ufunc_trait<decltype(&F::operator())> {};

template <class F, class Ret, class... Args>
struct ufunc_trait<Ret (F::*)(Args...) const> {
  static PyObject *get(const char *name, F func) {
    constexpr size_t nin = sizeof...(Args);
    constexpr size_t nout = 1;

    return make_ufunc_imp(
        name,
        {
            loop1d<F, Ret, Args...>(func),
        },
        {NumpyTypes.at(typeid(Args))..., NumpyTypes.at(typeid(Ret))}, nin,
        nout);
  }
};

template <class F> void ufunc(ExtModule &m, const char *name, F f) {
  m.var(name, ufunc_trait<F>::get(name, f));
}

} // namespace pybindcpp

#endif // NUMPY_H
