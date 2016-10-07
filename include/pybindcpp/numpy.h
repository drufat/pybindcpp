// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef NUMPY_H
#define NUMPY_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>

#include "capsule.h"

namespace pybindcpp {
namespace {

const std::map<std::type_index, char> NumpyTypes = {

    {typeid(double), NPY_DOUBLE},
    {typeid(float), NPY_FLOAT},
    {typeid(int), NPY_INT},
    {typeid(unsigned int), NPY_UINT},
    {typeid(long), NPY_LONG},
    {typeid(unsigned long), NPY_ULONG},
    {typeid(PyObject *), NPY_OBJECT},

};

template<typename Out, typename F, typename... In, std::size_t... Is>
decltype(auto)
loop1d_imp(F func, std::index_sequence<Is...>) {
  return [func](
      char **args,
      npy_intp *dimensions,
      npy_intp *steps,
      void *data
  ) {
    constexpr size_t nin = sizeof...(In);

    const auto N = dimensions[0];
    for (auto i = 0; i < N; i++) {
      auto &out = (*reinterpret_cast<Out * >(args[nin] + i * steps[nin]));
      out = func(*reinterpret_cast<In * >(args[Is] + i * steps[Is])...);
    }
  };
}

template<typename Out, typename F, typename... In>
decltype(auto)
loop1d(F func) {
  using IndexIn = std::make_index_sequence<sizeof...(In)>;
  return loop1d_imp<Out, F, In...>(func, IndexIn{});
}

using ftype = std::function<void(char **, npy_intp *, npy_intp *, void *)>;

void
generic_target(char **args,
               npy_intp *dimensions,
               npy_intp *steps,
               void *data) {
  auto f = *static_cast<ftype *>(data);
  return f(args, dimensions, steps, NULL);
}

struct UFuncObjects {
  std::vector<ftype> funcs;
  std::vector<char> types;
  std::vector<PyUFuncGenericFunction> cfuncs;
  std::vector<void *> cdata;
};

PyObject *
make_ufunc_imp(
    const char *name,
    std::vector<ftype> funcs,
    std::vector<char> types,
    int nin,
    int nout
) {
  auto ntypes = funcs.size();
  assert(types.size() == ntypes * (nin + nout));

  auto objs = std::make_shared<UFuncObjects>();
  objs->funcs = funcs;
  objs->types = types;

  for (auto &f : objs->funcs) {

    auto t = f.target<PyUFuncGenericFunction>();
    if (t) {
      objs->cfuncs.push_back(*t);
      objs->cdata.push_back(nullptr);
    } else {
      objs->cfuncs.push_back(generic_target);
      objs->cdata.push_back((void *) &f);
    }
  }
  assert(objs->cdata.size() == ntypes == objs->cfuncs.size());

  auto __func = PyUFunc_FromFuncAndData(
      objs->cfuncs.data(),
      objs->cdata.data(),
      objs->types.data(),
      ntypes,
      nin, nout,
      PyUFunc_None,
      name, NULL, 0
  );
  // store the __objs for as long as __func lives
  auto __objs = capsule_new(objs);

  PyObject *o;
  {
    auto mod = PyImport_ImportModule("pybindcpp.function");
    auto ufunc = PyObject_GetAttrString(mod, "ufunc");
    o = PyObject_CallFunctionObjArgs(ufunc, __func, __objs, nullptr);
    Py_DecRef(ufunc);
    Py_DecRef(mod);
  }

  Py_DecRef(__objs);
  Py_DecRef(__func);
  return o;
}

template<class O, class F, class... I>
PyObject *
make_ufunc(const char *name, F f) {
  constexpr auto nin = sizeof...(I);
  constexpr auto nout = 1;
  return make_ufunc_imp(
      name,
      {
          loop1d<O, F, I...>(f),
      },
      {
          NumpyTypes.at(typeid(I))...,
          NumpyTypes.at(typeid(O))
      },
      nin,
      nout
  );
}

inline
void
ufunc_raw(ExtModule &m,
          const char *name,
          std::vector<ftype> funcs,
          std::vector<char> types,
          int nin, int nout) {
  m.var(name, make_ufunc_imp(name, funcs, types, nin, nout));
}

template<class O, class F, class... I>
void
ufunc(ExtModule &m, const char *name, F f) {
  m.var(name, make_ufunc<O, F, I...>(name, f));
}

} // end anonymous namespace
} // end python namespace

#endif // NUMPY_H

