// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef NUMPY_H
#define NUMPY_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <vector>
#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>

#include "capsule.h"

namespace pybindcpp {
namespace {

const std::map<std::type_index, char> NumpyTypes = {

    {typeid(npy_bool), NPY_BOOL},
    {typeid(npy_byte), NPY_BYTE},

    {typeid(npy_double), NPY_DOUBLE},
    {typeid(npy_float), NPY_FLOAT},
    {typeid(npy_cdouble), NPY_CDOUBLE},
    {typeid(npy_cfloat), NPY_CFLOAT},

    {typeid(npy_short), NPY_SHORT},
    {typeid(npy_int), NPY_INT},
    {typeid(npy_long), NPY_LONG},
    {typeid(npy_ushort), NPY_USHORT},
    {typeid(npy_uint), NPY_UINT},
    {typeid(npy_ulong), NPY_ULONG},

    {typeid(PyObject *), NPY_OBJECT},

};

template<class Ret, class F, class... Args, std::size_t... Is>
decltype(auto)
loop1d_imp(F func, std::index_sequence<Is...>) {
  return [func](
      char **args,
      npy_intp *dimensions,
      npy_intp *steps,
      void *data
  ) {
    constexpr size_t nin = sizeof...(Args);

    const auto N = dimensions[0];
    for (auto i = 0; i < N; i++) {
      auto &out = (*reinterpret_cast<Ret * >(args[nin] + i * steps[nin]));
      out = func(*reinterpret_cast<Args * >(args[Is] + i * steps[Is])...);
    }
  };
}

template<class Ret, class F, class... Args>
decltype(auto)
loop1d(F func) {
  using IndexIn = std::make_index_sequence<sizeof...(Args)>;
  return loop1d_imp<Ret, F, Args...>(func, IndexIn{});
}

using pyufuncgenericfuncion = std::function<
    void(
        char **args,
        npy_intp *dimensions,
        npy_intp *strides,
        void *innerloopdata
    )
>;

void
generic_target(
    char **args,
    npy_intp *dimensions,
    npy_intp *steps,
    void *data
) {
  auto f = *static_cast<pyufuncgenericfuncion *>(data);
  return f(args, dimensions, steps, NULL);
}

struct UFuncObjects {
  std::vector<pyufuncgenericfuncion> funcs;
  std::vector<char> types;
  std::vector<PyUFuncGenericFunction> cfuncs;
  std::vector<void *> cdata;
};

PyObject *
make_ufunc_imp(
    const char *name,
    std::vector<pyufuncgenericfuncion> funcs,
    std::vector<char> types,
    int nin,
    int nout
) {
  auto ntypes = funcs.size();
  assert(types.size() == ntypes * (nin + nout));

  // to eventually store as a capsule
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
  assert(objs->cdata.size() == ntypes);
  assert(objs->cfuncs.size() == ntypes);

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

template<class Ret, class F, class... Args>
PyObject *
make_ufunc(const char *name, F f) {
  constexpr auto nin = sizeof...(Args);
  constexpr auto nout = 1;
  return make_ufunc_imp(
      name,
      {
          loop1d<Ret, F, Args...>(f),
      },
      {
          NumpyTypes.at(typeid(Args))...,
          NumpyTypes.at(typeid(Ret))
      },
      nin,
      nout
  );
}

inline
void
ufunc_raw(
    ExtModule &m,
    const char *name,
    std::vector<pyufuncgenericfuncion> funcs,
    std::vector<char> types,
    int nin, int nout
) {
  m.var(name, make_ufunc_imp(name, funcs, types, nin, nout));
}

template<class Ret, class F, class... Args>
void
ufunc(ExtModule &m, const char *name, F f) {
  m.var(name, make_ufunc<Ret, F, Args...>(name, f));
}

} // end anonymous namespace
} // end python namespace

#endif // NUMPY_H

