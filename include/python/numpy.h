// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef NUMPY_H
#define NUMPY_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

#include "python/module.h"

namespace python {
namespace {


const std::map<std::type_index, char> NumpyTypes = {

    {typeid(double),           NPY_DOUBLE},
    {typeid(float),            NPY_FLOAT},
    {typeid(int),              NPY_INT},
    {typeid(uint),             NPY_UINT},
    {typeid(long),             NPY_LONG},
    {typeid(ulong),            NPY_ULONG},
    {typeid(PyObject*),        NPY_OBJECT},

};


template <typename Out, typename F, typename... In, std::size_t... Is>
decltype(auto)
loop1d_imp(F func, std::index_sequence<Is...>)
{
    return [func](char **args,
            npy_intp *dimensions,
            npy_intp* steps,
            void* data) {

        constexpr size_t nin = sizeof...(In);
        //constexpr size_t nout = 1;

        const auto N = dimensions[0];
        for (auto i = 0; i < N; i++) {
            auto& out = (*(Out*)(args[nin] + i*steps[nin]));
            out = func(*(In*)(args[Is] + i*steps[Is])...);
        }
    };
}


template <typename Out, typename F, typename... In>
decltype(auto)
loop1d(F func)
{
    using IndexIn = std::make_index_sequence<sizeof...(In)>;
    return loop1d_imp<Out, F, In...>(func, IndexIn{});
}


using ftype = std::function<void(char**, npy_intp*, npy_intp*, void*)>;

void
dispatch(char **args,
         npy_intp *dimensions,
         npy_intp* steps,
         void* data)
{
    auto f = *(ftype*)data;
    return f(args, dimensions, steps, NULL);
}

PyObject*
create_ufunc(
        std::string name_,
        std::vector<ftype> funcs_,
        std::vector<char> types_,
        int nin,
        int nout
        )
{
    auto ntypes = funcs_.size();
    assert(types_.size() == ntypes*(nin + nout));

    auto name = store(name_);
    auto funcs = store(funcs_);
    auto types = store(types_);

    auto cfuncs = store(std::vector<PyUFuncGenericFunction>());
    auto cdata = store(std::vector<void*>());

    for (auto& f : *funcs) {

        auto t = f.target<PyUFuncGenericFunction>();
        if (t) {
            cfuncs->push_back(*t);
            cdata->push_back(NULL);
        } else {
            cfuncs->push_back(dispatch);
            cdata->push_back((void*)&f);
        }
    }
    assert(cdata->size() == ntypes == cfuncs->size());

    return PyUFunc_FromFuncAndData(
                cfuncs->data(),
                cdata->data(),
                types->data(),
                ntypes,
                nin, nout,
                PyUFunc_None,
                name->c_str(), NULL, 0);

}

template<class O, class F, class... I>
PyObject*
make_ufunc(std::string name, F f)
{
    constexpr auto nin = sizeof...(I);
    constexpr auto nout = 1;
    return create_ufunc(name, {
                            loop1d<O, F, I...>(f),
                        }, {
                            NumpyTypes.at(typeid(I))...,
                            NumpyTypes.at(typeid(O))
                        }, nin, nout);
}


inline
void
ufunc_raw(Module& m,
          std::string name,
          std::vector<ftype> funcs,
          std::vector<char> types,
          int nin, int nout)
{
    m.var(name, create_ufunc(name, funcs, types, nin, nout));
}

template<class O, class F, class... I>
void
ufunc(Module& m, std::string name, F f)
{
    m.var(name, make_ufunc<O, F, I...>(name, f));
}



} // end anonymous namespace
} // end python namespace

#endif // NUMPY_H

#ifndef NUMPY_H
#define NUMPY_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

#include "module.h"

namespace python {
namespace {


const std::map<std::type_index, char> NumpyTypes = {

    {typeid(double),           NPY_DOUBLE},
    {typeid(float),            NPY_FLOAT},
    {typeid(int),              NPY_INT},
    {typeid(uint),             NPY_UINT},
    {typeid(long),             NPY_LONG},
    {typeid(ulong),            NPY_ULONG},
    {typeid(PyObject*),        NPY_OBJECT},

};


template <typename Out, typename F, typename... In, std::size_t... Is>
decltype(auto)
loop1d_imp(F func, std::index_sequence<Is...>)
{
    return [func](char **args,
            npy_intp *dimensions,
            npy_intp* steps,
            void* data) {

        constexpr size_t nin = sizeof...(In);
        //constexpr size_t nout = 1;

        const auto N = dimensions[0];
        for (auto i = 0; i < N; i++) {
            auto& out = (*(Out*)(args[nin] + i*steps[nin]));
            out = func(*(In*)(args[Is] + i*steps[Is])...);
        }
    };
}


template <typename Out, typename F, typename... In>
decltype(auto)
loop1d(F func)
{
    using IndexIn = std::make_index_sequence<sizeof...(In)>;
    return loop1d_imp<Out, F, In...>(func, IndexIn{});
}


using ftype = std::function<void(char**, npy_intp*, npy_intp*, void*)>;

void
dispatch(char **args,
         npy_intp *dimensions,
         npy_intp* steps,
         void* data)
{
    auto f = *(ftype*)data;
    return f(args, dimensions, steps, NULL);
}

PyObject*
create_ufunc(
        std::string name_,
        std::vector<ftype> funcs_,
        std::vector<char> types_,
        int nin,
        int nout
        )
{
    auto ntypes = funcs_.size();
    assert(types_.size() == ntypes*(nin + nout));

    auto name = store(name_);
    auto funcs = store(funcs_);
    auto types = store(types_);

    auto cfuncs = store(std::vector<PyUFuncGenericFunction>());
    auto cdata = store(std::vector<void*>());

    for (auto& f : *funcs) {

        auto t = f.target<PyUFuncGenericFunction>();
        if (t) {
            cfuncs->push_back(*t);
            cdata->push_back(NULL);
        } else {
            cfuncs->push_back(dispatch);
            cdata->push_back((void*)&f);
        }
    }
    assert(cdata->size() == ntypes == cfuncs->size());

    return PyUFunc_FromFuncAndData(
                cfuncs->data(),
                cdata->data(),
                types->data(),
                ntypes,
                nin, nout,
                PyUFunc_None,
                name->c_str(), NULL, 0);

}

template<class O, class F, class... I>
PyObject*
make_ufunc(std::string name, F f)
{
    constexpr auto nin = sizeof...(I);
    constexpr auto nout = 1;
    return create_ufunc(name, {
                            loop1d<O, F, I...>(f),
                        }, {
                            NumpyTypes.at(typeid(I))...,
                            NumpyTypes.at(typeid(O))
                        }, nin, nout);
}


inline
void
ufunc_raw(Module& m,
          std::string name,
          std::vector<ftype> funcs,
          std::vector<char> types,
          int nin, int nout)
{
    m.var(name, create_ufunc(name, funcs, types, nin, nout));
}

template<class O, class F, class... I>
void
ufunc(Module& m, std::string name, F f)
{
    m.var(name, make_ufunc<O, F, I...>(name, f));
}



} // end anonymous namespace
} // end python namespace

#endif // NUMPY_H

