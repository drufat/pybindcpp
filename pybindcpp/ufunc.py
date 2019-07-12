# Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.
import ctypes as ct

import pybindcpp.core.ufunc as uf

[
    NPY_BOOL, NPY_BYTE, NPY_UBYTE, NPY_SHORT, NPY_USHORT,
    NPY_INT, NPY_UINT, NPY_LONG, NPY_ULONG, NPY_LONGLONG,
    NPY_ULONGLONG, NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
] = uf.numpy_types[:uf.numpy_types_size]

type_char = {
    ct.c_bool: NPY_BOOL,
    ct.c_char: NPY_BYTE, ct.c_ubyte: NPY_UBYTE,
    ct.c_short: NPY_SHORT, ct.c_ushort: NPY_USHORT,
    ct.c_int: NPY_INT, ct.c_uint: NPY_UINT,
    ct.c_long: NPY_LONG, ct.c_ulong: NPY_ULONG,
    ct.c_longlong: NPY_LONGLONG, ct.c_ulonglong: NPY_ULONGLONG,
    ct.c_float: NPY_FLOAT, ct.c_double: NPY_DOUBLE, ct.c_longdouble: NPY_LONGDOUBLE,
}

PyUFuncGenericFunction = type(uf.generic_function)


def args_ufunc(name, fn, loop1d):
    func = (PyUFuncGenericFunction * 1)()
    func[:] = PyUFuncGenericFunction(loop1d),
    ntypes = 1

    data = (ct.c_void_p * 1)()
    data[:] = None,

    restype = type_char[fn.restype]
    argtypes = [type_char[_] for _ in fn.argtypes]
    nout = 1
    nin = len(argtypes)
    n = (nin + nout) * ntypes
    types = (ct.c_char * n)()
    types[:] = [*argtypes, restype]

    name_ = ct.c_char_p(name)

    args = (func, data, types, ntypes, nin, nout, name_)
    return args


def make_ufunc(args):
    (func, data, types, ntypes, nin, nout, name_) = args
    return uf.pyufunc(func, data, types, ntypes, nin, nout, name_)
