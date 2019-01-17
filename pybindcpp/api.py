"""
>>> s = ct.c_char_p(b'(c_char_p,c_int,c_double,)')
>>> assert get_type(s)._restype_ == ct.c_char_p
>>> assert get_type(s)._argtypes_ == tuple([ct.c_int, ct.c_double])

>>> s = ct.c_char_p(b'(None,)')
>>> assert get_type(s)._restype_ == None
>>> assert get_type(s)._argtypes_ == tuple()
"""

import ctypes as ct
import importlib

storage = {}


@ct.PYFUNCTYPE(ct.py_object, ct.c_char_p)
def get_type(expr):
    expr = expr.decode()
    t = eval(expr, ct.__dict__)
    return ct.PYFUNCTYPE(*t)


@ct.PYFUNCTYPE(ct.c_void_p, ct.c_char_p, ct.c_char_p)
def get_cfunction(module, attr):
    module = module.decode()
    attr = attr.decode()

    mod = importlib.import_module(module)
    cfunc = getattr(mod, attr)
    addr = ct.addressof(cfunc)
    return addr


@ct.PYFUNCTYPE(ct.c_void_p, ct.c_char_p, ct.c_char_p, ct.c_char_p)
def get_pyfunction(module, attr, cfunctype):
    module = module.decode()
    attr = attr.decode()
    cfunc_type = get_type(cfunctype)

    mod = importlib.import_module(module)
    func = getattr(mod, attr)
    cfunc = cfunc_type(func)

    addr = ct.addressof(cfunc)

    # To ensure addr does not become dangling.
    storage[addr] = cfunc

    return addr


@ct.PYFUNCTYPE(ct.py_object, ct.c_void_p, ct.c_char_p)
def func_c(func, func_type):
    p = ct.cast(func, ct.POINTER(ct.c_void_p))
    t = get_type(func_type)
    f = ct.cast(p[0], t)
    return f


@ct.PYFUNCTYPE(ct.py_object, ct.py_object, ct.c_void_p)
def func_std(func_call, func_ptr):
    def func(*args):
        return func_call(func_ptr, *args)

    return func


@ct.PYFUNCTYPE(ct.py_object, ct.py_object)
def vararg(f):
    def v(*args):
        return f(None, args)

    return v


@ct.PYFUNCTYPE(None)
def error():
    raise RuntimeError('RuntimeError')


class API(ct.Structure):
    _fields_ = [
        ('get_cfunction', type(get_cfunction)),
        ('get_pyfunction', type(get_pyfunction)),
        ('func_c', type(func_c)),
        ('func_std', type(func_std)),
        ('vararg', type(vararg)),
        ('error', type(error)),
    ]


api = API(
    get_cfunction,
    get_pyfunction,
    func_c,
    func_std,
    vararg,
    error,
)

api_addr = ct.addressof(api)
