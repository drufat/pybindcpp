import ctypes as ct
import types

from pybindcpp.bind import capsule

VOIDFUNCTYPE = ct.CFUNCTYPE(None)
REGFUNCTYPE = ct.CFUNCTYPE(ct.py_object, ct.c_void_p, ct.py_object)


class Funcs(ct.Structure):
    _fields_ = [
        ('reg', REGFUNCTYPE),
    ]


def add(x, y):
    return x + y


def id(_):
    return _


def new_module(name):
    '''
    >>> m = new_module(b'abc')
    >>> type(m)
    <class 'module'>
    '''
    return types.ModuleType(name.decode())


c_new_module = ct.CFUNCTYPE(ct.py_object, ct.c_char_p)(new_module)
c_new_module_cap = capsule(c_new_module)
