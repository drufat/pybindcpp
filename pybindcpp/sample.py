import ctypes
import types

from pybindcpp.bind import capsule

VOIDFUNCTYPE = ctypes.CFUNCTYPE(None)
REGFUNCTYPE = ctypes.CFUNCTYPE(ctypes.py_object, ctypes.c_void_p, ctypes.py_object)


class Funcs(ctypes.Structure):
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


c_new_module = ctypes.CFUNCTYPE(ctypes.py_object, ctypes.c_char_p)(new_module)
c_new_module_cap = capsule(c_new_module)
