import ctypes as ct
from pybindcpp.api import PyCapsule_New


def id(arg):
    return arg


def capsule(cfunc):
    return PyCapsule_New(ct.cast(ct.pointer(cfunc), ct.c_void_p), None, None)


@ct.PYFUNCTYPE(ct.py_object, ct.c_char_p)
def cfunctype(signature):
    signature = signature.decode()
    types = tuple(getattr(ct, _) for _ in signature.split(","))
    fn_type = ct.PYFUNCTYPE(*types)
    return fn_type


cfunctype_cap = capsule(cfunctype)


@ct.CFUNCTYPE(ct.py_object, ct.c_void_p, ct.py_object)
def register(func, func_type):
    p = ct.cast(func, ct.POINTER(ct.c_void_p))
    f = ct.cast(p[0], func_type)
    return f


register_cap = capsule(register)
