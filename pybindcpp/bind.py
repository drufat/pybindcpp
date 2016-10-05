import ctypes as ct
from pybindcpp.api import PyCapsule_New


def capsule(cfunc):
    return PyCapsule_New(ct.cast(ct.pointer(cfunc), ct.c_void_p), None, None)


@ct.CFUNCTYPE(ct.py_object, ct.c_char_p)
def cfunctype(signature):
    signature = signature.decode()
    types = tuple(getattr(ct, _) for _ in signature.split(","))
    fn_type = ct.CFUNCTYPE(*types)
    return fn_type


cfunctype_cap = capsule(cfunctype)


def register(func, func_type):
    p = ct.cast(func, ct.POINTER(ct.c_void_p))
    f = ct.cast(p[0], func_type)
    return f


c_register = ct.CFUNCTYPE(ct.py_object, ct.c_void_p, ct.py_object)(register)

register_cap = capsule(c_register)
