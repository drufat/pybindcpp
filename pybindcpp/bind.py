import ctypes as ct
import importlib
import types

################
# Python.h API #
################
PyCapsule_Destructor = ct.CFUNCTYPE(
    None,
    ct.py_object
)

PyCapsule_New = ct.CFUNCTYPE(
    ct.py_object,
    ct.c_void_p, ct.c_char_p, ct.c_void_p
)(('PyCapsule_New', ct.pythonapi))

PyCapsule_GetPointer = ct.CFUNCTYPE(
    ct.c_void_p,
    ct.py_object, ct.c_char_p
)(('PyCapsule_GetPointer', ct.pythonapi))

PyLong_AsVoidPtr = ct.CFUNCTYPE(
    ct.c_void_p,
    ct.py_object
)(('PyLong_AsVoidPtr', ct.pythonapi))


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
    f = ct.cast(func, func_type)
    return f


c_register = ct.CFUNCTYPE(ct.py_object, ct.c_void_p, ct.py_object)(register)

register_cap = capsule(c_register)
