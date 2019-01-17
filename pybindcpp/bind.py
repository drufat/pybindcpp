"""
>>> p = get_capsule(b'pybindcpp.bind', b'register_cap')

>>> p = ct.c_char_p(b"abcdef")
>>> name = ct.c_char_p(b"char *")
>>> o = PyCapsule_New(p, name, None)
>>> name1 = ct.c_char_p(b"char *")
>>> pp = PyCapsule_GetPointer(o, name1)
>>> ct.cast(pp, ct.c_char_p).value
b'abcdef'

"""
import ctypes as ct

import importlib

PyCapsule_New = ct.PYFUNCTYPE(
    ct.py_object,
    ct.c_void_p, ct.c_char_p, ct.c_void_p
)(('PyCapsule_New', ct.pythonapi))

PyCapsule_GetPointer = ct.PYFUNCTYPE(
    ct.c_void_p,
    ct.py_object, ct.c_char_p
)(('PyCapsule_GetPointer', ct.pythonapi))

PyLong_AsVoidPtr = ct.PYFUNCTYPE(
    ct.c_void_p,
    ct.py_object
)(('PyLong_AsVoidPtr', ct.pythonapi))


@ct.PYFUNCTYPE(ct.c_void_p, ct.c_char_p, ct.c_char_p)
def get_capsule(module, attr):
    module = module.decode()
    attr = attr.decode()

    mod = importlib.import_module(module)
    cap = getattr(mod, attr)
    return PyCapsule_GetPointer(cap, None)


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
