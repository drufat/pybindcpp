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


def capsule(cfunc):
    return PyCapsule_New(ct.cast(cfunc, ct.c_void_p), None, None)


def cfunctype(signature):
    signature = signature.decode()
    types = tuple(getattr(ct, _) for _ in signature.split(","))
    fn_type = ct.CFUNCTYPE(*types)
    return fn_type


c_cfunctype = ct.CFUNCTYPE(ct.py_object, ct.c_char_p)(cfunctype)
c_cfunctype_cap = capsule(c_cfunctype)


def register(func, func_type):
    f = ct.cast(func, func_type)
    return f


c_register = ct.CFUNCTYPE(ct.py_object, ct.c_void_p, ct.py_object)(register)
c_register_cap = capsule(c_register)


def cfuncify(module, name, func_type, cfunc, cptr):
    mod = importlib.import_module(module.decode())
    func = getattr(mod, name.decode())
    cfunc[0] = func_type(func)
    cptr[0] = ct.cast(cfunc[0], ct.c_void_p)


c_cfuncify = ct.CFUNCTYPE(
    None,
    ct.c_char_p, ct.c_char_p, ct.py_object,
    ct.POINTER(ct.py_object), ct.POINTER(ct.c_void_p)
)(cfuncify)
c_cfuncify_cap = capsule(c_cfuncify)
