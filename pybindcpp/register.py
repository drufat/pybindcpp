import ctypes as ct
import types
import importlib

VOIDFUNCTYPE = ct.CFUNCTYPE(None)
REGFUNCTYPE = ct.CFUNCTYPE(ct.py_object, ct.c_void_p, ct.c_char_p)

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


class Funcs(ct.Structure):
    _fields_ = [
        ('reg', REGFUNCTYPE),
    ]


def add(x, y):
    return x + y


def capsule(cfunc):
    return PyCapsule_New(ct.cast(cfunc, ct.c_void_p), None, None)


def cfunctype(signature):
    signature = signature.decode()
    types = tuple(getattr(ct, _) for _ in signature.split(","))
    fn_type = ct.CFUNCTYPE(*types)
    return fn_type


c_cfunctype = ct.CFUNCTYPE(ct.py_object, ct.c_char_p)(cfunctype)
c_cfunctype_cap = capsule(c_cfunctype)


def register(func, signature):
    fn_type = cfunctype(signature)
    f = ct.cast(func, fn_type)
    return f


c_register = ct.CFUNCTYPE(ct.py_object, ct.c_void_p, ct.c_char_p)(register)
c_register_cap = capsule(c_register)


def new_module(name):
    '''
    >>> m = new_module(b'abc')
    >>> type(m)
    <class 'module'>
    '''
    return types.ModuleType(name.decode())


c_new_module = ct.CFUNCTYPE(ct.py_object, ct.c_char_p)(new_module)
c_new_module_cap = capsule(c_new_module)


def cfuncify(module, name, signature):
    module = module.value.decode()
    mod = importlib.import_module(module)
    name = name.value.decode()
    func = getattr(mod, name)
    return cfunctype(signature)(func)


c_cfuncify = ct.CFUNCTYPE(ct.py_object, ct.c_char_p, ct.c_char_p, ct.POINTER(ct.c_int))(cfuncify)
c_cfuncify_cap = capsule(c_cfuncify)
