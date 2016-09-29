import ctypes as ct
import types

CTYPE_enum = (
    ct.c_wchar,
    ct.c_char,
    ct.c_ubyte,
    ct.c_short,
    ct.c_ushort,
    ct.c_int,
    ct.c_uint,
    ct.c_long,
    ct.c_ulong,
    ct.c_longlong,
    ct.c_ulonglong,
    ct.c_size_t,
    ct.c_ssize_t,
    ct.c_float,
    ct.c_double,
    ct.c_longdouble,
    ct.c_char_p,
    ct.c_wchar_p,
    ct.c_void_p,
)

VOIDFUNCTYPE = ct.CFUNCTYPE(None)
REGFUNCTYPE = ct.CFUNCTYPE(ct.py_object, ct.c_void_p, ct.POINTER(ct.c_int), ct.c_size_t)

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


def register(func, signature, signature_size):
    types = tuple(CTYPE_enum[signature[_]] for _ in range(signature_size))
    fn_type = ct.CFUNCTYPE(*types)
    f = ct.cast(func, fn_type)
    return f


c_register = REGFUNCTYPE(register)
c_register_cap = PyCapsule_New(ct.cast(c_register, ct.c_void_p), None, None)


def new_module(name):
    '''
    >>> m = new_module(b'abc')
    >>> type(m)
    <class 'module'>
    '''
    return types.ModuleType(name.decode())


NEW_MODULE_FUNCTYPE = ct.CFUNCTYPE(ct.py_object, ct.c_char_p)
c_new_module = NEW_MODULE_FUNCTYPE(new_module)
c_new_module_cap = PyCapsule_New(ct.cast(c_new_module, ct.c_void_p), None, None)
