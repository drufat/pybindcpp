import ctypes as ct

CTYPE_enum = (
    ct.c_wchar,
    ct.c_byte,
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


class Funcs(ct.Structure):
    _fields_ = [
        ('reg', REGFUNCTYPE),
    ]


def register(func, signature, signature_size):
    types = tuple(CTYPE_enum[signature[_]] for _ in range(signature_size))
    f = ct.cast(func, ct.CFUNCTYPE(*types))
    return f


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

c_register_t = REGFUNCTYPE(register)
c_register_name = ct.c_char_p(b'pybindcpp.register.c_register')
c_register = PyCapsule_New(ct.cast(c_register_t, ct.c_void_p), c_register_name, None)


def add(x, y):
    return x + y
