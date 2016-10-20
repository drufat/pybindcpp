import ctypes as ct
import importlib

################
# Python.h API #
################

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


###############
# Struct API  #
###############

@ct.PYFUNCTYPE(ct.py_object, ct.c_char_p)
def get_type(typ):
    expr = typ.decode()
    t = eval(expr, ct.__dict__)
    return ct.PYFUNCTYPE(*t)


@ct.PYFUNCTYPE(ct.c_void_p, ct.c_char_p, ct.c_char_p)
def get_capsule(module, attr):
    module = module.decode()
    attr = attr.decode()

    mod = importlib.import_module(module)
    cap = getattr(mod, attr)
    return PyCapsule_GetPointer(cap, None)


@ct.PYFUNCTYPE(ct.c_void_p, ct.c_char_p, ct.c_char_p)
def get_cfunction(module, attr):
    module = module.decode()
    attr = attr.decode()

    mod = importlib.import_module(module)
    cfunc = getattr(mod, attr)
    addr = ct.addressof(cfunc)
    return PyLong_AsVoidPtr(addr)


@ct.PYFUNCTYPE(ct.py_object, ct.c_char_p, ct.c_char_p, ct.py_object)
def get_pyfunction(module, attr, cfunctype):
    module = module.decode()
    attr = attr.decode()

    mod = importlib.import_module(module)
    func = getattr(mod, attr)
    cfunc = cfunctype(func)
    return cfunc


@ct.PYFUNCTYPE(ct.c_void_p, ct.py_object)
def get_addr(cfunc):
    addr = ct.addressof(cfunc)
    return PyLong_AsVoidPtr(addr)


@ct.PYFUNCTYPE(ct.py_object, ct.c_void_p, ct.py_object)
def register_(func, func_type):
    p = ct.cast(func, ct.POINTER(ct.c_void_p))
    f = ct.cast(p[0], func_type)
    return f


@ct.PYFUNCTYPE(ct.py_object, ct.py_object, ct.py_object)
def apply(capsule_call, capsule):
    def func(*args):
        return capsule_call(capsule, *args)

    return func


@ct.PYFUNCTYPE(ct.py_object, ct.py_object)
def vararg(f):
    def v(*args):
        return f(None, args)

    return v


@ct.PYFUNCTYPE(None)
def error():
    raise RuntimeError('RuntimeError')


def api_test():
    '''
    >>> s = ct.c_char_p(b'(c_char_p,c_int,c_double,)')
    >>> assert get_type(s)._restype_ == ct.c_char_p
    >>> assert get_type(s)._argtypes_ == tuple([ct.c_int, ct.c_double])

    >>> s = ct.c_char_p(b'(None,)')
    >>> assert get_type(s)._restype_ == None
    >>> assert get_type(s)._argtypes_ == tuple()

    >>> p = get_capsule(b'pybindcpp.bind', b'register_cap')
    '''
    pass


class API(ct.Structure):
    _fields_ = [
        ('get_type', type(get_type)),
        ('get_capsule', type(get_capsule)),
        ('get_cfunction', type(get_cfunction)),
        ('get_pyfunction', type(get_pyfunction)),
        ('get_addr', type(get_addr)),
        ('register_', type(register_)),
        ('apply', type(apply)),
        ('vararg', type(vararg)),
        ('error', type(error)),
    ]


api = API(
    get_type,
    get_capsule,
    get_cfunction,
    get_pyfunction,
    get_addr,
    register_,
    apply,
    vararg,
    error,
)


@ct.PYFUNCTYPE(ct.py_object, ct.POINTER(ct.POINTER(API)))
def init(ptr):
    ptr[0] = ct.pointer(api)
    return api


init_addr = ct.addressof(init)
