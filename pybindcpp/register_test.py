import ctypes as ct
from pybindcpp import bindctypes

from pybindcpp.register import REGFUNCTYPE, Funcs, register

bindctypes_so = ct.PyDLL(bindctypes.__file__)


def fun(name, ret_t, *args_t):
    return ct.CFUNCTYPE(ret_t, *args_t)((name, bindctypes_so))


create_string = fun("create_string", ct.c_int, ct.c_char, ct.c_int, ct.c_char_p)
bind_init = fun('bind_init', None, REGFUNCTYPE)


def test_assign():
    funcs = Funcs.in_dll(bindctypes_so, "funcs")
    reg = register({})
    funcs.reg = REGFUNCTYPE(reg)


def test_reg():
    module = {}
    bind_init(REGFUNCTYPE(register(module)))
    assert module['add'](1, 2) == 3
    assert module['add'](100, 2) == 102
    assert module['minus'](2, 1) == 1
    assert module['add_d'](1., 2.) == 3.


def test_fun():
    '''
    >>> size = 20
    >>> buf = ct.create_string_buffer(size+1)
    >>> create_string(b'a', size, buf)
    0
    >>> buf.raw
    b'aaaaaaaaaaaaaaaaaaaa\\x00'
    >>> buf[0] = b'A'
    >>> buf.raw
    b'Aaaaaaaaaaaaaaaaaaaa\\x00'
    >>> ct.cast(ct.cast(buf, ct.c_void_p), ct.c_char_p).value
    b'Aaaaaaaaaaaaaaaaaaaa'
    '''
    pass


def test_module():
    '''
    >>> bindctypes.__dict__.update({'one':1, 'two':2})
    >>> bindctypes.one
    1
    >>> setattr(bindctypes, 'three', 3)
    >>> bindctypes.three
    3
    '''
    pass


PyCapsule_Destructor = ct.CFUNCTYPE(None, ct.py_object)

PyCapsule_New = ct.pythonapi.PyCapsule_New
PyCapsule_New.restype = ct.py_object
PyCapsule_New.argtypes = (ct.c_void_p, ct.c_char_p, ct.c_void_p)

PyCapsule_GetPointer = ct.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = ct.c_void_p
PyCapsule_GetPointer.argtypes = (ct.py_object, ct.c_char_p)


def test_capsule():
    '''
    >>> p = ct.c_char_p(b"abcdef")
    >>> name = ct.c_char_p(b"char *")
    >>> o = PyCapsule_New(p, name, None)
    >>> name1 = ct.c_char_p(b"char *")
    >>> pp = PyCapsule_GetPointer(o, name1)
    >>> ct.cast(pp, ct.c_char_p).value
    b'abcdef'
    '''
    pass
