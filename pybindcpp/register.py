import ctypes as ct

from pybindcpp import bindctypes

bindctypes_so = ct.PyDLL(bindctypes.__file__)


def fun(name, ret_t, *args_t):
    return ct.CFUNCTYPE(ret_t, *args_t)((name, bindctypes_so))


VOIDFUNCTYPE = ct.CFUNCTYPE(None)
REGFUNCTYPE = ct.CFUNCTYPE(ct.py_object, ct.c_char_p, ct.c_void_p, ct.c_char_p)

create_string = fun("create_string", ct.c_int, ct.c_char, ct.c_int, ct.c_char_p)
bind_init = fun('bind_init', None, REGFUNCTYPE)

module = {}


def register(name, func, signature):
    name = ct.string_at(name).decode()
    signature = ct.string_at(signature).decode()
    types = [getattr(ct, t) for t in signature.split()]
    f = ct.cast(func, ct.CFUNCTYPE(*types))
    module[name] = f


class Funcs(ct.Structure):
    _fields_ = [
        ('reg', REGFUNCTYPE),
    ]


def test_assign():
    funcs = Funcs.in_dll(bindctypes_so, "funcs")
    funcs.reg = REGFUNCTYPE(register)


def test_reg():
    bind_init(REGFUNCTYPE(register))
    assert module['add'](1, 2) == 3
    assert module['add'](100, 2) == 102
    assert module['minus'](2, 1) == 1


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
