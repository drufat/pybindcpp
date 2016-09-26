import ctypes as ct
from pybindcpp import bindctypes

from pybindcpp.register import REGFUNCTYPE, Funcs, register, PyCapsule_New, PyCapsule_GetPointer


def test_assign():
    so = ct.PyDLL(bindctypes.__file__)
    funcs = Funcs.in_dll(so, "funcs")
    funcs.reg = REGFUNCTYPE(register)


def test_reg():
    assert bindctypes.add(1, 2) == 3
    assert bindctypes.add(100, 2) == 102
    assert bindctypes.minus(2, 1) == 1
    assert bindctypes.add_d(1., 2.) == 3.


def test_fun():
    '''
    >>> so = ct.PyDLL(bindctypes.__file__)
    >>> create_string = ct.CFUNCTYPE(ct.c_int, ct.c_char, ct.c_int, ct.c_char_p)(("create_string", so))
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
