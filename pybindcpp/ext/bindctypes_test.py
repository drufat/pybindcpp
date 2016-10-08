import ctypes as ct
from pybindcpp.api import PyCapsule_New, PyCapsule_GetPointer
from pybindcpp.bind import register
import pybindcpp.ext.bindctypes as bc


def test_reg():
    '''
    >>> t = lambda _: (_._restype_,) + _._argtypes_
    >>> t(bc.id_type)
    (<class 'ctypes.py_object'>, <class 'ctypes.py_object'>)
    >>> t(bc.add_type)
    (<class 'ctypes.c_int'>, <class 'ctypes.c_int'>, <class 'ctypes.c_int'>)
    >>> t(bc.set_string_type)
    (<class 'ctypes.c_int'>, <class 'ctypes.c_char'>, <class 'ctypes.c_int'>, <class 'ctypes.c_char_p'>)
    >>> bc.error()
    Traceback (most recent call last):
    ...
    RuntimeError: Called error().
    >>> bc.func()
    Traceback (most recent call last):
    ...
    RuntimeError: Another error.
    '''
    assert bc.id(1) == 1
    assert bc.add(1, 2) == 3
    assert bc.add(100, 2) == 102
    assert bc.minus(2, 1) == 1
    assert bc.add_d(1., 2.) == 3.
    assert bc.mul(2, 3) == 6


def test_extern_c():
    '''
    >>> so = ct.PyDLL(bc.__file__)
    >>> set_string = ct.CFUNCTYPE(ct.c_int, ct.c_char, ct.c_int, ct.c_char_p)(("set_string", so))
    >>> size = 20

    >>> buf = b'\\x00'*(size+1)
    >>> set_string(b'a', size, buf)
    0
    >>> buf
    b'aaaaaaaaaaaaaaaaaaaa\\x00'
    >>> ct.cast(ct.cast(buf, ct.c_void_p), ct.c_char_p).value
    b'aaaaaaaaaaaaaaaaaaaa'

    >>> buf = b'\\x00'*(size+1)
    >>> bc.set_string(b'o', size, buf)
    0
    >>> buf
    b'oooooooooooooooooooo\\x00'
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
