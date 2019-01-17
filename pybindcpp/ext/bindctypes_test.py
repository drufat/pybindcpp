import ctypes as ct

import pybindcpp.ext.bindctypes as bc


def test_reg():
    '''
    >>> bc.id_type
    b'(py_object,py_object,)'
    >>> bc.add_type
    b'(c_int,c_int,c_int,)'
    >>> bc.set_string_type
    b'(c_int,c_char,c_int,c_char_p,)'
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
