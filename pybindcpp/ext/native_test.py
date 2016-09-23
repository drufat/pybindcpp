# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from pybindcpp.ext import native

def test_native():
    '''
    >>> native.__doc__
    >>> native.__name__
    'pybindcpp.ext.native'

    >>> native.parsing(1, 2, 20.0, 5)
    (1, 2, 20.0, 5, 'string', 'string', True, False)
    >>> native.func(4, 0, 0)
    4
    >>> native.one
    1
    >>> native.true
    True
    >>> native.name
    'native'
    >>> native.name1
    'native'

    >>> native.h(2, 3)
    5

    >>> native.f(2, 1, 0)
    3
    >>> native.closure()
    (2, 1, 0)
    >>> native.f(20, 3, 0)
    23
    >>> native.closure()
    (20, 3, 0)
    >>> native.f(3, 1, 0)
    4
    >>> native.closure()
    (3, 1, 0)
    >>> native.f(5, 1, 0)
    6
    >>> native.f(8, 1, 0)
    9
    >>> native.closure()
    (8, 1, 0)

    >>> native.f_one()
    1
    >>> native.f_func()()
    1

    >>> native.S("abc")
    'abc'

    >>> native.g_cfun(1, 2)
    3
    >>> native.g_fun(1, 2)
    3
    >>> native.g_afun(1, 2)
    3
    >>> native.g_ofun(1, 2)
    3

    >>> native.manytypes(3, 1)
    4
    >>> native.manytypes(5, 2)
    7
    >>> native.manytypes(5, "abc")
    'abc'
    >>> native.manytypes(5, 2)
    7
    >>> native.manytypes(5, 2, 1)
    Traceback (most recent call last):
    ...
    TypeError: function takes exactly 2 arguments (3 given)
    >>> native.manytypes(5)
    Traceback (most recent call last):
    ...
    TypeError: function takes exactly 2 arguments (1 given)

    >>> type(native.caps_int)
    <class 'PyCapsule'>
    >>> type(native.caps_string)
    <class 'PyCapsule'>
    >>> native.PyCapsule_GetName(native.caps_int)
    'i'
    >>> native.PyCapsule_GetName(native.caps_double)
    'd'
    '''
    assert native.f(2, 1, 0) == 3
    assert native.closure() == (2, 1, 0)
    assert native.f(20, 3, 0) == 23
    assert native.closure() == (20, 3, 0)


