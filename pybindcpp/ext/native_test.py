# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import pytest
from pybindcpp.ext import native, native_cpp


@pytest.mark.parametrize('m,name', [
    (native, 'pybindcpp.ext.native'),
    (native_cpp, 'pybindcpp.ext.native_cpp'),
])
def test_native(m, name):

    assert (m.__name__ == name)

    assert round(m.pi, 2) == 3.14
    assert m.half == 0.5

    assert m.one == 1
    assert m.true == True
    assert m.false == False
    assert m.name == b'native'
    assert m.name1 == b'native'

    assert (
        m.parsing(1, 2, 20.0, 5)
        ==
        (1, 2, 20.0, 5, b'string', b'string', True, False)
    )
    assert m.func(4, 0, 0) == 4

    assert m.h(2, 3) == 5

    assert m.f(2, 1, 0) == 3
    assert m.closure() == (2, 1, 0)
    assert m.f(20, 3, 0) == 23
    assert m.closure() == (20, 3, 0)
    assert m.f(3, 1, 0) == 4
    assert m.closure() == (3, 1, 0)
    assert m.f(5, 1, 0) == 6
    assert m.f(8, 1, 0) == 9
    assert m.closure() == (8, 1, 0)

    assert m.f_one() == 1
    assert m.f_func()() == 1

    assert m.S('abc') == 'abc'

    assert m.g_cfun(1, 2) == 3
    assert m.g_fun(1, 2) == 3
    assert m.g_afun(1, 2) == 3
    assert m.g_ofun(1, 2) == 3

    assert m.f(2, 1, 0) == 3
    assert m.closure() == (2, 1, 0)
    assert m.f(20, 3, 0) == 23
    assert m.closure() == (20, 3, 0)

    assert m.manytypes(3, 1) == 4
    assert m.manytypes(5, 2) == 7
    assert m.manytypes(5, 'abc') == 'abc'
    assert m.manytypes(5, 2) == 7

    with pytest.raises(TypeError):
        m.manytypes(5)
    with pytest.raises(TypeError):
        m.manytypes(5, 2, 1)

    assert m.PyCapsule_GetName(m.caps_int) == b'i'
    assert m.PyCapsule_GetName(m.caps_double) == b'd'
