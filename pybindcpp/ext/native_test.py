# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import ctypes as ct

import pybindcpp.ext.native_capi as m_capi
import pybindcpp.ext.native_ctyp as m_ctyp
import pytest


@pytest.mark.parametrize('m', [m_capi, m_ctyp])
def test_native(m):
    assert round(m.pi, 2) == 3.14
    assert m.half == 0.5

    assert m.one == 1
    assert m.true is True
    assert m.false is False
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


# core dumps on m_capi
@pytest.mark.parametrize('m', [m_ctyp])
def test_native1(m):
    N = 5
    a = (ct.c_double * N)(*range(N))
    assert tuple(a) == (0.0, 1.0, 2.0, 3.0, 4.0)
    m.add_one(N, a)
    assert tuple(a) == (1.0, 2.0, 3.0, 4.0, 5.0)
    m.add_one(N, a)
    assert tuple(a) == (2.0, 3.0, 4.0, 5.0, 6.0)
