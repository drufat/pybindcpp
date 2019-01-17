# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

import math

import pybindcpp.ext.example_capi as m_capi
import pybindcpp.ext.example_ctyp as m_ctyp
import pytest


@pytest.mark.parametrize('m', [m_capi, m_ctyp])
def test_example(m):
    assert round(m.pi, 2) == 3.14
    assert m.half == 0.5
    assert m.one == 1
    assert m.true == True
    assert m.false == False
    assert m.name == b'pybindcpp'

    assert m.f(1, 2, 3) == 1 + 2 + 3
    assert m.mycos(0.1) == math.cos(0.1)
    assert m.cos(0.1) == math.cos(0.1)
    assert m.sin(0.1) == math.sin(0.1)
