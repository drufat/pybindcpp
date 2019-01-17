# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import pybindcpp.ext.simple_capi as m_capi
import pybindcpp.ext.simple_ctyp as m_ctyp
import pytest


@pytest.mark.parametrize('m', [m_capi, m_ctyp])
def test_simple(m):
    assert m.g(1, 2) == 1 + 2
    assert m.f(3.0, 4.0) == 12.0
