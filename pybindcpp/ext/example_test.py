# Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.

from math import cos, sin

import pybindcpp.ext.example as m


def test_example():
    # numbers
    assert round(m.pi, 2) == 3.14
    assert m.half == 0.5
    assert m.one == 1

    # boolean
    assert m.true is True
    assert m.false is False

    # string
    assert m.name == b'pybindcpp'

    # functions
    assert m.f(1, 2, 3) == 1 + 2 + 3
    assert m.mycos(0.1) == cos(0.1)
    assert m.cos(0.1) == cos(0.1)
    assert m.sin(0.1) == sin(0.1)


if __name__ == '__main__':
    test_example()
