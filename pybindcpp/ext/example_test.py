# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

import math

import pybindcpp.ext.example as m


def test_example():
    assert round(m.pi, 2) == 3.14
    assert m.half == 0.5
    assert m.one == 1
    assert m.true is True
    assert m.false is False
    assert m.name == b'name'

    assert m.f(1, 2, 3) == 1 + 2 + 3
    assert m.mycos(0.1) == math.cos(0.1)
    assert m.sin(0.1) == math.sin(0.1)

    assert m.add_d(1.0, 2.0) == 3.0
    assert m.add_i(1, 2) == 3

    assert m.cos(0.1) == math.cos(0.1)
    assert m.mycos_(0.1) == math.cos(0.1)

    assert m.get_x() == 0
    m.set_x(1)
    assert m.get_x() == 1

    assert m.apply(lambda _: _ + 1, 2) == 3
    assert m.get(4)() == 4

    assert m.fapp(lambda _, x: _(x), 1) == 1
    assert m.farg(lambda _: _(2), lambda _: 3 + _) == 5
    assert m.fret(6)()() == 7

    def inc(x):
        return x + 1

    assert m.fidentity(inc)(8) == 9
    for i in range(100):
        inc = m.fidentity(inc)
    assert m.fidentity(inc)(8) == 9
