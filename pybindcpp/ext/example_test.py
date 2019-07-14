# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

import math

import pybindcpp.ext.example as m


def test_1():
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
    m.set_x(0)


def test_2():
    def _apply(f, arg):
        return f(arg)

    assert _apply(lambda _: _ + 1, 2) == 3
    assert m.apply(lambda _: _ + 1, 2) == 3

    def p_get(x):
        def f():
            return x

        return f

    assert p_get(4)() == 4
    assert m.get(4)() == 4

    def p_fapp(a, x):
        def f(y): return y

        return a(f, x)

    assert p_fapp(lambda _, x: _(x), 1) == 1
    assert m.fapp(lambda _, x: _(x), 1) == 1

    def p_farg(g, h):
        return g(h)

    assert p_farg(lambda _: _(2), lambda _: 3 + _) == 5
    assert m.farg(lambda _: _(2), lambda _: 3 + _) == 5

    def p_fret(x):
        return lambda: lambda: x + 1

    assert p_fret(6)()() == 7
    assert m.fret(6)()() == 7

    def p_fidentity(f):
        return f

    def inc(x):
        return x + 1

    assert p_fidentity(inc)(8) == 9
    assert m.fidentity(inc)(8) == 9
    for i in range(100):
        inc = m.fidentity(inc)
    assert m.fidentity(inc)(8) == 9


def test_3():
    assert m.import_func(b'log')(2.0) == math.log(2.0)
    assert m.import_func(b'cos')(2.0) == math.cos(2.0)
    assert m.import_func(b'sin')(2.0) == math.sin(2.0)

    assert m.import_sin(2.0) == math.sin(2.0)
    assert m.import_log(2.0) == math.log(2.0)


def test_4():
    assert m.py_double(3.0) == 6.0
    assert m.py_double((0, 1, 2)) == (0, 1, 2, 0, 1, 2)
    assert m.py_double([0, 1, 2]) == [0, 1, 2, 0, 1, 2]
    assert m.py_square(3.0) == 9.0


if __name__ == '__main__':
    test_1()
    test_2()
    test_3()
    test_4()
