# Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
import pybindcpp.ext.ufunc as uf

from pybindcpp.helper import eq


def test_1():
    x = np.linspace(0, 1, 5)
    assert tuple(uf.cos.types) == ('d->d',)
    x = x.astype('d')
    assert eq(np.cos(x), uf.cos(x))
    assert tuple(uf.sin.types) == ('f->f',)
    x = x.astype('f')
    assert eq(np.sin(x), uf.sin(x))


def test_2():
    fn = np.vectorize(lambda N, x: N * x)

    for N, x in ((1, 2.0),
                 ([[1], [2]], [1.0, 2.0, 3.0, 4.0]),
                 (1, [[1.0, 2.0], [3.0, 4.0]]),
                 ([[1], [2]], [[1.0, 2.0], [3.0, 4.0]])):
        assert eq(fn(N, x), uf.fn(N, x))

    N = 1
    for x in ([1, 2, 3, 4],
              np.ones(100),
              np.zeros(100)):
        assert eq(fn(N, x), uf.fn(N, x))


def test_3():
    assert eq(uf.add_one([1, 2, 3]), [2, 3, 4])
    assert eq(uf.add_one([3, 2, 1]), [4, 3, 2])


if __name__ == '__main__':
    test_1()
    test_2()
    test_3()
