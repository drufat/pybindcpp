# Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
import ctypes as ct

from pybindcpp.ext import eigen
from pybindcpp.helper import eq

def square(x):

    x = np.ascontiguousarray(x, dtype=np.double)
    y = np.empty_like(x)

    px = x.ctypes.data_as(ct.POINTER(ct.c_double))
    py = y.ctypes.data_as(ct.POINTER(ct.c_double))

    eigen.square(
        px, *x.shape,
        py, *y.shape,
    )

    return y


def test_eigen():
    assert eq(
        square(
            [[0, 0],
             [0, 0]]),
        np.array(
            [[0, 0],
             [0, 0]])
    )

    assert eq(
        square(
            [[1, 2],
             [3, 4]]),
        np.array(
            [[0., 10.],
             [15., 22.]])
    )


if __name__ == '__main__':
    test_eigen()
