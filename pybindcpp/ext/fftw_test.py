# Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.
import ctypes as ct

import numpy as np
import numpy.fft as nf
import pybindcpp.ext.fftw as m

from pybindcpp.complex import cdouble
from pybindcpp.helper import eq


def fft(x):
    x = np.ascontiguousarray(x, dtype=np.cdouble)
    y = np.empty_like(x)

    px = x.ctypes.data_as(ct.POINTER(cdouble))
    py = y.ctypes.data_as(ct.POINTER(cdouble))

    m.fft(x.shape[0], px, py)

    return y


def test_1():
    x = [0, 1, 2, 3]
    assert eq(fft(x), nf.fft(x))

    x = [0, 1, 2, 3, 4]
    assert eq(fft(x), nf.fft(x))

    x = [0, 0, 0, 0]
    assert eq(fft(x), nf.fft(x))

    x = [5, 2, 3, 3 + 2j]
    assert eq(fft(x), nf.fft(x))

    x = np.arange(1000)
    assert eq(fft(x), nf.fft(x))

    x = np.random.rand(100) + 1j * np.random.rand(100)
    assert eq(fft(x), nf.fft(x))


def fft2(x):
    if len(x.shape) == 2:
        M, N = x.shape
    else:
        M, N = 1, x.shape[0]
    x = np.ascontiguousarray(x, dtype=np.cdouble)
    y = np.empty_like(x)

    px = x.ctypes.data_as(ct.POINTER(cdouble))
    py = y.ctypes.data_as(ct.POINTER(cdouble))

    m.fft2(N, M, px, py)

    return y


def test_2():
    np_fft2 = lambda _: np.apply_along_axis(nf.fft, -1, _)

    x = np.arange(16).reshape(4, 4)
    assert eq(fft2(x), np_fft2(x))

    x = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    assert eq(fft2(x), np_fft2(x))

    x = np.random.rand(7, 7) + 1j * np.random.rand(7, 7)
    assert eq(fft2(x), np_fft2(x))

    x = np.random.rand(5, 8) + 1j * np.random.rand(5, 8)
    assert eq(fft2(x), np_fft2(x))

    x = np.random.rand(5) + 1j * np.random.rand(5)
    assert eq(fft2(x), np_fft2(x))


if __name__ == '__main__':
    test_1()
    test_2()
