# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
import numpy.fft as fft

from pybindcpp.ext import fftw
from pybindcpp.helper import eq


def test_fftw():
    x = [0, 1, 2, 3]
    assert eq(fftw.fft(x), fft.fft(x))

    x = [0, 1, 2, 3, 4]
    assert eq(fftw.fft(x), fft.fft(x))

    x = [0, 0, 0, 0]
    assert eq(fftw.fft(x), fft.fft(x))

    x = [5, 2, 3, 3 + 2j]
    assert eq(fftw.fft(x), fft.fft(x))

    x = np.arange(1000)
    assert eq(fftw.fft(x), fft.fft(x))

    x = np.random.rand(100) + 1j * np.random.rand(100)
    assert eq(fftw.fft(x), fft.fft(x))


def test_fftw2():
    fft2 = lambda _: np.apply_along_axis(fft.fft, -1, _)

    x = np.arange(16).reshape(4, 4)
    assert eq(fftw.fft2(x), fft2(x))

    x = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    assert eq(fftw.fft2(x), fft2(x))

    x = np.random.rand(7, 7) + 1j * np.random.rand(7, 7)
    assert eq(fftw.fft2(x), fft2(x))

    x = np.random.rand(5, 8) + 1j * np.random.rand(5, 8)
    assert eq(fftw.fft2(x), fft2(x))

    x = np.random.rand(5) + 1j * np.random.rand(5)
    assert eq(fftw.fft2(x), fft2(x))


if __name__ == '__main__':
    test_fftw()
    test_fftw2()
