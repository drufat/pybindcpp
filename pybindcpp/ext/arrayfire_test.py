# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
import numpy.fft as fft
import os
import pytest

from pybindcpp.helper import eq

try:
    from pybindcpp.ext import arrayfire as af
except:
    af = None


@pytest.mark.skipif(af is None, reason='Requires arrayfire')
def test_fft():
    af.set_backend('cpu')

    x = [0, 1, 2, 3]
    assert eq(af.fft(x), fft.fft(x))

    x = [0, 1, 2, 3, 4]
    assert eq(af.fft(x), fft.fft(x))

    x = [0, 0, 0, 0]
    assert eq(af.fft(x), fft.fft(x))

    x = [5, 2, 3, 3 + 2j]
    assert eq(af.fft(x), fft.fft(x))

    x = np.arange(1000)
    assert eq(af.fft(x), fft.fft(x))

    x = np.random.rand(100) + 1j * np.random.rand(100)
    assert eq(af.fft(x), fft.fft(x))
