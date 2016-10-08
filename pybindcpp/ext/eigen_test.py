# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
from pybindcpp.helper import eq
from pybindcpp.ext import eigen


def test_eigen():
    '''
    >>> eigen.square(None)
    Traceback (most recent call last):
    ...
    ValueError: object of too small depth for desired array
    '''
    assert eq(
        eigen.square(
            [[0, 0],
             [0, 0]]),
        np.array(
            [[0, 0],
             [0, 0]])
    )

    assert eq(
        eigen.square(
            [[1, 2],
             [3, 4]]),
        np.array(
            [[0., 10.],
             [15., 22.]])
    )
