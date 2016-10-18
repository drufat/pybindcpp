# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np
from pybindcpp.helper import eq
from pybindcpp.ext import numpy as nn


def test_numpy():
    fn = np.vectorize(lambda N, x: N * x)

    for N, x in (
            (1, 2),
            (1, [1, 2, 3, 4]),
            (1, [[1, 2], [3, 4]]),
            ([1, 2], [[1, 2], [3, 4]]),
    ):
        assert eq(fn(N, x),
                  nn.fn_ufunc(N, x),
                  nn.fn_ufunc1(N, x),
                  nn.fn_ufunc2(N, x),
                  nn.fn_ufunc3(N, x),
                  )

    for N, x in (
            (1, 2),
    ):
        assert eq(fn(N, x),
                  nn.fn(N, x))

    for N, x in (
            (1, [1, 2, 3, 4]),
            (1, np.ones(100)),
            (1, np.zeros(100)),
    ):
        assert eq(fn(N, x),
                  nn.fn_array(N, x),
                  nn.fn_array1(N, x),
                  nn.fn_array2(N, x),
                  )


def test_ufunc():
    assert eq(
        nn.add_one(np.array([1, 2, 3])),
        np.array([2, 3, 4])
    )
    assert eq(
        nn.add_one(np.array([3, 2, 1])),
        np.array([4, 3, 2])
    )
