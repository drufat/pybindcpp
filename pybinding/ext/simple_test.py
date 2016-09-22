# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from pybinding.ext import simple


def test_simple():
    assert simple.g(1, 2) == 1 + 2
