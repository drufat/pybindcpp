# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np


def eq(*args):
    '''
    >>> eq(1, 1)
    True
    >>> eq(1, 2)
    False
    >>> eq(1, 2, 3)
    False
    >>> eq(1, 1, 1)
    True

    '''
    for l, r in zip(args[:-1], args[1:]):
        if not np.allclose(l, r):
            return False
    return True
