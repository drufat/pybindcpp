# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from pybindcpp import dispatch


class function(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return dispatch.dispatch(self.func, args)


class ufunc(object):
    def __init__(self, __func, __objs):
        self.__func = __func
        self.__doc__ = __func.__doc__
        self.__objs = __objs

    def __call__(self, *args, **kwargs):
        return self.__func(*args, **kwargs)

