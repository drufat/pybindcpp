# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from pybinding import dispatch


class function(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return dispatch.dispatch(self.func, args)
