// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "pybindcpp/module.h"

using namespace pybindcpp;

int
g(int x, int y) {
    return x  + y;
}

auto
simple(Module& m)
{
    m.fun("g", g);
    return NULL;
}


PyMODINIT_FUNC
PyInit_simple(void)
{
    return module_init("simple", simple);
}
