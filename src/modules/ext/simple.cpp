// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "python/module.h"

using namespace python;

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
