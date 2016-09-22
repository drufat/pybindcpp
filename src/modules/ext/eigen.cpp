// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "python/module.h"
#include "python/numpy.h"

#include <complex>
#include <cmath>
#include <Eigen/Core>

using namespace python;
using namespace Eigen;


template<class T>
void
computation(const T& X, T& Y)
{
    Y = X.matrix()*X.matrix();
    Y(0, 0) = 0.0;
}


auto
eigen(Module& m)
{
    import_array();

    m.varargs("square", [](PyObject* self, PyObject* args) -> PyObject*
    {
        PyObject* o;
        if(!arg_parse_tuple(args, o))
            return NULL;

        const auto x = (PyArrayObject*)PyArray_ContiguousFromAny(o,
                                                                 NPY_DOUBLE,
                                                                 2, 2);
        if (!x) return NULL;

        auto y = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(x),
                                               PyArray_DIMS(x),
                                               NPY_DOUBLE,
                                               0);
        if (!y) return NULL;


        Map<ArrayXXd> X((double*)PyArray_DATA(x), PyArray_DIMS(x)[0], PyArray_DIMS(x)[1]);
        Map<ArrayXXd> Y((double*)PyArray_DATA(y), PyArray_DIMS(y)[0], PyArray_DIMS(y)[1]);

        computation(X, Y);

        Py_DECREF(x);
        return (PyObject*)y;
    });
}


PyMODINIT_FUNC
PyInit_eigen(void)
{
    return module_init("eigen", eigen);
}
