import ctypes as ct


def class_factory(name, ctype):
    fields = [
        ('real', ctype),
        ('imag', ctype),
    ]
    return type(name, (ct.Structure,), {'_fields_': fields})


cfloat = class_factory('cfloat', ct.c_float)
cdouble = class_factory('cdouble', ct.c_double)
clongdouble = class_factory('clongdouble', ct.c_longdouble)
