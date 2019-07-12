import ctypes as ct
import importlib
import sys

from pybindcpp.api import Box, TypeSystem

ctypemap = {
    'bool': ct.c_bool,
    'wchar_t': ct.c_wchar,
    'char': ct.c_char,
    'unsigned char': ct.c_ubyte,
    'short': ct.c_short,
    'unsigned short': ct.c_ushort,
    'int': ct.c_int,
    'unsigned int': ct.c_uint,
    'long': ct.c_long,
    'unsigned long': ct.c_ulong,
    'long long': ct.c_longlong,
    'unsigned long long': ct.c_ulonglong,

    # 'size_t': ct.c_size_t,
    # 'ssize_t': ct.c_ssize_t,

    'float': ct.c_float,
    'double': ct.c_double,
    'long double': ct.c_longdouble,

    'char *': ct.c_char_p,
    'wchar_t *': ct.c_wchar_p,

    'const char *': ct.c_char_p,
    'const wchar_t *': ct.c_wchar_p,

    'void *': ct.c_void_p,

    'PyObject *': ct.py_object,

    'void': None,
    'Box': Box,
}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()


storage = {}


@ct.CFUNCTYPE(None, ct.c_void_p)
def deleter(ptr):
    # eprint('py del', hex(ptr))
    del storage[int(ptr)]


def parse(types, typeid, top=None):
    """
    >>> types = {
    ... 0: ('double',),
    ... 1: ('int',),
    ... 2: ('unsigned long',),
    ... 3: ('bool',),
    ... 4: ('char *',),
    ... 5: ('POINTER', 0),
    ... 6: ('void *',),
    ... 7: ('POINTER', 1),
    ... 8: ('CFUNCTYPE', 3, 1),
    ... 9: ('CFUNCTYPE', 0, 0),
    ... 10: ('CFUNCTYPE', 1, 2, 3),
    ... }
    >>> ns = {
    ...    'POINTER': ct.POINTER,
    ...    'CFUNCTYPE': ct.CFUNCTYPE,
    ... }

    >>> tid = 1
    >>> parse(types, tid)
    'int'
    >>> ct.c_int == parse_ctype(ns, types, tid)
    True

    >>> tid = 9
    >>> parse(types, tid)
    'CFUNCTYPE(double,double)'
    >>> ct.CFUNCTYPE(ct.c_double, ct.c_double) == parse_ctype(ns, types, tid)
    True

    >>> tid = 5
    >>> parse(types, tid)
    'POINTER(double)'
    >>> ct.POINTER(ct.c_double) == parse_ctype(ns, types, tid)
    True

    >>> tid = 8
    >>> parse(types, tid, top='PYFUNCTYPE')
    'PYFUNCTYPE(bool,int)'
    >>> ct.CFUNCTYPE(ct.c_bool, ct.c_int) == parse_ctype(ns, types, tid, top=ct.CFUNCTYPE)
    True
    """
    name, *args = types[typeid]
    if not args:
        # simple type
        return name

    # composite type
    if top:
        name = top
    return '{}({})'.format(name, ','.join([parse(types, _) for _ in args]))


def parse_ctype(ns, types, typeid, top=None):
    name, *args = types[typeid]

    # simple type
    if not args:
        return ctypemap[name]

    # composite type
    if top:
        COMP = top
    else:
        COMP = ns[name]
    return COMP(*[parse_ctype(ns, types, _) for _ in args])


def identity(arg):
    return arg


class Func(object):

    def __init__(self, func, box):
        self.func = func
        self.box = box

    def __call__(self, *args):
        return self.func(*args)

    def __del__(self):
        b = self.box
        b.deleter(b.ptr)


def struct_env(struct, locals_, globals_):
    """
    Create ctypes struct from locals_ and globals_.
    """

    def lookup(name):
        return locals_.get(name, globals_.get(name))

    return struct(*[ctype(lookup(name)) for name, ctype in struct._fields_])


def type_system():
    def print_types():
        print('types:')
        for _ in types:
            print('\t', _, parse(types, _))

    def print_funcs():
        print('callers:')
        for _ in callers:
            print('\t', _, parse(types, _), parse(types, callers[_].tid))
        print('callbacks:')
        for _ in callers:
            print('\t', _, parse(types, _), parse(types, callbacks[_]))

    def pre_init():
        pass

    def post_init():
        # print_types()
        # print_funcs()
        pass

    # initialize type counter to zero
    type_counter = 0
    types = {}

    def add_type(tid, name, args, nargs):
        if tid in types:
            return tid
        name = name.decode()
        args = [args[i] for i in range(nargs)]
        types[tid] = (name, *args)

        return tid

    callers = {}

    def add_caller(tid, box):
        callers[tid] = box

    callbacks = {}

    def add_callback(tid, cid):
        callbacks[tid] = cid

    def cat(tid):
        name, *args = types[tid]
        if not args:
            return 'SIMPLE'
        return types[tid][0]

    def unbox(box):
        return {
            'SIMPLE': unbox_simple,
            'POINTER': unbox_pointer,
            'CFUNCTYPE': unbox_cfunc,
            'CPPFUNCTYPE': unbox_cppfunc,
        }[cat(box.tid)](box)

    ns = {
        'POINTER': ct.POINTER,
        'CFUNCTYPE': ct.CFUNCTYPE,
        'CPPFUNCTYPE': lambda *args: Box,
    }

    def unbox_simple(box):
        ctype = parse_ctype(ns, types, box.tid)
        ptr = ct.cast(box.ptr, ct.POINTER(ctype))
        obj = ptr[0]
        box.deleter(box.ptr)
        return obj

    def unbox_pointer(box):
        ctype = parse_ctype(ns, types, box.tid)
        obj = ct.cast(box.ptr, ctype)
        return obj

    def unbox_cfunc(box):
        ctype = parse_ctype(ns, types, box.tid)
        obj = ct.cast(box.ptr, ctype)
        return obj

    def unbox_cppfunc(box):

        _, ret_tid, *args_tid = types[box.tid]
        _box_args = box_args(args_tid)
        _unbox_ret = unbox_ret(ret_tid)

        cbox = callers[box.tid]
        caller = unbox_cfunc(cbox)

        def f(*args):
            # box args
            args = _box_args(args)
            # do actual call
            ret = caller(box.ptr, *args)
            # unbox ret
            ret = _unbox_ret(ret)
            return ret

        return Func(f, box)

    def unbox_rets(tids):
        funcs = [unbox_ret(_) for _ in tids]
        return lambda args: [b(a) for b, a in zip(funcs, args)]

    def unbox_ret(tid):
        if cat(tid) == 'CPPFUNCTYPE':
            return unbox_cppfunc
        return identity

    def box_args(tids):
        funcs = [box_arg(_) for _ in tids]
        return lambda args: [b(a) for b, a in zip(funcs, args)]

    def box_arg(tid):
        if cat(tid) == 'CPPFUNCTYPE':
            return box_cppfunc(tid)
        return identity

    def box_cppfunc(tid):
        ctype = parse_ctype(ns, types, tid, top=ct.CFUNCTYPE)
        _, ret_tid, *args_tid = types[tid]

        _unbox_args = unbox_rets(args_tid)
        _box_ret = box_arg(ret_tid)

        def _(callback):
            if type(callback) is Func:
                return callback.box

            def f(*args):
                # unbox args
                args = _unbox_args(args)
                # do actual call
                ret = callback(*args)
                # box ret
                ret = _box_ret(ret)
                return ret

            b = Box(tid=callbacks[tid],
                    ptr=ct.cast(ctype(f), ct.c_void_p),
                    deleter=deleter)
            storage[int(b.ptr)] = b
            # eprint('py new', hex(b.ptr))

            return b

        return _

    def add_box(m, name, box):
        name = name.decode()
        setattr(m, name, unbox(box))

    def import_func(module, name, tid, box_out):
        module = module.decode()
        name = name.decode()
        m = importlib.import_module(module)
        f = getattr(m, name)
        box_out[0] = box_cppfunc(tid)(f)

    ts = struct_env(TypeSystem, locals(), globals())
    return ts


type_systems = {}


def typesystem_init(ts_out):
    if ts_out not in type_systems:
        ts = type_system()
        type_systems[ts_out] = ts

        pp = ct.cast(ts_out, ct.POINTER(ct.POINTER(TypeSystem)))
        pp[0] = ct.pointer(ts)

    return type_systems[ts_out]
