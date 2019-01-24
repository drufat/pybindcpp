import ctypes as ct
import importlib


class Box(ct.Structure):
    _fields_ = [
        ('tid', ct.c_size_t),
        ('ptr', ct.c_void_p),
        ('deleter', ct.CFUNCTYPE(None, ct.c_void_p)),
    ]


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

    'void': None,
    'Box': Box,
}


class API(ct.Structure):
    _fields_ = [
        ('print_', ct.CFUNCTYPE(None, ct.c_char_p)),
        ('error', ct.CFUNCTYPE(None)),

        ('pre_init', ct.CFUNCTYPE(None)),
        ('post_init', ct.CFUNCTYPE(None)),

        ('add_type', ct.CFUNCTYPE(
            ct.c_size_t,
            ct.c_size_t, ct.POINTER(ct.c_size_t), ct.c_size_t, ct.c_char_p)),
        ('add', ct.CFUNCTYPE(None, ct.c_char_p, Box)),
        ('add_caller', ct.CFUNCTYPE(None, ct.c_size_t, Box)),
        ('add_callback', ct.CFUNCTYPE(None, ct.c_size_t, ct.c_size_t)),

        ('commit_add', ct.CFUNCTYPE(None, )),

        ('get_cfunction', ct.CFUNCTYPE(ct.c_void_p, ct.c_char_p, ct.c_char_p)),
        ('get_pyfunction', ct.CFUNCTYPE(ct.c_void_p, ct.c_char_p, ct.c_char_p, ct.c_char_p)),
        ('func_c', ct.CFUNCTYPE(ct.py_object, ct.c_void_p, ct.c_char_p)),
        ('func_std', ct.CFUNCTYPE(ct.py_object, ct.py_object, ct.c_void_p)),
        ('vararg', ct.CFUNCTYPE(ct.py_object, ct.py_object)),
    ]


storage = {}


@ct.CFUNCTYPE(None, ct.c_void_p)
def deleter(ptr):
    # print('py del', hex(ptr))
    del storage[int(ptr)]


def is_func(f):
    return hasattr(f, 'argtypes') and hasattr(f, 'restype')


def print_(name):
    print(name.decode())


def error():
    raise RuntimeError('RuntimeError')


def get_type(expr):
    expr = expr.decode()
    t = eval(expr, ct.__dict__)
    return ct.PYFUNCTYPE(*t)


def get_cfunction(module, attr):
    module = module.decode()
    attr = attr.decode()

    mod = importlib.import_module(module)
    cfunc = getattr(mod, attr)
    addr = ct.addressof(cfunc)
    return addr


def get_pyfunction(module, attr, cfunctype):
    module = module.decode()
    attr = attr.decode()
    cfunc_type = get_type(cfunctype)

    mod = importlib.import_module(module)
    func = getattr(mod, attr)
    cfunc = cfunc_type(func)

    addr = ct.addressof(cfunc)

    # To ensure addr does not become dangling.
    storage[addr] = cfunc

    return addr


def func_c(func, func_type):
    p = ct.cast(func, ct.POINTER(ct.c_void_p))
    t = get_type(func_type)
    f = ct.cast(p[0], t)
    return f


def func_std(func_call, func_ptr):
    def func(*args):
        return func_call(func_ptr, *args)

    return func


def vararg(f):
    def v(*args):
        return f(None, args)

    return v


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
    if not args:
        # simple type
        return ctypemap[name]

    # composite type
    if top:
        COMP = top
    else:
        COMP = ns[name]
    return COMP(*[parse_ctype(ns, types, _) for _ in args])


def identity(arg):
    return arg


class Func:

    def __init__(self, func, box):
        self.func = func
        self.box = box

    def __call__(self, *args):
        return self.func(*args)

    def __del__(self):
        box = self.box
        # print("cc del", hex(box.ptr))
        box.deleter(box.ptr)


def module_api(m):
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

    types = {}

    def add_type(tid, args, nargs, name):
        if tid in types:
            return tid
        name = name.decode()
        args = [args[i] for i in range(nargs)]
        types[tid] = (name, *args)

        return tid

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
        # print('cc del', hex(box.ptr))
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
        ctype = parse_ctype(ns, types, tid, top=ct.PYFUNCTYPE)
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
            # print('py new', hex(b.ptr))
            return b

        return _

    callers = {}

    def add_caller(tid, box):
        callers[tid] = box

    callbacks = {}

    def add_callback(tid, cid):
        callbacks[tid] = cid

    to_add = []

    def add(name, box):
        to_add.append((
            name.decode(),
            lambda: unbox(box)
        ))

    def commit_add():
        for name, obj in to_add:
            setattr(m, name, obj())

    locals_, globals_ = locals(), globals()

    def env(name):
        return locals_.get(name, globals_.get(name))

    return API(*[ctype(env(name)) for name, ctype in API._fields_])


def module_init(mod_addr, api_ret):
    pm = ct.cast(mod_addr, ct.POINTER(ct.py_object))
    m = pm[0]

    api = module_api(m)

    api_addr = ct.addressof(api)
    storage[api_addr] = api

    pp = ct.cast(api_ret, ct.POINTER(ct.POINTER(API)))
    pp[0] = ct.pointer(api)
