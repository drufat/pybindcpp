import ctypes as ct

from pybindcpp import bindctypes

bindctypes_so = ct.CDLL(bindctypes.__file__)


def fun(name, ret_t, *args_t):
    return ct.CFUNCTYPE(ret_t, *args_t)((name, bindctypes_so))


c_void_fn = ct.CFUNCTYPE(None)

add = fun("add", ct.c_int, ct.c_int, ct.c_int)
create_string = fun("create_string", ct.c_int, ct.c_char, ct.c_int, ct.c_char_p)
register_function = fun("register_function", ct.py_object, ct.POINTER(c_void_fn), ct.POINTER(ct.c_char))


def register(func, signature):
    return register_function(
        ct.cast(func, ct.POINTER(c_void_fn)),
        signature
    )


def test_fun():
    '''
    >>> add(1, 1)
    2

    >>> type(bindctypes_so.register_function)
    <class 'ctypes.CDLL.__init__.<locals>._FuncPtr'>

    >>> size = 20
    >>> buf = ct.create_string_buffer(size+1)
    >>> create_string(b'a', size, buf)
    0
    >>> buf.raw
    b'aaaaaaaaaaaaaaaaaaaa\\x00'
    >>> buf[0] = b'A'
    >>> buf.raw
    b'Aaaaaaaaaaaaaaaaaaaa\\x00'
    >>> ct.cast(ct.cast(buf, ct.c_void_p), ct.c_char_p).value
    b'Aaaaaaaaaaaaaaaaaaaa'
    '''
    pass


def test_register():
    register(add, b"int, int, int")
