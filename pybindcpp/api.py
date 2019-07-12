import ctypes as ct


class Box(ct.Structure):
    _fields_ = [
        ('tid', ct.c_size_t),
        ('ptr', ct.c_void_p),
        ('deleter', ct.CFUNCTYPE(None, ct.c_void_p)),
    ]


class TypeSystem(ct.Structure):
    _fields_ = [
        ('type_counter', ct.c_size_t),

        ('add_type', ct.CFUNCTYPE(ct.c_size_t, ct.c_size_t, ct.c_char_p, ct.POINTER(ct.c_size_t), ct.c_size_t)),
        ('add_caller', ct.CFUNCTYPE(None, ct.c_size_t, Box)),
        ('add_callback', ct.CFUNCTYPE(None, ct.c_size_t, ct.c_size_t)),

        ('pre_init', ct.CFUNCTYPE(None)),
        ('post_init', ct.CFUNCTYPE(None)),

        ('add_box', ct.CFUNCTYPE(None, ct.py_object, ct.c_char_p, Box)),

        ('import_func', ct.CFUNCTYPE(None, ct.c_char_p, ct.c_char_p, ct.c_size_t, ct.POINTER(Box))),
    ]
