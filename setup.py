# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from glob import glob

import numpy

print(numpy.__version__)
from setuptools import setup, Extension

# Read version number
ns = {}
with open("pybindcpp/version.py") as f:
    exec(f.read(), ns)

include_dirs = [
    'pybindcpp/include',
    '/usr/include/eigen3',
    '/usr/local/include/eigen3',
    numpy.get_include(),
]

headers = [
    *glob('pybindcpp/include/*.h'),
    *glob('pybindcpp/include/capi/*.h'),
    *glob('pybindcpp/include/pb/*.h'),
]

depends = ['setup.py', *headers]
extra_compile_args = [
    '-std=c++14',
    '-g',
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
]
libraries = []

ext_modules = [

    Extension(
        'pybindcpp.core.dispatch',
        sources=[
            'pybindcpp/core/dispatch.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        f'pybindcpp.ext.capi',
        sources=[
            f'pybindcpp/ext/capi.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'pybindcpp.ext.np.np',
        sources=[
            'pybindcpp/ext/np/np.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'pybindcpp.ext.np.eigen',
        sources=[
            'pybindcpp/ext/np/eigen.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'pybindcpp.ext.np.fftw',
        sources=[
            'pybindcpp/ext/np/fftw.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries + ['fftw3'],
    ),

    Extension(
        'pybindcpp.ext.pb',
        sources=[
            'pybindcpp/ext/pb.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'pybindcpp.core.ufunc',
        sources=[
            'pybindcpp/core/ufunc.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'pybindcpp.ext.ufunc',
        sources=[
            'pybindcpp/ext/ufunc.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

]

setup(
    name='pybindcpp',
    packages=[
        'pybindcpp'
    ],
    package_dir={
        'pybindcpp': 'pybindcpp'
    },
    package_data={
        'pybindcpp': ['include/*.h', 'include/pybindcpp/*.h'],
    },
    ext_modules=ext_modules,
    version=ns['__version__'],
    description='Python Bindings from C++',
    author='Dzhelil Rufat',
    author_email='d@rufat.be',
    license='GPLv3',
)
