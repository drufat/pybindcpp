# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import glob
import sys
import os

import numpy
from setuptools import setup, Extension

# Read version number
with open("pybindcpp/version.py") as f:
    exec(f.read())

include_dirs = [
    'include',
    '/usr/include/eigen3',
    '/usr/local/include/eigen3',
    numpy.get_include(),
]

headers = glob.glob(
    'include/*.h'
) + glob.glob(
    'include/pybindcpp/*.h'
)

depends = [
              'setup.py',
          ] + headers

extra_compile_args = [
    '-std=c++14',
]

libraries = []

OPENMP = False
if OPENMP:
    extra_compile_args += ['-fopenmp']
    if 'darwin' not in sys.platform:
        libraries += ['gomp']

ext_modules = [

    Extension(
        'pybindcpp.dispatch',
        sources=[
            'pybindcpp/dispatch.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'pybindcpp.ext.bindctypes',
        sources=[
            'pybindcpp/ext/bindctypes.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'pybindcpp.ext.simple',
        sources=[
            'pybindcpp/ext/simple.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'pybindcpp.ext.native',
        sources=[
            'pybindcpp/ext/native.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'pybindcpp.ext.native_cpp',
        sources=[
            'pybindcpp/ext/native.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args + ['-DNATIVE_CPP'],
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'pybindcpp.ext.numpy',
        sources=[
            'pybindcpp/ext/numpy.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'pybindcpp.ext.eigen',
        sources=[
            'pybindcpp/ext/eigen.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'pybindcpp.ext.fftw',
        sources=[
            'pybindcpp/ext/fftw.cpp',
        ],
        depends=depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries + ['fftw3'],
    ),
]

if 'AF_PATH' in os.environ:
    ext_modules += [
        Extension(
            'pybindcpp.ext.arrayfire',
            sources=[
                'pybindcpp/ext/arrayfire.cpp',
            ],
            depends=depends,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language="c++",
            libraries=libraries + ['af'],
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
    ext_modules=ext_modules,
    version=__version__,
    description='Python Bindings from C++',
    author='Dzhelil Rufat',
    author_email='drufat@caltech.edu',
    license='GNU GPLv3',
    url='http://dzhelil.info/pybindcpp',
)
