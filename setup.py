# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import glob
import sys

import numpy
from setuptools import setup, Extension

include_dirs = [
    'include',
    'src',
    numpy.get_include(),
    '/usr/include/eigen3',
    '/usr/local/include/eigen3',
]

headers = glob.glob(
    'include/pybindcpp/*.h'
) + glob.glob(
    'include/pybindcpp/ctypes/*.h'
)

depends = [
              'setup.py',
          ] + headers

extra_compile_args = [
    '-std=c++14',
    '-Wall',
    '-g',
    '-O3',
    '-DNDEBUG',
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
        'pybindcpp.bindctypes',
        sources=[
            'pybindcpp/bindctypes.cpp',
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

    # Extension(
    #     'pybindcpp.ext.arrayfire',
    #     sources=[
    #         'pybindcpp/ext/arrayfire.cpp',
    #     ],
    #     depends=depends,
    #     include_dirs=include_dirs,
    #     extra_compile_args=extra_compile_args,
    #     language="c++",
    #     libraries=libraries + ['af'],
    # ),

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
    version='0.1.0',
    description='Python Bindings from C++',
    author='Dzhelil Rufat',
    author_email='drufat@caltech.edu',
    license='GNU GPLv3',
    url='http://dzhelil.info/pybindcpp',
    requires=[
        'numpy',
    ],
)
