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

OPENMP = True
if OPENMP:
    extra_compile_args += ['-fopenmp']
    if 'darwin' not in sys.platform:
        libraries += ['gomp']

ext_modules = [

    Extension(
        'pybindcpp.dispatch',
        sources=[
            'src/modules/ext/dispatch.cpp',
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
            'src/modules/ext/bindctypes.cpp',
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
            'src/modules/ext/simple.cpp',
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
            'src/modules/ext/native.cpp',
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
            'src/modules/ext/numpy.cpp',
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
            'src/modules/ext/eigen.cpp',
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
            'src/modules/ext/fftw.cpp',
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
    #         'src/modules/ext/arrayfire.cpp',
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
