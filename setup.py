# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import glob

import numpy
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
    *glob.glob('pybindcpp/include/*.h'),
    *glob.glob('pybindcpp/include/pybindcpp/*.h'),
]

depends = ['setup.py', *headers]
extra_compile_args = ['-std=c++14']
libraries = []

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
        'pybindcpp.ext.example',
        sources=[
            'pybindcpp/ext/example.cpp',
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
