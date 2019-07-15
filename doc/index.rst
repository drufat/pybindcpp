PyBindCpp
=========

|Build Status| |Version Status| |Downloads|

.. |Build Status| image:: https://travis-ci.org/drufat/pybindcpp.png
   :target: https://travis-ci.org/drufat/pybindcpp
.. |Version Status| image:: https://img.shields.io/pypi/v/pybindcpp.svg
   :target: https://pypi.python.org/pypi/pybindcpp/
.. |Downloads| image:: https://img.shields.io/pypi/dm/pybindcpp.svg
   :target: https://pypi.python.org/pypi/pybindcpp/

*PyBindCpp* is a Python module that we have developed in order to make it easy to interface Python with C++. In what follows we describe the approach taken by PyBindCpp, and compare it to other similar libraries.

Install
--------

The PyBindCpp module can be easily installed from the terminal using the command

.. code-block:: bash

    $ pip install pybindcpp

The source code is available on Github, and can be downloaded using the command

.. code-block:: bash

    $ git clone https://github.com/drufat/pybindcpp.git

Introduction
-------------

Python has been widely successful as a programming language for scientific computations primarily due to two factors. First of all, as a high level language it is highly readable and expressive, which allows for rapid development. Second of all, because of its open source nature and the access to its internal CPython [#f1]_ *application programming interface* (API) it is possible to interface Python with the vast collection of existing scientific libraries without having to reimplement them from the ground up. However, despite these advantages, Python has one important drawback. Although adequate for most applications where speed is not a critical requirement, because of its highly dynamic nature, Python tends to be slow, especially when dealing with tight for-loops, and it is not suitable for computations where such low-level looping constructs are necessary.

To ameliorate this problem, a number of approaches have been proposed for a hybrid programming model where most of the high level logic is written in Python, and the time critical parts of the application, which usually tend to be a small fraction of the code base, are reimplemented in a low level language, significantly speeding up the application running time.

.. [#f1] The reference implementation of the Python programming language is done in C and is known as CPython. It can be downloaded from  `python.org <http://python.org>`_.

Existing Libraries
--------------------

Among the most popular approaches is the one taken by Cython :cite:`behnel2010cython`. In fact, Cython is not merely a library, but an entire programming language, which is a strict superset of the Python language. This means that all Python programs are valid Cython programs, but the converse is not true. Cython adds type annotations to the Python language, where each variable can be optionally assigned a static type. This extra information allows for the source code to be translated directly into C, and compiled and linked against Python, and then imported as a module. While the approach taken by Cython is quite powerful, it does have its drawbacks. First, it seems to be mainly geared towards speeding up existing Python applications by adding type annotations to the critical parts, and not towards porting existing C/C++ libraries. Interfacing with external libraries is quite cumbersome as one has to copy all of the header file declarations into a form that can be parsed by Cython. This violates the "don't repeat yourself" (DRY) principle where every piece of knowledge must have a single unambiguous representation within the system. In the case of Cython, however, declarations must exist both in the C/C++ header files as well as within Cython files. Additionally, just to bridge Python and C/C++ one has to learn a completely new third programming language with its new syntax and semantics.

Another approach for speeding up Python is to write the low level glue code manually in C/C++. This requires intimate knowledge of the CPython API as well as the addition of a large amount of boilerplate code for manual data type conversion and memory management. While this gives a lot of control to the programmer, it also places limitations on productivity due to the time consuming nature of programming at such a low level. Additionally, this approach intimately depends on the internals of the original Python implementation, and is not portable to other implementations of the same programming language such as PyPy, Jython, or IronPython.

A third approach is to use the Ctypes module that is included by default in the Python interpreter. Ctypes is a foreign function library for Python that provides C compatible data types that can be used to call functions directly from compiled DLL or shared library files. Ctypes works at the *application binary interface* (ABI) level rather than the *application programming interface* (API) level. One still needs to provide descriptions in Python of the exact signatures of the functions that are being exported. In fact one has to be quite careful, because at the ABI level if there is any mismatch, one can get quite nasty segmentation faults and data corruption errors, whereas the worst that can happen at the API level is for the program not to compile. Again, like the approach taken by Cython, this violates the DRY principle since declarations must be copied into Python.

Design
-------------

PyBindCpp is our own proposed solution to the problem which leverages Ctypes and C++11 in order to provide highly portable and seamless interoperability between C/C++ and Python. Most of the logic is implemented in Python via the Ctypes module which enables one to call functions from shared library files and do the appropriate type conversions. The key insight is to use C++11 for type deductions, and to feed that information back to Python in order to generate the appropriate Ctypes wrappers. In order to deduce the types of the variables and the signatures of the functions, we have relied heavily on recent features added to C++ such as template metaprogramming, type traits, parameter packs, and the auto specifier. Because of this, the minimum version of C++ necessary to compile PyBindCpp is C++11. Without these features, the only way to obtain the type and signature information of functions would have been to parse C++, which is a very difficult problem given the size and complexity of the language.

In designing PyBindCpp, we have been inspired by ChaiScrit :cite:`turnerchaiscript`, and we have borrowed heavily from its design. ChaiScript is a scripting language that targets C++ directly and takes advantage of its many modern features. The typesystem of ChaiScript is identical to the C++ type system with a few minor additions. Unlike ChaiScript, PyBindCpp is not a new language, but a module of Python, and it provides its own type conversion layer between C++ data types and Python data types. Like ChaiScript, PyBindCCpp is \emph{headers only} and does not require any external tools or libraries to compile, just a modern C++ compiler.

PyBind11 :cite:`pybind11` is another library that is very similar in its design goals to PyBindCpp. However, there are some significant differences. Most of the programming logic in PyBind11 is coded in C++ and the library relies heavily on the CPython API making it difficult to port to other implementations of the Python language. PyBindCpp, on the other hand, uses Ctypes which is provided by most Python interpreters, and can therefore be easily ported. A design goal of PyBindCpp is to minimize reliance on the API to a bare minimum, perhaps even eliminate it completely, and instead leverage the ABI via Ctypes. Additionally, PyBindCpp is coded mainly in Python, and C++ is minimally used for type deduction. Many of the Python functions are made available to the C++ level via Ctypes callbacks. This saves us the effort of having to code them in C++, especially for code that is rarely executed (e.g., only once at import time) where speed is not that critical. This small speed penalty is more than offset by the simplicity of the code when written in Python.

Example
-------

Next we present a minimal working example of PyBindCpp usage. Consider the following C++ code which includes variables and functions that we wish to export to Python:


.. literalinclude:: example.cpp
   :language: cpp
   :lines: 3, 5-10


To port the above code, we merely need to write a single function before compiling into a shared library that can be directly imported from Python. We must define a function which takes as its input a reference to an :code:`module` object, and uses that to register the entities that need to exported. We can use the :code:`add` attribute to register both simple types and functions. At the end we must add the :code:`PYBINDCPP_INIT` preprocessor macro which takes two arguments - the name of the module that we are creating and the name of initialization function that we just defined. This is necessary in order for Python to find the correct entry point when loading the extension module.

.. literalinclude:: example.cpp
   :language: cpp
   :lines: 4, 11-

After compiling the code above into a shared module, we can use the script below to import it from Python and run some tests to ensure that everything works as expected.

.. literalinclude:: example_test.py
   :language: py
   :lines: 3-

References
------------

.. bibliography:: references.bib
