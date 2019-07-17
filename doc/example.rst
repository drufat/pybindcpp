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



