Introduction
-------------

Python has been widely successful as a programming language for scientific computations primarily due to two factors. First of all, as a high level language it is highly readable and expressive, which allows for rapid development. Second of all, because of its open source nature and the access to its internal CPython [#f1]_ *application programming interface* (API) it is possible to interface Python with the vast collection of existing scientific libraries without having to reimplement them from the ground up. However, despite these advantages, Python has one important drawback. Although adequate for most applications where speed is not a critical requirement, because of its highly dynamic nature, Python tends to be slow, especially when dealing with tight for-loops, and it is not suitable for computations where such low-level looping constructs are necessary.

.. [#f1] The reference implementation of the Python programming language is done in C and is known as CPython. It can be downloaded from  `python.org <http://python.org>`_.

To ameliorate this problem, a number of approaches have been proposed for a hybrid programming model where most of the high level logic is written in Python, and the time critical parts of the application, which usually tend to be a small fraction of the code base, are reimplemented in a low level language, significantly speeding up the application running time.
