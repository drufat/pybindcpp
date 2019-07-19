# Copyright (C) 2010-2019 Dzhelil S. Rufat. All Rights Reserved.

__version__ = __import__('pkg_resources').get_distribution('pybindcpp').version


def get_include():
    """
    Get the include directory.
    """
    import pathlib
    return pathlib.Path(__file__).parent / 'include'
