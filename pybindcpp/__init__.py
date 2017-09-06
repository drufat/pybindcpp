import pathlib


def get_include():
    """
    Get the include directory.
    """
    return pathlib.Path(__file__).parent / 'include'
