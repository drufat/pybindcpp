import sysconfig
import os
import sys

suffix = sysconfig.get_config_var('EXT_SUFFIX')


def move(name, bin, src):
    src_name, src_ext = os.path.splitext(src)
    assert (src_ext == '.cpp')
    src_dir = os.path.dirname(src_name)
    dest = os.path.join(src_dir, name) + suffix
    try:
        os.remove(dest)
    except FileNotFoundError:
        pass
    os.symlink(bin, dest)


if __name__ == '__main__':
    _, name, bin, src = sys.argv
    move(name, bin, src)
