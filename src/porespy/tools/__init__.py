r"""

Collection of helper functions for manipulating images
######################################################

This module contains a variety of functions for manipulating images in
ways that do NOT return a modified version of the original image.

"""


from ._utils import *
from ._morphology import *
from ._funcs import *
from ._sphere_insertions import *
from ._marching_cubes import *
from ._marching_squares import *


def _get_version():
    from porespy.__version__ import __version__ as ver
    suffix = ".dev0"
    if ver.endswith(suffix):
        ver = ver[:-len(suffix)]
    return ver

settings = Settings()
