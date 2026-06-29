r"""

Collection of functions for altering images based on structural properties
##########################################################################

This module contains a variety of functions for altering images based on
the structural characteristics, such as pore sizes.  A definition of a
*filter* is a function that returns an image the same shape as the original
image, but with altered values.

"""

from ._fftmorphology import *
from ._funcs import *
from ._fill_and_find import *
from ._nlmeans import *
from ._size_seq_satn import *
from ._snows import *
from ._transforms import *
from ._displacement import *
from ._morphology import *
from ._lt_methods import *
