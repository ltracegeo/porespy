r"""

Collection of functions for extracting quantitative information from images
###########################################################################

This submodule contains functions for determining key metrics about an
image. Typically these are applied to an image after applying a filter,
but a few functions can be applied directly to the binary image.

"""

from ._regionprops import *
from ._funcs import *
from ._meshtools import *
from ._rev import *
