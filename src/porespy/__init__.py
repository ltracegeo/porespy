r"""
#######
PoreSpy
#######

PoreSpy is a package for performing image analysis on volumetric images of
porous materials.

PoreSpy consists of several key modules. Each module is consisted of
several functions. Here, you'll find a comprehensive documentation of the
modules, occasionally with basic embedded examples on how to use them.

"""

import logging
from pathlib import Path

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    format=FORMAT, datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("porespy")


from . import tools
from . import filters
from . import metrics
from . import networks
from . import generators
from . import simulations
from . import visualization
from . import io
from .visualization import imshow

try:
    import tomllib as _toml
except ModuleNotFoundError:
    import tomli as _toml
import importlib.metadata as _metadata
import numpy as _np

_np.seterr(divide="ignore", invalid="ignore")
settings = tools.Settings()

_pyproject = Path(__file__).parents[2] / "pyproject.toml"

if _pyproject.exists():
    with open(_pyproject, "rb") as f:
        data = _toml.load(f)
        __version__ = data["project"]["version"]
        logger.debug("Loaded version from pyproject.toml")
else:
    __version__ = _metadata.version(__package__ or __name__)
    logger.debug("Loaded version from importlib.metadata")
