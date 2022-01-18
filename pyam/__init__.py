import logging
from pathlib import Path
from setuptools_scm import get_version

# use standard library for Python >=3.8
try:
    from importlib.metadata import version, PackageNotFoundError
# use dedicated package for Python 3.7
except ModuleNotFoundError:
    from importlib_metadata import version, PackageNotFoundError

from pyam.core import *
from pyam.utils import *
from pyam.statistics import *
from pyam.timeseries import *
from pyam.read_ixmp import *
from pyam.logging import *
from pyam.run_control import *
from pyam.iiasa import read_iiasa
from pyam.datareader import read_worldbank
from pyam.unfccc import read_unfccc
from pyam.testing import assert_iamframe_equal

from pyam.logging import defer_logging_config

logger = logging.getLogger(__name__)

# get version number either from git (preferred) or metadata
try:
    __version__ = get_version(root=Path(__file__).parents[1])
except LookupError:
    try:
        __version__ = version("pyam-iamc")
    # the pyam package is distributed under different names on pypi and conda
    except PackageNotFoundError:
        __version__ = version("pyam")

# special handling in Jupyter notebooks
try:
    from ipykernel.zmqshell import ZMQInteractiveShell
    from IPython import get_ipython

    shell = get_ipython()
    if isinstance(shell, ZMQInteractiveShell):

        # set up basic logging if running in a notebook
        log_msg = "Running in a notebook, setting up a basic logging at level INFO"

        defer_logging_config(
            logger,
            log_msg,
            level="INFO",
            format="%(name)s - %(levelname)s: %(message)s",
        )

        # deactivate in-cell scrolling in a Jupyter notebook
        shell.run_cell_magic(
            "javascript",
            "",
            "if (typeof IPython !== 'undefined') "
            "{ IPython.OutputArea.prototype._should_scroll = function(lines)"
            "{ return false; }}",
        )

except Exception:
    pass
