import logging
from pathlib import Path
from setuptools_scm import get_version
from importlib.metadata import version

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
    __version__ = get_version(Path(__file__).parent.parent)
except LookupError:
    try:
        __version__ = version("pyam-iamc")
    except:  # the package is distributed under different names on pypi and conda
        __version__ = version("pyam")

# in Jupyter notebooks: disable autoscroll and set-up logging
try:
    from ipykernel.zmqshell import ZMQInteractiveShell
    from IPython import get_ipython

    shell = get_ipython()
    if isinstance(shell, ZMQInteractiveShell):
        shell.run_cell_magic(
            u"javascript",
            u"",
            u"IPython.OutputArea.prototype._should_scroll = "
            u"function(lines) { return false; }",
        )
        log_msg = "Running in a notebook, setting up a basic logging at level INFO"

        defer_logging_config(
            logger,
            log_msg,
            level="INFO",
            format="%(name)s - %(levelname)s: %(message)s",
        )

except Exception:
    pass
