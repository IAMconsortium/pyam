from importlib.metadata import version, PackageNotFoundError
import logging
from pathlib import Path
from setuptools_scm import get_version

from pyam.core import (
    IamDataFrame,
    categorize,
    check_aggregate,
    compare,
    concat,
    filter_by_meta,
    require_variable,
    read_datapackage,
    validate,
)
from pyam.statistics import Statistics
from pyam.iiasa import read_iiasa, lazy_read_iiasa  # noqa: F401
from pyam.datareader import read_worldbank  # noqa: F401
from pyam.unfccc import read_unfccc  # noqa: F401
from pyam.testing import assert_iamframe_equal  # noqa: F401
from pyam.run_control import run_control  # noqa: F401
from pyam.utils import IAMC_IDX  # noqa: F401

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
        # harmonize formatting of ixmp4 and pyam logging
        ixmp4_logger = logging.getLogger("ixmp4")
        ixmp4_logger.removeHandler(ixmp4_logger.handlers[0])

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s: %(message)s"))

        for _logger in [logger, ixmp4_logger]:
            _logger.addHandler(handler)

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
