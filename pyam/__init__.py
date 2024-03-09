from importlib.metadata import PackageNotFoundError, version

from pyam.core import (
    IamDataFrame,
    categorize,
    check_aggregate,
    compare,
    concat,
    filter_by_meta,
    read_datapackage,
    require_variable,
    validate,
)
from pyam.iiasa import lazy_read_iiasa, read_iiasa  # noqa: F401  # noqa: F401
from pyam.ixmp4 import read_ixmp4  # noqa: F401
from pyam.logging import configure_logging  # noqa: F401
from pyam.run_control import run_control  # noqa: F401
from pyam.statistics import Statistics  # noqa: F401
from pyam.testing import assert_iamframe_equal  # noqa: F401
from pyam.unfccc import read_unfccc  # noqa: F401
from pyam.utils import IAMC_IDX  # noqa: F401
from pyam.worldbank import read_worldbank  # noqa: F401

try:
    __version__ = version("pyam-iamc")
# the pyam package is distributed under different names on pypi and conda
except PackageNotFoundError:
    __version__ = version("pyam")

# Set up logging consistent with the ixmp4 "production" logging configuration
configure_logging()
