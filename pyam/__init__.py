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
from pyam.iiasa import lazy_read_iiasa, read_iiasa
from pyam.ixmp4 import read_ixmp4
from pyam.logging import configure_logging
from pyam.netcdf import read_netcdf
from pyam.run_control import run_control
from pyam.statistics import Statistics
from pyam.testing import assert_iamframe_equal
from pyam.unfccc import read_unfccc
from pyam.utils import IAMC_IDX
from pyam.worldbank import read_worldbank

try:
    __version__ = version("pyam-iamc")
# the pyam package is distributed under different names on pypi and conda
except PackageNotFoundError:
    __version__ = version("pyam")

# Set up logging consistent with the ixmp4 "production" logging configuration
configure_logging()
