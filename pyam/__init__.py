import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from pyam.core import (
    IamDataFrame,
    compare,
    concat,
    filter_by_meta,
    read_datapackage,
)
from pyam.iiasa import lazy_read_iiasa, read_iiasa
from pyam.ixmp4 import read_ixmp4
from pyam.netcdf import read_netcdf
from pyam.run_control import run_control
from pyam.statistics import Statistics
from pyam.testing import assert_iamframe_equal
from pyam.unfccc import read_unfccc
from pyam.utils import IAMC_IDX
from pyam.worldbank import read_worldbank

here = Path(__file__).parent

try:
    __IPYTHON__  # type: ignore
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

_sys_has_ps1 = hasattr(sys, "ps1")

# Logging is only configured by default when used in an interactive environment.
# This follows the setup in ixmp4 and nomenclature.
if _in_ipython_session or _sys_has_ps1:
    with open(here / "logging.yaml") as file:
        logging.config.dictConfig(yaml.safe_load(file))

try:
    __version__ = version("pyam-iamc")
# the pyam package is distributed under different names on pypi and conda
except PackageNotFoundError:
    __version__ = version("pyam")
