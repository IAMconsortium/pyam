import logging

from pyam.core import *
from pyam.utils import *
from pyam.statistics import *
from pyam.timeseries import *
from pyam.read_ixmp import *
from pyam.logging import *
from pyam.run_control import *
from pyam.iiasa import read_iiasa  # noqa: F401

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# in Jupyter notebooks: disable autoscroll and set logger to info
try:
    get_ipython().run_cell_magic(u'javascript', u'',
                                 u'IPython.OutputArea.prototype._should_scroll = function(lines) { return false; }')

    logger.setLevel(logging.INFO)

    stderr_info_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
    stderr_info_handler.setFormatter(formatter)
    logger.addHandler(stderr_info_handler)

    log_msg = (
        "Running in a notebook, setting `{}` logging level to `logging.INFO` "
        "and adding stderr handler".format(__name__)
    )
    logger.info(log_msg)

except Exception:
    pass

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
