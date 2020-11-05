import logging  # noqa: F401

from pyam.core import *
from pyam.utils import *
from pyam.statistics import *
from pyam.timeseries import *
from pyam.read_ixmp import *
from pyam.logging import *
from pyam.run_control import *
from pyam.iiasa import read_iiasa  # noqa: F401
from pyam.datareader import read_worldbank  # noqa: F401

from pyam.logging import defer_logging_config

logger = logging.getLogger(__name__)

# in Jupyter notebooks: disable autoscroll and set-up logging
try:
    from ipykernel.zmqshell import ZMQInteractiveShell
    from IPython import get_ipython

    shell = get_ipython()
    if isinstance(shell, ZMQInteractiveShell):
        shell.run_cell_magic(
            u'javascript', u'',
            u'IPython.OutputArea.prototype._should_scroll = '
            u'function(lines) { return false; }'
        )
        log_msg = (
            "Running in a notebook, "
            "setting up a basic logging config at level INFO"
        )
        defer_logging_config(
            logger, log_msg, level="INFO",
            format="%(name)s - %(levelname)s: %(message)s",
        )

except Exception:
    pass

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
