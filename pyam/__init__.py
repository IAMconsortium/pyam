from pyam.core import *
from pyam.utils import *
from pyam.statistics import *
from pyam.timeseries import *
from pyam.logger import *
from pyam.run_control import *
from pyam.iiasa import read_iiasa, read_iiasa_iamc15  # noqa: F401

# in Jupyter notebooks: disable autoscroll, activate warnings
try:
    get_ipython().run_cell_magic(u'javascript', u'',
                                 u'IPython.OutputArea.prototype._should_scroll = function(lines) { return false; }')
    warnings.simplefilter('default')

    def custom_formatwarning(msg, category, filename, lineno, line=''):
        # ignore everything except the message
        return str(msg) + '\n'
    warnings.formatwarning = custom_formatwarning
except Exception:
    pass
