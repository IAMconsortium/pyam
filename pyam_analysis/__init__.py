from pyam_analysis.core import *
from pyam_analysis.utils import *
from pyam_analysis.timeseries import *

# formatting for warnings


def custom_formatwarning(msg, category, filename, lineno, line=''):
    # ignore everything except the message
    return str(msg) + '\n'


# in Jupyter notebooks: disable autoscroll, activate warnings
try:
    get_ipython().run_cell_magic(u'javascript', u'',
                                 u'IPython.OutputArea.prototype._should_scroll = function(lines) { return false; }')
    warnings.simplefilter('default')
    warnings.formatwarning = custom_formatwarning
except Exception:
    pass
