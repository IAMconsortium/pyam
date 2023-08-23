from contextlib import contextmanager
from logging import *

import pandas as pd
import warnings


logger = getLogger(__name__)


@contextmanager
def adjust_log_level(logger="pyam", level="ERROR"):
    """Context manager to change log level"""
    if isinstance(logger, str):
        logger = getLogger(logger)
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    yield
    logger.setLevel(old_level)


def deprecation_warning(msg, item="This method", stacklevel=3):
    """Write deprecation warning to log"""
    warnings.simplefilter("always", DeprecationWarning)
    message = f"{item} is deprecated and will be removed in future versions. {msg}"
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def raise_data_error(msg, data):
    """Utils function to format error message from data formatting"""
    if isinstance(data, pd.MultiIndex):
        data = data.to_frame(index=False)
    data = data.drop_duplicates()
    msg = f"{msg}:\n{data.head()}" + ("\n..." if len(data) > 5 else "")
    raise ValueError(msg)
