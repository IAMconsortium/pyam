import logging
import pandas as pd

from pyam.utils import make_index, s

logger = logging.getLogger(__name__)


def exclude_on_fail(df, index):
    """Assign a selection of scenarios as `exclude: True`"""

    if not isinstance(index, pd.MultiIndex):
        index = make_index(index, cols=df.index.names)

    df.exclude[index] = True
    n = len(index)
    logger.info(
        f"{n} scenario{s(n)} failed validation and will be set as `exclude=True`."
    )
