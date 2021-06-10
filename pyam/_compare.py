import numpy as np
import pandas as pd


def _compare(
    left, right, left_label="left", right_label="right", drop_close=True, **kwargs
):
    """Internal implementation of comparison of IamDataFrames or pd.Series"""

    def as_series(s):
        return s if isinstance(s, pd.Series) else s._data

    ret = pd.merge(
        left=as_series(left).rename(index=left_label),
        right=as_series(right).rename(index=right_label),
        how="outer",
        left_index=True,
        right_index=True,
    )
    if drop_close:
        ret = ret[~np.isclose(ret[left_label], ret[right_label], **kwargs)]
    return ret
