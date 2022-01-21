import numpy as np
from pyam.testing import assert_iamframe_equal


def test_equal_meta_nan_col(test_df_year):
    """Test that a meta-column with only np.nan is seen as equal"""
    # https://github.com/IAMconsortium/pyam/issues/515
    df = test_df_year.copy()
    df.set_meta("foo", np.nan)

    assert_iamframe_equal(test_df_year, df)
