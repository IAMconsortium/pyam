import numpy as np
import pytest

from pyam.testing import assert_iamframe_equal, assert_iamframe_not_equal


def test_equal_meta_nan_col(test_df_year):
    """Test that a meta-column with only np.nan is seen as equal"""
    # https://github.com/IAMconsortium/pyam/issues/515
    df = test_df_year.copy()
    df.set_meta(meta=np.nan, name="nan-column")  # add a column of np.nan's to `meta`

    assert_iamframe_equal(test_df_year, df)


def test_equal_meta_different(test_df_year):
    """Test with different meta values"""
    # https://github.com/IAMconsortium/pyam/issues/515
    df = test_df_year.copy()
    df.set_meta(
        meta="this will be ignored", name="meta_difference"
    )  # add a column of np.nan's to `meta`

    # assert that ignoring meta does not raise an error
    assert_iamframe_equal(test_df_year, df, check_meta=False)
    # assert that checking meta does raise an error
    with pytest.raises(AssertionError):
        assert_iamframe_equal(test_df_year, df, check_meta=True)


def test_df_not_equal(test_df_year, test_df_time):
    assert_iamframe_not_equal(test_df_year, test_df_time, check_meta=False)
    assert_iamframe_not_equal(test_df_year, test_df_time, check_meta=True)
