import datetime

import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt

from pyam import IamDataFrame
from pyam.str import is_str
from pyam.testing import assert_iamframe_equal
from pyam.utils import IAMC_IDX


def test_interpolate(test_pd_df):
    _df = test_pd_df.copy()
    _df["foo"] = ["bar", "baz", 2]  # add extra_col (check for #351)
    df = IamDataFrame(_df)
    obs = df.interpolate(2007, inplace=False).filter(year=2007)._data.values
    npt.assert_allclose(obs, [3, 1.5, 4])

    # redo the interpolation and check that no duplicates are added
    df.interpolate(2007, inplace=False)
    assert not df._data.index.duplicated().any()

    # assert that extra_col does not have nan's (check for #351)
    assert all([True if is_str(i) else ~np.isnan(i) for i in df.foo])


def test_interpolate_time_exists(test_df_year):
    obs = test_df_year.interpolate(2005, inplace=False).filter(year=2005)._data.values
    npt.assert_allclose(obs, [1.0, 0.5, 2.0])


def test_interpolate_with_list(test_df_year):
    lst = [2007, 2008]
    obs = test_df_year.interpolate(lst, inplace=False).filter(year=lst)._data.values
    npt.assert_allclose(obs, [3, 4, 1.5, 2, 4, 5])


def test_interpolate_with_numpy_list(test_df_year):
    test_df_year.interpolate(np.r_[2007 : 2008 + 1], inplace=True)
    obs = test_df_year.filter(year=[2007, 2008])._data.values
    npt.assert_allclose(obs, [3, 4, 1.5, 2, 4, 5])


def test_interpolate_full_example():
    cols = ["model_a", "scen_a", "World"]
    df = IamDataFrame(
        pd.DataFrame(
            [
                cols + ["all", "EJ/yr", 0, 1, 6.0, 10],
                cols + ["last", "EJ/yr", 0, 0.5, 3, np.nan],
                cols + ["first", "EJ/yr", 0, np.nan, 2, 7],
                cols + ["middle", "EJ/yr", 0, 1, np.nan, 7],
                cols + ["first two", "EJ/yr", 0, np.nan, np.nan, 7],
                cols + ["last two", "EJ/yr", 0, 1, np.nan, np.nan],
            ],
            columns=IAMC_IDX + [2000, 2005, 2010, 2017],
        )
    )
    exp = IamDataFrame(
        pd.DataFrame(
            [
                cols + ["all", "EJ/yr", 0, 1, 6.0, 7.142857, 10],
                cols + ["last", "EJ/yr", 0, 0.5, 3, np.nan, np.nan],
                cols + ["first", "EJ/yr", 0, 1.0, 2, 3.428571, 7],
                cols + ["middle", "EJ/yr", 0, 1, np.nan, 4.5, 7],
                cols + ["first two", "EJ/yr", 0, 2.058824, np.nan, 4.941176, 7],
                cols + ["last two", "EJ/yr", 0, 1, np.nan, np.nan, np.nan],
            ],
            columns=IAMC_IDX + [2000, 2005, 2010, 2012, 2017],
        )
    )
    assert_iamframe_equal(df.interpolate([2005, 2012], inplace=False), exp)


def test_interpolate_extra_cols():
    # check that interpolation with non-matching extra_cols has no effect
    # (#351)
    EXTRA_COL_DF = pd.DataFrame(
        [
            ["foo", 2005, 1],
            ["foo", 2010, 2],
            ["bar", 2005, 2],
            ["bar", 2010, 3],
        ],
        columns=["extra_col", "year", "value"],
    )
    df = IamDataFrame(
        EXTRA_COL_DF,
        model="model_a",
        scenario="scen_a",
        region="World",
        variable="Primary Energy",
        unit="EJ/yr",
    )

    # create a copy from interpolation
    df2 = df.interpolate(2007, inplace=False)

    # interpolate should work as if extra_cols is in the _data index
    assert_iamframe_equal(df, df2.filter(year=2007, keep=False))
    obs = df2.filter(year=2007)._data.values
    npt.assert_allclose(obs, [2.4, 1.4])


def test_interpolate_datetimes(test_df):
    # test that interpolation also works with date-times.
    some_date = datetime.datetime(2007, 7, 1)
    if test_df.time_col == "year":
        pytest.raises(ValueError, test_df.interpolate, time=some_date)
    else:
        test_df.interpolate(some_date, inplace=True)
        obs = test_df.filter(time=some_date).data["value"].reset_index(drop=True)
        exp = pd.Series([3, 1.5, 4], name="value")
        pd.testing.assert_series_equal(obs, exp, rtol=0.01)
        # redo the interpolation and check that no duplicates are added
        test_df.interpolate(some_date, inplace=True)
        assert not test_df.filter()._data.index.duplicated().any()
