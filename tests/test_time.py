import pytest
import pandas as pd
from datetime import datetime
from pyam import IamDataFrame
from pyam.testing import assert_iamframe_equal

from .conftest import TEST_DTS, TEST_TIME_STR, TEST_TIME_STR_HR


def get_subannual_df(date1, date2):
    df = pd.DataFrame(
        [
            ["scen_a", "Primary Energy", 2005, date1, 1.0],
            ["scen_a", "Primary Energy", 2010, date2, 6.0],
            ["scen_a", "Primary Energy|Coal", 2005, date1, 0.5],
            ["scen_a", "Primary Energy|Coal", 2010, date2, 3],
            ["scen_b", "Primary Energy", 2005, date1, 2],
            ["scen_b", "Primary Energy", 2010, date2, 7],
        ],
        columns=["scenario", "variable", "year", "subannual", "value"],
    )
    return IamDataFrame(df, model="model_a", region="World", unit="EJ/yr")


# this is the subannual column format used in the openENTRANCE project
OE_DATETIME = ["2005-10-01 23:15+01:00", "2010-10-02 23:15+01:00"]
OE_SUBANNUAL_FORMAT = lambda x: x.strftime("%m-%d %H:%M%z").replace("+0100", "+01:00")


@pytest.mark.parametrize("inplace", [True, False])
def test_swap_time_to_year(test_df, inplace):
    """Swap time column for year (int) dropping subannual time resolution (default)"""

    if test_df.time_col == "year":
        pytest.skip("IamDataFrame with time domain `year` not relevant for this test.")

    exp = test_df.data
    exp["year"] = exp["time"].apply(lambda x: x.year)
    exp = exp.drop("time", axis="columns")
    exp = IamDataFrame(exp, meta=test_df.meta)

    obs = test_df.swap_time_for_year(inplace=inplace)

    if inplace:
        assert obs is None
        obs = test_df

    assert_iamframe_equal(obs, exp)
    match = "'IamDataFrame' object has no attribute 'time'"
    with pytest.raises(AttributeError, match=match):
        obs.time


@pytest.mark.parametrize(
    "columns, subannual, dates",
    [
        # use the default datetime format without `year`
        [TEST_DTS, True, ["06-17 00:00", "07-21 00:00"]],
        [TEST_TIME_STR, True, ["06-17 00:00", "07-21 00:00"]],
        [TEST_TIME_STR_HR, True, ["06-17 00:00", "07-21 12:00"]],
        # apply formatting for strftime as str
        [TEST_DTS, "%m-%d", ["06-17", "07-21"]],
        # apply openENTRANCE formatting with timezone
        [OE_DATETIME, OE_SUBANNUAL_FORMAT, ["10-01 23:15+01:00", "10-02 23:15+01:00"]],
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_swap_time_to_year_subannual(test_pd_df, columns, subannual, dates, inplace):
    """Swap time column for year (int) keeping subannual resolution as extra-column"""

    test_pd_df.rename({2005: columns[0], 2010: columns[1]}, axis=1, inplace=True)

    # check swapping time for year
    df = IamDataFrame(test_pd_df)
    obs = df.swap_time_for_year(subannual=subannual, inplace=inplace)

    if inplace:
        assert obs is None
        obs = df

    exp = get_subannual_df(dates[0], dates[1])
    assert_iamframe_equal(obs, exp)

    # check that reverting using `swap_year_for_time` yields the original data
    assert_iamframe_equal(obs.swap_year_for_time(), IamDataFrame(test_pd_df))


def test_swap_time_to_year_errors(test_df):
    """Assert that swapping time column for year (int) raises the expected errors"""

    # swapping time to year raises when the IamDataFrame has time domain `year`
    if test_df.time_col == "year":
        match = "Time domain must be datetime to use this method"
        with pytest.raises(ValueError, match=match):
            test_df.swap_time_for_year()

    else:
        # set time column to same year so that dropping month/day leads to duplicates
        tdf = test_df.data
        tdf["time"] = tdf["time"].apply(lambda x: datetime(2005, x.month, x.day))

        with pytest.raises(ValueError, match="Swapping time for year causes duplicate"):
            IamDataFrame(tdf).swap_time_for_year()
