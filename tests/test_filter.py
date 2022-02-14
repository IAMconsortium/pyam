import datetime
import re
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from pyam import IamDataFrame, IAMC_IDX

from .conftest import EXP_DATETIME_INDEX


def test_filter_error_illegal_column(test_df):
    # filtering by column `foo` is not valid
    pytest.raises(ValueError, test_df.filter, foo="test")


def test_filter_error_keep(test_df):
    # string or non-starred dict was mis-interpreted as `keep` kwarg, see #253
    pytest.raises(ValueError, test_df.filter, model="foo", keep=1)
    pytest.raises(ValueError, test_df.filter, dict(model="foo"))


def test_filter_year(test_df):
    obs = test_df.filter(year=2005)
    if test_df.time_col == "year":
        assert obs.year == [2005]
    else:
        pdt.assert_index_equal(obs.time, EXP_DATETIME_INDEX)


@pytest.mark.parametrize(
    "arg_year, arg_time",
    [
        (dict(year=2005), dict(year=2010)),
        (dict(time_domain="year"), dict(time_domain="datetime")),
    ],
)
def test_filter_mixed_time_domain(test_df_mixed, arg_year, arg_time):
    """Assert that reassigning attributes works for filtering from mixed time domain"""

    assert test_df_mixed.time_domain == "mixed"

    # filtering to datetime-only works as expected
    obs = test_df_mixed.filter(**arg_time)
    assert obs.time_domain == "datetime"
    pdt.assert_index_equal(obs.time, pd.DatetimeIndex(["2010-07-21"]))

    # filtering to year-only works as expected including changing of time domain
    obs = test_df_mixed.filter(**arg_year)
    assert obs.time_col == "year"
    assert obs.time_domain == "year"
    assert obs.year == [2005]
    pdt.assert_index_equal(obs.time, pd.Int64Index([2005]))


def test_filter_time_domain_raises(test_df_year):
    """Assert that error is raised for invalid time_domain filter value"""

    match = "Filter by `time_domain='mixed'` not supported!"
    with pytest.raises(ValueError, match=match):
        test_df_year.filter(time_domain="mixed")


def test_filter_time_match_raises(test_df_time):
    """Assert that error is raised for invalid time-component filter value"""

    match = r"Could not convert months to integer: \['year'\]"
    with pytest.raises(ValueError, match=match):
        test_df_time.filter(month="year")


@pytest.mark.parametrize("test_month", [6, "June", "Jun", "jun", ["Jun", "jun"]])
def test_filter_month(test_df, test_month):
    if test_df.time_col == "year":
        # Filtering data with yearly time domain returns empty
        assert test_df.filter(month=test_month).empty
    else:
        obs = test_df.filter(month=test_month)
        pdt.assert_index_equal(obs.time, EXP_DATETIME_INDEX)


@pytest.mark.parametrize("test_month", [6, "Jun", "jun", ["Jun", "jun"]])
def test_filter_year_month(test_df, test_month):
    if test_df.time_col == "year":
        # Filtering data with yearly time domain returns empty
        assert test_df.filter(year=2005, month=test_month).empty
    else:
        obs = test_df.filter(year=2005, month=test_month)
        pdt.assert_index_equal(obs.time, EXP_DATETIME_INDEX)


@pytest.mark.parametrize("test_day", [17, "Fri", "Friday", "friday", ["Fri", "fri"]])
def test_filter_day(test_df, test_day):
    if test_df.time_col == "year":
        # Filtering data with yearly time domain returns empty
        assert test_df.filter(day=test_day).empty
    else:
        obs = test_df.filter(day=test_day)
        pdt.assert_index_equal(obs.time, EXP_DATETIME_INDEX)


def test_filter_with_numpy_64_date_vals(test_df):
    dates = test_df[test_df.time_col].unique()
    key = "year" if test_df.time_col == "year" else "time"
    res_0 = test_df.filter(**{key: dates[0]})
    res = test_df.filter(**{key: dates})
    assert np.equal(res_0.data[res_0.time_col].values, dates[0]).all()
    assert res.equals(test_df)


@pytest.mark.parametrize("test_hour", [0, 12, [12, 13]])
def test_filter_hour(test_df, test_hour):
    if test_df.time_col == "year":
        # Filtering data with yearly time domain returns empty
        assert test_df.filter(hour=test_hour).empty
    else:
        obs = test_df.filter(hour=test_hour)
        test_hour = [test_hour] if isinstance(test_hour, int) else test_hour
        expected_rows = test_df.data["time"].apply(lambda x: x.hour).isin(test_hour)
        expected = test_df.data["time"].loc[expected_rows].unique()

        unique_time = np.array(obs["time"].unique(), dtype=np.datetime64)
        npt.assert_array_equal(unique_time, expected)


def test_filter_time_exact_match(test_df):
    if test_df.time_col == "year":
        error_msg = re.escape("Filter by `year` requires integers!")
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(year=datetime.datetime(2005, 6, 17))
    else:
        obs = test_df.filter(time=datetime.datetime(2005, 6, 17))
        pdt.assert_index_equal(obs.time, EXP_DATETIME_INDEX)


def test_filter_time_range(test_df):
    error_msg = r".*datetime.datetime.*"
    with pytest.raises(TypeError, match=error_msg):
        test_df.filter(
            year=range(datetime.datetime(2000, 6, 17), datetime.datetime(2009, 6, 17))
        )


def test_filter_time_range_year(test_df):
    obs = test_df.filter(year=range(2000, 2008))

    if test_df.time_col == "year":
        assert obs.year == [2005]
    else:
        pdt.assert_index_equal(obs.time, EXP_DATETIME_INDEX)


@pytest.mark.parametrize("month_range", [range(1, 7), "Mar-Jun"])
def test_filter_time_range_month(test_df, month_range):
    if test_df.time_col == "year":
        # Filtering data with yearly time domain returns empty
        assert test_df.filter(hour=month_range).empty
    else:
        obs = test_df.filter(month=month_range)
        pdt.assert_index_equal(obs.time, EXP_DATETIME_INDEX)


@pytest.mark.parametrize("month_range", [["Mar-Jun", "Nov-Feb"]])
def test_filter_time_range_round_the_clock_error(test_df, month_range):
    if test_df.time_col == "year":
        # Filtering data with yearly time domain returns empty
        assert test_df.filter(month=month_range).empty
    else:
        error_msg = re.escape(
            "string ranges must lead to increasing integer ranges, "
            "Nov-Feb becomes [11, 2]"
        )
        with pytest.raises(ValueError, match=error_msg):
            test_df.filter(month=month_range)


@pytest.mark.parametrize("day_range", [range(14, 20), "Thu-Sat"])
def test_filter_time_range_day(test_df, day_range):
    if test_df.time_col == "year":
        # Filtering data with yearly time domain returns empty
        assert test_df.filter(day=day_range).empty
    else:
        obs = test_df.filter(day=day_range)
        pdt.assert_index_equal(obs.time, EXP_DATETIME_INDEX)


@pytest.mark.parametrize("hour_range", [range(10, 14)])
def test_filter_time_range_hour(test_df, hour_range):
    if test_df.time_col == "year":
        # Filtering data with yearly time domain returns empty
        assert test_df.filter(hour=hour_range).empty
    else:
        obs = test_df.filter(hour=hour_range)

        expected_rows = test_df.data["time"].apply(lambda x: x.hour).isin(hour_range)
        expected = test_df.data["time"].loc[expected_rows].unique()

        unique_time = np.array(obs["time"].unique(), dtype=np.datetime64)
        npt.assert_array_equal(unique_time, expected)


def test_filter_time_no_match(test_df):
    if test_df.time_col == "year":
        # Filtering data with yearly time domain returns empty
        assert test_df.filter(time=datetime.datetime(2004, 6, 18)).empty
    else:
        obs = test_df.filter(time=datetime.datetime(2004, 6, 18))
        assert obs.data.empty


def test_filter_time_not_datetime_error(test_df):
    if test_df.time_col == "year":
        # Filtering data with yearly time domain returns empty
        assert test_df.filter(time=2005).empty
    else:
        error_msg = re.escape("`time` can only be filtered by datetimes")
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(time=2005)
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(time="summer")


def test_filter_time_not_datetime_range_error(test_df):
    if test_df.time_col == "year":
        # Filtering data with yearly time domain returns empty
        assert test_df.filter(time=range(2000, 2008)).empty
    else:
        error_msg = re.escape("`time` can only be filtered by datetimes")
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(time=range(2000, 2008))
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(time=["summer", "winter"])


def test_filter_year_with_time_col(test_pd_df):
    test_pd_df["subannual"] = ["summer", "summer", "winter"]
    df = IamDataFrame(test_pd_df)
    obs = df.filter(subannual="summer").timeseries()

    exp = test_pd_df.set_index(IAMC_IDX + ["subannual"])
    exp.columns = list(map(int, exp.columns))
    pd.testing.assert_frame_equal(obs, exp[0:2])


def test_filter_as_kwarg(test_df):
    _df = test_df.filter(variable="Primary Energy|Coal")
    assert _df.scenario == ["scen_a"]


def test_filter_keep_false(test_df):
    df = test_df.filter(variable="Primary Energy|Coal", year=2005, keep=False)
    obs = df.data[df.data.scenario == "scen_a"].value
    npt.assert_array_equal(obs, [1, 6, 3])


def test_filter_by_regexp(test_df):
    obs = test_df.filter(scenario="sce._a$", regexp=True)
    assert obs["scenario"].unique() == "scen_a"
