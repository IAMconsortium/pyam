import logging
import pytest
import re
import datetime

import numpy as np
import pandas as pd
from numpy import testing as npt

from pyam import IamDataFrame, validate, categorize, \
    require_variable, filter_by_meta, META_IDX, IAMC_IDX, sort_data, compare
from pyam.core import _meta_idx, concat

from conftest import TEST_DTS


df_filter_by_meta_matching_idx = pd.DataFrame([
    ['model_a', 'scen_a', 'region_1', 1],
    ['model_a', 'scen_a', 'region_2', 2],
    ['model_a', 'scen_b', 'region_3', 3],
], columns=['model', 'scenario', 'region', 'col'])


df_filter_by_meta_nonmatching_idx = pd.DataFrame([
    ['model_a', 'scen_c', 'region_1', 1, 2],
    ['model_a', 'scen_c', 'region_2', 2, 3],
    ['model_a', 'scen_b', 'region_3', 3, 4],
], columns=['model', 'scenario', 'region', 2010, 2020]
).set_index(['model', 'region'])

df_with_na_columns = pd.DataFrame([
    ['model_a', 'scen_a', 'World', 'Primary Energy', np.nan, 1, 6.],
    ['model_a', 'scen_a', 'World', 'Primary Energy|Coal', 'EJ/yr', 0.5, 3],
    ['model_a', 'scen_b', 'World', 'Primary Energy', 'EJ/yr', 2, 7],
],
    columns=IAMC_IDX + [2005, 2010],
)

df_empty = pd.DataFrame([], columns=IAMC_IDX + [2005, 2010])


def test_init_df_with_index(test_pd_df):
    df = IamDataFrame(test_pd_df.set_index(META_IDX))
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), test_pd_df)


def test_init_df_with_float_cols_raises(test_pd_df):
    _test_df = test_pd_df.rename(columns={2005: 2005.5, 2010: 2010.})
    pytest.raises(ValueError, IamDataFrame, data=_test_df)


def test_init_df_with_duplicates_raises(test_df):
    _df = test_df.timeseries()
    _df = _df.append(_df.iloc[0]).reset_index()
    pytest.raises(ValueError, IamDataFrame, data=_df)


def test_init_df_with_na_unit(test_df):
    pytest.raises(ValueError, IamDataFrame, data=df_with_na_columns)


def test_init_df_with_float_cols(test_pd_df):
    _test_df = test_pd_df.rename(columns={2005: 2005., 2010: 2010.})
    obs = IamDataFrame(_test_df).timeseries().reset_index()
    pd.testing.assert_series_equal(obs[2005], test_pd_df[2005])


def test_init_df_from_timeseries(test_df):
    df = IamDataFrame(test_df.timeseries())
    pd.testing.assert_frame_equal(df.timeseries(), test_df.timeseries())


def test_init_df_with_extra_col(test_pd_df):
    tdf = test_pd_df.copy()

    extra_col = "climate model"
    extra_value = "scm_model"
    tdf[extra_col] = extra_value

    df = IamDataFrame(tdf)

    assert df.extra_cols == [extra_col]
    pd.testing.assert_frame_equal(df.timeseries().reset_index(),
                                  tdf, check_like=True)


def test_init_datetime(test_pd_df):
    tdf = test_pd_df.copy()
    tmin = datetime.datetime(2005, 6, 17)
    tmax = datetime.datetime(2010, 6, 17)
    tdf = tdf.rename(
        {
            2005: tmin,
            2010: tmax,
        },
        axis="columns"
    )

    df = IamDataFrame(tdf)

    assert df["time"].max() == tmax
    assert df["time"].min() == tmin


@pytest.mark.xfail(reason=(
    "pandas datetime is limited to the time period of ~1677-2262, see "
    "https://stackoverflow.com/a/37226672"
))
def test_init_datetime_long_timespan(test_pd_df):
    tdf = test_pd_df.copy()
    tmin = datetime.datetime(2005, 6, 17)
    tmax = datetime.datetime(3005, 6, 17)
    tdf = tdf.rename(
        {
            2005: tmin,
            2010: tmax,
        },
        axis="columns"
    )

    df = IamDataFrame(tdf)

    assert df["time"].max() == tmax
    assert df["time"].min() == tmin


def test_init_datetime_subclass_long_timespan(test_pd_df):
    class TempSubClass(IamDataFrame):
        def _format_datetime_col(self):
            # the subclass does not try to coerce the datetimes to pandas
            # datetimes, instead simply leaving the time column as object type,
            # so we don't run into the problem of pandas limited time period as
            # discussed in https://stackoverflow.com/a/37226672
            pass

    tdf = test_pd_df.copy()
    tmin = datetime.datetime(2005, 6, 17)
    tmax = datetime.datetime(3005, 6, 17)
    tdf = tdf.rename(
        {
            2005: tmin,
            2010: tmax,
        },
        axis="columns"
    )

    df = TempSubClass(tdf)

    assert df["time"].max() == tmax
    assert df["time"].min() == tmin


def test_init_empty_message(test_pd_df, caplog):
    IamDataFrame(data=df_empty)
    drop_message = (
        "Formatted data is empty!"
    )
    message_idx = caplog.messages.index(drop_message)
    assert caplog.records[message_idx].levelno == logging.WARNING


def test_empty_attribute(test_df_year):
    assert not test_df_year.empty
    assert test_df_year.filter(model='foo').empty


def test_equals(test_df_year):
    test_df_year.set_meta([1, 2], name='test')

    # assert that a copy (with changed index-sort) is equal
    df = test_df_year.copy()
    df.data = df.data.sort_values(by='value')
    assert test_df_year.equals(df)

    # assert that adding a new timeseries is not equal
    df = test_df_year.rename(variable={'Primary Energy': 'foo'}, append=True)
    assert not test_df_year.equals(df)

    # assert that adding a new meta indicator is not equal
    df = test_df_year.copy()
    df.set_meta(['foo', ' bar'], name='string')
    assert not test_df_year.equals(df)


def test_equals_raises(test_pd_df):
    df = IamDataFrame(test_pd_df)
    pytest.raises(ValueError, df.equals, test_pd_df)


def test_get_item(test_df):
    assert test_df['model'].unique() == ['model_a']


def test_model(test_df):
    exp = pd.Series(data=['model_a'], name='model')
    pd.testing.assert_series_equal(test_df.models(), exp)


def test_scenario(test_df):
    exp = pd.Series(data=['scen_a', 'scen_b'], name='scenario')
    pd.testing.assert_series_equal(test_df.scenarios(), exp)


def test_region(test_df):
    exp = pd.Series(data=['World'], name='region')
    pd.testing.assert_series_equal(test_df.regions(), exp)


def test_variable(test_df):
    exp = pd.Series(
        data=['Primary Energy', 'Primary Energy|Coal'], name='variable')
    pd.testing.assert_series_equal(test_df.variables(), exp)


def test_variable_unit(test_df):
    dct = {'variable': ['Primary Energy', 'Primary Energy|Coal'],
           'unit': ['EJ/yr', 'EJ/yr']}
    exp = pd.DataFrame.from_dict(dct)[['variable', 'unit']]
    npt.assert_array_equal(test_df.variables(include_units=True), exp)


def test_filter_empty_df():
    # test for issue seen in #254
    df = IamDataFrame(data=df_empty)
    obs = df.filter(variable='foo')
    assert len(obs) == 0


def test_filter_variable_and_depth(test_df):
    obs = list(test_df.filter(variable='*rimary*C*', level=0).variables())
    exp = ['Primary Energy|Coal']
    assert obs == exp

    obs = list(test_df.filter(variable='*rimary*C*', level=1).variables())
    assert len(obs) == 0


def test_variable_depth_0_keep_false(test_df):
    obs = list(test_df.filter(level=0, keep=False)['variable'].unique())
    exp = ['Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_raises(test_df):
    pytest.raises(ValueError, test_df.filter, level='1/')


def test_filter_error_illegal_column(test_df):
    # filtering by column `foo` is not valid
    pytest.raises(ValueError, test_df.filter, foo='test')


def test_filter_error_keep(test_df):
    # string or non-starred dict was mis-interpreted as `keep` kwarg, see #253
    pytest.raises(ValueError, test_df.filter, model='foo', keep=1)
    pytest.raises(ValueError, test_df.filter, dict(model='foo'))


def test_filter_year(test_df):
    obs = test_df.filter(year=2005)
    if "year" in test_df.data.columns:
        npt.assert_equal(obs['year'].unique(), 2005)
    else:
        expected = np.array(pd.to_datetime('2005-06-17T00:00:00.0'),
                            dtype=np.datetime64)
        unique_time = obs['time'].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


@pytest.mark.parametrize("test_month",
                         [6, "June", "Jun", "jun", ["Jun", "jun"]])
def test_filter_month(test_df, test_month):
    if "year" in test_df.data.columns:
        error_msg = re.escape("filter by `month` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_df.filter(month=test_month)
    else:
        obs = test_df.filter(month=test_month)
        expected = np.array(pd.to_datetime('2005-06-17T00:00:00.0'),
                            dtype=np.datetime64)
        unique_time = obs['time'].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


@pytest.mark.parametrize("test_month", [6, "Jun", "jun", ["Jun", "jun"]])
def test_filter_year_month(test_df, test_month):
    if "year" in test_df.data.columns:
        error_msg = re.escape("filter by `month` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_df.filter(year=2005, month=test_month)
    else:
        obs = test_df.filter(year=2005, month=test_month)
        expected = np.array(pd.to_datetime('2005-06-17T00:00:00.0'),
                            dtype=np.datetime64)
        unique_time = obs['time'].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


@pytest.mark.parametrize("test_day",
                         [17, "Fri", "Friday", "friday", ["Fri", "fri"]])
def test_filter_day(test_df, test_day):
    if "year" in test_df.data.columns:
        error_msg = re.escape("filter by `day` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_df.filter(day=test_day)
    else:
        obs = test_df.filter(day=test_day)
        expected = np.array(pd.to_datetime('2005-06-17T00:00:00.0'),
                            dtype=np.datetime64)
        unique_time = obs['time'].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


@pytest.mark.parametrize("test_hour", [0, 12, [12, 13]])
def test_filter_hour(test_df, test_hour):
    if "year" in test_df.data.columns:
        error_msg = re.escape("filter by `hour` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_df.filter(hour=test_hour)
    else:
        obs = test_df.filter(hour=test_hour)
        test_hour = [test_hour] if isinstance(test_hour, int) else test_hour
        expected_rows = (test_df.data["time"]
                         .apply(lambda x: x.hour).isin(test_hour))
        expected = test_df.data["time"].loc[expected_rows].unique()

        unique_time = obs['time'].unique()
        npt.assert_array_equal(unique_time, expected)


def test_filter_time_exact_match(test_df):
    if "year" in test_df.data.columns:
        error_msg = re.escape(
            "`year` can only be filtered with ints or lists of ints"
        )
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(year=datetime.datetime(2005, 6, 17))
    else:
        obs = test_df.filter(time=datetime.datetime(2005, 6, 17))
        expected = np.array(pd.to_datetime('2005-06-17T00:00:00.0'),
                            dtype=np.datetime64)
        unique_time = obs['time'].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


def test_filter_time_range(test_df):
    error_msg = r".*datetime.datetime.*"
    with pytest.raises(TypeError, match=error_msg):
        test_df.filter(year=range(
            datetime.datetime(2000, 6, 17),
            datetime.datetime(2009, 6, 17)
        ))


def test_filter_time_range_year(test_df):
    obs = test_df.filter(year=range(2000, 2008))

    if "year" in test_df.data.columns:
        unique_time = obs['year'].unique()
        expected = np.array([2005])
    else:
        unique_time = obs['time'].unique()
        expected = np.array(pd.to_datetime('2005-06-17T00:00:00.0'),
                            dtype=np.datetime64)

    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("month_range", [range(1, 7), "Mar-Jun"])
def test_filter_time_range_month(test_df, month_range):
    if "year" in test_df.data.columns:
        error_msg = re.escape("filter by `month` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_df.filter(month=month_range)
    else:
        obs = test_df.filter(month=month_range)
        expected = np.array(pd.to_datetime('2005-06-17T00:00:00.0'),
                            dtype=np.datetime64)

        unique_time = obs['time'].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


@pytest.mark.parametrize("month_range", [["Mar-Jun", "Nov-Feb"]])
def test_filter_time_range_round_the_clock_error(test_df, month_range):
    if "year" in test_df.data.columns:
        error_msg = re.escape("filter by `month` not supported")
        with pytest.raises(ValueError, match=error_msg):
            test_df.filter(month=month_range)
    else:
        error_msg = re.escape(
            "string ranges must lead to increasing integer ranges, "
            "Nov-Feb becomes [11, 2]"
        )
        with pytest.raises(ValueError, match=error_msg):
            test_df.filter(month=month_range)


@pytest.mark.parametrize("day_range", [range(14, 20), "Thu-Sat"])
def test_filter_time_range_day(test_df, day_range):
    if "year" in test_df.data.columns:
        error_msg = re.escape("filter by `day` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_df.filter(day=day_range)
    else:
        obs = test_df.filter(day=day_range)
        expected = np.array(pd.to_datetime('2005-06-17T00:00:00.0'),
                            dtype=np.datetime64)
        unique_time = obs['time'].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


@pytest.mark.parametrize("hour_range", [range(10, 14)])
def test_filter_time_range_hour(test_df, hour_range):
    if "year" in test_df.data.columns:
        error_msg = re.escape("filter by `hour` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_df.filter(hour=hour_range)
    else:
        obs = test_df.filter(hour=hour_range)

        expected_rows = (test_df.data["time"]
                         .apply(lambda x: x.hour).isin(hour_range))
        expected = test_df.data["time"].loc[expected_rows].unique()

        unique_time = obs['time'].unique()
        npt.assert_array_equal(unique_time, expected)


def test_filter_time_no_match(test_df):
    if "year" in test_df.data.columns:
        error_msg = re.escape(
            "`year` can only be filtered with ints or lists of ints"
        )
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(year=datetime.datetime(2004, 6, 18))
    else:
        obs = test_df.filter(time=datetime.datetime(2004, 6, 18))
        assert obs.data.empty


def test_filter_time_not_datetime_error(test_df):
    if "year" in test_df.data.columns:
        with pytest.raises(ValueError, match=re.escape("`time`")):
            test_df.filter(time=datetime.datetime(2004, 6, 18))
    else:
        error_msg = re.escape(
            "`time` can only be filtered by datetimes"
        )
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(time=2005)
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(time='summer')


def test_filter_time_not_datetime_range_error(test_df):
    if "year" in test_df.data.columns:
        with pytest.raises(ValueError, match=re.escape("`time`")):
            test_df.filter(time=range(2000, 2008))
    else:
        error_msg = re.escape(
            "`time` can only be filtered by datetimes"
        )
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(time=range(2000, 2008))
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(time=['summer', 'winter'])


def test_filter_year_with_time_col(test_pd_df):
    test_pd_df['time'] = ['summer', 'summer', 'winter']
    df = IamDataFrame(test_pd_df)
    obs = df.filter(time='summer').timeseries()

    exp = test_pd_df.set_index(IAMC_IDX + ['time'])
    exp.columns = list(map(int, exp.columns))
    pd.testing.assert_frame_equal(obs, exp[0:2])


def test_filter_as_kwarg(test_df):
    obs = list(test_df.filter(variable='Primary Energy|Coal').scenarios())
    assert obs == ['scen_a']


def test_filter_keep_false(test_df):
    df = test_df.filter(variable='Primary Energy|Coal', year=2005, keep=False)
    obs = df.data[df.data.scenario == 'scen_a'].value
    npt.assert_array_equal(obs, [1, 6, 3])


def test_filter_by_regexp(test_df):
    obs = test_df.filter(scenario='sce._a$', regexp=True)
    assert obs['scenario'].unique() == 'scen_a'


def test_timeseries(test_df):
    dct = {'model': ['model_a'] * 2, 'scenario': ['scen_a'] * 2,
           'years': [2005, 2010], 'value': [1, 6]}
    exp = pd.DataFrame(dct).pivot_table(index=['model', 'scenario'],
                                        columns=['years'], values='value')
    obs = test_df.filter(scenario='scen_a',
                         variable='Primary Energy').timeseries()
    npt.assert_array_equal(obs, exp)


def test_timeseries_raises(test_df_year):
    _df = test_df_year.filter(model='foo')
    pytest.raises(ValueError, _df.timeseries)


def test_filter_meta_index(test_df):
    obs = test_df.filter(scenario='scen_b').meta.index
    exp = pd.MultiIndex(levels=[['model_a'], ['scen_b']],
                        codes=[[0], [0]],
                        names=['model', 'scenario'])
    pd.testing.assert_index_equal(obs, exp)


def test_meta_idx(test_df):
    # assert that the `drop_duplicates()` in `_meta_idx()` returns right length
    assert len(_meta_idx(test_df.data)) == 2


def test_require_variable(test_df):
    obs = test_df.require_variable(variable='Primary Energy|Coal',
                                   exclude_on_fail=True)
    assert len(obs) == 1
    assert obs.loc[0, 'scenario'] == 'scen_b'

    assert list(test_df['exclude']) == [False, True]


def test_require_variable_top_level(test_df):
    obs = require_variable(test_df, variable='Primary Energy|Coal',
                           exclude_on_fail=True)
    assert len(obs) == 1
    assert obs.loc[0, 'scenario'] == 'scen_b'

    assert list(test_df['exclude']) == [False, True]


def test_require_variable_year_list(test_df):
    years = [2005, 2010]

    # checking for variables that have ANY of the years in the list
    df = IamDataFrame(test_df.data[1:])
    df.require_variable(variable='Primary Energy',
                        year=years,
                        exclude_on_fail=True)
    df.filter(exclude=False, inplace=True)

    assert len(df.variables()) == 2
    assert len(df.scenarios()) == 2

    # checking for variables that have ALL of the years in the list
    df = IamDataFrame(test_df.data[1:])
    for y in years:
        df.require_variable(variable='Primary Energy',
                            year=y,
                            exclude_on_fail=True)
    df.filter(exclude=False, inplace=True)

    assert len(df.variables()) == 1
    assert len(df.scenarios()) == 1


def test_validate_all_pass(test_df):
    obs = test_df.validate(
        {'Primary Energy': {'up': 10}}, exclude_on_fail=True)
    assert obs is None
    assert len(test_df.data) == 6  # data unchanged

    assert list(test_df['exclude']) == [False, False]  # none excluded


def test_validate_nonexisting(test_df):
    obs = test_df.validate({'Primary Energy|Coal': {'up': 2}},
                           exclude_on_fail=True)
    assert len(obs) == 1
    assert obs['scenario'].values[0] == 'scen_a'

    assert list(test_df['exclude']) == [True, False]  # scenario with failed
    # validation excluded, scenario with non-defined value passes validation


def test_validate_up(test_df):
    obs = test_df.validate({'Primary Energy': {'up': 6.5}},
                           exclude_on_fail=False)
    assert len(obs) == 1
    if 'year' in test_df.data:
        assert obs['year'].values[0] == 2010
    else:
        exp_time = pd.to_datetime(datetime.datetime(2010, 7, 21))
        print(exp_time)
        assert pd.to_datetime(obs['time'].values[0]).date() == exp_time

    assert list(test_df['exclude']) == [False, False]  # assert none excluded


def test_validate_lo(test_df):
    obs = test_df.validate({'Primary Energy': {'up': 8, 'lo': 2.0}})
    assert len(obs) == 1
    if 'year' in test_df.data:
        assert obs['year'].values[0] == 2005
    else:
        exp_year = pd.to_datetime(datetime.datetime(2005, 6, 17))
        assert pd.to_datetime(obs['time'].values[0]).date() == exp_year

    assert list(obs['scenario'].values) == ['scen_a']


def test_validate_both(test_df):
    obs = test_df.validate({'Primary Energy': {'up': 6.5, 'lo': 2.0}})
    assert len(obs) == 2
    if 'year' in test_df.data:
        assert list(obs['year'].values) == [2005, 2010]
    else:
        exp_time = pd.to_datetime(TEST_DTS)
        obs.time = obs.time.dt.normalize()
        assert (pd.to_datetime(obs['time'].values) == exp_time).all()

    assert list(obs['scenario'].values) == ['scen_a', 'scen_b']


def test_validate_year(test_df):
    obs = test_df.validate({'Primary Energy': {'up': 5.0, 'year': 2005}},
                           exclude_on_fail=False)
    assert obs is None

    obs = test_df.validate({'Primary Energy': {'up': 5.0, 'year': 2010}},
                           exclude_on_fail=False)
    assert len(obs) == 2


def test_validate_exclude(test_df):
    test_df.validate({'Primary Energy': {'up': 6.0}}, exclude_on_fail=True)
    assert list(test_df['exclude']) == [False, True]


def test_validate_top_level(test_df):
    obs = validate(test_df, criteria={'Primary Energy': {'up': 6.0}},
                   exclude_on_fail=True, variable='Primary Energy')
    assert len(obs) == 1
    if 'year' in test_df.data:
        assert obs['year'].values[0] == 2010
    else:
        exp_time = pd.to_datetime(datetime.datetime(2010, 7, 21))
        assert (pd.to_datetime(obs['time'].values[0]).date() == exp_time)
    assert list(test_df['exclude']) == [False, True]


def test_category_none(test_df):
    test_df.categorize('category', 'Testing', {'Primary Energy': {'up': 0.8}})
    assert 'category' not in test_df.meta.columns


def test_category_pass(test_df):
    dct = {'model': ['model_a', 'model_a'],
           'scenario': ['scen_a', 'scen_b'],
           'category': ['foo', None]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    test_df.categorize('category', 'foo', {'Primary Energy':
                                           {'up': 6, 'year': 2010}})
    obs = test_df['category']
    pd.testing.assert_series_equal(obs, exp)


def test_category_top_level(test_df):
    dct = {'model': ['model_a', 'model_a'],
           'scenario': ['scen_a', 'scen_b'],
           'category': ['foo', None]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    categorize(test_df, 'category', 'foo',
               criteria={'Primary Energy': {'up': 6, 'year': 2010}},
               variable='Primary Energy')
    obs = test_df['category']
    pd.testing.assert_series_equal(obs, exp)


def test_interpolate(test_df_year):
    test_df_year.interpolate(2007)
    obs = test_df_year.filter(year=2007).data['value'].reset_index(drop=True)
    exp = pd.Series([3, 1.5, 4], name='value')
    pd.testing.assert_series_equal(obs, exp)

    # redo the interpolation and check that no duplicates are added
    test_df_year.interpolate(2007)
    assert not test_df_year.filter().data.duplicated().any()


def test_interpolate_datetimes(test_df):
    # test that interpolation also works with date-times.
    some_date = datetime.datetime(2007, 7, 1)
    if test_df.time_col == "year":
        pytest.raises(ValueError, test_df.interpolate, time=some_date)
    else:
        test_df.interpolate(some_date)
        obs = test_df.filter(time=some_date).data['value']\
            .reset_index(drop=True)
        exp = pd.Series([3, 1.5, 4], name='value')
        pd.testing.assert_series_equal(obs, exp, check_less_precise=True)
        # redo the interpolation and check that no duplicates are added
        test_df.interpolate(some_date)
        assert not test_df.filter().data.duplicated().any()


def test_filter_by_bool(test_df):
    test_df.set_meta([True, False], name='exclude')
    obs = test_df.filter(exclude=True)
    assert obs['scenario'].unique() == 'scen_a'


def test_filter_by_int(test_df):
    test_df.set_meta([1, 2], name='test')
    obs = test_df.filter(test=[1, 3])
    assert obs['scenario'].unique() == 'scen_a'


def _r5_regions_exp(df):
    df = df.filter(region='World', keep=False)
    df['region'] = 'R5MAF'
    return sort_data(df.data, df._LONG_IDX)


def test_map_regions_r5(reg_df):
    obs = reg_df.map_regions('r5_region').data
    exp = _r5_regions_exp(reg_df)
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_map_regions_r5_region_col(reg_df):
    df = reg_df.filter(model='MESSAGE-GLOBIOM')
    obs = df.map_regions(
        'r5_region', region_col='MESSAGE-GLOBIOM.REGION').data
    exp = _r5_regions_exp(df)
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_map_regions_r5_inplace(reg_df):
    exp = _r5_regions_exp(reg_df)
    reg_df.map_regions('r5_region', inplace=True)
    obs = reg_df.data
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_map_regions_r5_agg(reg_df):
    columns = reg_df.data.columns
    obs = reg_df.map_regions('r5_region', agg='sum').data

    exp = _r5_regions_exp(reg_df)
    grp = list(columns)
    grp.remove('value')
    exp = exp.groupby(grp).sum().reset_index()
    exp = exp[columns]
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_48a():
    # tests fix for #48 mapping many->few
    df = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'SSD', 'var', 'unit', 1, 6],
        ['model', 'scen', 'SDN', 'var', 'unit', 2, 7],
        ['model', 'scen1', 'SSD', 'var', 'unit', 2, 7],
        ['model', 'scen1', 'SDN', 'var', 'unit', 2, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))

    exp = _r5_regions_exp(df)
    columns = df.data.columns
    grp = list(columns)
    grp.remove('value')
    exp = exp.groupby(grp).sum().reset_index()
    exp = exp[columns]

    obs = df.map_regions('r5_region', region_col='iso', agg='sum').data

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_48b():
    # tests fix for #48 mapping few->many

    exp = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'SSD', 'var', 'unit', 1, 6],
        ['model', 'scen', 'SDN', 'var', 'unit', 1, 6],
        ['model', 'scen1', 'SSD', 'var', 'unit', 2, 7],
        ['model', 'scen1', 'SDN', 'var', 'unit', 2, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    )).data

    df = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'R5MAF', 'var', 'unit', 1, 6],
        ['model', 'scen1', 'R5MAF', 'var', 'unit', 2, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))
    obs = df.map_regions('iso', region_col='r5_region').data
    obs = sort_data(obs[obs.region.isin(['SSD', 'SDN'])], df._LONG_IDX)

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_48c():
    # tests fix for #48 mapping few->many, dropping duplicates

    exp = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'AGO', 'var', 'unit', 1, 6],
        ['model', 'scen1', 'AGO', 'var', 'unit', 2, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    )).data.reset_index(drop=True)

    df = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'R5MAF', 'var', 'unit', 1, 6],
        ['model', 'scen1', 'R5MAF', 'var', 'unit', 2, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))
    obs = df.map_regions('iso', region_col='r5_region',
                         remove_duplicates=True).data
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_pd_filter_by_meta(test_df):
    data = df_filter_by_meta_matching_idx.set_index(['model', 'region'])

    test_df.set_meta([True, False], 'boolean')
    test_df.set_meta(0, 'integer')

    obs = filter_by_meta(data, test_df, join_meta=True,
                         boolean=True, integer=None)
    obs = obs.reindex(columns=['scenario', 'col', 'boolean', 'integer'])

    exp = data.iloc[0:2].copy()
    exp['boolean'] = True
    exp['integer'] = 0

    pd.testing.assert_frame_equal(obs, exp)


def test_pd_filter_by_meta_no_index(test_df):
    data = df_filter_by_meta_matching_idx

    test_df.set_meta([True, False], 'boolean')
    test_df.set_meta(0, 'int')

    obs = filter_by_meta(data, test_df, join_meta=True,
                         boolean=True, int=None)
    obs = obs.reindex(columns=META_IDX + ['region', 'col', 'boolean', 'int'])

    exp = data.iloc[0:2].copy()
    exp['boolean'] = True
    exp['int'] = 0

    pd.testing.assert_frame_equal(obs, exp)


def test_pd_filter_by_meta_nonmatching_index(test_df):
    data = df_filter_by_meta_nonmatching_idx
    test_df.set_meta(['a', 'b'], 'string')

    obs = filter_by_meta(data, test_df, join_meta=True, string='b')
    obs = obs.reindex(columns=['scenario', 2010, 2020, 'string'])

    exp = data.iloc[2:3].copy()
    exp['string'] = 'b'

    pd.testing.assert_frame_equal(obs, exp)


def test_pd_join_by_meta_nonmatching_index(test_df):
    data = df_filter_by_meta_nonmatching_idx
    test_df.set_meta(['a', 'b'], 'string')

    obs = filter_by_meta(data, test_df, join_meta=True, string=None)
    obs = obs.reindex(columns=['scenario', 2010, 2020, 'string'])

    exp = data.copy()
    exp['string'] = [np.nan, np.nan, 'b']

    pd.testing.assert_frame_equal(obs.sort_index(level=1), exp)


def test_concat_fails_iter():
    pytest.raises(TypeError, concat, 1)


def test_concat_fails_notdf():
    pytest.raises(TypeError, concat, 'foo')


def test_concat(test_df):
    left = IamDataFrame(test_df.data.copy())
    right = left.data.copy()
    right['model'] = 'not left'
    right = IamDataFrame(right)

    result = concat([left, right])

    obs = result.data.reset_index(drop=True)
    exp = pd.concat([left.data, right.data]).reset_index(drop=True)
    pd.testing.assert_frame_equal(obs, exp)

    obs = result.meta.reset_index(drop=True)
    exp = pd.concat([left.meta, right.meta]).reset_index(drop=True)
    pd.testing.assert_frame_equal(obs, exp)


def test_normalize(test_df):
    exp = test_df.data.copy().reset_index(drop=True)
    exp.loc[1::2, 'value'] /= exp['value'][::2].values
    exp.loc[::2, 'value'] /= exp['value'][::2].values
    if "year" in test_df.data:
        obs = test_df.normalize(year=2005).data.reset_index(drop=True)
    else:
        obs = test_df.normalize(
            time=datetime.datetime(2005, 6, 17)
        ).data.reset_index(drop=True)
    pd.testing.assert_frame_equal(obs, exp)


def test_normalize_not_time(test_df):
    pytest.raises(ValueError, test_df.normalize, variable='foo')
    pytest.raises(ValueError, test_df.normalize, year=2015, variable='foo')


@pytest.mark.parametrize("inplace", [True, False])
def test_swap_time_to_year(test_df, inplace):
    if "year" in test_df.data:
        return  # year df not relevant for this test

    exp = test_df.data.copy()
    exp["year"] = exp["time"].apply(lambda x: x.year)
    exp = exp.drop("time", axis="columns")
    exp = IamDataFrame(exp)

    obs = test_df.swap_time_for_year(inplace=inplace)

    if inplace:
        assert obs is None
        assert compare(test_df, exp).empty
    else:
        assert compare(obs, exp).empty
        assert "year" not in test_df.data.columns


@pytest.mark.parametrize("inplace", [True, False])
def test_swap_time_to_year_errors(test_df, inplace):
    if "year" in test_df.data:
        with pytest.raises(ValueError):
            test_df.swap_time_for_year(inplace=inplace)
        return

    tdf = test_df.data.copy()
    tdf["time"] = tdf["time"].apply(
        lambda x: datetime.datetime(2005, x.month, x.day)
    )

    with pytest.raises(ValueError):
        IamDataFrame(tdf).swap_time_for_year(inplace=inplace)
