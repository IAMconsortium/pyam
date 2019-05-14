import os
import pytest
import re
import datetime

import numpy as np
import pandas as pd
from numpy import testing as npt

from pyam import IamDataFrame, validate, categorize, \
    require_variable, filter_by_meta, META_IDX, IAMC_IDX, sort_data, compare
from pyam.core import _meta_idx, concat

from conftest import TEST_DATA_DIR


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


def test_to_excel(test_df):
    fname = 'foo_testing.xlsx'
    test_df.to_excel(fname)
    assert os.path.exists(fname)
    os.remove(fname)


def test_to_csv(test_df):
    fname = 'foo_testing.csv'
    test_df.to_csv(fname)
    assert os.path.exists(fname)
    os.remove(fname)


def test_get_item(test_df):
    assert test_df['model'].unique() == ['model_a']


def test_model(test_df):
    pd.testing.assert_series_equal(test_df.models(),
                                   pd.Series(data=['model_a'], name='model'))


def test_scenario(test_df):
    exp = pd.Series(data=['scen_a'], name='scenario')
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
           'unit': ['EJ/y', 'EJ/y']}
    exp = pd.DataFrame.from_dict(dct)[['variable', 'unit']]
    npt.assert_array_equal(test_df.variables(include_units=True), exp)


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


def test_filter_error(test_df):
    pytest.raises(ValueError, test_df.filter, foo='foo')


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


def test_filter_as_kwarg(meta_df):
    obs = list(meta_df.filter(variable='Primary Energy|Coal').scenarios())
    assert obs == ['scen_a']


def test_filter_keep_false(meta_df):
    df = meta_df.filter(variable='Primary Energy|Coal', year=2005, keep=False)
    obs = df.data[df.data.scenario == 'scen_a'].value
    npt.assert_array_equal(obs, [1, 6, 3])


def test_filter_by_regexp(meta_df):
    obs = meta_df.filter(scenario='sce._a$', regexp=True)
    assert obs['scenario'].unique() == 'scen_a'


def test_timeseries(test_df):
    dct = {'model': ['model_a'] * 2, 'scenario': ['scen_a'] * 2,
           'years': [2005, 2010], 'value': [1, 6]}
    exp = pd.DataFrame(dct).pivot_table(index=['model', 'scenario'],
                                        columns=['years'], values='value')
    obs = test_df.filter(variable='Primary Energy').timeseries()
    npt.assert_array_equal(obs, exp)


def test_read_pandas():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'testing_data_2.csv'))
    assert list(df.variables()) == ['Primary Energy']


def test_filter_meta_index(meta_df):
    obs = meta_df.filter(scenario='scen_b').meta.index
    exp = pd.MultiIndex(levels=[['model_a'], ['scen_b']],
                        labels=[[0], [0]],
                        names=['model', 'scenario'])
    pd.testing.assert_index_equal(obs, exp)


def test_meta_idx(meta_df):
    # assert that the `drop_duplicates()` in `_meta_idx()` returns right length
    assert len(_meta_idx(meta_df.data)) == 2


def test_require_variable(meta_df):
    obs = meta_df.require_variable(variable='Primary Energy|Coal',
                                   exclude_on_fail=True)
    assert len(obs) == 1
    assert obs.loc[0, 'scenario'] == 'scen_b'

    assert list(meta_df['exclude']) == [False, True]


def test_require_variable_top_level(meta_df):
    obs = require_variable(meta_df, variable='Primary Energy|Coal',
                           exclude_on_fail=True)
    assert len(obs) == 1
    assert obs.loc[0, 'scenario'] == 'scen_b'

    assert list(meta_df['exclude']) == [False, True]


def test_validate_all_pass(meta_df):
    obs = meta_df.validate(
        {'Primary Energy': {'up': 10}}, exclude_on_fail=True)
    assert obs is None
    assert len(meta_df.data) == 6  # data unchanged

    assert list(meta_df['exclude']) == [False, False]  # none excluded


def test_validate_nonexisting(meta_df):
    obs = meta_df.validate({'Primary Energy|Coal': {'up': 2}},
                           exclude_on_fail=True)
    assert len(obs) == 1
    assert obs['scenario'].values[0] == 'scen_a'

    assert list(meta_df['exclude']) == [True, False]  # scenario with failed
    # validation excluded, scenario with non-defined value passes validation


def test_validate_up(meta_df):
    obs = meta_df.validate({'Primary Energy': {'up': 6.5}},
                           exclude_on_fail=False)
    assert len(obs) == 1
    assert obs['year'].values[0] == 2010

    assert list(meta_df['exclude']) == [False, False]  # assert none excluded


def test_validate_lo(meta_df):
    obs = meta_df.validate({'Primary Energy': {'up': 8, 'lo': 2.0}})
    assert len(obs) == 1
    assert obs['year'].values[0] == 2005
    assert list(obs['scenario'].values) == ['scen_a']


def test_validate_both(meta_df):
    obs = meta_df.validate({'Primary Energy': {'up': 6.5, 'lo': 2.0}})
    assert len(obs) == 2
    assert list(obs['year'].values) == [2005, 2010]
    assert list(obs['scenario'].values) == ['scen_a', 'scen_b']


def test_validate_year(meta_df):
    obs = meta_df.validate({'Primary Energy': {'up': 5.0, 'year': 2005}},
                           exclude_on_fail=False)
    assert obs is None

    obs = meta_df.validate({'Primary Energy': {'up': 5.0, 'year': 2010}},
                           exclude_on_fail=False)
    assert len(obs) == 2


def test_validate_exclude(meta_df):
    meta_df.validate({'Primary Energy': {'up': 6.0}}, exclude_on_fail=True)
    assert list(meta_df['exclude']) == [False, True]


def test_validate_top_level(meta_df):
    obs = validate(meta_df, criteria={'Primary Energy': {'up': 6.0}},
                   exclude_on_fail=True, variable='Primary Energy')
    assert len(obs) == 1
    assert obs['year'].values[0] == 2010
    assert list(meta_df['exclude']) == [False, True]


def test_category_none(meta_df):
    meta_df.categorize('category', 'Testing', {'Primary Energy': {'up': 0.8}})
    assert 'category' not in meta_df.meta.columns


def test_category_pass(meta_df):
    dct = {'model': ['model_a', 'model_a'],
           'scenario': ['scen_a', 'scen_b'],
           'category': ['foo', None]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    meta_df.categorize('category', 'foo', {'Primary Energy':
                                           {'up': 6, 'year': 2010}})
    obs = meta_df['category']
    pd.testing.assert_series_equal(obs, exp)


def test_category_top_level(meta_df):
    dct = {'model': ['model_a', 'model_a'],
           'scenario': ['scen_a', 'scen_b'],
           'category': ['foo', None]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    categorize(meta_df, 'category', 'foo',
               criteria={'Primary Energy': {'up': 6, 'year': 2010}},
               variable='Primary Energy')
    obs = meta_df['category']
    pd.testing.assert_series_equal(obs, exp)


def test_load_metadata(meta_df):
    meta_df.load_metadata(os.path.join(
        TEST_DATA_DIR, 'testing_metadata.xlsx'), sheet_name='meta')
    obs = meta_df.meta

    dct = {'model': ['model_a'] * 2, 'scenario': ['scen_a', 'scen_b'],
           'category': ['imported', np.nan], 'exclude': [False, False]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])
    pd.testing.assert_series_equal(obs['exclude'], exp['exclude'])
    pd.testing.assert_series_equal(obs['category'], exp['category'])


def test_load_SSP_database_downloaded_file(test_df_year):
    obs_df = IamDataFrame(os.path.join(
        TEST_DATA_DIR, 'test_SSP_database_raw_download.xlsx')
    )
    pd.testing.assert_frame_equal(obs_df.as_pandas(), test_df_year.as_pandas())


def test_load_RCP_database_downloaded_file(test_df_year):
    obs_df = IamDataFrame(os.path.join(
        TEST_DATA_DIR, 'test_RCP_database_raw_download.xlsx')
    )
    pd.testing.assert_frame_equal(obs_df.as_pandas(), test_df_year.as_pandas())


def test_interpolate(test_df_year):
    test_df_year.interpolate(2007)
    dct = {'model': ['a_model'] * 3, 'scenario': ['a_scenario'] * 3,
           'years': [2005, 2007, 2010], 'value': [1, 3, 6]}
    exp = pd.DataFrame(dct).pivot_table(index=['model', 'scenario'],
                                        columns=['years'], values='value')
    variable = {'variable': 'Primary Energy'}
    obs = test_df_year.filter(**variable).timeseries()
    npt.assert_array_equal(obs, exp)

    # redo the inpolation and check that no duplicates are added
    test_df_year.interpolate(2007)
    assert not test_df_year.filter(**variable).data.duplicated().any()


def test_set_meta_no_name(meta_df):
    idx = pd.MultiIndex(levels=[['a_scenario'], ['a_model'], ['some_region']],
                        labels=[[0], [0], [0]],
                        names=['scenario', 'model', 'region'])
    s = pd.Series(data=[0.3], index=idx)
    pytest.raises(ValueError, meta_df.set_meta, s)


def test_set_meta_as_named_series(meta_df):
    idx = pd.MultiIndex(levels=[['scen_a'], ['model_a'], ['some_region']],
                        labels=[[0], [0], [0]],
                        names=['scenario', 'model', 'region'])

    s = pd.Series(data=[0.3], index=idx)
    s.name = 'meta_values'
    meta_df.set_meta(s)

    idx = pd.MultiIndex(levels=[['model_a'], ['scen_a', 'scen_b']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])
    exp = pd.Series(data=[0.3, np.nan], index=idx)
    exp.name = 'meta_values'

    obs = meta_df['meta_values']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_unnamed_series(meta_df):
    idx = pd.MultiIndex(levels=[['scen_a'], ['model_a'], ['some_region']],
                        labels=[[0], [0], [0]],
                        names=['scenario', 'model', 'region'])

    s = pd.Series(data=[0.3], index=idx)
    meta_df.set_meta(s, name='meta_values')

    idx = pd.MultiIndex(levels=[['model_a'], ['scen_a', 'scen_b']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])
    exp = pd.Series(data=[0.3, np.nan], index=idx)
    exp.name = 'meta_values'

    obs = meta_df['meta_values']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_non_unique_index_fail(meta_df):
    idx = pd.MultiIndex(levels=[['model_a'], ['scen_a'], ['reg_a', 'reg_b']],
                        labels=[[0, 0], [0, 0], [0, 1]],
                        names=['model', 'scenario', 'region'])
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, meta_df.set_meta, s)


def test_set_meta_non_existing_index_fail(meta_df):
    idx = pd.MultiIndex(levels=[['model_a', 'fail_model'],
                                ['scen_a', 'fail_scenario']],
                        labels=[[0, 1], [0, 1]], names=['model', 'scenario'])
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, meta_df.set_meta, s)


def test_set_meta_by_df(meta_df):
    df = pd.DataFrame([
        ['model_a', 'scen_a', 'some_region', 1],
    ], columns=['model', 'scenario', 'region', 'col'])

    meta_df.set_meta(meta=0.3, name='meta_values', index=df)

    idx = pd.MultiIndex(levels=[['model_a'], ['scen_a', 'scen_b']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])
    exp = pd.Series(data=[0.3, np.nan], index=idx)
    exp.name = 'meta_values'

    obs = meta_df['meta_values']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_series(meta_df):
    s = pd.Series([0.3, 0.4])
    meta_df.set_meta(s, 'meta_series')

    idx = pd.MultiIndex(levels=[['model_a'],
                                ['scen_a', 'scen_b']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])

    exp = pd.Series(data=[0.3, 0.4], index=idx)
    exp.name = 'meta_series'

    obs = meta_df['meta_series']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_int(meta_df):
    meta_df.set_meta(3.2, 'meta_int')

    idx = pd.MultiIndex(levels=[['model_a'],
                                ['scen_a', 'scen_b']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])

    exp = pd.Series(data=[3.2, 3.2], index=idx, name='meta_int')

    obs = meta_df['meta_int']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_str(meta_df):
    meta_df.set_meta('testing', name='meta_str')

    idx = pd.MultiIndex(levels=[['model_a'],
                                ['scen_a', 'scen_b']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])

    exp = pd.Series(data=['testing', 'testing'], index=idx, name='meta_str')

    obs = meta_df['meta_str']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_str_list(meta_df):
    meta_df.set_meta(['testing', 'testing2'], name='category')
    obs = meta_df.filter(category='testing')
    assert obs['scenario'].unique() == 'scen_a'


def test_set_meta_as_str_by_index(meta_df):
    idx = pd.MultiIndex(levels=[['model_a'], ['scen_a']],
                        labels=[[0], [0]], names=['model', 'scenario'])

    meta_df.set_meta('foo', 'meta_str', idx)

    obs = pd.Series(meta_df['meta_str'].values)
    pd.testing.assert_series_equal(obs, pd.Series(['foo', None]))


def test_filter_by_bool(meta_df):
    meta_df.set_meta([True, False], name='exclude')
    obs = meta_df.filter(exclude=True)
    assert obs['scenario'].unique() == 'scen_a'


def test_filter_by_int(meta_df):
    meta_df.set_meta([1, 2], name='test')
    obs = meta_df.filter(test=[1, 3])
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


def test_pd_filter_by_meta(meta_df):
    data = df_filter_by_meta_matching_idx.set_index(['model', 'region'])

    meta_df.set_meta([True, False], 'boolean')
    meta_df.set_meta(0, 'integer')

    obs = filter_by_meta(data, meta_df, join_meta=True,
                         boolean=True, integer=None)
    obs = obs.reindex(columns=['scenario', 'col', 'boolean', 'integer'])

    exp = data.iloc[0:2].copy()
    exp['boolean'] = True
    exp['integer'] = 0

    pd.testing.assert_frame_equal(obs, exp)


def test_pd_filter_by_meta_no_index(meta_df):
    data = df_filter_by_meta_matching_idx

    meta_df.set_meta([True, False], 'boolean')
    meta_df.set_meta(0, 'int')

    obs = filter_by_meta(data, meta_df, join_meta=True,
                         boolean=True, int=None)
    obs = obs.reindex(columns=META_IDX + ['region', 'col', 'boolean', 'int'])

    exp = data.iloc[0:2].copy()
    exp['boolean'] = True
    exp['int'] = 0

    pd.testing.assert_frame_equal(obs, exp)


def test_pd_filter_by_meta_nonmatching_index(meta_df):
    data = df_filter_by_meta_nonmatching_idx
    meta_df.set_meta(['a', 'b'], 'string')

    obs = filter_by_meta(data, meta_df, join_meta=True, string='b')
    obs = obs.reindex(columns=['scenario', 2010, 2020, 'string'])

    exp = data.iloc[2:3].copy()
    exp['string'] = 'b'

    pd.testing.assert_frame_equal(obs, exp)


def test_pd_join_by_meta_nonmatching_index(meta_df):
    data = df_filter_by_meta_nonmatching_idx
    meta_df.set_meta(['a', 'b'], 'string')

    obs = filter_by_meta(data, meta_df, join_meta=True, string=None)
    obs = obs.reindex(columns=['scenario', 2010, 2020, 'string'])

    exp = data.copy()
    exp['string'] = [np.nan, np.nan, 'b']

    pd.testing.assert_frame_equal(obs.sort_index(level=1), exp)


def test_concat_fails_iter():
    pytest.raises(TypeError, concat, 1)


def test_concat_fails_notdf():
    pytest.raises(TypeError, concat, 'foo')


def test_concat(meta_df):
    left = IamDataFrame(meta_df.data.copy())
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


def test_normalize(meta_df):
    exp = meta_df.data.copy().reset_index(drop=True)
    exp['value'][1::2] /= exp['value'][::2].values
    exp['value'][::2] /= exp['value'][::2].values
    obs = meta_df.normalize(year=2005).data.reset_index(drop=True)
    pd.testing.assert_frame_equal(obs, exp)


def test_normalize_not_time(meta_df):
    pytest.raises(ValueError, meta_df.normalize, variable='foo')
    pytest.raises(ValueError, meta_df.normalize, year=2015, variable='foo')


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("drop", [True, False])
def test_swap_time_to_year(test_df, inplace, drop):
    if "year" in test_df.data:
        pytest.skip("Can't test conversion if we already have year")

    obs = test_df.swap_time_for_year(inplace=inplace, drop=drop)

    exp = test_df.data
    exp["year"] = exp["time"].apply(lambda x: x.year)
    if drop:
        exp = exp.drop("time", axis="columns")
    exp = IamDataFrame(exp)

    if inplace:
        assert obs is None
        assert compare(test_df, exp).empty
    else:
        assert compare(obs, exp).empty
        assert not compare(test_df, exp).empty
