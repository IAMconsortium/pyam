import os
import copy
import pytest
import re
import datetime

import numpy as np
import pandas as pd
from numpy import testing as npt

from pyam import IamDataFrame, plotting, validate, categorize, \
    require_variable, check_aggregate, filter_by_meta, META_IDX, IAMC_IDX
from pyam.core import _meta_idx

from conftest import TEST_DATA_DIR


df_filter_by_meta_matching_idx = pd.DataFrame([
    ['a_model', 'a_scenario', 'a_region1', 1],
    ['a_model', 'a_scenario', 'a_region2', 2],
    ['a_model', 'a_scenario2', 'a_region3', 3],
], columns=['model', 'scenario', 'region', 'col'])


df_filter_by_meta_nonmatching_idx = pd.DataFrame([
    ['a_model', 'a_scenario3', 'a_region1', 1, 2],
    ['a_model', 'a_scenario3', 'a_region2', 2, 3],
    ['a_model', 'a_scenario2', 'a_region3', 3, 4],
], columns=['model', 'scenario', 'region', 2010, 2020]
).set_index(['model', 'region'])


def test_init_df_with_index(test_pd_df):
    df = IamDataFrame(test_pd_df.set_index(META_IDX))
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), test_pd_df)


def test_init_df_with_float_cols_raises(test_pd_df):
    _test_df = test_pd_df.rename(columns={2005: 2005.5, 2010: 2010.})
    pytest.raises(ValueError, IamDataFrame, data=_test_df)


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
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), tdf, check_like=True)

    exp_meta = pd.DataFrame([
        ['b_model', 'b_scen', extra_value, False],
        ['b_model', 'a_scenario2', extra_value, False],
    ], columns=['model', 'scenario', extra_col, 'exclude']
    ).set_index(['model', 'scenario', extra_col])
    pd.testing.assert_frame_equal(df.meta, exp_meta)


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
    assert test_df['model'].unique() == ['a_model']


def test_model(test_df):
    pd.testing.assert_series_equal(test_df.models(),
                                   pd.Series(data=['a_model'], name='model'))


def test_scenario(test_df):
    exp = pd.Series(data=['a_scenario'], name='scenario')
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


def test_variable_depth_0(test_df):
    obs = list(test_df.filter(level=0)['variable'].unique())
    exp = ['Primary Energy']
    assert obs == exp


def test_variable_depth_0_keep_false(test_df):
    obs = list(test_df.filter(level=0, keep=False)['variable'].unique())
    exp = ['Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_0_minus(test_df):
    obs = list(test_df.filter(level='0-')['variable'].unique())
    exp = ['Primary Energy']
    assert obs == exp


def test_variable_depth_0_plus(test_df):
    obs = list(test_df.filter(level='0+')['variable'].unique())
    exp = ['Primary Energy', 'Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_1(test_df):
    obs = list(test_df.filter(level=1)['variable'].unique())
    exp = ['Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_1_minus(test_df):
    obs = list(test_df.filter(level='1-')['variable'].unique())
    exp = ['Primary Energy', 'Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_1_plus(test_df):
    obs = list(test_df.filter(level='1+')['variable'].unique())
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


@pytest.mark.parametrize("test_month", [6, "June", "Jun", "jun", ["Jun", "jun"]])
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


@pytest.mark.parametrize("test_day", [17, "Fri", "Friday", "friday", ["Fri", "fri"]])
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
        expected_rows = test_df.data["time"].apply(lambda x: x.hour).isin(test_hour)
        expected = test_df.data["time"].loc[expected_rows].unique()

        unique_time = obs['time'].unique()
        npt.assert_array_equal(unique_time, expected)


def test_filter_time_exact_match(test_df):
    if "year" in test_df.data.columns:
        error_msg = re.escape("`year` can only be filtered with ints or lists of ints")
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
    error_msg = re.escape(
        "'datetime.datetime' object cannot be interpreted as an integer"
    )
    with pytest.raises(TypeError, match=error_msg):
        obs = test_df.filter(year=range(
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
            obs = test_df.filter(month=month_range)
    else:
        error_msg = re.escape("string ranges must lead to increasing integer ranges, Nov-Feb becomes [11, 2]")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_df.filter(month=month_range)


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

        expected_rows = test_df.data["time"].apply(lambda x: x.hour).isin(hour_range)
        expected = test_df.data["time"].loc[expected_rows].unique()

        unique_time = obs['time'].unique()
        npt.assert_array_equal(unique_time, expected)


def test_filter_time_no_match(test_df):
    if "year" in test_df.data.columns:
        error_msg = re.escape("`year` can only be filtered with ints or lists of ints")
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(year=datetime.datetime(2004, 6, 18))
    else:
        obs = test_df.filter(time=datetime.datetime(2004, 6, 18))
        assert obs.data.empty


def test_filter_time_not_datetime_error(test_df):
    if "year" in test_df.data.columns:
        with pytest.raises(KeyError, match=re.escape("'time")):
            test_df.filter(time=datetime.datetime(2004, 6, 18))
    else:
        error_msg = re.escape("`time` can only be filtered with datetimes or lists of datetimes")
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(time=2005)


def test_filter_time_not_datetime_range_error(test_df):
    if "year" in test_df.data.columns:
        with pytest.raises(KeyError, match=re.escape("'time")):
            test_df.filter(time=range(2000, 2008))
    else:
        error_msg = re.escape("`time` can only be filtered with datetimes or lists of datetimes")
        with pytest.raises(TypeError, match=error_msg):
            test_df.filter(time=range(2000, 2008))


def test_filter_as_kwarg(meta_df):
    obs = list(meta_df.filter(variable='Primary Energy|Coal').scenarios())
    assert obs == ['a_scenario']


def test_filter_keep_false(meta_df):
    df = meta_df.filter(variable='Primary Energy|Coal', year=2005, keep=False)
    obs = df.data[df.data.scenario == 'a_scenario'].value
    npt.assert_array_equal(obs, [1, 6, 3])


def test_filter_by_regexp(meta_df):
    obs = meta_df.filter(scenario='a_scenari.$', regexp=True)
    assert obs['scenario'].unique() == 'a_scenario'


def test_timeseries(test_df):
    dct = {'model': ['a_model'] * 2, 'scenario': ['a_scenario'] * 2,
           'years': [2005, 2010], 'value': [1, 6]}
    exp = pd.DataFrame(dct).pivot_table(index=['model', 'scenario'],
                                        columns=['years'], values='value')
    obs = test_df.filter(variable='Primary Energy').timeseries()
    npt.assert_array_equal(obs, exp)


def test_read_pandas():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'testing_data_2.csv'))
    assert list(df.variables()) == ['Primary Energy']


def test_filter_meta_index(meta_df):
    obs = meta_df.filter(scenario='a_scenario2').meta.index
    exp = pd.MultiIndex(levels=[['a_model'], ['a_scenario2']],
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
    assert obs.loc[0, 'scenario'] == 'a_scenario2'

    assert list(meta_df['exclude']) == [False, True]


def test_require_variable_top_level(meta_df):
    obs = require_variable(meta_df, variable='Primary Energy|Coal',
                           exclude_on_fail=True)
    assert len(obs) == 1
    assert obs.loc[0, 'scenario'] == 'a_scenario2'

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
    assert obs['scenario'].values[0] == 'a_scenario'

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
    assert list(obs['scenario'].values) == ['a_scenario']


def test_validate_both(meta_df):
    obs = meta_df.validate({'Primary Energy': {'up': 6.5, 'lo': 2.0}})
    assert len(obs) == 2
    assert list(obs['year'].values) == [2005, 2010]
    assert list(obs['scenario'].values) == ['a_scenario', 'a_scenario2']


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
    dct = {'model': ['a_model', 'a_model'],
           'scenario': ['a_scenario', 'a_scenario2'],
           'category': ['foo', None]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    meta_df.categorize('category', 'foo', {'Primary Energy':
                                           {'up': 6, 'year': 2010}})
    obs = meta_df['category']
    pd.testing.assert_series_equal(obs, exp)


def test_category_top_level(meta_df):
    dct = {'model': ['a_model', 'a_model'],
           'scenario': ['a_scenario', 'a_scenario2'],
           'category': ['Testing', None]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    categorize(meta_df, 'category', 'Testing',
               criteria={'Primary Energy': {'up': 6, 'year': 2010}},
               variable='Primary Energy')
    obs = meta_df['category']
    pd.testing.assert_series_equal(obs, exp)


def test_load_metadata(meta_df):
    meta_df.load_metadata(os.path.join(
        TEST_DATA_DIR, 'testing_metadata.xlsx'), sheet_name='meta')
    obs = meta_df.meta

    dct = {'model': ['a_model'] * 2, 'scenario': ['a_scenario', 'a_scenario2'],
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


def test_append_other_scenario(meta_df):
    other = meta_df.filter(scenario='a_scenario2')\
        .rename({'scenario': {'a_scenario2': 'a_scenario3'}})

    meta_df.set_meta([0, 1], name='col1')
    meta_df.set_meta(['a', 'b'], name='col2')

    other.set_meta(2, name='col1')
    other.set_meta('x', name='col3')

    df = meta_df.append(other)

    # check that the original meta dataframe is not updated
    obs = meta_df.meta.index.get_level_values(1)
    npt.assert_array_equal(obs, ['a_scenario', 'a_scenario2'])

    # assert that merging of meta works as expected
    exp = pd.DataFrame([
        ['a_model', 'a_scenario', False, 0, 'a', np.nan],
        ['a_model', 'a_scenario2', False, 1, 'b', np.nan],
        ['a_model', 'a_scenario3', False, 2, np.nan, 'x'],
    ], columns=['model', 'scenario', 'exclude', 'col1', 'col2', 'col3']
    ).set_index(['model', 'scenario'])

    # sort columns for assertion in older pandas versions
    df.meta = df.meta.reindex(columns=exp.columns)
    pd.testing.assert_frame_equal(df.meta, exp)

    # assert that appending data works as expected
    ts = df.timeseries()
    npt.assert_array_equal(ts.iloc[2].values, ts.iloc[3].values)


def test_append_same_scenario(meta_df):
    other = meta_df.filter(scenario='a_scenario2')\
        .rename({'variable': {'Primary Energy': 'Primary Energy clone'}})

    meta_df.set_meta([0, 1], name='col1')

    other.set_meta(2, name='col1')
    other.set_meta('b', name='col2')

    # check that non-matching meta raise an error
    pytest.raises(ValueError, meta_df.append, other=other)

    # check that ignoring meta conflict works as expetced
    df = meta_df.append(other, ignore_meta_conflict=True)

    # check that the new meta.index is updated, but not the original one
    npt.assert_array_equal(meta_df.meta.columns, ['exclude', 'col1'])

    # assert that merging of meta works as expected
    exp = meta_df.meta.copy()
    exp['col2'] = [np.nan, 'b']
    pd.testing.assert_frame_equal(df.meta, exp)

    # assert that appending data works as expected
    ts = df.timeseries()
    npt.assert_array_equal(ts.iloc[2], ts.iloc[3])


def test_append_duplicates(test_df_year):
    other = copy.deepcopy(test_df_year)
    pytest.raises(ValueError, test_df_year.append, other=other)


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
    idx = pd.MultiIndex(levels=[['a_scenario'], ['a_model'], ['a_region']],
                        labels=[[0], [0], [0]],
                        names=['scenario', 'model', 'region'])
    s = pd.Series(data=[0.3], index=idx)
    pytest.raises(ValueError, meta_df.set_meta, s)


def test_set_meta_as_named_series(meta_df):
    idx = pd.MultiIndex(levels=[['a_scenario'], ['a_model'], ['a_region']],
                        labels=[[0], [0], [0]],
                        names=['scenario', 'model', 'region'])

    s = pd.Series(data=[0.3], index=idx)
    s.name = 'meta_values'
    meta_df.set_meta(s)

    idx = pd.MultiIndex(levels=[['a_model'], ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])
    exp = pd.Series(data=[0.3, np.nan], index=idx)
    exp.name = 'meta_values'

    obs = meta_df['meta_values']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_unnamed_series(meta_df):
    idx = pd.MultiIndex(levels=[['a_scenario'], ['a_model'], ['a_region']],
                        labels=[[0], [0], [0]],
                        names=['scenario', 'model', 'region'])

    s = pd.Series(data=[0.3], index=idx)
    meta_df.set_meta(s, name='meta_values')

    idx = pd.MultiIndex(levels=[['a_model'], ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])
    exp = pd.Series(data=[0.3, np.nan], index=idx)
    exp.name = 'meta_values'

    obs = meta_df['meta_values']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_non_unique_index_fail(meta_df):
    idx = pd.MultiIndex(levels=[['a_model'], ['a_scenario'], ['a', 'b']],
                        labels=[[0, 0], [0, 0], [0, 1]],
                        names=['model', 'scenario', 'region'])
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, meta_df.set_meta, s)


def test_set_meta_non_existing_index_fail(meta_df):
    idx = pd.MultiIndex(levels=[['a_model', 'fail_model'],
                                ['a_scenario', 'fail_scenario']],
                        labels=[[0, 1], [0, 1]], names=['model', 'scenario'])
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, meta_df.set_meta, s)


def test_set_meta_by_df(meta_df):
    df = pd.DataFrame([
        ['a_model', 'a_scenario', 'a_region1', 1],
    ], columns=['model', 'scenario', 'region', 'col'])

    meta_df.set_meta(meta=0.3, name='meta_values', index=df)

    idx = pd.MultiIndex(levels=[['a_model'], ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])
    exp = pd.Series(data=[0.3, np.nan], index=idx)
    exp.name = 'meta_values'

    obs = meta_df['meta_values']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_series(meta_df):
    s = pd.Series([0.3, 0.4])
    meta_df.set_meta(s, 'meta_series')

    idx = pd.MultiIndex(levels=[['a_model'],
                                ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])

    exp = pd.Series(data=[0.3, 0.4], index=idx)
    exp.name = 'meta_series'

    obs = meta_df['meta_series']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_int(meta_df):
    meta_df.set_meta(3.2, 'meta_int')

    idx = pd.MultiIndex(levels=[['a_model'],
                                ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])

    exp = pd.Series(data=[3.2, 3.2], index=idx, name='meta_int')

    obs = meta_df['meta_int']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_str(meta_df):
    meta_df.set_meta('testing', name='meta_str')

    idx = pd.MultiIndex(levels=[['a_model'],
                                ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])

    exp = pd.Series(data=['testing', 'testing'], index=idx, name='meta_str')

    obs = meta_df['meta_str']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_str_list(meta_df):
    meta_df.set_meta(['testing', 'testing2'], name='category')
    obs = meta_df.filter(category='testing')
    assert obs['scenario'].unique() == 'a_scenario'


def test_set_meta_as_str_by_index(meta_df):
    idx = pd.MultiIndex(levels=[['a_model'], ['a_scenario']],
                        labels=[[0], [0]], names=['model', 'scenario'])

    meta_df.set_meta('foo', 'meta_str', idx)

    obs = pd.Series(meta_df['meta_str'].values)
    pd.testing.assert_series_equal(obs, pd.Series(['foo', None]))


def test_filter_by_bool(meta_df):
    meta_df.set_meta([True, False], name='exclude')
    obs = meta_df.filter(exclude=True)
    assert obs['scenario'].unique() == 'a_scenario'


def test_filter_by_int(meta_df):
    meta_df.set_meta([1, 2], name='test')
    obs = meta_df.filter(test=[1, 3])
    assert obs['scenario'].unique() == 'a_scenario'


def _r5_regions_exp(df):
    df = df.filter(region='World', keep=False)
    df['region'] = 'R5MAF'
    return df.data.reset_index(drop=True)


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
    )).data.reset_index(drop=True)

    df = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'R5MAF', 'var', 'unit', 1, 6],
        ['model', 'scen1', 'R5MAF', 'var', 'unit', 2, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))
    obs = df.map_regions('iso', region_col='r5_region').data
    obs = obs[obs.region.isin(['SSD', 'SDN'])].reset_index(drop=True)

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


def test_rename_variable():
    df = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'SST', 'test_1', 'unit', 1, 5],
        ['model', 'scen', 'SDN', 'test_2', 'unit', 2, 6],
        ['model', 'scen', 'SST', 'test_3', 'unit', 3, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))

    mapping = {'variable': {'test_1': 'test', 'test_3': 'test'}}

    obs = df.rename(mapping).data.reset_index(drop=True)

    exp = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'SST', 'test', 'unit', 4, 12],
        ['model', 'scen', 'SDN', 'test_2', 'unit', 2, 6],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    )).data.sort_values(by='region').reset_index(drop=True)

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_rename_index_fail(meta_df):
    mapping = {'scenario': {'a_scenario': 'a_scenario2'}}
    pytest.raises(ValueError, meta_df.rename, mapping)


def test_rename_index(meta_df):
    mapping = {'model': {'a_model': 'b_model'},
               'scenario': {'a_scenario': 'b_scen'}}
    obs = meta_df.rename(mapping)

    # test data changes
    exp = pd.DataFrame([
        ['b_model', 'b_scen', 'World', 'Primary Energy', 'EJ/y', 1., 6.],
        ['b_model', 'b_scen', 'World', 'Primary Energy|Coal', 'EJ/y', .5, 3.],
        ['b_model', 'a_scenario2', 'World', 'Primary Energy', 'EJ/y', 2., 7.],
    ], columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010]
    ).set_index(IAMC_IDX).sort_index()
    exp.columns = exp.columns.map(int)
    pd.testing.assert_frame_equal(obs.timeseries().sort_index(), exp)

    # test meta changes
    exp = pd.DataFrame([
        ['b_model', 'b_scen', False],
        ['b_model', 'a_scenario2', False],
    ], columns=['model', 'scenario', 'exclude']
    ).set_index(META_IDX)
    pd.testing.assert_frame_equal(obs.meta, exp)


def test_convert_unit():
    df = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'SST', 'test_1', 'A', 1, 5],
        ['model', 'scen', 'SDN', 'test_2', 'unit', 2, 6],
        ['model', 'scen', 'SST', 'test_3', 'C', 3, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))

    unit_conv = {'A': ['B', 5], 'C': ['D', 3]}

    obs = df.convert_unit(unit_conv).data.reset_index(drop=True)

    exp = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'SST', 'test_1', 'B', 5, 25],
        ['model', 'scen', 'SDN', 'test_2', 'unit', 2, 6],
        ['model', 'scen', 'SST', 'test_3', 'D', 9, 21],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    )).data.reset_index(drop=True)

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
