import pytest
import numpy as np
import pandas as pd


EXP_IDX = pd.MultiIndex(levels=[['model_a'], ['scen_a', 'scen_b']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])


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

    s = pd.Series(data=[0.3], index=idx, name='meta_values')
    meta_df.set_meta(s)

    exp = pd.Series(data=[0.3, np.nan], index=EXP_IDX, name='meta_values')
    pd.testing.assert_series_equal(meta_df['meta_values'], exp)


def test_set_meta_as_unnamed_series(meta_df):
    idx = pd.MultiIndex(levels=[['scen_a'], ['model_a'], ['some_region']],
                        labels=[[0], [0], [0]],
                        names=['scenario', 'model', 'region'])

    s = pd.Series(data=[0.3], index=idx)
    meta_df.set_meta(s, name='meta_values')

    exp = pd.Series(data=[0.3, np.nan], index=EXP_IDX, name='meta_values')
    pd.testing.assert_series_equal(meta_df['meta_values'], exp)


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

    exp = pd.Series(data=[0.3, np.nan], index=EXP_IDX, name='meta_values')
    pd.testing.assert_series_equal(meta_df['meta_values'], exp)


def test_set_meta_as_series(meta_df):
    s = pd.Series([0.3, 0.4])
    meta_df.set_meta(s, 'meta_series')

    exp = pd.Series(data=[0.3, 0.4], index=EXP_IDX, name='meta_series')
    pd.testing.assert_series_equal(meta_df['meta_series'], exp)


def test_set_meta_as_int(meta_df):
    meta_df.set_meta(3.2, 'meta_int')

    exp = pd.Series(data=[3.2, 3.2], index=EXP_IDX, name='meta_int')

    obs = meta_df['meta_int']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_str(meta_df):
    meta_df.set_meta('testing', name='meta_str')

    exp = pd.Series(data=['testing'] * 2, index=EXP_IDX, name='meta_str')
    pd.testing.assert_series_equal(meta_df['meta_str'], exp)


def test_set_meta_as_str_list(meta_df):
    meta_df.set_meta(['testing', 'testing2'], name='category')
    obs = meta_df.filter(category='testing')
    assert obs['scenario'].unique() == 'scen_a'


def test_set_meta_as_str_by_index(meta_df):
    idx = pd.MultiIndex(levels=[['model_a'], ['scen_a']],
                        labels=[[0], [0]], names=['model', 'scenario'])

    meta_df.set_meta('foo', 'meta_str', idx)

    exp = pd.Series(data=['foo', None], index=EXP_IDX, name='meta_str')
    pd.testing.assert_series_equal(meta_df['meta_str'], exp)


def test_set_meta_from_data(meta_df):
    meta_df.set_meta_from_data('pe_2005', variable='Primary Energy', year=2005)
    exp = pd.Series(data=[1., 2.], index=EXP_IDX, name='pe_2005')
    pd.testing.assert_series_equal(meta_df['pe_2005'], exp)


def test_set_meta_from_data_max(meta_df):
    meta_df.set_meta_from_data('pe_max_yr', variable='Primary Energy',
                               method=np.max)
    exp = pd.Series(data=[6., 7.], index=EXP_IDX, name='pe_max_yr')
    pd.testing.assert_series_equal(meta_df['pe_max_yr'], exp)


def test_set_meta_from_data_mean(meta_df):
    meta_df.set_meta_from_data('pe_mean', variable='Primary Energy',
                               method=np.mean)
    exp = pd.Series(data=[3.5, 4.5], index=EXP_IDX, name='pe_mean')
    pd.testing.assert_series_equal(meta_df['pe_mean'], exp)


def test_set_meta_from_data_method_other_column(meta_df):
    if 'year' in meta_df.data.columns:
        col, value = 'year', 2010
    else:
        col, value = 'time', pd.to_datetime('2010-07-21T00:00:00.0')
    meta_df.set_meta_from_data('pe_max_yr', variable='Primary Energy',
                               method=np.max, column=col)
    exp = pd.Series(data=[value] * 2, index=EXP_IDX, name='pe_max_yr')
    pd.testing.assert_series_equal(meta_df['pe_max_yr'], exp)


def test_set_meta_from_data_nonunique(meta_df):
    # the filtered `data` dataframe is not unique with regard to META_IDX
    pytest.raises(ValueError, meta_df.set_meta_from_data, 'fail',
                  variable='Primary Energy')
