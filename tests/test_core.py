import os
import copy
import pytest

import numpy as np
import pandas as pd
from numpy import testing as npt

from pyam_analysis import IamDataFrame, plotting, validate, categorize

from testing_utils import here, meta_df, test_df, TEST_DF


def test_model(test_df):
    assert test_df['model'].unique() == ['a_model']


def test_scenario(test_df):
    assert test_df['scenario'].unique() == ['a_scenario']


def test_region(test_df):
    assert test_df['region'].unique() == ['World']


def test_variable(test_df):
    assert list(test_df['variable'].unique()) == ['Primary Energy',
                                                  'Primary Energy|Coal']


def test_variable_depth_0(test_df):
    obs = list(test_df.filter({'level': 0})['variable'].unique())
    exp = ['Primary Energy']
    assert obs == exp


def test_variable_depth_0_minus(test_df):
    obs = list(test_df.filter({'level': '0-'})['variable'].unique())
    exp = ['Primary Energy']
    assert obs == exp


def test_variable_depth_0_plus(test_df):
    obs = list(test_df.filter({'level': '0+'})['variable'].unique())
    exp = ['Primary Energy', 'Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_1(test_df):
    obs = list(test_df.filter({'level': 1})['variable'].unique())
    exp = ['Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_1_minus(test_df):
    obs = list(test_df.filter({'level': '1-'})['variable'].unique())
    exp = ['Primary Energy', 'Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_1_plus(test_df):
    obs = list(test_df.filter({'level': '1+'})['variable'].unique())
    exp = ['Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_raises(test_df):
    pytest.raises(ValueError, test_df.filter, {'level': '1/'})


def test_variable_unit(test_df):
    dct = {'variable': ['Primary Energy', 'Primary Energy|Coal'],
           'unit': ['EJ/y', 'EJ/y']}
    cols = ['variable', 'unit']
    exp = pd.DataFrame.from_dict(dct)[cols]
    npt.assert_array_equal(test_df[cols].drop_duplicates(), exp)


def test_timeseries(test_df):
    dct = {'model': ['a_model'] * 2, 'scenario': ['a_scenario'] * 2,
           'years': [2005, 2010], 'value': [1, 6]}
    exp = pd.DataFrame(dct).pivot_table(index=['model', 'scenario'],
                                        columns=['years'], values='value')
    obs = test_df.filter({'variable': 'Primary Energy'}).timeseries()
    npt.assert_array_equal(obs, exp)


def test_read_pandas():
    ia = IamDataFrame(os.path.join(here, 'testing_data_2.csv'))
    assert list(ia['variable'].unique()) == ['Primary Energy']


def test_validate_none(meta_df):
    obs = meta_df.validate(
        {'Primary Energy': {'up': 10}}, exclude=True)
    assert obs is None
    assert len(meta_df.data) == 6  # data unchanged

    assert list(meta_df['exclude']) == [False, False]  # none excluded


def test_validate_null(meta_df):
    obs = meta_df.validate({'Secondary Energy': {'up': 10}}, exclude=True)
    assert obs is None


def test_validate_up(meta_df):
    obs = meta_df.validate({'Primary Energy': {'up': 6.5}}, exclude=False)
    assert len(obs) == 1
    assert obs['year'].values[0] == 2010


def test_validate_lo(meta_df):
    obs = meta_df.validate({'Primary Energy': {'lo': 2.0}}, exclude=False)
    assert len(obs) == 1
    assert obs['year'].values[0] == 2005


def test_validate_year(meta_df):
    obs = meta_df.validate({'Primary Energy': {'up': 5.0, 'year': 2005}},
                           exclude=False)
    assert obs is None

    obs = meta_df.validate({'Primary Energy': {'up': 5.0, 'year': 2010}},
                           exclude=False)
    assert len(obs) == 2


def test_validate_exclude(meta_df):
    obs = meta_df.validate({'Primary Energy': {'up': 6.0}}, exclude=True)
    assert list(meta_df['exclude']) == [False, True]


def test_validate_top_level(meta_df):
    obs = validate(meta_df,
                   filters={'variable': 'Primary Energy'},
                   criteria={'Primary Energy': {'up': 6.0}},
                   exclude=True)
    assert len(obs) == 1
    assert obs['year'].values[0] == 2010
    assert list(meta_df['exclude']) == [False, True]


def test_category_none(meta_df):
    meta_df.categorize('category', 'Testing', {'Primary Energy': {'up': 0.8}})
    obs = meta_df['category'].values
    # old usage when default was None
    # exp = [np.nan, np.nan]
    # assert list(obs) == exp
    assert np.isnan(obs).all()


def test_category_pass(meta_df):
    dct = {'model': ['a_model', 'a_model'],
           'scenario': ['a_scenario', 'a_scenario2'],
           'category': ['Testing', np.nan]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    meta_df.categorize('category', 'Testing', {'Primary Energy': {'up': 6}})
    obs = meta_df['category']
    pd.testing.assert_series_equal(obs, exp)


def test_category_top_level(meta_df):
    dct = {'model': ['a_model', 'a_model'],
           'scenario': ['a_scenario', 'a_scenario2'],
           'category': ['Testing', np.nan]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    categorize(meta_df, 'category', 'Testing',
               criteria={'Primary Energy': {'up': 6}},
               filters={'variable': 'Primary Energy'})
    obs = meta_df['category']
    pd.testing.assert_series_equal(obs, exp)


def test_load_metadata(test_df):
    test_df.load_metadata(os.path.join(
        here, 'testing_metadata.xlsx'), sheet_name='metadata')

    obs = test_df.meta
    dct = {'model': ['a_model'], 'scenario': ['a_scenario'],
           'category': ['imported']}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])
    pd.testing.assert_series_equal(obs['category'], exp['category'])


def test_append(test_df):
    df2 = test_df.append(other=os.path.join(here, 'testing_data_2.csv'))

    obs = test_df['scenario'].unique()
    exp = ['a_scenario']
    npt.assert_array_equal(obs, exp)

    obs = df2['scenario'].unique()
    exp = ['a_scenario', 'append_scenario']
    npt.assert_array_equal(obs, exp)


def test_append_duplicates(test_df):
    other = copy.deepcopy(test_df)
    pytest.raises(ValueError, test_df.append, other=other)


def test_interpolate(test_df):
    test_df.interpolate(2007)
    dct = {'model': ['a_model'] * 3, 'scenario': ['a_scenario'] * 3,
           'years': [2005, 2007, 2010], 'value': [1, 3, 6]}
    exp = pd.DataFrame(dct).pivot_table(index=['model', 'scenario'],
                                        columns=['years'], values='value')
    obs = test_df.filter({'variable': 'Primary Energy'}).timeseries()
    npt.assert_array_equal(obs, exp)


def test_add_metadata_as_named_series(meta_df):
    idx = pd.MultiIndex(levels=[['a_model'], ['a_scenario']],
                        labels=[[0], [0]], names=['model', 'scenario'])

    s = pd.Series(data=[0.3], index=idx)
    s.name = 'meta_values'
    meta_df.metadata(s)

    idx = pd.MultiIndex(levels=[['a_model'],
                                ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])
    exp = pd.Series(data=[0.3, np.nan], index=idx)
    exp.name = 'meta_values'

    obs = meta_df['meta_values']
    pd.testing.assert_series_equal(obs, exp)


def test_add_metadata_index_fail(meta_df):
    idx = pd.MultiIndex(levels=[['a_model', 'fail_model'],
                                ['a_scenario', 'fail_scenario']],
                        labels=[[0, 1], [0, 1]], names=['model', 'scenario'])
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, meta_df.metadata, s)


def test_add_metadata_as_series(meta_df):
    s = pd.Series([0.3, 0.4])
    meta_df.metadata(s, 'meta_series')

    idx = pd.MultiIndex(levels=[['a_model'],
                                ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])

    exp = pd.Series(data=[0.3, 0.4], index=idx)
    exp.name = 'meta_series'

    obs = meta_df['meta_series']
    pd.testing.assert_series_equal(obs, exp)


def test_add_metadata_as_int(meta_df):
    meta_df.metadata(3.2, 'meta_int')

    idx = pd.MultiIndex(levels=[['a_model'],
                                ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])

    exp = pd.Series(data=[3.2, 3.2], index=idx)
    exp.name = 'meta_int'

    obs = meta_df['meta_int']
    pd.testing.assert_series_equal(obs, exp)


def test_filter_by_metadata_bool(meta_df):
    meta_df.metadata([True, False], name='exclude')
    obs = meta_df.filter({'exclude': True})
    assert obs['scenario'].unique() == 'a_scenario'


def test_filter_by_metadata_int(meta_df):
    meta_df.metadata([1, 2], name='value')
    obs = meta_df.filter({'value': [1, 3]})
    assert obs['scenario'].unique() == 'a_scenario'
