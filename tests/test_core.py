import os
import copy
import pytest

import pandas as pd
from numpy import testing as npt

from pyam_analysis import IamDataFrame, plotting, validate, categorize

from testing_utils import here, test_df, TEST_DF


def test_model(test_df):
    assert test_df['model'].unique() == ['a_model']


def test_scenario(test_df):
    assert test_df['scenario'].unique() == ['a_scenario']


def test_region(test_df):
    assert test_df['region'].unique() == ['World']


def test_variable(test_df):
    assert list(test_df['variable'].unique()) == ['Primary Energy',
                                                  'Primary Energy|Coal']


def test_variable_depth(test_df):
    obs = list(test_df.filter({'level': 0})['variable'].unique())
    exp = ['Primary Energy']
    assert obs == exp


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
    ia = IamDataFrame(TEST_DF)
    assert list(ia['variable'].unique()) == [
        'Primary Energy', 'Primary Energy|Coal']


def test_validate_none(test_df):
    obs = test_df.validate(
        {'Primary Energy': {'up': 10}}, exclude=True)
    assert obs is None
    assert len(test_df.data) == 4  # data unchanged

    assert list(test_df['exclude']) == [False]  # none excluded


def test_validate_null(test_df):
    obs = test_df.validate({'Secondary Energy': {'up': 10}}, exclude=True)
    assert obs is None


def test_validate_up(test_df):
    obs = test_df.validate({'Primary Energy': {'up': 5.0}}, exclude=False)
    assert len(obs) == 1
    assert obs['year'].values[0] == 2010


def test_validate_lo(test_df):
    obs = test_df.validate({'Primary Energy': {'lo': 5.0}}, exclude=False)
    assert len(obs) == 1
    assert obs['year'].values[0] == 2005


def test_validate_year(test_df):
    obs = test_df.validate({'Primary Energy': {'up': 5.0, 'year': 2005}},
                           exclude=False)
    assert obs is None

    obs = test_df.validate({'Primary Energy': {'up': 5.0, 'year': 2010}},
                           exclude=False)
    assert len(obs) == 1


def test_validate_exclude(test_df):
    obs = test_df.validate({'Primary Energy': {'up': 5.0}}, exclude=True)
    assert list(test_df['exclude']) == [True]  # one scenario in dataset


def test_validate_top_level(test_df):
    obs = validate(test_df,
                   filters={'variable': 'Primary Energy'},
                   criteria={'Primary Energy': {'up': 5.0}},
                   exclude=True)
    assert len(obs) == 1
    assert obs['year'].values[0] == 2010
    assert list(test_df['exclude']) == [True]  # one scenario in dataset


def test_category_none(test_df):
    test_df.categorize('category', 'Testing', {'Primary Energy': {'up': 0.8}})
    obs = test_df['category'].values
    exp = [None]
    assert obs == exp


def test_category_pass(test_df):
    dct = {'model': ['a_model'], 'scenario': ['a_scenario'],
           'category': ['Testing']}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    test_df.categorize('category', 'Testing', {'Primary Energy': {'up': 10}})
    obs = test_df['category']
    pd.testing.assert_series_equal(obs, exp)


def test_category_top_level(test_df):
    dct = {'model': ['a_model'], 'scenario': ['a_scenario'],
           'category': ['Testing']}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    categorize(test_df, 'category', 'Testing',
               criteria={'Primary Energy': {'up': 10}},
               filters={'variable': 'Primary Energy'})
    obs = test_df['category']
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
