from pyam_analysis import IamDataFrame
from testing_utils import test_ia

import pytest
import pandas as pd
from numpy import testing as npt


def test_model(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    assert df.models() == ['test_model']


def test_scenario(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    assert df.scenarios() == ['test_scenario']


def test_region(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    assert df.regions() == ['World']


def test_variable(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    assert df.variables() == ['Primary Energy']


def test_variable_unit(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    dct = {'variable': ['Primary Energy'], 'unit': ['EJ/y']}
    exp = pd.DataFrame.from_dict(dct)[['variable', 'unit']]
    npt.assert_array_equal(df.variables(include_units=True), exp)


def test_validate_pass(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    assert df.validate(criteria='Primary Energy', exclude=True) is None


def test_validate_fail(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    dct = {'model': ['test_model'], 'scenario': ['test_scenario']}
    exp = pd.DataFrame(dct)[['model', 'scenario']]
    npt.assert_array_equal(df.validate(criteria='Secondary Energy'), exp)

    assert df.category('exclude', display='df').empty


def test_validate_exclude(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    obs = df.validate(criteria='Secondary Energy', exclude=True)

    dct = {'model': ['test_model'], 'scenario': ['test_scenario']}
    exp = pd.DataFrame(dct)[['model', 'scenario']]
    npt.assert_array_equal(obs, exp)

    dct = {'model': ['test_model'], 'scenario': ['test_scenario']}
    exp = pd.DataFrame(dct)[['model', 'scenario']]
    obs = df.category('exclude', display='df')
    npt.assert_array_equal(obs[['model', 'scenario']], exp)


def test_category_none(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    obs = df.category('Testing', {'Primary Energy': {'up': 0.8}}, display='df')
    assert obs is None


def test_category_pass(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    dct = {'model': ['test_model'], 'scenario': ['test_scenario']}
    exp = pd.DataFrame(dct)[['model', 'scenario']]
    obs = df.category('Testing', {'Primary Energy': {'up': 1}}, display='df')
    npt.assert_array_equal(obs, exp)
    
    obs2 = df.category('Testing', display='df')
    npt.assert_array_equal(obs2[['model', 'scenario']], exp)


def test_append(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    df2 = df.append(path='tests', file='testing_data_2', ext='csv')
    assert df.scenarios() == ['test_scenario']
    assert df2.scenarios() == ['test_scenario', 'append_scenario']


def test_append_duplicates(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    pytest.raises(ValueError, df.append,
                  path='tests', file='testing_data', ext='csv')
