from pyam_analysis import IamDataFrame
from testing_utils import test_ia, test_data_path

import pytest
import pandas as pd
from numpy import testing as npt


def test_model(test_ia):
    assert test_ia.models() == ['test_model']


def test_scenario(test_ia):
    assert test_ia.scenarios() == ['test_scenario']


def test_region(test_ia):
    assert test_ia.regions() == ['World']


def test_variable(test_ia):
    assert test_ia.variables() == ['Primary Energy', 'Primary Energy|Coal']


def test_variable_depth(test_ia):
    assert test_ia.variables({'level': 0}) == ['Primary Energy']


def test_variable_unit(test_ia):
    dct = {'variable': ['Primary Energy', 'Primary Energy|Coal'],
           'unit': ['EJ/y', 'EJ/y']}
    exp = pd.DataFrame.from_dict(dct)[['variable', 'unit']]
    npt.assert_array_equal(test_ia.variables(include_units=True), exp)


def test_validate_pass(test_ia):
    assert test_ia.validate(criteria='Primary Energy', exclude=True) is None


def test_validate_fail(test_ia):
    df = test_ia
    dct = {'model': ['test_model'], 'scenario': ['test_scenario']}
    exp = pd.DataFrame(dct)[['model', 'scenario']]
    npt.assert_array_equal(df.validate(criteria='Secondary Energy'), exp)

    assert df.category('exclude', display='df').empty


def test_validate_exclude(test_ia):
    df = test_ia
    obs = df.validate(criteria='Secondary Energy', exclude=True)

    dct = {'model': ['test_model'], 'scenario': ['test_scenario']}
    exp = pd.DataFrame(dct)[['model', 'scenario']]
    npt.assert_array_equal(obs, exp)

    dct = {'model': ['test_model'], 'scenario': ['test_scenario']}
    exp = pd.DataFrame(dct)[['model', 'scenario']]
    obs = df.category('exclude', display='df')
    npt.assert_array_equal(obs[['model', 'scenario']], exp)


def test_category_none(test_ia):
    obs = test_ia.category('Testing', {'Primary Energy': {'up': 0.8}}, 
                           display='df')
    assert obs is None


def test_category_pass(test_data_path):
    df = IamDataFrame(data=test_data_path)
    dct = {'model': ['test_model'], 'scenario': ['test_scenario']}
    exp = pd.DataFrame(dct)[['model', 'scenario']]
    obs = df.category('Testing', {'Primary Energy': {'up': 10}},
                      color='red', display='df')
    npt.assert_array_equal(obs, exp)

    obs2 = df.category('Testing', display='df')
    npt.assert_array_equal(obs2[['model', 'scenario']], exp)


def test_load_metadata(test_ia):
    test_ia.load_metadata('tests/testing_metadata.xlsx')
    obs = test_ia.metadata()
    dct = {'model': ['test_model'], 'scenario': ['test_scenario'],
           'category': ['imported']}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])
    npt.assert_array_equal(obs, exp)

    # test importing category color assignment
    assert test_ia.cat_color['imported'] == 'red'


def test_append(test_data_path):
    df = IamDataFrame(data=test_data_path)
    df2 = df.append(other='tests/testing_data_2.csv')
    assert df.scenarios() == ['test_scenario']
    assert df2.scenarios() == ['test_scenario', 'append_scenario']


def test_append_duplicates(test_ia):
    pytest.raises(ValueError, test_ia.append,
                  other='tests/testing_data.csv')


def test_interpolate(test_data_path):
    df = IamDataFrame(data=test_data_path)
    df.interpolate(2007)
    dct = {'model': ['test_model'] * 3, 'scenario': ['test_scenario'] * 3,
           'years': [2005, 2007, 2010], 'value': [1, 3, 6]}
    exp = pd.DataFrame(dct).pivot_table(index=['model', 'scenario'],
                                        columns=['years'], values='value')
    obs = df.timeseries({'variable': 'Primary Energy'})
    npt.assert_array_equal(obs, exp)
