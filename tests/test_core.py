from pyam_analysis import IamDataFrame
from testing_utils import test_ia
import pytest


def test_model(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    assert df.models() == ['test_model']


def test_scenario(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    assert df.scenarios() == ['test_scenario']


def test_variable(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    assert df.variables() == ['Primary Energy']


def test_append(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    df2 = df.append(path='tests', file='testing_data_2', ext='csv')
    assert df.scenarios() == ['test_scenario']
    assert df2.scenarios() == ['test_scenario', 'append_scenario']

 
def test_append_duplicates(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    pytest.raises(ValueError, df.append,
                  path='tests', file='testing_data', ext='csv')
