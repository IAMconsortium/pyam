from pyam_analysis import IamDataFrame
from testing_utils import test_ia


def test_model(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    assert df.models() == ['test_model']


def test_scenario(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    assert df.scenarios() == ['test_scenario']


def test_variable(test_ia):
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    assert df.variables() == ['Primary Energy']
