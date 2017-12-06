import os
import pytest

from pyam_analysis import IamDataFrame

here = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="session")
def test_ia():
    df = IamDataFrame(data='tests/testing_data.csv')
    yield df


@pytest.fixture(scope="session")
def test_data_path():
    pth = 'tests/testing_data.csv'
    yield pth
