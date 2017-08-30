import os
import pytest

from pyam_analysis import IamDataFrame

here = os.path.dirname(os.path.realpath(__file__))

@pytest.fixture(scope="session")
def test_ia():
    df = IamDataFrame(path='tests', file='testing_data', ext='csv')
    yield df
