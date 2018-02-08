import os
import pytest

from pyam_analysis import IamDataFrame

here = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(here, 'testing_data.csv')


@pytest.fixture(scope="session")
def test_ia():
    df = IamDataFrame(data=data_path)
    yield df
