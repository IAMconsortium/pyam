import os
import pytest

from pyam_analysis import IamDataFrame

here = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(here, 'testing_data.csv')
plot_path = os.path.join(here, 'plot_data.csv')


@pytest.fixture(scope="session")
def test_ia():
    df = IamDataFrame(data=data_path)
    yield df


@pytest.fixture(scope="session")
def plot_idf():
    df = IamDataFrame(data=plot_path)
    yield df
