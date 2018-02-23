import os
import pytest

from pyam_analysis import IamDataFrame

here = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(here, 'testing_data.csv')
plot_path = os.path.join(here, 'plot_data.csv')
IMAGE_BASELINE_DIR = os.path.join(here, 'expected_figs')


@pytest.fixture(scope="session")
def test_df():
    df = IamDataFrame(data=data_path)
    yield df


@pytest.fixture(scope="session")
def plot_df():
    df = IamDataFrame(data=plot_path)
    yield df
