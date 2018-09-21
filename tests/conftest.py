import os
import pytest

import pandas as pd

from pyam import IamDataFrame

here = os.path.dirname(os.path.realpath(__file__))
IMAGE_BASELINE_DIR = os.path.join(here, 'expected_figs')
TEST_DATA_DIR = os.path.join(here, 'data')


TEST_DF = pd.DataFrame([
    ['a_model', 'a_scenario', 'World', 'Primary Energy', 'EJ/y', 1, 6.],
    ['a_model', 'a_scenario', 'World', 'Primary Energy|Coal', 'EJ/y', 0.5, 3],
    ['a_model', 'a_scenario2', 'World', 'Primary Energy', 'EJ/y', 2, 7],
],
    columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010],
)


REG_DF = pd.DataFrame([
    ['IMAGE', 'a_scenario', 'NAF', 'Primary Energy', 'EJ/y', 1, 6],
    ['IMAGE', 'a_scenario', 'ME', 'Primary Energy', 'EJ/y', 2, 7],
    ['IMAGE', 'a_scenario', 'World', 'Primary Energy', 'EJ/y', 3, 13],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'MEA', 'Primary Energy', 'EJ/y', 1, 6],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'AFR', 'Primary Energy', 'EJ/y', 2, 7],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'World', 'Primary Energy', 'EJ/y', 3, 13],
],
    columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010],
)


@pytest.fixture(scope="function")
def test_df():
    df = IamDataFrame(data=TEST_DF.iloc[:2])
    yield df


@pytest.fixture(scope="function")
def test_pd_df():
    yield TEST_DF


@pytest.fixture(scope="function")
def meta_df():
    df = IamDataFrame(data=TEST_DF)
    yield df


@pytest.fixture(scope="function")
def reg_df():
    df = IamDataFrame(data=REG_DF)
    yield df

# PLOT_DF = pd.DataFrame([
#     ['a_model', 'a_scenario', 'World', 'Primary Energy', 'EJ/y', 1, 6, 10],
#     ['a_model', 'a_scenario', 'World', 'Primary Energy|Coal', 'EJ/y', 0.5, 3, 4],
#     ['a_model', 'a_scenario1', 'World', 'Primary Energy', 'EJ/y', 2, 6, 8],
#     ['a_model', 'a_scenario1', 'World', 'Primary Energy|Coal', 'EJ/y', 0.5, 2, 5],
#     ['a_model1', 'a_scenario', 'World', 'Primary Energy', 'EJ/y', 0.7, 4.2, 7],
#     ['a_model1', 'a_scenario', 'World', 'Primary Energy|Coal', 'EJ/y', 0.35, 2.1, 2.8],
#     ['a_model1', 'a_scenario1', 'World', 'Primary Energy', 'EJ/y', 1.4, 4.2, 5.6],
#     ['a_model1', 'a_scenario1', 'World',
#         'Primary Energy|Coal', 'EJ/y', 0.35, 1.4, 3.5],
# ],
#     columns=['model', 'scenario', 'region',
#              'variable', 'unit', 2005, 2010, 2020],
# )


@pytest.fixture(scope="session")
def plot_df():
    df = IamDataFrame(data=os.path.join(TEST_DATA_DIR, 'plot_data.csv'))
    yield df
