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


CHECK_AGG_DF = pd.DataFrame([
    ['IMAGE', 'a_scenario', 'R5ASIA', 'Primary Energy', 'EJ/y', 1, 6],
    ['IMAGE', 'a_scenario', 'R5ASIA', 'Primary Energy|Coal', 'EJ/y', 0.75, 5],
    ['IMAGE', 'a_scenario', 'R5ASIA', 'Primary Energy|Gas', 'EJ/y', 0.25, 1],
    ['IMAGE', 'a_scenario', 'R5ASIA', 'Emissions|CO2', 'Mt CO2/yr', 3, 8],
    ['IMAGE', 'a_scenario', 'R5ASIA', 'Emissions|CO2|Cars', 'Mt CO2/yr', 1, 3],
    ['IMAGE', 'a_scenario', 'R5ASIA', 'Emissions|CO2|Power', 'Mt CO2/yr', 2, 5],
    ['IMAGE', 'a_scenario', 'R5REF', 'Primary Energy', 'EJ/y', 0.3, 0.6],
    ['IMAGE', 'a_scenario', 'R5REF', 'Primary Energy|Coal', 'EJ/y', 0.15, 0.4],
    ['IMAGE', 'a_scenario', 'R5REF', 'Primary Energy|Gas', 'EJ/y', 0.15, 0.2],
    ['IMAGE', 'a_scenario', 'R5REF', 'Emissions|CO2', 'Mt CO2/yr', 1, 1.4],
    ['IMAGE', 'a_scenario', 'R5REF', 'Emissions|CO2|Cars', 'Mt CO2/yr', 0.6, 0.8],
    ['IMAGE', 'a_scenario', 'R5REF', 'Emissions|CO2|Power', 'Mt CO2/yr', 0.4, 0.6],
    ['IMAGE', 'a_scenario', 'World', 'Primary Energy', 'EJ/y', 1.3, 6.6],
    ['IMAGE', 'a_scenario', 'World', 'Primary Energy|Coal', 'EJ/y', 0.9, 5.4],
    ['IMAGE', 'a_scenario', 'World', 'Primary Energy|Gas', 'EJ/y', 0.4, 1.2],
    ['IMAGE', 'a_scenario', 'World', 'Emissions|CO2', 'Mt CO2/yr', 4, 9.4],
    ['IMAGE', 'a_scenario', 'World', 'Emissions|CO2|Cars', 'Mt CO2/yr', 1.6, 3.8],
    ['IMAGE', 'a_scenario', 'World', 'Emissions|CO2|Power', 'Mt CO2/yr', 2.4, 5.6],
    ['IMAGE', 'a_scenario_2', 'R5ASIA', 'Primary Energy', 'EJ/y', 1.4, 6.4],
    ['IMAGE', 'a_scenario_2', 'R5ASIA', 'Primary Energy|Coal', 'EJ/y', 0.95, 5.2],
    ['IMAGE', 'a_scenario_2', 'R5ASIA', 'Primary Energy|Gas', 'EJ/y', 0.45, 1.2],
    ['IMAGE', 'a_scenario_2', 'R5ASIA', 'Emissions|CO2', 'Mt CO2/yr', 3.4, 8.4],
    ['IMAGE', 'a_scenario_2', 'R5ASIA', 'Emissions|CO2|Cars', 'Mt CO2/yr', 1.2, 3.2],
    ['IMAGE', 'a_scenario_2', 'R5ASIA', 'Emissions|CO2|Power', 'Mt CO2/yr', 2.2, 5.2],
    ['IMAGE', 'a_scenario_2', 'R5REF', 'Primary Energy', 'EJ/y', 0.7, 1.0],
    ['IMAGE', 'a_scenario_2', 'R5REF', 'Primary Energy|Coal', 'EJ/y', 0.35, 0.6],
    ['IMAGE', 'a_scenario_2', 'R5REF', 'Primary Energy|Gas', 'EJ/y', 0.35, 0.4],
    ['IMAGE', 'a_scenario_2', 'R5REF', 'Emissions|CO2', 'Mt CO2/yr', 1.4, 1.8],
    ['IMAGE', 'a_scenario_2', 'R5REF', 'Emissions|CO2|Cars', 'Mt CO2/yr', 0.8, 1.0],
    ['IMAGE', 'a_scenario_2', 'R5REF', 'Emissions|CO2|Power', 'Mt CO2/yr', 0.6, 0.8],
    ['IMAGE', 'a_scenario_2', 'World', 'Primary Energy', 'EJ/y', 2.1, 7.4],
    ['IMAGE', 'a_scenario_2', 'World', 'Primary Energy|Coal', 'EJ/y', 1.3, 5.8],
    ['IMAGE', 'a_scenario_2', 'World', 'Primary Energy|Gas', 'EJ/y', 0.8, 1.6],
    ['IMAGE', 'a_scenario_2', 'World', 'Emissions|CO2', 'Mt CO2/yr', 4.8, 10.2],
    ['IMAGE', 'a_scenario_2', 'World', 'Emissions|CO2|Cars', 'Mt CO2/yr', 2.0, 4.2],
    ['IMAGE', 'a_scenario_2', 'World', 'Emissions|CO2|Power', 'Mt CO2/yr', 2.8, 6.0],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'R5ASIA', 'Primary Energy', 'EJ/y', 0.8, 5.8],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'R5ASIA', 'Primary Energy|Coal', 'EJ/y', 0.65, 4.9],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'R5ASIA', 'Primary Energy|Gas', 'EJ/y', 0.15, 0.9],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'R5ASIA', 'Emissions|CO2', 'Mt CO2/yr', 2.8, 7.8],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'R5ASIA', 'Emissions|CO2|Cars', 'Mt CO2/yr', 0.9, 2.9],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'R5ASIA', 'Emissions|CO2|Power', 'Mt CO2/yr', 1.9, 4.9],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'R5REF', 'Primary Energy', 'EJ/y', 0.1, 0.4],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'R5REF', 'Primary Energy|Coal', 'EJ/y', 0.05, 0.3],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'R5REF', 'Primary Energy|Gas', 'EJ/y', 0.05, 0.1],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'R5REF', 'Emissions|CO2', 'Mt CO2/yr', 0.8, 1.2],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'R5REF', 'Emissions|CO2|Cars', 'Mt CO2/yr', 0.5, 0.7],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'R5REF', 'Emissions|CO2|Power', 'Mt CO2/yr', 0.3, 0.5],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'World', 'Primary Energy', 'EJ/y', 0.9, 6.2],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'World', 'Primary Energy|Coal', 'EJ/y', 0.7, 5.2],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'World', 'Primary Energy|Gas', 'EJ/y', 0.2, 1.0],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'World', 'Emissions|CO2', 'Mt CO2/yr', 3.6, 9.0],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'World', 'Emissions|CO2|Cars', 'Mt CO2/yr', 1.4, 3.6],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'World', 'Emissions|CO2|Power', 'Mt CO2/yr', 2.2, 5.4],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5ASIA', 'Primary Energy', 'EJ/y', -1.4, -6.4],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5ASIA', 'Primary Energy|Coal', 'EJ/y', -0.95, -5.2],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5ASIA', 'Primary Energy|Gas', 'EJ/y', -0.45, -1.2],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5ASIA', 'Emissions|CO2', 'Mt CO2/yr', -3.4, -8.4],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5ASIA', 'Emissions|CO2|Cars', 'Mt CO2/yr', -1.2, -3.2],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5ASIA', 'Emissions|CO2|Power', 'Mt CO2/yr', -2.2, -5.2],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5REF', 'Primary Energy', 'EJ/y', -0.7, -1.0],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5REF', 'Primary Energy|Coal', 'EJ/y', -0.35, -0.6],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5REF', 'Primary Energy|Gas', 'EJ/y', -0.35, -0.4],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5REF', 'Emissions|CO2', 'Mt CO2/yr', -1.4, -1.8],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5REF', 'Emissions|CO2|Cars', 'Mt CO2/yr', -0.8, -1.0],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5REF', 'Emissions|CO2|Power', 'Mt CO2/yr', -0.6, -0.8],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'World', 'Primary Energy', 'EJ/y', -2.1, -7.4],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'World', 'Primary Energy|Coal', 'EJ/y', -1.3, -5.8],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'World', 'Primary Energy|Gas', 'EJ/y', -0.8, -1.6],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'World', 'Emissions|CO2', 'Mt CO2/yr', -5.0, -10.6],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'World', 'Emissions|CO2|Cars', 'Mt CO2/yr', -2.0, -4.2],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'World', 'Emissions|CO2|Power', 'Mt CO2/yr', -2.8, -6.0],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'World', 'Emissions|CO2|Aggregate Agg', 'Mt CO2/yr', -0.2, -0.4],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'World', 'Emissions|CF4', 'kt CF4/yr', 54, 56],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'World', 'Emissions|C2F6', 'kt C2F6/yr', 32, 27],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'World', 'Emissions|C2F6|Solvents', 'kt C2F6/yr', 30, 33],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'World', 'Emissions|C2F6|Industry', 'kt C2F6/yr', 2, -6],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'World', 'Emissions|CH4', 'Mt CH4/yr', 322, 217],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5REF', 'Emissions|CH4', 'Mt CH4/yr', 30, 201],
    ['MESSAGE-GLOBIOM', 'a_scenario_2', 'R5ASIA', 'Emissions|CH4', 'Mt CH4/yr', 292, 16],
],
    columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010],
)


CHECK_AGG_REGIONAL_DF = pd.DataFrame([
    ['AIM/CGE', 'c_scen', 'World', 'Emissions|N2O', 'Mt N/yr', 1.8, 15.6],
    ['AIM/CGE', 'c_scen', 'World', 'Emissions|N2O|Shipping', 'Mt N/yr', 1, 6],
    ['AIM/CGE', 'c_scen', 'World', 'Emissions|N2O|Solvents', 'Mt N/yr', 1.6, 3.8],
    ['AIM/CGE', 'c_scen', 'World', 'Emissions|N2O|Transport', 'Mt N/yr', -0.8, 5.8],
    ['AIM/CGE', 'c_scen', 'RASIA', 'Emissions|N2O', 'Mt N/yr', 0, 5.9],
    ['AIM/CGE', 'c_scen', 'RASIA', 'Emissions|N2O|Solvents', 'Mt N/yr', 0.8, 2.6],
    ['AIM/CGE', 'c_scen', 'RASIA', 'Emissions|N2O|Transport', 'Mt N/yr', -0.8, 3.3],
    ['AIM/CGE', 'c_scen', 'REUROPE', 'Emissions|N2O', 'Mt N/yr', 0.8, 3.7],
    ['AIM/CGE', 'c_scen', 'REUROPE', 'Emissions|N2O|Solvents', 'Mt N/yr', 0.8, 1.2],
    ['AIM/CGE', 'c_scen', 'REUROPE', 'Emissions|N2O|Transport', 'Mt N/yr', 0, 2.5],
    ['AIM/CGE', 'c_scen', 'China', 'Emissions|N2O', 'Mt N/yr', 0.2, 1.3],
    ['AIM/CGE', 'c_scen', 'China', 'Emissions|N2O|Transport', 'Mt N/yr', 0.2, 1.3],
    ['AIM/CGE', 'c_scen', 'Japan', 'Emissions|N2O', 'Mt N/yr', -1, 2],
    ['AIM/CGE', 'c_scen', 'Japan', 'Emissions|N2O|Transport', 'Mt N/yr', -1, 2],
    ['AIM/CGE', 'c_scen', 'Germany', 'Emissions|N2O', 'Mt N/yr', 2, 3],
    ['AIM/CGE', 'c_scen', 'Germany', 'Emissions|N2O|Transport', 'Mt N/yr', 2, 3],
    ['AIM/CGE', 'c_scen', 'UK', 'Emissions|N2O', 'Mt N/yr', -2, -0.5],
    ['AIM/CGE', 'c_scen', 'UK', 'Emissions|N2O|Transport', 'Mt N/yr', -2, -0.5],

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
def check_aggregate_df():
    df = IamDataFrame(data=CHECK_AGG_DF)
    yield df


@pytest.fixture(scope="function")
def check_aggregate_regional_df():
    df = IamDataFrame(data=CHECK_AGG_REGIONAL_DF)
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
