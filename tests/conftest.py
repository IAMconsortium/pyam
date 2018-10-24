import os
import pytest

import pandas as pd

from pyam import IamDataFrame, OpenSCMDataFrame

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


mg_ascen = ['MSG-GLB', 'a_scen']
mg_ascen_2 = ['MSG-GLB', 'a_scen_2']
CHECK_AGG_DF = pd.DataFrame([
    ['IMG', 'a_scen', 'R5ASIA', 'Primary Energy', 'EJ/y', 1, 6],
    ['IMG', 'a_scen', 'R5ASIA', 'Primary Energy|Coal', 'EJ/y', 0.75, 5],
    ['IMG', 'a_scen', 'R5ASIA', 'Primary Energy|Gas', 'EJ/y', 0.25, 1],
    ['IMG', 'a_scen', 'R5ASIA', 'Emissions|CO2', 'Mt CO2/yr', 3, 8],
    ['IMG', 'a_scen', 'R5ASIA', 'Emissions|CO2|Cars', 'Mt CO2/yr', 1, 3],
    ['IMG', 'a_scen', 'R5ASIA', 'Emissions|CO2|Tar', 'Mt CO2/yr', 2, 5],
    ['IMG', 'a_scen', 'R5REF', 'Primary Energy', 'EJ/y', 0.3, 0.6],
    ['IMG', 'a_scen', 'R5REF', 'Primary Energy|Coal', 'EJ/y', 0.15, 0.4],
    ['IMG', 'a_scen', 'R5REF', 'Primary Energy|Gas', 'EJ/y', 0.15, 0.2],
    ['IMG', 'a_scen', 'R5REF', 'Emissions|CO2', 'Mt CO2/yr', 1, 1.4],
    ['IMG', 'a_scen', 'R5REF', 'Emissions|CO2|Cars', 'Mt CO2/yr', 0.6, 0.8],
    ['IMG', 'a_scen', 'R5REF', 'Emissions|CO2|Tar', 'Mt CO2/yr', 0.4, 0.6],
    ['IMG', 'a_scen', 'World', 'Primary Energy', 'EJ/y', 1.3, 6.6],
    ['IMG', 'a_scen', 'World', 'Primary Energy|Coal', 'EJ/y', 0.9, 5.4],
    ['IMG', 'a_scen', 'World', 'Primary Energy|Gas', 'EJ/y', 0.4, 1.2],
    ['IMG', 'a_scen', 'World', 'Emissions|CO2', 'Mt CO2/yr', 4, 9.4],
    ['IMG', 'a_scen', 'World', 'Emissions|CO2|Cars', 'Mt CO2/yr', 1.6, 3.8],
    ['IMG', 'a_scen', 'World', 'Emissions|CO2|Tar', 'Mt CO2/yr', 2.4, 5.6],
    ['IMG', 'a_scen_2', 'R5ASIA', 'Primary Energy', 'EJ/y', 1.4, 6.4],
    ['IMG', 'a_scen_2', 'R5ASIA', 'Primary Energy|Coal', 'EJ/y', 0.95, 5.2],
    ['IMG', 'a_scen_2', 'R5ASIA', 'Primary Energy|Gas', 'EJ/y', 0.45, 1.2],
    ['IMG', 'a_scen_2', 'R5ASIA', 'Emissions|CO2', 'Mt CO2/yr', 3.4, 8.4],
    ['IMG', 'a_scen_2', 'R5ASIA', 'Emissions|CO2|Cars', 'Mt CO2/yr', 1.2, 3.2],
    ['IMG', 'a_scen_2', 'R5ASIA', 'Emissions|CO2|Tar', 'Mt CO2/yr', 2.2, 5.2],
    ['IMG', 'a_scen_2', 'R5REF', 'Primary Energy', 'EJ/y', 0.7, 1.0],
    ['IMG', 'a_scen_2', 'R5REF', 'Primary Energy|Coal', 'EJ/y', 0.35, 0.6],
    ['IMG', 'a_scen_2', 'R5REF', 'Primary Energy|Gas', 'EJ/y', 0.35, 0.4],
    ['IMG', 'a_scen_2', 'R5REF', 'Emissions|CO2', 'Mt CO2/yr', 1.4, 1.8],
    ['IMG', 'a_scen_2', 'R5REF', 'Emissions|CO2|Cars', 'Mt CO2/yr', 0.8, 1.0],
    ['IMG', 'a_scen_2', 'R5REF', 'Emissions|CO2|Tar', 'Mt CO2/yr', 0.6, 0.8],
    ['IMG', 'a_scen_2', 'World', 'Primary Energy', 'EJ/y', 2.1, 7.4],
    ['IMG', 'a_scen_2', 'World', 'Primary Energy|Coal', 'EJ/y', 1.3, 5.8],
    ['IMG', 'a_scen_2', 'World', 'Primary Energy|Gas', 'EJ/y', 0.8, 1.6],
    ['IMG', 'a_scen_2', 'World', 'Emissions|CO2', 'Mt CO2/yr', 4.8, 10.2],
    ['IMG', 'a_scen_2', 'World', 'Emissions|CO2|Cars', 'Mt CO2/yr', 2.0, 4.2],
    ['IMG', 'a_scen_2', 'World', 'Emissions|CO2|Tar', 'Mt CO2/yr', 2.8, 6.0],
    mg_ascen + ['R5ASIA', 'Primary Energy', 'EJ/y', 0.8, 5.8],
    mg_ascen + ['R5ASIA', 'Primary Energy|Coal', 'EJ/y', 0.65, 4.9],
    mg_ascen + ['R5ASIA', 'Primary Energy|Gas', 'EJ/y', 0.15, 0.9],
    mg_ascen + ['R5ASIA', 'Emissions|CO2', 'Mt CO2/yr', 2.8, 7.8],
    mg_ascen + ['R5ASIA', 'Emissions|CO2|Cars', 'Mt CO2/yr', 0.9, 2.9],
    mg_ascen + ['R5ASIA', 'Emissions|CO2|Tar', 'Mt CO2/yr', 1.9, 4.9],
    mg_ascen + ['R5REF', 'Primary Energy', 'EJ/y', 0.1, 0.4],
    mg_ascen + ['R5REF', 'Primary Energy|Coal', 'EJ/y', 0.05, 0.3],
    mg_ascen + ['R5REF', 'Primary Energy|Gas', 'EJ/y', 0.05, 0.1],
    mg_ascen + ['R5REF', 'Emissions|CO2', 'Mt CO2/yr', 0.8, 1.2],
    mg_ascen + ['R5REF', 'Emissions|CO2|Cars', 'Mt CO2/yr', 0.5, 0.7],
    mg_ascen + ['R5REF', 'Emissions|CO2|Tar', 'Mt CO2/yr', 0.3, 0.5],
    mg_ascen + ['World', 'Primary Energy', 'EJ/y', 0.9, 6.2],
    mg_ascen + ['World', 'Primary Energy|Coal', 'EJ/y', 0.7, 5.2],
    mg_ascen + ['World', 'Primary Energy|Gas', 'EJ/y', 0.2, 1.0],
    mg_ascen + ['World', 'Emissions|CO2', 'Mt CO2/yr', 3.6, 9.0],
    mg_ascen + ['World', 'Emissions|CO2|Cars', 'Mt CO2/yr', 1.4, 3.6],
    mg_ascen + ['World', 'Emissions|CO2|Tar', 'Mt CO2/yr', 2.2, 5.4],
    mg_ascen_2 + ['R5ASIA', 'Primary Energy', 'EJ/y', -1.4, -6.4],
    mg_ascen_2 + ['R5ASIA', 'Primary Energy|Coal', 'EJ/y', -0.95, -5.2],
    mg_ascen_2 + ['R5ASIA', 'Primary Energy|Gas', 'EJ/y', -0.45, -1.2],
    mg_ascen_2 + ['R5ASIA', 'Emissions|CO2', 'Mt CO2/yr', -3.4, -8.4],
    mg_ascen_2 + ['R5ASIA', 'Emissions|CO2|Cars', 'Mt CO2/yr', -1.2, -3.2],
    mg_ascen_2 + ['R5ASIA', 'Emissions|CO2|Tar', 'Mt CO2/yr', -2.2, -5.2],
    mg_ascen_2 + ['R5REF', 'Primary Energy', 'EJ/y', -0.7, -1.0],
    mg_ascen_2 + ['R5REF', 'Primary Energy|Coal', 'EJ/y', -0.35, -0.6],
    mg_ascen_2 + ['R5REF', 'Primary Energy|Gas', 'EJ/y', -0.35, -0.4],
    mg_ascen_2 + ['R5REF', 'Emissions|CO2', 'Mt CO2/yr', -1.4, -1.8],
    mg_ascen_2 + ['R5REF', 'Emissions|CO2|Cars', 'Mt CO2/yr', -0.8, -1.0],
    mg_ascen_2 + ['R5REF', 'Emissions|CO2|Tar', 'Mt CO2/yr', -0.6, -0.8],
    mg_ascen_2 + ['World', 'Primary Energy', 'EJ/y', -2.1, -7.4],
    mg_ascen_2 + ['World', 'Primary Energy|Coal', 'EJ/y', -1.3, -5.8],
    mg_ascen_2 + ['World', 'Primary Energy|Gas', 'EJ/y', -0.8, -1.6],
    mg_ascen_2 + ['World', 'Emissions|CO2', 'Mt CO2/yr', -5.0, -10.6],
    mg_ascen_2 + ['World', 'Emissions|CO2|Cars', 'Mt CO2/yr', -2.0, -4.2],
    mg_ascen_2 + ['World', 'Emissions|CO2|Tar', 'Mt CO2/yr', -2.8, -6.0],
    mg_ascen_2 + ['World', 'Emissions|CO2|Agg Agg', 'Mt CO2/yr', -0.2, -0.4],
    mg_ascen_2 + ['World', 'Emissions|CF4', 'kt CF4/yr', 54, 56],
    mg_ascen_2 + ['World', 'Emissions|C2F6', 'kt C2F6/yr', 32, 27],
    mg_ascen_2 + ['World', 'Emissions|C2F6|Solvents', 'kt C2F6/yr', 30, 33],
    mg_ascen_2 + ['World', 'Emissions|C2F6|Industry', 'kt C2F6/yr', 2, -6],
    mg_ascen_2 + ['World', 'Emissions|CH4', 'Mt CH4/yr', 322, 217],
    mg_ascen_2 + ['R5REF', 'Emissions|CH4', 'Mt CH4/yr', 30, 201],
    mg_ascen_2 + ['R5ASIA', 'Emissions|CH4', 'Mt CH4/yr', 292, 16],
],
    columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010],
)


CHECK_AGG_REGIONAL_DF = pd.DataFrame([
    ['AIM', 'cscen', 'World', 'Emissions|N2O', 'Mt N/yr', 1.8, 15.6],
    ['AIM', 'cscen', 'World', 'Emissions|N2O|Shipping', 'Mt N/yr', 1, 6],
    ['AIM', 'cscen', 'World', 'Emissions|N2O|Solvents', 'Mt N/yr', 1.6, 3.8],
    ['AIM', 'cscen', 'World', 'Emissions|N2O|Transport', 'Mt N/yr', -0.8, 5.8],
    ['AIM', 'cscen', 'RASIA', 'Emissions|N2O', 'Mt N/yr', 0, 5.9],
    ['AIM', 'cscen', 'RASIA', 'Emissions|N2O|Solvents', 'Mt N/yr', 0.8, 2.6],
    ['AIM', 'cscen', 'RASIA', 'Emissions|N2O|Transport', 'Mt N/yr', -0.8, 3.3],
    ['AIM', 'cscen', 'REUROPE', 'Emissions|N2O', 'Mt N/yr', 0.8, 3.7],
    ['AIM', 'cscen', 'REUROPE', 'Emissions|N2O|Solvents', 'Mt N/yr', 0.8, 1.2],
    ['AIM', 'cscen', 'REUROPE', 'Emissions|N2O|Transport', 'Mt N/yr', 0, 2.5],
    ['AIM', 'cscen', 'China', 'Emissions|N2O', 'Mt N/yr', 0.2, 1.3],
    ['AIM', 'cscen', 'China', 'Emissions|N2O|Transport', 'Mt N/yr', 0.2, 1.3],
    ['AIM', 'cscen', 'Japan', 'Emissions|N2O', 'Mt N/yr', -1, 2],
    ['AIM', 'cscen', 'Japan', 'Emissions|N2O|Transport', 'Mt N/yr', -1, 2],
    ['AIM', 'cscen', 'Germany', 'Emissions|N2O', 'Mt N/yr', 2, 3],
    ['AIM', 'cscen', 'Germany', 'Emissions|N2O|Transport', 'Mt N/yr', 2, 3],
    ['AIM', 'cscen', 'UK', 'Emissions|N2O', 'Mt N/yr', -2, -0.5],
    ['AIM', 'cscen', 'UK', 'Emissions|N2O|Transport', 'Mt N/yr', -2, -0.5],

],
    columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010],
)


@pytest.fixture(scope="function", params=[IamDataFrame, OpenSCMDataFrame])
def pyam_df(request):
    return request.param


@pytest.fixture(scope="function")
def test_df(pyam_df):
    df = pyam_df(data=TEST_DF.iloc[:2])
    yield df


@pytest.fixture(scope="function")
def test_df_iam():
    df = IamDataFrame(data=TEST_DF.iloc[:2])
    yield df


@pytest.fixture(scope="function")
def test_df_openscm():
    df = OpenSCMDataFrame(data=TEST_DF.iloc[:2])
    yield df


@pytest.fixture(scope="function")
def test_pd_df():
    yield TEST_DF


@pytest.fixture(scope="function")
def meta_df(pyam_df):
    df = pyam_df(data=TEST_DF)
    yield df


@pytest.fixture(scope="function")
def meta_df_iam():
    df = IamDataFrame(data=TEST_DF)
    yield df


@pytest.fixture(scope="function")
def check_aggregate_df_iam():
    df = IamDataFrame(data=CHECK_AGG_DF)
    yield df


@pytest.fixture(scope="function")
def check_aggregate_regional_df_iam():
    df = IamDataFrame(data=CHECK_AGG_REGIONAL_DF)
    yield df


@pytest.fixture(scope="function")
def reg_df(pyam_df):
    df = pyam_df(data=REG_DF)
    yield df


@pytest.fixture(scope="function")
def reg_df_iam():
    df = IamDataFrame(data=REG_DF)
    yield df


@pytest.fixture(scope="session")
def plot_df_iam():
    df = IamDataFrame(data=os.path.join(TEST_DATA_DIR, 'plot_data.csv'))
    yield df

@pytest.fixture(scope="function")
def float_time_pd_df():
    return pd.DataFrame([
        ['a_model', 'a_scenario', 'World', 'Primary Energy', 'EJ/y', 1, 6.],
    ],
        columns=['model', 'scenario', 'region', 'variable', 'unit', 2005.5, 2010.5],
    )

