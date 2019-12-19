# has to go first for environment setup reasons
import matplotlib
matplotlib.use('agg')

import os
import pytest
import pandas as pd


from datetime import datetime
from pyam import IamDataFrame, IAMC_IDX


here = os.path.dirname(os.path.realpath(__file__))
IMAGE_BASELINE_DIR = os.path.join(here, 'expected_figs')
TEST_DATA_DIR = os.path.join(here, 'data')


TEST_DF = pd.DataFrame([
    ['model_a', 'scen_a', 'World', 'Primary Energy', 'EJ/y', 1, 6.],
    ['model_a', 'scen_a', 'World', 'Primary Energy|Coal', 'EJ/y', 0.5, 3],
    ['model_a', 'scen_b', 'World', 'Primary Energy', 'EJ/y', 2, 7],
],
    columns=IAMC_IDX + [2005, 2010],
)


FULL_FEATURE_DF = pd.DataFrame([
    ['World', 'Primary Energy', 'EJ/y', 12, 15],
    ['reg_a', 'Primary Energy', 'EJ/y', 8, 9],
    ['reg_b', 'Primary Energy', 'EJ/y', 4, 6],
    ['World', 'Primary Energy|Coal', 'EJ/y', 9, 10],
    ['reg_a', 'Primary Energy|Coal', 'EJ/y', 6, 6],
    ['reg_b', 'Primary Energy|Coal', 'EJ/y', 3, 4],
    ['World', 'Primary Energy|Wind', 'EJ/y', 3, 5],
    ['reg_a', 'Primary Energy|Wind', 'EJ/y', 2, 3],
    ['reg_b', 'Primary Energy|Wind', 'EJ/y', 1, 2],
    ['World', 'Emissions|CO2', 'EJ/y', 10, 14],
    ['World', 'Emissions|CO2|Energy', 'EJ/y', 6, 8],
    ['World', 'Emissions|CO2|AFOLU', 'EJ/y', 3, 4],
    ['World', 'Emissions|CO2|Bunkers', 'EJ/y', 1, 2],
    ['reg_a', 'Emissions|CO2', 'EJ/y', 6, 8],
    ['reg_a', 'Emissions|CO2|Energy', 'EJ/y', 4, 5],
    ['reg_a', 'Emissions|CO2|AFOLU', 'EJ/y', 2, 3],
    ['reg_b', 'Emissions|CO2', 'EJ/y', 3, 4],
    ['reg_b', 'Emissions|CO2|Energy', 'EJ/y', 2, 3],
    ['reg_b', 'Emissions|CO2|AFOLU', 'EJ/y', 1, 1],
    ['World', 'Price|Carbon', 'USD/tCO2', 4, 27],
    ['reg_a', 'Price|Carbon', 'USD/tCO2', 1, 30],
    ['reg_b', 'Price|Carbon', 'USD/tCO2', 10, 21],
    ['World', 'Population', 'm', 3, 5],
    ['reg_a', 'Population', 'm', 2, 3],
    ['reg_b', 'Population', 'm', 1, 2],
],
    columns=['region', 'variable', 'unit', 2005, 2010],
)


REG_DF = pd.DataFrame([
    ['IMAGE', 'a_scenario', 'NAF', 'Primary Energy', 'EJ/y', 1, 6],
    ['IMAGE', 'a_scenario', 'ME', 'Primary Energy', 'EJ/y', 2, 7],
    ['IMAGE', 'a_scenario', 'World', 'Primary Energy', 'EJ/y', 3, 13],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'MEA', 'Primary Energy', 'EJ/y', 1, 6],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'AFR', 'Primary Energy', 'EJ/y', 2, 7],
    ['MESSAGE-GLOBIOM', 'a_scenario', 'World', 'Primary Energy', 'EJ/y', 3, 13],
],
    columns=IAMC_IDX + [2005, 2010],
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


ms = ['AIM', 'cscen']
CHECK_AGG_REGIONAL_DF = pd.DataFrame([
    ms + ['World', 'Emissions|N2O', 'Mt N/yr', 1.9, 15.7],
    ms + ['World', 'Emissions|N2O|AFOLU', 'Mt N/yr', 0.1, 0.1],
    ms + ['World', 'Emissions|N2O|Ind', 'Mt N/yr', 1.8, 15.6],
    ms + ['World', 'Emissions|N2O|Ind|Shipping', 'Mt N/yr', 1, 6],
    ms + ['World', 'Emissions|N2O|Ind|Solvents', 'Mt N/yr', 1.6, 3.8],
    ms + ['World', 'Emissions|N2O|Ind|Transport', 'Mt N/yr', -0.8, 5.8],
    ms + ['RASIA', 'Emissions|N2O', 'Mt N/yr', 0, 5.9],
    ms + ['RASIA', 'Emissions|N2O|Ind', 'Mt N/yr', 0, 5.9],
    ms + ['RASIA', 'Emissions|N2O|Ind|Solvents', 'Mt N/yr', 0.8, 2.6],
    ms + ['RASIA', 'Emissions|N2O|Ind|Transport', 'Mt N/yr', -0.8, 3.3],
    ms + ['REUROPE', 'Emissions|N2O', 'Mt N/yr', 0.8, 3.7],
    ms + ['REUROPE', 'Emissions|N2O|Ind', 'Mt N/yr', 0.8, 3.7],
    ms + ['REUROPE', 'Emissions|N2O|Ind|Solvents', 'Mt N/yr', 0.8, 1.2],
    ms + ['REUROPE', 'Emissions|N2O|Ind|Transport', 'Mt N/yr', 0, 2.5],
    ms + ['China', 'Emissions|N2O', 'Mt N/yr', 0.2, 1.3],
    ms + ['China', 'Emissions|N2O|Ind', 'Mt N/yr', 0.2, 1.3],
    ms + ['China', 'Emissions|N2O|Ind|Transport', 'Mt N/yr', 0.2, 1.3],
    ms + ['Japan', 'Emissions|N2O', 'Mt N/yr', -1, 2],
    ms + ['Japan', 'Emissions|N2O|Ind', 'Mt N/yr', -1, 2],
    ms + ['Japan', 'Emissions|N2O|Ind|Transport', 'Mt N/yr', -1, 2],
    ms + ['Germany', 'Emissions|N2O', 'Mt N/yr', 2, 3],
    ms + ['Germany', 'Emissions|N2O|Ind', 'Mt N/yr', 2, 3],
    ms + ['Germany', 'Emissions|N2O|Ind|Transport', 'Mt N/yr', 2, 3],
    ms + ['UK', 'Emissions|N2O', 'Mt N/yr', -2, -0.5],
    ms + ['UK', 'Emissions|N2O|Ind', 'Mt N/yr', -2, -0.5],
    ms + ['UK', 'Emissions|N2O|Ind|Transport', 'Mt N/yr', -2, -0.5],

],
    columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010],
)


TEST_STACKPLOT_DF = pd.DataFrame([
    ['World', 'Emissions|CO2|Energy|Oil', 'Mt CO2/yr', 2, 3.2, 2.0, 1.8],
    ['World', 'Emissions|CO2|Energy|Gas', 'Mt CO2/yr', 1.3, 1.6, 1.0, 0.7],
    ['World', 'Emissions|CO2|Energy|BECCS', 'Mt CO2/yr', 0.0, 0.4, -0.4, 0.3],
    ['World', 'Emissions|CO2|Cars', 'Mt CO2/yr', 1.6, 3.8, 3.0, 2.5],
    ['World', 'Emissions|CO2|Tar', 'Mt CO2/yr', 0.3, 0.35, 0.35, 0.33],
    ['World', 'Emissions|CO2|Agg', 'Mt CO2/yr', 0.5, -0.1, -0.5, -0.7],
    ['World', 'Emissions|CO2|LUC', 'Mt CO2/yr', -0.3, -0.6, -1.2, -1.0]
],
    columns=['region', 'variable', 'unit', 2005, 2010, 2015, 2020],
)
# appease stickler
TEST_STACKPLOT_DF['model'] = 'IMG'
TEST_STACKPLOT_DF['scenario'] = 'a_scen'


TEST_YEARS = [2005, 2010]
TEST_DTS = [datetime(2005, 6, 17), datetime(2010, 7, 21)]
TEST_TIME_STR = ['2005-06-17', '2010-07-21']
TEST_TIME_STR_HR = ['2005-06-17 00:00:00', '2010-07-21 12:00:00']


# minimal IamDataFrame with four different time formats
@pytest.fixture(
    scope="function",
    params=[
        TEST_YEARS,
        TEST_DTS,
        TEST_TIME_STR,
        TEST_TIME_STR_HR
    ]
)
def test_df(request):
    tdf = TEST_DF.rename({2005: request.param[0], 2010: request.param[1]},
                         axis="columns")
    df = IamDataFrame(data=tdf)
    yield df


# minimal IamDataFrame for specifically testing 'year'-column features
@pytest.fixture(scope="function")
def test_df_year():
    df = IamDataFrame(data=TEST_DF)
    yield df


# minimal test data provided as pandas.DataFrame (only 'year' time format)
@pytest.fixture(scope="function")
def test_pd_df():
    yield TEST_DF.copy()


# IamDataFrame with variable-and-region-structure for testing aggregation tools
@pytest.fixture(scope="function")
def aggregate_df():
    df = IamDataFrame(model='model_a', scenario='scen_a', data=FULL_FEATURE_DF)
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


@pytest.fixture(scope="session")
def plot_df():
    df = IamDataFrame(data=os.path.join(TEST_DATA_DIR, 'plot_data.csv'))
    yield df


@pytest.fixture(scope="session")
def plot_stack_plot_df():
    df = IamDataFrame(TEST_STACKPLOT_DF)
    yield df
