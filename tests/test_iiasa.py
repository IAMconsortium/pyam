import copy
import pytest

import numpy.testing as npt

from pyam import iiasa


def test_auth():
    conn = iiasa.Connection('iamc15')
    assert conn.base_url == 'https://db1.ene.iiasa.ac.at/iamc15-api/rest/v2.1/'
    conn.auth()


def test_connection_raises():
    pytest.raises(ValueError, iiasa.Connection, 'foo')


def test_variables():
    conn = iiasa.Connection('iamc15')
    obs = conn.variables().values
    assert 'Emissions|CO2' in obs


def test_regions():
    conn = iiasa.Connection('iamc15')
    obs = conn.regions().values
    assert 'World' in obs


def test_metadata():
    conn = iiasa.Connection('iamc15')
    obs = conn.metadata()['model'].values
    assert 'MESSAGEix-GLOBIOM 1.0' in obs


def test_available_indicators():
    conn = iiasa.Connection('iamc15')
    obs = conn.available_scenario_indicators()
    assert 'carbon price|2050' in list(obs)


QUERY_DATA_EXP = {
    "filters": {
        "regions": [],
        "variables": [],
        "runs": [],
        "years": [],
        "units": [],
        "times": []
    }
}


def test_query_data_model_scen():
    conn = iiasa.Connection('iamc15')
    obs = conn._query_post_data(model='AIM*', scenario='ADVANCE_2020_Med2C')
    exp = copy.deepcopy(QUERY_DATA_EXP)
    exp['filters']['runs'] = [2]
    assert obs == exp


def test_query_data_region():
    conn = iiasa.Connection('iamc15')
    obs = conn._query_post_data(model='AIM*', scenario='ADVANCE_2020_Med2C',
                                region='*World*')
    exp = copy.deepcopy(QUERY_DATA_EXP)
    exp['filters']['runs'] = [2]
    exp['filters']['regions'] = ['World']
    assert obs == exp


def test_query_data_variables():
    conn = iiasa.Connection('iamc15')
    obs = conn._query_post_data(model='AIM*', scenario='ADVANCE_2020_Med2C',
                                variable='Emissions|CO2*')
    exp = copy.deepcopy(QUERY_DATA_EXP)
    exp['filters']['runs'] = [2]
    exp['filters']['variables'] = [
        'Emissions|CO2', 'Emissions|CO2|AFOLU', 'Emissions|CO2|Energy',
        'Emissions|CO2|Energy and Industrial Processes',
        'Emissions|CO2|Energy|Demand', 'Emissions|CO2|Energy|Demand|AFOFI',
        'Emissions|CO2|Energy|Demand|Industry',
        'Emissions|CO2|Energy|Demand|Other Sector',
        'Emissions|CO2|Energy|Demand|Residential and Commercial',
        'Emissions|CO2|Energy|Demand|Transportation',
        'Emissions|CO2|Energy|Supply',
        'Emissions|CO2|Energy|Supply|Electricity',
        'Emissions|CO2|Energy|Supply|Gases',
        'Emissions|CO2|Energy|Supply|Heat',
        'Emissions|CO2|Energy|Supply|Liquids',
        'Emissions|CO2|Energy|Supply|Other Sector',
        'Emissions|CO2|Energy|Supply|Solids',
        'Emissions|CO2|Industrial Processes', 'Emissions|CO2|Other'
    ]
    for k in obs['filters']:
        npt.assert_array_equal(obs['filters'][k], exp['filters'][k])


def test_query_iamc15():
    df = iiasa.read_iiasa_iamc15(model='AIM*', scenario='ADVANCE_2020_Med2C',
                                 variable='Emissions|CO2', region='World')
    assert len(df) == 20


def test_query_iamc15_with_metadata():
    df = iiasa.read_iiasa_iamc15(
        model='MESSAGEix*',
        variable=['Emissions|CO2', 'Primary Energy|Coal'],
        region='World',
        meta=['carbon price|2100 (NPV)', 'category'],
    )
    assert len(df) == 168
    assert len(df.data) == 168
    assert len(df.meta) == 7
