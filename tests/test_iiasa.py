import copy
import pytest
import os
import yaml

import numpy.testing as npt

from pyam import iiasa

# check to see if we can do online testing of db authentication
TEST_ENV_USER = 'IIASA_CONN_TEST_USER'
TEST_ENV_PW = 'IIASA_CONN_TEST_PW'
CONN_ENV_AVAILABLE = TEST_ENV_USER in os.environ and TEST_ENV_PW in os.environ
CONN_ENV_REASON = 'Requires env variables defined: {} and {}'.format(
    TEST_ENV_USER, TEST_ENV_PW
)


def test_anon_conn():
    conn = iiasa.Connection('IXSE_SR15')
    assert conn.current_connection == 'IXSE_SR15'


def test_anon_conn_warning():
    conn = iiasa.Connection('iamc15')
    assert conn.current_connection == 'IXSE_SR15'


@pytest.mark.skipif(not CONN_ENV_AVAILABLE, reason=CONN_ENV_REASON)
def test_conn_creds_file(tmp_path):
    user, pw = os.environ[TEST_ENV_USER], os.environ[TEST_ENV_PW]
    path = tmp_path / 'config.yaml'
    with open(path, 'w') as f:
        yaml.dump({'username': user, 'password': pw}, f)
    conn = iiasa.Connection('IXSE_SR15', creds=path)
    assert conn.current_connection == 'IXSE_SR15'


@pytest.mark.skipif(not CONN_ENV_AVAILABLE, reason=CONN_ENV_REASON)
def test_conn_creds_tuple():
    user, pw = os.environ[TEST_ENV_USER], os.environ[TEST_ENV_PW]
    conn = iiasa.Connection('IXSE_SR15', creds=(user, pw))
    assert conn.current_connection == 'IXSE_SR15'


def test_conn_bad_creds():
    pytest.raises(RuntimeError, iiasa.Connection,
                  'IXSE_SR15', creds=('_foo', '_bar'))


def test_anon_conn_tuple_raises():
    pytest.raises(ValueError, iiasa.Connection, 'foo')


@pytest.mark.skipif(not CONN_ENV_AVAILABLE, reason=CONN_ENV_REASON)
def test_conn_creds_dict():
    user, pw = os.environ[TEST_ENV_USER], os.environ[TEST_ENV_PW]
    conn = iiasa.Connection(
        'IXSE_SR15', creds={'username': user, 'password': pw})
    assert conn.current_connection == 'IXSE_SR15'


def test_conn_creds_dict_raises():
    pytest.raises(KeyError, iiasa.Connection,
                  'IXSE_SR15', creds={'username': 'foo'})


def test_variables():
    conn = iiasa.Connection('IXSE_SR15')
    obs = conn.variables().values
    assert 'Emissions|CO2' in obs


def test_regions():
    conn = iiasa.Connection('IXSE_SR15')
    obs = conn.regions().values
    assert 'World' in obs


def test_metadata():
    conn = iiasa.Connection('IXSE_SR15')
    obs = conn.scenario_list()['model'].values
    assert 'MESSAGEix-GLOBIOM 1.0' in obs


def test_available_indicators():
    conn = iiasa.Connection('IXSE_SR15')
    obs = conn.available_metadata()
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
    conn = iiasa.Connection('IXSE_SR15')
    obs = conn._query_post_data(model='AIM*', scenario='ADVANCE_2020_Med2C')
    exp = copy.deepcopy(QUERY_DATA_EXP)
    exp['filters']['runs'] = [2]
    assert obs == exp


def test_query_data_region():
    conn = iiasa.Connection('IXSE_SR15')
    obs = conn._query_post_data(model='AIM*', scenario='ADVANCE_2020_Med2C',
                                region='*World*')
    exp = copy.deepcopy(QUERY_DATA_EXP)
    exp['filters']['runs'] = [2]
    exp['filters']['regions'] = ['World']
    assert obs == exp


def test_query_data_variables():
    conn = iiasa.Connection('IXSE_SR15')
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


def test_query_IXSE_SR15():
    df = iiasa.read_iiasa('IXSE_SR15',
                          model='AIM*',
                          scenario='ADVANCE_2020_Med2C',
                          variable='Emissions|CO2',
                          region='World',
                          )
    assert len(df) == 20


def test_query_IXSE_AR6():
    with pytest.raises(RuntimeError) as excinfo:
        variable = 'Emissions|CO2|Energy|Demand|Transportation'
        creds = dict(username='mahamba', password='verysecret')
        iiasa.read_iiasa('IXSE_AR6',
                         scenario='ADVANCE_2020_WB2C',
                         model='AIM/CGE 2.0',
                         region='World',
                         variable=variable,
                         creds=creds)
    assert str(excinfo.value).startswith('Login failed for user: mahamba')


def test_query_IXSE_SR15_with_metadata():
    df = iiasa.read_iiasa('IXSE_SR15',
                          model='MESSAGEix*',
                          variable=['Emissions|CO2', 'Primary Energy|Coal'],
                          region='World',
                          meta=['carbon price|2100 (NPV)', 'category'],
                          )
    assert len(df) == 168
    assert len(df.data) == 168
    assert len(df.meta) == 7
