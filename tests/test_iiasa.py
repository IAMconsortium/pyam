import os
import copy
import pytest
import pandas as pd
import numpy as np

import numpy.testing as npt
import pandas.testing as pdt

from pyam import iiasa, META_IDX
from conftest import IIASA_UNAVAILABLE, TEST_API, TEST_API_NAME

if IIASA_UNAVAILABLE:
    pytest.skip('IIASA database API unavailable', allow_module_level=True)

# check to see if we can do online testing of db authentication
TEST_ENV_USER = 'IIASA_CONN_TEST_USER'
TEST_ENV_PW = 'IIASA_CONN_TEST_PW'
CONN_ENV_AVAILABLE = TEST_ENV_USER in os.environ and TEST_ENV_PW in os.environ
CONN_ENV_REASON = 'Requires env variables defined: {} and {}'.format(
    TEST_ENV_USER, TEST_ENV_PW
)

META_COLS = ['number', 'string']
META_DF = pd.DataFrame([
    ['model_a', 'scen_a', 1, True, 1, 'foo'],
    ['model_a', 'scen_b', 1, True, 2, np.nan],
    ['model_a', 'scen_a', 2, False, 1, 'bar'],
    ['model_b', 'scen_a', 1, True, 3, 'baz']
], columns=META_IDX+['version', 'is_default']+META_COLS).set_index(META_IDX)


def test_unknown_conn():
    # connecting to an unknown API raises an error
    pytest.raises(ValueError, iiasa.Connection, 'foo')


def test_anon_conn(conn):
    assert conn.current_connection == TEST_API_NAME


@pytest.mark.skipif(not CONN_ENV_AVAILABLE, reason=CONN_ENV_REASON)
def test_conn_creds_config():
    iiasa.set_config(os.environ[TEST_ENV_USER], os.environ[TEST_ENV_PW])
    conn = iiasa.Connection(TEST_API)
    assert conn.current_connection == TEST_API_NAME


@pytest.mark.skipif(not CONN_ENV_AVAILABLE, reason=CONN_ENV_REASON)
def test_conn_creds_tuple():
    user, pw = os.environ[TEST_ENV_USER], os.environ[TEST_ENV_PW]
    conn = iiasa.Connection(TEST_API, creds=(user, pw))
    assert conn.current_connection == TEST_API_NAME


@pytest.mark.skipif(not CONN_ENV_AVAILABLE, reason=CONN_ENV_REASON)
def test_conn_creds_dict():
    user, pw = os.environ[TEST_ENV_USER], os.environ[TEST_ENV_PW]
    conn = iiasa.Connection(TEST_API, creds={'username': user, 'password': pw})
    assert conn.current_connection == TEST_API_NAME


def test_conn_bad_creds():
    # connecting with invalid credentials raises an error
    creds = ('_foo', '_bar')
    pytest.raises(RuntimeError, iiasa.Connection, TEST_API, creds=creds)


def test_conn_creds_dict_raises():
    # connecting with incomplete credentials as dictionary raises an error
    creds = {'username': 'foo'}
    pytest.raises(KeyError, iiasa.Connection, TEST_API, creds=creds)



def test_variables(conn):
    # check that connection returns the correct variables
    npt.assert_array_equal(conn.variables(),
                           ['Primary Energy', 'Primary Energy|Coal'])


def test_regions(conn):
    # check that connection returns the correct regions
    npt.assert_array_equal(conn.regions(), ['World', 'region_a'])


def test_regions_with_synonyms(conn):
    obs = conn.regions(include_synonyms=True)
    exp = pd.DataFrame([['World', None], ['region_a', 'ISO_a']],
                       columns=['region', 'synonym'])
    pdt.assert_frame_equal(obs, exp)


def test_regions_empty_response():
    obs = iiasa.Connection.convert_regions_payload('[]', include_synonyms=True)
    assert obs.empty


def test_regions_no_synonyms_response():
    json = '[{"id":1,"name":"World","parent":"World","hierarchy":"common"}]'
    obs = iiasa.Connection.convert_regions_payload(json, include_synonyms=True)
    assert not obs.empty


def test_regions_with_synonyms_response():
    json = '''
    [
        {
            "id":1,"name":"World","parent":"World","hierarchy":"common",
            "synonyms":[]
        },
        {
            "id":2,"name":"USA","parent":"World","hierarchy":"country",
            "synonyms":["US","United States"]
        },
        {
            "id":3,"name":"Germany","parent":"World","hierarchy":"country",
            "synonyms":["Deutschland","DE"]
        }
    ]
    '''
    obs = iiasa.Connection.convert_regions_payload(json, include_synonyms=True)
    assert not obs.empty
    assert (obs[obs.region == 'USA']
            .synonym.isin(['US', 'United States'])).all()
    assert (obs[obs.region == 'Germany']
            .synonym.isin(['Deutschland', 'DE'])).all()


def test_meta_columns(conn):
    # test that connection returns the correct list of meta indicators
    npt.assert_array_equal(conn.meta_columns, META_COLS)

    # test for deprecated version of the function
    npt.assert_array_equal(conn.available_metadata(), META_COLS)

@pytest.mark.parametrize("default", [True, False])
def test_index(conn, default):
    # test that connection returns the correct index
    if default:
        exp = META_DF.loc[META_DF.is_default, ['version']]
    else:
        exp = META_DF[['version', 'is_default']]

    pdt.assert_frame_equal(conn.index(default=default), exp, check_dtype=False)


@pytest.mark.parametrize("default", [True, False])
def test_meta(conn, default):
    # test that connection returns the correct meta dataframe
    if default:
        exp = META_DF.loc[META_DF.is_default, ['version'] + META_COLS]
    else:
        exp = META_DF[['version', 'is_default'] + META_COLS]

    pdt.assert_frame_equal(conn.meta(default=default), exp, check_dtype=False)

    # test for deprecated version of the function
    pdt.assert_frame_equal(conn.metadata(default=default), exp,
                           check_dtype=False)


def test_query(conn):
    # test reading timeseries data
    df = conn.query()
    print(df)

QUERY_DATA_EXP = {
    "filters": {
        "regions": [],
        "variables": [],
        "runs": [],
        "years": [],
        "units": [],
        "timeslices": []
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
