import os
import pytest
import pandas as pd
import numpy as np

import numpy.testing as npt
import pandas.testing as pdt

from pyam import IamDataFrame, iiasa, read_iiasa, META_IDX
from pyam.testing import assert_iamframe_equal

from .conftest import IIASA_UNAVAILABLE, META_COLS, TEST_API, TEST_API_NAME

if IIASA_UNAVAILABLE:
    pytest.skip("IIASA database API unavailable", allow_module_level=True)

# check to see if we can do online testing of db authentication
TEST_ENV_USER = "IIASA_CONN_TEST_USER"
TEST_ENV_PW = "IIASA_CONN_TEST_PW"
CONN_ENV_AVAILABLE = TEST_ENV_USER in os.environ and TEST_ENV_PW in os.environ
CONN_ENV_REASON = "Requires env variables defined: {} and {}".format(
    TEST_ENV_USER, TEST_ENV_PW
)

VERSION_COLS = ["version", "is_default"]
META_DF = pd.DataFrame(
    [
        ["model_a", "scen_a", 1, True, 1, "foo"],
        ["model_a", "scen_b", 1, True, 2, np.nan],
        ["model_a", "scen_a", 2, False, 1, "bar"],
        ["model_b", "scen_a", 1, True, 3, "baz"],
    ],
    columns=META_IDX + VERSION_COLS + META_COLS,
).set_index(META_IDX)

MODEL_B_DF = pd.DataFrame(
    [
        ["Primary Energy", "EJ/yr", "Summer", 1, 3],
        ["Primary Energy", "EJ/yr", "Year", 3, 8],
        ["Primary Energy|Coal", "EJ/yr", "Summer", 0.4, 2],
        ["Primary Energy|Coal", "EJ/yr", "Year", 0.9, 5],
    ],
    columns=["variable", "unit", "subannual", 2005, 2010],
)

NON_DEFAULT_DF = pd.DataFrame(
    [
        ["model_a", "scen_a", 2, "Primary Energy", "EJ/yr", "Year", 2, 7],
        ["model_a", "scen_a", 2, "Primary Energy|Coal", "EJ/yr", "Year", 0.8, 4],
        ["model_b", "scen_a", 1, "Primary Energy", "EJ/yr", "Summer", 1, 3],
        ["model_b", "scen_a", 1, "Primary Energy", "EJ/yr", "Year", 3, 8],
        ["model_b", "scen_a", 1, "Primary Energy|Coal", "EJ/yr", "Summer", 0.4, 2],
        ["model_b", "scen_a", 1, "Primary Energy|Coal", "EJ/yr", "Year", 0.9, 5],
    ],
    columns=META_IDX + ["version", "variable", "unit", "subannual", 2005, 2010],
)


def test_unknown_conn():
    # connecting to an unknown API raises an error
    pytest.raises(ValueError, iiasa.Connection, "foo")


def test_valid_connections():
    # connecting to an unknown API raises an error
    assert TEST_API in iiasa.Connection().valid_connections


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
    conn = iiasa.Connection(TEST_API, creds={"username": user, "password": pw})
    assert conn.current_connection == TEST_API_NAME


def test_conn_cleartext_raises():
    # connecting with invalid credentials raises an error
    creds = ("_foo", "_bar")
    pytest.raises(DeprecationWarning, iiasa.Connection, TEST_API, creds=creds)


def test_variables(conn):
    # check that connection returns the correct variables
    npt.assert_array_equal(conn.variables(), ["Primary Energy", "Primary Energy|Coal"])


def test_regions(conn):
    # check that connection returns the correct regions
    npt.assert_array_equal(conn.regions(), ["World", "region_a"])


def test_regions_with_synonyms(conn):
    obs = conn.regions(include_synonyms=True)
    exp = pd.DataFrame(
        [["World", None], ["region_a", "ISO_a"]], columns=["region", "synonym"]
    )
    pdt.assert_frame_equal(obs, exp)


def test_regions_empty_response():
    obs = iiasa.Connection.convert_regions_payload("[]", include_synonyms=True)
    assert obs.empty


def test_regions_no_synonyms_response():
    json = '[{"id":1,"name":"World","parent":"World","hierarchy":"common"}]'
    obs = iiasa.Connection.convert_regions_payload(json, include_synonyms=True)
    assert not obs.empty


def test_regions_with_synonyms_response():
    json = """
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
    """
    obs = iiasa.Connection.convert_regions_payload(json, include_synonyms=True)
    assert not obs.empty
    assert (obs[obs.region == "USA"].synonym.isin(["US", "United States"])).all()
    assert (obs[obs.region == "Germany"].synonym.isin(["Deutschland", "DE"])).all()


def test_meta_columns(conn):
    # test that connection returns the correct list of meta indicators
    npt.assert_array_equal(conn.meta_columns, META_COLS)


@pytest.mark.parametrize("default", [True, False])
def test_index(conn, default):
    # test that connection returns the correct index
    if default:
        exp = META_DF.loc[META_DF.is_default, ["version"]]
    else:
        exp = META_DF[VERSION_COLS]

    pdt.assert_frame_equal(conn.index(default=default), exp, check_dtype=False)


@pytest.mark.parametrize("default", [True, False])
def test_meta(conn, default):
    # test that connection returns the correct meta dataframe
    v = "version"
    if default:
        exp = META_DF.loc[META_DF.is_default, [v] + META_COLS]
    else:
        exp = META_DF[VERSION_COLS + META_COLS].set_index(v, append=True)

    pdt.assert_frame_equal(conn.meta(default=default), exp, check_dtype=False)


@pytest.mark.parametrize("default", [True, False])
def test_properties(conn, default):
    # test that connection returns the correct properties dataframe
    obs = conn.properties(default=default)
    if default:
        exp_cols = ["version"]
        exp = META_DF.loc[META_DF.is_default, exp_cols]
    else:
        exp_cols = VERSION_COLS
        exp = META_DF[exp_cols]

    # assert that the expected audit columns are included
    for col in ["create_user", "create_date", "update_user", "update_date"]:
        assert col in obs.columns
    # assert that the values of some columns is as expected
    pdt.assert_frame_equal(obs[exp_cols], exp, check_dtype=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(variable="Primary Energy"),
        dict(scenario="scen_a", variable="Primary Energy"),
    ],
)
def test_query_year(conn, test_df_year, kwargs):
    # test reading timeseries data (`model_a` has only yearly data)
    exp = test_df_year.copy()
    for i in ["version"] + META_COLS:
        exp.set_meta(META_DF.iloc[[0, 1]][i])

    # test method via Connection
    df = conn.query(model="model_a", **kwargs)
    assert_iamframe_equal(df, exp.filter(**kwargs))

    # test top-level method
    df = read_iiasa(TEST_API, model="model_a", **kwargs)
    assert_iamframe_equal(df, exp.filter(**kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(variable="Primary Energy"),
        dict(scenario="scen_a", variable="Primary Energy"),
    ],
)
def test_query_with_subannual(conn, test_pd_df, kwargs):
    # test reading timeseries data (including subannual data)
    exp = IamDataFrame(test_pd_df, subannual="Year").append(
        MODEL_B_DF, model="model_b", scenario="scen_a", region="World"
    )
    for i in ["version"] + META_COLS:
        exp.set_meta(META_DF.iloc[[0, 1, 3]][i])

    # test method via Connection
    df = conn.query(**kwargs)
    assert_iamframe_equal(df, exp.filter(**kwargs))

    # test top-level method
    df = read_iiasa(TEST_API, **kwargs)
    assert_iamframe_equal(df, exp.filter(**kwargs))


@pytest.mark.parametrize(
    "meta",
    [
        ["string"],  # version column is added whether or not stated explicitly
        ["string", "version"],
    ],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(variable="Primary Energy"),
        dict(scenario="scen_a", variable="Primary Energy"),
    ],
)
def test_query_with_meta_arg(conn, test_pd_df, meta, kwargs):
    # test reading timeseries data (including subannual data)
    exp = IamDataFrame(test_pd_df, subannual="Year").append(
        MODEL_B_DF, model="model_b", scenario="scen_a", region="World"
    )
    for i in ["version", "string"]:
        exp.set_meta(META_DF.iloc[[0, 1, 3]][i])

    # test method via Connection
    df = conn.query(meta=meta, **kwargs)
    assert_iamframe_equal(df, exp.filter(**kwargs))

    # test top-level method
    df = read_iiasa(TEST_API, meta=meta, **kwargs)
    assert_iamframe_equal(df, exp.filter(**kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(variable="Primary Energy"),
        dict(scenario="scen_a", variable="Primary Energy"),
    ],
)
def test_query_with_meta_false(conn, test_pd_df, kwargs):
    # test reading timeseries data (including subannual data)
    exp = IamDataFrame(test_pd_df, subannual="Year").append(
        MODEL_B_DF, model="model_b", scenario="scen_a", region="World"
    )

    # test method via Connection
    df = conn.query(meta=False, **kwargs)
    assert_iamframe_equal(df, exp.filter(**kwargs))

    # test top-level method
    df = read_iiasa(TEST_API, meta=False, **kwargs)
    assert_iamframe_equal(df, exp.filter(**kwargs))


def test_query_non_default(conn, test_pd_df):
    # test reading timeseries data with non-default versions & index
    test_pd_df["subannual"] = "Year"
    test_pd_df["version"] = 1
    df = pd.concat([test_pd_df[NON_DEFAULT_DF.columns], NON_DEFAULT_DF])

    meta = META_DF.set_index("version", append=True)
    index = ["model", "scenario", "version"]
    exp = IamDataFrame(df, meta=meta, index=index, region="World")

    # test method via Connection
    df = conn.query(default=False)
    assert_iamframe_equal(df, exp)

    # test top-level method
    df = read_iiasa(TEST_API, default=False)
    assert_iamframe_equal(df, exp)
