import os
from pathlib import Path
import pytest
import pandas as pd
import pandas.testing as pdt
import numpy as np
import numpy.testing as npt
import yaml

from pyam import IamDataFrame, iiasa, lazy_read_iiasa, read_iiasa, META_IDX
from pyam.testing import assert_iamframe_equal

from .conftest import META_COLS, IIASA_UNAVAILABLE, TEST_API, TEST_API_NAME


if IIASA_UNAVAILABLE:
    pytest.skip("IIASA database API unavailable", allow_module_level=True)


# TODO environment variables are currently not set up on GitHub Actions
TEST_ENV_USER = "IIASA_CONN_TEST_USER"
TEST_ENV_PW = "IIASA_CONN_TEST_PW"
CONN_ENV_AVAILABLE = TEST_ENV_USER in os.environ and TEST_ENV_PW in os.environ
CONN_ENV_REASON = "Requires env variables defined: {} and {}".format(
    TEST_ENV_USER, TEST_ENV_PW
)


FILTER_ARGS = [{}, dict(model="model_a"), dict(model=["model_a"]), dict(model="m*_a")]

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
    match = "You do not have access to instance 'foo' or it does not exist."
    with pytest.raises(ValueError, match=match):
        iiasa.Connection("foo")


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


def test_conn_nonexisting_creds_file():
    # pointing to non-existing creds file raises
    with pytest.raises(FileNotFoundError):
        iiasa.Connection(TEST_API, creds="foo")


@pytest.mark.parametrize(
    "creds, match",
    [
        (dict(username="user", password="password"), "Credentials not valid "),
        (dict(username="user"), "Unknown API error:*."),
    ],
)
def test_conn_invalid_creds_file(creds, match, tmpdir):
    # invalid credentials raises the expected errors
    with open(tmpdir / "creds.yaml", mode="w") as f:
        yaml.dump(creds, f)
    with pytest.raises(ValueError, match=match):
        iiasa.Connection(TEST_API, creds=Path(tmpdir) / "creds.yaml")


def test_conn_cleartext_creds_raises():
    # connecting with clear-text credentials raises an error
    match = "Passing credentials as clear-text is not allowed."
    with pytest.raises(DeprecationWarning, match=match):
        iiasa.Connection(TEST_API, creds=("user", "password"))


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


@pytest.mark.parametrize("kwargs", FILTER_ARGS)
@pytest.mark.parametrize("default", [True, False])
def test_index(conn, kwargs, default):
    # test that connection returns the correct index
    obs = conn.index(default=default, **kwargs)

    if default:
        exp = META_DF.loc[META_DF.is_default, ["version"]]
        if kwargs:
            exp = exp.iloc[0:2]
    else:
        exp = META_DF[VERSION_COLS]
        if kwargs:
            exp = exp.iloc[0:3]

    pdt.assert_frame_equal(obs, exp, check_dtype=False)


def test_index_empty(conn):
    # test that an empty filter does not yield an error
    # solves https://github.com/IAMconsortium/pyam/issues/676
    conn.index(model="foo").empty


def test_index_illegal_column(conn):
    # test that filtering by an illegal column raises an error
    with pytest.raises(ValueError, match="Invalid filter: 'foo'"):
        conn.index(foo="bar")


@pytest.mark.parametrize("kwargs", FILTER_ARGS)
@pytest.mark.parametrize("default", [True, False])
def test_meta(conn, kwargs, default):
    # test that connection returns the correct meta dataframe
    obs = conn.meta(default=default, **kwargs)

    v = "version"
    if default:
        exp = META_DF.loc[META_DF.is_default, [v] + META_COLS]
        if kwargs:
            exp = exp.iloc[0:2]
    else:
        exp = META_DF[VERSION_COLS + META_COLS].set_index(v, append=True)
        if kwargs:
            exp = exp.iloc[0:3]

    pdt.assert_frame_equal(obs, exp, check_dtype=False)


@pytest.mark.parametrize("kwargs", FILTER_ARGS)
@pytest.mark.parametrize("default", [True, False])
def test_properties(conn, kwargs, default):
    # test that connection returns the correct properties dataframe
    obs = conn.properties(default, **kwargs)

    if default:
        exp_cols = ["version"]
        exp = META_DF.loc[META_DF.is_default, exp_cols]
        if kwargs:
            exp = exp.iloc[0:2]
    else:
        exp_cols = VERSION_COLS
        exp = META_DF[exp_cols]
        if kwargs:
            exp = exp.iloc[0:3]

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
    assert df.model == ["model_a"]
    assert_iamframe_equal(df, exp.filter(**kwargs))

    # test top-level method
    df = read_iiasa(TEST_API, model="model_a", **kwargs)
    assert df.model == ["model_a"]
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


def test_query_empty_response(conn):
    """Check that querying with an empty response returns an empty IamDataFrame"""
    # solves https://github.com/IAMconsortium/pyam/issues/676
    assert conn.query(model="foo").empty


def test_lazy_read(tmpdir):
    tmp_file = tmpdir / "test_database.csv"
    df = lazy_read_iiasa(tmp_file, TEST_API, model="model_a")
    writetime = os.path.getmtime(tmp_file)
    assert df.model == ["model_a"]
    # This is read from the file, so the filter is not applied.
    df2 = lazy_read_iiasa(tmp_file, TEST_API)
    assert df.data.equals(df2.data)
    # If requesting with an inconsistent filter, get nothing back. Strings and filters
    # work interchangably.
    tmp_file = str(tmp_file)
    df_newfilt = lazy_read_iiasa(tmp_file, TEST_API, model="model_b")
    assert df_newfilt.empty
    assert writetime == os.path.getmtime(tmp_file)
    # Filter correctly applied if the file is deleted
    os.remove(tmp_file)
    df_newfilt = lazy_read_iiasa(tmp_file, TEST_API, model="model_b")
    assert df_newfilt.model == ["model_b"]
    assert os.path.getmtime(tmp_file) > writetime
    # file can also be xls or xlsx
    xlsx_file = tmpdir / "test_database.xlsx"
    df_xlsx = lazy_read_iiasa(xlsx_file, TEST_API, model="model_b")
    assert df_newfilt.equals(df_xlsx)
    xls_file = tmpdir / "test_database.xls"
    df_xls = lazy_read_iiasa(xls_file, TEST_API, model="model_b")
    assert df_xls.equals(df_xlsx)
