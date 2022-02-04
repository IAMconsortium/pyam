import pytest
import pandas as pd
from pyam import IamDataFrame, compare


# when making any updates to this file,
# please also update the `data_table_formats` tutorial notebook!


def test_cast_from_value_col(test_df_year):
    df_with_value_cols = pd.DataFrame(
        [
            ["model_a", "scen_a", "World", "EJ/yr", 2005, 1, 0.5],
            ["model_a", "scen_a", "World", "EJ/yr", 2010, 6.0, 3],
            ["model_a", "scen_b", "World", "EJ/yr", 2005, 2, None],
            ["model_a", "scen_b", "World", "EJ/yr", 2010, 7, None],
        ],
        columns=[
            "model",
            "scenario",
            "region",
            "unit",
            "year",
            "Primary Energy",
            "Primary Energy|Coal",
        ],
    )
    df = IamDataFrame(
        df_with_value_cols, value=["Primary Energy", "Primary Energy|Coal"]
    )

    assert compare(test_df_year, df).empty
    pd.testing.assert_frame_equal(df.data, test_df_year.data, check_like=True)


def test_cast_from_value_col_and_args(test_df_year):
    # checks for issue [#210](https://github.com/IAMconsortium/pyam/issues/210)
    df_with_value_cols = pd.DataFrame(
        [
            ["scen_a", "World", "EJ/yr", 2005, 1, 0.5],
            ["scen_a", "World", "EJ/yr", 2010, 6.0, 3],
            ["scen_b", "World", "EJ/yr", 2005, 2, None],
            ["scen_b", "World", "EJ/yr", 2010, 7, None],
        ],
        columns=[
            "scenario",
            "iso",
            "unit",
            "year",
            "Primary Energy",
            "Primary Energy|Coal",
        ],
    )
    df = IamDataFrame(
        df_with_value_cols,
        model="model_a",
        region="iso",
        value=["Primary Energy", "Primary Energy|Coal"],
    )

    assert compare(test_df_year, df).empty
    pd.testing.assert_frame_equal(df.data, test_df_year.data, check_like=True)


def test_cast_with_model_arg_raises():
    df = pd.DataFrame(
        [
            ["model_a", "scen_a", "World", "EJ/yr", 2005, 1, 0.5],
        ],
        columns=[
            "model",
            "scenario",
            "region",
            "unit",
            "year",
            "Primary Energy",
            "Primary Energy|Coal",
        ],
    )
    pytest.raises(ValueError, IamDataFrame, df, model="foo")


def test_cast_with_model_arg(test_df):
    df = test_df.timeseries().reset_index()
    df.rename(columns={"model": "foo"}, inplace=True)

    df = IamDataFrame(df, model="foo")
    assert compare(test_df, df).empty
    pd.testing.assert_frame_equal(df.data, test_df.data)


def test_cast_by_column_concat(test_df_year):
    df = pd.DataFrame(
        [
            ["scen_a", "World", "Primary Energy", None, "EJ/yr", 1, 6.0],
            ["scen_a", "World", "Primary Energy", "Coal", "EJ/yr", 0.5, 3],
            ["scen_b", "World", "Primary Energy", None, "EJ/yr", 2, 7],
        ],
        columns=["scenario", "region", "var_1", "var_2", "unit", 2005, 2010],
    )

    df = IamDataFrame(df, model="model_a", variable=["var_1", "var_2"])

    assert compare(test_df_year, df).empty
    pd.testing.assert_frame_equal(df.data, test_df_year.data, check_like=True)


def test_cast_with_variable_and_value(test_df):
    pe_df = test_df.filter(variable="Primary Energy")
    df = pe_df.data.rename(columns={"value": "lvl"}).drop("variable", axis=1)

    df = IamDataFrame(df, variable="Primary Energy", value="lvl")

    assert compare(pe_df, df).empty
    pd.testing.assert_frame_equal(df.data, pe_df.data.reset_index(drop=True))


def test_cast_from_r_df(test_pd_df):
    df = test_pd_df.copy()
    # last two columns are years
    df.columns = list(df.columns[:-2]) + ["X{}".format(c) for c in df.columns[-2:]]
    obs = IamDataFrame(df)
    exp = IamDataFrame(test_pd_df)
    assert compare(obs, exp).empty
    pd.testing.assert_frame_equal(obs.data, exp.data)


def test_cast_from_r_df_err(test_pd_df):
    df = test_pd_df.copy()
    # last two columns are years
    df.columns = list(df.columns[:-2]) + ["Xfoo", "Xbar"]
    pytest.raises(ValueError, IamDataFrame, df)
