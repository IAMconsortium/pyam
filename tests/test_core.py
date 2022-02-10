import logging
import pytest
import re
import datetime

import numpy as np
import pandas as pd
from numpy import testing as npt
from pandas import testing as pdt

from pyam import IamDataFrame, filter_by_meta, META_IDX, IAMC_IDX, sort_data
from pyam.core import _meta_idx
from pyam.utils import isstr
from pyam.testing import assert_iamframe_equal


df_filter_by_meta_matching_idx = pd.DataFrame(
    [
        ["model_a", "scen_a", "region_1", 1],
        ["model_a", "scen_a", "region_2", 2],
        ["model_a", "scen_b", "region_3", 3],
    ],
    columns=["model", "scenario", "region", "col"],
)


df_filter_by_meta_nonmatching_idx = pd.DataFrame(
    [
        ["model_a", "scen_c", "region_1", 1, 2],
        ["model_a", "scen_c", "region_2", 2, 3],
        ["model_a", "scen_b", "region_3", 3, 4],
    ],
    columns=["model", "scenario", "region", 2010, 2020],
).set_index(["model", "region"])

META_DF = pd.DataFrame(
    [
        ["model_a", "scen_a", 1, False],
        ["model_a", "scen_b", np.nan, False],
        ["model_a", "scen_c", 2, False],
    ],
    columns=META_IDX + ["foo", "exclude"],
).set_index(META_IDX)


df_empty = pd.DataFrame([], columns=IAMC_IDX + [2005, 2010])


def test_init_df_with_index(test_pd_df):
    df = IamDataFrame(test_pd_df.set_index(META_IDX))
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), test_pd_df)


def test_init_from_iamdf(test_df_year):
    # casting an IamDataFrame instance again works
    df = IamDataFrame(test_df_year)

    # inplace-operations on the new object have effects on the original object
    df.rename(scenario={"scen_a": "scen_foo"}, inplace=True)
    assert test_df_year.scenario == ["scen_b", "scen_foo"]

    # overwrites on the new object do not have effects on the original object
    df = df.rename(scenario={"scen_foo": "scen_bar"})
    assert df.scenario == ["scen_b", "scen_bar"]
    assert test_df_year.scenario == ["scen_b", "scen_foo"]


def test_init_from_iamdf_raises(test_df_year):
    # casting an IamDataFrame instance again with extra args fails
    match = "Invalid arguments for initializing from IamDataFrame: {'model': 'foo'}"
    with pytest.raises(ValueError, match=match):
        IamDataFrame(test_df_year, model="foo")


def test_init_df_with_float_cols_raises(test_pd_df):
    _test_df = test_pd_df.rename(columns={2005: 2005.5, 2010: 2010.0})
    pytest.raises(ValueError, IamDataFrame, data=_test_df)


def test_init_df_with_duplicates_raises(test_df):
    _df = test_df.timeseries()
    _df = _df.append(_df.iloc[0]).reset_index()
    match = "0  model_a   scen_a  World  Primary Energy  EJ/yr"
    with pytest.raises(ValueError, match=match):
        IamDataFrame(_df)


@pytest.mark.parametrize("illegal_value", [" ", "x0.5"])
def test_init_df_with_illegal_values_raises(test_pd_df, illegal_value):
    # values that cannot be cast to float should raise a value error and be specified by
    # index for user
    test_pd_df.loc[0, 2005] = illegal_value
    msg = (
        f'.*string "{illegal_value}" in `data`:'
        r"(\n.*){2}model_a.*scen_a.*World.*Primary Energy.*EJ/yr.*2005"
    )

    with pytest.raises(ValueError, match=msg):
        IamDataFrame(test_pd_df)


def test_init_df_with_na_scenario(test_pd_df):
    # missing values in an index dimension raises an error
    test_pd_df.loc[1, "scenario"] = np.nan
    pytest.raises(ValueError, IamDataFrame, data=test_pd_df)


def test_init_df_with_float_cols(test_pd_df):
    _test_df = test_pd_df.rename(columns={2005: 2005.0, 2010: 2010.0})
    obs = IamDataFrame(_test_df).timeseries().reset_index()
    pd.testing.assert_series_equal(obs[2005], test_pd_df[2005])


def test_init_df_from_timeseries(test_df):
    df = IamDataFrame(test_df.timeseries())
    pd.testing.assert_frame_equal(df.timeseries(), test_df.timeseries())


def test_init_df_with_extra_col(test_pd_df):
    tdf = test_pd_df.copy()

    extra_col = "climate model"
    extra_value = "scm_model"
    tdf[extra_col] = extra_value

    df = IamDataFrame(tdf)

    # check that timeseries data is as expected
    obs = df.timeseries().reset_index()
    exp = tdf[obs.columns]  # get the columns into the right order
    pd.testing.assert_frame_equal(obs, exp)


def test_init_df_with_meta(test_pd_df):
    # pass explicit meta dataframe with a scenario that doesn't exist in data
    df = IamDataFrame(test_pd_df, meta=META_DF.iloc[[0, 2]][["foo"]])

    # check that scenario not existing in data is removed during initialization
    pd.testing.assert_frame_equal(df.meta, META_DF.iloc[[0, 1]])


def test_init_df_with_meta_incompatible_index(test_pd_df):
    # define a meta dataframe with a non-standard index
    index = ["source", "scenario"]
    meta = pd.DataFrame(
        [False, False, False], columns=["exclude"], index=META_DF.index.rename(index)
    )

    # assert that using an incompatible index for the meta arg raises
    match = "Incompatible `index=\['model', 'scenario'\]` with `meta` *."
    with pytest.raises(ValueError, match=match):
        IamDataFrame(test_pd_df, meta=meta)


def test_init_df_with_custom_index(test_pd_df):
    # rename 'model' column and add a version column to the dataframe
    test_pd_df.rename(columns={"model": "source"}, inplace=True)
    test_pd_df["version"] = [1, 2, 3]

    # initialize with custom index columns, check that index is set correctly
    index = ["source", "scenario", "version"]
    df = IamDataFrame(test_pd_df, index=index)
    assert df.index.names == index

    # check that index attributes were set correctly and that df.model fails
    assert df.source == ["model_a"]
    assert df.version == [1, 2, 3]
    with pytest.raises(KeyError, match="Index `model` does not exist!"):
        df.model


def test_init_empty_message(caplog):
    IamDataFrame(data=df_empty)
    drop_message = "Formatted data is empty!"
    message_idx = caplog.messages.index(drop_message)
    assert caplog.records[message_idx].levelno == logging.WARNING


def test_init_with_column_conflict(test_pd_df):
    # add a column to the timeseries data with a conflict to the meta attribute
    test_pd_df["meta"] = "foo"

    # check that initialising an instance with an extra-column `meta` raises
    msg = re.compile(r"Column name \['meta'\] is illegal for timeseries data.")
    with pytest.raises(ValueError, match=msg):
        IamDataFrame(test_pd_df)

    # check that recommended fix works
    df = IamDataFrame(test_pd_df, meta_1="meta")
    assert df.meta_1 == ["foo"]


def test_set_meta_with_column_conflict(test_df_year):
    # check that setting a `meta` column with a name conflict raises
    msg = "Column model already exists in `data`!"
    with pytest.raises(ValueError, match=msg):
        test_df_year.set_meta(name="model", meta="foo")

    msg = "Name meta is illegal for meta indicators!"
    with pytest.raises(ValueError, match=msg):
        test_df_year.set_meta(name="meta", meta="foo")


def test_print(test_df_year):
    """Assert that `print(IamDataFrame)` (and `info()`) returns as expected"""
    exp = "\n".join(
        [
            "<class 'pyam.core.IamDataFrame'>",
            "Index:",
            " * model    : model_a (1)",
            " * scenario : scen_a, scen_b (2)",
            "Timeseries data coordinates:",
            "   region   : World (1)",
            "   variable : Primary Energy, Primary Energy|Coal (2)",
            "   unit     : EJ/yr (1)",
            "   year     : 2005, 2010 (2)",
            "Meta indicators:",
            "   exclude (bool) False (1)",
            "   number (int64) 1, 2 (2)",
            "   string (object) foo, nan (2)",
        ]
    )
    obs = test_df_year.info()
    assert obs == exp


def test_print_empty(test_df_year):
    """Assert that `print(IamDataFrame)` (and `info()`) returns as expected"""
    exp = "\n".join(
        [
            "<class 'pyam.core.IamDataFrame'>",
            "Index:",
            " * model    : (0)",
            " * scenario : (0)",
            "Timeseries data coordinates:",
            "   region   : (0)",
            "   variable : (0)",
            "   unit     : (0)",
            "   year     : (0)",
            "Meta indicators:",
            "   exclude (bool) (0)",
            "   number (int64) (0)",
            "   string (object) (0)",
        ]
    )
    obs = test_df_year.filter(model="foo").info()
    assert obs == exp


def test_as_pandas(test_df):
    # test that `as_pandas()` returns the right columns
    df = test_df.copy()
    df.set_meta(["foo", "bar"], name="string")
    df.set_meta([1, 2], name="number")

    # merge all columns (default)
    obs = df.as_pandas()
    cols = ["string", "number"]
    assert all(i in obs.columns for i in cols)  # assert relevant columns exist

    exp = pd.concat([pd.DataFrame([["foo", 1]] * 4), pd.DataFrame([["bar", 2]] * 2)])
    npt.assert_array_equal(obs[cols], exp)  # assert meta columns are merged

    # merge only one column
    obs = df.as_pandas(["string"])
    assert "string" in obs.columns
    assert "number" not in obs.columns
    npt.assert_array_equal(obs["string"], ["foo"] * 4 + ["bar"] * 2)

    # do not merge any columns
    npt.assert_array_equal(df.as_pandas(False), df.data)


def test_empty_attribute(test_df_year):
    assert not test_df_year.empty
    assert test_df_year.filter(model="foo").empty


def test_equals(test_df_year):
    test_df_year.set_meta([1, 2], name="test")

    # assert that a copy (with changed index-sort) is equal
    df = test_df_year.copy()
    df._data = df._data.sort_values()
    assert test_df_year.equals(df)

    # assert that adding a new timeseries is not equal
    df = test_df_year.rename(variable={"Primary Energy": "foo"}, append=True)
    assert not test_df_year.equals(df)

    # assert that adding a new meta indicator is not equal
    df = test_df_year.copy()
    df.set_meta(["foo", " bar"], name="string")
    assert not test_df_year.equals(df)


def test_equals_raises(test_pd_df):
    df = IamDataFrame(test_pd_df)
    pytest.raises(ValueError, df.equals, test_pd_df)


@pytest.mark.parametrize("column", ["model", "variable", "value"])
def test_get_item(test_df, column):
    """Assert that getting a column from `data` via the direct getter works"""
    pdt.assert_series_equal(test_df[column], test_df.data[column])


def test_index(test_df_year):
    # assert that the correct index is shown for the IamDataFrame
    exp = pd.MultiIndex.from_arrays(
        [["model_a"] * 2, ["scen_a", "scen_b"]], names=["model", "scenario"]
    )
    pd.testing.assert_index_equal(test_df_year.index, exp)


def test_index_attributes(test_df):
    # assert that the index and data column attributes are set correctly
    assert test_df.model == ["model_a"]
    assert test_df.scenario == ["scen_a", "scen_b"]
    assert test_df.region == ["World"]
    assert test_df.variable == ["Primary Energy", "Primary Energy|Coal"]
    assert test_df.unit == ["EJ/yr"]
    if test_df.time_col == "year":
        assert test_df.year == [2005, 2010]
    else:
        match = "'IamDataFrame' object has no attribute 'year'"
        with pytest.raises(AttributeError, match=match):
            test_df.year
    assert test_df.time.equals(pd.Index(test_df.data[test_df.time_col].unique()))


def test_index_attributes_extra_col(test_pd_df):
    test_pd_df["subannual"] = ["summer", "summer", "winter"]
    df = IamDataFrame(test_pd_df)
    assert df.subannual == ["summer", "winter"]


def test_unit_mapping(test_pd_df):
    """assert that the `unit_mapping` returns the expected dictionary"""
    test_pd_df.loc[2, "unit"] = "foo"  # replace unit of one row of Primary Energy data
    obs = IamDataFrame(test_pd_df).unit_mapping

    assert obs == {"Primary Energy": ["EJ/yr", "foo"], "Primary Energy|Coal": "EJ/yr"}


def test_dimensions(test_df):
    """Assert that the dimensions attribute works as expected"""
    assert test_df.dimensions == IAMC_IDX + [test_df.time_col]


def test_get_data_column(test_df):
    """Assert that getting a column from the `data` dataframe works"""

    obs = test_df.get_data_column("model")
    pdt.assert_series_equal(obs, pd.Series(["model_a"] * 6, name="model"))

    obs = test_df.get_data_column(test_df.time_col)
    pdt.assert_series_equal(obs, test_df.data[test_df.time_col])


def test_filter_empty_df():
    # test for issue seen in #254
    df = IamDataFrame(data=df_empty)
    obs = df.filter(variable="foo")
    assert len(obs) == 0


def test_filter_variable_and_depth(test_df):
    obs = test_df.filter(variable="*rimary*C*", level=0).variable
    assert obs == ["Primary Energy|Coal"]

    obs = test_df.filter(variable="*rimary*C*", level=1).variable
    assert obs == []


def test_variable_depth_0_keep_false(test_df):
    obs = test_df.filter(level=0, keep=False).variable
    assert obs == ["Primary Energy|Coal"]


def test_variable_depth_raises(test_df):
    pytest.raises(ValueError, test_df.filter, level="1/")


def test_timeseries(test_df):
    dct = {
        "model": ["model_a"] * 2,
        "scenario": ["scen_a"] * 2,
        "years": [2005, 2010],
        "value": [1, 6],
    }
    exp = pd.DataFrame(dct).pivot_table(
        index=["model", "scenario"], columns=["years"], values="value"
    )
    obs = test_df.filter(scenario="scen_a", variable="Primary Energy").timeseries()
    npt.assert_array_equal(obs, exp)


def test_timeseries_empty_raises(test_df_year):
    """Calling `timeseries()` on an empty IamDataFrame raises"""
    _df = test_df_year.filter(model="foo")
    with pytest.raises(ValueError, match="This IamDataFrame is empty!"):
        _df.timeseries()


def test_timeseries_time_iamc_raises(test_df_time):
    """Calling `timeseries(iamc_index=True)` on a continuous-time IamDataFrame raises"""
    match = "Cannot use `iamc_index=True` with 'datetime' time-domain!"
    with pytest.raises(ValueError, match=match):
        test_df_time.timeseries(iamc_index=True)


def test_timeseries_to_iamc_index(test_pd_df, test_df_year):
    """Reducing timeseries() of an IamDataFrame with extra-columns to IAMC-index"""
    test_pd_df["foo"] = "bar"
    exta_col_df = IamDataFrame(test_pd_df)
    assert exta_col_df.extra_cols == ["foo"]

    # assert that reducing to IAMC-columns (dropping extra-columns) with timeseries()
    obs = exta_col_df.timeseries(iamc_index=True)
    exp = test_df_year.timeseries()
    pdt.assert_frame_equal(obs, exp)


def test_timeseries_to_iamc_index_duplicated_raises(test_pd_df):
    """Assert that using `timeseries(iamc_index=True)` raises if there are duplicates"""
    test_pd_df = pd.concat([test_pd_df, test_pd_df])
    # adding an extra-col creates a unique index
    test_pd_df["foo"] = ["bar", "bar", "bar", "baz", "baz", "baz"]
    exta_col_df = IamDataFrame(test_pd_df)
    assert exta_col_df.extra_cols == ["foo"]

    # dropping the extra-column by setting `iamc_index=True` creates duplicated index
    match = "Index contains duplicate entries, cannot reshape"
    with pytest.raises(ValueError, match=match):
        exta_col_df.timeseries(iamc_index=True)


def test_pivot_table(test_df):
    dct = {
        "model": ["model_a"] * 2,
        "scenario": ["scen_a"] * 2,
        "years": [2005, 2010],
        "value": [1, 6],
    }
    args = dict(index=["model", "scenario"], columns=["years"], values="value")
    exp = pd.DataFrame(dct).pivot_table(**args)
    obs = test_df.filter(scenario="scen_a", variable="Primary Energy").pivot_table(
        index=["model", "scenario"], columns=test_df.time_col, aggfunc="sum"
    )
    npt.assert_array_equal(obs, exp)


def test_pivot_table_raises(test_df):
    # using the same dimension in both index and columns raises an error
    pytest.raises(
        ValueError,
        test_df.pivot_table,
        index=["model", "scenario"] + [test_df.time_col],
        columns=test_df.time_col,
    )


def test_filter_meta_index(test_df):
    obs = test_df.filter(scenario="scen_b").meta.index
    exp = pd.MultiIndex(
        levels=[["model_a"], ["scen_b"]], codes=[[0], [0]], names=["model", "scenario"]
    )
    pd.testing.assert_index_equal(obs, exp)


def test_meta_idx(test_df):
    # assert that the `drop_duplicates()` in `_meta_idx()` returns right length
    assert len(_meta_idx(test_df.data)) == 2


def test_interpolate(test_pd_df):
    _df = test_pd_df.copy()
    _df["foo"] = ["bar", "baz", 2]  # add extra_col (check for #351)
    df = IamDataFrame(_df)
    obs = df.interpolate(2007, inplace=False).filter(year=2007)._data.values
    npt.assert_allclose(obs, [3, 1.5, 4])

    # redo the interpolation and check that no duplicates are added
    df.interpolate(2007, inplace=False)
    assert not df._data.index.duplicated().any()

    # assert that extra_col does not have nan's (check for #351)
    assert all([True if isstr(i) else ~np.isnan(i) for i in df.foo])


def test_interpolate_time_exists(test_df_year):
    obs = test_df_year.interpolate(2005, inplace=False).filter(year=2005)._data.values
    npt.assert_allclose(obs, [1.0, 0.5, 2.0])


def test_interpolate_with_list(test_df_year):
    lst = [2007, 2008]
    obs = test_df_year.interpolate(lst, inplace=False).filter(year=lst)._data.values
    npt.assert_allclose(obs, [3, 4, 1.5, 2, 4, 5])


def test_interpolate_with_numpy_list(test_df_year):
    test_df_year.interpolate(np.r_[2007 : 2008 + 1], inplace=True)
    obs = test_df_year.filter(year=[2007, 2008])._data.values
    npt.assert_allclose(obs, [3, 4, 1.5, 2, 4, 5])


def test_interpolate_full_example():
    cols = ["model_a", "scen_a", "World"]
    df = IamDataFrame(
        pd.DataFrame(
            [
                cols + ["all", "EJ/yr", 0, 1, 6.0, 10],
                cols + ["last", "EJ/yr", 0, 0.5, 3, np.nan],
                cols + ["first", "EJ/yr", 0, np.nan, 2, 7],
                cols + ["middle", "EJ/yr", 0, 1, np.nan, 7],
                cols + ["first two", "EJ/yr", 0, np.nan, np.nan, 7],
                cols + ["last two", "EJ/yr", 0, 1, np.nan, np.nan],
            ],
            columns=IAMC_IDX + [2000, 2005, 2010, 2017],
        )
    )
    exp = IamDataFrame(
        pd.DataFrame(
            [
                cols + ["all", "EJ/yr", 0, 1, 6.0, 7.142857, 10],
                cols + ["last", "EJ/yr", 0, 0.5, 3, np.nan, np.nan],
                cols + ["first", "EJ/yr", 0, 1.0, 2, 3.428571, 7],
                cols + ["middle", "EJ/yr", 0, 1, np.nan, 4.5, 7],
                cols + ["first two", "EJ/yr", 0, 2.058824, np.nan, 4.941176, 7],
                cols + ["last two", "EJ/yr", 0, 1, np.nan, np.nan, np.nan],
            ],
            columns=IAMC_IDX + [2000, 2005, 2010, 2012, 2017],
        )
    )
    assert_iamframe_equal(df.interpolate([2005, 2012], inplace=False), exp)


def test_interpolate_extra_cols():
    # check that interpolation with non-matching extra_cols has no effect
    # (#351)
    EXTRA_COL_DF = pd.DataFrame(
        [
            ["foo", 2005, 1],
            ["foo", 2010, 2],
            ["bar", 2005, 2],
            ["bar", 2010, 3],
        ],
        columns=["extra_col", "year", "value"],
    )
    df = IamDataFrame(
        EXTRA_COL_DF,
        model="model_a",
        scenario="scen_a",
        region="World",
        variable="Primary Energy",
        unit="EJ/yr",
    )

    # create a copy from interpolation
    df2 = df.interpolate(2007, inplace=False)

    # interpolate should work as if extra_cols is in the _data index
    assert_iamframe_equal(df, df2.filter(year=2007, keep=False))
    obs = df2.filter(year=2007)._data.values
    npt.assert_allclose(obs, [2.4, 1.4])


def test_interpolate_datetimes(test_df):
    # test that interpolation also works with date-times.
    some_date = datetime.datetime(2007, 7, 1)
    if test_df.time_col == "year":
        pytest.raises(ValueError, test_df.interpolate, time=some_date)
    else:
        test_df.interpolate(some_date, inplace=True)
        obs = test_df.filter(time=some_date).data["value"].reset_index(drop=True)
        exp = pd.Series([3, 1.5, 4], name="value")
        pd.testing.assert_series_equal(obs, exp, check_less_precise=True)
        # redo the interpolation and check that no duplicates are added
        test_df.interpolate(some_date, inplace=True)
        assert not test_df.filter()._data.index.duplicated().any()


def test_filter_by_bool(test_df):
    test_df.set_meta([True, False], name="exclude")
    obs = test_df.filter(exclude=True)
    assert obs["scenario"].unique() == "scen_a"


def test_filter_by_int(test_df):
    test_df.set_meta([1, 2], name="test")
    obs = test_df.filter(test=[1, 3])
    assert obs["scenario"].unique() == "scen_a"


def _r5_regions_exp(df):
    df = df.filter(region="World", keep=False)
    data = df.data
    data["region"] = "R5MAF"
    return sort_data(data, df.dimensions)


def test_map_regions_r5(reg_df):
    obs = reg_df.map_regions("r5_region").data
    exp = _r5_regions_exp(reg_df)
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_map_regions_r5_region_col(reg_df):
    df = reg_df.filter(model="MESSAGE-GLOBIOM")
    obs = df.map_regions("r5_region", region_col="MESSAGE-GLOBIOM.REGION").data
    exp = _r5_regions_exp(df)
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_map_regions_r5_inplace(reg_df):
    exp = _r5_regions_exp(reg_df)
    reg_df.map_regions("r5_region", inplace=True)
    obs = reg_df.data
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_map_regions_r5_agg(reg_df):
    columns = reg_df.data.columns
    obs = reg_df.map_regions("r5_region", agg="sum").data

    exp = _r5_regions_exp(reg_df)
    grp = list(columns)
    grp.remove("value")
    exp = exp.groupby(grp).sum().reset_index()
    exp = exp[columns]
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_48a():
    # tests fix for #48 mapping many->few
    df = IamDataFrame(
        pd.DataFrame(
            [
                ["model", "scen", "SSD", "var", "unit", 1, 6],
                ["model", "scen", "SDN", "var", "unit", 2, 7],
                ["model", "scen1", "SSD", "var", "unit", 2, 7],
                ["model", "scen1", "SDN", "var", "unit", 2, 7],
            ],
            columns=["model", "scenario", "region", "variable", "unit", 2005, 2010],
        )
    )

    exp = _r5_regions_exp(df)
    columns = df.data.columns
    grp = list(columns)
    grp.remove("value")
    exp = exp.groupby(grp).sum().reset_index()
    exp = exp[columns]

    obs = df.map_regions("r5_region", region_col="iso", agg="sum").data

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_48b():
    # tests fix for #48 mapping few->many

    exp = IamDataFrame(
        pd.DataFrame(
            [
                ["model", "scen", "SSD", "var", "unit", 1, 6],
                ["model", "scen", "SDN", "var", "unit", 1, 6],
                ["model", "scen1", "SSD", "var", "unit", 2, 7],
                ["model", "scen1", "SDN", "var", "unit", 2, 7],
            ],
            columns=["model", "scenario", "region", "variable", "unit", 2005, 2010],
        )
    ).data

    df = IamDataFrame(
        pd.DataFrame(
            [
                ["model", "scen", "R5MAF", "var", "unit", 1, 6],
                ["model", "scen1", "R5MAF", "var", "unit", 2, 7],
            ],
            columns=["model", "scenario", "region", "variable", "unit", 2005, 2010],
        )
    )
    obs = df.map_regions("iso", region_col="r5_region").data
    obs = sort_data(obs[obs.region.isin(["SSD", "SDN"])], df.dimensions)

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_48c():
    # tests fix for #48 mapping few->many, dropping duplicates

    exp = IamDataFrame(
        pd.DataFrame(
            [
                ["model", "scen", "AGO", "var", "unit", 1, 6],
                ["model", "scen1", "AGO", "var", "unit", 2, 7],
            ],
            columns=["model", "scenario", "region", "variable", "unit", 2005, 2010],
        )
    ).data.reset_index(drop=True)

    df = IamDataFrame(
        pd.DataFrame(
            [
                ["model", "scen", "R5MAF", "var", "unit", 1, 6],
                ["model", "scen1", "R5MAF", "var", "unit", 2, 7],
            ],
            columns=["model", "scenario", "region", "variable", "unit", 2005, 2010],
        )
    )
    obs = df.map_regions("iso", region_col="r5_region", remove_duplicates=True).data
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_pd_filter_by_meta(test_df):
    data = df_filter_by_meta_matching_idx.set_index(["model", "region"])

    test_df.set_meta([True, False], "boolean")
    test_df.set_meta(0, "integer")

    obs = filter_by_meta(data, test_df, join_meta=True, boolean=True, integer=None)
    obs = obs.reindex(columns=["scenario", "col", "boolean", "integer"])

    exp = data.iloc[0:2].copy()
    exp["boolean"] = True
    exp["integer"] = 0

    pd.testing.assert_frame_equal(obs, exp)


def test_pd_filter_by_meta_no_index(test_df):
    data = df_filter_by_meta_matching_idx

    test_df.set_meta([True, False], "boolean")
    test_df.set_meta(0, "int")

    obs = filter_by_meta(data, test_df, join_meta=True, boolean=True, int=None)
    obs = obs.reindex(columns=META_IDX + ["region", "col", "boolean", "int"])

    exp = data.iloc[0:2].copy()
    exp["boolean"] = True
    exp["int"] = 0

    pd.testing.assert_frame_equal(obs, exp)


def test_pd_filter_by_meta_nonmatching_index(test_df):
    data = df_filter_by_meta_nonmatching_idx
    test_df.set_meta(["a", "b"], "string")

    obs = filter_by_meta(data, test_df, join_meta=True, string="b")
    obs = obs.reindex(columns=["scenario", 2010, 2020, "string"])

    exp = data.iloc[2:3].copy()
    exp["string"] = "b"

    pd.testing.assert_frame_equal(obs, exp)


def test_pd_join_by_meta_nonmatching_index(test_df):
    data = df_filter_by_meta_nonmatching_idx
    test_df.set_meta(["a", "b"], "string")

    obs = filter_by_meta(data, test_df, join_meta=True, string=None)
    obs = obs.reindex(columns=["scenario", 2010, 2020, "string"])

    exp = data.copy()
    exp["string"] = [np.nan, np.nan, "b"]

    pd.testing.assert_frame_equal(obs.sort_index(level=1), exp)


def test_normalize(test_df):
    exp = test_df.data.copy().reset_index(drop=True)
    exp.loc[1::2, "value"] /= exp["value"][::2].values
    exp.loc[::2, "value"] /= exp["value"][::2].values
    if "year" in test_df.data:
        obs = test_df.normalize(year=2005).data.reset_index(drop=True)
    else:
        obs = test_df.normalize(time=datetime.datetime(2005, 6, 17)).data.reset_index(
            drop=True
        )
    pd.testing.assert_frame_equal(obs, exp)


def test_normalize_not_time(test_df):
    pytest.raises(ValueError, test_df.normalize, variable="foo")
    pytest.raises(ValueError, test_df.normalize, year=2015, variable="foo")
