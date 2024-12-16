import datetime
import logging

import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt
from pandas import testing as pdt

from pyam import IamDataFrame, filter_by_meta
from pyam.core import _meta_idx
from pyam.utils import IAMC_IDX, META_IDX

from .conftest import TEST_DF

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
        ["model_a", "scen_a", 1, "foo"],
        ["model_a", "scen_b", np.nan, "bar"],
        ["model_a", "scen_c", 2, "baz"],
    ],
    columns=META_IDX + ["number", "string"],
).set_index(META_IDX)

df_empty = pd.DataFrame([], columns=IAMC_IDX + [2005, 2010])


@pytest.mark.parametrize("index", (None, META_IDX, ["model"]))
def test_init_df_with_non_default_index(test_pd_df, index):
    """Casting to IamDataFrame and returning as `timeseries()` yields original frame"""

    # set a value to `nan` to check that timeseries columns are ordered correctly
    test_pd_df.loc[0, 2010] = np.nan

    # any number of columns can be set as index
    df = test_pd_df.copy() if index is None else test_pd_df.set_index(index)
    obs = IamDataFrame(df).timeseries()
    pdt.assert_frame_equal(obs, test_pd_df.set_index(IAMC_IDX), check_column_type=False)


def test_init_df_unsorted(test_pd_df):
    """Casting unsorted timeseries data does not sort on init"""

    columns = IAMC_IDX + list(test_pd_df.columns[[6, 5]])
    unsorted_data = test_pd_df.iloc[[2, 0, 1]][columns]
    df = IamDataFrame(unsorted_data)

    # `data` is not sorted
    assert list(df.data.scenario.unique()) == ["scen_b", "scen_a"]
    assert list(df.data.year.unique()) == [2010, 2005]
    assert not df._data.index.is_monotonic_increasing


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
    _df = test_df.timeseries().reset_index()
    _df = pd.concat([_df, _df.iloc[0].to_frame().T])
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
    msg = (
        r"Empty cells in `data` \(columns: 'scenario'\):"
        r"(\n.*){2}model_a.*NaN.*World.*Primary Energy|Coal.*EJ/yr.*2005.*"
    )
    with pytest.raises(ValueError, match=msg):
        IamDataFrame(test_pd_df)


def test_init_df_with_float_cols(test_pd_df):
    _test_df = test_pd_df.rename(columns={2005: 2005.0, 2010: 2010.0})
    obs = IamDataFrame(_test_df).timeseries().reset_index()
    pdt.assert_series_equal(obs[2005], test_pd_df[2005])


def test_init_df_from_timeseries(test_df):
    df = IamDataFrame(test_df.timeseries())
    pdt.assert_frame_equal(df.timeseries(), test_df.timeseries())


def test_init_df_from_timeseries_unused_levels(test_df):
    # this test guards against regression for the bug
    # reported in https://github.com/IAMconsortium/pyam/issues/762

    for (model, scenario), data in test_df.timeseries().groupby(["model", "scenario"]):
        # we're only interested in the second model-scenario combination
        if model == "model_a" and scenario == "scen_b":
            df = IamDataFrame(data)

    # pandas 2.0 does not remove unused levels (here: "Primary Energy|Coal") in groupby
    # we check that unused levels are removed at initialization of the IamDataFrame
    assert df.variable == ["Primary Energy"]


def test_init_df_with_extra_col(test_pd_df):
    tdf = test_pd_df.copy()

    extra_col = "climate model"
    extra_value = "scm_model"
    tdf[extra_col] = extra_value

    df = IamDataFrame(tdf)

    # check that timeseries data is as expected
    obs = df.timeseries().reset_index()
    exp = tdf[obs.columns]  # get the columns into the right order
    pdt.assert_frame_equal(obs, exp)


def test_init_df_with_meta_with_index(test_pd_df):
    # pass indexed meta dataframe with a scenario that doesn't exist in data
    df = IamDataFrame(test_pd_df, meta=META_DF)

    # check that scenario not existing in data is removed during initialization
    pdt.assert_frame_equal(df.meta, META_DF.iloc[[0, 1]])
    assert df.scenario == ["scen_a", "scen_b"]


def test_init_df_with_meta_no_index(test_pd_df):
    # pass meta without index with a scenario that doesn't exist in data
    df = IamDataFrame(test_pd_df, meta=META_DF.reset_index())

    # check that scenario not existing in data is removed during initialization
    pdt.assert_frame_equal(df.meta, META_DF.iloc[[0, 1]])
    assert df.scenario == ["scen_a", "scen_b"]


def test_init_df_with_meta_key_value(test_pd_df):
    # pass meta with key-value columns with a scenario that doesn't exist in data

    meta_df = pd.DataFrame(
        [
            ["model_a", "scen_a", "number", 1],
            ["model_a", "scen_a", "string", "foo"],
            ["model_a", "scen_b", "string", "bar"],
            ["model_a", "scen_c", "number", 2],
        ],
        columns=META_IDX + ["key", "value"],
    )
    df = IamDataFrame(test_pd_df, meta=meta_df)

    # check that scenario not existing in data is removed during initialization
    pdt.assert_frame_equal(df.meta, META_DF.iloc[[0, 1]], check_dtype=False)
    assert df.scenario == ["scen_a", "scen_b"]


def test_init_df_with_meta_exclude_raises(test_pd_df):
    # pass explicit meta dataframe with a legacy "exclude" column
    meta = META_DF.copy()
    meta["exclude"] = False
    with pytest.raises(ValueError, match="Illegal columns in `meta`: 'exclude'"):
        IamDataFrame(test_pd_df, meta=meta)


def test_init_df_with_meta_incompatible_index_raises(test_pd_df):
    # define a meta dataframe with a non-standard index
    index = ["source", "scenario"]
    meta = pd.DataFrame(
        [False, False, False], columns=["foo"], index=META_DF.index.rename(index)
    )

    # assert that using an incompatible index for the meta arg raises
    match = r"Incompatible `index=\['model', 'scenario'\]` with `meta.index=*."
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
    drop_message = "Formatted data is empty."
    message_idx = caplog.messages.index(drop_message)
    assert caplog.records[message_idx].levelno == logging.WARNING


def test_init_with_unnamed_column(test_pd_df):
    # add a column to the timeseries data with an unnamed column
    test_pd_df[None] = "foo"

    # check that initialising an instance with an unnamed column raises
    with pytest.raises(ValueError, match="Unnamed column in timeseries data: None"):
        IamDataFrame(test_pd_df)


@pytest.mark.parametrize("illegal", ["meta", ""])
def test_init_with_illegal_column(test_pd_df, illegal):
    # add a column to the timeseries data with an illegal column name
    test_pd_df[illegal] = "foo"

    # check that initialising an instance with an illegal column name raises
    msg = f"Illegal column for timeseries data: '{illegal}'"
    with pytest.raises(ValueError, match=msg):
        IamDataFrame(test_pd_df)

    # check that recommended fix works
    df = IamDataFrame(test_pd_df, valid=illegal)
    assert df.valid == ["foo"]


def test_set_meta_with_column_conflict(test_df_year):
    # check that setting a `meta` column with a name conflict raises
    msg = "Column 'model' already exists in `data`."
    with pytest.raises(ValueError, match=msg):
        test_df_year.set_meta(name="model", meta="foo")

    msg = "Name 'meta' is illegal for meta indicators."
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
    pdt.assert_index_equal(test_df_year.index, exp)


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


def test_filter_variable_and_measurand_raises(test_df):
    with pytest.raises(ValueError, match="Filter by `variable` and `measurand` not"):
        test_df.filter(variable="foo", measurand=("foo", "bar"))


def test_filter_level_and_depth_raises(test_df):
    with pytest.raises(ValueError, match="Filter by `level` and `depth` not"):
        test_df.filter(level=1, depth=2)


@pytest.mark.parametrize(
    "filter_args",
    (dict(variable="*rimary*C*"), dict(measurand=("*rimary*C*", "EJ/*"))),
)
def test_filter_variable_and_level(test_df, filter_args):
    obs = test_df.filter(**filter_args, level=0).variable
    assert obs == ["Primary Energy|Coal"]

    obs = test_df.filter(**filter_args, level="0+").variable
    assert obs == ["Primary Energy|Coal"]

    obs = test_df.filter(**filter_args, level=1).variable
    assert obs == []


@pytest.mark.parametrize(
    "filter_args",
    (dict(variable="*rimary*C*"), dict(measurand=("*rimary*C*", "EJ/*"))),
)
def test_filter_variable_and_depth(test_df, filter_args):
    obs = test_df.filter(**filter_args, depth=1).variable
    assert obs == ["Primary Energy|Coal"]

    obs = test_df.filter(**filter_args, depth="0+").variable
    assert obs == ["Primary Energy|Coal"]

    obs = test_df.filter(**filter_args, depth=0).variable
    assert obs == []


def test_filter_measurand_list(test_df):
    data = test_df.data
    data.loc[4, "variable"] = "foo"
    data.loc[5, "unit"] = "bar"
    df = IamDataFrame(data)

    obs = df.filter(measurand=(("foo", "EJ/yr"), ("Primary Energy", "bar")))

    assert set(obs.variable) == {"Primary Energy", "foo"}
    assert set(obs.unit) == {"EJ/yr", "bar"}
    assert obs.scenario == ["scen_b"]


@pytest.mark.parametrize(
    "filter_name",
    ("level", "depth"),
)
def test_variable_depth_0_keep_false(test_df, filter_name):
    obs = test_df.filter(**{filter_name: 0}, keep=False).variable
    assert obs == ["Primary Energy|Coal"]


@pytest.mark.parametrize(
    "filter_name",
    ("level", "depth"),
)
def test_variable_depth_raises(test_df, filter_name):
    pytest.raises(ValueError, test_df.filter, **{filter_name: "1/"})


@pytest.mark.parametrize(
    "filter_name",
    ("level", "depth"),
)
def test_variable_depth_with_list_raises(test_df, filter_name):
    pytest.raises(ValueError, test_df.filter, **{filter_name: ["1", "2"]})
    pytest.raises(ValueError, test_df.filter, **{filter_name: [1, 2]})


@pytest.mark.parametrize("unsort", [False, True])
def test_timeseries_long(test_df, unsort):
    """Assert that timeseries is shown as expected from (unsorted) long data"""
    exp = TEST_DF.set_index(IAMC_IDX)

    if unsort:
        # revert order of _data, then check that the index and columns are sorted anyway
        data = test_df.data
        if test_df.time_col == "time":
            time = test_df.time
            data.time = data.time.replace(
                dict([(year, time[i]) for i, year in enumerate([2005, 2010])])
            )
        test_df = IamDataFrame(data.iloc[[5, 4, 3, 2, 1, 0]])
        # check that `data` is not sorted internally
        unsorted_data = test_df.data
        assert list(unsorted_data.scenario.unique()) == ["scen_b", "scen_a"]
        if test_df.time_col == "year":
            time = unsorted_data.year.unique()
        else:
            time = unsorted_data.time.unique()
        assert time[0] > time[1]

    if test_df.time_col == "time":
        exp.columns = test_df.time
        exp.columns.name = None

    obs = test_df.timeseries()
    pdt.assert_frame_equal(obs, exp, check_like=True, check_column_type=False)


@pytest.mark.parametrize("unsort", [False, True])
def test_timeseries_wide(test_pd_df, unsort):
    """Assert that timeseries is shown as expected from (unsorted) wide data"""

    # for some reason, `unstack` behaves differently if columns or rows are not sorted
    exp = test_pd_df.set_index(IAMC_IDX)

    if unsort:
        obs = IamDataFrame(test_pd_df[IAMC_IDX + [2010, 2005]]).timeseries()
    else:
        obs = IamDataFrame(test_pd_df).timeseries()
    pdt.assert_frame_equal(obs, exp, check_column_type=False)


def test_timeseries_empty_raises(test_df_year):
    """Calling `timeseries()` on an empty IamDataFrame raises"""
    _df = test_df_year.filter(model="foo")
    with pytest.raises(ValueError, match="This IamDataFrame is empty."):
        _df.timeseries()


def test_timeseries_time_iamc_raises(test_df_time):
    """Calling `timeseries(iamc_index=True)` on a continuous-time IamDataFrame raises"""
    match = "Cannot use `iamc_index=True` with 'datetime' time-domain."
    with pytest.raises(ValueError, match=match):
        test_df_time.timeseries(iamc_index=True)


def test_timeseries_to_iamc_index(test_pd_df, test_df_year):
    """Reducing timeseries() of an IamDataFrame with extra-columns to IAMC-index"""
    test_pd_df["foo"] = "bar"
    extra_col_df = IamDataFrame(test_pd_df)
    assert extra_col_df.extra_cols == ["foo"]

    # assert that reducing to IAMC-columns (dropping extra-columns) with timeseries()
    obs = extra_col_df.timeseries(iamc_index=True)
    exp = test_df_year.timeseries()
    pdt.assert_frame_equal(obs, exp)


def test_timeseries_to_iamc_index_duplicated_raises(test_pd_df):
    """Assert that using `timeseries(iamc_index=True)` raises if there are duplicates"""
    test_pd_df = pd.concat([test_pd_df, test_pd_df])
    # adding an extra-col creates a unique index
    test_pd_df["foo"] = ["bar", "bar", "bar", "baz", "baz", "baz"]

    extra_col_df = IamDataFrame(test_pd_df)
    assert extra_col_df.extra_cols == ["foo"]

    # dropping the extra-column by setting `iamc_index=True` creates duplicated index
    match = "Dropping non-IAMC-index causes duplicated index"
    with pytest.raises(ValueError, match=match):
        extra_col_df.timeseries(iamc_index=True)


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
    pdt.assert_index_equal(obs, exp)


def test_meta_idx(test_df):
    # assert that the `drop_duplicates()` in `_meta_idx()` returns right length
    assert len(_meta_idx(test_df.data)) == 2


def test_filter_meta_by_bool(test_df):
    test_df.set_meta([True, False], name="meta_bool")
    obs = test_df.filter(meta_bool=True)
    assert obs.scenario == ["scen_a"]


def test_filter_meta_by_int(test_df):
    test_df.set_meta([1, 2], name="meta_int")
    obs = test_df.filter(meta_int=[1, 3])
    assert obs.scenario == ["scen_a"]


def test_pd_filter_by_meta(test_df):
    data = df_filter_by_meta_matching_idx.set_index(["model", "region"])

    test_df.set_meta([True, False], "boolean")
    test_df.set_meta(0, "integer")

    obs = filter_by_meta(data, test_df, join_meta=True, boolean=True, integer=None)
    obs = obs.reindex(columns=["scenario", "col", "boolean", "integer"])

    exp = data.iloc[0:2].copy()
    exp["boolean"] = True
    exp["integer"] = 0

    pdt.assert_frame_equal(obs, exp)


def test_pd_filter_by_meta_no_index(test_df):
    data = df_filter_by_meta_matching_idx

    test_df.set_meta([True, False], "boolean")
    test_df.set_meta(0, "int")

    obs = filter_by_meta(data, test_df, join_meta=True, boolean=True, int=None)
    obs = obs.reindex(columns=META_IDX + ["region", "col", "boolean", "int"])

    exp = data.iloc[0:2].copy()
    exp["boolean"] = True
    exp["int"] = 0

    pdt.assert_frame_equal(obs, exp)


def test_pd_filter_by_meta_nonmatching_index(test_df):
    data = df_filter_by_meta_nonmatching_idx
    test_df.set_meta(["a", "b"], "string")

    obs = filter_by_meta(data, test_df, join_meta=True, string="b")
    obs = obs.reindex(columns=["scenario", 2010, 2020, "string"])

    exp = data.iloc[2:3].copy()
    exp["string"] = "b"

    pdt.assert_frame_equal(obs, exp)


def test_pd_join_by_meta_nonmatching_index(test_df):
    data = df_filter_by_meta_nonmatching_idx
    test_df.set_meta(["a", "b"], "string")

    obs = filter_by_meta(data, test_df, join_meta=True, string=None)
    obs = obs.reindex(columns=["scenario", 2010, 2020, "string"])

    exp = data.copy()
    exp["string"] = [np.nan, np.nan, "b"]

    pdt.assert_frame_equal(obs.sort_index(level=1), exp)


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
    pdt.assert_frame_equal(obs, exp)


def test_normalize_not_time(test_df):
    pytest.raises(ValueError, test_df.normalize, variable="foo")
    pytest.raises(ValueError, test_df.normalize, year=2015, variable="foo")


@pytest.mark.parametrize("padding", [0, 2])
def test_offset(test_df, padding):
    exp = test_df.data.copy().reset_index(drop=True)
    exp.loc[1::2, "value"] -= exp["value"][::2].values - padding
    exp.loc[::2, "value"] -= exp["value"][::2].values - padding
    # only call with kwarg if padding != 0 (the default)
    kwargs = {"padding": padding} if padding else {}
    if "year" in test_df.data:
        obs = test_df.offset(year=2005, **kwargs).data.reset_index(drop=True)
    else:
        obs = test_df.offset(
            time=datetime.datetime(2005, 6, 17), **kwargs
        ).data.reset_index(drop=True)
    pdt.assert_frame_equal(obs, exp)


def test_offset_not_time(test_df):
    pytest.raises(ValueError, test_df.offset, variable="foo")
    pytest.raises(ValueError, test_df.offset, year=2015, variable="foo")
