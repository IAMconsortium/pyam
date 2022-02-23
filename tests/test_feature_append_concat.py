import pytest
import numpy as np
from numpy import testing as npt
import pandas as pd
import pandas.testing as pdt
from datetime import datetime

from pyam import IamDataFrame, IAMC_IDX, META_IDX, assert_iamframe_equal, concat

from .conftest import TEST_DF, META_COLS, META_DF


# assert that merging of meta works as expected
EXP_META = pd.DataFrame(
    [
        ["model_a", "scen_a", False, 1, "foo", 0, "a", np.nan],
        ["model_a", "scen_b", False, 2, np.nan, 1, "b", np.nan],
        ["model_a", "scen_c", False, 2, np.nan, 2, np.nan, "x"],
    ],
    columns=META_IDX + ["exclude", "number", "string", "col1", "col2", "col3"],
).set_index(META_IDX)


def test_concat_single_item(test_df):
    """Check that calling concat on a single-item list returns identical object"""
    obs = concat([test_df])
    assert_iamframe_equal(obs, test_df)


def test_concat_fails_empty():
    """Check that calling concat with empty or none raises"""
    match = "No objects to concatenate"
    with pytest.raises(ValueError, match=match):
        concat([])


@pytest.mark.parametrize(
    "arg, msg", ((None, "NoneType"), (1, "int"), ("foo", "str"), (TEST_DF, "DataFrame"))
)
def test_concat_fails_iterable(arg, msg):
    """Check that calling concat with a non-iterable raises"""
    match = f"'{msg}' object is not iterable"
    with pytest.raises(TypeError, match=match):
        concat(arg)


def test_concat_incompatible_cols(test_pd_df):
    """Check that calling concat on a single-item list returns identical object"""
    df1 = IamDataFrame(test_pd_df)
    test_pd_df["extra_col"] = "foo"
    df2 = IamDataFrame(test_pd_df)

    match = "Items have incompatible timeseries data dimensions"
    with pytest.raises(ValueError, match=match):
        concat([df1, df2])


@pytest.mark.parametrize("inplace", (True, False))
def test_append_duplicates_raises(test_df_year, inplace):
    # Merging objects with overlapping values (merge conflict) raises an error

    other = test_df_year.copy()
    with pytest.raises(ValueError, match="Timeseries data has overlapping values:"):
        test_df_year.append(other=other, inplace=inplace)


@pytest.mark.parametrize("inplace", (True, False))
def test_append_incompatible_col_raises(test_df_year, test_pd_df, inplace):
    # Merging objects with different data index dimensions raises an error
    test_pd_df["foo"] = "baz"

    with pytest.raises(ValueError, match="Incompatible timeseries data index"):
        test_df_year.append(other=test_pd_df, inplace=inplace)


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("iterable", (False, True))
def test_concat(test_df, reverse, iterable):
    other = test_df.filter(scenario="scen_b").rename({"scenario": {"scen_b": "scen_c"}})

    test_df.set_meta([0, 1], name="col1")
    test_df.set_meta(["a", "b"], name="col2")

    other.set_meta(2, name="col1")
    other.set_meta("x", name="col3")

    dfs = [test_df, other]
    if reverse:
        dfs = list(reversed(dfs))
    if iterable:
        dfs = iter(dfs)

    result = concat(dfs)

    # check that the original object is not updated
    assert test_df.scenario == ["scen_a", "scen_b"]
    assert other.scenario == ["scen_c"]

    # assert that merging of meta works as expected (reorder columns)
    pdt.assert_frame_equal(result.meta[EXP_META.columns], EXP_META)

    # assert that appending data works as expected
    ts = result.timeseries()
    npt.assert_array_equal(ts.iloc[2].values, ts.iloc[3].values)


@pytest.mark.parametrize("reverse", (False, True))
def test_concat_with_pd_dataframe(test_df, reverse):
    other = test_df.filter(scenario="scen_b").rename({"scenario": {"scen_b": "scen_c"}})

    # merge with only the timeseries `data` DataFrame of `other`
    if reverse:
        result = concat([other.data, test_df])
    else:
        result = concat([test_df, other.data])

    # check that the original object is not updated
    assert test_df.scenario == ["scen_a", "scen_b"]

    # assert that merging meta from `other` is ignored
    exp_meta = META_DF.copy()
    exp_meta.loc[("model_a", "scen_c"), "number"] = np.nan
    exp_meta["exclude"] = False
    pdt.assert_frame_equal(result.meta, exp_meta[["exclude"] + META_COLS])

    # assert that appending data works as expected
    ts = result.timeseries()
    npt.assert_array_equal(ts.iloc[2].values, ts.iloc[3].values)


def test_append(test_df):
    other = test_df.filter(scenario="scen_b").rename({"scenario": {"scen_b": "scen_c"}})

    test_df.set_meta([0, 1], name="col1")
    test_df.set_meta(["a", "b"], name="col2")

    other.set_meta(2, name="col1")
    other.set_meta("x", name="col3")

    df = test_df.append(other)

    # check that the original object is not updated
    assert test_df.scenario == ["scen_a", "scen_b"]

    # assert that merging of meta works as expected
    pdt.assert_frame_equal(df.meta, EXP_META)

    # assert that appending data works as expected
    ts = df.timeseries()
    npt.assert_array_equal(ts.iloc[2].values, ts.iloc[3].values)


@pytest.mark.parametrize("time", (datetime(2010, 7, 21), "2010-07-21 00:00:00"))
@pytest.mark.parametrize("reverse", (False, True))
def test_concat_time_domain(test_pd_df, test_df_mixed, time, reverse):

    df_year = IamDataFrame(test_pd_df[IAMC_IDX + [2005]], meta=test_df_mixed.meta)
    df_time = IamDataFrame(
        test_pd_df[IAMC_IDX + [2010]].rename({2010: time}, axis="columns")
    )

    # concat `df_time` to `df_year`
    if reverse:
        obs = concat([df_time, df_year])
    else:
        obs = concat([df_year, df_time])

    # assert that original objects were not modified
    assert df_year.year == [2005]
    assert df_time.time == pd.Index([datetime(2010, 7, 21)])

    assert_iamframe_equal(obs, test_df_mixed)


@pytest.mark.parametrize("other", ("time", "year"))
@pytest.mark.parametrize("time", (datetime(2010, 7, 21), "2010-07-21 00:00:00"))
@pytest.mark.parametrize("inplace", (True, False))
def test_append_time_domain(test_pd_df, test_df_mixed, other, time, inplace):

    df_year = IamDataFrame(test_pd_df[IAMC_IDX + [2005]], meta=test_df_mixed.meta)
    df_time = IamDataFrame(
        test_pd_df[IAMC_IDX + [2010]].rename({2010: time}, axis="columns")
    )

    # append `df_time` to `df_year`
    if other == "time":
        if inplace:
            obs = df_year.copy()
            obs.append(df_time, inplace=True)
        else:
            obs = df_year.append(df_time)
            # assert that original object was not modified
            assert df_year.year == [2005]

    # append `df_year` to `df_time`
    else:
        if inplace:
            obs = df_time.copy()
            obs.append(df_year, inplace=True)
        else:
            obs = df_time.append(df_year)
            # assert that original object was not modified
            assert df_time.time == pd.Index([datetime(2010, 7, 21)])

    assert_iamframe_equal(obs, test_df_mixed)


def test_append_reconstructed_time(test_df):
    # check appending dfs with equal time cols created by different methods
    other = test_df.filter(scenario="scen_b").rename({"scenario": {"scen_b": "scen_c"}})
    other.time_col = other.time_col[0:1] + other.time_col[1:]
    test_df.append(other, inplace=True)
    assert "scen_c" in test_df.scenario


def test_append_same_scenario(test_df):
    other = test_df.filter(scenario="scen_b").rename(
        {"variable": {"Primary Energy": "Primary Energy clone"}}
    )

    test_df.set_meta([0, 1], name="col1")

    other.set_meta(2, name="col1")
    other.set_meta("b", name="col2")

    # check that non-matching meta raise an error
    pytest.raises(ValueError, test_df.append, other=other)

    # check that ignoring meta conflict works as expected
    df = test_df.append(other, ignore_meta_conflict=True)

    # check that the new meta.index is updated, but not the original one
    cols = ["exclude"] + META_COLS + ["col1"]
    npt.assert_array_equal(test_df.meta.columns, cols)

    # assert that merging of meta works as expected
    exp = test_df.meta.copy()
    exp["col2"] = [np.nan, "b"]
    pd.testing.assert_frame_equal(df.meta, exp)

    # assert that appending data works as expected
    ts = df.timeseries()
    npt.assert_array_equal(ts.iloc[2], ts.iloc[3])


@pytest.mark.parametrize("shuffle_cols", [True, False])
def test_append_extra_col(test_df, shuffle_cols):
    base_data = test_df.data.copy()

    base_data["col_1"] = "hi"
    base_data["col_2"] = "bye"
    base_df = IamDataFrame(base_data)

    other_data = base_data[base_data["variable"] == "Primary Energy"].copy()
    other_data["variable"] = "Primary Energy|Gas"
    other_df = IamDataFrame(other_data)

    if shuffle_cols:
        c1_idx = other_df.dimensions.index("col_1")
        c2_idx = other_df.dimensions.index("col_2")
        other_df.dimensions[c1_idx] = "col_2"
        other_df.dimensions[c2_idx] = "col_1"

    res = base_df.append(other_df)

    def check_meta_is(iamdf, meta_col, val):
        for checker in [iamdf.timeseries().reset_index(), iamdf.data]:
            meta_vals = checker[meta_col].unique()
            assert len(meta_vals) == 1, meta_vals
            assert meta_vals[0] == val, meta_vals

    # ensure meta merged correctly
    check_meta_is(res, "col_1", "hi")
    check_meta_is(res, "col_2", "bye")
