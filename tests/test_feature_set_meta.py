import pytest
import numpy as np
import pandas as pd


EXP_IDX = pd.MultiIndex(
    levels=[["model_a"], ["scen_a", "scen_b"]],
    codes=[[0, 0], [0, 1]],
    names=["model", "scenario"],
)


def test_set_meta_no_name(test_df):
    idx = pd.MultiIndex(
        levels=[["a_scenario"], ["a_model"], ["some_region"]],
        codes=[[0], [0], [0]],
        names=["scenario", "model", "region"],
    )
    s = pd.Series(data=[0.3], index=idx)
    pytest.raises(ValueError, test_df.set_meta, s)


def test_set_meta_as_named_series(test_df):
    idx = pd.MultiIndex(
        levels=[["scen_a"], ["model_a"], ["some_region"]],
        codes=[[0], [0], [0]],
        names=["scenario", "model", "region"],
    )

    s = pd.Series(data=[0.3], index=idx, name="meta_values")
    test_df.set_meta(s)

    exp = pd.Series(data=[0.3, np.nan], index=EXP_IDX, name="meta_values")
    pd.testing.assert_series_equal(test_df["meta_values"], exp)


def test_set_meta_as_unnamed_series(test_df):
    idx = pd.MultiIndex(
        levels=[["scen_a"], ["model_a"], ["some_region"]],
        codes=[[0], [0], [0]],
        names=["scenario", "model", "region"],
    )

    s = pd.Series(data=[0.3], index=idx)
    test_df.set_meta(s, name="meta_values")

    exp = pd.Series(data=[0.3, np.nan], index=EXP_IDX, name="meta_values")
    pd.testing.assert_series_equal(test_df["meta_values"], exp)


def test_set_meta_non_unique_index_fail(test_df):
    idx = pd.MultiIndex(
        levels=[["model_a"], ["scen_a"], ["reg_a", "reg_b"]],
        codes=[[0, 0], [0, 0], [0, 1]],
        names=["model", "scenario", "region"],
    )
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, test_df.set_meta, s)


def test_set_meta_non_existing_index_fail(test_df):
    idx = pd.MultiIndex(
        levels=[["model_a", "fail_model"], ["scen_a", "fail_scenario"]],
        codes=[[0, 1], [0, 1]],
        names=["model", "scenario"],
    )
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, test_df.set_meta, s)


def test_set_meta_by_df(test_df):
    df = pd.DataFrame(
        [
            ["model_a", "scen_a", "some_region", 1],
        ],
        columns=["model", "scenario", "region", "col"],
    )

    test_df.set_meta(meta=0.3, name="meta_values", index=df)

    exp = pd.Series(data=[0.3, np.nan], index=EXP_IDX, name="meta_values")
    pd.testing.assert_series_equal(test_df["meta_values"], exp)


def test_set_meta_as_series(test_df):
    s = pd.Series([0.3, 0.4])
    test_df.set_meta(s, "meta_series")

    exp = pd.Series(data=[0.3, 0.4], index=EXP_IDX, name="meta_series")
    pd.testing.assert_series_equal(test_df["meta_series"], exp)


def test_set_meta_as_int(test_df):
    test_df.set_meta(3.2, "meta_int")

    exp = pd.Series(data=[3.2, 3.2], index=EXP_IDX, name="meta_int")

    obs = test_df["meta_int"]
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_str(test_df):
    test_df.set_meta("testing", name="meta_str")

    exp = pd.Series(data=["testing"] * 2, index=EXP_IDX, name="meta_str")
    pd.testing.assert_series_equal(test_df["meta_str"], exp)


def test_set_meta_as_str_list(test_df):
    test_df.set_meta(["testing", "testing2"], name="category")
    obs = test_df.filter(category="testing")
    assert obs["scenario"].unique() == "scen_a"


def test_set_meta_as_str_by_index(test_df):
    idx = pd.MultiIndex(
        levels=[["model_a"], ["scen_a"]], codes=[[0], [0]], names=["model", "scenario"]
    )

    test_df.set_meta("foo", "meta_str", idx)

    exp = pd.Series(data=["foo", None], index=EXP_IDX, name="meta_str")
    pd.testing.assert_series_equal(test_df["meta_str"], exp)


def test_set_meta_from_data(test_df):
    test_df.set_meta_from_data("pe_2005", variable="Primary Energy", year=2005)
    exp = pd.Series(data=[1.0, 2.0], index=EXP_IDX, name="pe_2005")
    pd.testing.assert_series_equal(test_df["pe_2005"], exp)


def test_set_meta_from_data_max(test_df):
    test_df.set_meta_from_data("pe_max_yr", variable="Primary Energy", method=np.max)
    exp = pd.Series(data=[6.0, 7.0], index=EXP_IDX, name="pe_max_yr")
    pd.testing.assert_series_equal(test_df["pe_max_yr"], exp)


def test_set_meta_from_data_mean(test_df):
    test_df.set_meta_from_data("pe_mean", variable="Primary Energy", method=np.mean)
    exp = pd.Series(data=[3.5, 4.5], index=EXP_IDX, name="pe_mean")
    pd.testing.assert_series_equal(test_df["pe_mean"], exp)


def test_set_meta_from_data_method_other_column(test_df):
    if "year" in test_df.data.columns:
        col, value = "year", 2010
    else:
        col, value = "time", max(test_df.data.time)
    test_df.set_meta_from_data(
        "pe_max_yr", variable="Primary Energy", method=np.max, column=col
    )
    exp = pd.Series(data=[value] * 2, index=EXP_IDX, name="pe_max_yr")
    pd.testing.assert_series_equal(test_df["pe_max_yr"], exp)


def test_set_meta_from_data_nonunique(test_df):
    # the filtered `data` dataframe is not unique with regard to META_IDX
    pytest.raises(
        ValueError, test_df.set_meta_from_data, "fail", variable="Primary Energy"
    )
