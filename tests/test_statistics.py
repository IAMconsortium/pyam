# -*- coding: utf-8 -*-

import pytest
import pandas as pd
from pyam import Statistics


def stats_add(stats, plot_df):
    # test describing as pd.DataFrame
    primary = plot_df.filter(variable="Primary Energy", year=2005).timeseries()
    stats.add(data=primary, header="primary")
    # test describing as unamed pd.Series with `subheader` arg
    coal = plot_df.filter(variable="Primary Energy|Coal").timeseries()[2010]
    coal.name = None
    stats.add(data=coal, header="coal", subheader=2005)

    # repeat `add()` to ensure that previous data is overwritten
    coal = plot_df.filter(variable="Primary Energy|Coal").timeseries()[2005]
    coal.name = None
    stats.add(data=coal, header="coal", subheader=2005)
    return stats


def stats_add_with_rows(stats, plot_df):
    # test describing as pd.DataFrame
    primary = plot_df.filter(variable="Primary Energy", year=2005).timeseries()
    stats.add(data=primary, header="primary", row="first")
    # test describing as unamed pd.Series with `subheader` arg
    coal = plot_df.filter(variable="Primary Energy|Coal").timeseries()[2010]
    coal.name = None
    stats.add(data=coal, header="coal", subheader=2005, row="another")
    # repeat `add()` to ensure that previous data is overwritten
    coal = plot_df.filter(variable="Primary Energy|Coal").timeseries()[2005]
    coal.name = None
    stats.add(data=coal, header="coal", subheader=2005, row="another")
    return stats


def test_statistics(plot_df):
    plot_df.set_meta(meta=["a", "b", "b", "a"], name="category")
    stats = Statistics(
        df=plot_df,
        groupby={"category": ["b", "a"]},
        filters=[(("scen", "test"), {"scenario": "test_scenario"})],
    )
    obs = stats_add(stats, plot_df).summarize(custom_format="{:.0f}")

    idx = pd.MultiIndex(
        levels=[["category", "scen"], ["b", "a", "test"]],
        codes=[[0, 0, 1], [0, 1, 2]],
        names=["", ""],
    )
    cols = pd.MultiIndex(
        levels=[["count", "primary", "coal"], ["", 2005]],
        codes=[[0, 1, 2], [0, 1, 1]],
        names=[None, "mean (max, min)"],
    )
    exp = pd.DataFrame(
        data=[
            ["2", "1 (2, 1)", "0 (0, 0)"],
            ["2", "1 (1, 1)", "0 (0, 0)"],
            ["2", "1 (1, 1)", "0 (0, 0)"],
        ],
        index=idx,
        columns=cols,
    )
    pd.testing.assert_frame_equal(obs, exp)


def test_statistics_mismatching_groupby_index(plot_df):
    pytest.raises(
        ValueError,
        Statistics,
        df=plot_df,
        groupby={"category": ["b", "a"]},
        filters=[(("test"), {"scenario": "test_scenario"})],
    )


def test_statistics_mismatching_filters_depth(plot_df):
    pytest.raises(
        ValueError,
        Statistics,
        df=plot_df,
        groupby={"category": ["b", "a"]},
        filters=[
            (("test"), {"scenario": "test_scenario"}),
            (("test", "test"), {"scenario": "test_scenario"}),
        ],
    )


def test_statistics_by_filter(plot_df):
    stats = Statistics(df=plot_df, filters=[("test", {"scenario": "test_scenario"})])
    obs = stats_add(stats, plot_df).summarize(interquartile=True)

    idx = pd.MultiIndex(levels=[["test"]], codes=[[0]])
    cols = pd.MultiIndex(
        levels=[["count", "primary", "coal"], ["", 2005]],
        codes=[[0, 1, 2], [0, 1, 1]],
        names=[None, "mean (interquartile range)"],
    )
    exp = pd.DataFrame(
        data=["2", "0.85 (0.93, 0.77)", "0.42 (0.46, 0.39)"], index=cols, columns=idx
    ).T
    pd.testing.assert_frame_equal(obs, exp)


def test_statistics_with_row_required(plot_df):
    stats = Statistics(df=plot_df)
    primary = plot_df.filter(variable="Primary Energy", year=2005).timeseries()
    # test that kwarg `row` is required for `add`
    pytest.raises(ValueError, stats.add, data=primary, header="primary")


def test_statistics_with_rows(plot_df):
    stats = Statistics(
        df=plot_df, filters=[("test", {"scenario": "test_scenario"})], rows=True
    )
    obs = stats_add_with_rows(stats, plot_df).summarize(center="50%")

    idx = pd.MultiIndex(levels=[["test"], ["first", "another"]], codes=[[0, 0], [0, 1]])
    cols = pd.MultiIndex(
        levels=[["count", "primary", "coal"], ["", 2005]],
        codes=[[0, 1, 2], [0, 1, 1]],
        names=[None, "median (max, min)"],
    )
    exp = pd.DataFrame(
        data=[["2", "0.85 (1.00, 0.70)", ""], ["2", "", "0.42 (0.50, 0.35)"]],
        index=idx,
        columns=cols,
    )
    pd.testing.assert_frame_equal(obs, exp)


def test_statistics_with_percentiles(plot_df):
    stats = Statistics(
        df=plot_df,
        filters=[("test", {"scenario": "test_scenario"})],
        percentiles=[0.05, 0.95],
    )
    stats = stats_add(stats, plot_df)
    obs = set(stats.stats.columns.get_level_values(2))
    assert obs == set(["count", "mean", "std", "min", "5%", "50%", "95%", "max"])
