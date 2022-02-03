import pytest
import os
import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from contextlib import contextmanager

import pyam
from pyam import plotting, run_control, reset_rc_defaults
from .conftest import IMAGE_BASELINE_DIR


logger = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"
TOLERANCE = 6 if IS_WINDOWS else 2

MPL_KWARGS = {
    "style": "ggplot",
    "baseline_dir": str(IMAGE_BASELINE_DIR),
    "tolerance": TOLERANCE,
    "savefig_kwargs": {"dpi": 300, "bbox_inches": "tight"},
}


RC_TEST_DICT = {"color": {"scenario": {"test_scenario": "black"}}}


@contextmanager
def update_run_control(update):
    run_control().update(update)
    yield
    reset_rc_defaults()


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot(plot_df):
    _plot_df = copy.deepcopy(plot_df)
    _plot_df.set_meta(meta=[np.nan] * 4, name="test")
    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_df.plot(ax=ax, legend=True)
    return fig


def test_line_plot_cmap(plot_df):
    # need to provide cmap and color both
    _plot_df = copy.deepcopy(plot_df)
    _plot_df.set_meta(meta=[np.nan] * 4, name="test")
    pytest.raises(ValueError, _plot_df.plot, cmap="magma")


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_cmap_color_arg(plot_df):
    _plot_df = copy.deepcopy(plot_df)
    _plot_df.set_meta(meta=[np.nan] * 4, name="test")
    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_df.plot(ax=ax, legend=True, cmap="magma", color="variable")
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_dict_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot(ax=ax, legend=dict(loc="outside right"))
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_bottom_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot(ax=ax, legend=dict(loc="outside bottom"))
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_no_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot(ax=ax, legend=False)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_label(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot(ax=ax, label="foo")
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_label_color(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot(ax=ax, label="foo", color="red")
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_color(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot(ax=ax, color="model", legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_PYAM_COLORS(plot_df):
    # add a family of lines for each color in plotting.PYAM_COLORS separated by
    # a small offset
    update = {"color": {"model": {}}}
    _df = plot_df.filter(
        model="test_model",
        variable="Primary Energy",
        scenario="test_scenario1",
    ).data.copy()
    dfs = []
    for i, color in enumerate(plotting.PYAM_COLORS):
        df = _df.copy()
        model = color
        df["model"] = model
        df["value"] += i
        update["color"]["model"][model] = color
        dfs.append(df)
    df = pyam.IamDataFrame(pd.concat(dfs))
    fig, ax = plt.subplots(figsize=(8, 8))
    with update_run_control(update):
        df.plot(ax=ax, color="model", legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_color_fill_between(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot(ax=ax, color="model", fill_between=True, legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_color_fill_between_interpolate(plot_df):
    # designed to create the sawtooth behavior at a midpoint with missing data
    df = pyam.IamDataFrame(plot_df.data.copy())
    fig, ax = plt.subplots(figsize=(8, 8))
    newdata = [
        "test_model1",
        "test_scenario1",
        "World",
        "Primary Energy|Coal",
        "EJ/y",
        2010,
        3.50,
    ]
    df.data.loc[len(df.data) - 1] = newdata
    newdata = [
        "test_model1",
        "test_scenario1",
        "World",
        "Primary Energy|Coal",
        "EJ/y",
        2012,
        3.50,
    ]
    df.data.loc[len(df.data)] = newdata
    newdata = [
        "test_model1",
        "test_scenario1",
        "World",
        "Primary Energy|Coal",
        "EJ/y",
        2015,
        3.50,
    ]
    df.data.loc[len(df.data) + 1] = newdata
    df.plot(ax=ax, color="model", fill_between=True, legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_color_final_ranges(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot(ax=ax, color="model", final_ranges=True, legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_marker_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot(ax=ax, marker="model", legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_rm_legend_label(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot(
        ax=ax,
        marker="model",
        linestyle="scenario",
        legend=True,
        rm_legend_label="marker",
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_linestyle_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot(ax=ax, linestyle="model", legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_single_color(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot(ax=ax, color="b", linestyle="model", legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_filter_title(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.filter(variable="Primary Energy|Coal").plot(
        ax=ax, color="model", marker="scenario", legend=True
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_update_rc(plot_df):
    with update_run_control({"color": {"model": {"test_model1": "cyan"}}}):
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_df.plot(ax=ax, color="model", legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_1_var(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    (
        plot_df.filter(model="test_model", scenario="test_scenario").plot(
            x="Primary Energy", y="year", ax=ax, legend=False
        )
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_2_vars(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    (
        plot_df.filter(model="test_model", scenario="test_scenario").plot(
            x="Primary Energy|Coal", y="Primary Energy", ax=ax, legend=False
        )
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_order_by_dict(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    order = dict(
        model=["test_model1", "test_model"],
        scenario=["test_scenario1", "test_scenario"],
    )
    plot_df.plot(order=order, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_order_by_rc(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    order = dict(model=["test_model1"], scenario=["test_scenario1"])
    with update_run_control(order):  # first item from rc, then alphabetical
        plot_df.plot(order=order, ax=ax)
    return fig


def test_barplot_raises(plot_df):
    pytest.raises(ValueError, plot_df.plot.bar)


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_barplot(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.filter(variable="Primary Energy", model="test_model").plot.bar(
        ax=ax, bars="scenario"
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_barplot_h(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.filter(variable="Primary Energy", model="test_model").plot.bar(
        ax=ax, bars="scenario", orient="h"
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_barplot_stacked(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.filter(variable="Primary Energy", model="test_model").plot.bar(
        ax=ax, bars="scenario", stacked=True
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_barplot_stacked_order_by_list(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.filter(variable="Primary Energy", model="test_model").plot.bar(
        ax=ax,
        bars="scenario",
        order=[2015, 2010],
        bars_order=["test_scenario1", "test_scenario"],
        stacked=True,
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_barplot_stacked_order_by_rc(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    rc_order = dict(order={"year": [2010], "scenario": ["test_scenario1"]})
    with update_run_control(rc_order):  # first item from rc, then alphabetical
        (
            plot_df.filter(variable="Primary Energy", model="test_model").plot.bar(
                ax=ax, bars="scenario", stacked=True
            )
        )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_barplot_stacked_net_line(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    # explicitly add negative contributions for net lines
    df = pyam.IamDataFrame(plot_df.data.copy())
    vals = [(2005, 0.35), (2010, -1.0), (2015, -4.0)]
    for i, (y, v) in enumerate(vals):
        newdata = [
            "test_model1",
            "test_scenario1",
            "World",
            "Primary Energy|foo",
            "EJ/y",
            y,
            v,
        ]
        df.data.loc[len(df) + i] = newdata
    df.filter(
        variable="Primary Energy|*",
        model="test_model1",
        scenario="test_scenario1",
        region="World",
    ).plot.bar(ax=ax, bars="variable", stacked=True)
    plotting.add_net_values_to_bar_plot(ax, color="r")
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_barplot_title(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.filter(variable="Primary Energy", model="test_model").plot.bar(
        ax=ax, bars="scenario", title="foo"
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_barplot_rc(plot_df):
    with update_run_control({"color": {"scenario": {"test_scenario": "black"}}}):
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_df.filter(variable="Primary Energy", model="test_model").plot.bar(
            ax=ax, bars="scenario"
        )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_boxplot(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot.box(ax=ax)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_boxplot_hue(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot.box(ax=ax, by="model")
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_pie_plot_labels(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.filter(variable="Primary Energy", model="test_model", year=2010).plot.pie(
        ax=ax, category="scenario"
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_pie_plot_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.filter(variable="Primary Energy", model="test_model", year=2010).plot.pie(
        ax=ax, category="scenario", labels=None, legend=True
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_pie_plot_colors(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.filter(variable="Primary Energy", model="test_model", year=2010).plot.pie(
        ax=ax, category="scenario", colors=["green", "yellow"], title="foo"
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_pie_plot_other(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    with update_run_control(RC_TEST_DICT):
        (
            plot_df.filter(
                variable="Primary Energy", model="test_model", year=2010
            ).plot.pie(ax=ax, category="scenario", cmap="viridis", title="foo")
        )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_stackplot(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.filter(variable="Primary Energy", model="test_model").plot.stack(
        ax=ax, stack="scenario"
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_stackplot_order_by_list(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    (
        plot_df.filter(variable="Primary Energy", model="test_model").plot.stack(
            ax=ax, stack="scenario", order=["test_scenario1", "test_scenario"]
        )
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_stackplot_order_by_rc(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    scen_list = ["test_scenario1"]  # first list from rc, then alphabetical
    with update_run_control({"order": {"scenario": scen_list}}):
        plot_df.filter(variable="Primary Energy", model="test_model").plot.stack(
            ax=ax, stack="scenario"
        )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_stackplot_other(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    with update_run_control({"color": {"scenario": {"test_scenario": "black"}}}):
        plot_df.filter(variable="Primary Energy", model="test_model").plot.stack(
            ax=ax, stack="scenario", cmap="viridis", title="foo"
        )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_stackplot_negative():
    # test that data with both positive & negative values are shown correctly
    TEST_STACKPLOT_NEGATIVE = pd.DataFrame(
        [
            ["var1", 1, -1, 0],
            ["var2", 1, 1, -2],
            ["var3", -1, 1, 1],
        ],
        columns=["variable", 2005, 2010, 2015],
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    df = pyam.IamDataFrame(
        TEST_STACKPLOT_NEGATIVE,
        model="model_a",
        scenario="scen_a",
        region="World",
        unit="foo",
    )
    df.plot.stack(ax=ax, total=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_scatter(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot.scatter(ax=ax, x="Primary Energy", y="Primary Energy|Coal")
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_scatter_variables_with_meta_color(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.categorize(
        "foo", "a", criteria={"Primary Energy": {"up": 5, "year": 2010}}, color="blue"
    )
    plot_df.categorize(
        "foo", "b", criteria={"Primary Energy": {"lo": 5, "year": 2010}}, color="red"
    )
    plot_df.plot.scatter(
        ax=ax, x="Primary Energy", y="Primary Energy|Coal", color="foo"
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_scatter_with_lines(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.plot.scatter(
        ax=ax, x="Primary Energy", y="Primary Energy|Coal", with_lines=True
    )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_scatter_meta(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.set_meta(
        meta=plot_df.filter(variable="Primary Energy").timeseries()[2010], name="Total"
    )
    plot_df.set_meta(
        meta=plot_df.filter(variable="Primary Energy|Coal").timeseries()[2010],
        name="Coal",
    )
    plot_df.plot.scatter(ax=ax, x="Total", y="Coal")
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_add_panel_label(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plotting.set_panel_label("test", ax=ax, x=0.5, y=0.5)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_stackplot_negative_emissions(plot_stackplot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_stackplot_df.plot.stack(ax=ax)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_stackplot_negative_emissions_with_total(plot_stackplot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_stackplot_df.plot.stack(ax=ax, total=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_stackplot_negative_emissions_kwargs_def_total(plot_stackplot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_stackplot_df.plot.stack(ax=ax, alpha=0.5, total=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_stackplot_negative_emissions_kwargs_custom_total(plot_stackplot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    total = {"color": "grey", "ls": "--", "lw": 2.0}
    plot_stackplot_df.plot.stack(ax=ax, alpha=0.5, total=total)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_stackplot_missing_zero_issue_266(plot_stackplot_df):
    df = pyam.IamDataFrame(
        pd.DataFrame(
            [
                ["a", 1, 2, 3, 4],
                ["b", 0, 1, 2, 3],
                ["c", -1, 1, -1, -1],
                ["d", 1, 1, 1, -1],
            ],
            columns=["variable", 2010, 2020, 2030, 2040],
        ),
        model="model_a",
        scenario="scen_a",
        region="World",
        unit="some_unit",
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    df.plot.stack(ax=ax)
    return fig
