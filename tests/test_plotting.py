import pytest

import matplotlib.pyplot as plt

from pyam_analysis import plotting

from testing_utils import plot_idf, IMAGE_BASELINE_DIR


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_plot(plot_idf):
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_idf.line_plot(ax=ax)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_color(plot_idf):
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_idf.line_plot(ax=ax, color='model')
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_color_legend(plot_idf):
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_idf.line_plot(ax=ax, color='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_marker_legend(plot_idf):
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_idf.line_plot(ax=ax, marker='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_linestyle_legend(plot_idf):
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_idf.line_plot(ax=ax, linestyle='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_single_color(plot_idf):
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_idf.line_plot(ax=ax, color='b', linestyle='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_filter_title(plot_idf):
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_idf.filter({'variable': 'Primary Energy|Coal'}).line_plot(
        ax=ax, color='model', marker='scenario', legend=True)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_update_rc(plot_idf):
    update = {'color': {'model': {'test_model1': 'cyan'}}}
    plotting.run_control().update(update)
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_idf.line_plot(ax=ax, color='model', legend=True)
    return fig
