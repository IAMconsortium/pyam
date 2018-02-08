import pytest

import matplotlib.pyplot as plt

from testing_utils import plot_idf


@pytest.mark.mpl_image_compare
def test_line_plot(plot_idf):
    fig, ax = plt.subplots()
    plot_idf.line_plot(ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_line_color(plot_idf):
    fig, ax = plt.subplots()
    plot_idf.line_plot(ax=ax, color='model')
    return fig


@pytest.mark.mpl_image_compare
def test_line_color_legend(plot_idf):
    fig, ax = plt.subplots()
    plot_idf.line_plot(ax=ax, color='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare
def test_line_marker_legend(plot_idf):
    fig, ax = plt.subplots()
    plot_idf.line_plot(ax=ax, marker='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare
def test_line_linestyle_legend(plot_idf):
    fig, ax = plt.subplots()
    plot_idf.line_plot(ax=ax, linestyle='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare
def test_line_single_color(plot_idf):
    fig, ax = plt.subplots()
    plot_idf.line_plot(ax=ax, color='b', linestyle='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare
def test_line_filter_title(plot_idf):
    fig, ax = plt.subplots()
    plot_idf.filter({'variable': 'Primary Energy|Coal'}).line_plot(
        ax=ax, color='model', marker='scenario', legend=True)
    return fig
