import pytest

import matplotlib.pyplot as plt

from testing_utils import test_ia


@pytest.mark.mpl_image_compare
def test_line_plot(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_line_color_1(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax, color='model')
    return fig


@pytest.mark.mpl_image_compare
def test_line_plot_color_2(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax, color='variable')
    return fig


@pytest.mark.mpl_image_compare
def test_line_plot_color_2_legend(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax, color='variable', legend=True)
    return fig


@pytest.mark.mpl_image_compare
def test_line_color_1_legend(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax, color='model', legend=True)
    return fig

#


@pytest.mark.mpl_image_compare
def test_line_marker_1(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax, marker='model')
    return fig


@pytest.mark.mpl_image_compare
def test_line_plot_marker_2(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax, marker='variable')
    return fig


@pytest.mark.mpl_image_compare
def test_line_plot_marker_2_legend(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax, marker='variable', legend=True)
    return fig


@pytest.mark.mpl_image_compare
def test_line_marker_1_legend(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax, marker='model', legend=True)
    return fig

#


@pytest.mark.mpl_image_compare
def test_line_linestyle_1(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax, linestyle='model')
    return fig


@pytest.mark.mpl_image_compare
def test_line_plot_linestyle_2(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax, linestyle='variable')
    return fig


@pytest.mark.mpl_image_compare
def test_line_plot_linestyle_2_legend(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax, linestyle='variable', legend=True)
    return fig


@pytest.mark.mpl_image_compare
def test_line_linestyle_1_legend(test_ia):
    fig, ax = plt.subplots()
    test_ia.line_plot(ax=ax, linestyle='model', legend=True)
    return fig
