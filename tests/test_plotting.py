import cartopy
import pytest
import os

import matplotlib.pyplot as plt

from pyam_analysis import IamDataFrame, plotting

from testing_utils import plot_df, IMAGE_BASELINE_DIR, TEST_DATA_DIR


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_plot(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_color(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, color='model')
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_color_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, color='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_marker_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, marker='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_linestyle_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, linestyle='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_single_color(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, color='b', linestyle='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_filter_title(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.filter({'variable': 'Primary Energy|Coal'}).line_plot(
        ax=ax, color='model', marker='scenario', legend=True)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_line_update_rc(plot_df):
    update = {'color': {'model': {'test_model1': 'cyan'}}}
    plotting.run_control().update(update)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, color='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_region():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_iso_data.csv'))
    fig, ax = plt.subplots(
        subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(10, 7))
    df.region_plot(
        ax=ax,
    )
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_region_vmin_vmax():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_iso_data.csv'))
    fig, ax = plt.subplots(
        subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(10, 7))
    df.region_plot(
        ax=ax,
        vmin=0.2,
        vmax=0.4,
    )
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_region_cmap():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_iso_data.csv'))
    fig, ax = plt.subplots(
        subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(10, 7))
    df.region_plot(
        ax=ax,
        cmap='magma_r',
    )
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_region_crs():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_iso_data.csv'))
    crs = cartopy.crs.Robinson()
    fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=(10, 7))
    df.region_plot(
        ax=ax,
        crs=crs,
    )
    return fig


def test_region_axes_raises():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_iso_data.csv'))
    fig, ax = plt.subplots(figsize=(10, 7))
    pytest.raises(ValueError, df.region_plot, ax=ax)


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR)
def test_region_map_regions():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_region_data.csv'))
    fig, ax = plt.subplots(
        subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(10, 7))
    df.region_plot(
        ax=ax,
        map_regions=True,
    )
    return fig


@pytest.mark.mpl_image_compare(style='ggplot', baseline_dir=IMAGE_BASELINE_DIR,
                               savefig_kwargs={'bbox_inches': 'tight'})
def test_region_map_regions_legend():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_region_data.csv'))
    fig, ax = plt.subplots(
        subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(10, 7))
    df.region_plot(
        ax=ax,
        map_regions=True,
        legend=True,
    )
    return fig
