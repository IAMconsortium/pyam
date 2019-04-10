import matplotlib
import pytest
import os
import copy
import numpy as np
import pandas as pd
import pyam
import warnings

# on CI, freetype version 2.6.1 works, but 2.8.0 does not
# if we want to move to 2.8.0, then we will need to regenerate images
FREETYPE_VERSION = matplotlib.ft2font.__freetype_version__
if int(FREETYPE_VERSION.replace('.', '')) < 291:
    msg = 'Freetype version < 2.9.1: {}'.format(FREETYPE_VERSION)
    warnings.warn('test_plotting.py is being skipped due to a '
                  'Freetype Version mismatch: {}'.format(msg))
    pytest.skip(msg, allow_module_level=True)

try:
    import cartopy
    has_cartopy = True
except ImportError:
    has_cartopy = False

import matplotlib.pyplot as plt

from contextlib import contextmanager

from pyam import IamDataFrame, plotting, run_control, reset_rc_defaults

from conftest import IMAGE_BASELINE_DIR, TEST_DATA_DIR

IS_WINDOWS = os.name == 'nt'
TOLERANCE = 6 if IS_WINDOWS else 2

MPL_KWARGS = {
    'style': 'ggplot',
    'baseline_dir': IMAGE_BASELINE_DIR,
    'tolerance': TOLERANCE,
    'savefig_kwargs': {'dpi': 300, 'bbox_inches': 'tight'},
}


@contextmanager
def update_run_control(update):
    run_control().update(update)
    yield
    reset_rc_defaults()


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot(plot_df):
    _plot_df = copy.deepcopy(plot_df)
    _plot_df.set_meta(meta=[np.nan] * 4, name='test')
    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_df.line_plot(ax=ax, legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_dict_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, legend=dict(loc='outside right'))
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_bottom_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, legend=dict(loc='outside bottom'))
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_no_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, legend=False)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_color(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, color='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_PYAM_COLORS(plot_df):
    # add a family of lines for each color in plotting.PYAM_COLORS separated by
    # a small offset
    update = {'color': {'model': {}}}
    _df = plot_df.filter(
        model='test_model',
        variable='Primary Energy',
        scenario='test_scenario1',
    ).data.copy()
    dfs = []
    for i, color in enumerate(plotting.PYAM_COLORS):
        df = _df.copy()
        model = color
        df['model'] = model
        df['value'] += i
        update['color']['model'][model] = color
        dfs.append(df)
    df = pyam.IamDataFrame(pd.concat(dfs))
    fig, ax = plt.subplots(figsize=(8, 8))
    with update_run_control(update):
        df.line_plot(ax=ax, color='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_color_fill_between(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, color='model', fill_between=True, legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_color_fill_between_interpolate(plot_df):
    # designed to create the sawtooth behavior at a midpoint with missing data
    df = pyam.IamDataFrame(plot_df.data.copy())
    fig, ax = plt.subplots(figsize=(8, 8))
    newdata = ['test_model1', 'test_scenario1', 'World', 'Primary Energy|Coal',
               'EJ/y', 2010, 3.50]
    df.data.loc[len(df.data) - 1] = newdata
    newdata = ['test_model1', 'test_scenario1', 'World', 'Primary Energy|Coal',
               'EJ/y', 2012, 3.50]
    df.data.loc[len(df.data)] = newdata
    newdata = ['test_model1', 'test_scenario1', 'World', 'Primary Energy|Coal',
               'EJ/y', 2015, 3.50]
    df.data.loc[len(df.data) + 1] = newdata
    df.line_plot(ax=ax, color='model', fill_between=True, legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_color_final_ranges(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, color='model', final_ranges=True, legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_marker_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, marker='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_rm_legend_label(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, marker='model', linestyle='scenario', legend=True,
                      rm_legend_label='marker')
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_linestyle_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, linestyle='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_single_color(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.line_plot(ax=ax, color='b', linestyle='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_filter_title(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.filter(variable='Primary Energy|Coal').line_plot(
        ax=ax, color='model', marker='scenario', legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_update_rc(plot_df):
    with update_run_control({'color': {'model': {'test_model1': 'cyan'}}}):
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_df.line_plot(ax=ax, color='model', legend=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_1_var(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    (plot_df
     .filter(model='test_model', scenario='test_scenario')
     .line_plot(x='Primary Energy', y='year', ax=ax, legend=False)
     )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_line_plot_2_vars(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    (plot_df
     .filter(model='test_model', scenario='test_scenario')
     .line_plot(x='Primary Energy|Coal', y='Primary Energy', ax=ax, legend=False)
     )
    return fig


@pytest.mark.skipif(not has_cartopy, reason="requires cartopy")
@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_region():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_iso_data.csv'))
    fig, ax = plt.subplots(
        subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(10, 7))
    (df
        .region_plot(
            ax=ax,
            cbar=False,
        )
     )
    return fig


@pytest.mark.skipif(not has_cartopy, reason="requires cartopy")
@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_region_cbar():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_iso_data.csv'))
    fig, ax = plt.subplots(
        subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(10, 7))
    (df
        .region_plot(
            ax=ax,
            cbar=True,
        )
     )
    return fig


@pytest.mark.skipif(not has_cartopy, reason="requires cartopy")
@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_region_cbar_args():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_iso_data.csv'))
    fig, ax = plt.subplots(
        subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(10, 7))
    (df
        .region_plot(
            ax=ax,
            cbar={'extend': 'both'},
        )
     )
    return fig


@pytest.mark.skipif(not has_cartopy, reason="requires cartopy")
@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_region_vmin_vmax():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_iso_data.csv'))
    fig, ax = plt.subplots(
        subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(10, 7))
    (df
        .region_plot(
            ax=ax,
            vmin=0.2,
            vmax=0.4,
            cbar=False,
        )
     )
    return fig


@pytest.mark.skipif(not has_cartopy, reason="requires cartopy")
@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_region_cmap():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_iso_data.csv'))
    fig, ax = plt.subplots(
        subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(10, 7))
    (df
        .region_plot(
            ax=ax,
            cmap='magma_r',
            cbar=False,
        )
     )
    return fig


@pytest.mark.skipif(not has_cartopy, reason="requires cartopy")
@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_region_crs():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_iso_data.csv'))
    crs = cartopy.crs.Robinson()
    fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=(10, 7))
    (df
        .region_plot(
            ax=ax,
            crs=crs,
            cbar=False,
        )
     )
    return fig


@pytest.mark.skipif(not has_cartopy, reason="requires cartopy")
@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_region_map_regions():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_region_data.csv'))
    fig, ax = plt.subplots(
        subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(10, 7))
    (df
        .map_regions('iso')
        .region_plot(
            ax=ax,
            cbar=False,
        )
     )
    return fig


@pytest.mark.skipif(not has_cartopy, reason="requires cartopy")
@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_region_map_regions_legend():
    df = IamDataFrame(os.path.join(TEST_DATA_DIR, 'plot_region_data.csv'))
    fig, ax = plt.subplots(
        subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(10, 7))
    (df
        .map_regions('iso')
        .region_plot(
            ax=ax,
            legend=True,
            cbar=False,
        )
     )
    return fig


def test_bar_plot_raises(plot_df):
    pytest.raises(ValueError, plot_df.bar_plot)


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_bar_plot(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    (plot_df
     .filter(variable='Primary Energy', model='test_model')
     .bar_plot(ax=ax, bars='scenario')
     )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_bar_plot_h(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    (plot_df
     .filter(variable='Primary Energy', model='test_model')
     .bar_plot(ax=ax, bars='scenario',
               orient='h')
     )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_bar_plot_stacked(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    (plot_df
     .filter(variable='Primary Energy', model='test_model')
     .bar_plot(ax=ax, bars='scenario',
               stacked=True)
     )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_bar_plot_stacked_net_line(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    # explicitly add negative contributions for net lines
    df = pyam.IamDataFrame(plot_df.data.copy())
    vals = [(2005, 0.35), (2010, -1.0), (2015, -4.0)]
    for i, (y, v) in enumerate(vals):
        newdata = ['test_model1', 'test_scenario1', 'World',
                   'Primary Energy|foo', 'EJ/y', y, v]
        df.data.loc[len(df) + i] = newdata
    (df
     .filter(variable='Primary Energy|*', model='test_model1',
             scenario='test_scenario1', region='World')
     .bar_plot(ax=ax, bars='variable',
               stacked=True)
     )
    plotting.add_net_values_to_bar_plot(ax, color='r')
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_bar_plot_title(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    (plot_df
     .filter(variable='Primary Energy', model='test_model')
     .bar_plot(ax=ax, bars='scenario',
               title='foo')
     )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_bar_plot_rc(plot_df):
    with update_run_control({'color': {'scenario': {'test_scenario': 'black'}}}):
        fig, ax = plt.subplots(figsize=(8, 8))
        (plot_df
         .filter(variable='Primary Energy', model='test_model')
         .bar_plot(ax=ax, bars='scenario')
         )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_pie_plot_labels(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    (plot_df
     .filter(variable='Primary Energy', model='test_model', year=2010)
     .pie_plot(ax=ax, category='scenario')
     )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_pie_plot_legend(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    (plot_df
     .filter(variable='Primary Energy', model='test_model', year=2010)
     .pie_plot(ax=ax, category='scenario', labels=None, legend=True)
     )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_pie_plot_other(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    with update_run_control({'color': {'scenario': {'test_scenario': 'black'}}}):
        (plot_df
         .filter(variable='Primary Energy', model='test_model', year=2010)
         .pie_plot(ax=ax, category='scenario', cmap='viridis', title='foo')
         )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_stack_plot(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    (plot_df
     .filter(variable='Primary Energy', model='test_model')
     .stack_plot(ax=ax, stack='scenario')
     )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_stack_plot_other(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    with update_run_control({'color': {'scenario': {'test_scenario': 'black'}}}):
        (plot_df
         .filter(variable='Primary Energy', model='test_model')
         .stack_plot(ax=ax, stack='scenario', cmap='viridis', title='foo')
         )
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_scatter(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.scatter(ax=ax, x='Primary Energy', y='Primary Energy|Coal')
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_scatter_variables_with_meta_color(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.categorize(
        'foo', 'a',
        criteria={'Primary Energy': {'up': 5, 'year': 2010}},
        color='blue'
    )
    plot_df.categorize(
        'foo', 'b',
        criteria={'Primary Energy': {'lo': 5, 'year': 2010}},
        color='red'
    )
    plot_df.scatter(ax=ax, x='Primary Energy', y='Primary Energy|Coal',
                    color='foo')
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_scatter_with_lines(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.scatter(ax=ax, x='Primary Energy', y='Primary Energy|Coal',
                    with_lines=True)
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_scatter_meta(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df.set_meta(meta=plot_df.filter(variable='Primary Energy')
                     .timeseries()[2010], name='Total')
    plot_df.set_meta(meta=plot_df.filter(variable='Primary Energy|Coal')
                     .timeseries()[2010], name='Coal')
    plot_df.scatter(ax=ax, x='Total', y='Coal')
    return fig


@pytest.mark.mpl_image_compare(**MPL_KWARGS)
def test_add_panel_label(plot_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    plotting.set_panel_label('test', ax=ax, x=0.5, y=0.5)
    return fig
