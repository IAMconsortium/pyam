import itertools
import os
import warnings

try:
    import cartopy
except ImportError:
    cartopy = None

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches
import numpy as np

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from collections import defaultdict
from contextlib import contextmanager

try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

from pyam.utils import SORT_IDX
from pyam.run_control import run_control
from pyam.utils import requires_package

from pandas.plotting._style import _get_standard_colors

# line colors, markers, and styles that are cycled through when not
# explicitly declared
_DEFAULT_PROPS = None


def reset_default_props(**kwargs):
    """Reset properties to initial cycle point"""
    global _DEFAULT_PROPS
    pcycle = plt.rcParams['axes.prop_cycle']
    _DEFAULT_PROPS = {
        'color': itertools.cycle(_get_standard_colors(**kwargs))
        if len(kwargs) > 0 else itertools.cycle([x['color'] for x in pcycle]),
        'marker': itertools.cycle(['o', 'x', '.', '+', '*']),
        'linestyle': itertools.cycle(['-', '--', '-.', ':']),
    }


def default_props(reset=False, **kwargs):
    """Return current default properties

    Parameters
    ----------
    reset : bool
            if True, reset properties and return
            default: False
    """
    global _DEFAULT_PROPS
    if _DEFAULT_PROPS is None or reset:
        reset_default_props(**kwargs)
    return _DEFAULT_PROPS


def reshape_line_plot(df, x, y):
    """Reshape data from long form to "line plot form".

    Line plot form has x value as the index with one column for each line.
    Each column has data points as values and all metadata as column headers.
    """
    idx = list(df.columns.drop(y))
    if df.duplicated(idx).any():
        warnings.warn('Duplicated index found.')
        df = df.drop_duplicates(idx, keep='last')
    df = df.set_index(idx)[y].unstack(x).T
    return df


def reshape_bar_plot(df, x, y, bars):
    """Reshape data from long form to "bar plot form".

    Bar plot form has x value as the index with one column for bar grouping.
    Table values come from y values.
    """
    idx = [bars, x]
    if df.duplicated(idx).any():
        warnings.warn('Duplicated index found.')
        df = df.drop_duplicates(idx, keep='last')
    df = df.set_index(idx)[y].unstack(x).T
    return df


@requires_package(gpd, 'Requires geopandas')
@lru_cache()
def read_shapefile(fname, region_col=None, **kwargs):
    """Read a shapefile for use in regional plots. Shapefiles must have a 
    column denoted as "region".

    Parameters
    ----------
    fname : string
        path to shapefile to be read by geopandas
    region_col : string, default None
        if provided, rename a column in the shapefile to "region"
    """
    gdf = gpd.read_file(fname, **kwargs)
    if region_col is not None:
        gdf = gdf.rename(columns={region_col: 'region'})
    if 'region' not in gdf.columns:
        raise IOError('Must provide a region column')
    gdf['region'] = gdf['region'].str.upper()
    return gdf


@requires_package(gpd, 'Requires geopandas')
@requires_package(cartopy, 'Requires cartopy')
def region_plot(df, column='value', ax=None, crs=None, gdf=None, add_features=True,
                vmin=None, vmax=None, cmap=None, cbar=True, legend=False,
                title=True):
    """Plot data on a map.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot as a long-form data frame
    column : string, optional, default: 'value'
        The column to use for plotting values
    ax : matplotlib.Axes, optional
    crs : cartopy.crs, optional
        The crs to plot, PlateCarree is used by default.
    gdf : geopandas.GeoDataFrame, optional
        The geometries to plot. The gdf must have a "region" column.
    add_features : bool, optional, default: True
        If true, add land, ocean, coastline, and border features.
    vmin : numeric, optional
        The minimum value to plot.
    vmax : numeric, optional
        The maximum value to plot.
    cmap : string, optional
        The colormap to use.
    cbar : bool or dictionary, optional, default: True
        Add a colorbar. If a dictionary is provided, it will be used as keyword 
        arguments in creating the colorbar.
    legend : bool or dictionary, optional, default: False
        Add a legend. If a dictionary is provided, it will be used as keyword 
        arguments in creating the legend.
    title : bool or string, optional
        Display a default or custom title.
    """
    for col in ['model', 'scenario', 'year', 'variable']:
        if len(df[col].unique()) > 1:
            msg = 'Can not plot multiple {}s in region_plot'
            raise ValueError(msg.format(col))

    crs = crs or cartopy.crs.PlateCarree()
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection=crs))
    elif not isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot):
        msg = 'Must provide a cartopy axes object, not: {}'
        raise ValueError(msg.format(type(ax)))

    gdf = gdf or read_shapefile(gpd.datasets.get_path('naturalearth_lowres'),
                                region_col='iso_a3')
    data = gdf.merge(df, on='region', how='inner').to_crs(crs.proj4_init)
    if data.empty:  # help users with iso codes
        df['region'] = df['region'].str.upper()
        data = gdf.merge(df, on='region', how='inner').to_crs(crs.proj4_init)
    if data.empty:
        raise ValueError('No data to plot')

    if add_features:
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.BORDERS)

    vmin = vmin or data['value'].min()
    vmax = vmax or data['value'].max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)
    labels = []
    handles = []
    for _, row in data.iterrows():
        label = row['label'] if 'label' in row else row['region']
        color = scalar_map.to_rgba(row['value'])
        ax.add_geometries(
            [row['geometry']],
            crs,
            facecolor=color,
            label=label,
        )
        if label not in labels:
            labels.append(label)
            handle = mpatches.Rectangle((0, 0), 5, 5, facecolor=color)
            handles.append(handle)

    if cbar:
        scalar_map._A = []  # for some reason you have to clear this
        if cbar is True:  # use some defaults
            cbar = dict(
                fraction=0.022,  # these are magic numbers
                pad=0.02,       # that just seem to "work"
            )
        cb = plt.colorbar(scalar_map, **cbar)

    if legend:
        if legend is True:  # use some defaults
            legend = dict(
                bbox_to_anchor=(1.32, 0.5) if cbar else (1.2, 0.5),
                loc='right',
            )
        ax.legend(handles, labels, **legend)

    if title:
        var = df['variable'].unique()[0]
        unit = df['unit'].unique()[0]
        year = df['year'].unique()[0]
        default_title = '{} ({}) in {}'.format(var, unit, year)
        title = default_title if title is True else title
        ax.set_title(title)

    return ax


def pie_plot(df, value='value', category='variable',
             ax=None, legend=False, title=True, cmap=None,
             **kwargs):
    """Plot data as a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot as a long-form data frame
    value : string, optional
        The column to use for data values
        default: value
    category : string, optional
        The column to use for labels
        default: variable
    ax : matplotlib.Axes, optional
    legend : bool, optional
        Include a legend
        default: False
    title : bool or string, optional
        Display a default or custom title.
    cmap : string, optional
        A colormap to use.
        default: None
    kwargs : Additional arguments to pass to the pd.DataFrame.plot() function
    """
    for col in set(SORT_IDX) - set([category]):
        if len(df[col].unique()) > 1:
            msg = 'Can not plot multiple {}s in pie_plot with value={}, category={}'
            raise ValueError(msg.format(col, value, category))

    if ax is None:
        fig, ax = plt.subplots()

    # get data, set negative values to explode
    _df = df.groupby(category)[value].sum()
    where = _df > 0
    explode = tuple(0 if _ else 0.2 for _ in where)
    _df = _df.abs()

    # explicitly get colors
    defaults = default_props(reset=True, num_colors=len(_df.index),
                             colormap=cmap)['color']
    rc = run_control()
    color = []
    for key, c in zip(_df.index, defaults):
        if 'color' in rc and \
           category in rc['color'] and \
           key in rc['color'][category]:
            c = rc['color'][category][key]
        color.append(c)

    # plot data
    _df.plot(kind='pie', colors=color, ax=ax, explode=explode, **kwargs)

    # add legend
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), labels=_df.index)
    if not legend:
        ax.legend_.remove()

    # remove label
    ax.set_ylabel('')

    return ax


def stack_plot(df, x='year', y='value', stack='variable',
               ax=None, legend=True, title=True, cmap=None,
               **kwargs):
    """Plot data as a stack chart.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot as a long-form data frame
    x : string, optional
        The column to use for x-axis values
        default: year
    y : string, optional
        The column to use for y-axis values
        default: value
    stack: string, optional
        The column to use for stack groupings
        default: variable
    ax : matplotlib.Axes, optional
    legend : bool, optional
        Include a legend
        default: False
    title : bool or string, optional
        Display a default or custom title.
    cmap : string, optional
        A colormap to use.
        default: None
    kwargs : Additional arguments to pass to the pd.DataFrame.plot() function
    """
    for col in set(SORT_IDX) - set([x, stack]):
        if len(df[col].unique()) > 1:
            msg = 'Can not plot multiple {}s in stack_plot with x={}, stack={}'
            raise ValueError(msg.format(col, x, stack))

    if ax is None:
        fig, ax = plt.subplots()

    # long form to one column per bar group
    _df = reshape_bar_plot(df, x, y, stack)

    # explicitly get colors
    defaults = default_props(reset=True, num_colors=len(_df.columns),
                             colormap=cmap)['color']
    rc = run_control()
    colors = []
    for key in _df.columns:
        c = next(defaults)
        if 'color' in rc and stack in rc['color'] and key in rc['color'][stack]:
            c = rc['color'][stack][key]
        colors.append(c)

    ax.stackplot(_df.index, _df.T, labels=_df.columns, colors=colors, **kwargs)

    # add legend
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    if not legend:
        ax.legend_.remove()

    # add default labels if possible
    ax.set_xlabel(x.capitalize())
    units = df['unit'].unique()
    if len(units) == 1:
        ax.set_ylabel(units[0])

    # build a default title if possible
    _title = []
    for var in ['model', 'scenario', 'region', 'variable']:
        values = df[var].unique()
        if len(values) == 1:
            _title.append('{}: {}'.format(var, values[0]))
    if title and _title:
        title = ' '.join(_title) if title is True else title
        ax.set_title(title)

    return ax


def bar_plot(df, x='year', y='value', bars='variable',
             ax=None, orient='v', legend=True, title=True, cmap=None,
             **kwargs):
    """Plot data as a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot as a long-form data frame
    x : string, optional
        The column to use for x-axis values
        default: year
    y : string, optional
        The column to use for y-axis values
        default: value
    bars: string, optional
        The column to use for bar groupings
        default: variable
    ax : matplotlib.Axes, optional
    orient : string, optional
        Vertical or horizontal orientation.
        default: variable
    legend : bool, optional
        Include a legend
        default: False
    title : bool or string, optional
        Display a default or custom title.
    cmap : string, optional
        A colormap to use.
        default: None
    kwargs : Additional arguments to pass to the pd.DataFrame.plot() function
    """
    for col in set(SORT_IDX) - set([x, bars]):
        if len(df[col].unique()) > 1:
            msg = 'Can not plot multiple {}s in bar_plot with x={}, bars={}'
            raise ValueError(msg.format(col, x, bars))

    if ax is None:
        fig, ax = plt.subplots()

    # long form to one column per bar group
    _df = reshape_bar_plot(df, x, y, bars)

    # explicitly get colors
    defaults = default_props(reset=True, num_colors=len(_df.columns),
                             colormap=cmap)['color']
    rc = run_control()
    color = []
    for key in _df.columns:
        c = next(defaults)
        if 'color' in rc and bars in rc['color'] and key in rc['color'][bars]:
            c = rc['color'][bars][key]
        color.append(c)

    # plot data
    kind = 'bar' if orient.startswith('v') else 'barh'
    _df.plot(kind=kind, color=color, ax=ax, **kwargs)

    # add legend
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    if not legend:
        ax.legend_.remove()

    # add default labels if possible
    if orient == 'v':
        ax.set_xlabel(x.capitalize())
    else:
        ax.set_ylabel(x.capitalize())
    units = df['unit'].unique()
    if len(units) == 1 and y == 'value':
        if orient == 'v':
            ax.set_ylabel(units[0])
        else:
            ax.set_xlabel(units[0])

    # build a default title if possible
    _title = []
    for var in ['model', 'scenario', 'region', 'variable']:
        values = df[var].unique()
        if len(values) == 1:
            _title.append('{}: {}'.format(var, values[0]))
    if title and _title:
        title = ' '.join(_title) if title is True else title
        ax.set_title(title)

    return ax


def line_plot(df, x='year', y='value', ax=None, legend=None, title=True,
              color=None, marker=None, linestyle=None, cmap=None,
              **kwargs):
    """Plot data as lines with or without markers.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot as a long-form data frame
    x : string, optional
        The column to use for x-axis values
        default: year
    y : string, optional
        The column to use for y-axis values
        default: value
    ax : matplotlib.Axes, optional
    legend : bool, optional
        Include a legend (`None` displays legend only if less than 13 entries)
        default: None
    title : bool or string, optional
        Display a default or custom title.
    color : string, optional
        A valid matplotlib color or column name. If a column name, common
        values will be provided the same color.
        default: None
    marker : string, optional
        A valid matplotlib marker or column name. If a column name, common
        values will be provided the same marker.
        default: None
    linestyle : string, optional
        A valid matplotlib linestyle or column name. If a column name, common
        values will be provided the same linestyle.
        default: None
    cmap : string, optional
        A colormap to use.
        default: None
    kwargs : Additional arguments to pass to the pd.DataFrame.plot() function
    """

    if ax is None:
        fig, ax = plt.subplots()

    df = reshape_line_plot(df, x, y)  # long form to one column per line

    # determine color, marker, and linestyle for each line
    defaults = default_props(reset=True, num_colors=len(df.columns),
                             colormap=cmap)
    props = {}
    prop_idx = {}
    rc = run_control()
    for kind, var in [('color', color), ('marker', marker), ('linestyle', linestyle)]:
        rc_has_kind = kind in rc
        if var in df.columns.names:
            rc_has_var = rc_has_kind and var in rc[kind]
            props_for_kind = {}
            for val in df.columns.get_level_values(var).unique():
                if rc_has_var and val in rc[kind][var]:
                    props_for_kind[val] = rc[kind][var][val]
                    # cycle any way to keep defaults the same
                    next(defaults[kind])
                else:
                    props_for_kind[val] = next(defaults[kind])
            props[kind] = props_for_kind
            prop_idx[kind] = df.columns.names.index(var)

    # plot data
    legend_data = []
    for col, data in df.iteritems():
        pargs = {}
        labels = []
        for key, kind, var in [('c', 'color', color),
                               ('marker', 'marker', marker),
                               ('linestyle', 'linestyle', linestyle)]:
            if kind in props:
                label = col[prop_idx[kind]]
                pargs[key] = props[kind][label]
                labels.append(repr(label).lstrip("u'").strip("'"))
            else:
                pargs[key] = var

        legend_data.append(' '.join(labels))
        kwargs.update(pargs)
        data = data.dropna()
        data.plot(ax=ax, **kwargs)

    # build legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    if legend_data != [''] * len(legend_data):
        labels = sorted(list(set(tuple(legend_data))))
        idxs = [legend_data.index(d) for d in labels]
        handles = [handles[i] for i in idxs]
    if legend is None and len(labels) < 13 or legend is True:
        ax.legend(handles, labels)

    # add default labels if possible
    ax.set_xlabel(x.title())
    units = df.columns.get_level_values('unit').unique()
    units_for_ylabel = len(units) == 1 and x == 'year' and y == 'value'
    ylabel = units[0] if units_for_ylabel else y.title()
    ax.set_ylabel(ylabel)

    # build a default title if possible
    _title = []
    for var in ['model', 'scenario', 'region', 'variable']:
        if var in df.columns.names:
            values = df.columns.get_level_values(var).unique()
            if len(values) == 1:
                _title.append('{}: {}'.format(var, values[0]))
    if title and _title:
        ax.set_title(' '.join(_title))

    return ax, handles, labels
