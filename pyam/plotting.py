import itertools
import logging

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from collections import defaultdict, Iterable


from pyam.run_control import run_control
from pyam.utils import requires_package, IAMC_IDX, SORT_IDX, isstr
# TODO: this is a hotfix for changes in pandas 0.25.0, per discussions on the
# pandas-dev listserv, we should try to ask if matplotlib would make it a
# standard feature in their library
from pyam._style import _get_standard_colors

logger = logging.getLogger(__name__)

# line colors, markers, and styles that are cycled through when not
# explicitly declared
_DEFAULT_PROPS = None

# maximum number of labels after which do not show legends by default
MAX_LEGEND_LABELS = 13

# default legend kwargs for putting legends outside of plots
OUTSIDE_LEGEND = {
    'right': dict(loc='center left', bbox_to_anchor=(1.0, 0.5)),
    'bottom': dict(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3),
}

PYAM_COLORS = {
    # AR6 colours from https://github.com/IPCC-WG1/colormaps
    # where each file is processed to generate hex values, e.g.:
    # for liwith open('rcp_cat.txt') as f:
    #   for l in f.readlines():
    #     rgb = np.array([int(x) for x in l.strip().split()]) / 256
    #     print(matplotlib.colors.rgb2hex(rgb))
    'AR6-SSP1': "#1e9583",
    'AR6-SSP2': "#4576be",
    'AR6-SSP3': "#f11111",
    'AR6-SSP4': "#e78731",
    'AR6-SSP5': "#8036a7",
    'AR6-SSP1-1.9': "#1e9583",
    'AR6-SSP1-2.6': "#1d3354",
    'AR6-SSP2-4.5': "#e9dc3d",
    'AR6-SSP3-7.0': "#f11111",
    'AR6-SSP3-LowNTCF': "#f11111",
    'AR6-SSP4-3.4': "#63bce4",
    'AR6-SSP4-6.0': "#e78731",
    'AR6-SSP5-3.4-OS': "#996dc8",
    'AR6-SSP5-8.5': "#830b22",
    'AR6-RCP-2.6': "#980002",
    'AR6-RCP-4.5': "#c37900",
    'AR6-RCP-6.0': "#709fcc",
    'AR6-RCP-8.5': "#003466",
    # AR5 colours from
    # https://tdaviesbarnard.co.uk/1202/ipcc-official-colors-rcp/
    'AR5-RCP-2.6': "#0000FF",
    'AR5-RCP-4.5': "#79BCFF",
    'AR5-RCP-6.0': "#FF822D",
    'AR5-RCP-8.5': "#FF0000",
}


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


def assign_style_props(df, color=None, marker=None, linestyle=None,
                       cmap=None):
    """Assign the style properties for a plot

    Parameters
    ----------
    df : pd.DataFrame
        data to be used for style properties
    """
    if color is None and cmap is not None:
        raise ValueError('`cmap` must be provided with the `color` argument')

    # determine color, marker, and linestyle for each line
    n = len(df[color].unique()) if color in df.columns else \
        len(df[list(set(df.columns) & set(IAMC_IDX))].drop_duplicates())
    defaults = default_props(reset=True, num_colors=n, colormap=cmap)

    props = {}
    rc = run_control()

    kinds = [('color', color), ('marker', marker), ('linestyle', linestyle)]

    for kind, var in kinds:
        rc_has_kind = kind in rc
        if var in df.columns:
            rc_has_var = rc_has_kind and var in rc[kind]
            props_for_kind = {}

            for val in df[var].unique():
                if rc_has_var and val in rc[kind][var]:
                    props_for_kind[val] = rc[kind][var][val]
                    # cycle any way to keep defaults the same
                    next(defaults[kind])
                else:
                    props_for_kind[val] = next(defaults[kind])
            props[kind] = props_for_kind

    # update for special properties only if they exist in props
    if 'color' in props:
        d = props['color']
        values = list(d.values())
        # find if any colors in our properties corresponds with special colors
        # we know about
        overlap_idx = np.in1d(values, list(PYAM_COLORS.keys()))
        if overlap_idx.any():  # some exist in our special set
            keys = np.array(list(d.keys()))[overlap_idx]
            values = np.array(values)[overlap_idx]
            # translate each from pyam name, like AR6-SSP2-45 to proper color
            # designation
            for k, v in zip(keys, values):
                d[k] = PYAM_COLORS[v]
            # replace props with updated dict without special colors
            props['color'] = d
    return props


def reshape_line_plot(df, x, y):
    """Reshape data from long form to "line plot form".

    Line plot form has x value as the index with one column for each line.
    Each column has data points as values and all metadata as column headers.
    """
    idx = list(df.columns.drop(y))
    if df.duplicated(idx).any():
        logger.warning('Duplicated index found.')
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
        logger.warning('Duplicated index found.')
        df = df.drop_duplicates(idx, keep='last')
    df = df.set_index(idx)[y].unstack(x).T
    return df


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
            msg = 'Can not plot multiple {}s in pie_plot with value={},' +\
                ' category={}'
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
               ax=None, legend=True, title=True, cmap=None, total=None,
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
    total : bool or dict, optional
        If True, plot a total line with default pyam settings. If a dict, then
        plot the total line using the dict key-value pairs as keyword arguments
        to ax.plot(). If None, do not plot the total line.
        default : None
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

    # Line below is for interpolation. On datetimes I think you'd downcast to
    # seconds first and then cast back to datetime at the end..?
    _df.index = _df.index.astype(float)

    time_original = _df.index.values
    first_zero_times = pd.DataFrame(index=["first_zero_time"])

    both_positive_and_negative = _df.apply(
        lambda x: (x >= 0).any() and (x < 0).any()
    )
    for col in _df.loc[:, both_positive_and_negative]:
        values = _df[col].dropna().values
        positive = (values >= 0)
        negative = (values < 0)
        pos_to_neg = positive[:-1] & negative[1:]
        neg_to_pos = positive[1:] & negative[:-1]
        crosses = np.argwhere(pos_to_neg | neg_to_pos)
        for i, cross in enumerate(crosses):
            cross = cross[0]  # get location
            x_1 = time_original[cross]
            x_2 = time_original[cross + 1]
            y_1 = values[cross]
            y_2 = values[cross + 1]

            zero_time = x_1 - y_1 * (x_2 - x_1) / (y_2 - y_1)
            if i == 0:
                first_zero_times.loc[:, col] = zero_time
            if zero_time not in _df.index.values:
                _df.loc[zero_time, :] = np.nan

    first_zero_times = first_zero_times.sort_values(
        by="first_zero_time",
        axis=1,
    )
    _df = _df.reindex(sorted(_df.index)).interpolate(method="values")

    # Sort lines so that negative timeseries are on the right, positive
    # timeseries are on the left and timeseries which go from positive to
    # negative are ordered such that the timeseries which goes negative first
    # is on the right (case of timeseries which go from negative to positive
    # is an edge case we haven't thought about as it's unlikely to apply to
    # us).
    pos_cols = [c for c in _df if (_df[c] >= 0).all()]
    cross_cols = first_zero_times.columns[::-1].tolist()
    neg_cols = [c for c in _df if (_df[c] < 0).all()]
    col_order = pos_cols + cross_cols + neg_cols
    _df = _df[col_order]

    # explicitly get colors
    defaults = default_props(reset=True, num_colors=len(_df.columns),
                             colormap=cmap)['color']
    rc = run_control()
    colors = {}
    for key in _df.columns:
        c = next(defaults)
        c_in_rc = 'color' in rc
        if c_in_rc and stack in rc['color'] and key in rc['color'][stack]:
            c = rc['color'][stack][key]
        colors[key] = c

    # plot stacks, starting from the top and working our way down to the bottom
    negative_only_cumulative = _df.applymap(
        lambda x: x if x < 0 else 0
    ).cumsum(axis=1)
    positive_only_cumulative = _df.applymap(lambda x: x if x >= 0 else 0)[
        col_order[::-1]
    ].cumsum(axis=1)[
        col_order
    ]
    time = _df.index.values
    upper = positive_only_cumulative.iloc[:, 0].values
    for j, col in enumerate(_df):
        noc_tr = negative_only_cumulative.iloc[:, j].values
        try:
            poc_nr = positive_only_cumulative.iloc[:, j + 1].values
        except IndexError:
            poc_nr = np.zeros_like(upper)
        lower = poc_nr.copy()
        if (noc_tr < 0).any():
            lower[np.where(poc_nr == 0)] = noc_tr[np.where(poc_nr == 0)]

        ax.fill_between(time, lower, upper, label=col,
                        color=colors[col], **kwargs)
        upper = lower.copy()

    # add total
    if (total is not None) and total:  # cover case where total=False
        if isinstance(total, bool):  # can now assume total=True
            total = {}
        total.setdefault("label", "Total")
        total.setdefault("color", "black")
        total.setdefault("lw", 4.0)
        ax.plot(time, _df.sum(axis=1), **total)

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


def _get_boxes(ax, xoffset=0.05, width_weight=0.1):
    xys = {}
    widths = {}
    heights = defaultdict(list)
    for b in ax.get_children():
        if isinstance(b, mpatches.Rectangle) and b.xy != (0, 0):
            x, y = b.xy
            heights[x].append(b.get_height())
            widths[x] = b.get_width() * width_weight
            xys[x] = ((x + b.get_width()) + xoffset, 0)
    return {x: (xys[x], widths[x], sum(heights[x])) for x in xys.keys()}


def add_net_values_to_bar_plot(axs, color='k'):
    """Add net values next to an existing vertical stacked bar chart

    Parameters
    ----------
    axs : matplotlib.Axes or list thereof
    color : str, optional, default: black
        the color of the bars to add
    """
    axs = axs if isinstance(axs, Iterable) else [axs]
    for ax in axs:
        box_args = _get_boxes(ax)
        for x, args in box_args.items():
            rect = mpatches.Rectangle(*args, color=color)
            ax.add_patch(rect)


def scatter(df, x, y, ax=None, legend=None, title=None,
            color=None, marker='o', linestyle=None, cmap=None,
            groupby=['model', 'scenario'], with_lines=False, **kwargs):
    """Plot data as a scatter chart.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot as a long-form data frame
    x : str
        column to be plotted on the x-axis
    y : str
        column to be plotted on the y-axis
    ax : matplotlib.Axes, optional
    legend : bool, optional
        Include a legend (`None` displays legend only if less than 13 entries)
        default: None
    title : bool or string, optional
        Display a custom title.
    color : string, optional
        A valid matplotlib color or column name. If a column name, common
        values will be provided the same color.
        default: None
    marker : string
        A valid matplotlib marker or column name. If a column name, common
        values will be provided the same marker.
        default: 'o'
    linestyle : string, optional
        A valid matplotlib linestyle or column name. If a column name, common
        values will be provided the same linestyle.
        default: None
    cmap : string, optional
        A colormap to use.
        default: None
    groupby : list-like, optional
        Data grouping for plotting.
        default: ['model', 'scenario']
    with_lines : bool, optional
        Make the scatter plot with lines connecting common data.
        default: False
    kwargs : Additional arguments to pass to the pd.DataFrame.plot() function
    """
    if ax is None:
        fig, ax = plt.subplots()

    # assign styling properties
    props = assign_style_props(df, color=color, marker=marker,
                               linestyle=linestyle, cmap=cmap)

    # group data
    groups = df.groupby(groupby)

    # loop over grouped dataframe, plot data
    legend_data = []
    for name, group in groups:
        pargs = {}
        labels = []
        for key, kind, var in [('c', 'color', color),
                               ('marker', 'marker', marker),
                               ('linestyle', 'linestyle', linestyle)]:
            if kind in props:
                label = group[var].values[0]
                pargs[key] = props[kind][group[var].values[0]]
                labels.append(repr(label).lstrip("u'").strip("'"))
            else:
                pargs[key] = var

        if len(labels) > 0:
            legend_data.append(' '.join(labels))
        else:
            legend_data.append(' '.join(name))
        kwargs.update(pargs)
        if with_lines:
            ax.plot(group[x], group[y], **kwargs)
        else:
            kwargs.pop('linestyle')  # scatter() can't take a linestyle
            ax.scatter(group[x], group[y], **kwargs)

    # build legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    if legend_data != [''] * len(legend_data):
        labels = sorted(list(set(tuple(legend_data))))
        idxs = [legend_data.index(d) for d in labels]
        handles = [handles[i] for i in idxs]
    if legend is None and len(labels) < 13 or legend is not False:
        _add_legend(ax, handles, labels, legend)

    # add labels and title
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)

    return ax


def line_plot(df, x='year', y='value', ax=None, legend=None, title=True,
              color=None, marker=None, linestyle=None, cmap=None,
              fill_between=None, final_ranges=None,
              rm_legend_label=[], **kwargs):
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
    legend : bool or dictionary, optional
        Add a legend. If a dictionary is provided, it will be used as keyword
        arguments in creating the legend.
        default: None (displays legend only if less than 13 entries)
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
    fill_between : boolean or dict, optional
        Fill lines between minima/maxima of the 'color' argument. This can only
        be used if also providing a 'color' argument. If this is True, then
        default arguments will be provided to `ax.fill_between()`. If this is a
        dictionary, those arguments will be provided instead of defaults.
        default: None
    final_ranges : boolean or dict, optional
        Add vertical line between minima/maxima of the 'color' argument in the
        last period plotted.  This can only be used if also providing a 'color'
        argument. If this is True, then default arguments will be provided to
        `ax.axvline()`. If this is a dictionary, those arguments will be
        provided instead of defaults.
        default: None
    rm_legend_label : string, list, optional
        Remove the color, marker, or linestyle label in the legend.
        default: []
    kwargs : Additional arguments to pass to the pd.DataFrame.plot() function
    """
    if ax is None:
        fig, ax = plt.subplots()

    # assign styling properties
    props = assign_style_props(df, color=color, marker=marker,
                               linestyle=linestyle, cmap=cmap)

    if fill_between and 'color' not in props:
        raise ValueError('Must use `color` kwarg if using `fill_between`')
    if final_ranges and 'color' not in props:
        raise ValueError('Must use `color` kwarg if using `final_ranges`')

    # reshape data for use in line_plot
    df = reshape_line_plot(df, x, y)  # long form to one column per line

    # determine index of column name in reshaped dataframe
    prop_idx = {}
    for kind, var in [('color', color), ('marker', marker),
                      ('linestyle', linestyle)]:
        if var is not None and var in df.columns.names:
            prop_idx[kind] = df.columns.names.index(var)

    # plot data, keeping track of which legend labels to apply
    no_label = [rm_legend_label] if isstr(rm_legend_label) else rm_legend_label

    for col, data in df.iteritems():
        pargs = {}
        labels = []
        # build plotting args and line legend labels
        for key, kind, var in [('c', 'color', color),
                               ('marker', 'marker', marker),
                               ('linestyle', 'linestyle', linestyle)]:
            if kind in props:
                label = col[prop_idx[kind]]
                pargs[key] = props[kind][label]
                if kind not in no_label:
                    labels.append(repr(label).lstrip("u'").strip("'"))
            else:
                pargs[key] = var
        kwargs.update(pargs)
        data = data.dropna()
        data.plot(ax=ax, **kwargs)
        if labels:
            ax.lines[-1].set_label(' '.join(labels))

    if fill_between:
        _kwargs = {'alpha': 0.25} if fill_between in [True, None] \
            else fill_between
        data = df.T
        columns = data.columns
        # get outer boundary mins and maxes
        allmins = data.groupby(color).min()
        intermins = (
            data.dropna(axis=1).groupby(color).min()  # nonan data
            .reindex(columns=columns)  # refill with nans
            .T.interpolate(method='index').T  # interpolate
        )
        mins = pd.concat([allmins, intermins]).min(level=0)
        allmaxs = data.groupby(color).max()
        intermaxs = (
            data.dropna(axis=1).groupby(color).max()  # nonan data
            .reindex(columns=columns)  # refill with nans
            .T.interpolate(method='index').T  # interpolate
        )
        maxs = pd.concat([allmaxs, intermaxs]).max(level=0)
        # do the fill
        for idx in mins.index:
            ymin = mins.loc[idx]
            ymax = maxs.loc[idx]
            ax.fill_between(ymin.index, ymin, ymax,
                            facecolor=props['color'][idx], **_kwargs)

    # add bars to the end of the plot showing range
    if final_ranges:
        # have to explicitly draw it to get the tick labels (these change once
        # you add the vlines)
        plt.gcf().canvas.draw()
        _kwargs = {'linewidth': 2} if final_ranges in [True, None] \
            else final_ranges
        first = df.index[0]
        final = df.index[-1]
        mins = df.T.groupby(color).min()[final]
        maxs = df.T.groupby(color).max()[final]
        ymin, ymax = ax.get_ylim()
        ydiff = ymax - ymin
        xmin, xmax = ax.get_xlim()
        xdiff = xmax - xmin
        xticks = ax.get_xticks()
        xlabels = ax.get_xticklabels()
        # 1.5% increase seems to be ok per extra line
        extra_space = 0.015
        for i, idx in enumerate(mins.index):
            xpos = final + xdiff * extra_space * (i + 1)
            _ymin = (mins[idx] - ymin) / ydiff
            _ymax = (maxs[idx] - ymin) / ydiff
            ax.axvline(xpos, ymin=_ymin, ymax=_ymax,
                       color=props['color'][idx], **_kwargs)
        # for equal spacing between xmin and first datapoint and xmax and last
        # line
        ax.set_xlim(xmin, xpos + first - xmin)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

    # build unique legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = np.array(handles), np.array(labels)
    _, idx = np.unique(labels, return_index=True)
    handles, labels = handles[idx], labels[idx]
    if legend is not False:
        _add_legend(ax, handles, labels, legend)

    # add default labels if possible
    ax.set_xlabel(x.title())
    units = df.columns.get_level_values('unit').unique()
    units_for_ylabel = len(units) == 1 and x == 'year' and y == 'value'
    ylabel = units[0] if units_for_ylabel else y.title()
    ax.set_ylabel(ylabel)

    # build a default title if possible
    if title:
        default_title = []
        for var in ['model', 'scenario', 'region', 'variable']:
            if var in df.columns.names:
                values = df.columns.get_level_values(var).unique()
                if len(values) == 1:
                    default_title.append('{}: {}'.format(var, values[0]))
        title = ' '.join(default_title) if title is True else title
        ax.set_title(title)

    return ax, handles, labels


def _add_legend(ax, handles, labels, legend):
    if legend is None and len(labels) >= MAX_LEGEND_LABELS:
        logger.info('>={} labels, not applying legend'.format(
            MAX_LEGEND_LABELS))
    else:
        legend = {} if legend in [True, None] else legend
        loc = legend.pop('loc', 'best')
        outside = loc.split(' ')[1] if loc.startswith('outside ') else False
        _legend = OUTSIDE_LEGEND[outside] if outside else dict(loc=loc)
        _legend.update(legend)
        ax.legend(handles, labels, **_legend)


def set_panel_label(label, ax=None, x=0.05, y=0.9):
    """Add a panel label to the figure/axes, by default in the top-left corner

    Parameters
    ----------
    label : str
        text to be added as panel label
    ax : matplotlib.Axes, optional
        panel to which to add the panel label
    x : number, default 0.05
        relative location of label to x-axis
    y : number, default 0.9
        relative location of label to y-axis
    """
    def _lim_loc(lim, loc):
        return lim[0] + (lim[1] - lim[0]) * loc

    if ax is not None:
        ax.text(_lim_loc(ax.get_xlim(), x), _lim_loc(ax.get_ylim(), y), label)
    else:
        plt.text(_lim_loc(plt.xlim(), x), _lim_loc(plt.ylim(), y), label)
