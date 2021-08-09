import itertools
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict
from collections.abc import Iterable

from pyam.run_control import run_control
from pyam.figures import sankey
from pyam.timeseries import cross_threshold
from pyam.utils import (
    META_IDX,
    IAMC_IDX,
    SORT_IDX,
    YEAR_IDX,
    isstr,
    to_list,
    _raise_data_error,
)
from pyam.index import get_index_levels

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
    "right": dict(loc="center left", bbox_to_anchor=(1.0, 0.5)),
    "bottom": dict(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3),
}

PYAM_COLORS = {
    # AR6 colours originally from https://github.com/IPCC-WG1/colormaps
    # Final values are used as communicated from the TSU and documented
    # in: https://github.com/IAMconsortium/pyam/pull/566
    # where each file is processed to generate hex values, e.g.:
    # with open('rcp_cat.txt') as f:
    #   for l in f.readlines():
    #     rgb = np.array([int(x) for x in l.strip().split()]) / 256
    #     print(matplotlib.colors.rgb2hex(rgb))
    "AR6-SSP1": "#1e9583",
    "AR6-SSP2": "#4576be",
    "AR6-SSP3": "#f11111",
    "AR6-SSP4": "#e78731",
    "AR6-SSP5": "#8036a7",
    "AR6-SSP1-1.9": "#00a9cf",
    "AR6-SSP1-2.6": "#003466",
    "AR6-SSP2-4.5": "#f69320",
    "AR6-SSP3-7.0": "#df0000",
    "AR6-SSP3-LowNTCF": "#e61d25",
    "AR6-SSP4-3.4": "#2274ae",
    "AR6-SSP4-6.0": "#b0724e",
    "AR6-SSP5-3.4-OS": "#92397a",
    "AR6-SSP5-8.5": "#980002",
    "AR6-RCP-2.6": "#003466",
    "AR6-RCP-4.5": "#709fcc",
    "AR6-RCP-6.0": "#c37900",
    "AR6-RCP-8.5": "#980002",
    # AR5 colours from
    # https://tdaviesbarnard.co.uk/1202/ipcc-official-colors-rcp/
    "AR5-RCP-2.6": "#0000FF",
    "AR5-RCP-4.5": "#79BCFF",
    "AR5-RCP-6.0": "#FF822D",
    "AR5-RCP-8.5": "#FF0000",
}


class PlotAccessor:
    """Make plots of IamDataFrame instances"""

    def __init__(self, df):
        self._parent = df

        # assign plotting functions as attributes
        PLOT_MAPPING = {
            "line": line,
            "bar": bar,
            "stack": stack,
            "box": box,
            "pie": pie,
            "scatter": scatter,
            "sankey": sankey,
        }
        # inherit the docstring from the plot function
        for name, func in PLOT_MAPPING.items():
            getattr(self, name).__func__.__doc__ = func.__doc__

    def __call__(self, kind="line", *args, **kwargs):
        return getattr(self, kind)(**kwargs)

    def line(self, **kwargs):
        return line(self._parent, **kwargs)

    def bar(self, **kwargs):
        return bar(self._parent, **kwargs)

    def stack(self, **kwargs):
        return stack(self._parent, **kwargs)

    def hist(self, **kwargs):
        raise NotImplementedError("Histogram plot not implemented yet!")

    def box(self, **kwargs):
        return box(self._parent, **kwargs)

    def pie(self, **kwargs):
        return pie(self._parent, **kwargs)

    def scatter(self, *args, **kwargs):
        return scatter(self._parent, *args, **kwargs)

    def sankey(self, *args, **kwargs):
        return sankey(self._parent, *args, **kwargs)


def reset_default_props(**kwargs):
    """Reset properties to initial cycle point"""
    global _DEFAULT_PROPS
    pcycle = plt.rcParams["axes.prop_cycle"]
    _DEFAULT_PROPS = {
        "color": itertools.cycle(_get_standard_colors(**kwargs))
        if len(kwargs) > 0
        else itertools.cycle([x["color"] for x in pcycle]),
        "marker": itertools.cycle(["o", "x", ".", "+", "*"]),
        "linestyle": itertools.cycle(["-", "--", "-.", ":"]),
    }


def default_props(reset=False, **kwargs):
    """Return current default properties

    Parameters
    ----------
    reset : bool
            if True, reset properties and return
    """
    global _DEFAULT_PROPS
    if _DEFAULT_PROPS is None or reset:
        reset_default_props(**kwargs)
    return _DEFAULT_PROPS


def mpl_args_to_meta_cols(df, **kwargs):
    """Return the kwargs values (not keys) matching a `df.meta` column name"""
    cols = set()
    for arg, value in kwargs.items():
        if isstr(value) and value in df.meta.columns:
            cols.add(value)
    return list(cols)


def assign_style_props(df, color=None, marker=None, linestyle=None, cmap=None):
    """Assign the style properties for a plot

    Parameters
    ----------
    df : pd.DataFrame
        data to be used for style properties
    """
    if color is None and cmap is not None:
        raise ValueError("`cmap` must be provided with the `color` argument")

    # determine color, marker, and linestyle for each line
    n = (
        len(df[color].unique())
        if color in df.columns
        else len(df[list(set(df.columns) & set(IAMC_IDX))].drop_duplicates())
    )
    defaults = default_props(reset=True, num_colors=n, colormap=cmap)

    props = {}
    rc = run_control()

    kinds = [("color", color), ("marker", marker), ("linestyle", linestyle)]

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
    if "color" in props:
        d = props["color"]
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
            props["color"] = d
    return props


def reshape_mpl(df, x, y, idx_cols, **kwargs):
    """Reshape data from long form to "bar plot form".

    Matplotlib requires x values as the index with one column for bar grouping.
    Table values come from y values.
    """
    idx_cols = to_list(idx_cols)
    if x not in idx_cols:
        idx_cols += [x]

    # check for duplicates
    rows = df[idx_cols].duplicated()
    if any(rows):
        _raise_data_error("Duplicates in plot data", df.loc[rows, idx_cols])

    # reshape the data
    df = df.set_index(idx_cols)[y].unstack(x).T

    # reindex to get correct order
    for key, value in kwargs.items():
        level = None
        if df.columns.name == key:  # single-dimension index
            axis, _values = "columns", df.columns.values
        elif df.index.name == key:  # single-dimension index
            axis, _values = "index", list(df.index)
        elif key in df.columns.names:  # several dimensions -> pd.MultiIndex
            axis, _values = "columns", get_index_levels(df.columns, key)
            level = key
        else:
            raise ValueError(f"No dimension {key} in the data!")

        # if not given, determine order based on run control (if possible)
        if value is None and key in run_control()["order"]:
            # select relevant items from run control, then add other cols
            value = [i for i in run_control()["order"][key] if i in _values]
            value += [i for i in _values if i not in value]
        df = df.reindex(**{axis: value, "level": level})

    return df


def time_col_or_year(df):
    """Return the time-col (if `df` is an IamDataFrame) or 'year'"""
    try:
        return df.time_col
    except AttributeError:
        return "year"


def pie(
    df,
    value="value",
    category="variable",
    legend=False,
    title=None,
    ax=None,
    cmap=None,
    **kwargs,
):
    """Plot data as a pie chart.

    Parameters
    ----------
    df : :class:`pyam.IamDataFrame`, :class:`pandas.DataFrame`
        Data to be plotted
    value : string, optional
        The column to use for data values
    category : string, optional
        The column to use for labels
    legend : bool, optional
        Include a legend.
    title : string, optional
        Text to use for the title.
    ax : :class:`matplotlib.axes.Axes`, optional
    cmap : string, optional
        The name of a registered colormap.
    kwargs
        Additional arguments passed to :meth:`pandas.DataFrame.plot`.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        Modified `ax` or new instance
    """

    # cast to DataFrame if necessary
    # TODO: select only relevant meta columns
    if not isinstance(df, pd.DataFrame):
        df = df.as_pandas()

    for col in set(SORT_IDX) - set([category]):
        if len(df[col].unique()) > 1:
            msg = (
                "Can not plot multiple {}s in a pie plot with value={} and category={}"
            )
            raise ValueError(msg.format(col, value, category))

    if ax is None:
        fig, ax = plt.subplots()

    # get data, set negative values to explode
    _df = df.groupby(category)[value].sum()
    where = _df > 0
    explode = tuple(0 if _ else 0.2 for _ in where)
    _df = _df.abs()

    # explicitly get colors
    defaults = default_props(reset=True, num_colors=len(_df.index), colormap=cmap)[
        "color"
    ]
    rc = run_control()

    if "colors" in kwargs:
        colors = kwargs.pop("colors")
    else:
        colors = []
        for key, c in zip(_df.index, defaults):
            if category in rc["color"] and key in rc["color"][category]:
                c = rc["color"][category][key]
            colors.append(c)

    # plot data
    _df.plot(kind="pie", colors=colors, ax=ax, explode=explode, **kwargs)

    # add legend and title
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), labels=_df.index)
    if not legend:
        ax.legend_.remove()
    if title:
        ax.set_title(title)

    # remove label
    ax.set_ylabel("")

    return ax


def stack(
    df,
    x=None,
    y="value",
    stack="variable",
    order=None,
    total=None,
    legend=True,
    title=True,
    ax=None,
    cmap=None,
    **kwargs,
):
    """Plot a stacked area chart of timeseries data

    Parameters
    ----------
    df : :class:`pyam.IamDataFrame`, :class:`pandas.DataFrame`
        Data to be plotted
    x : string, optional
        The coordinates or column of the data points for the horizontal axis;
        defaults to the time domain (if `df` is IamDataFrame) or 'year'.
    y : string, optional
        The coordinates or column of the data points for the vertical axis.
    stack : string, optional
        The column to use for stack groupings
    order : list, optional
         The order to plot the stack levels and the legend. If not specified,
         order by :meth:`run_control()['order'][\<stack\>] <pyam.run_control>`
         (where available) or alphabetical.
    total : bool or dict, optional
        If True, plot a total line with default |pyam| settings. If a dict,
        then plot the total line using the dict key-value pairs as keyword
        arguments to :meth:`matplotlib.axes.Axes.plot`.
        If None, do not plot the total line.
    legend : bool, optional
        Include a legend.
    title : bool or string, optional
        Text to use for the title, display a default if True.
    ax : :class:`matplotlib.axes.Axes`, optional
    cmap : string, optional
        The name of a registered colormap.
    kwargs
        Additional arguments passed to :meth:`pandas.DataFrame.plot`

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        Modified `ax` or new instance
    """

    # default x-axis to time-col attribute from an IamDataFrame, else use "year"
    x = x or time_col_or_year(df)

    # cast to DataFrame if necessary
    # TODO: select only relevant meta columns
    if not isinstance(df, pd.DataFrame):
        df = df.as_pandas()

    for col in set(SORT_IDX) - set([x, stack]):
        if len(df[col].unique()) > 1:
            msg = "Can not plot multiple {}s in stack_plot with x={}, stack={}"
            raise ValueError(msg.format(col, x, stack))

    if ax is None:
        fig, ax = plt.subplots()

    # long form to one column per stack group
    _df = reshape_mpl(df, x, y, stack, **{stack: order})

    # cannot plot timeseries that do not extend for the entire range
    has_na = _df.iloc[[0, -1]].isna().any()
    if any(has_na):
        msg = "Can not plot data that does not extend for the entire {} range"
        raise ValueError(msg.format(x))

    def as_series(index, name):
        _idx = [i[0] for i in index]
        return pd.Series([0] * len(index), index=_idx, name=name)

    # determine all time-indices where a timeseries crosses 0 and add to data
    _rows = pd.concat(
        [as_series(cross_threshold(_df[c], return_type=float), c) for c in _df.columns],
        axis=1,
    )
    _df = (
        _df.append(_rows.loc[_rows.index.difference(_df.index)])
        .sort_index()
        .interpolate(method="index")
    )

    # explicitly get colors
    defaults = default_props(reset=True, num_colors=len(_df.columns), colormap=cmap)[
        "color"
    ]
    rc = run_control()
    colors = {}
    for key in _df.columns:
        c = next(defaults)
        c_in_rc = "color" in rc
        if c_in_rc and stack in rc["color"] and key in rc["color"][stack]:
            c = rc["color"][stack][key]
        colors[key] = c

    # determine positive and negative parts of the timeseries data
    _df_pos = _df.applymap(lambda x: max(x, 0))
    _df_neg = _df.applymap(lambda x: min(x, 0))

    lower = [0] * len(_df_pos)
    for col in reversed(_df_pos.columns):
        upper = _df_pos[col].fillna(0) + lower
        ax.fill_between(
            _df_pos.index,
            upper,
            lower,
            label=None,
            color=colors[col],
            linewidth=0,
            **kwargs,
        )
        lower = upper

    upper = [0] * len(_df_neg)
    for col in _df_neg.columns:
        lower = _df_neg[col].fillna(0) + upper
        # add label only on negative to have it in right order
        ax.fill_between(
            _df_neg.index,
            upper,
            lower,
            label=col,
            color=colors[col],
            linewidth=0,
            **kwargs,
        )
        upper = lower

    # add total
    if (total is not None) and total:  # cover case where total=False
        if isinstance(total, bool):  # can now assume total=True
            total = {}
        total.setdefault("label", "Total")
        total.setdefault("color", "black")
        total.setdefault("lw", 4.0)
        ax.plot(_df.index, _df.sum(axis=1), **total)

    # add legend
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    if not legend:
        ax.legend_.remove()

    # add default labels if possible
    ax.set_xlabel(x.capitalize())
    units = df["unit"].unique()
    if len(units) == 1:
        ax.set_ylabel(units[0])

    # build a default title if possible
    _title = []
    for var in ["model", "scenario", "region", "variable"]:
        values = df[var].unique()
        if len(values) == 1:
            _title.append("{}: {}".format(var, values[0]))
    if title and _title:
        title = " ".join(_title) if title is True else title
        ax.set_title(title)

    return ax


def bar(
    df,
    x=None,
    y="value",
    bars="variable",
    order=None,
    bars_order=None,
    orient="v",
    legend=True,
    title=True,
    ax=None,
    cmap=None,
    **kwargs,
):
    """Plot data as a stacked or grouped bar chart

    Parameters
    ----------
    df : :class:`pyam.IamDataFrame`, :class:`pandas.DataFrame`
        Data to be plotted
    x : string, optional
        The coordinates or column of the data points for the horizontal axis;
        defaults to the time domain (if `df` is IamDataFrame) or 'year'.
    y : string, optional
        The coordinates or column of the data points for the vertical axis.
    bars : string, optional
        The column to use for bar groupings
    order, bars_order : list, optional
         The order to plot the levels on the x-axis and the bars (and legend).
         If not specified, order
         by :meth:`run_control()['order'][\<stack\>] <pyam.run_control>`
         (where available) or alphabetical.
    orient : string, optional
        Vertical or horizontal orientation.
    legend : bool, optional
        Include a legend.
    title : bool or string, optional
        Text to use for the title, display a default if True.
    ax : :class:`matplotlib.axes.Axes`, optional
    cmap : string, optional
        The name of a registered colormap.
    kwargs
        Additional arguments passed to :meth:`pandas.DataFrame.plot`

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        Modified `ax` or new instance
    """

    # default x-axis to time-col attribute from an IamDataFrame, else use "year"
    x = x or time_col_or_year(df)

    # cast to DataFrame if necessary
    # TODO: select only relevant meta columns
    if not isinstance(df, pd.DataFrame):
        df = df.as_pandas()

    for col in set(SORT_IDX) - set([x, bars]):
        if len(df[col].unique()) > 1:
            msg = "Can not plot multiple {}s in bar plot with x={}, bars={}"
            raise ValueError(msg.format(col, x, bars))

    if ax is None:
        fig, ax = plt.subplots()

    # long form to one column per bar group
    _df = reshape_mpl(df, x, y, bars, **{x: order, bars: bars_order})

    # explicitly get colors
    defaults = default_props(reset=True, num_colors=len(_df.columns), colormap=cmap)[
        "color"
    ]
    rc = run_control()
    color = []
    for key in _df.columns:
        c = next(defaults)
        if "color" in rc and bars in rc["color"] and key in rc["color"][bars]:
            c = rc["color"][bars][key]
        color.append(c)

    # change year to str to prevent pandas/matplotlib from auto-ordering (#474)
    if _df.index.name == "year":
        _df.index = map(str, _df.index)

    # plot data
    kind = "bar" if orient.startswith("v") else "barh"
    _df.plot(kind=kind, color=color, ax=ax, **kwargs)

    # add legend
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    if not legend:
        ax.legend_.remove()

    # add default labels if possible
    if orient == "v":
        ax.set_xlabel(x.capitalize())
    else:
        ax.set_ylabel(x.capitalize())
    units = df["unit"].unique()
    if len(units) == 1 and y == "value":
        if orient == "v":
            ax.set_ylabel(units[0])
        else:
            ax.set_xlabel(units[0])

    # build a default title if possible
    _title = []
    for var in ["model", "scenario", "region", "variable"]:
        values = df[var].unique()
        if len(values) == 1:
            _title.append("{}: {}".format(var, values[0]))
    if title and _title:
        title = " ".join(_title) if title is True else title
        ax.set_title(title)

    return ax


def box(df, y="value", x=None, by=None, legend=True, title=None, ax=None, **kwargs):
    """Plot boxplot of data using seaborn.boxplot

    Parameters
    ----------
    df : :class:`pyam.IamDataFrame`, :class:`pandas.DataFrame`
        Data to be plotted
    y : string, optional
        The column to use for y-axis values representing the distribution
        within the boxplot
    x : string, optional
        The coordinates or column of the data points for the horizontal axis;
        defaults to the time domain (if `df` is IamDataFrame) or 'year'.
    by : string, optional
        The column for grouping y-axis values at each x-axis point,
        i.e. a 3rd dimension. Data should be categorical, not a contiuous
        variable.
    legend : bool, optional
        Include a legend.
    title : bool or string, optional
        Text to use for the title, display a default if True.
    ax : :class:`matplotlib.axes.Axes`, optional
    kwargs
        Additional arguments passed to :meth:`pandas.DataFrame.plot`.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        Modified `ax` or new instance
    """

    # default x-axis to time-col attribute from an IamDataFrame, else use "year"
    x = x or time_col_or_year(df)

    # cast to DataFrame if necessary
    # TODO: select only relevant meta columns
    if not isinstance(df, pd.DataFrame):
        df = df.as_pandas()

    if by:
        rc = run_control()
        if "palette" not in kwargs and "color" in rc and by in rc["color"]:
            # TODO this only works if all categories are defined in run_control
            palette = rc["color"][by]
            df[by] = df[by].astype("category")
            df[by].cat.set_categories(list(palette), inplace=True)
            kwargs["palette"] = palette
        else:
            df.sort_values(by, inplace=True)

    if ax is None:
        fig, ax = plt.subplots()

    # Create the plot
    sns.boxplot(x=x, y=y, hue=by, data=df, ax=ax, **kwargs)

    # Add legend
    if legend:
        ax.legend(loc=2)
        ax.legend_.set_title(
            "n=" + str(len(df[META_IDX].drop_duplicates())),
        )

    # Axes labels
    if y == "value":
        ax.set_ylabel(df.unit.unique()[0])
    else:
        ax.set_ylabel(y)

    if title:
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


def add_net_values_to_bar_plot(axs, color="k"):
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


def scatter(
    df,
    x,
    y,
    legend=None,
    title=None,
    color=None,
    marker="o",
    linestyle=None,
    groupby=["model", "scenario"],
    with_lines=False,
    ax=None,
    cmap=None,
    **kwargs,
):
    """Plot data as a scatter chart.

    Parameters
    ----------
    df : class:`pyam.IamDataFrame`
        Data to be plotted
    x : str
        The coordinates or columns of the data points for the horizontal axis.
    y : str
        The coordinates or columns of the data points for the vertical axis.
    legend : bool, optional
        Include a legend. By default, show legend only if less than 13 entries.
        If a dictionary is provided, it will be used as keyword arguments
        in creating the legend.
    title : bool or string, optional
        Text to use for the title, display a default if True.
    color : string, optional
        A valid matplotlib color or column name. If a column name, common
        values will be provided the same color.
    marker : string
        A valid matplotlib marker or column name. If a column name, common
        values will be provided the same marker.
    linestyle : string, optional
        A valid matplotlib linestyle or column name. If a column name, common
        values will be provided the same linestyle.
        default: None
    groupby : list-like, optional
        Data grouping for plotting.
    with_lines : bool, optional
        Make the scatter plot with lines connecting common data.
    ax : :class:`matplotlib.axes.Axes`, optional
    cmap : string, optional
        The name of a registered colormap.
    kwargs
        Additional arguments passed to :meth:`pandas.DataFrame.plot`.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        Modified `ax` or new instance
    """

    # process the data
    xisvar = x in df.variable
    yisvar = y in df.variable

    meta_col_args = dict(color=color, marker=marker, linestyle=linestyle)
    meta_cols = mpl_args_to_meta_cols(df, **meta_col_args)

    if not xisvar and not yisvar:
        cols = [x, y] + meta_cols
        data = df.meta[cols].reset_index()
    elif xisvar and yisvar:
        # filter pivot both and rename
        dfx = (
            df.filter(variable=x)
            .as_pandas(meta_cols=meta_cols)
            .rename(columns={"value": x, "unit": "xunit"})
            .set_index(YEAR_IDX)
            .drop("variable", axis=1)
        )
        dfy = (
            df.filter(variable=y)
            .as_pandas(meta_cols=meta_cols)
            .rename(columns={"value": y, "unit": "yunit"})
            .set_index(YEAR_IDX)
            .drop("variable", axis=1)
        )
        data = dfx.join(dfy, lsuffix="_left", rsuffix="").reset_index()
    else:
        # filter, merge with meta, and rename value column to match var
        var = x if xisvar else y
        data = (
            df.filter(variable=var)
            .as_pandas(meta_cols=mpl_args_to_meta_cols(df, **kwargs))
            .rename(columns={"value": var})
        )

    # drop nan
    data.dropna(inplace=True)

    # create a plotting axes (if not given as kwarg)
    if ax is None:
        fig, ax = plt.subplots()

    # assign styling properties
    props = assign_style_props(data, **meta_col_args, cmap=cmap)

    # group data
    groups = data.dropna().groupby(groupby)

    # loop over grouped dataframe, plot data
    legend_data = []
    for name, group in groups:
        pargs = {}
        labels = []
        for key, kind, var in [
            ("c", "color", color),
            ("marker", "marker", marker),
            ("linestyle", "linestyle", linestyle),
        ]:
            if kind in props:
                label = group[var].values[0]
                pargs[key] = props[kind][group[var].values[0]]
                labels.append(repr(label).lstrip("u'").strip("'"))
            else:
                pargs[key] = var

        if len(labels) > 0:
            legend_data.append(" ".join(labels))
        else:
            legend_data.append(" ".join(name))
        kwargs.update(pargs)
        label = " ".join(group[g].iloc[0] for g in groupby)
        if with_lines:
            ax.plot(group[x], group[y], label=label, **kwargs)
        else:
            kwargs.pop("linestyle")  # scatter() can't take a linestyle
            ax.scatter(group[x], group[y], label=label, **kwargs)

    # build legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    if legend_data != [""] * len(legend_data):
        labels = sorted(list(set(tuple(legend_data))))
        idxs = [legend_data.index(d) for d in labels]
        handles = [handles[i] for i in idxs]
    if legend is not False:
        _add_legend(ax, handles, labels, legend)

    # add labels and title
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)

    return ax


def line(
    df,
    x=None,
    y="value",
    order=None,
    legend=None,
    title=True,
    color=None,
    marker=None,
    linestyle=None,
    fill_between=None,
    final_ranges=None,
    rm_legend_label=[],
    ax=None,
    cmap=None,
    **kwargs,
):
    """Plot data as lines with or without markers.

    Parameters
    ----------
    df : :class:`pyam.IamDataFrame`, :class:`pandas.DataFrame`
        Data to be plotted
    x : string, optional
        The coordinates or column of the data points for the horizontal axis;
        defaults to the time domain (if `df` is IamDataFrame) or 'year'.
    y : string, optional
        The column to use as y-axis
    order : dict or list, optional
         The order of lines and the legend as :code:`{<column>: [<order>]}` or
         a list of columns where ordering should be applied. If not specified,
         order by :meth:`run_control()['order'][\<column\>] <pyam.run_control>`
         (where available) or alphabetical.
    legend : bool or dictionary, optional
        Include a legend. By default, show legend only if less than 13 entries.
        If a dictionary is provided, it will be used as keyword arguments
        in creating the legend.
    title : bool or string, optional
        Text to use for the title, display a default if True.
    color : string, optional
        A valid matplotlib color or column name. If a column name, common
        values will be provided the same color.
    marker : string, optional
        A valid matplotlib marker or column name. If a column name, common
        values will be provided the same marker.
    linestyle : string, optional
        A valid matplotlib linestyle or column name. If a column name, common
        values will be provided the same linestyle.
    fill_between : boolean or dict, optional
        Fill lines between minima/maxima of the 'color' argument. This can only
        be used if also providing a 'color' argument. If this is True, then
        default arguments will be provided to `ax.fill_between()`. If this is a
        dictionary, those arguments will be provided instead of defaults.
    final_ranges : boolean or dict, optional
        Add vertical line between minima/maxima of the 'color' argument in the
        last period plotted.  This can only be used if also providing a 'color'
        argument. If this is True, then default arguments will be provided to
        `ax.axvline()`. If this is a dictionary, those arguments will be
        provided instead of defaults.
    rm_legend_label : string or list, optional
        Remove the color, marker, or linestyle label in the legend.
    ax : :class:`matplotlib.axes.Axes`, optional
    cmap : string, optional
        The name of a registered colormap.
    kwargs
        Additional arguments passed to :meth:`pandas.DataFrame.plot`.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        Modified `ax` or new instance
    """

    # default x-axis to time-col attribute from an IamDataFrame, else use "year"
    x = x or time_col_or_year(df)

    # cast to DataFrame if necessary
    if not isinstance(df, pd.DataFrame):
        meta_col_args = dict(color=color, marker=marker, linestyle=linestyle)
        df = df.as_pandas(meta_cols=mpl_args_to_meta_cols(df, **meta_col_args))

    # pivot data if asked for explicit variable name
    variables = df["variable"].unique()
    if x in variables or y in variables:
        keep_vars = set([x, y]) & set(variables)
        df = df[df["variable"].isin(keep_vars)]
        idx = list(set(df.columns) - set(["value"]))
        df = (
            df.reset_index()
            .set_index(idx)
            .value.unstack(level="variable")  # df -> series  # keep_vars are columns
            .rename_axis(None, axis=1)  # rm column index name
            .reset_index()
            .set_index(META_IDX)
        )
        if x != "year" and y != "year":
            df = df.drop("year", axis=1)  # years causes nan's

    if ax is None:
        fig, ax = plt.subplots()

    # assign styling properties
    props = assign_style_props(
        df, color=color, marker=marker, linestyle=linestyle, cmap=cmap
    )

    if fill_between and "color" not in props:
        raise ValueError("Must use `color` kwarg if using `fill_between`")
    if final_ranges and "color" not in props:
        raise ValueError("Must use `color` kwarg if using `final_ranges`")

    # prepare a dict for ordering, reshape data for use in line_plot
    idx_cols = list(df.columns.drop(y))
    if not isinstance(order, dict):
        order = dict([(i, None) for i in order or idx_cols])
    df = reshape_mpl(df, x, y, idx_cols, **order)

    # determine the columns that should go into the legend
    idx_cols.remove(x)
    title_cols = []
    y_label = None
    for col in idx_cols:
        values = get_index_levels(df.columns, col)
        if len(values) == 1 and col not in [color, marker, linestyle]:
            if col == "unit" and y == "value":
                y_label = values[0]
            elif col == y and col != "value":
                y_label = values[0]
            else:
                if col != "unit":
                    title_cols.append(f"{col}: {values[0]}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(col)
            else:  # cannot drop last remaining level, replace by empty list
                df.columns = [""]

    # determine index of column name in reshaped dataframe
    prop_idx = {}
    for kind, var in [("color", color), ("marker", marker), ("linestyle", linestyle)]:
        if var is not None and var in df.columns.names:
            prop_idx[kind] = df.columns.names.index(var)

    # pop label to avoid multiple values for plot-kwarg
    label = kwargs.pop("label", None)

    # plot data, keeping track of which legend labels to apply
    for col, data in df.iteritems():
        # handle case where columns are not strings or only have 1 dimension
        col = list(map(str, to_list(col)))
        pargs = {}
        labels = []
        # build plotting args and line legend labels
        for key, kind, var in [
            ("c", "color", color),
            ("marker", "marker", marker),
            ("linestyle", "linestyle", linestyle),
        ]:
            if kind in props:
                _label = col[prop_idx[kind]]
                pargs[key] = props[kind][_label]
                if kind not in to_list(rm_legend_label):
                    labels.append(repr(_label).lstrip("u'").strip("'"))
            else:
                pargs[key] = var
        kwargs.update(pargs)
        data = data.dropna()
        data.plot(ax=ax, label=label or " - ".join(labels if labels else col), **kwargs)

    if fill_between:
        _kwargs = {"alpha": 0.25} if fill_between in [True, None] else fill_between
        data = df.T
        columns = data.columns
        # get outer boundary mins and maxes
        allmins = data.groupby(color).min()
        intermins = (
            data.dropna(axis=1)
            .groupby(color)
            .min()  # nonan data
            .reindex(columns=columns)  # refill with nans
            .T.interpolate(method="index")
            .T  # interpolate
        )
        mins = pd.concat([allmins, intermins]).groupby(level=0).min()
        allmaxs = data.groupby(color).max()
        intermaxs = (
            data.dropna(axis=1)
            .groupby(color)
            .max()  # nonan data
            .reindex(columns=columns)  # refill with nans
            .T.interpolate(method="index")
            .T  # interpolate
        )
        maxs = pd.concat([allmaxs, intermaxs]).groupby(level=0).max()
        # do the fill
        for idx in mins.index:
            ymin = mins.loc[idx]
            ymax = maxs.loc[idx]
            ax.fill_between(
                ymin.index, ymin, ymax, facecolor=props["color"][idx], **_kwargs
            )

    # add bars to the end of the plot showing range
    if final_ranges:
        # have to explicitly draw it to get the tick labels (these change once
        # you add the vlines)
        plt.gcf().canvas.draw()
        _kwargs = {"linewidth": 2} if final_ranges in [True, None] else final_ranges
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
            ax.axvline(
                xpos, ymin=_ymin, ymax=_ymax, color=props["color"][idx], **_kwargs
            )
        # for equal spacing between xmin and first datapoint and xmax and last
        # line
        ax.set_xlim(xmin, xpos + first - xmin)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

    # build unique legend handles and labels
    if legend is not False:
        handles, labels = [np.array(i) for i in ax.get_legend_handles_labels()]
        if label is not None:  # label given explicitly via kwarg
            _add_legend(ax, handles, labels, legend)
        else:
            _, idx = np.unique(labels, return_index=True)
            idx.sort()
            _add_legend(ax, handles[idx], labels[idx], legend)

    # add default labels if possible
    ax.set_xlabel(x.title())
    ax.set_ylabel(y_label or y.title())

    # show a default title from columns with a unique value or a custom title
    if title:
        ax.set_title(" - ".join(title_cols) if title is True else title)

    return ax


def _add_legend(ax, handles, labels, legend):
    if legend is None and len(labels) >= MAX_LEGEND_LABELS:
        logger.info(f">={MAX_LEGEND_LABELS} labels, not applying legend")
    else:
        legend = {} if legend in [True, None] else legend
        loc = legend.pop("loc", "best")
        outside = loc.split(" ")[1] if loc.startswith("outside ") else False
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
