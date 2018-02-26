import collections
import itertools
import os
import yaml

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

# line colors, markers, and styles that are cycled through when not
# explicitly declared
_DEFAULT_PROPS = None

# user-defined defaults for various plot settings
_RUN_CONTROL = None

_RC_DEFAULTS = {
    'color': {},
    'marker': {},
    'linestyle': {},
}


def isstr(x):
    """Returns True if x is a string"""
    try:
        return isinstance(x, (str, unicode))
    except NameError:
        return isinstance(x, str)


def run_control():
    """Global run control for determining user-defined defaults for plotting style"""
    global _RUN_CONTROL
    if _RUN_CONTROL is None:
        _RUN_CONTROL = RunControl()
    return _RUN_CONTROL


def _recursive_update(d, u):
    """recursively update a dictionary d with a dictionary u"""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = _recursive_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


class RunControl(collections.Mapping):
    """A thin wrapper around a Python Dictionary to support configuration of
    harmonization execution. Input can be provided as dictionaries or YAML
    files.
    """

    def __init__(self, rc=None, defaults=None):
        """
        Parameters
        ----------
        rc : string, file, dictionary, optional
            a path to a YAML file, a file handle for a YAML file, or a 
            dictionary describing run control configuration
        defaults : string, file, dictionary, optional
            a path to a YAML file, a file handle for a YAML file, or a 
            dictionary describing **default** run control configuration
        """
        rc = rc or {}
        defaults = defaults or _RC_DEFAULTS

        rc = self._load_yaml(rc)
        defaults = self._load_yaml(defaults)
        self.store = _recursive_update(defaults, rc)

    def update(self, rc):
        """Add additional run control parameters

        Parameters
        ----------
        rc : string, file, dictionary, optional
            a path to a YAML file, a file handle for a YAML file, or a 
            dictionary describing run control configuration
        """
        rc = self._load_yaml(rc)
        self.store = _recursive_update(self.store, rc)

    def __getitem__(self, k):
        return self.store[k]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return self.store.__repr__()

    def _get_path(self, key, fyaml, fname):
        if os.path.exists(fname):
            return fname

        _fname = os.path.join(os.path.dirname(fyaml), fname)
        if not os.path.exists(_fname):
            msg = "YAML key '{}' in {}: {} is not a valid relative " + \
                "or absolute path"
            raise IOError(msg.format(key, fyaml, fname))
        return _fname

    def _load_yaml(self, obj):
        check_rel_paths = False
        if hasattr(obj, 'read'):  # it's a file
            obj = obj.read()
        if isstr(obj) and os.path.exists(obj):
            check_rel_paths = True
            fname = obj
            with open(fname) as f:
                obj = f.read()
        if not isinstance(obj, dict):
            obj = yaml.load(obj)
        return obj

    def recursive_update(self, k, d):
        """Recursively update a top-level option in the run control

        Parameters
        ----------
        k : string
            the top-level key
        d : dictionary or similar
            the dictionary to use for updating
        """
        u = self.__getitem__(k)
        self.store[k] = _recursive_update(u, d)


def reset_default_props():
    """Reset properties to initial cycle point"""
    global _DEFAULT_PROPS
    _DEFAULT_PROPS = {
        'color': itertools.cycle([x['color'] for x in plt.rcParams['axes.prop_cycle']]),
        'marker': itertools.cycle(['o', 'x', '.', '+', '*']),
        'linestyle': itertools.cycle(['-', '--', '-.', ':']),
    }


def default_props(reset=False):
    """Return current default properties

    Parameters
    ----------
    reset : bool
            if True, reset properties and return
            default: False
    """
    global _DEFAULT_PROPS
    if _DEFAULT_PROPS is None or reset:
        reset_default_props()
    return _DEFAULT_PROPS


def reshape_line_plot(df, x, y):
    """Reshape data from long form to "plot form".

    Plot form has x value as the index with one column for each line. 
    Each column has data points as values and all metadata as column headers.
    """
    idx = list(df.columns.drop(y))
    if df.duplicated(idx).any():
        warnings.warn('Duplicated index found.')
        df = df.drop_duplicates(idx, keep='last')
    df = df.set_index(idx)[y].unstack(x).T
    return df


def line_plot(df, x='year', y='value', ax=None, legend=False,
              color=None, marker=None, linestyle=None, **kwargs):
    """Plot data as lines with or without markers.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot as a long-form data frame
    x : str, optional
        The column to use for x-axis values
        default: year
    y : str, optional
        The column to use for y-axis values
        default: value
    ax : matplotlib.Axes, optional
    legend : bool, optional
        Include a legend
        default: False
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
    kwargs : Additional arguments to pass to the pd.DataFrame.plot() function
    """

    if ax is None:
        fig, ax = plt.subplots()

    df = reshape_line_plot(df, x, y)  # long form to one column per line

    # determine color, marker, and linestyle for each line
    defaults = default_props(reset=True)
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
        data.plot(ax=ax, **kwargs)

    # build legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    if legend_data != [''] * len(legend_data):
        labels = sorted(list(set(tuple(legend_data))))
        idxs = [legend_data.index(d) for d in labels]
        handles = [handles[i] for i in idxs]
    if legend:
        ax.legend(handles, labels)

    # add default labels if possible
    ax.set_xlabel(x.capitalize())
    units = df.columns.get_level_values('unit').unique()
    if len(units) == 1:
        ax.set_ylabel(units[0])

    # build a default title if possible
    title = []
    for var in ['model', 'scenario', 'region', 'variable']:
        values = df.columns.get_level_values(var).unique()
        if len(values) == 1:
            title.append('{}: {}'.format(var, values[0]))
    if title:
        ax.set_title(' '.join(title))

    return ax, handles, labels
